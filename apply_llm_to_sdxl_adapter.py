import torch
import torch.nn as nn
import logging

logger = logging.getLogger("LLM-SDXL-Adapter")

def pad_to_length(tensor, target_length, dim=1, value=0):
    """Pad or trim tensor to target length on a given dim."""
    current_length = tensor.size(dim)
    if current_length == target_length:
        return tensor
    if current_length > target_length:
        return tensor.narrow(dim, 0, target_length)

    pad_shape = list(tensor.shape)
    pad_shape[dim] = target_length - current_length
    padding = torch.full(pad_shape, value, device=tensor.device, dtype=tensor.dtype)
    return torch.cat([tensor, padding], dim=dim)

class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads=16, mlp_ratio=4.0, dropout=0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(
            dim, num_heads, batch_first=True, dropout=dropout
        )
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, int(dim * mlp_ratio)),
            nn.GELU(),
            nn.Linear(int(dim * mlp_ratio), dim),
        )

    def forward(self, x, mask=None):
        q = self.norm1(x)
        key_padding_mask = (~mask.bool()) if mask is not None else None
        attn_out, _ = self.attn(q, q, q, key_padding_mask=key_padding_mask, need_weights=False)
        x = x + attn_out
        x = x + self.mlp(self.norm2(x))
        return x

class LLMToSDXLAdapter(nn.Module):
    """
    Adapter: LLM hidden_states -> SDXL conditioning.
    Includes enhanced debugging for compression attention.
    """
    def __init__(self, llm_dim=1152, sdxl_seq_dim=2048, sdxl_pooled_dim=1280, max_input_len=512,
                 target_seq_len=308, n_wide_blocks=3, n_narrow_blocks=3, num_heads=16, dropout=0.0):
        super().__init__()
        self.max_input_len = max_input_len
        self.target_seq_len = target_seq_len
        self.num_heads = num_heads

        self.seq_projection = nn.Linear(llm_dim, sdxl_seq_dim) if llm_dim != sdxl_seq_dim else None
        self.input_position_embeddings = nn.Parameter(torch.randn(1, max_input_len, sdxl_seq_dim))
        self.output_position_embeddings = nn.Parameter(torch.randn(1, target_seq_len, sdxl_seq_dim))
        self.wide_attention_blocks = nn.ModuleList([TransformerBlock(sdxl_seq_dim, num_heads=num_heads, dropout=dropout) for _ in range(n_wide_blocks)])
        self.compression_queries = nn.Parameter(torch.randn(1, target_seq_len, sdxl_seq_dim))
        self.compression_attention = nn.MultiheadAttention(embed_dim=sdxl_seq_dim, num_heads=num_heads, batch_first=True, dropout=dropout)
        self.compression_norm = nn.LayerNorm(sdxl_seq_dim)
        self.compression_gate = nn.Sequential(nn.Linear(sdxl_seq_dim * 2, sdxl_seq_dim), nn.Sigmoid())
        self.narrow_attention_blocks = nn.ModuleList([TransformerBlock(sdxl_seq_dim, num_heads=num_heads, dropout=dropout) for _ in range(n_narrow_blocks)])
        self.pooling_attention = nn.MultiheadAttention(embed_dim=sdxl_seq_dim, num_heads=num_heads, batch_first=True, dropout=dropout)
        self.pooling_token = nn.Parameter(torch.randn(1, 1, sdxl_seq_dim))
        self.pooled_projection = nn.Sequential(
            nn.Linear(sdxl_seq_dim, sdxl_seq_dim), nn.LayerNorm(sdxl_seq_dim), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(sdxl_seq_dim, sdxl_pooled_dim)
        )

    @staticmethod
    def _decode_context_window(tokenizer, input_ids_tensor, center_idx, window=2):
        ids = input_ids_tensor.squeeze(0)
        start = max(center_idx - window, 0)
        end = min(center_idx + window + 1, ids.shape[0])
        chunk_ids = ids[start:end].tolist()
        try:
            return tokenizer.decode(chunk_ids)
        except Exception:
            tokens = tokenizer.convert_ids_to_tokens(chunk_ids)
            return " ".join(tokens).replace(" ", " ").strip()

    def forward(self, llm_hidden_states, attention_mask=None, **kwargs):
        tokenizer = kwargs.get("tokenizer")
        input_ids = kwargs.get("input_ids")

        bsz, seq_len, _ = llm_hidden_states.shape
        hidden_states = self.seq_projection(llm_hidden_states) if self.seq_projection else llm_hidden_states

        if seq_len != self.max_input_len:
            hidden_states = pad_to_length(hidden_states, self.max_input_len, dim=1)
            if attention_mask is not None:
                attention_mask = pad_to_length(attention_mask, self.max_input_len, dim=1, value=0)
            else:
                attention_mask = torch.ones(bsz, self.max_input_len, device=hidden_states.device)
                if seq_len < self.max_input_len:
                    attention_mask[:, seq_len:] = 0

        hidden_states = hidden_states + self.input_position_embeddings

        for blk in self.wide_attention_blocks:
            hidden_states = blk(hidden_states, attention_mask)

        queries = self.compression_queries.expand(bsz, -1, -1)
        key_padding_mask = (~attention_mask.bool()) if attention_mask is not None else None
        compressed, attn_weights = self.compression_attention(
            queries, hidden_states, hidden_states,
            key_padding_mask=key_padding_mask,
            need_weights=True,
            average_attn_weights=False,
        )

        if tokenizer is not None and input_ids is not None:
            try:
                avg_weights = attn_weights.mean(dim=(1, 2)).squeeze(0)
                top_k = min(15, avg_weights.shape[0])
                top_vals, top_idxs = torch.topk(avg_weights, top_k)
                all_tokens = tokenizer.convert_ids_to_tokens(input_ids.squeeze(0))
                print("\n--- [Adapter Attention Debug] ---")
                print(f"Top {top_k} attended tokens during compression (Source Length: {avg_weights.shape[0]}):")
                print("-" * 40)
                for rank in range(top_k):
                    idx = top_idxs[rank].item()
                    weight = top_vals[rank].item()
                    token_piece = all_tokens[idx] if 0 <= idx < len(all_tokens) else "[N/A]"
                    pretty_token = token_piece.replace(" ", " ").replace("</s>", "[EOS]").replace("<pad>", "[PAD]").strip()
                    context = self._decode_context_window(tokenizer, input_ids, idx, window=3)
                    print(f"#{rank+1:02d} | Index: {idx:03d} | Attention: {weight:.4f} | Token: '{pretty_token}' | Context: '{context}'")
                print("---------------------------------\n")
            except Exception as e:
                logger.error(f"Adapter debug print failed: {e}")

        gate_in = torch.cat([queries, compressed], dim=-1)
        gate = self.compression_gate(gate_in)
        compressed = gate * compressed + (1.0 - gate) * queries
        compressed = self.compression_norm(compressed)
        compressed = compressed + self.output_position_embeddings

        for blk in self.narrow_attention_blocks:
            compressed = blk(compressed, None)

        pool_tok = self.pooling_token.expand(bsz, -1, -1)
        pooled, _ = self.pooling_attention(pool_tok, compressed, compressed, need_weights=False)
        pooled = self.pooled_projection(pooled.squeeze(1))

        return compressed, pooled

# ===============================================================================
#  MODIFIED ComfyUI Node with Integrated Debugging
# ===============================================================================
class ApplyLLMAdapter:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "llm_adapter": ("LLM_ADAPTER",),
                "hidden_states": ("LLM_HIDDEN_STATES",),
            },
            # --- MODIFICATION: Debug inputs are now optional ---
            "optional": {
                "tokenizer": ("LLM_TOKENIZER",),
                "input_ids": ("LLM_INPUT_IDS",),
                "attention_mask": ("LLM_ATTENTION_MASK",),
            }
        }

    RETURN_TYPES = ("CONDITIONING",)
    FUNCTION = "apply_adapter"
    CATEGORY = "llm_sdxl"

    def apply_adapter(self, llm_adapter, hidden_states, tokenizer=None, input_ids=None, attention_mask=None):
        # Pass optional args to the adapter's forward method
        seq, pooled = llm_adapter(
            llm_hidden_states=hidden_states,
            attention_mask=attention_mask,
            tokenizer=tokenizer,
            input_ids=input_ids,
        )
        conditioning = [[seq, {"pooled_output": pooled}]]
        return (conditioning,)

# --- MODIFICATION: Register the single, unified node ---
NODE_CLASS_MAPPINGS = {
    "ApplyLLMAdapter": ApplyLLMAdapter,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "ApplyLLMAdapter": "Apply LLM Adapter",
}
