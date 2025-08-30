import torch
import torch.nn as nn
import logging
import torch.nn.functional as F
from torch.nn.functional import linear, softmax, dropout

logger = logging.getLogger("LLM-SDXL-Adapter")


def pad_to_length(tensor, target_length, dim=1, value=0):
    current_length = tensor.size(dim)
    if current_length >= target_length:
        return tensor.narrow(dim, 0, target_length)
    pad_size = list(tensor.shape)
    pad_size[dim] = target_length - current_length
    padding = torch.full(pad_size, value, device=tensor.device, dtype=tensor.dtype)
    return torch.cat([tensor, padding], dim=dim)


class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads=16, mlp_ratio=4.0, dropout=0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, batch_first=True, dropout=dropout)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, int(dim * mlp_ratio)),
            nn.GELU(),
            nn.Linear(int(dim * mlp_ratio), dim)
        )

    def forward(self, x, mask=None):
        normed = self.norm1(x)
        key_padding_mask = ~mask.bool() if mask is not None else None
        attn_out, _ = self.attn(normed, normed, normed, key_padding_mask=key_padding_mask, need_weights=False)
        x = x + attn_out
        x = x + self.mlp(self.norm2(x))
        return x

# --- ★★★ 修正版 ★★★ ---
# MultiheadAttentionのロジックを正しく実装し直し、温度を適用するクラス
class SharpenedMultiheadAttention(nn.MultiheadAttention):
    def __init__(self, *args, temperature=0.75, **kwargs):
        super().__init__(*args, **kwargs)
        self.temperature = temperature
        print(f"[Adapter] SharpenedMultiheadAttention initialized with temperature: {self.temperature}")

    def forward(self, query, key, value, key_padding_mask=None, need_weights=True, attn_mask=None, average_attn_weights=True):
        # このforwardメソッドはPyTorchの内部実装を忠実に再現しつつ、温度を適用します
        
        # --- ここが修正点：batch_firstフラグに応じて次元を正しく処理します ---
        if self.batch_first:
            bsz, tgt_len, embed_dim = query.shape
            bsz, src_len, _ = key.shape
            query, key, value = [x.transpose(1, 0) for x in (query, key, value)]
        else:
            tgt_len, bsz, embed_dim = query.shape
            src_len, _, _ = key.shape
        # --------------------------------------------------------------------

        # 1. 入力の射影 (Projecting Q, K, V)
        q, k, v = F._in_projection_packed(query, key, value, self.in_proj_weight, self.in_proj_bias)

        # 2. ヘッドへの分割と再配置 (Reshaping and splitting into heads)
        if attn_mask is not None:
            if attn_mask.dim() == 2:
                attn_mask = attn_mask.unsqueeze(0)
        
        if key_padding_mask is not None and key_padding_mask.dim() == 0:
            key_padding_mask = None

        q = q.contiguous().view(tgt_len, bsz * self.num_heads, self.head_dim).transpose(0, 1)
        k = k.contiguous().view(src_len, bsz * self.num_heads, self.head_dim).transpose(0, 1)
        v = v.contiguous().view(src_len, bsz * self.num_heads, self.head_dim).transpose(0, 1)

        # 3. Attentionスコアの計算
        attn_output_weights = torch.bmm(q, k.transpose(1, 2))
        
        # --- ここが核心部分 ---
        # 4. 温度を適用してスコアの分布をシャープにする
        if self.temperature < 1.0 and self.temperature > 0:
            attn_output_weights = attn_output_weights / self.temperature
        
        # 5. マスクとSoftmaxの適用
        if key_padding_mask is not None:
             attn_output_weights = attn_output_weights.view(bsz, self.num_heads, tgt_len, src_len)
             attn_output_weights = attn_output_weights.masked_fill(
                 key_padding_mask.unsqueeze(1).unsqueeze(2),
                 float('-inf'),
             )
             attn_output_weights = attn_output_weights.view(bsz * self.num_heads, tgt_len, src_len)

        attn_output_weights = softmax(attn_output_weights, dim=-1)
        attn_output_weights = dropout(attn_output_weights, p=self.dropout, training=self.training)

        # 6. Attentionの適用と出力の射影
        attn_output = torch.bmm(attn_output_weights, v)
        attn_output = attn_output.transpose(0, 1).contiguous().view(tgt_len, bsz, embed_dim)
        attn_output = linear(attn_output, self.out_proj.weight, self.out_proj.bias)

        if self.batch_first:
            attn_output = attn_output.transpose(1, 0)

        # 7. 戻り値の整形
        if need_weights:
            attn_output_weights = attn_output_weights.view(bsz, self.num_heads, tgt_len, src_len)
            if average_attn_weights:
                attn_output_weights = attn_output_weights.sum(dim=1) / self.num_heads
            return attn_output, attn_output_weights
        else:
            return attn_output, None


class LLMToSDXLAdapter(nn.Module):
    def __init__(self,
                 llm_dim=1152,
                 sdxl_seq_dim=2048,
                 sdxl_pooled_dim=1280,
                 max_input_len=512,
                 target_seq_len=308,
                 n_wide_blocks=3,
                 n_narrow_blocks=3,
                 num_heads=16,
                 dropout=0):
        super().__init__()
        self.max_input_len = max_input_len
        self.target_seq_len = target_seq_len
        self.num_heads = num_heads

        if llm_dim != sdxl_seq_dim:
            self.seq_projection = nn.Linear(llm_dim, sdxl_seq_dim)
        else:
            self.seq_projection = None

        self.input_position_embeddings = nn.Parameter(torch.randn(1, max_input_len, sdxl_seq_dim))
        self.output_position_embeddings = nn.Parameter(torch.randn(1, target_seq_len, sdxl_seq_dim))
        
        self.wide_attention_blocks = nn.ModuleList([
            TransformerBlock(sdxl_seq_dim, num_heads=num_heads, dropout=dropout)
            for _ in range(n_wide_blocks)
        ])

        self.compression_queries = nn.Parameter(torch.randn(1, target_seq_len, sdxl_seq_dim))
        
        self.compression_attention = SharpenedMultiheadAttention(
            embed_dim=sdxl_seq_dim,
            num_heads=num_heads,
            batch_first=True,
            dropout=dropout,
            temperature=0.60 
        )
        
        self.compression_norm = nn.LayerNorm(sdxl_seq_dim)
        self.compression_gate = nn.Sequential(
            nn.Linear(sdxl_seq_dim * 2, sdxl_seq_dim),
            nn.Sigmoid()
        )

        self.narrow_attention_blocks = nn.ModuleList([
            TransformerBlock(sdxl_seq_dim, num_heads=num_heads, dropout=dropout)
            for _ in range(n_narrow_blocks)
        ])
        
        self.pooling_attention = nn.MultiheadAttention(
            embed_dim=sdxl_seq_dim,
            num_heads=num_heads,
            batch_first=True,
            dropout=dropout
        )
        self.pooling_token = nn.Parameter(torch.randn(1, 1, sdxl_seq_dim))
        self.pooled_projection = nn.Sequential(
            nn.Linear(sdxl_seq_dim, sdxl_seq_dim),
            nn.LayerNorm(sdxl_seq_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(sdxl_seq_dim, sdxl_pooled_dim)
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
        batch_size, seq_len, _ = llm_hidden_states.shape

        hidden_states = self.seq_projection(llm_hidden_states) if self.seq_projection else llm_hidden_states

        if seq_len != self.max_input_len:
            hidden_states = pad_to_length(hidden_states, self.max_input_len, dim=1)
            if attention_mask is not None:
                attention_mask = pad_to_length(attention_mask, self.max_input_len, dim=1, value=0)
            else:
                attention_mask = torch.ones(batch_size, self.max_input_len, device=hidden_states.device)
                if seq_len < self.max_input_len:
                    attention_mask[:, seq_len:] = 0

        hidden_states = hidden_states + self.input_position_embeddings

        for block in self.wide_attention_blocks:
            hidden_states = block(hidden_states, attention_mask)

        queries = self.compression_queries.expand(batch_size, -1, -1)
        key_padding_mask = (~attention_mask.bool()) if attention_mask is not None else None

        compressed_sequence, compression_weights = self.compression_attention(
            queries,
            hidden_states,
            hidden_states,
            key_padding_mask=key_padding_mask,
            need_weights=True,
            average_attn_weights=True
        )

        if tokenizer is not None and input_ids is not None and compression_weights is not None:
            try:
                # average_attn_weights=Trueなので、weightsは既に (bsz, tgt_len, src_len) のはず
                avg_weights = compression_weights.mean(dim=(1)).squeeze(0)
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

        gate_input = torch.cat([queries, compressed_sequence], dim=-1)
        gate_weights = self.compression_gate(gate_input)
        compressed_sequence = gate_weights * compressed_sequence + (1 - gate_weights) * queries
        compressed_sequence = self.compression_norm(compressed_sequence)
        compressed_sequence = compressed_sequence + self.output_position_embeddings

        for block in self.narrow_attention_blocks:
            compressed_sequence = block(compressed_sequence)

        pooling_tokens = self.pooling_token.expand(batch_size, -1, -1)
        pooled_output, _ = self.pooling_attention(pooling_tokens, compressed_sequence, compressed_sequence, need_weights=False)
        pooled_output = pooled_output.squeeze(1)
        pooled_output = self.pooled_projection(pooled_output)

        return compressed_sequence, pooled_output
