import torch
import torch.nn as nn
import logging
# --- Refactor: Import F, linear, softmax, dropout ---
import torch.nn.functional as F
from torch.nn.functional import linear, softmax, dropout

logger = logging.getLogger("LLM-SDXL-Adapter")


def pad_to_length(tensor, target_length, dim=1, value=0):
    """
    Universal function for padding tensors to a target length along a specific dimension.
    If the tensor is already longer, it will be truncated.
    """
    current_length = tensor.size(dim)

    if current_length >= target_length:
        return tensor.narrow(dim, 0, target_length)

    pad_size = list(tensor.shape)
    pad_size[dim] = target_length - current_length

    padding = torch.full(
        pad_size,
        value,
        device=tensor.device,
        dtype=tensor.dtype
    )

    return torch.cat([tensor, padding], dim=dim)


class TransformerBlock(nn.Module):
    """
    A standard Transformer block consisting of Multihead Self-Attention
    and a Feed-Forward (MLP) network, with LayerNorm and residual connections.
    """
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
            nn.Linear(int(dim * mlp_ratio), dim)
        )

    def forward(self, x, mask=None):
        # Self-attention
        normed = self.norm1(x)

        # Use key_padding_mask instead of attn_mask
        if mask is not None:
            # key_padding_mask: True means "ignore this token"
            # Our mask: 1 = real token, 0 = padding
            # So we invert
            key_padding_mask = ~mask.bool()
        else:
            key_padding_mask = None

        attn_out, _ = self.attn(
            normed, normed, normed,
            key_padding_mask=key_padding_mask,
            need_weights=False
        )
        x = x + attn_out

        # MLP
        x = x + self.mlp(self.norm2(x))

        return x


# --- ★★★ Refactored Class ★★★ ---
# Added SharpenedMultiheadAttention from the reference script
# This class applies a temperature to the attention scores to sharpen the distribution.
class SharpenedMultiheadAttention(nn.MultiheadAttention):
    """
    Standard nn.MultiheadAttention with an added `temperature` parameter.
    This module is based on the official PyTorch v2.3.0 implementation of
    MultiheadAttention, modified to apply temperature scaling to the
    attention scores before the softmax.
    A temperature < 1.0 sharpens the attention distribution, forcing the
    model to focus on a smaller, more relevant subset of tokens.
    """
    def __init__(self, *args, temperature=1.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.temperature = temperature
        if self.temperature != 1.0 and self.temperature > 0:
            logger.info(f"SharpenedMultiheadAttention initialized with temperature: {self.temperature}")
        elif self.temperature <= 0:
            logger.warning(f"Invalid temperature: {self.temperature}. Setting to 1.0")
            self.temperature = 1.0

    def forward(self, query, key, value, key_padding_mask=None, need_weights=True, attn_mask=None, average_attn_weights=True):
        
        # --- Handle batch_first flag ---
        if self.batch_first:
            bsz, tgt_len, embed_dim = query.shape
            bsz, src_len, _ = key.shape
            query, key, value = [x.transpose(1, 0) for x in (query, key, value)]
        else:
            tgt_len, bsz, embed_dim = query.shape
            src_len, _, _ = key.shape
        # -------------------------------

        # 1. Input projection
        q, k, v = F._in_projection_packed(query, key, value, self.in_proj_weight, self.in_proj_bias)

        # 2. Reshape and split into heads
        if attn_mask is not None:
            if attn_mask.dim() == 2:
                attn_mask = attn_mask.unsqueeze(0)
        
        if key_padding_mask is not None and key_padding_mask.dim() == 0:
            key_padding_mask = None

        q = q.contiguous().view(tgt_len, bsz * self.num_heads, self.head_dim).transpose(0, 1)
        k = k.contiguous().view(src_len, bsz * self.num_heads, self.head_dim).transpose(0, 1)
        v = v.contiguous().view(src_len, bsz * self.num_heads, self.head_dim).transpose(0, 1)

        # 3. Calculate attention scores
        attn_output_weights = torch.bmm(q, k.transpose(1, 2))
        
        # --- ★★★ Core Modification ★★★ ---
        # 4. Apply temperature to sharpen scores
        if self.temperature != 1.0:
            attn_output_weights = attn_output_weights / self.temperature
        # ---------------------------------
        
        # 5. Apply masks and Softmax
        if key_padding_mask is not None:
             attn_output_weights = attn_output_weights.view(bsz, self.num_heads, tgt_len, src_len)
             attn_output_weights = attn_output_weights.masked_fill(
                 key_padding_mask.unsqueeze(1).unsqueeze(2),
                 float('-inf'),
             )
             attn_output_weights = attn_output_weights.view(bsz * self.num_heads, tgt_len, src_len)

        attn_output_weights = softmax(attn_output_weights, dim=-1)
        attn_output_weights = dropout(attn_output_weights, p=self.dropout, training=self.training)

        # 6. Apply attention weights to value
        attn_output = torch.bmm(attn_output_weights, v)
        attn_output = attn_output.transpose(0, 1).contiguous().view(tgt_len, bsz, embed_dim)
        attn_output = linear(attn_output, self.out_proj.weight, self.out_proj.bias)

        # 7. Reshape output for batch_first
        if self.batch_first:
            attn_output = attn_output.transpose(1, 0)

        # 8. Format return values
        if need_weights:
            attn_output_weights = attn_output_weights.view(bsz, self.num_heads, tgt_len, src_len)
            if average_attn_weights:
                attn_output_weights = attn_output_weights.sum(dim=1) / self.num_heads
            return attn_output, attn_output_weights
        else:
            return attn_output, None
# --- End of Refactored Class ---


class LLMToSDXLAdapter(nn.Module):
    """
    Core adapter module to transform LLM embeddings into SDXL conditioning format.
    This adapter performs the following stages:
    1.  Projection: Projects LLM hidden states (e.g., 2304 dim) to the
        adapter's working dimension (e.g., 2048 dim).
    2.  Wide Processing: Applies Transformer blocks to the full-length
        sequence (e.g., 512 tokens) to capture context.
    3.  Compression: Uses a cross-attention mechanism with learnable queries
        to compress the sequence from max_input_len (e.g., 512) to
        target_seq_len (e.g., 308). This is the critical step for
        distilling prompt information.
    4.  Narrow Processing: Applies Transformer blocks to the compressed
        sequence.
    5.  Pooling: Generates a final pooled vector embedding (e.g., 1280 dim)
        using a learnable pooling token.
    """
    def __init__(self,
                 llm_dim=1152,
                 sdxl_seq_dim=2048,
                 sdxl_pooled_dim=1280,
                 max_input_len=512,
                 target_seq_len=308,
                 n_wide_blocks=3,        # Blocks BEFORE compression
                 n_narrow_blocks=3,      # Blocks AFTER compression
                 num_heads=16,
                 dropout=0,
                 attention_temp=1.0):   # --- Refactor: Added attention_temp ---
        super().__init__()

        self.max_input_len = max_input_len
        self.target_seq_len = target_seq_len
        self.num_heads = num_heads

        # Projections
        if llm_dim != sdxl_seq_dim:
            self.seq_projection = nn.Linear(llm_dim, sdxl_seq_dim)
        else:
            # Avoid unnecessary projection if dimensions match
            self.seq_projection = None

        # Positional embeddings for full sequence
        self.input_position_embeddings = nn.Parameter(
            torch.randn(1, max_input_len, sdxl_seq_dim)
        )
        # Positional embeddings for compressed sequence
        self.output_position_embeddings = nn.Parameter(
            torch.randn(1, target_seq_len, sdxl_seq_dim)
        )
        
        # Wide blocks - processing full sequence (e.g., 512 tokens)
        self.wide_attention_blocks = nn.ModuleList([
            TransformerBlock(sdxl_seq_dim, num_heads=num_heads, dropout=dropout)
            for _ in range(n_wide_blocks)
        ])

        # Compression: Cross-attention with learnable queries
        self.compression_queries = nn.Parameter(
            torch.randn(1, target_seq_len, sdxl_seq_dim)
        )
        
        # --- ★★★ Refactor: Use SharpenedMultiheadAttention ★★★ ---
        self.compression_attention = SharpenedMultiheadAttention(
            embed_dim=sdxl_seq_dim,
            num_heads=num_heads,
            batch_first=True,
            dropout=dropout,
            temperature=attention_temp  # Pass the temperature
        )
        # ---------------------------------------------------------

        # Norm layer after compression for stability
        self.compression_norm = nn.LayerNorm(sdxl_seq_dim)
        # Optional gate mechanism for weighting information
        self.compression_gate = nn.Sequential(
            nn.Linear(sdxl_seq_dim * 2, sdxl_seq_dim),
            nn.Sigmoid()
        )

        # Narrow blocks - processing compressed sequence (e.g., 308 tokens)
        self.narrow_attention_blocks = nn.ModuleList([
            TransformerBlock(sdxl_seq_dim, num_heads=num_heads, dropout=dropout)
            for _ in range(n_narrow_blocks)
        ])
        
        # Pooling head - now works with processed sequence
        self.pooling_attention = nn.MultiheadAttention(
            embed_dim=sdxl_seq_dim,
            num_heads=num_heads,
            batch_first=True,
            dropout=dropout
        )

        # Learnable [CLS]-like token for pooling
        self.pooling_token = nn.Parameter(torch.randn(1, 1, sdxl_seq_dim))

        # Final projection for pooled embeddings
        self.pooled_projection = nn.Sequential(
            nn.Linear(sdxl_seq_dim, sdxl_seq_dim),
            nn.LayerNorm(sdxl_seq_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(sdxl_seq_dim, sdxl_pooled_dim)
        )

    def forward(self, llm_hidden_states, attention_mask=None, **kwargs):
        """
        Forward pass of the adapter.
        Args:
            llm_hidden_states (torch.Tensor):
                Hidden states from the LLM (batch_size, seq_len, llm_dim)
            attention_mask (torch.Tensor, optional):
                Attention mask from the LLM (batch_size, seq_len)
                where 1 indicates a valid token and 0 indicates padding.
        Returns:
            Tuple[torch.Tensor, torch.Tensor]:
                - compressed_sequence (batch_size, target_seq_len, sdxl_seq_dim)
                - pooled_output (batch_size, sdxl_pooled_dim)
        """
        batch_size, seq_len, _ = llm_hidden_states.shape

        # ===== STAGE 1: Projection & Padding =====
        # Project to target dimension
        if self.seq_projection is not None:
            hidden_states = self.seq_projection(llm_hidden_states)
        else:
            hidden_states = llm_hidden_states  

        # Padding/truncation to max_input_len
        if seq_len != self.max_input_len:
            hidden_states = pad_to_length(hidden_states, self.max_input_len, dim=1)
            if attention_mask is not None:
                attention_mask = pad_to_length(attention_mask, self.max_input_len, dim=1, value=0)
            else:
                # Create a default mask if none is provided
                attention_mask = torch.ones(batch_size, self.max_input_len, device=hidden_states.device)
                if seq_len < self.max_input_len:
                    attention_mask[:, seq_len:] = 0
        
        # Add positional embeddings
        hidden_states = hidden_states + self.input_position_embeddings

        # ===== STAGE 2: Wide Processing (full sequence) =====
        for block in self.wide_attention_blocks:
            hidden_states = block(hidden_states, attention_mask)

        # ===== STAGE 3: Compression (e.g., 512 -> 308) =====
        # Prepare queries for compression
        queries = self.compression_queries.expand(batch_size, -1, -1)

        # Cross-attention for compression
        # We use the attention_mask from the LLM as the key_padding_mask
        if attention_mask is not None:
            # True indicates "ignore", so we invert the mask
            key_padding_mask = ~attention_mask.bool()
        else:
            key_padding_mask = None

        # Call the (potentially sharpened) compression attention
        compressed_sequence, compression_weights = self.compression_attention(
            query=queries,
            key=hidden_states,
            value=hidden_states,
            key_padding_mask=key_padding_mask,
            need_weights=True,
            average_attn_weights=True # Get averaged weights for potential debug
        )

        # Optional: Gate mechanism for mixing with queries
        gate_input = torch.cat([queries, compressed_sequence], dim=-1)
        gate_weights = self.compression_gate(gate_input)
        compressed_sequence = gate_weights * compressed_sequence + (1 - gate_weights) * queries

        # Apply normalization
        compressed_sequence = self.compression_norm(compressed_sequence)

        # Add output positional embeddings
        compressed_sequence = compressed_sequence + self.output_position_embeddings

        # ===== STAGE 4: Narrow Processing (compressed sequence) =====
        # Process the compressed sequence
        for block in self.narrow_attention_blocks:
            compressed_sequence = block(compressed_sequence, mask=None) # No mask needed here

        # ===== STAGE 5: Pooling for Vector Embeddings =====
        # Pool the compressed sequence for vector embeddings
        pooling_tokens = self.pooling_token.expand(batch_size, -1, -1)
        pooled_output, _ = self.pooling_attention(
            query=pooling_tokens,
            key=compressed_sequence,
            value=compressed_sequence,
            need_weights=False
        )
        pooled_output = pooled_output.squeeze(1)  # Remove sequence dimension

        # Final projection for pooled embeddings
        pooled_output = self.pooled_projection(pooled_output)

        return compressed_sequence, pooled_output
