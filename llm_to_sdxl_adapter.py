import torch
import torch.nn as nn
import logging

logger = logging.getLogger("LLM-SDXL-Adapter")


def pad_to_length(tensor, target_length, dim=1, value=0):
    """Universal function for padding tensors"""
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


class LLMToSDXLAdapter(nn.Module):
    """
    Universal adapter for converting any LLM embeddings to SDXL format
    Supports various LLM architectures (Gemma, Llama, Mistral, etc.)
    """
    def __init__(self,
                 llm_dim=1152,              # Changed from gemma_dim to llm_dim
                 sdxl_seq_dim=2048,
                 sdxl_pooled_dim=1280,
                 max_input_len=512,
                 target_seq_len=308,
                 n_wide_blocks=3,        # Blocks BEFORE compression
                 n_narrow_blocks=3,      # Blocks AFTER compression
                 num_heads=16,
                 dropout=0):
        super().__init__()

        self.max_input_len = max_input_len
        self.target_seq_len = target_seq_len
        self.num_heads = num_heads

        # Projections
        if llm_dim != sdxl_seq_dim:
            self.seq_projection = nn.Linear(llm_dim, sdxl_seq_dim)

        # Positional embeddings for full sequence
        self.input_position_embeddings = nn.Parameter(
            torch.randn(1, max_input_len, sdxl_seq_dim)
        )
        # Positional embeddings for compressed sequence
        self.output_position_embeddings = nn.Parameter(
            torch.randn(1, target_seq_len, sdxl_seq_dim)
        )
        
        # Wide blocks - processing full sequence (512 tokens)
        self.wide_attention_blocks = nn.ModuleList([
            TransformerBlock(sdxl_seq_dim, num_heads=num_heads, dropout=dropout)
            for _ in range(n_wide_blocks)
        ])

        # Compression: Cross-attention with learnable queries
        self.compression_queries = nn.Parameter(
            torch.randn(1, target_seq_len, sdxl_seq_dim)
        )
        self.compression_attention = nn.MultiheadAttention(
            embed_dim=sdxl_seq_dim,
            num_heads=num_heads,
            batch_first=True,
            dropout=dropout
        )
        # Norm layer after compression for stability
        self.compression_norm = nn.LayerNorm(sdxl_seq_dim)
        # Optional gate mechanism for weighting information
        self.compression_gate = nn.Sequential(
            nn.Linear(sdxl_seq_dim * 2, sdxl_seq_dim),
            nn.Sigmoid()
        )

        # Narrow blocks - processing compressed sequence (308 tokens)
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

    def forward(self, llm_hidden_states, attention_mask=None):
        batch_size, seq_len, _ = llm_hidden_states.shape

        # Project to target dimension
        if self.seq_projection:
            hidden_states = self.seq_projection(llm_hidden_states)
        else:
            hidden_states = llm_hidden_states  

        # Padding/truncation to max_input_len
        if seq_len > self.max_input_len:
            hidden_states = hidden_states[:, :self.max_input_len, :]
            if attention_mask is not None:
                attention_mask = attention_mask[:, :self.max_input_len]
        else:
            if seq_len < self.max_input_len:
                hidden_states = pad_to_length(hidden_states, self.max_input_len, dim=1)
                if attention_mask is not None:
                    attention_mask = pad_to_length(attention_mask, self.max_input_len, dim=1, value=0)
                else:
                    attention_mask = torch.ones(batch_size, self.max_input_len, device=hidden_states.device)
                    attention_mask[:, seq_len:] = 0

        # Add positional embeddings
        hidden_states = hidden_states + self.input_position_embeddings

        # ===== STAGE 1: Wide Processing (full sequence) =====
        for block in self.wide_attention_blocks:
            hidden_states = block(hidden_states, attention_mask)

        # ===== STAGE 2: Compression (512 -> 308) =====
        # Prepare queries for compression
        queries = self.compression_queries.expand(batch_size, -1, -1)

        # Cross-attention for compression
        if attention_mask is not None:
            key_padding_mask = ~attention_mask.bool()
        else:
            key_padding_mask = None

        compressed_sequence, compression_weights = self.compression_attention(
            queries,
            hidden_states,
            hidden_states,
            key_padding_mask=key_padding_mask,
            need_weights=True,
            average_attn_weights=False  # Get weights for each head
        )

        # Optional: Gate mechanism for mixing with queries
        gate_input = torch.cat([queries, compressed_sequence], dim=-1)
        gate_weights = self.compression_gate(gate_input)
        compressed_sequence = gate_weights * compressed_sequence + (1 - gate_weights) * queries

        # Apply normalization
        compressed_sequence = self.compression_norm(compressed_sequence)

        # Add output positional embeddings
        compressed_sequence = compressed_sequence + self.output_position_embeddings

        # ===== STAGE 3: Narrow Processing (compressed sequence) =====
        for block in self.narrow_attention_blocks:
            compressed_sequence = block(compressed_sequence)

        # ===== STAGE 4: Pooling for Vector Embeddings =====
        # Pool the compressed sequence for vector embeddings
        pooling_tokens = self.pooling_token.expand(batch_size, -1, -1)
        pooled_output, pooling_weights = self.pooling_attention(
            pooling_tokens,
            compressed_sequence,
            compressed_sequence,
            need_weights=False
        )
        pooled_output = pooled_output.squeeze(1)  # Remove sequence dimension

        # Final projection for pooled embeddings
        pooled_output = self.pooled_projection(pooled_output)

        return compressed_sequence, pooled_output 