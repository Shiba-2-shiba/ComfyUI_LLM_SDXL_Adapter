import logging
from typing import Any, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger("LLM-SDXL-Adapter")


def pad_to_length(tensor: torch.Tensor, target_length: int, dim: int = 1, value: float = 0) -> torch.Tensor:
    """Pad or truncate ``tensor`` along ``dim`` to ``target_length`` using :func:`torch.nn.functional.pad`.

    Args:
        tensor: Input tensor.
        target_length: Desired length along ``dim``.
        dim: Dimension along which to pad.
        value: Padding value for additional elements.

    Returns:
        Tensor padded or truncated to the requested length.
    """

    current_length = tensor.size(dim)
    if current_length == target_length:
        return tensor
    if current_length > target_length:
        return tensor.narrow(dim, 0, target_length)

    pad_width = target_length - current_length
    pads = [0] * (tensor.dim() * 2)
    pads[(tensor.dim() - dim - 1) * 2 + 1] = pad_width
    return F.pad(tensor, pads, value=value)


def _init_positional_embedding(param: torch.nn.Parameter) -> None:
    nn.init.normal_(param, mean=0.0, std=0.02)


def _normalize_attention_mask(mask: Optional[torch.Tensor], batch: int, seq: int, device: torch.device) -> torch.Tensor:
    """Return a boolean attention mask with shape ``(batch, seq)`` where ``True`` denotes a valid token."""

    if mask is None:
        return torch.ones(batch, seq, dtype=torch.bool, device=device)

    if mask.dtype != torch.bool:
        mask = mask != 0

    if mask.dim() != 2 or mask.size(0) != batch:
        raise ValueError(f"attention_mask shape expected (B,S) got {tuple(mask.shape)}")

    if mask.size(1) != seq:
        if mask.size(1) > seq:
            mask = mask[:, :seq]
        else:
            pad = torch.zeros(mask.size(0), seq - mask.size(1), dtype=torch.bool, device=device)
            mask = torch.cat([mask, pad], dim=1)

    return mask


def _mask_to_key_padding(mask: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
    if mask is None:
        return None
    return (~mask).to(torch.bool)


def sdpa_with_temperature(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    *,
    temperature: float = 0.7,
    attn_mask: Optional[torch.Tensor] = None,
    key_padding_mask: Optional[torch.Tensor] = None,
    dropout_p: float = 0.0,
    training: bool = False,
) -> torch.Tensor:
    """Apply :func:`torch.nn.functional.scaled_dot_product_attention` with temperature control."""

    scale = max(temperature, 1e-6)

    if key_padding_mask is not None:
        kp_mask = key_padding_mask[:, None, None, :]
        kp_mask = kp_mask.to(torch.bool)
        if attn_mask is None:
            attn_mask = kp_mask
        else:
            attn_mask = attn_mask.logical_or(kp_mask)

    return torch.nn.functional.scaled_dot_product_attention(
        query / scale,
        key,
        value,
        attn_mask=attn_mask,
        dropout_p=dropout_p if training else 0.0,
        is_causal=False,
    )


class TemperaturedMultiheadAttention(nn.Module):
    """Multi-head attention module that prefers SDPA with temperature and falls back to ``nn.MultiheadAttention``."""

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        *,
        temperature: float = 0.7,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.temperature = temperature
        self.dropout = dropout

        head_dim = embed_dim // num_heads
        if head_dim * num_heads != embed_dim:
            raise ValueError("embed_dim must be divisible by num_heads")
        self.head_dim = head_dim

        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

        self.fallback_attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        *,
        key_padding_mask: Optional[torch.Tensor] = None,
        need_weights: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        bsz, tgt_len, _ = query.shape

        def project(x: torch.Tensor, proj: nn.Linear) -> torch.Tensor:
            x = proj(x)
            x = x.view(bsz, -1, self.num_heads, self.head_dim)
            return x.transpose(1, 2)

        q = project(query, self.q_proj)
        k = project(key, self.k_proj)
        v = project(value, self.v_proj)

        key_padding_mask = key_padding_mask.to(torch.bool) if key_padding_mask is not None else None
        try:
            context = sdpa_with_temperature(
                q,
                k,
                v,
                temperature=self.temperature,
                attn_mask=None,
                key_padding_mask=key_padding_mask,
                dropout_p=self.dropout,
                training=self.training,
            )
            context = context.transpose(1, 2).contiguous().view(bsz, tgt_len, self.embed_dim)
            output = self.out_proj(context)
            if need_weights:
                scale = max(self.temperature, 1e-6)
                attn_scores = torch.matmul((q / scale), k.transpose(-2, -1))
                if key_padding_mask is not None:
                    mask = key_padding_mask[:, None, None, :]
                    attn_scores = attn_scores.masked_fill(mask, float('-inf'))
                attn_weights = torch.softmax(attn_scores, dim=-1)
                weights = attn_weights
            else:
                weights = None
        except RuntimeError:
            logger.debug("Falling back to torch.nn.MultiheadAttention for compression attention")
            output, weights = self.fallback_attn(
                query,
                key,
                value,
                key_padding_mask=key_padding_mask,
                need_weights=need_weights,
                average_attn_weights=False,
            )

        return output, weights


class TransformerBlock(nn.Module):
    def __init__(self, dim: int, num_heads: int = 16, mlp_ratio: float = 4.0, dropout: float = 0.0) -> None:
        super().__init__()

        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, batch_first=True, dropout=dropout)

        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, int(dim * mlp_ratio)),
            nn.GELU(),
            nn.Linear(int(dim * mlp_ratio), dim),
        )

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        normed = self.norm1(x)

        key_padding_mask = _mask_to_key_padding(mask)

        attn_out, _ = self.attn(normed, normed, normed, key_padding_mask=key_padding_mask, need_weights=False)
        x = x + attn_out

        x = x + self.mlp(self.norm2(x))

        return x


class LLMToSDXLAdapter(nn.Module):
    """Adapter module that refines LLM hidden states for SDXL consumption."""

    def __init__(
        self,
        llm_dim: int = 1152,
        sdxl_seq_dim: int = 2048,
        sdxl_pooled_dim: int = 1280,
        max_input_len: int = 512,
        target_seq_len: int = 308,
        n_wide_blocks: int = 3,
        n_narrow_blocks: int = 3,
        num_heads: int = 16,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()

        self.max_input_len = max_input_len
        self.target_seq_len = target_seq_len
        self.num_heads = num_heads
        self.pool_mode = "attn"
        self.debug_attention = False

        self.seq_projection = nn.Linear(llm_dim, sdxl_seq_dim) if llm_dim != sdxl_seq_dim else None

        self.input_position_embeddings = nn.Parameter(torch.empty(1, max_input_len, sdxl_seq_dim))
        self.output_position_embeddings = nn.Parameter(torch.empty(1, target_seq_len, sdxl_seq_dim))
        _init_positional_embedding(self.input_position_embeddings)
        _init_positional_embedding(self.output_position_embeddings)

        self.input_pos_ln = nn.LayerNorm(sdxl_seq_dim)
        self.output_pos_ln = nn.LayerNorm(sdxl_seq_dim)

        self.wide_attention_blocks = nn.ModuleList(
            [TransformerBlock(sdxl_seq_dim, num_heads=num_heads, dropout=dropout) for _ in range(n_wide_blocks)]
        )

        self.compression_queries = nn.Parameter(torch.empty(1, target_seq_len, sdxl_seq_dim))
        nn.init.xavier_uniform_(self.compression_queries)
        self.compression_attention = TemperaturedMultiheadAttention(
            embed_dim=sdxl_seq_dim,
            num_heads=num_heads,
            temperature=0.7,
            dropout=dropout,
        )
        self.compression_norm = nn.LayerNorm(sdxl_seq_dim)
        self.compression_gate = nn.Sequential(
            nn.LayerNorm(sdxl_seq_dim * 2),
            nn.Linear(sdxl_seq_dim * 2, sdxl_seq_dim),
            nn.Sigmoid(),
        )
        self.skip_alpha = nn.Parameter(torch.tensor(0.5))

        self.narrow_attention_blocks = nn.ModuleList(
            [TransformerBlock(sdxl_seq_dim, num_heads=num_heads, dropout=dropout) for _ in range(n_narrow_blocks)]
        )

        self.pooling_attention = nn.MultiheadAttention(
            embed_dim=sdxl_seq_dim,
            num_heads=num_heads,
            batch_first=True,
            dropout=dropout,
        )
        self.pooling_token = nn.Parameter(torch.empty(1, 1, sdxl_seq_dim))
        nn.init.xavier_uniform_(self.pooling_token)

        self.pooled_projection = nn.Sequential(
            nn.Linear(sdxl_seq_dim, sdxl_seq_dim),
            nn.LayerNorm(sdxl_seq_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(sdxl_seq_dim, sdxl_pooled_dim),
        )

    def forward(
        self,
        llm_hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        *,
        tokenizer: Optional[Any] = None,
        input_ids: Optional[torch.Tensor] = None,
        top_k: int = 5,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size, seq_len, _ = llm_hidden_states.shape
        device = llm_hidden_states.device

        hidden_states = self.seq_projection(llm_hidden_states) if self.seq_projection is not None else llm_hidden_states

        normalized_mask = _normalize_attention_mask(attention_mask, batch_size, seq_len, device)

        if seq_len > self.max_input_len:
            hidden_states = hidden_states[:, : self.max_input_len, :]
            normalized_mask = normalized_mask[:, : self.max_input_len]
            seq_len = self.max_input_len
        elif seq_len < self.max_input_len:
            hidden_states = pad_to_length(hidden_states, self.max_input_len, dim=1, value=0.0)
            pad = torch.zeros(batch_size, self.max_input_len - seq_len, dtype=torch.bool, device=device)
            normalized_mask = torch.cat([normalized_mask, pad], dim=1)
            seq_len = self.max_input_len

        pos_emb = self.input_position_embeddings[:, :seq_len, :]
        hidden_states = self.input_pos_ln(hidden_states + pos_emb)

        for block in self.wide_attention_blocks:
            hidden_states = block(hidden_states, normalized_mask)

        queries = self.compression_queries.expand(batch_size, -1, -1)

        key_padding_mask = _mask_to_key_padding(normalized_mask)
        compressed_sequence, compression_weights = self.compression_attention(
            queries,
            hidden_states,
            hidden_states,
            key_padding_mask=key_padding_mask,
            need_weights=self.debug_attention,
        )

        gate_input = torch.cat([queries, compressed_sequence], dim=-1)
        gate_weights = self.compression_gate(gate_input)
        mixed = gate_weights * compressed_sequence + (1 - gate_weights) * queries
        compressed_sequence = self.skip_alpha * mixed + (1 - self.skip_alpha) * queries
        compressed_sequence = self.compression_norm(compressed_sequence)

        pos_out = self.output_position_embeddings[:, : self.target_seq_len, :]
        compressed_sequence = self.output_pos_ln(compressed_sequence + pos_out)

        for block in self.narrow_attention_blocks:
            compressed_sequence = block(compressed_sequence, None)

        pooling_tokens = self.pooling_token.expand(batch_size, -1, -1)
        pooled_tensor, _ = self.pooling_attention(
            pooling_tokens,
            compressed_sequence,
            compressed_sequence,
            need_weights=False,
        )
        pooled_tensor = pooled_tensor.squeeze(1)

        if self.pool_mode == "mean":
            pooled_sequence = compressed_sequence.mean(dim=1)
            pooled_output = self.pooled_projection(pooled_sequence)
        elif self.pool_mode == "hybrid":
            pooled_sequence = compressed_sequence.mean(dim=1)
            pooled_tensor = 0.5 * pooled_tensor + 0.5 * pooled_sequence
            pooled_output = self.pooled_projection(pooled_tensor)
        else:
            pooled_output = self.pooled_projection(pooled_tensor)

        if self.debug_attention and compression_weights is not None and tokenizer is not None and input_ids is not None:
            with torch.no_grad():
                attn_scores = compression_weights.mean(dim=(1, 2))
                top_scores, top_indices = attn_scores.topk(min(top_k, attn_scores.size(-1)), dim=-1)
                ids_list = input_ids[0].tolist()
                valid = [(idx, score) for idx, score in zip(top_indices[0].tolist(), top_scores[0].tolist()) if idx < len(ids_list)]
                if valid:
                    chosen_ids = [ids_list[idx] for idx, _ in valid]
                    tokens = tokenizer.convert_ids_to_tokens(chosen_ids)  # type: ignore[arg-type]
                    chosen_scores = [float(score) for _, score in valid]
                    logger.debug(
                        "Top-%d tokens by compression attention: %s (scores: %s)",
                        len(valid),
                        tokens,
                        chosen_scores,
                    )

        return compressed_sequence, pooled_output

    def infer(
        self,
        *args: Any,
        amp_dtype: Optional[torch.dtype] = None,
        **kwargs: Any,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        with torch.no_grad():
            if amp_dtype in (torch.float16, torch.bfloat16):
                device = args[0].device if args else torch.device("cuda")
                device_type = device.type
                with torch.autocast(device_type=device_type, dtype=amp_dtype):
                    return self.forward(*args, **kwargs)
            return self.forward(*args, **kwargs)

