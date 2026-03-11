"""
models/transformer/blocks.py
Encoder and Decoder blocks with per-sublayer activation capture.
"""

import torch
import torch.nn as nn
from .attention import MultiHeadAttention
from .ffn import FeedForward
from typing import Optional


class EncoderBlock(nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.self_attn  = MultiHeadAttention(d_model, num_heads, dropout)
        self.ffn        = FeedForward(d_model, d_ff, dropout)
        self.norm1      = nn.LayerNorm(d_model)
        self.norm2      = nn.LayerNorm(d_model)
        self.dropout    = nn.Dropout(dropout)
        self.cache: dict = {}

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        capture: bool = False,
    ) -> torch.Tensor:
        if capture:
            self.cache["input"] = x.detach().cpu()

        # Self-attention + residual
        attn_out = self.self_attn(x, x, x, mask=mask, capture=capture)
        x = self.norm1(x + self.dropout(attn_out))
        if capture:
            self.cache.update(self.self_attn.cache)
            self.cache["post_attn_norm"] = x.detach().cpu()

        # FFN + residual
        ffn_out = self.ffn(x, capture=capture)
        x = self.norm2(x + self.dropout(ffn_out))
        if capture:
            self.cache.update(self.ffn.cache)
            self.cache["output"] = x.detach().cpu()

        return x


class DecoderBlock(nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.self_attn  = MultiHeadAttention(d_model, num_heads, dropout)
        self.cross_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.ffn        = FeedForward(d_model, d_ff, dropout)
        self.norm1      = nn.LayerNorm(d_model)
        self.norm2      = nn.LayerNorm(d_model)
        self.norm3      = nn.LayerNorm(d_model)
        self.dropout    = nn.Dropout(dropout)
        self.cache: dict = {}

    def forward(
        self,
        x: torch.Tensor,
        enc_out: torch.Tensor,
        tgt_mask: Optional[torch.Tensor] = None,
        src_mask: Optional[torch.Tensor] = None,
        capture: bool = False,
        en_labels: Optional[list] = None,
        tr_labels: Optional[list] = None,
    ) -> torch.Tensor:
        if capture:
            self.cache["input"] = x.detach().cpu()
            self.cache["en_labels"] = en_labels
            self.cache["tr_labels"] = tr_labels

        # Masked self-attention
        sa_out = self.self_attn(x, x, x, mask=tgt_mask, capture=capture)
        x = self.norm1(x + self.dropout(sa_out))
        if capture:
            self.cache["self_attn_Q"] = self.self_attn.cache.get("Q")
            self.cache["self_attn_K"] = self.self_attn.cache.get("K")
            self.cache["self_attn_V"] = self.self_attn.cache.get("V")
            self.cache["self_attn_out"] = self.self_attn.cache.get("attn_out")
            self.cache["post_self_attn_norm"] = x.detach().cpu()

        # Cross-attention: query=decoder, key/value=encoder
        ca_out = self.cross_attn(x, enc_out, enc_out, mask=src_mask, capture=capture)
        x = self.norm2(x + self.dropout(ca_out))
        if capture:
            self.cache["cross_attn_Q"]   = self.cross_attn.cache.get("Q")
            self.cache["cross_attn_K"]   = self.cross_attn.cache.get("K")
            self.cache["cross_attn_V"]   = self.cross_attn.cache.get("V")
            self.cache["cross_attn_out"] = self.cross_attn.cache.get("attn_out")
            self.cache["post_cross_attn_norm"] = x.detach().cpu()

        # FFN
        ffn_out = self.ffn(x, capture=capture)
        x = self.norm3(x + self.dropout(ffn_out))
        if capture:
            self.cache["ffn_hidden"] = self.ffn.cache.get("ffn_hidden")
            self.cache["ffn_out"]    = self.ffn.cache.get("ffn_out")
            self.cache["output"]     = x.detach().cpu()

        return x