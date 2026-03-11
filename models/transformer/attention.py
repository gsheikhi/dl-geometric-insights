"""
models/transformer/attention.py
Multi-head self-attention and cross-attention with activation capture.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % num_heads == 0
        self.d_model   = d_model
        self.num_heads = num_heads
        self.d_k       = d_model // num_heads

        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=False)
        self.W_o = nn.Linear(d_model, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

        # Captured intermediate tensors for visualisation
        self.cache: dict = {}

    def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
        B, T, _ = x.shape
        return x.view(B, T, self.num_heads, self.d_k).transpose(1, 2)

    def _merge_heads(self, x: torch.Tensor) -> torch.Tensor:
        B, H, T, _ = x.shape
        return x.transpose(1, 2).contiguous().view(B, T, self.d_model)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        capture: bool = False,
    ) -> torch.Tensor:
        Q = self.W_q(query)
        K = self.W_k(key)
        V = self.W_v(value)

        if capture:
            self.cache["Q"] = Q.detach().cpu()
            self.cache["K"] = K.detach().cpu()
            self.cache["V"] = V.detach().cpu()

        Qh = self._split_heads(Q)
        Kh = self._split_heads(K)
        Vh = self._split_heads(V)

        scores = (Qh @ Kh.transpose(-2, -1)) / math.sqrt(self.d_k)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float("-inf"))
        attn = self.dropout(F.softmax(scores, dim=-1))

        if capture:
            self.cache["attn_weights"] = attn.detach().cpu()

        out = self._merge_heads(attn @ Vh)
        out = self.W_o(out)

        if capture:
            self.cache["attn_out"] = out.detach().cpu()

        return out