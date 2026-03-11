"""
models/transformer/ffn.py
Position-wise feed-forward network with activation capture.
"""

import torch
import torch.nn as nn


class FeedForward(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.relu    = nn.ReLU()
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        self.cache: dict = {}

    def forward(self, x: torch.Tensor, capture: bool = False) -> torch.Tensor:
        h = self.relu(self.linear1(x))
        if capture:
            self.cache["ffn_hidden"] = h.detach().cpu()
        out = self.dropout(self.linear2(h))
        if capture:
            self.cache["ffn_out"] = out.detach().cpu()
        return out