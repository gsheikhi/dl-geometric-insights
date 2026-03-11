"""
models/transformer/model.py
Encoder-decoder transformer for EN-->TR translation.
Frozen word embeddings + frozen positional offsets (from en/tr_rope_pe.npz).
No PE computation inside the model.
"""

import torch
import torch.nn as nn
import numpy as np
from .blocks import EncoderBlock, DecoderBlock
from typing import Optional


class Transformer(nn.Module):
    def __init__(
        self,
        en_embeddings: np.ndarray,
        tr_embeddings: np.ndarray,
        cfg: dict,
        en_pe: np.ndarray = None,
        tr_pe: np.ndarray = None,
    ):
        super().__init__()
        d_model   = cfg["embedding_dim"]
        num_heads = cfg["num_heads"]
        d_ff      = cfg["ffn_dim"]
        dropout   = cfg["dropout"]
        n_enc     = cfg["num_encoder_blocks"]
        n_dec     = cfg["num_decoder_blocks"]
        self.pad_idx = cfg["pad_idx"]
        self.sos_idx = cfg["sos_idx"]
        self.eos_idx = cfg["eos_idx"]

        self.en_emb = nn.Embedding.from_pretrained(
            torch.tensor(en_embeddings, dtype=torch.float32), freeze=True
        )
        self.tr_emb = nn.Embedding.from_pretrained(
            torch.tensor(tr_embeddings, dtype=torch.float32), freeze=True
        )

        self.register_buffer("en_pe", torch.tensor(en_pe, dtype=torch.float32))
        self.register_buffer("tr_pe", torch.tensor(tr_pe, dtype=torch.float32))

        self.encoder_blocks = nn.ModuleList(
            [EncoderBlock(d_model, num_heads, d_ff, dropout) for _ in range(n_enc)]
        )
        self.decoder_blocks = nn.ModuleList(
            [DecoderBlock(d_model, num_heads, d_ff, dropout) for _ in range(n_dec)]
        )

        self.output_proj = nn.Linear(d_model, cfg["tr_vocab_size"])
        self._init_weights()

    def _init_weights(self):
        for p in self.parameters():
            if p.requires_grad and p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def _add_pe(self, x: torch.Tensor, pe: torch.Tensor) -> torch.Tensor:
        T = x.size(1)
        return x + pe[:T].unsqueeze(0)

    def _pad_mask(self, seq: torch.Tensor) -> torch.Tensor:
        return (seq != self.pad_idx).unsqueeze(1).unsqueeze(2)

    def _causal_mask(self, T: int, device: torch.device) -> torch.Tensor:
        return (
            torch.tril(torch.ones(T, T, device=device, dtype=torch.bool))
            .unsqueeze(0).unsqueeze(0)
        )

    def encode(self, src: torch.Tensor, capture: bool = False) -> torch.Tensor:
        x    = self._add_pe(self.en_emb(src), self.en_pe)
        mask = self._pad_mask(src)
        for block in self.encoder_blocks:
            x = block(x, mask=mask, capture=capture)
        return x

    def decode(
        self,
        tgt: torch.Tensor,
        enc_out: torch.Tensor,
        src: torch.Tensor,
        capture: bool = False,
        en_labels: Optional[list] = None,
        tr_labels: Optional[list] = None,
    ) -> torch.Tensor:
        x        = self._add_pe(self.tr_emb(tgt), self.tr_pe)
        src_mask = self._pad_mask(src)
        T        = tgt.size(1)
        tgt_mask = self._causal_mask(T, tgt.device) & self._pad_mask(tgt)
        for block in self.decoder_blocks:
            x = block(
                x, enc_out,
                tgt_mask=tgt_mask,
                src_mask=src_mask,
                capture=capture,
                en_labels=en_labels,
                tr_labels=tr_labels,
            )
        return x

    def forward(
        self,
        src: torch.Tensor,
        tgt: torch.Tensor,
        capture: bool = False,
        en_labels: Optional[list] = None,
        tr_labels: Optional[list] = None,
    ) -> torch.Tensor:
        enc_out = self.encode(src, capture=capture)
        dec_out = self.decode(tgt, enc_out, src, capture=capture,
                              en_labels=en_labels, tr_labels=tr_labels)
        return self.output_proj(dec_out)

    def get_encoder_caches(self) -> list[dict]:
        return [b.cache for b in self.encoder_blocks]

    def get_decoder_caches(self) -> list[dict]:
        return [b.cache for b in self.decoder_blocks]