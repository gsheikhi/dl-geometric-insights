import json
import numpy as np
from .vocabulary import Vocabulary
from pathlib import Path


class SkipGramEmbedding:
    def __init__(self, vocab_size: int, dim: int = 100, window: int = 5,
                 lr: float = 0.01, epochs: int = 1000):
        self.vocab_size = vocab_size
        self.dim        = dim
        self.window     = window
        self.lr         = lr
        self.epochs     = epochs
        rng     = np.random.default_rng(0)
        self.W  = rng.normal(0, 0.01, (vocab_size, dim))
        self.W_ = rng.normal(0, 0.01, (vocab_size, dim))

    @staticmethod
    def _sigmoid(x: np.ndarray) -> np.ndarray:
        return 1.0 / (1.0 + np.exp(-np.clip(x, -10, 10)))

    def _build_pairs(self, sequences: list[list[int]]) -> list[tuple[int, int]]:
        pairs = []
        for seq in sequences:
            for i, c in enumerate(seq):
                lo = max(0, i - self.window)
                hi = min(len(seq), i + self.window + 1)
                for j in range(lo, hi):
                    if j != i:
                        pairs.append((c, seq[j]))
        return pairs

    def fit(self, sequences: list[list[int]], special_ids: set[int] | None = None):
        """
        Train embeddings.
        special_ids: token indices (pad, unk, sos, eos, sep) excluded from
                     negative sampling so specials don't pollute content vectors.
        """
        pairs      = self._build_pairs(sequences)
        rng        = np.random.default_rng(42)
        sample_ids = (
            [i for i in range(self.vocab_size) if i not in special_ids]
            if special_ids else list(range(self.vocab_size))
        )
        sample_ids = np.array(sample_ids, dtype=np.int64)

        for epoch in range(self.epochs):
            rng.shuffle(pairs)
            for c, ctx in pairs:
                s   = self._sigmoid(self.W[c] @ self.W_[ctx])
                g   = self.lr * (1 - s)
                dW  = g * self.W_[ctx].copy()
                self.W_[ctx] += g * self.W[c]
                self.W[c]    += dW

                neg = sample_ids[rng.integers(0, len(sample_ids))]
                sn  = self._sigmoid(self.W[c] @ self.W_[neg])
                gn  = self.lr * sn
                dWn = gn * self.W_[neg].copy()
                self.W_[neg] -= gn * self.W[c]
                self.W[c]    -= dWn

            if (epoch + 1) % 100 == 0:
                print(f"  epoch {epoch+1}/{self.epochs}")

    @property
    def embeddings(self) -> np.ndarray:
        return self.W


class EmbeddingStore:
    @staticmethod
    def save(embeddings: np.ndarray, vocab: Vocabulary, stem: str | Path):
        stem = Path(stem)
        np.savez(
            str(stem) + ".npz",
            embeddings=embeddings,
            words=np.array([vocab.i2w[i] for i in range(len(vocab))], dtype=object),
        )
        mapping = {vocab.i2w[i]: embeddings[i].tolist() for i in range(len(vocab))}
        with open(str(stem) + ".json", "w", encoding="utf-8") as f:
            json.dump(mapping, f, ensure_ascii=False, indent=2)
        print(f"Saved --> {stem}.npz  |  {stem}.json")

    @staticmethod
    def load_npz(path: str | Path) -> tuple[np.ndarray, list[str]]:
        data = np.load(path, allow_pickle=True)
        return data["embeddings"], list(data["words"])
