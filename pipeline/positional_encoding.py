import numpy as np
from pathlib import Path


class RotaryPositionalEncoding:
    """
    Rotary Position Embedding (RoPE).

    Each pair of embedding dimensions (d0, d1), (d2, d3), ... is treated as a 2D
    plane. A token at position p rotates that plane by angle p * θ_k, where
    θ_k = 1 / (base^(2k/dim)) and k is the pair index.

    The additive pe matrix is provided for visualisation only.
    """

    def __init__(self, dim: int, max_len: int = 25, base: float = 10_000.0):
        assert dim % 2 == 0, "Embedding dim must be even for RoPE."
        self.dim     = dim
        self.max_len = max_len
        self.base    = base

        k          = np.arange(dim // 2, dtype=np.float64)
        self.theta = 1.0 / (base ** (2 * k / dim))           # (dim/2,)

        positions  = np.arange(max_len, dtype=np.float64)
        angles     = np.outer(positions, self.theta)          # (max_len, dim/2)
        self.cos   = np.cos(angles)                           # (max_len, dim/2)
        self.sin   = np.sin(angles)                           # (max_len, dim/2)

        self.pe    = self._build_additive_pe()

    # ------------------------------------------------------------------

    def _rotate_half(self, x: np.ndarray) -> np.ndarray:
        x1 = x[..., 0::2]
        x2 = x[..., 1::2]
        return np.stack([-x2, x1], axis=-1).reshape(x.shape)

    def apply(self, x: np.ndarray) -> np.ndarray:
        """
        Apply RoPE to token vectors.
        x : (seq_len, dim)  or  (batch, seq_len, dim)
        """
        seq_len  = x.shape[-2]
        cos_full = np.repeat(self.cos[:seq_len], 2, axis=-1)  # (seq_len, dim)
        sin_full = np.repeat(self.sin[:seq_len], 2, axis=-1)
        return x * cos_full + self._rotate_half(x) * sin_full

    def _build_additive_pe(self) -> np.ndarray:
        unit = np.ones((self.max_len, self.dim), dtype=np.float64)
        return self.apply(unit)                                # (max_len, dim)

    # ------------------------------------------------------------------

    def save(self, path: str | Path):
        path = Path(path)
        np.savez(str(path), pe=self.pe, cos=self.cos, sin=self.sin, theta=self.theta)
        print(f"PE saved --> {path}.npz")

    @classmethod
    def load(cls, path: str | Path, dim: int, max_len: int = 25) -> "RotaryPositionalEncoding":
        obj        = cls.__new__(cls)
        data       = np.load(str(path), allow_pickle=True)
        obj.dim    = dim
        obj.max_len = max_len
        obj.base   = 10_000.0
        obj.pe     = data["pe"]
        obj.cos    = data["cos"]
        obj.sin    = data["sin"]
        obj.theta  = data["theta"]
        return obj