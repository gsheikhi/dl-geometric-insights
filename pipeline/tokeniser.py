import json
import re
import numpy as np  
from .vocabulary import Vocabulary

class Tokeniser:
    def __init__(self, vocab: Vocabulary, cfg: dict):
        self.vocab  = vocab
        self.sos    = cfg.get("start_token", "")
        self.eos    = cfg.get("end_token",   "")
        self.pad    = cfg.get("pad_token",   "")
        self.sep    = cfg.get("sep_token",   "")
        self.lower  = cfg.get("lower", True)
        self.pad_id = vocab.w2i.get(self.pad, 0)

    def _tokens(self, sentence: str) -> list[str]:
        text  = sentence.lower() if self.lower else sentence
        sents = [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]
        tokens: list[str] = []
        if self.sos:
            tokens.append(self.sos)
        for i, s in enumerate(sents):
            if i > 0 and self.sep:
                tokens.append(self.sep)
            tokens.extend(re.findall(r"\w+|[^\w\s]", s))
        if self.eos:
            tokens.append(self.eos)
        return tokens

    def encode(self, sentence: str) -> list[int]:
        return [self.vocab.encode(t) for t in self._tokens(sentence)]

    def encode_batch(self, sentences: list[str]) -> list[list[int]]:
        return [self.encode(s) for s in sentences]

    def pad_batch(self, sequences: list[list[int]]) -> np.ndarray:
        L = max(len(s) for s in sequences)
        return np.array(
            [s + [self.pad_id] * (L - len(s)) for s in sequences], dtype=np.int32
        )

    def save(self, encoded: list[list[int]], padded: np.ndarray, path_encoded: str, path_padded: str):
        with open(path_encoded, "w", encoding="utf-8") as f:
            json.dump(encoded, f)
        np.save(path_padded, padded)

    @staticmethod
    def load(path_encoded: str, path_padded: str) -> tuple[list[list[int]], np.ndarray]:
        with open(path_encoded, encoding="utf-8") as f:
            encoded = json.load(f)
        padded = np.load(path_padded)
        return encoded, padded