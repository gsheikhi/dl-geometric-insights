import json
import re
from collections import Counter

class Vocabulary:
    def __init__(self, sentences: list[str], specials: list[str], min_count: int = 1):
        tokens = [t for s in sentences for t in self._split(s)]
        counts = Counter(tokens)
        sp     = [s for s in specials if s]
        words  = sp + [
            w for w, c in counts.most_common()
            if w not in sp and c >= min_count
        ]
        self.w2i: dict[str, int] = {w: i for i, w in enumerate(words)}
        self.i2w: dict[int, str] = {i: w for w, i in self.w2i.items()}
        self.unk_id: int = self.w2i.get(sp[1], 0) if len(sp) > 1 else 0

    @staticmethod
    def _split(s: str) -> list[str]:
        return re.findall(r"\w+|[^\w\s]", s.lower())

    def __len__(self):
        return len(self.w2i)

    def encode(self, token: str) -> int:
        return self.w2i.get(token, self.unk_id)

    def to_json(self, path: str):
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.w2i, f, ensure_ascii=False, indent=2)

    @classmethod
    def from_json(cls, path: str) -> "Vocabulary":
        obj = cls.__new__(cls)
        with open(path, encoding="utf-8") as f:
            obj.w2i = json.load(f)
        obj.i2w    = {i: w for w, i in obj.w2i.items()}
        obj.unk_id = 0
        return obj