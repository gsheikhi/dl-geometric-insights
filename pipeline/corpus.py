import json

class Corpus:
    def __init__(self, path: str):
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        self.en = [entry["en"] for entry in data]
        self.tr = [entry["tr"] for entry in data]

    def __len__(self):
        return len(self.en)