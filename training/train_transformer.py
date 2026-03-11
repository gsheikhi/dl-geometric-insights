"""
training/train_transformer.py
End-to-end training + geometric visualisation for EN-->TR transformer.

Usage:
    python -m training.train_transformer
"""

import json
import sys
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from models.transformer.model import Transformer
from visualiser.transformer_visualiser import (
    CorpusActivationCollector,
    CorpusVisualiser,
    _ordered_stage_labels,
)


# ─────────────────────────────────────────────────────────────────────────────
# Dataset
# ─────────────────────────────────────────────────────────────────────────────

class TranslationDataset(Dataset):
    def __init__(self, en_seqs: list, tr_seqs: list, pad_idx: int = 0, max_len: int = 20):
        self.en      = [s[:max_len] for s in en_seqs]
        self.tr      = [s[:max_len] for s in tr_seqs]
        self.pad_idx = pad_idx

    def __len__(self):
        return len(self.en)

    def __getitem__(self, idx):
        return self.en[idx], self.tr[idx]

    def _pad(self, seqs):
        L = max(len(s) for s in seqs)
        return torch.tensor(
            [s + [self.pad_idx] * (L - len(s)) for s in seqs], dtype=torch.long
        )

    def collate(self, batch):
        en, tr = zip(*batch)
        return self._pad(en), self._pad(tr)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def load_json(path: str):
    with open(path) as f:
        return json.load(f)


def load_npz_embeddings(path: str) -> np.ndarray:
    data = np.load(path, allow_pickle=True)
    return data["embeddings"].astype(np.float32)


def load_npz_pe(path: str) -> np.ndarray:
    data = np.load(path, allow_pickle=True)
    return data["pe"].astype(np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# Training
# ─────────────────────────────────────────────────────────────────────────────

def train(model_cfg_path: str, train_cfg_path: str):
    model_cfg = load_json(model_cfg_path)
    train_cfg = load_json(train_cfg_path)
    data_cfg  = train_cfg["data"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    en_emb = load_npz_embeddings(data_cfg["en_embeddings_path"])
    tr_emb = load_npz_embeddings(data_cfg["tr_embeddings_path"])
    en_pe  = load_npz_pe(data_cfg["en_rope_pe_path"])
    tr_pe  = load_npz_pe(data_cfg["tr_rope_pe_path"])

    en_vocab = load_json(data_cfg["en_vocab_path"])
    tr_vocab = load_json(data_cfg["tr_vocab_path"])
    en_i2w   = {v: k for k, v in en_vocab.items()}
    tr_i2w   = {v: k for k, v in tr_vocab.items()}

    en_seqs = load_json(data_cfg["en_tokenised_path"])
    tr_seqs = load_json(data_cfg["tr_tokenised_path"])

    model_cfg["en_vocab_size"] = en_emb.shape[0]
    model_cfg["tr_vocab_size"] = tr_emb.shape[0]

    dataset = TranslationDataset(en_seqs, tr_seqs, pad_idx=model_cfg["pad_idx"])
    loader  = DataLoader(dataset, batch_size=train_cfg["batch_size"],
                         shuffle=True, collate_fn=dataset.collate)

    model = Transformer(en_emb, tr_emb, model_cfg, en_pe=en_pe, tr_pe=tr_pe).to(device)
    print(f"Trainable parameters: "
          f"{sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    optimizer = torch.optim.Adam(
        [p for p in model.parameters() if p.requires_grad],
        lr=train_cfg["learning_rate"]
    )
    criterion = nn.CrossEntropyLoss(ignore_index=model_cfg["pad_idx"])

    out_dir = Path(train_cfg["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, train_cfg["epochs"] + 1):
        model.train()
        epoch_loss = 0.0
        for src, tgt in loader:
            src, tgt = src.to(device), tgt.to(device)
            tgt_in   = tgt[:, :-1]
            tgt_out  = tgt[:, 1:]
            logits   = model(src, tgt_in)
            loss     = criterion(
                logits.reshape(-1, model_cfg["tr_vocab_size"]),
                tgt_out.reshape(-1)
            )
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), train_cfg["clip_grad_norm"])
            optimizer.step()
            epoch_loss += loss.item()

        if epoch % train_cfg["eval_every"] == 0:
            print(f"Epoch {epoch:4d}/{train_cfg['epochs']}  "
                  f"loss={epoch_loss/len(loader):.4f}")

        if epoch % train_cfg["save_every"] == 0:
            torch.save(model.state_dict(), out_dir / "checkpoint.pt")

    torch.save(model.state_dict(), out_dir / "checkpoint_final.pt")
    print("Training complete.")

    # ─────────────────────────────────────────────────────────────────────────
    # Corpus-wide geometric visualisation
    # ─────────────────────────────────────────────────────────────────────────
    print("\nCollecting activations over full corpus…")
    model.eval()

    special_ids = {model_cfg["pad_idx"], 1, model_cfg["sos_idx"],
                   model_cfg["eos_idx"], en_vocab.get("<SEP>", 4)}
    n_enc       = model_cfg["num_encoder_blocks"]
    n_dec       = model_cfg["num_decoder_blocks"]
    stage_order = _ordered_stage_labels(n_enc, n_dec)
    collector   = CorpusActivationCollector(special_ids, n_enc=n_enc, n_dec=n_dec)

    with torch.no_grad():
        for en_ids, tr_ids in zip(en_seqs, tr_seqs):
            src = torch.tensor([en_ids], dtype=torch.long).to(device)
            tgt = torch.tensor([tr_ids[:-1]], dtype=torch.long).to(device)
            model(src, tgt, capture=True)

            # pe_embedding: en_emb[tid] + en_pe[raw_pos] for every token
            # including specials, so _content_pairs can index by raw_pos correctly
            pe_row = np.array([
                en_emb[tid] + en_pe[raw_pos]
                for raw_pos, tid in enumerate(en_ids)
                if raw_pos < len(en_pe)
            ], dtype=np.float32)

            collector.collect(
                pe_embeddings  = pe_row,
                en_ids         = en_ids,
                tr_ids         = tr_ids[:-1],
                en_i2w         = en_i2w,
                tr_i2w         = tr_i2w,
                encoder_caches = model.get_encoder_caches(),
                decoder_caches = model.get_decoder_caches(),
            )

    stage_data = collector.get()
    plots_dir  = Path(train_cfg["plots_dir"])
    corpus_vis = CorpusVisualiser(out_dir=plots_dir, perplexity=15)

    print("\nFull corpus stage plots…")
    corpus_vis.full_corpus_stages(stage_data, stage_order,
                                  title_prefix="Full corpus | ")

    print(f"\nAll plots saved to: {plots_dir}")


# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    train(
        model_cfg_path="models/transformer/model_config.json",
        train_cfg_path="training/transformer_train_config.json",
    )