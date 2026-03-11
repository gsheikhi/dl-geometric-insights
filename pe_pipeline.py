"""
pe_pipeline.py
--------------
Builds RoPE positional encodings and generates visualisation plots.

Usage:
  python pe_pipeline.py                   # skip steps whose outputs exist
  python pe_pipeline.py --force           # rerun everything
  python pe_pipeline.py --dim_pair 2      # plot dims 2 & 3 of PE (default 0)
"""
import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from pipeline import load_json, _exists
from pipeline.vocabulary import Vocabulary
from pipeline.embedding import EmbeddingStore
from pipeline.visualiser import EmbeddingVisualiser
from pipeline.positional_encoding import RotaryPositionalEncoding


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--force",    action="store_true")
    parser.add_argument("--dim_pair", type=int, default=0,
                        help="Index i: plot PE dims i and i+1 (default 0)")
    args = parser.parse_args()

    cfg      = load_json(PROJECT_ROOT / "pipeline_config.json")
    path_cfg = cfg["path_config"]
    emb_cfg  = cfg["embedding_config"]
    EMBED_DIM = emb_cfg["embedding_dim"]

    def P(key: str) -> Path:
        p = PROJECT_ROOT / path_cfg[key]
        p.parent.mkdir(parents=True, exist_ok=True)
        return p

    EN_VOCAB_PATH   = P("en_vocab_path")
    TR_VOCAB_PATH   = P("tr_vocab_path")
    EN_ENCODED_PATH = P("en_tokenised_path").with_suffix(".json")   
    TR_ENCODED_PATH = P("tr_tokenised_path").with_suffix(".json")    

    EMBEDDING_STEM          = P("embeddings_stem")
    EMBEDDING_VISUALS_STEM  = P("embedding_visuals_stem")

    EN_PE_NPZ   = EMBEDDING_STEM / "en_rope_pe.npz"
    TR_PE_NPZ   = EMBEDDING_STEM / "tr_rope_pe.npz"
    PLOT_PE_2D  = EMBEDDING_VISUALS_STEM / "en_pe_2d"
    PLOT_EMB_2D = EMBEDDING_VISUALS_STEM / "en_emb_pe_2d.png"
    PLOT_EMB_3D = EMBEDDING_VISUALS_STEM / "en_emb_pe_3d.png"

    # -- vocab --
    en_vocab = Vocabulary.from_json(EN_VOCAB_PATH)
    print(f"EN vocab: {len(en_vocab)} tokens")
    tr_vocab = Vocabulary.from_json(TR_VOCAB_PATH)
    print(f"TR vocab: {len(tr_vocab)} tokens")

    # -- encoded sequences --
    with open(EN_ENCODED_PATH, encoding="utf-8") as f:
        en_encoded: list[list[int]] = json.load(f)
    print(f"Loaded {len(en_encoded)} encoded sequences")

    with open(TR_ENCODED_PATH, encoding="utf-8") as f:
        tr_encoded: list[list[int]] = json.load(f)
    print(f"Loaded {len(tr_encoded)} encoded sequences")

    # -- embeddings --
    en_embeddings, _ = EmbeddingStore.load_npz(EMBEDDING_STEM / "en_embeddings.npz")
    print(f"English Embeddings: {en_embeddings.shape}")
    tr_embeddings, _ = EmbeddingStore.load_npz(EMBEDDING_STEM / "tr_embeddings.npz")
    print(f"Turkish Embeddings: {tr_embeddings.shape}")

    # -- RoPE --
    if args.force or not _exists(EN_PE_NPZ, TR_PE_NPZ):
        print("Building RoPE positional encodings...")
        en_rope = RotaryPositionalEncoding(dim=EMBED_DIM, max_len=max(map(len, en_encoded)))
        en_rope.save(EN_PE_NPZ)
        tr_rope = RotaryPositionalEncoding(dim=EMBED_DIM, max_len=max(map(len, tr_encoded)))
        tr_rope.save(TR_PE_NPZ)
    else:
        print("Loading RoPE from cache...")
        en_rope = RotaryPositionalEncoding.load(EN_PE_NPZ, dim=EMBED_DIM, max_len=max(map(len, en_encoded)))
        tr_rope = RotaryPositionalEncoding.load(TR_PE_NPZ, dim=EMBED_DIM, max_len=max(map(len, tr_encoded)))
    print(f"ENglish PE matrix: {en_rope.pe.shape}  (positions × dim)")
    print(f"TRglish PE matrix: {tr_rope.pe.shape}  (positions × dim)")

    # -- plots --
    tok_cfg     = cfg["tokeniser_config"]
    special_ids = {
        en_vocab.encode(tok_cfg[k])
        for k in ("pad_token", "unk_token", "start_token", "end_token", "sep_token")
        if tok_cfg.get(k)
    }

    vis = EmbeddingVisualiser(en_embeddings, en_vocab)

    print(f"Plotting PE space (dims {args.dim_pair} & {args.dim_pair + 1})...")
    vis.plot_pe(en_rope.pe, i=args.dim_pair, out_path=f"{PLOT_PE_2D}_dims{args.dim_pair}_{args.dim_pair + 1}.png")

    print("Plotting Emb+PE space (2D t-SNE)...")
    vis.plot_positioned_emb_2d(en_rope.pe, en_encoded, en_vocab.i2w,
                                special_ids=special_ids, out_path=PLOT_EMB_2D)

    print("Plotting Emb+PE space (3D t-SNE)...")
    vis.plot_positioned_emb_3d(en_rope.pe, en_encoded, en_vocab.i2w,
                                special_ids=special_ids, out_path=PLOT_EMB_3D)

    print("\nDone.")