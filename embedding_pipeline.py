import argparse
from os import mkdir
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from pipeline.corpus import Corpus
from pipeline.vocabulary import Vocabulary
from pipeline.tokeniser import Tokeniser
from pipeline.embedding import SkipGramEmbedding, EmbeddingStore
from pipeline.visualiser import EmbeddingVisualiser
from pipeline import load_json, _exists

"""
embedding_pipeline.py
─────────────────────
Usage:
  python embedding_pipeline.py           # skip steps whose outputs exist
  python embedding_pipeline.py --force   # rerun everything
"""

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    cfg      = load_json(PROJECT_ROOT / "pipeline_config.json")
    tok_cfg  = cfg["tokeniser_config"]
    path_cfg = cfg["path_config"]
    emb_cfg  = cfg["embedding_config"]

    def P(key: str) -> Path:
        p = PROJECT_ROOT / path_cfg[key]
        p.parent.mkdir(parents=True, exist_ok=True)
        return p

    CORPUS_PATH     = P("corpus_path")
    EN_VOCAB_PATH   = P("en_vocab_path")
    TR_VOCAB_PATH   = P("tr_vocab_path")
    EN_ENCODED_PATH = P("en_tokenised_path").with_suffix(".json")   
    TR_ENCODED_PATH = P("tr_tokenised_path").with_suffix(".json")    
    EN_PADDED_PATH  = P("en_tokenised_path").with_suffix(".npy")
    TR_PADDED_PATH  = P("tr_tokenised_path").with_suffix(".npy")

    EMBEDDING_STEM          = P("embeddings_stem")
    EMBEDDING_VISUALS_STEM  = P("embedding_visuals_stem")

    EMBED_DIM = emb_cfg["embedding_dim"]
    WINDOW    = emb_cfg["window"]
    LR        = emb_cfg.get("lr", 0.01)
    EPOCHS    = emb_cfg.get("epochs", 1000)
    MIN_COUNT = emb_cfg.get("min_count", 1)

    specials = [
        tok_cfg.get("pad_token",   ""),
        tok_cfg.get("unk_token",   ""),
        tok_cfg.get("start_token", ""),
        tok_cfg.get("end_token",   ""),
        tok_cfg.get("sep_token",   ""),
    ]

    corpus = Corpus(CORPUS_PATH)
    print(f"Corpus: {len(corpus)} pairs")

    # --- Vocabulary ---
    if args.force or not _exists(EN_VOCAB_PATH, TR_VOCAB_PATH):
        print("Building vocabularies...")
        en_vocab = Vocabulary(corpus.en, specials=specials, min_count=MIN_COUNT)
        tr_vocab = Vocabulary(corpus.tr, specials=specials, min_count=MIN_COUNT)
        en_vocab.to_json(EN_VOCAB_PATH)
        tr_vocab.to_json(TR_VOCAB_PATH)
        print(f"EN vocab: {len(en_vocab)}  |  TR vocab: {len(tr_vocab)}")
    else:
        print("Loading vocabularies from cache...")
        en_vocab = Vocabulary.from_json(EN_VOCAB_PATH)
        tr_vocab = Vocabulary.from_json(TR_VOCAB_PATH)

    # --- Tokenisation ---
    if args.force or not _exists(EN_ENCODED_PATH, TR_ENCODED_PATH, EN_PADDED_PATH, TR_PADDED_PATH):
        print("Tokenising English...")
        tokeniser  = Tokeniser(en_vocab, tok_cfg)
        en_encoded = tokeniser.encode_batch(corpus.en)
        en_padded  = tokeniser.pad_batch(en_encoded)
        tokeniser.save(en_encoded, en_padded, EN_ENCODED_PATH, EN_PADDED_PATH)
        print(f"Padded EN: {en_padded.shape}")

        print("Tokenising Turkish...")
        tokeniser  = Tokeniser(tr_vocab, tok_cfg)
        tr_encoded = tokeniser.encode_batch(corpus.tr)
        tr_padded  = tokeniser.pad_batch(tr_encoded)
        tokeniser.save(tr_encoded, tr_padded, TR_ENCODED_PATH, TR_PADDED_PATH)
        print(f"Padded TR: {tr_padded.shape}")
    else:
        print("Loading tokenised data from cache...")
        en_encoded, en_padded = Tokeniser.load(EN_ENCODED_PATH, EN_PADDED_PATH)
        tr_encoded, tr_padded = Tokeniser.load(TR_ENCODED_PATH, TR_PADDED_PATH)
        print(f"Padded EN: {en_padded.shape}  |  Padded TR: {tr_padded.shape}")

    # --- Embeddings ---
    EMBEDDING_STEM.mkdir(parents=True, exist_ok=True)
    en_embeddings_path  = EMBEDDING_STEM / "en_embeddings"
    tr_embeddings_path  = EMBEDDING_STEM / "tr_embeddings"
    EMBEDDING_STEM.parent.mkdir(parents=True, exist_ok=True)

    if args.force or not _exists(en_embeddings_path.with_suffix(".npz"), tr_embeddings_path.with_suffix(".npz")):
        print("Training Skip-gram embeddings English...")
        en_special_ids = {en_vocab.encode(t) for t in specials if t}
        model = SkipGramEmbedding(
            vocab_size=len(en_vocab),
            dim=EMBED_DIM,
            window=WINDOW,
            lr=LR,
            epochs=EPOCHS,
        )
        model.fit(en_encoded, special_ids=en_special_ids)
        EmbeddingStore.save(model.embeddings, en_vocab, en_embeddings_path)
        en_embeddings = model.embeddings
    
        print("Training Skip-gram embeddings Turkish...")
        tr_special_ids = {tr_vocab.encode(t) for t in specials if t}
        model = SkipGramEmbedding(
            vocab_size=len(tr_vocab),
            dim=EMBED_DIM,
            window=WINDOW,
            lr=LR,
            epochs=EPOCHS,
        )
        model.fit(tr_encoded, special_ids=tr_special_ids)
        EmbeddingStore.save(model.embeddings, tr_vocab, tr_embeddings_path)
        tr_embeddings = model.embeddings
    else:
        print("Loading embeddings from cache...")
        en_embeddings, _ = EmbeddingStore.load_npz(en_embeddings_path.with_suffix(".npz"))
        tr_embeddings, _ = EmbeddingStore.load_npz(tr_embeddings_path.with_suffix(".npz"))

    # --- Plots ---
    EMBEDDING_VISUALS_STEM.mkdir(parents=True, exist_ok=True)
    print("Generating plots for English embeddings...")
    en_vis = EmbeddingVisualiser(en_embeddings, en_vocab)
    en_vis.plot2d(out_path = EMBEDDING_VISUALS_STEM / "en_embedding_2d.png")
    en_vis.plot3d(out_path = EMBEDDING_VISUALS_STEM / "en_embedding_3d.png")

    print("Generating plots for Turkish embeddings...")
    tr_vis = EmbeddingVisualiser(tr_embeddings, tr_vocab)
    tr_vis.plot2d(out_path = EMBEDDING_VISUALS_STEM / "tr_embedding_2d.png")
    tr_vis.plot3d(out_path = EMBEDDING_VISUALS_STEM / "tr_embedding_3d.png")