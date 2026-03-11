import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from .vocabulary import Vocabulary


class EmbeddingVisualiser:
    def __init__(self, embeddings: np.ndarray, vocab: Vocabulary):
        self.embeddings   = embeddings
        self.vocab        = list(vocab.i2w.items())
        self.perplexity   = 15
        self.max_iter     = 2000
        self.random_state = 42

    # ------------------------------------------------------------------
    # Original embedding plots
    # ------------------------------------------------------------------

    def plot2d(self, out_path: str = "embedding_2d.png"):
        reduced = TSNE(n_components=2,
                       perplexity=self.perplexity,
                       max_iter=self.max_iter,
                       random_state=self.random_state
        ).fit_transform(self.embeddings)
        fig = plt.figure(figsize=(14, 10))
        ax  = fig.add_subplot(111)
        ax.scatter(reduced[:, 0], reduced[:, 1],
                   s=40, alpha=0.7, c=range(len(reduced)), cmap="tab20")
        for idx, word in self.vocab:
            ax.text(reduced[idx, 0], reduced[idx, 1],
                    word, fontsize=10, alpha=0.85)
        ax.set_title("English Embedding Space (t-SNE 2D)")
        ax.set_xlabel("Dim 1"); ax.set_ylabel("Dim 2")
        plt.tight_layout()
        plt.savefig(out_path, dpi=150)
        plt.close()
        print(f"Plot saved --> {out_path}")

    def plot3d(self, out_path: str = "embedding_3d.png"):
        reduced = TSNE(n_components=3,
                       perplexity=self.perplexity,
                       max_iter=self.max_iter,
                       random_state=self.random_state
        ).fit_transform(self.embeddings)
        fig = plt.figure(figsize=(14, 10))
        ax  = fig.add_subplot(111, projection="3d")
        ax.scatter(reduced[:, 0], reduced[:, 1], reduced[:, 2],
                   s=40, alpha=0.7, c=range(len(reduced)), cmap="tab20")
        for idx, word in self.vocab:
            ax.text(reduced[idx, 0], reduced[idx, 1], reduced[idx, 2],
                    word, fontsize=10, alpha=0.85)
        ax.set_title("English Embedding Space (t-SNE 3D)")
        ax.set_xlabel("Dim 1"); ax.set_ylabel("Dim 2"); ax.set_zlabel("Dim 3")
        plt.tight_layout()
        plt.savefig(out_path, dpi=150)
        plt.close()
        print(f"Plot saved --> {out_path}")

    # ------------------------------------------------------------------
    # PE space — raw dimensions, no reduction
    # pe  : (max_len, dim)   labels = 1…max_len
    # i   : first of the two dimensions to plot (plots dim i and dim i+1)
    # ------------------------------------------------------------------

    def plot_pe(self, pe: np.ndarray, i: int = 0,
                out_path: str = "pe_2d.png"):
        assert i + 1 < pe.shape[1], f"dim index {i} out of range for dim={pe.shape[1]}"
        labels = list(range(1, len(pe) + 1))
        fig, ax = plt.subplots(figsize=(10, 8))
        ax.scatter(pe[:, i], pe[:, i + 1],
                   s=50, alpha=0.8, c=labels, cmap="plasma")
        for pos, lbl in enumerate(labels):
            ax.annotate(str(lbl), (pe[pos, i], pe[pos, i + 1]),
                        fontsize=9, alpha=0.85,
                        textcoords="offset points", xytext=(4, 4))
        ax.set_title(f"RoPE Positional Encoding — dims {i} & {i+1} (Raw)")
        ax.set_xlabel(f"Dim {i}"); ax.set_ylabel(f"Dim {i+1}")
        plt.tight_layout()
        plt.savefig(out_path, dpi=150)
        plt.close()
        print(f"Plot saved --> {out_path}")

    # ------------------------------------------------------------------
    # Embedding + PE space — t-SNE reduction, labels = "word@position"
    # special_ids: token ids to skip (specials + unk, i.e. rare words
    #              that were collapsed to <UNK> by min_count filtering)
    # ------------------------------------------------------------------

    def _build_positioned_embeddings(
        self,
        pe: np.ndarray,
        encoded_sequences: list[list[int]],
        vocab_i2w: dict[int, str],
        special_ids: set[int],
    ) -> tuple[np.ndarray, list[str]]:
        seen  = {}
        max_p = len(pe)
        for seq in encoded_sequences:
            for pos, tok_id in enumerate(seq):
                if pos >= max_p:
                    break
                if tok_id in special_ids:
                    continue
                key = (tok_id, pos)
                if key not in seen:
                    word      = vocab_i2w.get(tok_id, f"id{tok_id}")
                    seen[key] = (self.embeddings[tok_id] + pe[pos],
                                 f"{word}@{pos + 1}")
        vecs   = np.array([v for v, _ in seen.values()])
        labels = [l for _, l in seen.values()]
        return vecs, labels

    def plot_positioned_emb_2d(
        self,
        pe: np.ndarray,
        encoded_sequences: list[list[int]],
        vocab_i2w: dict[int, str],
        special_ids: set[int] = frozenset({0, 1, 2, 3, 4}),
        out_path: str = "emb_pe_2d.png",
    ):
        vecs, labels = self._build_positioned_embeddings(
            pe, encoded_sequences, vocab_i2w, special_ids
        )
        reduced = TSNE(n_components=2,
                       perplexity=min(self.perplexity, len(vecs) - 1),
                       max_iter=self.max_iter,
                       random_state=self.random_state
        ).fit_transform(vecs)
        fig, ax = plt.subplots(figsize=(14, 10))
        ax.scatter(reduced[:, 0], reduced[:, 1],
                   s=35, alpha=0.7, c=range(len(reduced)), cmap="tab20")
        for i, lbl in enumerate(labels):
            ax.annotate(lbl, reduced[i], fontsize=7, alpha=0.8,
                        textcoords="offset points", xytext=(3, 3))
        ax.set_title("Embedding + RoPE Space (t-SNE 2D)")
        ax.set_xlabel("Dim 1"); ax.set_ylabel("Dim 2")
        plt.tight_layout()
        plt.savefig(out_path, dpi=150)
        plt.close()
        print(f"Plot saved --> {out_path}")

    def plot_positioned_emb_3d(
        self,
        pe: np.ndarray,
        encoded_sequences: list[list[int]],
        vocab_i2w: dict[int, str],
        special_ids: set[int] = frozenset({0, 1, 2, 3, 4}),
        out_path: str = "emb_pe_3d.png",
    ):
        vecs, labels = self._build_positioned_embeddings(
            pe, encoded_sequences, vocab_i2w, special_ids
        )
        reduced = TSNE(n_components=3,
                       perplexity=min(self.perplexity, len(vecs) - 1),
                       max_iter=self.max_iter,
                       random_state=self.random_state
        ).fit_transform(vecs)
        fig = plt.figure(figsize=(14, 10))
        ax  = fig.add_subplot(111, projection="3d")
        ax.scatter(reduced[:, 0], reduced[:, 1], reduced[:, 2],
                   s=35, alpha=0.7, c=range(len(reduced)), cmap="tab20")
        for i, lbl in enumerate(labels):
            ax.text(reduced[i, 0], reduced[i, 1], reduced[i, 2],
                    lbl, fontsize=7, alpha=0.8)
        ax.set_title("Embedding + RoPE Space (t-SNE 3D)")
        ax.set_xlabel("Dim 1"); ax.set_ylabel("Dim 2"); ax.set_zlabel("Dim 3")
        plt.tight_layout()
        plt.savefig(out_path, dpi=150)
        plt.close()
        print(f"Plot saved --> {out_path}")