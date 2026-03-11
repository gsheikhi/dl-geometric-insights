"""
pipeline/corpus_visualiser.py

Two visualisation modes that operate on the *full corpus* rather than a
single sentence:

CorpusActivationCollector
    Runs every sentence through the model with capture=True and accumulates
    hidden-state tensors at every named stage (pe_embedding --> enc_block1_Q
    --> … --> dec_block2_output).  Skips PAD / special tokens so only
    meaningful word@position points enter the plots.

CorpusVisualiser
    Option 1 — full_corpus_stages()
        One t-SNE scatter per stage, full population of word@position points.
        Directly comparable to the existing pe_pipeline plots.

    Option 3 — animated_trajectory()
        Fits a single global t-SNE on all stages stacked together, then
        saves per-stage frames as a GIF *and* as a self-contained HTML file
        with a play/pause slider so students can scrub through the journey
        interactively.
"""

import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from sklearn.manifold import TSNE
from pathlib import Path
from typing import Optional
import io, base64


# ─────────────────────────────────────────────────────────────────────────────
# Stage ordering
# ─────────────────────────────────────────────────────────────────────────────

ENCODER_STAGE_KEYS = [
    "input",
    "Q", "K", "V",
    "attn_out",
    "post_attn_norm",
    # ffn_hidden excluded: lives in d_ff space, incompatible with residual stream
    "ffn_out",
    "output",
]

DECODER_STAGE_KEYS = [
    "input",
    "self_attn_Q", "self_attn_K", "self_attn_V",
    "self_attn_out",
    "post_self_attn_norm",
    "cross_attn_Q", "cross_attn_out",
    "post_cross_attn_norm",
    # ffn_hidden excluded: lives in d_ff space, incompatible with residual stream
    "ffn_out",
    "output",
]


def _stage_label(side: str, block: int, key: str) -> str:
    return f"{side}_b{block+1}_{key}"


def _ordered_stage_labels(n_enc: int, n_dec: int) -> list[str]:
    labels = ["pe_embedding"]
    for b in range(n_enc):
        for k in ENCODER_STAGE_KEYS:
            labels.append(_stage_label("enc", b, k))
    for b in range(n_dec):
        for k in DECODER_STAGE_KEYS:
            labels.append(_stage_label("dec", b, k))
    return labels


# ─────────────────────────────────────────────────────────────────────────────
# Collector
# ─────────────────────────────────────────────────────────────────────────────

class CorpusActivationCollector:
    """
    Accumulates word@position activations across all sentences at every stage.

    Usage
    -----
    collector = CorpusActivationCollector(special_ids, n_enc=2, n_dec=2)
    for each sentence:
        run model forward with capture=True
        collector.collect(
            pe_emb_row,          # np.ndarray (T_en, d)
            en_ids, tr_ids,
            model.get_encoder_caches(),
            model.get_decoder_caches(),
        )
    data = collector.get()   # dict[stage_label] -> {"vecs": ndarray, "labels": list[str]}
    """

    def __init__(self, special_ids: set, n_enc: int = 2, n_dec: int = 2):
        self.special_ids = special_ids
        self.n_enc       = n_enc
        self.n_dec       = n_dec
        self._buckets: dict[str, dict] = {}   # stage --> {vecs: list, labels: list}

    def _push(self, stage: str, vecs: np.ndarray, labels: list[str]):
        if stage not in self._buckets:
            self._buckets[stage] = {"vecs": {}, "labels": {}}   # keyed by label for dedup
        bucket = self._buckets[stage]
        for vec, label in zip(vecs, labels):
            if label not in bucket["labels"]:
                bucket["vecs"][label]   = vec.tolist()
                bucket["labels"][label] = label

    def _content_pairs(
        self, ids: list[int], tensor_row: np.ndarray, i2w: dict, prefix: str = ""
    ) -> tuple[list[str], np.ndarray]:
        """
        Walk ids and tensor_row together, skipping special tokens.
        Position is the 1-based index among content tokens only, matching
        the pipeline's convention.  Returns (labels, filtered_vecs) with
        guaranteed 1-to-1 alignment.
        """
        labels, vecs = [], []
        content_pos  = 1
        for raw_pos, tid in enumerate(ids):
            if tid in self.special_ids:
                continue
            if raw_pos >= len(tensor_row):
                break
            word = i2w.get(tid, f"id{tid}")
            labels.append(f"{prefix}{word}@{content_pos}")
            vecs.append(tensor_row[raw_pos])
            content_pos += 1
        return labels, np.array(vecs, dtype=np.float32)

    # ── public ────────────────────────────────────────────────────────────────

    def collect(
        self,
        pe_embeddings: np.ndarray,     # (T_en, d)  — pre-network, from the npz
        en_ids: list[int],
        tr_ids: list[int],
        en_i2w: dict,
        tr_i2w: dict,
        encoder_caches: list[dict],
        decoder_caches: list[dict],
    ):
        # ── PE+embedding stage ────────────────────────────────────────────────
        en_labels, pe_vecs = self._content_pairs(en_ids, pe_embeddings, en_i2w)
        if len(pe_vecs):
            self._push("pe_embedding", pe_vecs, en_labels)

        # ── Encoder stages ────────────────────────────────────────────────────
        for b, cache in enumerate(encoder_caches):
            for key in ENCODER_STAGE_KEYS:
                tensor = cache.get(key)
                if tensor is None:
                    continue
                arr = tensor[0].numpy()          # remove batch dim --> (T, d)
                labels, vecs = self._content_pairs(en_ids, arr, en_i2w)
                if len(vecs):
                    self._push(_stage_label("enc", b, key), vecs, labels)

        # ── Decoder stages ────────────────────────────────────────────────────
        for b, cache in enumerate(decoder_caches):
            for key in DECODER_STAGE_KEYS:
                tensor = cache.get(key)
                if tensor is None:
                    continue
                arr = tensor[0].numpy()
                labels, vecs = self._content_pairs(tr_ids, arr, tr_i2w)
                if len(vecs):
                    self._push(_stage_label("dec", b, key), vecs, labels)

    def get(self) -> dict:
        return {
            stage: {
                "vecs":   np.array(list(v["vecs"].values()),   dtype=np.float32),
                "labels": list(v["labels"].values()),
            }
            for stage, v in self._buckets.items()
            if len(v["vecs"]) >= 3
        }


# ─────────────────────────────────────────────────────────────────────────────
# t-SNE fit on a shared embedding (for trajectory consistency)
# ─────────────────────────────────────────────────────────────────────────────

def _fit_global_tsne(
    stage_data: dict,
    stage_order: list[str],
    perplexity: int = 15,
) -> dict[str, np.ndarray]:
    """
    Stack all stages into one matrix, fit a single t-SNE, then split back.
    This ensures every stage lives in the same 2D coordinate system so the
    animated trajectory is geometrically meaningful.
    """
    present   = [s for s in stage_order if s in stage_data]

    # Only keep stages whose vector dimension matches the first present stage
    ref_dim   = stage_data[present[0]]["vecs"].shape[1]
    present   = [s for s in present if stage_data[s]["vecs"].shape[1] == ref_dim]

    sizes     = [len(stage_data[s]["vecs"]) for s in present]
    all_vecs  = np.vstack([stage_data[s]["vecs"] for s in present])

    perp = min(perplexity, max(2, len(all_vecs) - 1))
    reduced = TSNE(
        n_components=2,
        perplexity=perp,
        max_iter=2000,
        random_state=42,
    ).fit_transform(all_vecs)

    result, start = {}, 0
    for stage, size in zip(present, sizes):
        result[stage] = reduced[start : start + size]
        start += size
    return result


def _fit_per_stage_tsne(
    stage_data: dict,
    perplexity: int = 15,
) -> dict[str, np.ndarray]:
    """Independent t-SNE per stage — better local structure, used for Option 1."""
    result = {}
    for stage, data in stage_data.items():
        vecs = data["vecs"]
        perp = min(perplexity, max(2, len(vecs) - 1))
        result[stage] = TSNE(
            n_components=2,
            perplexity=perp,
            max_iter=2000,
            random_state=42,
        ).fit_transform(vecs)
    return result


# ─────────────────────────────────────────────────────────────────────────────
# Colour helpers
# ─────────────────────────────────────────────────────────────────────────────

_STAGE_PALETTE = {
    "pe_embedding":       "#607D8B",
    "enc":                "#1976D2",   # blue family for encoder
    "dec_self":           "#E64A19",   # deep orange for decoder self-attn
    "dec_cross":          "#7B1FA2",   # purple for cross-attn
    "dec_ffn":            "#F57C00",   # amber for ffn
    "dec_output":         "#C62828",   # dark red for block output
}

def _stage_colour(stage: str) -> str:
    if stage == "pe_embedding":
        return _STAGE_PALETTE["pe_embedding"]
    if stage.startswith("enc"):
        return _STAGE_PALETTE["enc"]
    if "self_attn" in stage or stage.endswith("self_attn_Q") or stage.endswith("self_attn_K"):
        return _STAGE_PALETTE["dec_self"]
    if "cross" in stage:
        return _STAGE_PALETTE["dec_cross"]
    if "ffn" in stage:
        return _STAGE_PALETTE["dec_ffn"]
    return _STAGE_PALETTE["dec_output"]


# ─────────────────────────────────────────────────────────────────────────────
# CorpusVisualiser
# ─────────────────────────────────────────────────────────────────────────────

class CorpusVisualiser:

    def __init__(self, out_dir: str = "plots/transformer", perplexity: int = 15):
        self.out_dir    = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.perplexity = perplexity

    # ── Option 1 ──────────────────────────────────────────────────────────────

    def full_corpus_stages(
        self,
        stage_data: dict,
        stage_order: list[str],
        title_prefix: str = "",
    ):
        """
        One t-SNE scatter per stage, independent fit, full corpus population.
        Saved as  plots/transformer/corpus_<stage>.png
        """
        print("  Fitting per-stage t-SNE…")
        reduced = _fit_per_stage_tsne(stage_data, self.perplexity)

        for stage in stage_order:
            if stage not in reduced:
                continue
            coords = reduced[stage]
            labels = stage_data[stage]["labels"]
            colour = _stage_colour(stage)

            fig, ax = plt.subplots(figsize=(12, 9))
            ax.scatter(coords[:, 0], coords[:, 1],
                       c=colour, s=45, alpha=0.75, edgecolors="k", linewidths=0.3)
            for i, lbl in enumerate(labels):
                ax.annotate(lbl, coords[i], fontsize=6.5, alpha=0.85,
                            textcoords="offset points", xytext=(3, 3))
            nice = stage.replace("_", " ")
            ax.set_title(f"{title_prefix}{nice}", fontsize=10)
            ax.set_xlabel("t-SNE 1"); ax.set_ylabel("t-SNE 2")
            fig.tight_layout()
            path = self.out_dir / f"corpus_{stage}.png"
            fig.savefig(path, dpi=150, bbox_inches="tight")
            plt.close(fig)
            print(f"    saved --> {path.name}")

    # ── Option 3 ──────────────────────────────────────────────────────────────

    def animated_trajectory(
        self,
        stage_data: dict,
        stage_order: list[str],
        fps: int = 2,
        gif_name: str = "trajectory.gif",
        html_name: str = "trajectory.html",
    ):
        """
        Fits one global t-SNE (all stages stacked --> shared coordinate system),
        then saves:
          • a GIF  — one frame per stage
          • an HTML — self-contained, with a play/pause button and a scrub slider
        """
        present = [s for s in stage_order if s in stage_data]
        print(f"  Fitting global t-SNE over {len(present)} stages…")
        global_coords = _fit_global_tsne(stage_data, stage_order, self.perplexity)

        # ── compute axis limits from ALL points so axes never jump ────────────
        all_xy = np.vstack(list(global_coords.values()))
        pad    = (all_xy.max(0) - all_xy.min(0)) * 0.08
        xlim   = (all_xy[:, 0].min() - pad[0], all_xy[:, 0].max() + pad[0])
        ylim   = (all_xy[:, 1].min() - pad[1], all_xy[:, 1].max() + pad[1])

        # ── GIF ───────────────────────────────────────────────────────────────
        self._save_gif(present, global_coords, stage_data,
                       xlim, ylim, fps, gif_name)

        # ── HTML ──────────────────────────────────────────────────────────────
        self._save_html(present, global_coords, stage_data,
                        xlim, ylim, fps, html_name)

    # ── GIF builder ───────────────────────────────────────────────────────────

    def _make_frame(
        self,
        stage: str,
        coords: np.ndarray,
        labels: list[str],
        xlim: tuple,
        ylim: tuple,
        frame_idx: int,
        total: int,
    ) -> plt.Figure:
        fig, ax = plt.subplots(figsize=(11, 8))
        colour  = _stage_colour(stage)
        ax.scatter(coords[:, 0], coords[:, 1],
                   c=colour, s=50, alpha=0.78, edgecolors="k", linewidths=0.3,
                   zorder=3)
        for i, lbl in enumerate(labels):
            ax.annotate(lbl, coords[i], fontsize=6.5, alpha=0.87,
                        textcoords="offset points", xytext=(3, 3), zorder=4)
        nice = stage.replace("_", " ")
        ax.set_title(f"Stage {frame_idx+1}/{total}:  {nice}", fontsize=11, pad=10)
        ax.set_xlim(*xlim); ax.set_ylim(*ylim)
        ax.set_xlabel("t-SNE 1"); ax.set_ylabel("t-SNE 2")
        ax.grid(True, linestyle="--", alpha=0.25)
        fig.tight_layout()
        return fig

    def _save_gif(self, present, global_coords, stage_data,
                  xlim, ylim, fps, gif_name):
        frames = []
        for i, stage in enumerate(present):
            fig = self._make_frame(
                stage, global_coords[stage], stage_data[stage]["labels"],
                xlim, ylim, i, len(present)
            )
            buf = io.BytesIO()
            fig.savefig(buf, format="png", dpi=120, bbox_inches="tight")
            plt.close(fig)
            buf.seek(0)
            frames.append(plt.imread(buf))

        if not frames:
            return
        h, w = frames[0].shape[:2]
        fig, ax = plt.subplots(figsize=(w / 120, h / 120))
        ax.axis("off")
        ims = [[ax.imshow(f, animated=True)] for f in frames]
        ani = animation.ArtistAnimation(fig, ims,
                                        interval=int(1000 / fps),
                                        blit=True, repeat=True)
        path = self.out_dir / gif_name
        ani.save(str(path), writer="pillow", fps=fps)
        plt.close(fig)
        print(f"    saved --> {path.name}")

    # ── HTML builder ──────────────────────────────────────────────────────────

    def _save_html(self, present, global_coords, stage_data,
                   xlim, ylim, fps, html_name):
        """
        Encodes each frame as a base64 PNG and inlines them all in a single
        HTML file with a JS play/pause slider.  No external dependencies.
        """
        b64_frames = []
        for i, stage in enumerate(present):
            fig = self._make_frame(
                stage, global_coords[stage], stage_data[stage]["labels"],
                xlim, ylim, i, len(present)
            )
            buf = io.BytesIO()
            fig.savefig(buf, format="png", dpi=110, bbox_inches="tight")
            plt.close(fig)
            buf.seek(0)
            b64_frames.append(base64.b64encode(buf.read()).decode())

        stage_names_js = json.dumps([s.replace("_", " ") for s in present])
        frames_js      = json.dumps(b64_frames)

        html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Transformer Embedding Trajectory</title>
<style>
  body  {{ font-family: sans-serif; background: #1a1a2e; color: #eee;
           display: flex; flex-direction: column; align-items: center;
           padding: 24px; }}
  h2    {{ margin-bottom: 6px; letter-spacing: 1px; }}
  #stage-label {{ font-size: 1.1em; color: #90caf9; margin: 8px 0 14px; min-height: 1.4em; }}
  img   {{ max-width: 900px; width: 100%; border-radius: 8px;
           box-shadow: 0 4px 24px #0006; }}
  .controls {{ margin-top: 18px; display: flex; gap: 14px; align-items: center; }}
  button {{ padding: 8px 22px; border-radius: 6px; border: none; cursor: pointer;
            background: #1976d2; color: #fff; font-size: 1em; }}
  button:hover {{ background: #1565c0; }}
  input[type=range] {{ width: 480px; accent-color: #90caf9; }}
  #counter {{ min-width: 80px; text-align: center; color: #aaa; font-size: 0.9em; }}
</style>
</head>
<body>
<h2>Transformer — Embedding Space Trajectory</h2>
<div id="stage-label">–</div>
<img id="frame-img" src="" alt="frame">
<div class="controls">
  <button id="btn-play">▶ Play</button>
  <input type="range" id="slider" min="0" max="{len(present)-1}" value="0" step="1">
  <span id="counter">1 / {len(present)}</span>
</div>

<script>
const frames     = {frames_js};
const stageNames = {stage_names_js};
const img        = document.getElementById('frame-img');
const slider     = document.getElementById('slider');
const label      = document.getElementById('stage-label');
const counter    = document.getElementById('counter');
const btn        = document.getElementById('btn-play');
let cur = 0, timer = null;

function show(i) {{
  cur = i;
  img.src = 'data:image/png;base64,' + frames[i];
  label.textContent = stageNames[i];
  counter.textContent = (i+1) + ' / ' + frames.length;
  slider.value = i;
}}

function step() {{
  show((cur + 1) % frames.length);
}}

btn.addEventListener('click', () => {{
  if (timer) {{ clearInterval(timer); timer = null; btn.textContent = '▶ Play'; }}
  else       {{ timer = setInterval(step, {int(1000/fps)}); btn.textContent = '⏸ Pause'; }}
}});

slider.addEventListener('input', () => show(parseInt(slider.value)));

show(0);
</script>
</body>
</html>"""

        path = self.out_dir / html_name
        path.write_text(html, encoding="utf-8")
        print(f"    saved --> {path.name}")