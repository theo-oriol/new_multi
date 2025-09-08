import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch

CSV_PATH    = "/home/oriol@newcefe.newage.fr/Models/new_multi/runs/runs_dino_relatif/DINOv2_relatif_multi_6/predictions_val.csv"
LABELS_PATH = "/home/oriol@newcefe.newage.fr/Models/new_multi/labels_habitat1_relatif.json" 
EPS = 1e-12

def _normalize_rows(a: np.ndarray, eps: float = EPS) -> np.ndarray:
    s = a.sum(axis=1, keepdims=True)
    s = np.where(s < eps, 1.0, s)
    return (a / s).astype(np.float32)

def _brier_per_sample(p: np.ndarray, q: np.ndarray) -> np.ndarray:
    # mean squared error per sample
    return ((q - p) ** 2).mean(axis=1)

def _kl_per_sample(p: np.ndarray, q: np.ndarray, eps: float = EPS) -> np.ndarray:
    # D_KL(p||q) per sample
    p = np.clip(p, eps, 1.0)
    q = np.clip(q, eps, 1.0)
    return (p * (np.log(p) - np.log(q))).sum(axis=1)

def _plot_grid(indices, probs, labels, images, class_names, out_path, fig_title, brier, kl):
    C = labels.shape[1]
    x = 0.5 + np.arange(C)
    fig, axs = plt.subplots(2, 2, figsize=(12, 8))
    axs = np.asarray(axs)

    for i, sample in enumerate(indices):
        ax = axs[i // 2, i % 2]
        ax.bar(x, probs[sample], alpha=0.5, label="preds")
        ax.bar(x, labels[sample], alpha=0.5, label="labels")

        title = f"idx={sample}  KL={kl[sample]:.4f}  Brier={brier[sample]:.4f}\n{images[sample]}"
        ax.set_title(title, fontsize=9)
        ax.set_xticks(x, class_names, rotation=60, ha="right")
        ax.legend(fontsize=8, loc="best")

    fig.suptitle(fig_title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close(fig)

def main():
    csv_path = Path(CSV_PATH)
    out_dir = csv_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(csv_path)
    prob_cols  = sorted([c for c in df.columns if c.startswith("prob_")],  key=lambda x: int(x.split("_")[1]))
    label_cols = sorted([c for c in df.columns if c.startswith("label_")], key=lambda x: int(x.split("_")[1]))
    image_cols = sorted([c for c in df.columns if c.startswith("image_")], key=lambda x: x.split("_"))

    probs_raw  = df[prob_cols].to_numpy(np.float32)
    labels_raw = df[label_cols].to_numpy(np.float32)

    if image_cols:
        images = df[image_cols].to_numpy(str)
        # flatten multiple image_* columns if present
        if images.ndim > 1:
            images = np.array([" | ".join(row.tolist()) for row in images])
    else:
        images = df.get("image_path", pd.Series(["?"] * len(df))).to_numpy(str)

    # normalize (defensive)
    probs  = _normalize_rows(probs_raw)
    labels = _normalize_rows(labels_raw)

    N, C = labels.shape
    class_names = json.loads(Path(LABELS_PATH).read_text(encoding="utf-8"))

    # per-sample metrics
    brier = _brier_per_sample(labels, probs)
    kl    = _kl_per_sample(labels, probs)

    # 1) Random 4
    rng_idx = [np.random.randint(0, N) for _ in range(4)]
    _plot_grid(
        rng_idx, probs, labels, images, class_names,
        out_dir / "distri_ex_random.png",
        "Random samples: predicted vs. label distributions",
        brier=brier, kl=kl
    )

    # 2) Best Brier (lowest 4)
    best_brier_idx = np.argsort(brier)[:4].tolist()
    _plot_grid(
        best_brier_idx, probs, labels, images, class_names,
        out_dir / "distri_ex_best_brier.png",
        "Best Brier (lowest per-sample Brier)",
        brier=brier, kl=kl
    )

    # 3) Best Brier but bad KL (from top-K best Brier, pick highest KL)
    K = max(20, min(N, 100))
    topK_brier = np.argsort(brier)[:K]
    pick_bad_kl_among_best_brier = topK_brier[np.argsort(kl[topK_brier])[::-1][:4]].tolist()
    _plot_grid(
        pick_bad_kl_among_best_brier, probs, labels, images, class_names,
        out_dir / "distri_ex_best_brier_bad_kl.png",
        f"Best Brier but bad KL (chosen from top-{K} Brier)",
        brier=brier, kl=kl
    )

    # 4) Worst KL (highest 4)
    worst_kl_idx = np.argsort(kl)[-4:][::-1].tolist()
    _plot_grid(
        worst_kl_idx, probs, labels, images, class_names,
        out_dir / "distri_ex_worst_kl.png",
        "Worst KL (highest per-sample KL)",
        brier=brier, kl=kl
    )

if __name__ == "__main__":
    main()