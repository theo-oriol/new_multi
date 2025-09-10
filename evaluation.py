#!/usr/bin/env python3
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from torcheval.metrics import MultilabelPrecisionRecallCurve
from torcheval.metrics.functional.classification.auprc import binary_auprc

CSV_PATH    = "/home/oriol@newcefe.newage.fr/Models/new_multi/runs/runs_dino_sans_couleur/DINOv2_sans_couleur_multi_0/predictions_val.csv"
LABELS_PATH = "/home/oriol@newcefe.newage.fr/Models/new_multi/labels_habitat1.json"  

def safe_name(s):
    return "".join(c if c.isalnum() or c in "-_." else "_" for c in s)

def plot_pr(recall, precision, title, out_path):
    plt.figure()
    plt.step(recall, precision, where="post")
    plt.xlabel("Recall"); plt.ylabel("Precision"); plt.title(title)
    plt.xlim(0, 1); plt.ylim(0, 1.05); plt.grid(alpha=0.3)
    plt.tight_layout(); plt.savefig(out_path, dpi=150); plt.close()

def main():
    csv_path = Path(CSV_PATH)
    out_dir = csv_path.parent / "pr_curves"
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(csv_path)
    prob_cols  = sorted([c for c in df.columns if c.startswith("prob_")],  key=lambda x: int(x.split("_")[1]))
    label_cols = sorted([c for c in df.columns if c.startswith("label_")], key=lambda x: int(x.split("_")[1]))

    probs  = torch.tensor(df[prob_cols].to_numpy(np.float32))  
    labels = torch.tensor(df[label_cols].to_numpy(np.int64))   
    N, C = labels.shape

    class_names = json.loads(Path(LABELS_PATH).read_text(encoding="utf-8"))

    pos_counts = labels.sum(dim=0)  
    valid_idx = [i for i in range(C) if pos_counts[i].item() > 0]

 

    metric = MultilabelPrecisionRecallCurve(num_labels=C)
    metric.update(probs, labels)
    precisions, recalls, _ = metric.compute()  

    aps = []
    for i in valid_idx:
        p = precisions[i].cpu().numpy()
        r = recalls[i].cpu().numpy()
        ap_i = float(binary_auprc(probs[:, i], labels[:, i]))
        aps.append(ap_i)

        fname = f"PR_{i:02d}_{safe_name(class_names[i])}.png"
        plot_pr(r, p, f"{class_names[i]} (AP={ap_i:.3f})", out_dir / fname)

    plt.figure(figsize=(8, 6))
    for i, ap_i in zip(valid_idx, aps):
        p = precisions[i].cpu().numpy()
        r = recalls[i].cpu().numpy()
        plt.step(r, p, where="post", label=f"{class_names[i]} (AP={ap_i:.3f})", alpha=0.9)

    mAP = float(np.mean(aps))
    plt.xlabel("Recall"); plt.ylabel("Precision")
    plt.title(f"Precision–Recall  mAP={mAP:.3f}")
    plt.xlim(0, 1); plt.ylim(0, 1.05); plt.grid(alpha=0.3)
    plt.legend(fontsize=8, loc="best")
    plt.tight_layout()
    plt.savefig(out_dir / "PR_all_classes.png", dpi=150)
    plt.close()


if __name__ == "__main__":
    main()
