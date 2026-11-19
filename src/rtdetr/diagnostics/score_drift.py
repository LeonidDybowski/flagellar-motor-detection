"""RT-DETR score drift across epochs — diagnostic.

Launch via score_drift.sbatch:
    python src/rtdetr/score_drift.py

Whole-tomogram F2 went 0.516 (epoch 6) -> 0 (epochs 7-9) -> flood of ~0.55
positions on every tile (epoch 10 on). This runs a few checkpoints on the
same validation tiles (empty and motor-containing) and records every query's
score, to show whether the score scale of background moved as a whole and
whether fp16 inference plays a part. It also prints the classification
heads' biases per checkpoint.

Output:
    results/rtdetr/baseline_first_run/score_drift.csv   one row per checkpoint x precision x tile kind
    results/rtdetr/baseline_first_run/score_drift.png   score histograms, empty vs motor tiles
"""

import glob
import os
import random

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from ultralytics import RTDETR

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
WEIGHTS_DIR = os.path.join(ROOT, "results", "rtdetr", "baseline_first_run", "train", "weights")
RESULT_DIR = os.path.join(ROOT, "results", "rtdetr", "baseline_first_run")
TILES = "/data/quokka/ws/15/ledy730h-flagella/train_tiles"

# Ultralytics' epoch{N}.pt is 0-based: label = 1-based epoch as in fbeta_history.csv.
CHECKPOINTS = {4: "epoch3.pt", 6: "epoch5.pt", 7: "epoch6.pt", 9: "epoch8.pt",
               10: "epoch9.pt", 14: "epoch13.pt", 20: "last.pt"}
N_TILES = 400            # per kind (empty / motor)
IMG_SIZE = 640
MATCH_PX = 10            # a query "hits" the motor if its centre is within this of the label centre
SEED = 57


def sample_tiles():
    labels = sorted(glob.glob(os.path.join(TILES, "labels", "val", "*.txt")))
    empty, motor = [], []
    for p in labels:
        (motor if os.path.getsize(p) > 0 else empty).append(p)
    rng = random.Random(SEED)
    pick = lambda xs: rng.sample(xs, min(N_TILES, len(xs)))
    def to_img(p):
        stem = os.path.splitext(os.path.basename(p))[0]
        return glob.glob(os.path.join(TILES, "images", "val", stem + ".*"))[0], p
    print(f"{len(empty)} empty / {len(motor)} motor val tiles")
    return [to_img(p) for p in pick(empty)], [to_img(p) for p in pick(motor)]


def motor_centres(label_path):
    rows = np.loadtxt(label_path, ndmin=2)
    return rows[:, 1:3] * IMG_SIZE


def head_biases(model):
    dec = model.model.model[-1]
    out = {}
    for name in ("enc_score_head", "dec_score_head"):
        head = getattr(dec, name, None)
        if head is None:
            continue
        lin = head[-1] if isinstance(head, torch.nn.ModuleList) else head
        out[name] = float(lin.bias.detach().float().mean())
    return out


def scores_of(model, tiles, half, with_hits):
    """All query scores, top score per tile, and (motor tiles) best score of a hit."""
    all_s, top, hit = [], [], []
    for i in range(0, len(tiles), 32):
        chunk = tiles[i:i + 32]
        res = model.predict([t[0] for t in chunk], imgsz=IMG_SIZE, conf=0.0, max_det=300,
                            half=half, device=0, batch=len(chunk), verbose=False)
        for r, (_, lab) in zip(res, chunk):
            s = r.boxes.conf.float().cpu().numpy()
            all_s.append(s)
            top.append(s.max() if len(s) else 0.0)
            if with_hits:
                c = r.boxes.xywh[:, :2].float().cpu().numpy()
                d = np.linalg.norm(c[:, None] - motor_centres(lab)[None], axis=2).min(1)
                hit.append(s[d <= MATCH_PX].max() if (d <= MATCH_PX).any() else 0.0)
    return np.concatenate(all_s), np.array(top), np.array(hit)


def main():
    empty, motor = sample_tiles()
    rows, hists = [], {}
    for ep, name in CHECKPOINTS.items():
        model = RTDETR(os.path.join(WEIGHTS_DIR, name))
        print(f"epoch {ep:2d} ({name}) biases {head_biases(model)}", flush=True)
        for half in (True, False):
            se, te, _ = scores_of(model, empty, half, False)
            sm, tm, hm = scores_of(model, motor, half, True)
            hists[(ep, half)] = (se, sm)
            for kind, s, t, h in (("empty", se, te, None), ("motor", sm, tm, hm)):
                row = dict(epoch=ep, half=half, kind=kind,
                           q_median=np.median(s), q_p99=np.percentile(s, 99),
                           top_median=np.median(t), top_p10=np.percentile(t, 10),
                           top_p90=np.percentile(t, 90),
                           n_queries_ge_0p5=float((s >= 0.5).sum() / len(t)))
                if h is not None:
                    row.update(hit_median=np.median(h), hit_rate=float((h > 0).mean()))
                rows.append(row)
                print("  " + "  ".join(f"{k}={v:.3f}" if isinstance(v, float) else f"{k}={v}"
                                       for k, v in row.items()), flush=True)

    keys = ["epoch", "half", "kind", "q_median", "q_p99", "top_median", "top_p10",
            "top_p90", "n_queries_ge_0p5", "hit_median", "hit_rate"]
    with open(os.path.join(RESULT_DIR, "score_drift.csv"), "w") as f:
        f.write(",".join(keys) + "\n")
        for r in rows:
            f.write(",".join(str(r.get(k, "")) for k in keys) + "\n")

    fig, axes = plt.subplots(1, len(CHECKPOINTS), figsize=(3 * len(CHECKPOINTS), 3), sharey=True)
    bins = np.linspace(0, 1, 51)
    for ax, ep in zip(axes, CHECKPOINTS):
        se, sm = hists[(ep, True)]
        ax.hist(se, bins, density=True, alpha=0.6, label="empty tiles")
        ax.hist(sm, bins, density=True, alpha=0.6, label="motor tiles")
        ax.set_title(f"epoch {ep}")
        ax.set_xlabel("query score")
        ax.set_yscale("log")
    axes[0].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(os.path.join(RESULT_DIR, "score_drift.png"), dpi=150)
    print("saved score_drift.csv and score_drift.png")


if __name__ == "__main__":
    main()
