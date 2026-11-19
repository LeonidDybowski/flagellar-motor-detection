"""YOLO first baseline run — score at each motor by slice offset (diagnostic).

Launch via slice_offset_scores.sbatch:
    python src/yolo/diagnostics/slice_offset_scores.py

Training labelled every motor on all slices within +/-3 (DEPTH) of its
centre, including faint outer slices. If that ambiguity holds confidence
down, the score should be high on the centre slice and fall with the
offset; if even the centre slice scores ~0.3, the cause is elsewhere (e.g.
the IoU-aware classification target).

For every motor in the validation tomograms and every slice offset in
OFFSETS (+/-4..6 are unlabelled controls), the 640x640 grid tile holding the
motor (same grid and zero padding as fbeta.py) is predicted with the first
run's best.pt, and the best confidence of a box whose centre lies within
HIT_PX of the motor is recorded (0 if none), plus the tile's top score.
Nothing is trained.

Output (results/yolo/baseline_first_run/diagnostics/slice_offset/):
    scores.csv    one row per motor x offset
    summary.csv   per offset: n, median / quartiles of the motor score,
                  share of motors scoring >= 0.05 and >= 0.2
    scores.png    motor score vs offset (median and quartiles)
"""

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from ultralytics import YOLO

DATA_ROOT = os.environ.get("FLAGELLA_DATA", "/data/quokka/ws/15/ledy730h-flagella")
TRAIN_NPY_DIR = f"{DATA_ROOT}/train_npy/"
LABELS = f"{DATA_ROOT}/train_labels.csv"

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
CHECKPOINT = os.path.join(ROOT, "results", "yolo", "baseline_first_run", "train", "weights", "best.pt")
OUT_DIR = os.path.join(ROOT, "results", "yolo", "baseline_first_run", "diagnostics", "slice_offset")

SEED = 57
VALIDATION_FRAC = 0.2

TILE_SIZE = 640                          # matches prepare_dataset.py
IMG_SIZE = 640
BATCH = 64
DEPTH = 3                                # labelled slices: centre +/- DEPTH (prepare_dataset.py)
OFFSETS = list(range(-6, 7))
HIT_PX = 9                               # half-width of the 19 px training box
MIN_CONF = 0.001                         # keep (almost) every box


def train_val_split(tomo_ids, validation_frac, seed):
    """Identical to unet2d/unet3d's train_val_split."""
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(tomo_ids))
    n_val = int(len(tomo_ids) * validation_frac)
    val_ids = {tomo_ids[i] for i in perm[:n_val]}
    train_ids = [t for t in tomo_ids if t not in val_ids]
    assert val_ids.isdisjoint(set(train_ids))
    return train_ids, sorted(val_ids)


def grid_tile(vol, zi, y0, x0, tile_size=TILE_SIZE):
    """One zero-padded grid tile as a 3-channel uint8 image (as fbeta.py)."""
    tile = np.zeros((tile_size, tile_size), dtype=np.uint8)
    part = vol[zi, y0:y0 + tile_size, x0:x0 + tile_size]
    tile[:part.shape[0], :part.shape[1]] = part
    return np.repeat(tile[:, :, None], 3, axis=2)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    labels = pd.read_csv(LABELS)
    _, val_ids = train_val_split(sorted(labels.tomo_id.unique()), VALIDATION_FRAC, SEED)
    model = YOLO(CHECKPOINT)

    jobs = []                            # (tile, row) per motor x offset
    for tomo in val_ids:
        rows = labels[(labels.tomo_id == tomo) & (labels["Number of motors"] > 0)]
        if rows.empty:
            continue
        vol = np.load(os.path.join(TRAIN_NPY_DIR, f"{tomo}.npy"), mmap_mode="r")
        D = vol.shape[0]
        for mi, (_, r) in enumerate(rows.iterrows()):
            cz, cy, cx = int(r["Motor axis 0"]), int(r["Motor axis 1"]), int(r["Motor axis 2"])
            y0, x0 = cy // TILE_SIZE * TILE_SIZE, cx // TILE_SIZE * TILE_SIZE
            for off in OFFSETS:
                zi = cz + off
                if 0 <= zi < D:
                    jobs.append((grid_tile(vol, zi, y0, x0),
                                 dict(tomo_id=tomo, motor=mi, offset=off,
                                      labelled=abs(off) <= DEPTH, ly=cy - y0, lx=cx - x0)))
    print(f"{len(val_ids)} val tomograms, {len(jobs)} motor x offset tiles", flush=True)

    out = []
    for i in range(0, len(jobs), BATCH):
        chunk = jobs[i:i + BATCH]
        results = model.predict([t for t, _ in chunk], imgsz=IMG_SIZE, conf=MIN_CONF,
                                device=0, half=True, batch=len(chunk), verbose=False)
        for res, (_, row) in zip(results, chunk):
            conf = res.boxes.conf.float().cpu().numpy()
            xy = res.boxes.xywh[:, :2].float().cpu().numpy()
            d = np.hypot(xy[:, 0] - row["lx"], xy[:, 1] - row["ly"]) if len(conf) else np.zeros(0)
            hit = conf[d <= HIT_PX]
            out.append({**{k: v for k, v in row.items() if k not in ("ly", "lx")},
                        "motor_score": float(hit.max()) if len(hit) else 0.0,
                        "tile_top_score": float(conf.max()) if len(conf) else 0.0})
    df = pd.DataFrame(out)
    df.to_csv(os.path.join(OUT_DIR, "scores.csv"), index=False)

    g = df.groupby("offset").motor_score
    summary = pd.DataFrame({
        "n": g.size(), "median": g.median(), "q25": g.quantile(0.25), "q75": g.quantile(0.75),
        "share_ge_0.05": g.apply(lambda s: (s >= 0.05).mean()),
        "share_ge_0.2": g.apply(lambda s: (s >= 0.2).mean()),
    }).reset_index()
    summary.to_csv(os.path.join(OUT_DIR, "summary.csv"), index=False)
    print(summary.to_string(index=False, float_format=lambda v: f"{v:.3f}"), flush=True)

    fig, ax = plt.subplots(figsize=(6.5, 4))
    ax.axvspan(-DEPTH - 0.5, DEPTH + 0.5, color="#ebeae5", zorder=0,
               label=f"labelled as motor (±{DEPTH} slices)")
    ax.fill_between(summary.offset, summary.q25, summary.q75, color="#9ec5f4", alpha=0.6,
                    linewidth=0, label="25–75 % of motors")
    ax.plot(summary.offset, summary["median"], "o-", color="#2a78d6", label="median")
    ax.axhline(0.05, color="#52514e", linewidth=1, linestyle="--")
    ax.text(OFFSETS[-1], 0.055, "0.05 (best validation threshold)", ha="right", va="bottom",
            fontsize=8, color="#52514e")
    ax.set_xlabel("slice offset from the motor's centre slice")
    ax.set_ylabel("best box score within 9 px of the motor")
    ax.set_xticks(OFFSETS)
    ax.set_ylim(0, max(0.5, float(summary.q75.max()) * 1.1))
    ax.set_title("YOLO first run: score at the motor by slice offset (validation)")
    ax.legend(loc="upper left", fontsize=8, frameon=False)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "scores.png"), dpi=150)
    print(f"saved {OUT_DIR}/scores.csv, summary.csv, scores.png", flush=True)


if __name__ == "__main__":
    main()
