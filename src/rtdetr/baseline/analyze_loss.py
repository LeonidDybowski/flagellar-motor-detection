"""Study RT-DETR's set-prediction loss on sparse, low-SNR tomogram tiles.

Launch via analyze_loss.sbatch (CPU-only, no GPU needed — this only reads
results.csv, no model inference):
    python src/rtdetr/analyze_loss.py

RT-DETR is query-based: a fixed set of learned object queries is
bipartite-matched (Hungarian algorithm) against ground-truth boxes each
step, and the loss decomposes into three independently-weighted components
(see ultralytics.models.rtdetr.train.RTDETRTrainer.loss_names):
  * cls_loss  — classification loss on matched queries (is this query a
    motor, and which one).
  * l1_loss   — L1 box-coordinate regression loss on matched queries.
  * giou_loss — generalised IoU loss on matched queries (box shape/overlap,
    scale-invariant unlike raw L1).
Unmatched queries only contribute to cls_loss (pushed towards
"no object"). On data this sparse — most tiles have zero or one motor
against a fixed query budget, and most negatives are visually
indistinguishable low-SNR background — the informative signal is not any
single component's absolute value but how the three move relative to each
other over training:
  * cls_loss dominating while giou/l1 stay flat near their initial values
    suggests the model is mostly learning "is there anything here" (a
    detection-existence problem) without yet refining box geometry, which
    is expected early given how rarely a query is actually matched to a
    real motor per step.
  * giou_loss failing to drop even as cls_loss falls suggests the model
    can find motors but not localise them precisely — plausible given
    tomograms' low SNR makes exact motor boundaries ambiguous even to the
    Gaussian-heatmap ground truth this dataset derives boxes from
    (yolo/prepare_dataset.py's BOX_SIZE is a fixed 19px proxy, not a
    measured extent).
  * All three plateauing together, early, points at the bipartite matcher
    itself struggling to find consistent matches step-to-step on such a
    sparse target set (few ground-truth boxes per image relative to the
    query count) rather than a per-component optimisation problem.

This script does not draw those conclusions automatically — it produces the
per-component train/val curves (results/rtdetr/baseline/loss_components.png) and a
tidy CSV (results/rtdetr/baseline/loss_components.csv) so the actual run's behaviour
can be read off and interpreted against the patterns above.

Prerequisite: src/rtdetr/train.py must have run, producing
results/rtdetr/baseline/train/results.csv (written automatically by Ultralytics'
BaseTrainer every epoch — no custom logging needed during training).
"""

import os
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
RESULT_DIR = os.path.join(ROOT, "results/rtdetr/baseline")
RESULTS_CSV = f"{RESULT_DIR}/train/results.csv"

LOSS_COMPONENTS = ["giou_loss", "cls_loss", "l1_loss"]


def main():
    if not os.path.exists(RESULTS_CSV):
        raise FileNotFoundError(f"{RESULTS_CSV} not found — run src/rtdetr/train.py first.")

    df = pd.read_csv(RESULTS_CSV)
    df.columns = [c.strip() for c in df.columns]

    epoch_col = "epoch" if "epoch" in df.columns else df.columns[0]

    tidy_rows = []
    fig, axes = plt.subplots(1, len(LOSS_COMPONENTS), figsize=(15, 4.5), sharex=True)
    for ax, comp in zip(axes, LOSS_COMPONENTS):
        train_col = f"train/{comp}"
        val_col = f"val/{comp}"
        if train_col in df.columns:
            ax.plot(df[epoch_col], df[train_col], label="train", marker="o", markersize=3)
        if val_col in df.columns:
            ax.plot(df[epoch_col], df[val_col], label="val", marker="o", markersize=3)
        ax.set_title(comp)
        ax.set_xlabel("epoch")
        ax.grid(True)
        ax.legend()

        for _, row in df.iterrows():
            tidy_rows.append({
                "epoch": row[epoch_col], "component": comp,
                "train": row.get(train_col), "val": row.get(val_col),
            })

    axes[0].set_ylabel("loss")
    fig.suptitle("RT-DETR set-prediction loss components — train vs val")
    fig.tight_layout()

    os.makedirs(RESULT_DIR, exist_ok=True)
    fig.savefig(f"{RESULT_DIR}/loss_components.png", dpi=150)
    pd.DataFrame(tidy_rows).to_csv(f"{RESULT_DIR}/loss_components.csv", index=False)

    print(f"saved {RESULT_DIR}/loss_components.png and {RESULT_DIR}/loss_components.csv",
          flush=True)

    # Quick printed summary: first vs last epoch per component, so the trend
    # direction is visible without opening the plot.
    for comp in LOSS_COMPONENTS:
        train_col, val_col = f"train/{comp}", f"val/{comp}"
        if train_col in df.columns:
            first, last = df[train_col].iloc[0], df[train_col].iloc[-1]
            print(f"{comp}: train {first:.4f} -> {last:.4f}", flush=True)
        if val_col in df.columns:
            first, last = df[val_col].iloc[0], df[val_col].iloc[-1]
            print(f"{comp}: val   {first:.4f} -> {last:.4f}", flush=True)


if __name__ == "__main__":
    main()
