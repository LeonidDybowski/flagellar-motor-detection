"""RT-DETR flagellar motor detector — evaluation via Ultralytics.

Launch via evaluate.sbatch:
    python src/rtdetr/evaluate.py

Runs Ultralytics' built-in validator against the held-out val split defined
in src/yolo/data.yaml (same tomograms unet2d/unet3d/yolo validate on, via
the shared train_val_split), and writes a metrics summary to
results/rtdetr/baseline/pr.csv, mirroring the other detectors' evaluate.py output.

Loads the best checkpoint from the most recent train.py run
(results/rtdetr/baseline/train/weights/best.pt) unless FLAGELLA_RTDETR_CKPT
overrides it (matches unet2d/unet3d/yolo evaluate.py's checkpoint-env-var
convention).
"""

import os
import pandas as pd
from ultralytics import RTDETR

DATA_YAML = os.path.join(os.path.dirname(__file__), "data.yaml")
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
RESULT_DIR = os.path.join(ROOT, "results/rtdetr/baseline")
CHECKPOINT = os.environ.get("FLAGELLA_RTDETR_CKPT", f"{RESULT_DIR}/train/weights/best.pt")

DEVICE = "0,1,2,3"
IMG_SIZE = 640
BATCH = 64
WORKERS = 6


def main():
    if not os.path.exists(CHECKPOINT):
        raise FileNotFoundError(f"{CHECKPOINT} not found — run src/rtdetr/train.py first.")
    if not os.path.exists(DATA_YAML):
        raise FileNotFoundError(
            f"{DATA_YAML} not found — run src/rtdetr/baseline/prepare_dataset.py first."
        )

    os.makedirs(RESULT_DIR, exist_ok=True)

    model = RTDETR(CHECKPOINT)
    metrics = model.val(
        data=DATA_YAML,
        imgsz=IMG_SIZE,
        batch=BATCH,
        workers=WORKERS,
        device=DEVICE,
        project=RESULT_DIR,
        name="evaluate",
        exist_ok=True,
    )

    # box.p / box.r / box.map / box.map50 are per-class arrays (nc=1 here,
    # so index [0]); mp/mr/map/map50 are the same values pre-averaged.
    row = {
        "precision": float(metrics.box.mp),
        "recall": float(metrics.box.mr),
        "map50": float(metrics.box.map50),
        "map50_95": float(metrics.box.map),
    }
    f1 = (2 * row["precision"] * row["recall"] / (row["precision"] + row["recall"])
          if (row["precision"] + row["recall"]) > 0 else 0.0)
    row["f1"] = f1

    print(f"P={row['precision']:.3f}  R={row['recall']:.3f}  "
          f"F1={f1:.3f}  mAP50={row['map50']:.3f}  mAP50-95={row['map50_95']:.3f}",
          flush=True)

    pd.DataFrame([row]).to_csv(f"{RESULT_DIR}/pr.csv", index=False)
    print(f"saved {RESULT_DIR}/pr.csv", flush=True)


if __name__ == "__main__":
    main()
