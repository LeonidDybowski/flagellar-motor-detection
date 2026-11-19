"""YOLOv8 flagellar motor detector — training via Ultralytics.

Launch via train.sbatch:
    python src/yolo/train.py

Prerequisite: src/yolo/prepare_dataset.py must have been run first, to
tile {DATA_ROOT}/train_npy/ into {DATA_ROOT}/train_tiles/ (YOLO-format
images + labels) and write src/yolo/data.yaml.

Uses all 4 GPUs — Ultralytics handles its own multi-GPU via the `device`
kwarg (spawns its own DDP processes internally; no torchrun needed).

Changes vs src/legacy/yolo/train.py: every epoch's weights are kept
(SAVE_PERIOD) for src/yolo/fbeta.py, which train.sbatch runs after training
to log the per-motor F-beta per epoch. Ultralytics' own validation still
selects best.pt.
"""

import os
from ultralytics import YOLO

DATA_YAML = os.path.join(os.path.dirname(__file__), "data.yaml")
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
RESULT_DIR = os.path.join(ROOT, "results/yolo/baseline")

MODEL = "yolov8n.pt"
DEVICE = "0,1,2,3"

IMG_SIZE = 640                           # matches prepare_dataset.py's TILE_SIZE
EPOCHS = 20                              # matches unet2d/unet3d NUM_EPOCHS
BATCH = 64                               # matches unet2d/unet3d GLOBAL_BATCH_SIZE
WORKERS = 6

SEED = 57

# Keep every epoch's weights (weights/epoch{N}.pt) so fbeta.py can score each
# epoch on whole validation tomograms with the per-motor F-beta afterwards.
SAVE_PERIOD = 1


def main():
    if not os.path.exists(DATA_YAML):
        raise FileNotFoundError(
            f"{DATA_YAML} not found — run src/yolo/prepare_dataset.py first."
        )

    os.makedirs(RESULT_DIR, exist_ok=True)

    model = YOLO(MODEL)
    model.train(
        data=DATA_YAML,
        epochs=EPOCHS,
        imgsz=IMG_SIZE,
        batch=BATCH,
        workers=WORKERS,
        device=DEVICE,
        seed=SEED,
        save_period=SAVE_PERIOD,
        project=RESULT_DIR,
        name="train",
        exist_ok=True,
        # Motors are small point-like objects with a fixed 19px box (see
        # prepare_dataset.py) — plain flips/rotation/translation preserve
        # that; mosaic/mixup are left at Ultralytics' defaults since they
        # don't distort box geometry either.
        fliplr=0.5,
        flipud=0.5,
    )


if __name__ == "__main__":
    main()
