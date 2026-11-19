"""YOLO center-slice experiment — training via Ultralytics.

Launch via train.sbatch:
    python src/yolo/center_slice/train.py

Identical to src/yolo/train.py except for the dataset: motors labelled only
on their centre slice (center_slice/prepare_dataset.py, run that first).
"""

import os
from ultralytics import YOLO

DATA_YAML = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data.yaml")
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
RESULT_DIR = os.path.join(ROOT, "results/yolo/center_slice")

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
            f"{DATA_YAML} not found — run src/yolo/center_slice/prepare_dataset.py first."
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
        fliplr=0.5,
        flipud=0.5,
    )


if __name__ == "__main__":
    main()
