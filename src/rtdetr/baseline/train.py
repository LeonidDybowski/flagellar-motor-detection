"""RT-DETR flagellar motor detector — training via Ultralytics.

Launch via train.sbatch:
    python src/rtdetr/train.py

Reuses YOLO's tiled dataset (src/yolo/data.yaml, {DATA_ROOT}/train_tiles/) —
same tiles, same YOLO-format labels, same train/val split. Run
src/yolo/prepare_dataset.py first if that hasn't been done.

RT-DETR is query-based (bipartite/Hungarian-matched set prediction, not
anchor-based like YOLO), so its loss has three separately-named components
(giou_loss, cls_loss, l1_loss) instead of YOLO's box/cls/dfl split — see
src/rtdetr/analyze_loss.py, which reads Ultralytics' auto-written
results/rtdetr/baseline/train/results.csv and plots each component's train/val curve.
That is the primary artifact for studying how the set-prediction loss
behaves on this sparse, low-SNR data — training itself needs no custom loss
logging code, Ultralytics writes results.csv every epoch by default.

Uses all 4 GPUs — Ultralytics handles its own multi-GPU via the `device`
kwarg (spawns its own DDP processes internally; no torchrun needed).

Changes vs src/legacy/rtdetr/train.py: every epoch's weights are kept
(SAVE_PERIOD) for src/rtdetr/fbeta.py, which train.sbatch runs after training
to log the per-motor F-beta per epoch. Ultralytics' own validation still
selects best.pt.
"""

import os
from ultralytics import RTDETR

from fp32_trainer import Fp32RTDETRTrainer  # fp32 checkpoints (see fp32_trainer.py)

DATA_YAML = os.path.join(os.path.dirname(__file__), "data.yaml")
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
RESULT_DIR = os.path.join(ROOT, "results/rtdetr/baseline")

MODEL = os.path.join(os.path.dirname(__file__), "rtdetr-l.pt")
DEVICE = "0,1,2,3"

IMG_SIZE = 640                           # matches prepare_dataset.py's TILE_SIZE
EPOCHS = 20                              # matches unet2d/unet3d/yolo NUM_EPOCHS
BATCH = 64                               # matches GLOBAL_BATCH_SIZE; the 1280px mosaic OOM that forced 32 is gone (MOSAIC = 0.0)
WORKERS = 6

SEED = 57

# Keep every epoch's weights (weights/epoch{N}.pt) so fbeta.py can score each
# epoch on whole validation tomograms with the per-motor F-beta afterwards.
SAVE_PERIOD = 1

# Run 2318275 diverged: optimizer="auto" picked MuSGD(lr=0.01), warmup ramped
# it to 0.027 by epoch 3, and losses went NaN at epoch 4 — after which
# "EMA contains NaN/Inf" blocked every checkpoint save and mAP50 sat at 0.
# These are RT-DETR's reference optimiser settings instead.
OPTIMIZER = "AdamW"
LR0 = 1e-4
MOMENTUM = 0.9                           # AdamW beta1 (trainer.py:1085 builds betas=(momentum, 0.999))
WEIGHT_DECAY = 1e-4
WARMUP_EPOCHS = 1.0
# Naming an optimiser skips trainer.py's "auto" branch, which is also the only
# place that clamps warmup_bias_lr for Adam-family optimisers (trainer.py:1064).
# Left at its default 0.1 it would warm the bias groups at 1000x LR0.
WARMUP_BIAS_LR = 0.0

# Ultralytics' RT-DETR path builds transforms via v8_transforms(..., stretch=True)
# (models/rtdetr/val.py:86), which sets RandomPerspective(size=None)
# (data/augment.py:2739) — so Mosaic's 2*imgsz canvas is never cropped back down
# and training ran at 1280 while validation ran at 640 (val gets no LetterBox at
# all, val.py:89). Disabling mosaic keeps both at IMG_SIZE, and measured 2.23 ->
# 1.50 s/it on run 2318275.
MOSAIC = 0.0


def main():
    if not os.path.exists(DATA_YAML):
        raise FileNotFoundError(
            f"{DATA_YAML} not found — run src/rtdetr/baseline/prepare_dataset.py first."
        )

    os.makedirs(RESULT_DIR, exist_ok=True)

    model = RTDETR(MODEL)
    model.train(
        trainer=Fp32RTDETRTrainer,
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
        optimizer=OPTIMIZER,
        lr0=LR0,
        momentum=MOMENTUM,
        weight_decay=WEIGHT_DECAY,
        warmup_epochs=WARMUP_EPOCHS,
        warmup_bias_lr=WARMUP_BIAS_LR,
        # Motors are small point-like objects with a fixed 19px box (see
        # yolo/prepare_dataset.py) — plain flips preserve that.
        fliplr=0.5,
        flipud=0.5,
        mosaic=MOSAIC,
        close_mosaic=0,                  # nothing to close; avoids a misleading log line
    )


if __name__ == "__main__":
    main()
