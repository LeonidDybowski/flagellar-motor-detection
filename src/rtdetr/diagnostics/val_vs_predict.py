"""RT-DETR: Ultralytics validation vs. predict on the same tiles — diagnostic.

Launch via val_vs_predict.sbatch:
    python src/rtdetr/val_vs_predict.py

Ultralytics' validation during training reported P 0.63 / val cls_loss 0.074
at epoch 20, while predict() on the same val tiles gives ~0.6 to 252 of 300
queries on every empty tile (score_drift.py). Both cannot hold. This runs,
on the same checkpoint:
  1. model.val() on the full val split, recording every image's raw query
     scores (before the conf filter) inside the validator;
  2. model.predict() on a sample of those same files;
  3. a bare forward pass (cv2 read, /255, model.model) on the same files;
and prints, per path, top score and queries >= 0.5 for empty vs motor tiles,
plus P/R/mAP50 from (1) and what the checkpoint file holds (ema / model).

Output: printed only (logs/rtdetr/rtdetr-val-vs-predict-<jobid>.out).
"""

import os
import random

import cv2
import numpy as np
import torch
from ultralytics import RTDETR
from ultralytics.models.rtdetr.val import RTDETRValidator

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
CKPT = os.environ.get("FLAGELLA_RTDETR_CKPT",
                      os.path.join(ROOT, "results/rtdetr/baseline_first_run/train/weights/last.pt"))
# The existing baseline weights (job 2366332) were trained on YOLO's tiles.
DATA_YAML = os.path.join(os.path.dirname(os.path.abspath(__file__)), "yolo_tiles.yaml")
IMG_SIZE = 640
N_SAMPLE = 300           # files per kind compared across the three paths
SEED = 57

record = {}              # im_file -> raw query scores seen inside the validator
_orig_pre, _orig_post = RTDETRValidator.preprocess, RTDETRValidator.postprocess


def _pre(self, batch):
    self._files = list(batch["im_file"])
    return _orig_pre(self, batch)


def _post(self, preds):
    raw = preds[0] if isinstance(preds, (list, tuple)) else preds
    for f, s in zip(self._files, raw[..., 4].float().cpu().numpy()):
        record[f] = s
    return _orig_post(self, preds)


RTDETRValidator.preprocess, RTDETRValidator.postprocess = _pre, _post


def is_empty(img_path):
    lab = img_path.replace("/images/", "/labels/").rsplit(".", 1)[0] + ".txt"
    return os.path.getsize(lab) == 0


def summary(name, scores_by_file, files):
    for kind in ("empty", "motor"):
        fs = [f for f in files if is_empty(f) == (kind == "empty")]
        top = np.array([scores_by_file[f].max() for f in fs])
        n05 = np.array([(scores_by_file[f] >= 0.5).sum() for f in fs])
        print(f"  {name:8s} {kind:5s} n={len(fs):5d}  top median {np.median(top):.4f}  "
              f"top p90 {np.percentile(top, 90):.4f}  queries>=0.5 mean {n05.mean():.1f}", flush=True)


def main():
    ck = torch.load(CKPT, map_location="cpu", weights_only=False)
    print(f"checkpoint {CKPT}")
    print(f"  keys {sorted(ck)}  ema is None: {ck.get('ema') is None}  "
          f"model is None: {ck.get('model') is None}  epoch {ck.get('epoch')}", flush=True)
    if ck.get("ema") is not None and ck.get("model") is not None:
        a, b = ck["ema"].float().state_dict(), ck["model"].float().state_dict()
        d = max((a[k] - b[k]).abs().max().item() for k in a if a[k].is_floating_point())
        print(f"  max |ema - model| weight difference {d:.4g}", flush=True)

    model = RTDETR(CKPT)
    m = model.val(data=DATA_YAML, imgsz=IMG_SIZE, batch=32, device=0, plots=False, verbose=False)
    print(f"model.val: P {m.box.mp:.4f}  R {m.box.mr:.4f}  mAP50 {m.box.map50:.4f}  "
          f"({len(record)} images recorded)", flush=True)

    rng = random.Random(SEED)
    files = sorted(record)
    empty = [f for f in files if is_empty(f)]
    motor = [f for f in files if not is_empty(f)]
    sample = rng.sample(empty, min(N_SAMPLE, len(empty))) + rng.sample(motor, min(N_SAMPLE, len(motor)))

    pred = {}
    for i in range(0, len(sample), 32):
        chunk = sample[i:i + 32]
        for f, r in zip(chunk, model.predict(chunk, imgsz=IMG_SIZE, conf=0.0, max_det=300,
                                             device=0, batch=len(chunk), verbose=False)):
            pred[f] = r.boxes.conf.float().cpu().numpy()

    net = model.model.to(0).eval()
    fwd = {}
    with torch.no_grad():
        for i in range(0, len(sample), 32):
            chunk = sample[i:i + 32]
            x = np.stack([cv2.imread(f)[..., ::-1] for f in chunk]).transpose(0, 3, 1, 2)
            x = torch.from_numpy(np.ascontiguousarray(x)).to(0).float() / 255
            out = net(x)
            out = out[0] if isinstance(out, (list, tuple)) else out
            for f, s in zip(chunk, out[..., 4].float().cpu().numpy()):
                fwd[f] = s

    print("same files, three paths:")
    summary("val", record, sample)
    summary("predict", pred, sample)
    summary("forward", fwd, sample)
    print("full val split, validator path:")
    summary("val", record, files)


if __name__ == "__main__":
    main()
