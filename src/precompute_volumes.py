"""Precompute each tomogram's JPEG slice stack into a single .npy volume.

Replaces per-slice JPEG decoding in unet3d/train.py and unet3d/evaluate.py's
load_volume with a single strided read via np.load(..., mmap_mode="r").

Mirrors train/ and test/ 1:1 as train_npy/ and test_npy/ — no train/val
split, no normalization (normalization stays in load_volume).

Usage:
    python src/precompute_volumes.py [--workers N]

Output:
    {DATA_ROOT}/train_npy/{tomo_id}.npy
    {DATA_ROOT}/test_npy/{tomo_id}.npy
one uint8 (D, H, W) array per tomogram.
"""

import os
import argparse
from multiprocessing import Pool
import numpy as np
from torchvision.io import read_image, ImageReadMode

DATA_ROOT = os.environ.get("FLAGELLA_DATA", "/data/quokka/ws/15/ledy730h-flagella")


def precompute_volume(tomo, in_dir, out_dir):
    out_path = os.path.join(out_dir, f"{tomo}.npy")
    if os.path.exists(out_path):
        return

    tomo_dir = os.path.join(in_dir, tomo)
    slice_names = sorted(f for f in os.listdir(tomo_dir) if f.endswith(".jpg"))

    first = read_image(os.path.join(tomo_dir, slice_names[0]), mode=ImageReadMode.GRAY)
    H, W = first.shape[1], first.shape[2]
    D = len(slice_names)

    vol = np.empty((D, H, W), dtype=np.uint8)
    vol[0] = first[0].numpy()
    for zi, name in enumerate(slice_names[1:], start=1):
        img = read_image(os.path.join(tomo_dir, name), mode=ImageReadMode.GRAY)
        vol[zi] = img[0].numpy()

    tmp_path = os.path.join(out_dir, f"{tomo}.tmp.npy")
    np.save(tmp_path, vol)
    os.replace(tmp_path, out_path)  # atomic: no partial file on crash
    print(f"{tomo}: {vol.shape} saved", flush=True)


def _precompute_volume_star(args):
    precompute_volume(*args)


def precompute_volumes(in_dir, out_dir, num_workers):
    os.makedirs(out_dir, exist_ok=True)
    tomos = sorted(os.listdir(in_dir))
    tasks = [(tomo, in_dir, out_dir) for tomo in tomos]
    if num_workers <= 1:
        for task in tasks:
            _precompute_volume_star(task)
    else:
        with Pool(num_workers) as pool:
            pool.map(_precompute_volume_star, tasks)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=os.cpu_count() or 1)
    args = parser.parse_args()

    precompute_volumes(f"{DATA_ROOT}/train/", f"{DATA_ROOT}/train_npy/", args.workers)
    precompute_volumes(f"{DATA_ROOT}/test/", f"{DATA_ROOT}/test_npy/", args.workers)
