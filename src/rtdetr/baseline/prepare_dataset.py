"""Tile precomputed tomogram volumes into a YOLOv8-format detection dataset.

Ultralytics' trainer expects real image files on disk plus a parallel
directory of YOLO-format label .txt files (class cx cy w h, normalised
0-1), discovered by filename convention — it does not accept an in-memory
Dataset/crop function the way the unet2d/unet3d PyTorch pipelines do. This
script materialises that on-disk layout by tiling each z-slice of every
precomputed {DATA_ROOT}/train_npy/*.npy volume (see src/precompute_volumes.py)
into non-overlapping 640x640 PNG tiles, and writing one YOLO-format label
file per tile that contains a motor.

Tiling is a fixed grid (tile boundaries are deterministic multiples of
TILE_SIZE), unlike unet2d/unet3d's randomly-jittered crops — so positives
are not upsampled by redrawing; augmentation instead relies on Ultralytics'
own built-in augmentation (flips, mosaic, HSV, etc.) at train time.

Motor "extent" is derived from HEATMAP_RADIUS (= 3*SIGMA = 9), the same
truncation radius unet2d/unet3d use for heatmap targets and peak-matching —
this keeps all three detectors' notion of "motor size" consistent, so
results are comparable across models. Box side = 2*HEATMAP_RADIUS + 1 = 19px.

A motor is placed on every z-slice within DEPTH (=3, same as unet2d) of its
centre — the ±3-slice "visible" window — not just its exact centre slice.

Usage:
    python src/yolo/prepare_dataset.py

Output:
    {DATA_ROOT}/train_tiles_rtdetr/images/{train,val}/{tomo}_z{zi}_y{y0}_x{x0}.png
    {DATA_ROOT}/train_tiles_rtdetr/labels/{train,val}/{tomo}_z{zi}_y{y0}_x{x0}.txt
    {DATA_ROOT}/test_tiles_rtdetr/images/{tomo}_z{zi}_y{y0}_x{x0}.png
        (test/ has no labels — inference-only, no labels/ dir written)
    src/rtdetr/baseline/data.yaml
"""

import os
import math
import zlib
from multiprocessing import Pool
import numpy as np
import pandas as pd
from PIL import Image

DATA_ROOT = os.environ.get("FLAGELLA_DATA", "/data/quokka/ws/15/ledy730h-flagella")
TRAIN_NPY_DIR = f"{DATA_ROOT}/train_npy/"
TEST_NPY_DIR = f"{DATA_ROOT}/test_npy/"
LABELS = f"{DATA_ROOT}/train_labels.csv"

TRAIN_TILES_DIR = f"{DATA_ROOT}/train_tiles_rtdetr/"
TEST_TILES_DIR = f"{DATA_ROOT}/test_tiles_rtdetr/"
DATA_YAML = os.path.join(os.path.dirname(__file__), "data.yaml")

SEED = 57
VALIDATION_FRAC = 0.2

SIGMA = 3.0
HEATMAP_RADIUS = int(3 * SIGMA)          # same truncation radius as unet2d/unet3d
BOX_SIZE = 2 * HEATMAP_RADIUS + 1        # 19px square, centred on the motor
DEPTH = int(SIGMA)                       # motor "visible" within +/- 3 slices, matches unet2d

TILE_SIZE = 640                          # YOLOv8 default input size, no overlap

NEG2MOTOR_RATIO = 282                    # same ratio as unet2d/unet3d training


def train_val_split(tomo_ids, validation_frac, seed):
    """Identical to unet2d/unet3d's train_val_split."""
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(tomo_ids))
    n_val = int(len(tomo_ids) * validation_frac)
    val_ids = {tomo_ids[i] for i in perm[:n_val]}
    train_ids = [t for t in tomo_ids if t not in val_ids]
    assert val_ids.isdisjoint(set(train_ids))
    return train_ids, sorted(val_ids)


def visible_motors(motors, zi, depth=DEPTH):
    return [(cz, cy, cx) for cz, cy, cx in motors if abs(zi - cz) <= depth]


def tile_origin(coord, tile_size):
    """Which fixed-grid tile index a coordinate falls in, and the tile's origin."""
    idx = coord // tile_size
    return idx, idx * tile_size


def yolo_label_line(cy, cx, y0, x0, tile_h, tile_w, box_size):
    """One YOLO-format label line: class cx cy w h, normalised 0-1 within the tile."""
    ly, lx = cy - y0, cx - x0
    return (f"0 {lx / tile_w:.6f} {ly / tile_h:.6f} "
            f"{box_size / tile_w:.6f} {box_size / tile_h:.6f}")


def _save_tile(vol, zi, y0, x0, tile_size, out_path):
    H, W = vol.shape[1], vol.shape[2]
    y1, x1 = min(y0 + tile_size, H), min(x0 + tile_size, W)
    tile = np.asarray(vol[zi, y0:y1, x0:x1])
    # Pad ragged edge tiles (tomogram dims aren't exact multiples of TILE_SIZE)
    # up to the full tile size, so every tile YOLO sees is TILE_SIZE x TILE_SIZE.
    if tile.shape != (tile_size, tile_size):
        padded = np.zeros((tile_size, tile_size), dtype=tile.dtype)
        padded[:tile.shape[0], :tile.shape[1]] = tile
        tile = padded
    Image.fromarray(tile, mode="L").save(out_path)
    return tile.shape[0], tile.shape[1]


def _build_tomo_tiles(args):
    (tomo, npy_dir, images_dir, labels_dir, motors, D, H, W,
     tile_size, box_size, depth, neg_ratio, avg_motors) = args

    vol = np.load(os.path.join(npy_dir, f"{tomo}.npy"), mmap_mode="r")
    n_tiles_y = math.ceil(H / tile_size)
    n_tiles_x = math.ceil(W / tile_size)

    written = 0

    # Positives: every (motor, visible z-slice) pair -> the one grid tile
    # containing that motor's (cy, cx). Multiple motors landing in the same
    # tile on the same slice share one label file with multiple lines.
    tile_labels = {}  # (zi, tile_iy, tile_ix) -> list of label lines
    for cz, cy, cx in motors:
        for zi in range(max(cz - depth, 0), min(cz + depth + 1, D)):
            tile_iy, y0 = tile_origin(cy, tile_size)
            tile_ix, x0 = tile_origin(cx, tile_size)
            key = (zi, tile_iy, tile_ix)
            tile_labels.setdefault(key, []).append((cy, cx, y0, x0))

    for (zi, tile_iy, tile_ix), motor_list in tile_labels.items():
        y0, x0 = tile_iy * tile_size, tile_ix * tile_size
        name = f"{tomo}_z{zi:04d}_y{y0:05d}_x{x0:05d}"
        img_path = os.path.join(images_dir, f"{name}.png")
        th, tw = _save_tile(vol, zi, y0, x0, tile_size, img_path)
        lines = [yolo_label_line(cy, cx, y0, x0, th, tw, box_size)
                 for cy, cx, y0, x0 in motor_list]
        with open(os.path.join(labels_dir, f"{name}.txt"), "w") as f:
            f.write("\n".join(lines) + "\n")
        written += 1

    # Negatives: neg_ratio * motor_count (or avg_motors if this tomogram has
    # none) random (zi, tile) grid cells with no motor visible on that slice.
    # An empty label file is written (YOLO convention for "no objects").
    # Seeded per tomogram, so the empty tiles are reproducible regardless of
    # worker order.
    rng = np.random.default_rng([SEED, zlib.crc32(tomo.encode())])
    n_motors = len(motors) if motors else avg_motors
    n_neg = round(neg_ratio * n_motors)
    tries = 0
    max_tries = n_neg * 20 + 20
    n_written_neg = 0
    while n_written_neg < n_neg and tries < max_tries:
        tries += 1
        zi = int(rng.integers(0, D))
        tile_iy = int(rng.integers(0, n_tiles_y))
        tile_ix = int(rng.integers(0, n_tiles_x))
        key = (zi, tile_iy, tile_ix)
        if key in tile_labels or not visible_motors(motors, zi):
            if key in tile_labels:
                continue
            y0, x0 = tile_iy * tile_size, tile_ix * tile_size
            name = f"{tomo}_z{zi:04d}_y{y0:05d}_x{x0:05d}"
            img_path = os.path.join(images_dir, f"{name}.png")
            if os.path.exists(img_path):
                continue
            _save_tile(vol, zi, y0, x0, tile_size, img_path)
            open(os.path.join(labels_dir, f"{name}.txt"), "w").close()
            n_written_neg += 1
            written += 1

    return tomo, written


def build_train_tiles(tomo_ids, labels, split_name, images_dir, labels_dir, num_workers):
    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(labels_dir, exist_ok=True)

    tomo_infos = []
    for tomo in tomo_ids:
        rows = labels[labels["tomo_id"] == tomo]
        D = int(rows.iloc[0]["Array shape (axis 0)"])
        H = int(rows.iloc[0]["Array shape (axis 1)"])
        W = int(rows.iloc[0]["Array shape (axis 2)"])
        motors = [
            (int(r["Motor axis 0"]), int(r["Motor axis 1"]), int(r["Motor axis 2"]))
            for _, r in rows.iterrows() if r["Number of motors"] > 0
        ]
        tomo_infos.append((tomo, motors, D, H, W))

    motor_counts = [len(m) for _, m, _, _, _ in tomo_infos if m]
    avg_motors = sum(motor_counts) / len(motor_counts) if motor_counts else 0.0

    tasks = [
        (tomo, TRAIN_NPY_DIR, images_dir, labels_dir, motors, D, H, W,
         TILE_SIZE, BOX_SIZE, DEPTH, NEG2MOTOR_RATIO, avg_motors)
        for tomo, motors, D, H, W in tomo_infos
    ]

    total = 0
    if num_workers <= 1:
        for task in tasks:
            tomo, n = _build_tomo_tiles(task)
            total += n
            print(f"[{split_name}] {tomo}: {n} tiles", flush=True)
    else:
        with Pool(num_workers) as pool:
            for tomo, n in pool.imap_unordered(_build_tomo_tiles, tasks):
                total += n
                print(f"[{split_name}] {tomo}: {n} tiles", flush=True)

    print(f"[{split_name}] total: {total} tiles", flush=True)
    return total


def _build_test_tomo_tiles(args):
    tomo, npy_dir, images_dir, tile_size = args
    vol = np.load(os.path.join(npy_dir, f"{tomo}.npy"), mmap_mode="r")
    D, H, W = vol.shape
    n_tiles_y = math.ceil(H / tile_size)
    n_tiles_x = math.ceil(W / tile_size)

    written = 0
    for zi in range(D):
        for tile_iy in range(n_tiles_y):
            for tile_ix in range(n_tiles_x):
                y0, x0 = tile_iy * tile_size, tile_ix * tile_size
                name = f"{tomo}_z{zi:04d}_y{y0:05d}_x{x0:05d}"
                img_path = os.path.join(images_dir, f"{name}.png")
                _save_tile(vol, zi, y0, x0, tile_size, img_path)
                written += 1
    return tomo, written


def build_test_tiles(tomo_ids, images_dir, num_workers):
    os.makedirs(images_dir, exist_ok=True)
    tasks = [(tomo, TEST_NPY_DIR, images_dir, TILE_SIZE) for tomo in tomo_ids]

    total = 0
    if num_workers <= 1:
        for task in tasks:
            tomo, n = _build_test_tomo_tiles(task)
            total += n
            print(f"[test] {tomo}: {n} tiles", flush=True)
    else:
        with Pool(num_workers) as pool:
            for tomo, n in pool.imap_unordered(_build_test_tomo_tiles, tasks):
                total += n
                print(f"[test] {tomo}: {n} tiles", flush=True)

    print(f"[test] total: {total} tiles", flush=True)
    return total


def write_data_yaml(train_images_dir, val_images_dir, path):
    with open(path, "w") as f:
        f.write(
            f"train: {os.path.abspath(train_images_dir)}\n"
            f"val: {os.path.abspath(val_images_dir)}\n"
            f"nc: 1\n"
            f"names: ['motor']\n"
        )
    print(f"wrote {path}", flush=True)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=os.cpu_count() or 1)
    args = parser.parse_args()

    labels = pd.read_csv(LABELS)
    tomo_ids = sorted(labels.tomo_id.unique())
    train_ids, val_ids = train_val_split(tomo_ids, VALIDATION_FRAC, SEED)

    train_images = os.path.join(TRAIN_TILES_DIR, "images", "train")
    train_labels_dir = os.path.join(TRAIN_TILES_DIR, "labels", "train")
    val_images = os.path.join(TRAIN_TILES_DIR, "images", "val")
    val_labels_dir = os.path.join(TRAIN_TILES_DIR, "labels", "val")

    build_train_tiles(train_ids, labels, "train", train_images, train_labels_dir, args.workers)
    build_train_tiles(val_ids, labels, "val", val_images, val_labels_dir, args.workers)

    test_ids = sorted(
        d for d in os.listdir(TEST_NPY_DIR) if d.endswith(".npy")
    )
    test_ids = [t[:-4] for t in test_ids]
    if test_ids:
        test_images = os.path.join(TEST_TILES_DIR, "images")
        build_test_tiles(test_ids, test_images, args.workers)

    write_data_yaml(train_images, val_images, DATA_YAML)
