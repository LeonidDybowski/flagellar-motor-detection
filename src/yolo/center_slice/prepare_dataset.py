"""YOLO center-slice experiment — dataset.

Launch via prepare_dataset.sbatch (after src/yolo/prepare_dataset.py):
    python src/yolo/center_slice/prepare_dataset.py

Same tiles as the baseline ({DATA_ROOT}/train_tiles_yolo/), but a motor is
labelled only on its own centre slice instead of every slice within +/-3:
  * motor tiles: the baseline tile on each motor's centre slice, labelled
    with only the motors whose centre slice it is. Baseline motor tiles on
    the +/-1..3 slices are dropped (neither motor nor empty);
  * empty tiles: drawn from the baseline's empty tiles (motor-free slices),
    as many as keep motors POS_OVERSAMPLING times more frequent than in the
    raw data. The raw ratio is counted per split as (all 640x640 grid tiles
    of every slice - centre-slice motor tiles) / centre-slice motor tiles —
    the same rule that gives the baseline's ~4.9x (1:72 vs raw 1:354).
Images are hard links to the baseline PNGs (no copies); label files are new.

Output:
    {DATA_ROOT}/train_tiles_center/images/{train,val}/*.png
    {DATA_ROOT}/train_tiles_center/labels/{train,val}/*.txt
    src/yolo/center_slice/data.yaml
"""

import math
import os
import random
import shutil
import numpy as np
import pandas as pd

DATA_ROOT = os.environ.get("FLAGELLA_DATA", "/data/quokka/ws/15/ledy730h-flagella")
LABELS = f"{DATA_ROOT}/train_labels.csv"

SEED = 57
VALIDATION_FRAC = 0.2

SIGMA = 3.0
HEATMAP_RADIUS = int(3 * SIGMA)          # same truncation radius as unet2d/unet3d
BOX_SIZE = 2 * HEATMAP_RADIUS + 1        # 19px square, centred on the motor

TILE_SIZE = 640                          # YOLOv8 default input size, no overlap


BASE_DIR = f"{DATA_ROOT}/train_tiles_yolo/"
OUT_DIR = f"{DATA_ROOT}/train_tiles_center/"
DATA_YAML = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data.yaml")

POS_OVERSAMPLING = 5     # motor tiles this many times more frequent than in the raw data


def train_val_split(tomo_ids, validation_frac, seed):
    """Identical to unet2d/unet3d's train_val_split."""
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(tomo_ids))
    n_val = int(len(tomo_ids) * validation_frac)
    val_ids = {tomo_ids[i] for i in perm[:n_val]}
    train_ids = [t for t in tomo_ids if t not in val_ids]
    assert val_ids.isdisjoint(set(train_ids))
    return train_ids, sorted(val_ids)


def tile_origin(coord, tile_size):
    """Which fixed-grid tile index a coordinate falls in, and the tile's origin."""
    idx = coord // tile_size
    return idx, idx * tile_size


def yolo_label_line(cy, cx, y0, x0, tile_h, tile_w, box_size):
    """One YOLO-format label line: class cx cy w h, normalised 0-1 within the tile."""
    ly, lx = cy - y0, cx - x0
    return (f"0 {lx / tile_w:.6f} {ly / tile_h:.6f} "
            f"{box_size / tile_w:.6f} {box_size / tile_h:.6f}")


def write_data_yaml(train_images_dir, val_images_dir, path):
    with open(path, "w") as f:
        f.write(
            f"train: {os.path.abspath(train_images_dir)}\n"
            f"val: {os.path.abspath(val_images_dir)}\n"
            f"nc: 1\n"
            f"names: ['motor']\n"
        )
    print(f"wrote {path}", flush=True)


def link(src, dst):
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def build_split(split, tomo_ids, labels, rng):
    base_img = os.path.join(BASE_DIR, "images", split)
    base_lab = os.path.join(BASE_DIR, "labels", split)
    out_img = os.path.join(OUT_DIR, "images", split)
    out_lab = os.path.join(OUT_DIR, "labels", split)
    os.makedirs(out_img, exist_ok=True)
    os.makedirs(out_lab, exist_ok=True)

    tiles = {}                                       # name -> label lines
    total = 0                                        # all grid tiles of all slices
    for tomo in tomo_ids:
        rows = labels[labels["tomo_id"] == tomo]
        D, H, W = (int(rows.iloc[0][f"Array shape (axis {i})"]) for i in range(3))
        total += D * math.ceil(H / TILE_SIZE) * math.ceil(W / TILE_SIZE)
        for _, r in rows.iterrows():
            if r["Number of motors"] <= 0:
                continue
            cz, cy, cx = int(r["Motor axis 0"]), int(r["Motor axis 1"]), int(r["Motor axis 2"])
            _, y0 = tile_origin(cy, TILE_SIZE)
            _, x0 = tile_origin(cx, TILE_SIZE)
            name = f"{tomo}_z{cz:04d}_y{y0:05d}_x{x0:05d}"
            tiles.setdefault(name, []).append(
                yolo_label_line(cy, cx, y0, x0, TILE_SIZE, TILE_SIZE, BOX_SIZE))

    empty = sorted(e.name[:-4] for e in os.scandir(base_lab) if e.stat().st_size == 0)
    n_pos = len(tiles)
    raw = (total - n_pos) / n_pos
    n_empty = round(n_pos * raw / POS_OVERSAMPLING)
    if n_empty > len(empty):
        raise ValueError(f"[{split}] need {n_empty} empty tiles, baseline has {len(empty)}")
    empty = rng.sample(empty, n_empty)

    for name, lines in tiles.items():
        src = os.path.join(base_img, f"{name}.png")
        if not os.path.exists(src):
            raise FileNotFoundError(f"{src}: baseline is missing a centre-slice motor tile")
        link(src, os.path.join(out_img, f"{name}.png"))
        with open(os.path.join(out_lab, f"{name}.txt"), "w") as f:
            f.write("\n".join(lines) + "\n")
    for name in empty:
        link(os.path.join(base_img, f"{name}.png"), os.path.join(out_img, f"{name}.png"))
        open(os.path.join(out_lab, f"{name}.txt"), "w").close()

    print(f"[{split}] {n_pos} motor tiles ({sum(map(len, tiles.values()))} motors), "
          f"{n_empty} empty tiles = 1:{n_empty / n_pos:.1f}; raw 1:{raw:.0f} "
          f"({total} grid tiles); motors oversampled {raw / (n_empty / n_pos):.2f}x", flush=True)
    return out_img


def main():
    if os.path.exists(OUT_DIR):
        raise FileExistsError(f"{OUT_DIR} exists — remove it to rebuild")
    labels = pd.read_csv(LABELS)
    tomo_ids = sorted(labels.tomo_id.unique())
    train_ids, val_ids = train_val_split(tomo_ids, VALIDATION_FRAC, SEED)
    rng = random.Random(SEED)
    train_img = build_split("train", train_ids, labels, rng)
    val_img = build_split("val", val_ids, labels, rng)
    write_data_yaml(train_img, val_img, DATA_YAML)


if __name__ == "__main__":
    main()
