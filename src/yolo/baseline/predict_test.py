"""yolo baseline — predictions on the unlabelled test tomograms.

Launch via predict_test.sbatch:
    python src/yolo/baseline/predict_test.py

Copy of src/yolo/baseline/fbeta.py with the scoring removed: the test tomograms
({DATA_ROOT}/test_npy/) are scanned and decoded exactly as validation
tomograms are, with the baseline weights (results/yolo/baseline/train/weights/best.pt).
The test set has no labels, so nothing is scored.

Output:
    results/yolo/baseline/test_predictions.csv  positions (tomo_id, z, y, x, score) at the
        threshold that scored best on validation (results/yolo/baseline/fbeta.csv,
        required); no threshold sweep — the test set has no labels.
"""

import os
import re
import glob
import argparse
import tempfile
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from scipy.ndimage import label
from scipy.optimize import linear_sum_assignment
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree
from sklearn.metrics import fbeta_score
from ultralytics import YOLO

DATA_ROOT = os.environ.get("FLAGELLA_DATA", "/data/quokka/ws/15/ledy730h-flagella")
TEST_NPY_DIR = f"{DATA_ROOT}/test_npy/"
LABELS = f"{DATA_ROOT}/train_labels.csv"

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
# Experiment baseline (fixed for this folder); results go to results/yolo/baseline/.
#   baseline   non-overlapping 640x640 tiles
#   stitching  tiles overlapping by half (evaluation only, on the baseline
#              weights); the same motor boxed in two tiles gives one position
#              through the window decoder
# (The U-Nets' "mse" and "blobs" experiments do not apply: the detectors have
# their own losses and output boxes, not a probability map.)
EXPERIMENT = "baseline"
CFG = {"overlap": False, "weights": "baseline"}

RESULT_DIR = os.path.join(ROOT, "results", "yolo", EXPERIMENT)
WEIGHTS_DIR = os.path.join(ROOT, "results", "yolo", CFG["weights"], "train", "weights")
CHECKPOINT = os.environ.get("FLAGELLA_YOLO_CKPT", f"{WEIGHTS_DIR}/best.pt")

SEED = 57
VALIDATION_FRAC = 0.2

TILE_SIZE = 640                          # matches prepare_dataset.py
IMG_SIZE = 640
BATCH = 64                               # tiles per predict call, per GPU
# Same value as the U-Nets' NMS_RADIUS (3 * SIGMA) and the half-width of the
# 19 px training boxes (prepare_dataset.py's HEATMAP_RADIUS).
NMS_RADIUS = 9
MODEL_CLASS = YOLO
MODEL_NAME = "yolo"


# ----------------------------------------------------------------------------
# Evaluation: decode positions per tomogram, then score them per motor
# ----------------------------------------------------------------------------
# Step 1 — positions. The tiles' probabilities are written into one
# probability volume per tomogram (in GPU memory; where tiles overlap, as in
# the stitching experiment, the overlaps are averaged), and positions are
# decoded from that volume with one of two decoders. Scores are used here
# and nowhere else.
#   "window": a voxel is a position if it is the maximum of its
#       (2 * NMS_RADIUS + 1)^3 neighbourhood and at least the threshold.
#   "blobs":  voxels at least the threshold are grouped into connected
#       regions (26-connectivity); each region gives one position, at its
#       maximum.
#   Where several voxels share a maximum exactly (a tie), the one nearest
#   the centre of the tied voxels is kept.
#
# Step 2 — metric, on positions only. Distances are Euclidean, in angstrom
# (voxel distance times the tomogram's voxel spacing). Positions and motors
# are paired one-to-one by the Hungarian algorithm on these distances, pairs
# farther than MIN_RADIUS not allowed, so the pairing finds as many motors
# as possible. A paired motor is a TP, an unpaired motor an FN, an unpaired
# position an FP (including a second position at an already-found motor).
#     F_beta = (1 + beta^2) TP / ((1 + beta^2) TP + beta^2 FN + FP),  beta = 2
# computed with sklearn.metrics.fbeta_score.

BETA = 2
MIN_RADIUS = 1000.0                      # angstrom
def validation_threshold():
    """The threshold with the best per-motor F-beta on validation
    (RESULT_DIR/fbeta.csv, written by the validation run)."""
    sweep = f"{RESULT_DIR}/fbeta.csv"
    if not os.path.exists(sweep):
        raise FileNotFoundError(f"{sweep} not found — run the validation first.")
    rows = pd.read_csv(sweep)
    return float(rows.loc[rows.fbeta.idxmax(), "threshold"])


# Only the validation threshold: the test set has no labels to choose one by.
SCORE_THRESHOLDS = [validation_threshold()]
DECODE_CHUNK = 32                        # slices decoded at a time, bounds GPU memory


def box_max(x, radius):
    """Maximum over the (2 * radius + 1)-wide box around every element of a
    2D or 3D tensor, computed one axis at a time (equal to the full box)."""
    k = 2 * radius + 1
    y = x[None, None]
    if x.dim() == 2:
        y = torch.nn.functional.max_pool2d(y, (k, 1), 1, (radius, 0))
        y = torch.nn.functional.max_pool2d(y, (1, k), 1, (0, radius))
    else:
        y = torch.nn.functional.max_pool3d(y, (k, 1, 1), 1, (radius, 0, 0))
        y = torch.nn.functional.max_pool3d(y, (1, k, 1), 1, (0, radius, 0))
        y = torch.nn.functional.max_pool3d(y, (1, 1, k), 1, (0, 0, radius))
    return y[0, 0]


def central_of_groups(coords, groups):
    """Indices of one element per group: the one nearest the group's centre
    (first one on an exact tie). coords: (N, d) array, groups: (N,) labels."""
    keep = []
    for g in np.unique(groups):
        members = np.nonzero(groups == g)[0]
        centre = coords[members].mean(axis=0)
        keep.append(members[np.argmin(((coords[members] - centre) ** 2).sum(axis=1))])
    return np.sort(np.asarray(keep, dtype=np.int64))


def decode_peaks(heatmap, threshold, radius):
    """Window decoder on a 2D or 3D heatmap: (N, ndim + 1) of coordinates
    and score.

    A pixel/voxel is a candidate if it equals the maximum of its
    (2 * radius + 1)-wide box, is at least `threshold`, and the box is not
    flat (a flat region is a maximum of nothing). Two candidates inside one
    box always have the same value (each is the maximum of a box containing
    the other), so candidates within `radius` of each other are ties; each
    group of tied candidates keeps only its most central one.
    """
    pooled = box_max(heatmap, radius)
    floored = -box_max(-heatmap, radius)
    cand = (heatmap == pooled) & (heatmap >= threshold) & (pooled > floored)
    idx = torch.nonzero(cand).cpu().numpy()
    if len(idx) > 1:
        pairs = cKDTree(idx).query_pairs(r=radius, p=np.inf, output_type="ndarray")
        if len(pairs):
            graph = coo_matrix((np.ones(len(pairs)), (pairs[:, 0], pairs[:, 1])),
                               shape=(len(idx), len(idx)))
            _, groups = connected_components(graph, directed=False)
            idx = idx[central_of_groups(idx, groups)]
    idx_t = torch.from_numpy(idx).to(heatmap.device)
    scores = heatmap[tuple(idx_t.T)] if len(idx) else heatmap.new_zeros(0)
    return torch.cat([idx_t.float(), scores.float()[:, None]], dim=1)


def decode_volume(vol, threshold, radius, chunk=DECODE_CHUNK):
    """decode_peaks over a whole (D, H, W) volume, `chunk` slices at a time
    with a `radius`-slice halo, so candidates are the same as decoding it in
    one go (a tie group split by a chunk border is resolved on each side)."""
    D = vol.shape[0]
    out = []
    for z0 in range(0, D, chunk):
        a, b = max(0, z0 - radius), min(D, z0 + chunk + radius)
        pk = decode_peaks(vol[a:b], threshold, radius)
        pk[:, 0] += a
        out.append(pk[(pk[:, 0] >= z0) & (pk[:, 0] < z0 + chunk)])
    return torch.cat(out).cpu()


def decode_blobs(vol, threshold):
    """Blob decoder: one position per connected region (26-connectivity) of
    voxels >= threshold, at the region's maximum (most central voxel among
    equal maxima). (N, 4) z, y, x, score. Labelled on CPU (scipy)."""
    v = vol.cpu().numpy()
    labels, n = label(v >= threshold, structure=np.ones((3, 3, 3), dtype=bool))
    if n == 0:
        return torch.zeros((0, 4))
    inside = np.nonzero(labels)                  # only the voxels above threshold
    lab, val = labels[inside], v[inside]
    blob_max = np.full(n + 1, -np.inf, dtype=np.float32)
    np.maximum.at(blob_max, lab, val)
    at_max = val == blob_max[lab]
    idx = np.stack(inside, axis=1)[at_max]
    idx = idx[central_of_groups(idx, lab[at_max])]
    scores = v[tuple(idx.T)]
    return torch.from_numpy(np.concatenate([idx, scores[:, None]], axis=1).astype(np.float32))


def decode_tomogram(vol, decoder, radius, thresholds=SCORE_THRESHOLDS):
    """{threshold: (N, 4) z, y, x, score} for one tomogram's probability volume."""
    if decoder == "window":
        # Candidates do not depend on the threshold, only which ones pass it:
        # decode once at the lowest threshold, filter for the others.
        pk = decode_volume(vol, min(thresholds), radius)
        return {thr: pk[pk[:, 3] >= thr] for thr in thresholds}
    if decoder == "blobs":
        # Regions split as the threshold rises, so every threshold is decoded.
        return {thr: decode_blobs(vol, thr) for thr in thresholds}
    raise ValueError(decoder)


class TomogramVolumes:
    """Assembles tile/patch probabilities into one volume per tomogram (in
    GPU memory) and decodes it once the tomogram is complete.

    The loader must yield a rank's tiles grouped by tomogram (the datasets
    are ordered that way and each rank gets whole tomograms), so only one
    volume is held at a time. The padded part of an edge tile is cut off;
    where tiles overlap (stitching), their probabilities are averaged.
    """

    def __init__(self, shapes, device, radius, decoder):
        self.shapes = shapes
        self.device = device
        self.radius = radius
        self.decoder = decoder
        self.ti = None
        self.vol = None
        self.count = None
        self.peaks = {}

    def add(self, probs, meta):
        for b, (ti, z0, y0, x0) in enumerate(meta.tolist()):
            if ti != self.ti:
                self._finish()
                self.ti = ti
                self.vol = torch.zeros(self.shapes[ti], device=self.device)
                self.count = torch.zeros(self.shapes[ti], device=self.device, dtype=torch.uint8)
            D, H, W = self.shapes[ti]
            tile = probs[b, 0]
            if tile.dim() == 2:                       # one slice of a 2D model
                h, w = min(tile.shape[0], H - y0), min(tile.shape[1], W - x0)
                region = (z0, slice(y0, y0 + h), slice(x0, x0 + w))
                tile = tile[:h, :w]
            else:
                d = min(tile.shape[0], D - z0)
                h, w = min(tile.shape[1], H - y0), min(tile.shape[2], W - x0)
                region = (slice(z0, z0 + d), slice(y0, y0 + h), slice(x0, x0 + w))
                tile = tile[:d, :h, :w]
            self.vol[region] += tile
            self.count[region] += 1

    def _finish(self):
        if self.vol is not None:
            self.vol /= self.count.clamp(min=1)
            self.peaks[self.ti] = decode_tomogram(self.vol, self.decoder, self.radius)
            self.vol = self.count = None

    def result(self, n_tomos):
        """{threshold: (N, 4) z, y, x, score} per tomogram, merged across ranks."""
        self._finish()
        peaks = dict(self.peaks)
        if dist.is_available() and dist.is_initialized():
            parts = [None] * dist.get_world_size()
            dist.all_gather_object(parts, peaks)
            for part in parts:
                peaks.update(part)
        empty = {thr: torch.zeros((0, 4)) for thr in SCORE_THRESHOLDS}
        return [peaks.get(ti, empty) for ti in range(n_tomos)]


def tomo_truth(labels, tomo_ids):
    """{tomo_id: (motors, voxel_spacing)} straight from the label table."""
    truth = {}
    for tomo in tomo_ids:
        rows = labels[labels["tomo_id"] == tomo]
        motors = [
            (int(r["Motor axis 0"]), int(r["Motor axis 1"]), int(r["Motor axis 2"]))
            for _, r in rows.iterrows() if r["Number of motors"] > 0
        ]
        truth[tomo] = (motors, float(rows.iloc[0]["Voxel spacing"]))
    return truth


def match_motors(positions, motors, spacing, radius=MIN_RADIUS):
    """Pair predicted positions with motors one-to-one; positions only.

    Hungarian algorithm on the angstrom distances, with pairs farther than
    `radius` forbidden (cost above any sum of allowed distances), so the
    pairing has as many pairs as possible and, among those, the smallest
    total distance. Returns [(position index, motor index, distance in
    angstrom)] for the TP pairs; unpaired motors are FNs, unpaired positions
    FPs.
    """
    if len(positions) == 0 or not motors:
        return []
    d = spacing * np.linalg.norm(
        np.asarray(positions, dtype=np.float64)[:, None, :]
        - np.asarray(motors, dtype=np.float64)[None, :, :], axis=2)  # (N, n_motors)
    near = np.nonzero((d <= radius).any(axis=1))[0]   # only these can pair
    if len(near) == 0:
        return []
    sub = d[near]
    forbidden = 1.0 + radius * min(sub.shape)
    rows, cols = linear_sum_assignment(np.where(sub <= radius, sub, forbidden))
    return [(int(near[r]), int(c), float(sub[r, c]))
            for r, c in zip(rows, cols) if sub[r, c] <= radius]


def count_matches(truth, tomo_ids, peaks, threshold):
    """Total (tp, fp, fn) over all tomograms for the positions decoded at
    `threshold`."""
    tp = fp = fn = 0
    for tomo, p in zip(tomo_ids, peaks):
        motors, spacing = truth[tomo]
        positions = p[threshold][:, :3].tolist()
        hits = len(match_motors(positions, motors, spacing))
        tp += hits
        fp += len(positions) - hits
        fn += len(motors) - hits
    return tp, fp, fn


def fbeta_from_counts(tp, fp, fn, beta=BETA):
    """sklearn's F-beta on the per-motor / per-position outcomes."""
    y_true = [1] * tp + [1] * fn + [0] * fp
    y_pred = [1] * tp + [0] * fn + [1] * fp
    if not y_true:
        return 0.0
    return float(fbeta_score(y_true, y_pred, beta=beta, zero_division=0))


def fbeta_sweep(truth, tomo_ids, peaks, thresholds=SCORE_THRESHOLDS):
    """F-beta, precision, recall and tp/fp/fn at every score threshold."""
    rows = []
    for thr in thresholds:
        tp, fp, fn = count_matches(truth, tomo_ids, peaks, thr)
        rows.append({
            "threshold": thr, "fbeta": fbeta_from_counts(tp, fp, fn),
            "precision": tp / (tp + fp) if (tp + fp) > 0 else float("nan"),
            "recall": tp / (tp + fn) if (tp + fn) > 0 else float("nan"),
            "tp": tp, "fp": fp, "fn": fn,
        })
    return rows


def best_row(rows):
    return max(rows, key=lambda r: r["fbeta"])


def train_val_split(tomo_ids, validation_frac, seed):
    """Identical to unet2d/unet3d's train_val_split."""
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(tomo_ids))
    n_val = int(len(tomo_ids) * validation_frac)
    val_ids = {tomo_ids[i] for i in perm[:n_val]}
    train_ids = [t for t in tomo_ids if t not in val_ids]
    assert val_ids.isdisjoint(set(train_ids))
    return train_ids, sorted(val_ids)


def tile_origins(n, tile_size, stride):
    """Origins of tile_size-wide tiles every `stride` covering [0, n)."""
    origins = [0]
    while origins[-1] + tile_size < n:
        origins.append(origins[-1] + stride)
    return origins


def tile_batches(vol, tile_size=TILE_SIZE, batch=BATCH):
    """All tiles of a (D, H, W) volume, slice by slice on prepare_dataset.py's
    fixed grid, as batches of (3-channel uint8 tiles, (zi, y0, x0) origins).

    Ragged edge tiles are zero-padded to tile_size exactly as
    prepare_dataset.py's _save_tile does. The training PNGs are grayscale and
    Ultralytics reads them as 3 identical channels, hence the stacking.
    In the stitching experiment tiles overlap by half.
    """
    D, H, W = vol.shape
    stride = tile_size // 2 if CFG["overlap"] else tile_size
    imgs, metas = [], []
    for zi in range(D):
        for y0 in tile_origins(H, tile_size, stride):
            for x0 in tile_origins(W, tile_size, stride):
                tile = np.zeros((tile_size, tile_size), dtype=np.uint8)
                part = vol[zi, y0:y0 + tile_size, x0:x0 + tile_size]
                tile[:part.shape[0], :part.shape[1]] = part
                imgs.append(np.repeat(tile[:, :, None], 3, axis=2))
                metas.append((zi, y0, x0))
                if len(imgs) == batch:
                    yield imgs, metas
                    imgs, metas = [], []
    if imgs:
        yield imgs, metas


def centres_to_peaks(found, shape, device):
    """Box centres [(z, y, x, conf)] of one tomogram -> decoded positions
    {threshold: (N, 4) z, y, x, score}.

    Each centre's confidence goes into a (D, H, W) volume at its voxel (the
    higher one where two land on the same voxel); decode_volume then applies
    the same neighbourhood rule as the U-Nets.
    """
    D, H, W = shape
    pts = torch.tensor(found, dtype=torch.float64)
    z = pts[:, 0].round().long().clamp(0, D - 1)
    y = pts[:, 1].round().long().clamp(0, H - 1)       # centres in a zero-padded
    x = pts[:, 2].round().long().clamp(0, W - 1)       # edge tile can fall outside
    vol = torch.zeros(D * H * W, device=device)
    vol.scatter_reduce_(0, ((z * H + y) * W + x).to(device),
                        pts[:, 3].float().to(device), reduce="amax")
    return decode_tomogram(vol.view(D, H, W), "window", NMS_RADIUS)


def scan_worker(rank, n_ranks, ckpts, tomo_ids, out_dir):
    """Decoded positions for every checkpoint, for tomograms rank::n_ranks.

    Writes {out_dir}/peaks_{rank}.pt: peaks[ci][ti] = {threshold: (N, 4)
    z, y, x, score}, empty where this rank did not scan the tomogram.
    """
    device = torch.device("cuda", rank) if torch.cuda.is_available() else torch.device("cpu")
    models = [MODEL_CLASS(c) for c in ckpts]
    min_conf = min(SCORE_THRESHOLDS)
    empty = {thr: torch.zeros((0, 4)) for thr in SCORE_THRESHOLDS}
    peaks = [[empty for _ in tomo_ids] for _ in ckpts]
    for ti in range(rank, len(tomo_ids), n_ranks):
        vol = np.load(os.path.join(TEST_NPY_DIR, f"{tomo_ids[ti]}.npy"))
        found = [[] for _ in ckpts]
        for imgs, metas in tile_batches(vol):
            for ci, model in enumerate(models):
                results = model.predict(imgs, imgsz=IMG_SIZE, conf=min_conf, device=rank,
                                        half=True, batch=len(imgs), verbose=False)
                for res, (zi, y0, x0) in zip(results, metas):
                    for (cx, cy), s in zip(res.boxes.xywh[:, :2].tolist(),
                                           res.boxes.conf.tolist()):
                        found[ci].append((zi, y0 + cy, x0 + cx, s))
        for ci in range(len(ckpts)):
            if found[ci]:
                peaks[ci][ti] = centres_to_peaks(found[ci], vol.shape, device)
        print(f"[gpu {rank}] {tomo_ids[ti]} done", flush=True)
    torch.save(peaks, os.path.join(out_dir, f"peaks_{rank}.pt"))


def scan(ckpts, tomo_ids):
    """peaks[ci][ti] = {threshold: (N, 4) z, y, x, score} per checkpoint and tomogram,
    one process per GPU (tomograms are split, so ranks never overlap)."""
    n_ranks = max(1, torch.cuda.device_count())
    with tempfile.TemporaryDirectory() as tmp:
        mp.spawn(scan_worker, args=(n_ranks, ckpts, tomo_ids, tmp), nprocs=n_ranks)
        parts = [torch.load(os.path.join(tmp, f"peaks_{r}.pt")) for r in range(n_ranks)]
    return [[parts[ti % n_ranks][ci][ti] for ti in range(len(tomo_ids))]
            for ci in range(len(ckpts))]


def epoch_of(path):
    """Ultralytics names per-epoch weights epoch{N}.pt with a 0-based N."""
    return int(re.fullmatch(r"epoch(\d+)\.pt", os.path.basename(path)).group(1)) + 1


def write_outputs(model_name, rows, peaks, truth, tomo_ids):
    """fbeta.csv, fbeta.png, peaks.csv and matches.csv under RESULT_DIR."""
    df = pd.DataFrame(rows)
    df.to_csv(f"{RESULT_DIR}/fbeta.csv", index=False)
    for r in rows:
        print(f"thr={r['threshold']:.2f}  tp={r['tp']:4d} fp={r['fp']:5d} fn={r['fn']:4d}  "
              f"P={r['precision']:.3f} R={r['recall']:.3f} F{BETA}={r['fbeta']:.3f}", flush=True)

    # Every decoded position at every threshold, so any threshold can be
    # re-scored later.
    pd.DataFrame(
        [{"threshold": thr, "tomo_id": tomo, "z": z, "y": y, "x": x, "score": s}
         for thr in SCORE_THRESHOLDS for tomo, p in zip(tomo_ids, peaks)
         for z, y, x, s in p[thr].tolist()],
        columns=["threshold", "tomo_id", "z", "y", "x", "score"],
    ).to_csv(f"{RESULT_DIR}/peaks.csv", index=False)

    # Outcomes at the best threshold: one row per motor (TP/FN) and one per
    # unpaired position (FP); distance_A of an FP is to its nearest motor.
    top = best_row(rows)
    out = []
    for tomo, p in zip(tomo_ids, peaks):
        motors, spacing = truth[tomo]
        p = p[top["threshold"]]
        pairs = match_motors(p[:, :3].tolist(), motors, spacing)
        by_motor = {mi: (pi, d) for pi, mi, d in pairs}
        paired = {pi for pi, _, _ in pairs}
        for mi, (mz, my, mx) in enumerate(motors):
            row = {"tomo_id": tomo, "case": "TP" if mi in by_motor else "FN",
                   "motor_z": mz, "motor_y": my, "motor_x": mx, "voxel_spacing": spacing}
            if mi in by_motor:
                pi, d = by_motor[mi]
                z, y, x, s = p[pi].tolist()
                row.update(pred_z=z, pred_y=y, pred_x=x, score=s, distance_A=d)
            out.append(row)
        for pi, (z, y, x, s) in enumerate(p.tolist()):
            if pi in paired:
                continue
            near = (min(spacing * float(np.sqrt((z - mz) ** 2 + (y - my) ** 2 + (x - mx) ** 2))
                        for mz, my, mx in motors) if motors else float("nan"))
            out.append({"tomo_id": tomo, "case": "FP", "voxel_spacing": spacing,
                        "pred_z": z, "pred_y": y, "pred_x": x, "score": s,
                        "distance_A": near})
    pd.DataFrame(out, columns=["tomo_id", "case", "motor_z", "motor_y", "motor_x",
                               "pred_z", "pred_y", "pred_x", "score", "distance_A",
                               "voxel_spacing"]).to_csv(f"{RESULT_DIR}/matches.csv", index=False)

    plt.figure(figsize=(7, 5))
    plt.plot(df["threshold"], df["fbeta"], "o-", label=f"F{BETA}")
    plt.plot(df["threshold"], df["precision"], "s--", label="precision")
    plt.plot(df["threshold"], df["recall"], "^--", label="recall")
    plt.xlabel("Confidence threshold")
    plt.ylabel("Score")
    plt.title(f"{model_name} {EXPERIMENT}: val, whole tomograms, per motor "
              f"(best F{BETA} {top['fbeta']:.3f} @ thr {top['threshold']:.2f})")
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{RESULT_DIR}/fbeta.png", dpi=150)
    plt.close()
    print(f"best F{BETA} {top['fbeta']:.3f} at threshold {top['threshold']:.2f}", flush=True)
    print(f"saved {RESULT_DIR}/fbeta.csv, fbeta.png, peaks.csv and matches.csv", flush=True)


def test_ids():
    return sorted(f[:-4] for f in os.listdir(TEST_NPY_DIR) if f.endswith(".npy"))


def write_predictions(peaks, tomo_ids):
    """test_predictions.csv: positions at the validation threshold."""
    os.makedirs(RESULT_DIR, exist_ok=True)
    thr = SCORE_THRESHOLDS[0]
    pd.DataFrame(
        [{"tomo_id": tomo, "z": z, "y": y, "x": x, "score": s, "threshold": thr}
         for tomo, p in zip(tomo_ids, peaks) for z, y, x, s in p[thr].tolist()],
        columns=["tomo_id", "z", "y", "x", "score", "threshold"],
    ).to_csv(f"{RESULT_DIR}/test_predictions.csv", index=False)
    for tomo, p in zip(tomo_ids, peaks):
        print(f"{tomo}: {len(p[thr])} positions at threshold {thr:.2f}", flush=True)
    print(f"saved {RESULT_DIR}/test_predictions.csv", flush=True)


def main():
    if not os.path.exists(CHECKPOINT):
        raise FileNotFoundError(f"{CHECKPOINT} not found — run train.py first.")
    ids = test_ids()
    print(f"experiment={EXPERIMENT} {CFG}", flush=True)
    print(f"{len(ids)} test tomograms  ckpt={CHECKPOINT}  "
          f"{max(1, torch.cuda.device_count())} GPUs", flush=True)
    peaks = scan([CHECKPOINT], ids)[0]
    write_predictions(peaks, ids)


if __name__ == "__main__":
    main()
