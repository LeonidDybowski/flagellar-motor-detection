"""unet2d baseline — predictions on the unlabelled test tomograms.

Launch via predict_test.sbatch:
    torchrun ... src/unet2d/baseline/predict_test.py

Copy of src/unet2d/baseline/evaluate.py with the scoring removed: the test tomograms
({DATA_ROOT}/test_npy/) are scanned and decoded exactly as validation
tomograms are, with the baseline weights (results/unet2d/baseline/best.pt).
The test set has no labels, so nothing is scored.

Output:
    results/unet2d/baseline/test_predictions.csv  positions (tomo_id, z, y, x, score) at the
        threshold that scored best on validation (val_threshold of the best epoch in
        results/unet2d/baseline/history.csv); no threshold sweep — the test set has no labels.
"""

import os
import functools
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.distributed as dist
from torch.utils.data import Dataset, DataLoader, Subset
from monai.networks.nets import UNet
from scipy.ndimage import label
from scipy.optimize import linear_sum_assignment
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree
from sklearn.metrics import fbeta_score

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.benchmark = True

DATA_ROOT = os.environ.get("FLAGELLA_DATA", "/data/quokka/ws/15/ledy730h-flagella")
TEST_NPY_DIR = f"{DATA_ROOT}/test_npy/"
LABELS = f"{DATA_ROOT}/train_labels.csv"

# Experiment baseline (fixed for this folder); results go to results/unet2d/baseline/.
#   baseline   focal loss, window decoder, non-overlapping tiles
#   mse        MSE on the sigmoid output instead of the focal loss (retrained)
#   blobs      blob decoder instead of the window decoder (evaluation only,
#              on the baseline weights)
#   stitching  512x512 tiles overlap by half, overlaps averaged (evaluation only, on the
#              baseline weights)
EXPERIMENT = "baseline"
CFG = {"loss": "focal", "decoder": "window", "overlap": False, "weights": "baseline"}

RESULT_DIR = f"results/unet2d/{EXPERIMENT}"
# Evaluation-only experiments (blobs, stitching) score the baseline weights.
CHECKPOINT = os.environ.get("FLAGELLA_CKPT", f"results/unet2d/{CFG['weights']}/best.pt")

SEED = 57
VALIDATION_FRAC = 0.2

SIGMA = 3.0
HEATMAP_RADIUS = int(3 * SIGMA)
PATCH_WIDTH = 512
DEPTH = int(SIGMA)

BATCH_SIZE = 32                          # per rank
NUM_WORKERS = 6

UNET_CHANNELS = (16, 32, 64, 128)
UNET_STRIDES = (2, 2, 2)

NMS_RADIUS = int(3 * SIGMA)


def setup_distributed():
    if "RANK" in os.environ:
        dist.init_process_group("nccl")
        rank = int(os.environ["RANK"])
        local_rank = int(os.environ["LOCAL_RANK"])
        world_size = dist.get_world_size()
    else:
        rank, local_rank, world_size = 0, 0, 1
    torch.cuda.set_device(local_rank)
    return rank, local_rank, world_size


def is_distributed():
    return dist.is_available() and dist.is_initialized()


def train_val_split(tomo_ids, validation_frac, seed):
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(tomo_ids))
    n_val = int(len(tomo_ids) * validation_frac)
    val_ids = {tomo_ids[i] for i in perm[:n_val]}
    train_ids = [t for t in tomo_ids if t not in val_ids]
    assert val_ids.isdisjoint(set(train_ids))
    return train_ids, sorted(val_ids)


def visible_motors(motors, zi, depth=DEPTH):
    return [(cz, cy, cx) for cz, cy, cx in motors if abs(zi - cz) <= depth]


def heatmap_2d(motors, zi, y0, y1, x0, x1, sigma=SIGMA, radius=HEATMAP_RADIUS):
    """In-plane Gaussian target for slice zi over the crop [y0:y1, x0:x1].

    Every motor within DEPTH slices contributes a peak of exactly 1.0.
    This differs from the legacy precomputed heatmaps, which attenuated the peak
    by exp(-dz^2 / 2 sigma^2) across a 9-slice window; under a focal loss those
    attenuated slices carry no positive voxel at all, so the model would only
    ever be told to suppress them.
    """
    hm = torch.zeros(y1 - y0, x1 - x0)
    for _, cy, cx in visible_motors(motors, zi):
        ay0, ay1 = max(y0, cy - radius), min(y1, cy + radius + 1)
        ax0, ax1 = max(x0, cx - radius), min(x1, cx + radius + 1)
        if ay0 >= ay1 or ax0 >= ax1:
            continue
        yy, xx = torch.meshgrid(
            torch.arange(ay0, ay1, dtype=torch.float32),
            torch.arange(ax0, ax1, dtype=torch.float32),
            indexing="ij",
        )
        hm[ay0 - y0:ay1 - y0, ax0 - x0:ax1 - x0] += torch.exp(
            -((yy - cy) ** 2 + (xx - cx) ** 2) / (2 * sigma ** 2)
        )
    return hm


@functools.lru_cache(maxsize=8)
def _mmap_tomo(tomo):
    """Memory-map a precomputed (D, H, W) uint8 volume. Cached per worker
    process — persistent_workers keeps the cache alive across batches."""
    return np.load(os.path.join(TEST_NPY_DIR, f"{tomo}.npy"), mmap_mode="r")


def load_slice(tomo, zi, y0, y1, x0, x1):
    crop = np.asarray(_mmap_tomo(tomo)[zi, y0:y1, x0:x1], dtype=np.float32)
    img = torch.from_numpy(crop).unsqueeze(0)
    img = (img - img.mean()) / (img.std() + 1e-8)
    return img


def grid_origins(n, patch, stride):
    """Origins of `patch`-wide tiles every `stride` covering [0, n):
    non-overlapping when stride == patch, overlapping (stitching) when
    smaller. A tile running past the volume edge is zero-padded (see the
    dataset's __getitem__)."""
    origins = [0]
    while origins[-1] + patch < n:
        origins.append(origins[-1] + stride)
    return origins


class TomogramTileDataset(Dataset):
    """Whole-tomogram validation scan: every slice of every tomogram, cut
    into a grid of patch x patch tiles every `stride` pixels, in order
    (tomogram, slice, row, column). No sampling, no class balancing — the model sees each tomogram
    exactly the way it would at inference.

    Each item also returns (tomogram index, zi, y0, x0), so a tile's
    prediction can be put back into tomogram coordinates.
    """

    def __init__(self, tomo_ids, labels, patch, stride):
        self.patch = patch
        self.motors = []
        self.shapes = []
        samples = []
        for ti, tomo in enumerate(tomo_ids):
            rows = labels[labels["tomo_id"] == tomo]
            D = int(rows.iloc[0]["Array shape (axis 0)"])
            H = int(rows.iloc[0]["Array shape (axis 1)"])
            W = int(rows.iloc[0]["Array shape (axis 2)"])
            self.shapes.append((D, H, W))
            self.motors.append([
                (int(r["Motor axis 0"]), int(r["Motor axis 1"]), int(r["Motor axis 2"]))
                for _, r in rows.iterrows() if r["Number of motors"] > 0
            ])
            for zi in range(D):
                for y0 in grid_origins(H, patch, stride):
                    for x0 in grid_origins(W, patch, stride):
                        samples.append((ti, zi, y0, x0))
        self.tomo_ids = list(tomo_ids)
        # One int array instead of ~10^5 tuples: cheap to fork into workers.
        self.samples = np.asarray(samples, dtype=np.int64)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        ti, zi, y0, x0 = (int(v) for v in self.samples[idx])
        p = self.patch
        tomo = self.tomo_ids[ti]
        _, H, W = self.shapes[ti]
        y1, x1 = min(y0 + p, H), min(x0 + p, W)
        # Normalised over the real pixels, then zero-padded (= the crop mean)
        # up to p x p at the volume's edge.
        img = torch.nn.functional.pad(load_slice(tomo, zi, y0, y1, x0, x1),
                                      (0, p - (x1 - x0), 0, p - (y1 - y0)))
        target = heatmap_2d(self.motors[ti], zi, y0, y0 + p, x0, x0 + p).unsqueeze(0)
        return img, target, torch.from_numpy(self.samples[idx])


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
    """The validation threshold of best.pt: the val_threshold of the epoch
    with the best per-motor F-beta in RESULT_DIR/history.csv (the epoch
    train.py saved as best.pt)."""
    history = f"{RESULT_DIR}/history.csv"
    if not os.path.exists(history):
        raise FileNotFoundError(f"{history} not found — run train.py first.")
    rows = pd.read_csv(history)
    return float(rows.loc[rows.val_fbeta.idxmax(), "val_threshold"])


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
    plt.xlabel("Score threshold")
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


@torch.no_grad()


def test_labels(tomo_ids):
    """Rows in train_labels.csv's format (shape, no motors) for the test
    tomograms, so the scan dataset is used unchanged."""
    rows = []
    for tomo in tomo_ids:
        D, H, W = np.load(os.path.join(TEST_NPY_DIR, f"{tomo}.npy"), mmap_mode="r").shape
        rows.append({"tomo_id": tomo, "Array shape (axis 0)": D, "Array shape (axis 1)": H,
                     "Array shape (axis 2)": W, "Number of motors": 0,
                     "Motor axis 0": -1, "Motor axis 1": -1, "Motor axis 2": -1})
    return pd.DataFrame(rows)


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
    rank, local_rank, world_size = setup_distributed()
    device = torch.device("cuda", local_rank)

    ids = test_ids()
    labels = test_labels(ids)
    ds = TomogramTileDataset(ids, labels, PATCH_WIDTH,
                             PATCH_WIDTH // 2 if CFG["overlap"] else PATCH_WIDTH)
    # Whole tomograms per rank, tiles in dataset order (as evaluate.py).
    shard = Subset(ds, np.nonzero(ds.samples[:, 0] % world_size == rank)[0].tolist())
    dl = DataLoader(shard, batch_size=BATCH_SIZE, shuffle=False,
                    num_workers=NUM_WORKERS, pin_memory=True,
                    persistent_workers=NUM_WORKERS > 0)

    if rank == 0:
        print(f"experiment={EXPERIMENT} {CFG}", flush=True)
        print(f"world_size={world_size}  {len(ids)} test tomograms  "
              f"{len(ds)} tiles  ckpt={CHECKPOINT}", flush=True)

    model = UNet(spatial_dims=2, in_channels=1, out_channels=1,
                 channels=UNET_CHANNELS, strides=UNET_STRIDES).to(device)
    model.load_state_dict(torch.load(CHECKPOINT, map_location=device, weights_only=True))
    model.eval()

    volumes = TomogramVolumes(ds.shapes, device, NMS_RADIUS, CFG["decoder"])
    with torch.no_grad():
        for img, _, meta in dl:
            img = img.to(device, non_blocking=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits = model(img)
            volumes.add(torch.sigmoid(logits.float()), meta)
    peaks = volumes.result(len(ids))

    if rank == 0:
        write_predictions(peaks, ids)

    if is_distributed():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
