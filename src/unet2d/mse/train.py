"""2D UNet flagellar motor detector — training on 4 GPUs with DDP.

Launch via train.sbatch:
    torchrun --standalone --nproc_per_node=4 src/unet2d/train.py

Self-contained: everything this script needs is in this file.

Changes vs src/legacy/unet2d/train.py (validation only; training data,
model, loss and schedule are unchanged):
  * Validation scans every validation tomogram in full, every slice, cut
    into a grid of 512x512 tiles (TomogramTileDataset), instead of
    a frozen sample of random crops. Edge tiles are zero-padded after
    normalisation; the padded part is discarded.
  * The tiles' probabilities are assembled into one probability volume per
    tomogram (GPU memory; each GPU handles whole tomograms) and positions
    are decoded from it — window decoder: maximum of the
    (2 * NMS_RADIUS + 1)^3 neighbourhood and at least the threshold; blob
    decoder: maximum of each connected region above the threshold. Exact
    ties keep the most central voxel.
  * Scored per motor on positions only: positions and motors are paired
    one-to-one by the Hungarian algorithm on their distance in angstrom
    (voxel distance times voxel spacing), pairs beyond 1000 A not allowed.
    Paired motors are TPs, unpaired motors FNs, unpaired positions FPs.
    F-beta (beta=2) from sklearn, swept over SCORE_THRESHOLDS. Ground truth
    is the label table's coordinates.
  * best.pt is selected on the best F-beta.
  * Experiments (EXPERIMENTS, chosen with FLAGELLA_EXPERIMENT; results in
    results/unet2d/<name>/): baseline, mse (MSE loss, retrained), blobs
    (blob decoder) and stitching (tiles overlapping by half, averaged); the
    last two are evaluation-only on the baseline weights.

Changes vs the earlier src/legacy/train_UNet_2D_4GPU.py:
  * Positive crops are centred on a motor (with jitter). The legacy script took
    a uniform random 512x512 crop from a ~959x928 slice, so roughly 70% of its
    "positive" samples contained no motor at all and silently became negatives.
  * Penalty-reduced focal loss instead of MSE. Only ~361 of 890k pixels on a
    positive slice are non-zero (0.04%), so MSE's optimum is "predict zero
    everywhere" — which is exactly where the legacy runs landed (0 peaks
    detected on all 6717 validation slices).
  * The target is an in-plane Gaussian with peak 1.0 for every motor within
    DEPTH slices. Legacy used the z-attenuated 3D Gaussian over a 9-slice
    window, so a slice 9 away had a peak of 0.011 and no voxel the focal loss
    could treat as positive. See heatmap_2d below.
  * Targets are generated analytically rather than read from the 930 GB
    precomputed .pt tree.
  * DistributedDataParallel instead of DataParallel, LR scaled linearly.
  * Validation set frozen at construction; per-epoch val F1; best.pt.
  * Slices are read from the precomputed {DATA_ROOT}/train_npy/ tree
    (see src/precompute_volumes.py) via np.load(mmap_mode="r") instead of
    decoding train/*.jpg per slice on every __getitem__.
"""

import os
import math
import functools
import numpy as np
import pandas as pd
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
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
TRAIN_NPY_DIR = f"{DATA_ROOT}/train_npy/"
LABELS = f"{DATA_ROOT}/train_labels.csv"

# Experiment mse (fixed for this folder); results go to results/unet2d/mse/.
#   baseline   focal loss, window decoder, non-overlapping tiles
#   mse        MSE on the sigmoid output instead of the focal loss (retrained)
#   blobs      blob decoder instead of the window decoder (evaluation only,
#              on the baseline weights)
#   stitching  512x512 tiles overlap by half, overlaps averaged (evaluation only, on the
#              baseline weights)
EXPERIMENT = "mse"
CFG = {"loss": "mse",   "decoder": "window", "overlap": False, "weights": "mse"}

RESULT_DIR = f"results/unet2d/{EXPERIMENT}"
HISTORY_FILE = f"{RESULT_DIR}/history.csv"

SEED = 57
VALIDATION_FRAC = 0.2

SIGMA = 3.0
HEATMAP_RADIUS = int(3 * SIGMA)
PATCH_WIDTH = 512
DEPTH = int(SIGMA)                # a motor is "visible" within +/- 3 slices
NEG_RATIO = 2
NEG2MOTOR_RATIO = 1100              # training only; val/test see the true random distribution
POS_UPSAMPLING_RATIO = 100         # random-intersecting crops drawn per motor, training only
POS_JITTER = PATCH_WIDTH // 4
HARD_NEG_FRAC = 0.5                # fraction of negatives drawn near a motor instead of uniformly at random
HARD_NEG_RADIUS = 2 * PATCH_WIDTH  # window (voxels/slices) around a motor that "hard" negatives are drawn from

GLOBAL_BATCH_SIZE = 128
REFERENCE_BATCH_SIZE = 64
REFERENCE_LR = 1e-3
LEARNING_RATE = REFERENCE_LR * GLOBAL_BATCH_SIZE / REFERENCE_BATCH_SIZE
WARMUP_EPOCHS = 2
NUM_EPOCHS = 40
NUM_WORKERS = 6

UNET_CHANNELS = (16, 32, 64, 128)
UNET_STRIDES = (2, 2, 2)

FOCAL_ALPHA = 2.0
FOCAL_BETA = 4.0

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


def all_reduce_sum(values, device):
    t = torch.tensor(values, dtype=torch.float64, device=device)
    if is_distributed():
        dist.all_reduce(t, op=dist.ReduceOp.SUM)
    return t.tolist()


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
    return np.load(os.path.join(TRAIN_NPY_DIR, f"{tomo}.npy"), mmap_mode="r")


def load_slice(tomo, zi, y0, y1, x0, x1):
    crop = np.asarray(_mmap_tomo(tomo)[zi, y0:y1, x0:x1], dtype=np.float32)
    img = torch.from_numpy(crop).unsqueeze(0)
    img = (img - img.mean()) / (img.std() + 1e-8)
    return img


class BalancedSliceDataset(Dataset):
    """Motor-centred positive crops plus random negative crops.

    One positive sample per (motor, slice) pair within DEPTH. Negative
    count is neg_ratio * total positive count, drawn from a single pool of
    (tomogram, motor-free slice) pairs across the whole split — not per
    tomogram — so a tomogram with no motors doesn't inject a negative that
    has no matching positive. At neg_ratio=1 this makes positives and
    negatives exactly equal in number.
    """

    def __init__(self, tomo_ids, labels, neg_ratio, patch, jitter, frozen, seed=SEED):
        self.patch = patch
        self.jitter = jitter
        self.frozen = frozen
        self.samples = []
        rng = np.random.default_rng(seed)

        clear_pool = []  # (tomo, zi, shape, motors) across the whole split
        for tomo in tomo_ids:
            rows = labels[labels["tomo_id"] == tomo]
            D = int(rows.iloc[0]["Array shape (axis 0)"])
            H = int(rows.iloc[0]["Array shape (axis 1)"])
            W = int(rows.iloc[0]["Array shape (axis 2)"])
            if H < patch or W < patch:
                continue

            motors = [
                (int(r["Motor axis 0"]), int(r["Motor axis 1"]), int(r["Motor axis 2"]))
                for _, r in rows.iterrows() if r["Number of motors"] > 0
            ]

            for cz, cy, cx in motors:
                for zi in range(max(cz - DEPTH, 0), min(cz + DEPTH + 1, D)):
                    self.samples.append({
                        "tomo": tomo, "z": zi, "shape": (H, W), "motors": motors,
                        "center": (cy, cx), "positive": True, "origin": None,
                    })

            for zi in range(D):
                if not visible_motors(motors, zi):
                    clear_pool.append((tomo, zi, (H, W), motors))

        n_pos = len(self.samples)
        n_neg = min(len(clear_pool), round(neg_ratio * n_pos))
        for i in rng.choice(len(clear_pool), size=n_neg, replace=False) if clear_pool else []:
            tomo, zi, shape, motors = clear_pool[i]
            self.samples.append({
                "tomo": tomo, "z": zi, "shape": shape, "motors": motors,
                "center": None, "positive": False, "origin": None,
            })

        if frozen:
            for s in self.samples:
                s["origin"] = self._draw_origin(s, rng)

    def _draw_origin(self, s, rng):
        H, W = s["shape"]
        p = self.patch
        if s["positive"]:
            cy, cx = s["center"]
            j = self.jitter
            cy += int(rng.integers(-j, j + 1))
            cx += int(rng.integers(-j, j + 1))
            return (min(max(cy - p // 2, 0), H - p), min(max(cx - p // 2, 0), W - p))
        return (int(rng.integers(0, H - p + 1)), int(rng.integers(0, W - p + 1)))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        if self.frozen:
            y0, x0 = s["origin"]
        else:
            y0, x0 = self._draw_origin(s, np.random.default_rng())
        p = self.patch
        img = load_slice(s["tomo"], s["z"], y0, y0 + p, x0, x0 + p)
        target = heatmap_2d(s["motors"], s["z"], y0, y0 + p, x0, x0 + p).unsqueeze(0)
        return img, target


def _random_flip_rotate_2d(img, target, rng):
    """Random H/W flips + 90-degree rotation, training-time augmentation.

    img and target are (1, H, W); dims 1,2 are H,W. No Z-flip — a slice is a
    single z-index, not a volume, so flipping it doesn't apply here. Exact
    transforms (flip/rot90), no interpolation — applied identically to both
    tensors so they stay spatially aligned.
    """
    for d in (1, 2):
        if rng.random() < 0.5:
            img = img.flip(d)
            target = target.flip(d)

    k = int(rng.integers(0, 4))
    if k:
        img = torch.rot90(img, k, dims=(1, 2))
        target = torch.rot90(target, k, dims=(1, 2))

    return img.contiguous(), target.contiguous()


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


class BalancedRandomSliceDataset(Dataset):
    """Uniform random (slice, crop) pairs, class-balanced by rejection —
    training only.

    Crops are still drawn uniformly at random within a tomogram (no
    motor-centring, no jitter) — the only thing controlled is how many
    positive vs. negative (slice, crop) pairs end up in the sample pool.
    "Positive" means the crop, on that slice, has a motor visible within
    DEPTH (see visible_motors/heatmap_2d); "negative" means it doesn't.

    Ratio is applied per tomogram, not globally. A tomogram with n motors
    contributes n * pos_upsampling_ratio positive samples — each an
    independently drawn random (slice, crop) that has a motor visible on it,
    i.e. upsampling by redrawing, not by reusing one fixed crop — and
    round(neg2motor_ratio * n) negative samples. Motor-free tomograms are
    not excluded — they contribute round(neg2motor_ratio * avg_motors)
    negatives, where avg_motors is the mean motor count over tomograms that
    have at least one motor, so a 0-motor tomogram gets the same negative
    count an average motor-having tomogram would.

    Not used for validation/test — those should see the true random
    distribution (TomogramTileDataset), matching real inference.

    HARD_NEG_FRAC of each tomogram's negatives are drawn from a window near
    a random motor (see _draw) rather than uniformly across the whole
    volume, so the model sees background that's structurally close to a
    motor site, not just easy random background.
    """

    def __init__(self, tomo_ids, labels, neg2motor_ratio, pos_upsampling_ratio,
                 patch, max_tries=20):
        self.patch = patch
        self.max_tries = max_tries
        self.tomos = []

        for tomo in tomo_ids:
            rows = labels[labels["tomo_id"] == tomo]
            D = int(rows.iloc[0]["Array shape (axis 0)"])
            H = int(rows.iloc[0]["Array shape (axis 1)"])
            W = int(rows.iloc[0]["Array shape (axis 2)"])
            if H < patch or W < patch:
                continue

            motors = [
                (int(r["Motor axis 0"]), int(r["Motor axis 1"]), int(r["Motor axis 2"]))
                for _, r in rows.iterrows() if r["Number of motors"] > 0
            ]
            self.tomos.append({"tomo": tomo, "shape": (D, H, W), "motors": motors})

        motor_counts = [len(t["motors"]) for t in self.tomos if t["motors"]]
        avg_motors = sum(motor_counts) / len(motor_counts) if motor_counts else 0.0

        self.samples = []
        for t in self.tomos:
            for cz, cy, cx in t["motors"]:
                for _ in range(pos_upsampling_ratio):
                    self.samples.append({
                        "tomo": t["tomo"], "shape": t["shape"], "motors": t["motors"],
                        "positive": True,
                    })

            n_motors = len(t["motors"]) if t["motors"] else avg_motors
            n_neg = round(neg2motor_ratio * n_motors)
            # HARD_NEG_FRAC of negatives are drawn near a motor (see
            # _draw) instead of uniformly at random; only meaningful for
            # tomograms that actually have a motor to be "near".
            n_hard = round(HARD_NEG_FRAC * n_neg) if t["motors"] else 0
            for i in range(n_neg):
                self.samples.append({
                    "tomo": t["tomo"], "shape": t["shape"], "motors": t["motors"],
                    "positive": False, "hard": i < n_hard,
                })

    def _crop_is_positive(self, zi, origin, motors):
        y0, x0 = origin
        p = self.patch
        return len(visible_motors(motors, zi)) > 0 and any(
            y0 <= my < y0 + p and x0 <= mx < x0 + p
            for _, my, mx in visible_motors(motors, zi)
        )

    def _draw(self, s, rng):
        D, H, W = s["shape"]
        p = self.patch

        def draw():
            return (int(rng.integers(0, D)),
                    int(rng.integers(0, H - p + 1)),
                    int(rng.integers(0, W - p + 1)))

        if s["positive"]:
            # Draw uniformly from the window guaranteed to contain a
            # randomly chosen motor — direct draw, not rejection sampling,
            # so it always lands the motor in one try regardless of
            # tomogram size. zi is drawn from that motor's visible slices;
            # (y0, x0) is uniform over the whole patch, not centred.
            cz, cy, cx = s["motors"][int(rng.integers(0, len(s["motors"])))]
            zi = int(rng.integers(max(0, cz - DEPTH), min(cz + DEPTH, D - 1) + 1))
            y0 = int(rng.integers(max(0, cy - p + 1), min(cy, H - p) + 1))
            x0 = int(rng.integers(max(0, cx - p + 1), min(cx, W - p) + 1))
            return zi, (y0, x0)

        def draw_hard():
            # Window around a random motor instead of the whole volume, so
            # the crop's surroundings resemble a real motor site without
            # showing one. Clamped into the valid draw range per axis, then
            # swapped if the clamp inverted the bounds (small volumes /
            # motors near an edge).
            cz, cy, cx = s["motors"][int(rng.integers(0, len(s["motors"])))]
            r = HARD_NEG_RADIUS

            def bounded(c, limit):
                lo = max(0, min(c - r, limit))
                hi = max(0, min(c + r, limit))
                return (lo, hi) if lo <= hi else (hi, lo)

            zi = int(rng.integers(*bounded(cz, D - 1), endpoint=True))
            y0 = int(rng.integers(*bounded(cy, H - p), endpoint=True))
            x0 = int(rng.integers(*bounded(cx, W - p), endpoint=True))
            return zi, y0, x0

        # Negative: rejection-sample as before. "hard" negatives draw from
        # draw_hard() instead of the uniform draw().
        draw_fn = draw_hard if s.get("hard") and s["motors"] else draw
        zi, y0, x0 = draw_fn()
        for _ in range(self.max_tries):
            if not self._crop_is_positive(zi, (y0, x0), s["motors"]):
                return zi, (y0, x0)
            zi, y0, x0 = draw_fn()
        # Give up after max_tries — keep whatever was last drawn.
        return zi, (y0, x0)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        # Step 1: fresh random (slice, crop) every epoch.
        rng = np.random.default_rng()
        zi, (y0, x0) = self._draw(s, rng)

        p = self.patch
        img = load_slice(s["tomo"], zi, y0, y0 + p, x0, x0 + p)
        target = heatmap_2d(s["motors"], zi, y0, y0 + p, x0, x0 + p).unsqueeze(0)

        # Step 2: random flips + 90-degree rotation, applied identically to
        # img and target so they stay spatially aligned.
        img, target = _random_flip_rotate_2d(img, target, rng)
        return img, target


def mse_loss(logits, target):
    """Mean squared error between the sigmoid output and the heatmap target
    (the "mse" experiment). On the sigmoid, so decoding thresholds mean the
    same as with the focal loss."""
    return torch.mean((torch.sigmoid(logits.float()) - target.float().clamp(max=1.0)) ** 2)


def compute_loss(logits, target):
    """The experiment's training/validation loss (CFG["loss"])."""
    if CFG["loss"] == "mse":
        return mse_loss(logits, target)
    return focal_loss(logits, target)


def focal_loss(logits, target, alpha=FOCAL_ALPHA, beta=FOCAL_BETA):
    """CenterNet penalty-reduced focal loss, normalised by the positive count."""
    p = torch.sigmoid(logits.float()).clamp(1e-4, 1 - 1e-4)
    t = target.float().clamp(max=1.0)
    pos = t.ge(1.0 - 1e-6).float()
    neg = 1.0 - pos

    pos_loss = -((1 - p) ** alpha) * torch.log(p) * pos
    neg_loss = -((1 - t) ** beta) * (p ** alpha) * torch.log(1 - p) * neg
    n_pos = pos.sum().clamp(min=1.0)
    return (pos_loss.sum() + neg_loss.sum()) / n_pos


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
SCORE_THRESHOLDS = [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
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


def lr_at(epoch):
    if epoch < WARMUP_EPOCHS:
        return LEARNING_RATE * (epoch + 1) / WARMUP_EPOCHS
    progress = (epoch - WARMUP_EPOCHS) / max(1, NUM_EPOCHS - WARMUP_EPOCHS)
    return LEARNING_RATE * 0.5 * (1 + math.cos(math.pi * progress))


def train_epoch(model, loader, optimizer, device, epoch):
    model.train()
    for g in optimizer.param_groups:
        g["lr"] = lr_at(epoch)

    total, n = 0.0, 0
    for img, target in loader:
        img = img.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits = model(img)
        loss = compute_loss(logits, target)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        total += loss.item()
        n += 1
    return total / max(n, 1)


@torch.no_grad()
def validate(model, loader, device, truth, tomo_ids, shapes):
    """Scan every validation tomogram tile by tile, slice by slice, assemble
    each tomogram's probability volume, decode positions from it and score
    them per motor with F-beta, swept over SCORE_THRESHOLDS. Returns
    (val_loss, sweep rows, per-tomogram peaks); the loss is the mean
    experiment loss over all tiles."""
    model.eval()
    total, n = 0.0, 0
    volumes = TomogramVolumes(shapes, device, NMS_RADIUS, CFG["decoder"])
    for img, target, meta in loader:
        img = img.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits = model(img)
        total += compute_loss(logits, target).item()
        n += 1
        volumes.add(torch.sigmoid(logits.float()), meta)

    loss_sum, count = all_reduce_sum([total, n], device)
    peaks = volumes.result(len(tomo_ids))
    rows = fbeta_sweep(truth, tomo_ids, peaks)
    return loss_sum / max(count, 1), rows, peaks


def main():
    if CFG["weights"] != EXPERIMENT:
        raise SystemExit(f"experiment {EXPERIMENT!r} is evaluation-only (it reuses the "
                         f"{CFG['weights']!r} weights): run evaluate.py instead")
    rank, local_rank, world_size = setup_distributed()
    device = torch.device("cuda", local_rank)
    torch.manual_seed(SEED + rank)

    if rank == 0:
        os.makedirs(RESULT_DIR, exist_ok=True)
        print(f"experiment={EXPERIMENT} {CFG}", flush=True)
        print(f"world_size={world_size}  global_batch={GLOBAL_BATCH_SIZE}  lr={LEARNING_RATE:.2e}",
              flush=True)

    labels = pd.read_csv(LABELS)
    tomo_ids = sorted(labels.tomo_id.unique())
    train_ids, val_ids = train_val_split(tomo_ids, VALIDATION_FRAC, SEED)
    val_truth = tomo_truth(labels, val_ids)

    train_ds = BalancedRandomSliceDataset(train_ids, labels, NEG2MOTOR_RATIO,
                                          POS_UPSAMPLING_RATIO, PATCH_WIDTH)
    val_ds = TomogramTileDataset(val_ids, labels, PATCH_WIDTH,
                                 PATCH_WIDTH // 2 if CFG["overlap"] else PATCH_WIDTH)
    if rank == 0:
        print(f"{len(train_ids)} train / {len(val_ids)} val tomograms", flush=True)
        print(f"{len(train_ds)} train slices / {len(val_ds)} val tiles", flush=True)

    per_rank_batch = max(1, GLOBAL_BATCH_SIZE // world_size)
    train_sampler = (
        torch.utils.data.distributed.DistributedSampler(train_ds, shuffle=True, drop_last=True)
        if is_distributed() else None
    )
    train_dl = DataLoader(
        train_ds, batch_size=per_rank_batch, sampler=train_sampler,
        shuffle=(train_sampler is None), num_workers=NUM_WORKERS,
        pin_memory=True, persistent_workers=NUM_WORKERS > 0, drop_last=True,
    )
    # Whole tomograms per rank (each rank assembles and decodes its own),
    # tiles kept in dataset order so a tomogram's tiles arrive together.
    val_shard = Subset(val_ds, np.nonzero(val_ds.samples[:, 0] % world_size == rank)[0].tolist())
    val_dl = DataLoader(
        val_shard, batch_size=per_rank_batch, shuffle=False, num_workers=NUM_WORKERS,
        pin_memory=True, persistent_workers=NUM_WORKERS > 0,
    )

    core = UNet(spatial_dims=2, in_channels=1, out_channels=1,
                channels=UNET_CHANNELS, strides=UNET_STRIDES).to(device)
    model = DDP(core, device_ids=[local_rank]) if is_distributed() else core
    model = torch.compile(model)
    optimizer = torch.optim.Adam(core.parameters(), lr=LEARNING_RATE)

    history = []
    best_fbeta = -1.0
    for epoch in range(NUM_EPOCHS):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        train_loss = train_epoch(model, train_dl, optimizer, device, epoch)
        val_loss, rows, _ = validate(model, val_dl, device, val_truth, val_ids, val_ds.shapes)
        top = best_row(rows)

        if rank == 0:
            print(f"epoch {epoch+1:3d}/{NUM_EPOCHS}  lr {lr_at(epoch):.2e}  "
                  f"train {train_loss:.4f}  val {val_loss:.4f}  "
                  f"P {top['precision']:.3f}  R {top['recall']:.3f}  "
                  f"F{BETA} {top['fbeta']:.3f} @ thr {top['threshold']:.2f}", flush=True)
            history.append({
                "epoch": epoch + 1, "lr": lr_at(epoch),
                "train_loss": train_loss, "val_loss": val_loss,
                "val_fbeta": top["fbeta"], "val_precision": top["precision"],
                "val_recall": top["recall"], "val_threshold": top["threshold"],
                "val_tp": top["tp"], "val_fp": top["fp"], "val_fn": top["fn"],
            })
            pd.DataFrame(history).to_csv(HISTORY_FILE, index=False)
            torch.save(core.state_dict(), f"{RESULT_DIR}/last.pt")
            if top["fbeta"] > best_fbeta:
                best_fbeta = top["fbeta"]
                torch.save(core.state_dict(), f"{RESULT_DIR}/best.pt")
                print(f"  new best F{BETA} {best_fbeta:.3f} -> best.pt", flush=True)

    if rank == 0:
        print(f"done. best val F{BETA} {best_fbeta:.3f}", flush=True)
    if is_distributed():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
