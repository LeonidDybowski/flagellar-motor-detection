"""3D UNet flagellar motor detector — training on 4 GPUs with DDP.

Launch via train.sbatch:
    torchrun --standalone --nproc_per_node=4 src/unet3d/train.py

Self-contained: everything this script needs is in this file.

Changes vs src/legacy/unet3d/train.py (validation only; training data,
model, loss and schedule are unchanged):
  * Validation scans every validation tomogram in full, as a grid of
    64^3 patches (TomogramPatchDataset), instead of
    a frozen sample of random crops. Edge patches are zero-padded after
    normalisation; the padded part is discarded.
  * The patches' probabilities are assembled into one probability volume per
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
    results/unet3d/<name>/): baseline, mse (MSE loss, retrained), blobs
    (blob decoder) and stitching (tiles overlapping by half, averaged); the
    last two are evaluation-only on the baseline weights.

Changes vs the earlier src/legacy/train_UNet_3D_4GPU.py:
  * Penalty-reduced focal loss instead of MSE. MSE's global optimum on this data
    is "predict zero everywhere" (~1.3e-4 for a 64^3 patch), which is exactly
    where the legacy runs converged. Focal normalises by the positive count.
  * Heatmap targets are generated analytically instead of loaded from the
    930 GB precomputed .pt tree. Verified bit-exact against those files.
  * Volumes are read from the precomputed {DATA_ROOT}/train_npy/ tree
    (see src/precompute_volumes.py) via np.load(mmap_mode="r") instead of
    decoding train/*.jpg per slice on every __getitem__.
  * DistributedDataParallel instead of DataParallel, LR scaled linearly.
  * Positive patches are jittered, not always motor-centred.
  * Validation set is frozen at construction (legacy re-drew negatives on every
    __getitem__, so per-epoch val loss was measured on different data).
  * Per-epoch validation F1; best.pt is selected on it.
"""

import os
import datetime
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

# Experiment mse (fixed for this folder); results go to results/unet3d/mse/.
#   baseline   focal loss, window decoder, non-overlapping tiles
#   mse        MSE on the sigmoid output instead of the focal loss (retrained)
#   blobs      blob decoder instead of the window decoder (evaluation only,
#              on the baseline weights)
#   stitching  64^3 patches overlap by half, overlaps averaged (evaluation only, on the
#              baseline weights)
EXPERIMENT = "mse"
CFG = {"loss": "mse",   "decoder": "window", "overlap": False, "weights": "mse"}

RESULT_DIR = f"results/unet3d/{EXPERIMENT}"
HISTORY_FILE = f"{RESULT_DIR}/history.csv"

SEED = 57
VALIDATION_FRAC = 0.2

SIGMA = 3.0
HEATMAP_RADIUS = int(3 * SIGMA)          # box truncation, matches legacy precompute
PATCH_DEPTH = 64
PATCH_SIZE = 64
NEG2MOTOR_RATIO = 1100                   # training only; val/test see the true random distribution
POS_UPSAMPLING_RATIO = 100               # random-intersecting crops drawn per motor, training only
POS_JITTER = PATCH_SIZE // 4             # +/- voxels the motor is offset from patch centre
HARD_NEG_FRAC = 0.5                      # fraction of negatives drawn near a motor instead of uniformly at random
HARD_NEG_RADIUS = 2 * PATCH_SIZE         # window (voxels) around a motor that "hard" negatives are drawn from

# Global batch across all ranks; per-rank batch is GLOBAL_BATCH_SIZE // world_size.
GLOBAL_BATCH_SIZE = 64
# Reference LR 1e-3 was tuned at global batch 16; scaled linearly to the batch above.
REFERENCE_BATCH_SIZE = 16
REFERENCE_LR = 1e-3
LEARNING_RATE = REFERENCE_LR * GLOBAL_BATCH_SIZE / REFERENCE_BATCH_SIZE
WARMUP_EPOCHS = 2                        # linear warmup, standard with scaled LR
NUM_EPOCHS = 20
NUM_WORKERS = 4                          # cpus-per-task 24 / 4 ranks

UNET_CHANNELS = (16, 32, 64, 128)
UNET_STRIDES = (2, 2, 2)

FOCAL_ALPHA = 2.0
FOCAL_BETA = 4.0

NMS_RADIUS = int(3 * SIGMA)

# Final-conv bias init, so the model starts predicting ~this probability
# everywhere instead of 0.5. A 0 bias is why epoch 1 of the previous run
# opened at train loss 1469.
PRIOR_PROB = 0.01


# ----------------------------------------------------------------------------
# distributed setup
# ----------------------------------------------------------------------------

def setup_distributed():
    if "RANK" in os.environ:
        # Each GPU validates whole tomograms, so GPUs finish validation at
        # different times; the default 10 min collective timeout killed a run
        # while one GPU was still validating.
        dist.init_process_group("nccl", timeout=datetime.timedelta(hours=3))
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


# ----------------------------------------------------------------------------
# data
# ----------------------------------------------------------------------------

def train_val_split(tomo_ids, validation_frac, seed):
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(tomo_ids))
    n_val = int(len(tomo_ids) * validation_frac)
    val_ids = {tomo_ids[i] for i in perm[:n_val]}
    train_ids = [t for t in tomo_ids if t not in val_ids]
    assert val_ids.isdisjoint(set(train_ids))
    return train_ids, sorted(val_ids)


def heatmap_3d(centers, z0, z1, y0, y1, x0, x1, sigma=SIGMA, radius=HEATMAP_RADIUS):
    """Gaussian heatmap for the crop [z0:z1, y0:y1, x0:x1].

    Same closed form and same box truncation as the legacy precompute step, so
    this reproduces the precomputed .pt files exactly (verified bit-exact).
    """
    hm = torch.zeros(z1 - z0, y1 - y0, x1 - x0)
    for cz, cy, cx in centers:
        az0, az1 = max(z0, cz - radius), min(z1, cz + radius + 1)
        ay0, ay1 = max(y0, cy - radius), min(y1, cy + radius + 1)
        ax0, ax1 = max(x0, cx - radius), min(x1, cx + radius + 1)
        if az0 >= az1 or ay0 >= ay1 or ax0 >= ax1:
            continue
        zz, yy, xx = torch.meshgrid(
            torch.arange(az0, az1, dtype=torch.float32),
            torch.arange(ay0, ay1, dtype=torch.float32),
            torch.arange(ax0, ax1, dtype=torch.float32),
            indexing="ij",
        )
        hm[az0 - z0:az1 - z0, ay0 - y0:ay1 - y0, ax0 - x0:ax1 - x0] += torch.exp(
            -((zz - cz) ** 2 + (yy - cy) ** 2 + (xx - cx) ** 2) / (2 * sigma ** 2)
        )
    return hm


@functools.lru_cache(maxsize=8)
def _mmap_tomo(tomo):
    """Memory-map a precomputed (D, H, W) uint8 volume. Cached per worker
    process — persistent_workers keeps the cache alive across batches."""
    return np.load(os.path.join(TRAIN_NPY_DIR, f"{tomo}.npy"), mmap_mode="r")


def load_volume(tomo, z0, z1, y0, y1, x0, x1):
    """Read a (1, D, H, W) crop, normalised once over the whole crop."""
    crop = np.asarray(_mmap_tomo(tomo)[z0:z1, y0:y1, x0:x1], dtype=np.float32)
    vol = torch.from_numpy(crop)
    vol = (vol - vol.mean()) / (vol.std() + 1e-8)
    return vol.unsqueeze(0)


def _random_flip_rotate(volume, target, rng):
    """Random axis flips + 90-degree rotation, training-time augmentation.

    volume and target are (1, D, H, W); dims 1,2,3 are D,H,W. Exact
    transforms (flip/rot90), no interpolation — applied identically to both
    tensors so they stay spatially aligned.
    """
    dims = [1, 2, 3]
    for d in dims:
        if rng.random() < 0.5:
            volume = volume.flip(d)
            target = target.flip(d)

    k = int(rng.integers(0, 4))
    if k:
        volume = torch.rot90(volume, k, dims=(2, 3))
        target = torch.rot90(target, k, dims=(2, 3))

    return volume.contiguous(), target.contiguous()


class Balanced3DDataset(Dataset):
    """Motor-centred positive patches plus random negatives.

    Negative count is neg_ratio * total positive count, drawn as a whole
    across the split — not per tomogram — so a motor-free tomogram doesn't
    inject a negative that has no matching positive. At neg_ratio=1 this
    makes positives and negatives exactly equal in number. Each negative
    slot picks a tomogram uniformly at random from those with at least one
    valid crop origin (independent of whether that tomogram has motors).

    frozen=True precomputes every crop origin with a seeded RNG, so the
    sample is identical on every epoch and across processes — required for
    the validation set to be comparable epoch to epoch.
    """

    def __init__(self, tomo_ids, labels, neg_ratio, patch_depth, patch_size,
                 jitter, frozen, seed=SEED):
        self.pd = patch_depth
        self.ps = patch_size
        self.jitter = jitter
        self.frozen = frozen
        self.samples = []
        rng = np.random.default_rng(seed)

        valid_tomos = []
        for tomo in tomo_ids:
            rows = labels[labels["tomo_id"] == tomo]
            D = int(rows.iloc[0]["Array shape (axis 0)"])
            H = int(rows.iloc[0]["Array shape (axis 1)"])
            W = int(rows.iloc[0]["Array shape (axis 2)"])
            if D < patch_depth or H < patch_size or W < patch_size:
                continue

            motors = [
                (int(r["Motor axis 0"]), int(r["Motor axis 1"]), int(r["Motor axis 2"]))
                for _, r in rows.iterrows() if r["Number of motors"] > 0
            ]
            valid_tomos.append({"tomo": tomo, "shape": (D, H, W), "motors": motors})

            for cz, cy, cx in motors:
                self.samples.append({
                    "tomo": tomo, "shape": (D, H, W), "motors": motors,
                    "center": (cz, cy, cx), "positive": True, "origin": None,
                })

        n_pos = len(self.samples)
        n_neg = round(neg_ratio * n_pos)
        for i in rng.integers(0, len(valid_tomos), size=n_neg):
            t = valid_tomos[i]
            self.samples.append({
                "tomo": t["tomo"], "shape": t["shape"], "motors": t["motors"],
                "center": None, "positive": False, "origin": None,
            })

        if frozen:
            for s in self.samples:
                s["origin"] = self._draw_origin(s, rng)

    def _draw_origin(self, s, rng):
        D, H, W = s["shape"]
        pd, ps = self.pd, self.ps

        if s["positive"]:
            cz, cy, cx = s["center"]
            j = self.jitter
            cz += int(rng.integers(-j, j + 1))
            cy += int(rng.integers(-j, j + 1))
            cx += int(rng.integers(-j, j + 1))
            z0 = min(max(cz - pd // 2, 0), D - pd)
            y0 = min(max(cy - ps // 2, 0), H - ps)
            x0 = min(max(cx - ps // 2, 0), W - ps)
            return (z0, y0, x0)

        # Negative: reject crops that would contain a motor, then give up and
        # take whatever we drew (the target is generated from the same motor
        # list either way, so a leaked motor is still labelled correctly).
        for _ in range(10):
            z0 = int(rng.integers(0, D - pd + 1))
            y0 = int(rng.integers(0, H - ps + 1))
            x0 = int(rng.integers(0, W - ps + 1))
            clear = all(
                not (z0 <= mz < z0 + pd and y0 <= my < y0 + ps and x0 <= mx < x0 + ps)
                for mz, my, mx in s["motors"]
            )
            if clear:
                return (z0, y0, x0)
        return (z0, y0, x0)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        if self.frozen:
            z0, y0, x0 = s["origin"]
        else:
            # Fresh crop every epoch — this is the augmentation for training.
            z0, y0, x0 = self._draw_origin(s, np.random.default_rng())

        pd, ps = self.pd, self.ps
        volume = load_volume(s["tomo"], z0, z0 + pd, y0, y0 + ps, x0, x0 + ps)
        target = heatmap_3d(s["motors"], z0, z0 + pd, y0, y0 + ps, x0, x0 + ps).unsqueeze(0)
        return volume, target


def grid_origins(n, patch, stride):
    """Origins of `patch`-wide tiles every `stride` covering [0, n):
    non-overlapping when stride == patch, overlapping (stitching) when
    smaller. A tile running past the volume edge is zero-padded (see the
    dataset's __getitem__)."""
    origins = [0]
    while origins[-1] + patch < n:
        origins.append(origins[-1] + stride)
    return origins


class TomogramPatchDataset(Dataset):
    """Whole-tomogram validation scan: every tomogram cut into a grid of
    patch_depth x patch_size x patch_size patches every stride_depth /
    stride_size voxels, in order (tomogram, z, y, x). No sampling, no class balancing — the model sees each tomogram
    exactly the way it would at inference.

    Each item also returns (tomogram index, z0, y0, x0), so a patch's
    prediction can be put back into tomogram coordinates.
    """

    def __init__(self, tomo_ids, labels, patch_depth, patch_size, stride_depth, stride_size):
        self.pd = patch_depth
        self.ps = patch_size
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
            for z0 in grid_origins(D, patch_depth, stride_depth):
                for y0 in grid_origins(H, patch_size, stride_size):
                    for x0 in grid_origins(W, patch_size, stride_size):
                        samples.append((ti, z0, y0, x0))
        self.tomo_ids = list(tomo_ids)
        # One int array instead of ~10^5 tuples: cheap to fork into workers.
        self.samples = np.asarray(samples, dtype=np.int64)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        ti, z0, y0, x0 = (int(v) for v in self.samples[idx])
        pd, ps = self.pd, self.ps
        tomo = self.tomo_ids[ti]
        D, H, W = self.shapes[ti]
        z1, y1, x1 = min(z0 + pd, D), min(y0 + ps, H), min(x0 + ps, W)
        # Normalised over the real voxels, then zero-padded (= the crop mean)
        # up to the patch size at the volume's edge.
        volume = torch.nn.functional.pad(
            load_volume(tomo, z0, z1, y0, y1, x0, x1),
            (0, ps - (x1 - x0), 0, ps - (y1 - y0), 0, pd - (z1 - z0)))
        target = heatmap_3d(self.motors[ti], z0, z0 + pd, y0, y0 + ps, x0, x0 + ps).unsqueeze(0)
        return volume, target, torch.from_numpy(self.samples[idx])


class BalancedRandomSubvolumeDataset(Dataset):
    """Uniform random subvolumes, class-balanced by rejection — training only.

    Crops are still drawn uniformly at random within a tomogram (no
    motor-centring, no jitter) — the only thing controlled is how many
    positive vs. negative crops end up in the sample pool. "Positive" means
    the crop happens to intersect a motor's heatmap footprint; "negative"
    means it doesn't. The raw random-crop class ratio is on the order of
    2000:1 (see train_val_split-adjacent analysis); this trades that down to
    a training-friendly ratio without reintroducing motor-centred positives.

    Ratio is applied per tomogram, not globally. A tomogram with n motors
    contributes n * pos_upsampling_ratio positive samples — each an
    independently drawn random crop that intersects that motor, i.e.
    upsampling by redrawing, not by reusing one fixed crop — and
    round(neg2motor_ratio * n) negative samples. Motor-free tomograms are
    not excluded — they contribute round(neg2motor_ratio * avg_motors)
    negatives, where avg_motors is the mean motor count over tomograms that
    have at least one motor, so a 0-motor tomogram gets the same negative
    count an average motor-having tomogram would.

    Not used for validation/test — those should see the true random
    distribution (TomogramPatchDataset), matching real inference.

    HARD_NEG_FRAC of each tomogram's negatives are drawn from a window near
    a random motor (see _draw_origin) rather than uniformly across the whole
    volume, so the model sees background that's structurally close to a
    motor site, not just easy random background.
    """

    def __init__(self, tomo_ids, labels, neg2motor_ratio, pos_upsampling_ratio,
                 patch_depth, patch_size, max_tries=20):
        self.pd = patch_depth
        self.ps = patch_size
        self.max_tries = max_tries
        self.tomos = []

        for tomo in tomo_ids:
            rows = labels[labels["tomo_id"] == tomo]
            D = int(rows.iloc[0]["Array shape (axis 0)"])
            H = int(rows.iloc[0]["Array shape (axis 1)"])
            W = int(rows.iloc[0]["Array shape (axis 2)"])
            if D < patch_depth or H < patch_size or W < patch_size:
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
            # _draw_origin) instead of uniformly at random; only meaningful
            # for tomograms that actually have a motor to be "near".
            n_hard = round(HARD_NEG_FRAC * n_neg) if t["motors"] else 0
            for i in range(n_neg):
                self.samples.append({
                    "tomo": t["tomo"], "shape": t["shape"], "motors": t["motors"],
                    "positive": False, "hard": i < n_hard,
                })

    def _crop_is_positive(self, origin, motors):
        z0, y0, x0 = origin
        pd, ps = self.pd, self.ps
        return any(
            z0 <= mz < z0 + pd and y0 <= my < y0 + ps and x0 <= mx < x0 + ps
            for mz, my, mx in motors
        )

    def _draw_origin(self, s, rng):
        D, H, W = s["shape"]
        pd, ps = self.pd, self.ps

        def draw():
            return (int(rng.integers(0, D - pd + 1)),
                    int(rng.integers(0, H - ps + 1)),
                    int(rng.integers(0, W - ps + 1)))

        if s["positive"]:
            # Draw uniformly from the window of origins guaranteed to contain
            # a randomly chosen motor — direct draw, not rejection sampling,
            # so it always lands the motor in one try regardless of
            # tomogram size. The motor's position within the crop is
            # uniform over the whole patch, not centred.
            mz, my, mx = s["motors"][int(rng.integers(0, len(s["motors"])))]
            z0 = int(rng.integers(max(0, mz - pd + 1), min(mz, D - pd) + 1))
            y0 = int(rng.integers(max(0, my - ps + 1), min(my, H - ps) + 1))
            x0 = int(rng.integers(max(0, mx - ps + 1), min(mx, W - ps) + 1))
            return (z0, y0, x0)

        def draw_hard():
            # Window around a random motor instead of the whole volume, so
            # the crop's surroundings resemble a real motor site without
            # containing one. Clamped into the valid origin range per axis,
            # then swapped if the clamp inverted the bounds (small volumes /
            # motors near an edge).
            mz, my, mx = s["motors"][int(rng.integers(0, len(s["motors"])))]
            r = HARD_NEG_RADIUS

            def bounded(c, limit):
                lo = max(0, min(c - r, limit))
                hi = max(0, min(c + r, limit))
                return (lo, hi) if lo <= hi else (hi, lo)

            z0 = int(rng.integers(*bounded(mz, D - pd), endpoint=True))
            y0 = int(rng.integers(*bounded(my, H - ps), endpoint=True))
            x0 = int(rng.integers(*bounded(mx, W - ps), endpoint=True))
            return (z0, y0, x0)

        # Negative: rejection-sample as before. "hard" negatives draw from
        # draw_hard() instead of the uniform draw().
        draw_fn = draw_hard if s.get("hard") and s["motors"] else draw
        origin = draw_fn()
        for _ in range(self.max_tries):
            if not self._crop_is_positive(origin, s["motors"]):
                return origin
            origin = draw_fn()
        # Give up after max_tries — keep whatever was last drawn (matches
        # Balanced3DDataset's give-up behaviour).
        return origin

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        # Step 1: fresh random crop origin every epoch (see _draw_origin —
        # positive slots are rejection-sampled to intersect the motor, but the
        # motor's position within the crop is otherwise uniform random).
        rng = np.random.default_rng()
        z0, y0, x0 = self._draw_origin(s, rng)

        pd, ps = self.pd, self.ps
        volume = load_volume(s["tomo"], z0, z0 + pd, y0, y0 + ps, x0, x0 + ps)
        target = heatmap_3d(s["motors"], z0, z0 + pd, y0, y0 + ps, x0, x0 + ps).unsqueeze(0)

        # Step 2: random flips + 90-degree rotation, applied identically to
        # volume and target so they stay spatially aligned. Exact (no
        # interpolation) — cheap tensor flip/transpose ops.
        volume, target = _random_flip_rotate(volume, target, rng)
        return volume, target


# ----------------------------------------------------------------------------
# loss
# ----------------------------------------------------------------------------

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
    """CenterNet penalty-reduced focal loss, normalised by the positive count.

    Normalising by the number of peak voxels — not by the number of voxels — is
    what stops the all-zeros solution from being optimal.
    """
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



# ----------------------------------------------------------------------------
# train / validate
# ----------------------------------------------------------------------------

def init_final_bias(model, prior_prob=PRIOR_PROB):
    """Bias the output conv so the model opens at p ~= prior_prob everywhere.

    MONAI leaves it at 0, i.e. p = 0.5 on all 262144 voxels of every patch,
    which under a loss normalised by positive count is what opened the
    previous run at train loss 1469. CenterNet's -log((1-p)/p) init is the
    standard fix. Returns the module it touched, or None.
    """
    # MONAI's output layer is a ConvTranspose3d, not a Conv3d — match both.
    last = None
    for m in model.modules():
        if isinstance(m, (torch.nn.Conv3d, torch.nn.ConvTranspose3d)) and m.out_channels == 1:
            last = m
    if last is not None and last.bias is not None:
        torch.nn.init.constant_(last.bias, -math.log((1 - prior_prob) / prior_prob))
    return last


def lr_at(epoch):
    if epoch < WARMUP_EPOCHS:
        return LEARNING_RATE * (epoch + 1) / WARMUP_EPOCHS
    progress = (epoch - WARMUP_EPOCHS) / max(1, NUM_EPOCHS - WARMUP_EPOCHS)
    return LEARNING_RATE * 0.5 * (1 + math.cos(math.pi * progress))


def train_epoch(model, loader, optimizer, device, epoch):
    model.train()
    for g in optimizer.param_groups:
        g["lr"] = lr_at(epoch)

    total = torch.zeros((), device=device)
    n = 0
    for vol, target in loader:
        vol = vol.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits = model(vol)
        loss = compute_loss(logits, target)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        total += loss.detach()
        n += 1
    return (total / max(n, 1)).item()


@torch.no_grad()
def validate(core, loader, device, truth, tomo_ids, shapes):
    """Scan every validation tomogram patch by patch, assemble each
    tomogram's probability volume, decode positions from it and score them
    per motor with F-beta, swept over SCORE_THRESHOLDS. Returns (val_loss,
    sweep rows, per-tomogram peaks); the loss is the mean experiment loss
    over all patches.

    Scores the *unwrapped* module in fp32: training runs
    torch.compile(DDP(core)), and scoring through that wrapper in eval mode
    produced a near-constant output field in an earlier run — the metric was
    barely reading the weights. Dropping autocast also avoids the bf16
    quantisation that turns a smooth heatmap into tie-riddled steps, which
    decode_peaks handles worst.
    """
    core.eval()
    total, n = 0.0, 0
    volumes = TomogramVolumes(shapes, device, NMS_RADIUS, CFG["decoder"])
    for vol, target, meta in loader:
        vol = vol.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)
        logits = core(vol)
        total += compute_loss(logits, target).item()
        n += 1
        volumes.add(torch.sigmoid(logits.float()), meta)

    loss_sum, count = all_reduce_sum([total, n], device)
    peaks = volumes.result(len(tomo_ids))
    rows = fbeta_sweep(truth, tomo_ids, peaks)
    core.train()
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

    train_ds = BalancedRandomSubvolumeDataset(train_ids, labels, NEG2MOTOR_RATIO,
                                              POS_UPSAMPLING_RATIO, PATCH_DEPTH, PATCH_SIZE)
    stride = 2 if CFG["overlap"] else 1
    val_ds = TomogramPatchDataset(val_ids, labels, PATCH_DEPTH, PATCH_SIZE,
                                  PATCH_DEPTH // stride, PATCH_SIZE // stride)
    if rank == 0:
        print(f"{len(train_ids)} train / {len(val_ids)} val tomograms", flush=True)
        print(f"{len(train_ds)} train / {len(val_ds)} val patches", flush=True)

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
    # Contiguous strided shard so every val patch is seen exactly once, with no
    # padding duplicates (which DistributedSampler would introduce).
    # Whole tomograms per rank (each rank assembles and decodes its own),
    # tiles kept in dataset order so a tomogram's tiles arrive together.
    val_shard = Subset(val_ds, np.nonzero(val_ds.samples[:, 0] % world_size == rank)[0].tolist())
    val_dl = DataLoader(
        val_shard, batch_size=per_rank_batch, shuffle=False, num_workers=NUM_WORKERS,
        pin_memory=True, persistent_workers=NUM_WORKERS > 0,
    )

    core = UNet(spatial_dims=3, in_channels=1, out_channels=1,
                channels=UNET_CHANNELS, strides=UNET_STRIDES).to(device)
    biased = init_final_bias(core)
    if rank == 0:
        print(f"final-conv bias init: {'ok' if biased is not None else 'NOT FOUND'} "
              f"(prior_prob={PRIOR_PROB})", flush=True)
    model = DDP(core, device_ids=[local_rank]) if is_distributed() else core
    model = torch.compile(model)
    optimizer = torch.optim.Adam(core.parameters(), lr=LEARNING_RATE)

    history = []
    best_fbeta = -1.0
    for epoch in range(NUM_EPOCHS):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        train_loss = train_epoch(model, train_dl, optimizer, device, epoch)
        val_loss, rows, _ = validate(core, val_dl, device, val_truth, val_ids, val_ds.shapes)
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
            # Clean state dict: saved from the unwrapped module, so it loads into
            # a bare UNet without DDP/compile key prefixes.
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
