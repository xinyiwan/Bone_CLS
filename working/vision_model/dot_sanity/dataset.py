"""
Data loading for the synthetic-dot sanity experiment.

The experiment
--------------
We inject a small bright sphere ("dot") of intensity ``dot_frac * volume.max()``
near the centre of half the volumes, label those 1 and the rest 0, and ask the
ViT to learn the label. A correctly wired model + loader should reach ~perfect
train/val performance on this trivial task; failure to do so means something in
the pipeline (patchify, masking, normalisation, loss, optimisation) is broken.

Two dataset modes
------------------
* ``SyntheticDotDataset`` -- generates smooth random background volumes on the
  fly. Needs no files, so the whole sanity check runs anywhere. This is the
  primary mode.
* ``NiftiDotDataset``      -- loads real ``.nii.gz`` volumes (optionally cropped
  to a segmentation), runs the same preprocessing used downstream, and injects
  the dot *after* preprocessing. Use this once you want to confirm the model can
  pick up a known signal inside real anatomy.

Conventions follow the GIST pipeline: arrays are ``[Z, Y, X]`` and the dot is
injected after intensity normalisation, so ``dot_frac=0.95`` means "95% of the
post-preprocessing max intensity".
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

try:                                    # optional; only needed for NiftiDotDataset
    import nibabel as nib
except Exception:                       # pragma: no cover
    nib = None

from scipy.ndimage import rotate as nd_rotate


# ---------------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------------

def normalize_volume(vol: np.ndarray, mode: str = "minmax",
                     mask: Optional[np.ndarray] = None) -> np.ndarray:
    """Intensity-normalise a volume.

    * ``minmax``  -> [0, 1] over (optionally masked) voxels.
    * ``zscore``  -> zero mean / unit std over (optionally masked) voxels.

    For MRI there is no fixed intensity scale, so per-volume normalisation is
    the sensible default (unlike CT where HU clipping is used).
    """
    vol = vol.astype(np.float32)
    ref = vol[mask > 0] if mask is not None and mask.any() else vol
    if mode == "minmax":
        lo, hi = float(ref.min()), float(ref.max())
        if hi - lo < 1e-8:
            return np.zeros_like(vol)
        return np.clip((vol - lo) / (hi - lo), 0.0, 1.0)
    if mode == "zscore":
        mu, sd = float(ref.mean()), float(ref.std())
        if sd < 1e-8:
            return np.zeros_like(vol)
        return (vol - mu) / sd
    raise ValueError(f"unknown normalize mode: {mode}")


def resize_to_shape(vol: np.ndarray, target: Tuple[int, int, int],
                    pad_val: float = 0.0) -> np.ndarray:
    """Center-crop and/or symmetric-pad a [Z, Y, X] volume to ``target`` shape.

    Mirrors ``OpPadOrCropToFixedDivisibleShape`` from the GIST pipeline (minus
    the divisibility rounding, which the caller controls via ``target``).
    """
    out = vol
    for dim in range(3):
        cur, tgt = out.shape[dim], target[dim]
        if cur < tgt:
            total = tgt - cur
            before = total // 2
            pad = [(0, 0)] * 3
            pad[dim] = (before, total - before)
            out = np.pad(out, pad, mode="constant", constant_values=pad_val)
        elif cur > tgt:
            start = (cur - tgt) // 2
            out = np.take(out, range(start, start + tgt), axis=dim)
    assert out.shape == tuple(target), f"{out.shape} != {target}"
    return out


# ---------------------------------------------------------------------------
# Dot injection (the synthetic label)
# ---------------------------------------------------------------------------

def inject_dot(vol: np.ndarray, rng: np.random.Generator,
               radius: int = 4, value: float = 1.0,
               center_jitter: float = 0.35) -> np.ndarray:
    """Burn a bright sphere near the centre of ``vol`` (in place on a copy).

    radius        : sphere radius in voxels.
    value         : *absolute* intensity written into the sphere. The caller
                    decides the semantics: for the synthetic task we pass a
                    value above the background ceiling so the dot is genuinely
                    the brightest region; for real volumes we pass
                    ``dot_frac * vol.max()`` to mean "95% of the post-
                    preprocessing max", as in the original plan.
    center_jitter : max offset of the sphere centre from the volume centre,
                    as a fraction of each dimension (so the dot is "randomly
                    around the centre", not always exactly centred).
    """
    vol = vol.copy()
    z, y, x = vol.shape
    cz, cy, cx = z / 2.0, y / 2.0, x / 2.0
    cz += rng.uniform(-center_jitter, center_jitter) * z
    cy += rng.uniform(-center_jitter, center_jitter) * y
    cx += rng.uniform(-center_jitter, center_jitter) * x

    zz, yy, xx = np.ogrid[:z, :y, :x]
    sphere = ((zz - cz) ** 2 + (yy - cy) ** 2 + (xx - cx) ** 2) <= radius ** 2
    vol[sphere] = value
    return vol


# ---------------------------------------------------------------------------
# Augmentation (kept gentle so the dot survives)
# ---------------------------------------------------------------------------

@dataclass
class AugConfig:
    flip: bool = True                 # random axis flips
    rotate_deg: float = 10.0          # +/- small rotation (reshape=False -> shape kept)
    noise_std: float = 0.02           # additive Gaussian noise (in normalised units)


def augment(vol: np.ndarray, cfg: AugConfig, rng: np.random.Generator) -> np.ndarray:
    out = vol
    if cfg.flip:
        for ax in range(3):
            if rng.random() < 0.5:
                out = np.flip(out, axis=ax)
    if cfg.rotate_deg > 0:
        angle = rng.uniform(-cfg.rotate_deg, cfg.rotate_deg)
        axes = (int(rng.integers(0, 3)), 0)
        axes = tuple(sorted({axes[0], (axes[0] + 1) % 3}))
        out = nd_rotate(out, angle=angle, axes=axes, reshape=False,
                        order=1, mode="constant", cval=0.0)
    if cfg.noise_std > 0:
        out = out + rng.normal(0.0, cfg.noise_std, size=out.shape).astype(np.float32)
    return np.ascontiguousarray(out, dtype=np.float32)


# ---------------------------------------------------------------------------
# Synthetic dataset
# ---------------------------------------------------------------------------

class SyntheticDotDataset(Dataset):
    """Smooth random background volumes; half get a bright dot (label 1).

    Backgrounds use a per-sample random intensity scale/offset so the model
    cannot cheat by reading a global statistic -- it must localise the dot.
    """

    def __init__(
        self,
        n_samples: int,
        img_size: Tuple[int, int, int] = (64, 64, 64),
        radius: int = 4,
        dot_frac: float = 0.95,
        bg_ceiling: float = 0.6,
        normalize: str = "minmax",
        augment_cfg: Optional[AugConfig] = None,
        seed: int = 0,
    ):
        # The background is scaled into [0, bg_ceiling] and the dot is written at
        # the absolute value ``dot_frac`` (e.g. 0.95). With dot_frac > bg_ceiling
        # the dot is the brightest, localized region -> a clean, learnable signal.
        # Set bg_ceiling closer to dot_frac to make the task progressively harder.
        if not 0.0 < bg_ceiling < dot_frac <= 1.0:
            raise ValueError("require 0 < bg_ceiling < dot_frac <= 1")
        self.n = n_samples
        self.img_size = tuple(img_size)
        self.radius = radius
        self.dot_frac = dot_frac
        self.bg_ceiling = bg_ceiling
        self.normalize = normalize
        self.aug = augment_cfg
        # Fixed per-sample seeds -> deterministic dataset (reproducible splits).
        self.seeds = np.random.SeedSequence(seed).generate_state(n_samples)
        self.labels = (np.arange(n_samples) % 2).astype(np.float32)  # balanced

    def __len__(self) -> int:
        return self.n

    def _background(self, rng: np.random.Generator) -> np.ndarray:
        from scipy.ndimage import gaussian_filter
        base = rng.normal(0.0, 1.0, size=self.img_size).astype(np.float32)
        base = gaussian_filter(base, sigma=3.0)          # smooth, anatomy-like blobs
        scale = rng.uniform(0.5, 2.0)
        offset = rng.uniform(-1.0, 1.0)
        return base * scale + offset

    def __getitem__(self, idx: int):
        rng = np.random.default_rng(int(self.seeds[idx]))
        label = float(self.labels[idx])

        vol = self._background(rng)
        vol = normalize_volume(vol, mode=self.normalize)   # preprocess first ...
        vol = vol * self.bg_ceiling                        # ... cap background ...
        if label > 0.5:
            vol = inject_dot(vol, rng, radius=self.radius,  # ... then inject dot
                             value=self.dot_frac)
        if self.aug is not None:
            vol = augment(vol, self.aug, rng)

        x = torch.from_numpy(vol).unsqueeze(0).float()      # [1, Z, Y, X]
        return x, torch.tensor(label, dtype=torch.float32)


# ---------------------------------------------------------------------------
# Real-nifti dataset
# ---------------------------------------------------------------------------

class NiftiDotDataset(Dataset):
    """Load real volumes, preprocess like the downstream pipeline, inject a dot.

    ``items`` is a list of dicts: ``{"img": <path>, "seg": <path-or-None>}``.
    Half the items (by index parity, shuffled by ``seed``) receive a dot.
    If a seg is given, the volume is cropped to the seg bounding box first
    (so the dot lands inside the lesion region), matching ``OpTumorCrop``.
    """

    def __init__(
        self,
        items: Sequence[dict],
        img_size: Tuple[int, int, int] = (64, 64, 64),
        radius: int = 4,
        dot_frac: float = 0.95,
        normalize: str = "minmax",
        crop_to_seg: bool = True,
        augment_cfg: Optional[AugConfig] = None,
        seed: int = 0,
    ):
        if nib is None:
            raise ImportError("nibabel is required for NiftiDotDataset")
        self.items = list(items)
        self.img_size = tuple(img_size)
        self.radius = radius
        self.dot_frac = dot_frac
        self.normalize = normalize
        self.crop_to_seg = crop_to_seg
        self.aug = augment_cfg
        rng = np.random.default_rng(seed)
        self.labels = rng.integers(0, 2, size=len(self.items)).astype(np.float32)
        self.seeds = np.random.SeedSequence(seed + 1).generate_state(len(self.items))

    def __len__(self) -> int:
        return len(self.items)

    @staticmethod
    def _load_zyx(path: str) -> np.ndarray:
        """Load a nifti and return [Z, Y, X] (shortest axis -> Z), as in GIST."""
        arr = nib.load(path).get_fdata().astype(np.float32)     # [X, Y, Z]
        min_axis = int(np.argmin(arr.shape))
        order = [a for a in range(3) if a != min_axis] + [min_axis]
        arr = np.transpose(arr, order)                          # min axis last
        return np.transpose(arr, (2, 1, 0))                     # -> [Z, Y, X]

    def __getitem__(self, idx: int):
        item = self.items[idx]
        rng = np.random.default_rng(int(self.seeds[idx]))
        label = float(self.labels[idx])

        vol = self._load_zyx(item["img"])
        mask = None
        seg_path = item.get("seg")
        if seg_path and os.path.exists(seg_path):
            mask = self._load_zyx(seg_path)
            if self.crop_to_seg and mask.any():
                nz = np.argwhere(mask > 0)
                lo = nz.min(0)
                hi = nz.max(0) + 1
                vol = vol[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]]
                mask = mask[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]]

        vol = normalize_volume(vol, mode=self.normalize, mask=mask)
        vol = resize_to_shape(vol, self.img_size, pad_val=0.0)
        if label > 0.5:
            # "95% of the post-preprocessing max", per the original plan.
            vol = inject_dot(vol, rng, radius=self.radius,
                             value=self.dot_frac * float(vol.max()))
        if self.aug is not None:
            vol = augment(vol, self.aug, rng)

        x = torch.from_numpy(vol).unsqueeze(0).float()
        return x, torch.tensor(label, dtype=torch.float32)


def split_indices(n: int, val_frac: float, seed: int) -> Tuple[List[int], List[int]]:
    """Deterministic train/val index split."""
    rng = np.random.default_rng(seed)
    idx = rng.permutation(n)
    n_val = int(round(n * val_frac))
    return idx[n_val:].tolist(), idx[:n_val].tolist()
