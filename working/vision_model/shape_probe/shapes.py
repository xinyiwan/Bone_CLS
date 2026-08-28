"""
Synthetic shape rasterizers for the perception probe.

The whole point of the probe is that the drawn shape is the ONLY thing that
distinguishes classes, so every shape is drawn the same way the real
segmentation contour is drawn in `preprocess/overlay.py`:

    RGB uint8 image, red (255, 0, 0) polyline, thickness 2, anti-aliased off

so the model sees an overlay of the same "visual species" as the radiologist
contour it gets in the real run. If you change the real overlay's colour or
thickness, change `DEFAULT_COLOR` / `DEFAULT_THICKNESS` here too -- otherwise
the probe stops being a proxy for the real task.

All shapes are inscribed in a circle of radius `radius_px` around `center`, so
the four classes cover comparable image area and cannot be told apart by size.

TWO SHAPE SETS
--------------
`icons` (circle/square/triangle/star) is the original probe. Every class is a
polygon with a distinct VERTEX COUNT (3, 4, inf, 10-with-spikes), so a model can
solve it by counting corners -- a categorical cue that real tumour margins do
not have. Near-perfect accuracy there says "the overlay is visible", nothing
more.

`clinical` is the harder set: values of the `shape` feature in
medgemma_pilot/feature_prompts.yaml. All five generators live here, but
`build_shapes.py --skip-shapes` decides which are actually built -- by default
`geographic` and `exophytic` are left out, because their geometry here does not
carry the clinical meaning of the words (geographic is about how sharply
demarcated a border is, not concavity; exophytic is about growth out of the host
bone, which a free-floating outline cannot express). All five come out of ONE
radial equation

    r(theta) = R * [1 + a*sin(k*theta + phi)          # lobulated: k smooth bulges
                      - d*dent(theta)                 # geographic: one concave arc
                      + b*noise(theta)                # irregular: high-freq jaggedness
                      + c*bump(theta)                 # exophytic: one flat-topped stalk
                      + eps*surface(theta)]           # tiny texture on ALL families

with only the parameters differing, so corner-counting cannot separate them --
the model has to judge the CHARACTER of the boundary. Because a/d/b/c are
continuous, `difficulty` sweeps deformation amplitude and turns the probe from a
pass/fail number into a psychometric curve ("lobulated separates from round once
bulges exceed ~15% of R"), which can then be compared against the deformation
amplitude actually present in the annotated lesions.
"""

from __future__ import annotations

import math
import random
from typing import Dict, Sequence, Tuple

import cv2
import numpy as np

SHAPES = ("circle", "square", "triangle", "star")

# The five labels of the `shape` feature in medgemma_pilot/feature_prompts.yaml,
# in snake_case (CSV/JSON-safe). CLINICAL_LABEL_TEXT maps back to the exact
# ground-truth vocabulary used in the real run.
CLINICAL_SHAPES = ("round_oval", "lobulated", "geographic", "irregular", "exophytic")

CLINICAL_LABEL_TEXT = {
    "round_oval": "Round/Oval",
    "lobulated": "lobulated",
    "geographic": "geographic",
    "irregular": "irregular",
    "exophytic": "exophytic",
}

SHAPE_SETS: Dict[str, Tuple[str, ...]] = {"icons": SHAPES, "clinical": CLINICAL_SHAPES}

# Deformation amplitude at difficulty 1.0, i.e. the EASY end: each is a fraction
# of R. `difficulty` multiplies all of them, so 0.35 means "bulges/dents/spikes
# are ~1/3 as pronounced" -- the same five classes, closer together.
BASE = {
    "aspect": 0.45,   # round_oval: ellipse elongation (1 + aspect)
    "lobe": 0.15,     # lobulated: sinusoid amplitude
    "dent": 0.80,     # geographic: depth of the single concave arc
    "jag": 0.34,      # irregular: spike amplitude (see IRREGULAR_* below)
    "bump": 1.10,     # exophytic: height of the single outward stalk
}

# SEPARATING `irregular` FROM `lobulated`
# --------------------------------------
# These are the knobs, in descending order of how much they actually move the two
# classes apart. Two measurable descriptors track the effect (both computable from
# r(theta) alone -- see the FFT snippet at the bottom of this comment):
#
#   dom_k    dominant wavenumber = how many bulges go round the outline
#   kurt     kurtosis of (r - mean r) = 1.5 for an even wave, >6 for isolated
#            spikes. THIS is "jagged vs wavy", and it is the descriptor that
#            wavenumber alone does not touch.
#
# Reference values at difficulty 1.0:  lobulated dom_k 5.6, kurt 1.5, p2p 0.25
#                                      irregular dom_k 15,  kurt  7,  p2p 0.60
#
# 1. IRREGULAR_SHARPNESS (biggest effect, and the least obvious)
#    A sum of sinusoids is infinitely smooth, so raising the wavenumber only
#    converts a slow wave into a fast wave -- still a wave, still confusable with
#    lobulated. This exponent (see _sharpen) is what makes the margin ANGULAR.
#    1.0 = old pure-sine behaviour (kurt ~6); 2.0 -> kurt ~7 with visible cusps;
#    3.5 -> kurt ~19, spiky to the point of looking like rasterisation grit.
#    Costs nothing in amplitude: peak deviation is preserved by construction.
#
# 2. IRREGULAR_K / IRREGULAR_N_HARM
#    The band sets how many spikes fit round the perimeter. It stays strictly
#    ABOVE lobulated's k = 4-7 on purpose: wavenumber is the stated discriminating
#    cue, so overlapping bands would make some irregular images legitimately
#    lobulated-looking and cap achievable accuracy for reasons that have nothing
#    to do with the model. (10, 20) puts dom_k at ~15, a 3x gap. Going higher
#    stops helping once a period is only a few pixels -- see the resolution note.
#
# 3. BASE["jag"] vs BASE["lobe"]  (use with care)
#    The bluntest separator, currently 0.34 vs 0.15, so irregular deviates ~2.4x
#    further than lobulated. It works, but it is a *size* cue rather than a
#    character cue: a model can then pass by measuring how far the boundary wanders
#    without ever judging jaggedness. If you want the honest experiment -- "can it
#    see cusped vs smooth at matched amplitude" -- set jag to ~0.20 and rely on
#    SHARPNESS plus the band instead.
#
# 4. IRREGULAR_PATCHES / _PATCH_SIGMA / _FLOOR
#    The sparse envelope: where jaggedness is switched on. These control
#    appearance more than separability -- spikes everywhere at high frequency read
#    as a uniform texture, i.e. a *regular* look, which is wrong for a label whose
#    whole content is "no countable or repeatable geometry". Current values leave
#    ~70% of the perimeter actively jagged (recorded per image as `jag_cover`).
#    Note this interacts with SHARPNESS: both sparsify, and a low FLOOR together
#    with a high exponent leaves jaggedness in only one place, which reads as a
#    single localised disturbance rather than an irregular margin. If you raise
#    SHARPNESS, raise FLOOR with it. Keep the patch-count lower bound >= 2.
#
# 5. NOT a shape parameter, but often the real limit: RESOLUTION. At a 128px crop
#    the shape radius is ~30px, so a k=15 spike is ~2px wide -- no parameter value
#    makes that read as jagged. If the classes still look alike in the contact
#    sheet, try build_shapes.py --shape-scale 1.5 or a larger crop before
#    retuning anything here.
#
# To re-measure after a change:
#     u = r - r.mean(); F = abs(np.fft.rfft(u))**2; F[:2] = 0
#     dom_k = F.argmax(); kurt = ((u / u.std())**4).mean()
IRREGULAR_K = (10, 20)         # harmonic band: 10-20 oscillations round the perimeter
IRREGULAR_N_HARM = 6           # distinct wavenumbers drawn from that band
IRREGULAR_PATCHES = (3, 5)     # inclusive range for the number of jagged patches
IRREGULAR_PATCH_SIGMA = 0.50   # rad, angular half-width of one patch
IRREGULAR_SHARPNESS = 2.0      # cusp exponent; 1.0 = plain sine sum (see _sharpen)
IRREGULAR_FLOOR = 0.40         # min envelope value: keeps the "smooth" stretches
                               # slightly unsettled rather than perfectly round

DIFFICULTY_PRESETS = {"easy": 1.0, "medium": 0.6, "hard": 0.35}

# Roughly one pixel at typical radii; present on EVERY family so that "perfectly
# smooth rasterisation" is not itself a tell for round_oval. Keep it well below
# BASE["lobe"] * the smallest difficulty you sweep, or the texture itself starts
# to look lobulated and the round/lobulated boundary stops being controlled.
# With BASE["lobe"]=0.15 the margin at difficulty 0.35 is 0.15*0.35 = 0.05, i.e.
# ~5x this value -- still safe, but this is now the binding constraint on how low
# you can push either the lobe amplitude or the difficulty floor.
SURFACE_NOISE = 0.01

DEFAULT_COLOR = (255, 0, 0)  # RGB, matches preprocess.overlay.draw_contour_overlay
DEFAULT_THICKNESS = 2


def _polygon(center: Tuple[float, float], radius: float, n: int, rotation_deg: float) -> np.ndarray:
    """`n` vertices evenly spaced on the circumscribed circle."""
    cx, cy = center
    phi = math.radians(rotation_deg) - math.pi / 2  # -90deg -> first vertex points up
    pts = [
        (cx + radius * math.cos(phi + 2 * math.pi * i / n),
         cy + radius * math.sin(phi + 2 * math.pi * i / n))
        for i in range(n)
    ]
    return np.array(pts, dtype=np.int32)


def _star(center: Tuple[float, float], radius: float, rotation_deg: float,
          points: int = 5, inner_ratio: float = 0.4) -> np.ndarray:
    """A `points`-pointed star: outer/inner vertices alternating."""
    cx, cy = center
    phi = math.radians(rotation_deg) - math.pi / 2
    pts = []
    for i in range(2 * points):
        r = radius if i % 2 == 0 else radius * inner_ratio
        a = phi + math.pi * i / points
        pts.append((cx + r * math.cos(a), cy + r * math.sin(a)))
    return np.array(pts, dtype=np.int32)


def shape_polygon(shape: str, center: Tuple[float, float], radius: float,
                  rotation_deg: float = 0.0) -> np.ndarray | None:
    """Vertices for `shape`, or None for 'circle' (drawn analytically, not as a
    polygon, so it stays smooth at small radii)."""
    if shape == "circle":
        return None
    if shape == "square":
        return _polygon(center, radius, 4, rotation_deg + 45.0)  # +45 -> flat sides at rotation=0
    if shape == "triangle":
        return _polygon(center, radius, 3, rotation_deg)
    if shape == "star":
        return _star(center, radius, rotation_deg)
    raise ValueError(f"Unknown shape {shape!r} (expected one of {SHAPES})")


# --------------------------------------------------------------------------
# clinical set: one radial equation, five parameter regimes
# --------------------------------------------------------------------------

def _surface(theta: np.ndarray, rng: random.Random, k_lo: int, k_hi: int, n_harm: int) -> np.ndarray:
    """Band-limited periodic noise, unit-ish amplitude: a sum of `n_harm`
    harmonics with distinct integer wavenumbers from [k_lo, k_hi], random phases
    and random weights. Built from harmonics rather than per-pixel noise so the
    curve stays closed and smooth -- a jagged margin here means genuine high
    spatial frequency, not rasterisation grit.

    The weights matter: equal-amplitude harmonics produce an evenly-spaced,
    gear-like outline that reads as *regular*, which is exactly the wrong look
    for the `irregular` class. Random weights over distinct k give the
    unpredictable, non-repeating margin the label describes."""
    ks = rng.sample(range(k_lo, k_hi + 1), min(n_harm, k_hi - k_lo + 1))
    weights = [rng.uniform(0.35, 1.0) for _ in ks]
    norm = math.sqrt(sum(w * w for w in weights)) or 1.0
    acc = np.zeros_like(theta)
    for k, w in zip(ks, weights):
        acc += w * np.sin(k * theta + rng.uniform(0, 2 * math.pi)) / norm
    return acc


def _sharpen(u: np.ndarray, p: float) -> np.ndarray:
    """sign(u)*|u|^p, rescaled to keep the original peak amplitude.

    This is what makes `irregular` look ANGULAR rather than merely fast. A sum of
    sinusoids is infinitely smooth no matter how high the wavenumber, so raising
    the frequency alone turns a slow wave into a fast wave -- still a wave, and
    still easily confused with `lobulated`. Raising the deviation to a power > 1
    flattens everything near zero and leaves the extremes almost untouched, so
    the boundary sits quiet and then departs steeply: cusped spikes separated by
    calm stretches, which is the shape of the word.

    Peak amplitude is preserved deliberately: `p` must change the CHARACTER of the
    margin without changing how far it deviates, or it stops being independent of
    the amplitude knobs (BASE["jag"], difficulty) and the sweep confounds the two.

    p = 1 is the identity, i.e. the old pure-sine behaviour."""
    if p == 1.0:
        return u
    peak = float(np.abs(u).max())
    if peak <= 0:
        return u
    return np.sign(u) * (np.abs(u) / peak) ** p * peak


def _sparse_envelope(theta: np.ndarray, rng: random.Random) -> np.ndarray:
    """Angular mask in [IRREGULAR_FLOOR, 1] with a few randomly-placed maxima.

    A sum of wrapped Gaussians at random angles, normalised to peak at 1 and
    lifted off zero by IRREGULAR_FLOOR. Multiplying the high-frequency noise by
    this concentrates the jaggedness into a handful of patches, leaving the rest
    of the boundary comparatively smooth.

    Wrapping via np.angle(exp(i*dtheta)) rather than plain subtraction is what
    keeps a patch continuous across theta=0 -- without it a patch centred near
    the seam would be cut in half and read as two."""
    n = rng.randint(*IRREGULAR_PATCHES)
    env = np.zeros_like(theta)
    for _ in range(n):
        dth = np.angle(np.exp(1j * (theta - rng.uniform(0, 2 * math.pi))))
        env = np.maximum(env, np.exp(-(dth ** 2) / (2 * IRREGULAR_PATCH_SIGMA ** 2)))
    peak = float(env.max()) or 1.0
    return IRREGULAR_FLOOR + (1.0 - IRREGULAR_FLOOR) * env / peak


def clinical_radii(
    family: str,
    n_points: int,
    difficulty: float,
    rng: random.Random,
) -> Tuple[np.ndarray, np.ndarray, dict]:
    """(theta, r/R, params) for one clinical family. Radii are returned as a
    fraction of R so the caller controls absolute size, and `params` is written
    into the metadata CSV so eval can break accuracy down by the deformation
    amplitude that actually produced each image."""
    theta = np.linspace(0.0, 2.0 * math.pi, n_points, endpoint=False)
    p: dict = {"family": family, "difficulty": round(difficulty, 3)}

    # Every family carries the same faint surface texture and the same slight
    # ellipticity, so neither can be used as a shortcut cue for one class.
    # The texture band sits ABOVE IRREGULAR_K deliberately. When the two coincide,
    # the texture every class carries is a scaled-down copy of the irregular cue,
    # so round_oval and lobulated differ from irregular in amplitude at the same
    # wavenumber -- exactly the confound the shared texture exists to avoid.
    r = 1.0 + SURFACE_NOISE * _surface(theta, rng, 24, 40, 4)
    base_aspect = 1.0 + 0.10 * rng.random()

    if family == "round_oval":
        base_aspect = 1.0 + BASE["aspect"] * difficulty * rng.uniform(0.4, 1.0)

    elif family == "lobulated":
        # Several rounded convex lobes side by side. k is the discriminating cue
        # vs `irregular` (4-7 low harmonics vs IRREGULAR_K's high ones).
        #
        # BASE["lobe"] is deliberately SHALLOW (0.15R): the intended look is an
        # oval you can still see as an oval, with a gentle wave riding on it --
        # the way a real lobulated margin presents -- not a cauliflower of
        # deep-cut lobes. This makes lobulated vs round_oval the probe's hardest
        # pair by design, so read those two together in the confusion matrix.
        k = rng.randint(4, 7)
        a = BASE["lobe"] * difficulty * rng.uniform(0.8, 1.0)
        r = r + a * np.sin(k * theta + rng.uniform(0, 2 * math.pi))
        p.update(lobe_k=k, lobe_amp=round(a, 3))

    elif family == "geographic":
        # ONE broad, sharply demarcated concave arc -- a scalloped bite. Narrow
        # enough in angle to read as a bite, wide enough not to look like noise.
        d = BASE["dent"] * difficulty * rng.uniform(0.8, 1.0)
        sigma = rng.uniform(0.40, 0.55)          # rad, ~45-63 deg half-width
        dth = np.angle(np.exp(1j * (theta - rng.uniform(0, 2 * math.pi))))
        r = r - d * np.exp(-(dth ** 2) / (2 * sigma ** 2))
        p.update(dent_depth=round(d, 3), dent_sigma=round(sigma, 3))

    elif family == "irregular":
        # No repeatable geometry: unevenly-weighted harmonics, so bulges are
        # neither countable nor evenly spaced. The band is deliberately LOW
        # (IRREGULAR_K) and the noise is gated by a sparse angular envelope, so
        # spikes are few, wide apart and clustered in unpredictable patches
        # instead of running continuously round the whole contour -- continuous
        # high-frequency grit reads as a uniform texture, which is a *regular*
        # appearance and the opposite of what this label means.
        b = BASE["jag"] * difficulty * rng.uniform(0.8, 1.0)
        env = _sparse_envelope(theta, rng)
        jag = _sharpen(_surface(theta, rng, *IRREGULAR_K, IRREGULAR_N_HARM),
                       IRREGULAR_SHARPNESS)
        r = r + b * env * jag
        # jag_cover = mean envelope, i.e. roughly what fraction of the perimeter
        # is actually jagged -- recorded so eval can check whether sparser
        # examples are the ones being missed.
        p.update(jag_amp=round(b, 3), jag_cover=round(float(env.mean()), 3))

    elif family == "exophytic":
        # One dominant flat-topped protrusion on an otherwise smooth mass: the
        # mushroom/polypoid stalk. Super-Gaussian (^4) gives the flat cap that a
        # plain Gaussian would round off into a mere nipple.
        c = BASE["bump"] * difficulty * rng.uniform(0.8, 1.0)
        sigma = rng.uniform(0.22, 0.34)          # rad, narrow: it is a stalk
        dth = np.angle(np.exp(1j * (theta - rng.uniform(0, 2 * math.pi))))
        r = r + c * np.exp(-(dth ** 4) / (2 * sigma ** 4))
        p.update(bump_height=round(c, 3), bump_sigma=round(sigma, 3))

    else:
        raise ValueError(f"Unknown clinical family {family!r} (expected one of {CLINICAL_SHAPES})")

    r = np.clip(r, 0.15, None)
    p["aspect"] = round(base_aspect, 3)
    # Normalise so every family is inscribed in R: area/extent cannot separate
    # classes, only boundary character can.
    return theta, r / float(r.max()), p


def clinical_polygon(
    family: str,
    center: Tuple[float, float],
    radius: float,
    rotation_deg: float = 0.0,
    difficulty: float = 1.0,
    rng: random.Random | None = None,
    n_points: int = 512,
) -> Tuple[np.ndarray, dict]:
    """Closed contour vertices for one clinical family, plus its params."""
    rng = rng or random.Random()
    theta, r, p = clinical_radii(family, n_points, difficulty, rng)
    cx, cy = center
    phi = math.radians(rotation_deg)
    # Ellipticity applied in the shape's own frame, then rotated with it, so
    # elongation direction is not always image-vertical.
    # Elongate by squashing the minor axis (not stretching the major one), so
    # the contour still fits inside R and aspect cannot be read off as size.
    x = radius * r * np.cos(theta)
    y = radius * r * np.sin(theta) / p["aspect"]
    xr = cx + x * math.cos(phi) - y * math.sin(phi)
    yr = cy + x * math.sin(phi) + y * math.cos(phi)
    return np.round(np.stack([xr, yr], axis=1)).astype(np.int32), p


def draw_poly(
    rgb: np.ndarray,
    poly: np.ndarray,
    color: Sequence[int] = DEFAULT_COLOR,
    thickness: int = DEFAULT_THICKNESS,
    filled: bool = False,
) -> np.ndarray:
    """Draw an arbitrary closed contour with the real overlay's colour/thickness."""
    out = np.ascontiguousarray(np.asarray(rgb, dtype=np.uint8).copy())
    col = tuple(int(c) for c in color)
    if filled:
        cv2.fillPoly(out, [poly], col)
    else:
        cv2.polylines(out, [poly], isClosed=True, color=col, thickness=int(thickness), lineType=cv2.LINE_AA)
    return out


def draw_shape(
    rgb: np.ndarray,
    shape: str,
    center: Tuple[float, float],
    radius: float,
    rotation_deg: float = 0.0,
    color: Sequence[int] = DEFAULT_COLOR,
    thickness: int = DEFAULT_THICKNESS,
    filled: bool = False,
) -> np.ndarray:
    """Draw `shape` onto a copy of an RGB uint8 image and return it."""
    out = np.ascontiguousarray(np.asarray(rgb, dtype=np.uint8).copy())
    col = tuple(int(c) for c in color)
    t = -1 if filled else int(thickness)

    if shape == "circle":
        cv2.circle(out, (int(round(center[0])), int(round(center[1]))), int(round(radius)), col, t)
    else:
        # Rotation comes from the caller (build_shapes draws it from the seeded
        # RNG and records it in the metadata) -- do NOT re-draw it here, or the
        # recorded rotation_deg stops describing the image and the build stops
        # being reproducible from --seed.
        poly = shape_polygon(shape, center, radius, rotation_deg)
        if filled:
            cv2.fillPoly(out, [poly], col)
        else:
            cv2.polylines(out, [poly], isClosed=True, color=col, thickness=int(thickness))
    return out
