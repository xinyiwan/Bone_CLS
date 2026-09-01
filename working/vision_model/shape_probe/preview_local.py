"""
Eyeball the clinical shape generator on a laptop -- no build, no cv2, no GPU.

`preview.py` renders a contact sheet from a finished `shape_metadata.csv`, so it
needs `build_shapes.py` to have run against real preprocess crops. This one calls
`shapes.clinical_polygon` directly and draws with PIL, so it runs anywhere numpy
and Pillow are installed. Use it to check a generator change BEFORE rebuilding a
dataset or spending GPU time on it.

The README's standing instruction applies here more than anywhere: **look at the
tiles and check you would label them correctly yourself.** If a human cannot
separate the classes at a given difficulty, a model failing there says nothing
about the model.

    # one row per class, columns sweeping difficulty
    python preview_local.py --out /tmp/shapes.png

    # more examples per cell, and the descriptors printed as a table
    python preview_local.py --out /tmp/shapes.png --n 6 --stats

    # check a nuisance parameter is not a class cue
    python preview_local.py --stats --no-image

`--stats` prints, per class and difficulty, the three descriptors the README
tracks (`dom_k`, `kurt`, peak-to-peak) plus `aspect`, all measured from the
generated r(theta) rather than assumed. Read the OVERLAP between classes: a
descriptor whose ranges do not overlap is a shortcut cue a model can exploit
without judging boundary character at all, which is exactly how the round_oval
elongation bug went unnoticed.
"""

from __future__ import annotations

import argparse
import math
import random
import sys
import types
from pathlib import Path
from typing import Dict, List

import numpy as np

# shapes.py imports cv2 only for its drawing helpers, which this script does not
# use -- it draws with PIL instead. Stub the module so the import succeeds on a
# machine without OpenCV; if a cv2-backed function is ever called it will raise
# AttributeError loudly rather than silently drawing nothing.
if "cv2" not in sys.modules:
    try:
        import cv2  # noqa: F401
    except ImportError:
        sys.modules["cv2"] = types.ModuleType("cv2")

sys.path.insert(0, str(Path(__file__).resolve().parent))
import shapes as S  # noqa: E402

CLASSES = ["round_oval", "lobulated", "irregular"]


def descriptors(family: str, difficulty: float, rng: random.Random) -> Dict[str, float]:
    """The README's three separation descriptors, measured from r(theta).

    Computed here rather than read from `shape_params` because the generator does
    not record them, and because they are the only descriptors defined for ALL
    classes -- the recorded params (lobe_k, jag_amp, ...) exist for one class
    each, so any comparison across classes using them is meaningless.
    """
    theta, r, p = S.clinical_radii(family, 512, difficulty, rng)
    dev = r - r.mean()
    spec = np.abs(np.fft.rfft(dev))
    spec[0] = 0.0
    m2 = float((dev ** 2).mean())
    return {
        "dom_k": float(np.argmax(spec)),
        "kurt": float((dev ** 4).mean() / m2 ** 2) if m2 > 0 else float("nan"),
        "p2p": float(r.max() - r.min()),
        "aspect": float(p["aspect"]),
        # area = 0.5 * integral(r^2 dtheta) = pi * mean(r^2); reported over pi R^2
        # so 1.0 would mean "fills the inscribing circle". Must match across
        # classes or it is a shortcut cue.
        "area": float((r ** 2).mean()),
        # Fraction of R the longest spike reaches. >1.0 means it clips at the
        # crop edge, which silently truncates exactly the feature that defines
        # `irregular` -- so this is checked, not assumed.
        "max_r": float(r.max()),
    }


KEYS = ("dom_k", "kurt", "p2p", "aspect", "area", "max_r")

# Which descriptors are ALLOWED to separate the classes. dom_k and kurt are the
# stated discriminating cues -- the probe exists to test whether the model reads
# them. Everything else is a nuisance parameter, and a nuisance that separates the
# classes is a shortcut a model can take without judging boundary character at
# all. max_r is listed as intended because a spike reaching further than a smooth
# arc IS the feature, not a confound: it cannot be equalised without deleting the
# thing that makes `irregular` irregular.
INTENDED = {"dom_k", "kurt", "max_r"}

# Some separations are definitional rather than shortcuts. `round_oval` means
# "no deviation from a smooth curve", so p2p and max_r MUST distinguish it from
# the deformed classes -- equalising that would delete the class. What matters is
# that p2p does not separate lobulated from irregular, because then a model
# passes by measuring how far the boundary wanders instead of judging whether it
# wanders smoothly or in cusps. So the check is pair-aware, not key-aware.
DEFINITIONAL = {"p2p": {frozenset({"round_oval", "lobulated"}),
                        frozenset({"round_oval", "irregular"})}}


def is_shortcut(key: str, a: str, b: str) -> bool:
    if key in INTENDED:
        return False
    return frozenset({a, b}) not in DEFINITIONAL.get(key, set())


def stats_table(levels: List[float], n: int, seed: int) -> None:
    # Sample ONCE per (class, difficulty) and derive every table from the same
    # draws. Re-sampling per key would make the overlap check disagree with the
    # min/max printed right above it, and constructing a fresh Random() inside the
    # loop would silently make all n "samples" identical, collapsing every span to
    # a point and reporting everything as disjoint.
    rng = random.Random(seed)
    draws: Dict[tuple, List[Dict[str, float]]] = {
        (fam, d): [descriptors(fam, d, rng) for _ in range(n)]
        for fam in CLASSES for d in levels
    }

    print(f"\n{n} sample(s) per cell; mean [min, max] measured from r(theta)\n")
    for key in KEYS:
        tag = "" if key in INTENDED else "   (nuisance: must OVERLAP)"
        print(f"--- {key} ---{tag}")
        print(f"{'class':<12}" + "".join(f"{'d=' + str(d):>26}" for d in levels))
        for fam in CLASSES:
            cells = []
            for d in levels:
                vals = [x[key] for x in draws[(fam, d)]]
                cells.append(f"{np.mean(vals):>7.2f} [{min(vals):.2f}, {max(vals):.2f}]")
            print(f"{fam:<12}" + "".join(f"{c:>26}" for c in cells))

        for d in levels:
            spans = {fam: (min(x[key] for x in draws[(fam, d)]),
                           max(x[key] for x in draws[(fam, d)])) for fam in CLASSES}
            # Relative tolerance, or a column normalised to a constant (area,
            # after the RMS fix) reports itself as disjoint on 1e-16 of float
            # noise -- a false alarm on the very thing that was just fixed.
            tol = 1e-6 * (max(abs(v) for s in spans.values() for v in s) or 1.0)
            disjoint = [(a, b) for i, a in enumerate(CLASSES) for b in CLASSES[i + 1:]
                        if spans[a][1] < spans[b][0] - tol or spans[b][1] < spans[a][0] - tol]
            bad = [(a, b) for a, b in disjoint if is_shortcut(key, a, b)]
            ok = [(a, b) for a, b in disjoint if not is_shortcut(key, a, b)]
            if bad:
                print(f"    !! d={d}: DISJOINT for "
                      + ", ".join(f"{a}/{b}" for a, b in bad)
                      + " -- SHORTCUT CUE, fix the generator")
            if ok:
                print(f"    ok d={d}: separates "
                      + ", ".join(f"{a}/{b}" for a, b in ok)
                      + " (intended or definitional)")
        print()

    # Spike extent, as a fraction of the nominal radius R. R is NOT the image
    # half-width: build_shapes.py sets it from the lesion extent inside a crop
    # that carries a margin, so a shape may exceed R by a little and still sit
    # comfortably inside the image. What actually clips is
    #     max_r * R > half the crop,  with R = 0.5 * min_dim * lesion_frac * shape_scale
    # so the binding constraint is --shape-scale, not this number alone. Report
    # the distribution and let the caller do that arithmetic.
    allm = sorted(x["max_r"] for v in draws.values() for x in v)
    p50, p99, mx = (allm[len(allm) // 2], allm[int(0.99 * (len(allm) - 1))], allm[-1])
    print(f"spike extent (fraction of R, NORM_TARGET_RMS={S.NORM_TARGET_RMS}): "
          f"median {p50:.2f}  p99 {p99:.2f}  max {mx:.2f}")
    print(f"   keep --shape-scale <= {1.0 / (p99 * 0.7):.2f} for a crop with the usual "
          f"~0.7 lesion fraction, or the p99 spike lands outside the image")
    if p99 > 1.05:
        print("!! p99 is well past R -- lower shapes.NORM_TARGET_RMS")


def contact_sheet(levels: List[float], n: int, seed: int, size: int, out: Path) -> None:
    from PIL import Image, ImageDraw

    rng = random.Random(seed)
    pad, label_h = 6, 18
    cols = len(levels) * n
    sheet = Image.new("RGB", (cols * (size + pad) + pad + 90,
                              len(CLASSES) * (size + pad + label_h) + pad), "black")
    draw = ImageDraw.Draw(sheet)

    for ri, fam in enumerate(CLASSES):
        y0 = pad + ri * (size + pad + label_h)
        draw.text((4, y0 + size // 2), fam, fill="white")
        ci = 0
        for d in levels:
            for _ in range(n):
                # rotation randomised per tile, exactly as build_shapes.py does,
                # so a class cannot be recognised by a fixed orientation.
                poly, p = S.clinical_polygon(
                    fam, center=(size / 2, size / 2), radius=size * 0.40,
                    rotation_deg=rng.uniform(0, 360), difficulty=d, rng=rng)
                tile = Image.new("RGB", (size, size), "black")
                td = ImageDraw.Draw(tile)
                td.line([tuple(pt) for pt in poly] + [tuple(poly[0])],
                        fill=(255, 0, 0), width=2, joint="curve")
                x0 = 90 + pad + ci * (size + pad)
                sheet.paste(tile, (x0, y0))
                draw.text((x0 + 2, y0 + size + 3),
                          f"d={d} a={p['aspect']:.2f}", fill="gray")
                ci += 1
    out.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out)
    print(f"wrote {out}  ({sheet.size[0]}x{sheet.size[1]})")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=Path("/tmp/shape_preview.png"))
    ap.add_argument("--difficulty", default="0.65,1.0",
                    help="comma-separated levels, matching your build")
    ap.add_argument("--n", type=int, default=4, help="samples per class per difficulty")
    ap.add_argument("--size", type=int, default=128, help="tile size in px")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--stats", action="store_true", help="print the descriptor table")
    ap.add_argument("--no-image", action="store_true", help="skip the contact sheet")
    args = ap.parse_args()

    levels = [float(x) for x in str(args.difficulty).split(",") if x.strip()]
    if args.stats:
        stats_table(levels, max(args.n, 30), args.seed)
    if not args.no_image:
        contact_sheet(levels, args.n, args.seed, args.size, args.out)


if __name__ == "__main__":
    main()
