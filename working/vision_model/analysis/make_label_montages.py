"""Build example-slice montages per ground-truth label, for the radiologist review doc.

Reads output/label_out/jsons/<case_id>.json for tumor_shape / tumor_matrix_mri,
picks representative slices from output/preprocess/shape_256/<case_id>/shape/,
and saves a grid PNG per label into BONE-AI/results/label_examples/.

Shape montages: one tile per case, 3x3 grid.
Matrix montages: one column per case (subject), one row per modality
(T1W / T2W_FS / T1W_C), missing modality left blank. Each tile is
labeled with its imaging plane (orientation).
"""
import json
import re
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
from PIL import Image

ROOT = Path(__file__).resolve().parents[4]
LABELS_DIR = ROOT / "output/label_out/jsons"
SLICES_DIR = ROOT / "output/preprocess/shape_256_all_loc"
OUT_DIR = ROOT / "results/label_examples"
N_SUBJECTS = 10
SHAPE_GRID = 3  # 3x3
SHAPE_MODALITY_PREFERENCE = ["T1W"]
# row label -> acceptable modality tags, tried in order (T2W and T2* look
# alike and are grouped into one row)
MATRIX_ROW_MODALITIES = {
    "T1W": ("T1W",),
    "T1W_C": ("T1W_C",),
    "T2W_FS": ("T2W_FS"),
    "T1W_FS_C": ("T1W_FS_C",),
}
PLANE_RE = re.compile(r"_(coronal|sagittal|axial)_\d+(?:_overlay)?$")


def slug(label: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", label.lower()).strip("_")


def plane_of(path: Path) -> str:
    m = PLANE_RE.search(path.stem)
    return m.group(1) if m else "?"


def case_slices(case_id: str) -> list[Path]:
    case_dir = SLICES_DIR / case_id
    if not case_dir.is_dir():
        return []
    return [p for p in case_dir.glob("*/*.png") if "_overlay" not in p.stem]


def _modality_of(path: Path) -> str | None:
    m = re.match(r"^(.*)_(?:coronal|sagittal|axial)_\d+$", path.stem)
    return m.group(1) if m else None


def pick_slice(case_id: str) -> Path | None:
    candidates = case_slices(case_id)
    for modality in SHAPE_MODALITY_PREFERENCE:
        for p in candidates:
            if _modality_of(p) == modality:
                return p
    return candidates[0] if candidates else None


def pick_slice_for_modality(case_id: str, modality_group: tuple[str, ...]) -> Path | None:
    slices = case_slices(case_id)
    for modality in modality_group:
        for p in slices:
            if _modality_of(p) == modality:
                return p
        
    return None


def collect_cases_by_label() -> tuple[dict[str, list[str]], dict[str, list[str]]]:
    shape_cases = defaultdict(list)
    matrix_cases = defaultdict(list)
    for path in sorted(LABELS_DIR.glob("*.json")):
        case_id = path.stem
        data = json.loads(path.read_text())
        feats = data.get("imaging_features", {})
        shape = feats.get("tumor_shape")
        for s in (shape if isinstance(shape, list) else [shape]):
            if s:
                shape_cases[s].append(case_id)
        matrix = feats.get("tumor_matrix_mri")
        for m in (matrix if isinstance(matrix, list) else [matrix]):
            if m:
                matrix_cases[m].append(case_id)
    return shape_cases, matrix_cases


def _show(ax, path: Path | None):
    ax.axis("off")
    if path is None:
        return
    ax.imshow(Image.open(path).convert("RGB"))
    ax.text(
        0.5, -0.05, plane_of(path), transform=ax.transAxes,
        ha="center", va="top", fontsize=7, color="black",
    )


def make_shape_montage(label: str, case_ids: list[str], out_path: Path) -> None:
    slices = []
    for case_id in case_ids:
        p = pick_slice(case_id)
        if p is not None:
            slices.append((case_id, p))
        if len(slices) == SHAPE_GRID * SHAPE_GRID:
            break
    if not slices:
        print(f"  skip '{label}': no slices found")
        return

    fig, axes = plt.subplots(SHAPE_GRID, SHAPE_GRID, figsize=(9, 9.5))
    for ax in axes.flat:
        ax.axis("off")
    for ax, (case_id, p) in zip(axes.flat, slices):
        _show(ax, p)
        ax.set_title(case_id, fontsize=8)
    fig.suptitle(label, fontsize=14)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  wrote {out_path} ({len(slices)} slices)")


def make_matrix_montage(label: str, case_ids: list[str], out_path: Path) -> None:
    subjects = [c for c in case_ids if case_slices(c)][:N_SUBJECTS]
    if not subjects:
        print(f"  skip '{label}': no slices found")
        return

    n_rows, n_cols = len(MATRIX_ROW_MODALITIES), len(subjects)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(2.4 * n_cols, 2.6 * n_rows), squeeze=False)
    for row, (row_label, modality_group) in enumerate(MATRIX_ROW_MODALITIES.items()):
        for col, case_id in enumerate(subjects):
            ax = axes[row][col]
            _show(ax, pick_slice_for_modality(case_id, modality_group))
            if row == 0:
                ax.set_title(case_id, fontsize=8)
            if col == 0:
                ax.text(
                    -0.15, 0.5, row_label, transform=ax.transAxes,
                    ha="right", va="center", fontsize=9, rotation=90,
                )
    fig.suptitle(label, fontsize=14)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  wrote {out_path} ({n_cols} subjects)")


def main():
    shape_cases, matrix_cases = collect_cases_by_label()

    print("Shape labels:")
    for label, case_ids in shape_cases.items():
        make_shape_montage(label, case_ids, OUT_DIR / "shape" / f"{slug(label)}.png")

    print("Matrix labels:")
    for label, case_ids in matrix_cases.items():
        make_matrix_montage(label, case_ids, OUT_DIR / "matrix" / f"{slug(label)}.png")


if __name__ == "__main__":
    main()
