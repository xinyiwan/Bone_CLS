"""
Quick viewer for DICOM folders.
Shows 3 slices (25%, 50%, 75%) for each of the 3 planes (axial, coronal, sagittal).

Saves a preview.png inside each scan folder.
Set DATADIR to the root directory containing BONE_AI_* folders.
"""

from pathlib import Path
import os
import numpy as np
import pydicom
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


def load_volume(folder: Path) -> np.ndarray:
    dcm_files = sorted(folder.glob("*.dcm"))
    if not dcm_files:
        raise FileNotFoundError(f"No .dcm files found in {folder}")

    slices = []
    for f in dcm_files:
        ds = pydicom.dcmread(f)
        slices.append((float(getattr(ds, "InstanceNumber", 0)), ds.pixel_array.astype(np.float32)))

    slices.sort(key=lambda x: x[0])
    volume = np.stack([s[1] for s in slices], axis=0)  # (Z, H, W)
    return volume


def normalize(arr: np.ndarray) -> np.ndarray:
    lo, hi = np.percentile(arr, 1), np.percentile(arr, 99)
    arr = np.clip(arr, lo, hi)
    arr = (arr - lo) / (hi - lo + 1e-8)
    return arr


def plot_volume(volume: np.ndarray, title: str, save_path: Path):
    volume = normalize(volume)
    Z, H, W = volume.shape

    planes = {
        "Axial   (Z)": [volume[int(Z * q)] for q in (0.25, 0.50, 0.75)],
        "Coronal (Y)": [volume[:, int(H * q), :] for q in (0.25, 0.50, 0.75)],
        "Sagittal(X)": [volume[:, :, int(W * q)] for q in (0.25, 0.50, 0.75)],
    }

    fig = plt.figure(figsize=(10, 8))
    fig.suptitle(title, fontsize=12, y=0.98)
    gs = gridspec.GridSpec(3, 3, hspace=0.08, wspace=0.04)

    labels = ("25 %", "50 %", "75 %")
    for row, (plane_name, imgs) in enumerate(planes.items()):
        for col, img in enumerate(imgs):
            ax = fig.add_subplot(gs[row, col])
            ax.imshow(img, cmap="gray", origin="upper", aspect="equal")
            ax.axis("off")
            if col == 0:
                ax.set_ylabel(plane_name, fontsize=9, labelpad=4)
                ax.yaxis.set_label_position("left")
                ax.tick_params(left=False, labelleft=False)
            if row == 0:
                ax.set_title(labels[col], fontsize=9)

    plt.tight_layout(rect=[0, 0, 1, 0.97])

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved → {save_path}")
    else:
        plt.show()

    plt.close(fig)


def main():
    datadir = Path("/mnt/rimp/PROJECTS/BONE-AI/ADQUISICIONES/")  # root containing BONE_AI_* folders
    savedir = "/home/ext_xinwan/Bone_AI/preview"

    folders = sorted(f for f in datadir.rglob("*.dcm") if f.is_file())
    scan_dirs = sorted({f.parent for f in folders})

    for folder in scan_dirs:
        print(f"Loading from {folder} …")
        try:
            volume = load_volume(folder)
        except Exception as e:
            print(f"  Skipped ({e})")
            continue
        print(f"  Volume shape (Z, H, W): {volume.shape}")
        new_save_folder = Path(savedir + str(folder).replace('/mnt/rimp/PROJECTS/BONE-AI/ADQUISICIONES', ''))
        os.makedirs(new_save_folder, exist_ok=True)
        save_path = new_save_folder / "preview.png"
        plot_volume(volume, title=str(folder), save_path=save_path)


if __name__ == "__main__":
    main()
