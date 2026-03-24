"""
DICOM sequence reviewer for physics-informed labels.

Mirrors PRMG_Visualizacion_local.py but filters on the four physics columns
produced by label_physics.py:

  phys_sequence   – T1W | T2W | T2* | PD | DWI | …
  phys_fat_sat    – FS | STIR | ""
  phys_contrast   – Contrast | ""
  phys_acquisition – FSE | SE | GRE | IR | FLAIR  (shown as info only)

Optional --ref points to the DCM-classifier CSV.  When supplied, a colour-
coded agreement label appears next to each image:
  green  – DCM clf agrees with the current physics filter
  red    – DCM clf disagrees
  grey   – no reference data
"""

import os
import argparse
import pandas as pd
import pydicom
import matplotlib.pyplot as plt
import tkinter as tk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from skimage.transform import resize
from PIL import Image
import numpy as np
from datetime import datetime


# ---------------------------------------------------------------------------
# Reference-label helpers
# ---------------------------------------------------------------------------

def _yn_fat(val) -> str:
    """Normalise phys_fat_sat to Y / N."""
    v = str(val).strip()
    return "Y" if v in ("FS", "STIR") else "N"


def _yn_contrast(val) -> str:
    """Normalise phys_contrast to Y / N."""
    return "Y" if str(val).strip() == "Contrast" else "N"


def format_ref_label(row) -> str:
    """Format DCM-classifier reference as e.g. 'T1W  FS  CE'.
    Returns 'NaN' when no reference data is available."""
    seq = row.get("dcm_modality", "")
    fs  = row.get("dcm_fat_sat",  "")
    ce  = row.get("dcm_contrast", "")
    if pd.isna(seq) or str(seq).strip() == "":
        return "NaN"
    fs_str = "FS"   if str(fs).strip() == "Y" else "noFS"
    ce_str = "CE"   if str(ce).strip() == "Y" else "noCE"
    return f"{seq}  {fs_str}  {ce_str}"


def ref_label_color(row, modality: str, fatsat: str, contrast: str) -> str:
    """Green if DCM clf agrees with the active physics filter, red otherwise."""
    seq = row.get("dcm_modality", "")
    fs  = row.get("dcm_fat_sat",  "")
    ce  = row.get("dcm_contrast", "")
    if pd.isna(seq) or str(seq).strip() == "":
        return "#888888"
    dcm_fs = "Y" if str(fs).strip() == "Y" else "N"
    dcm_ce = "Y" if str(ce).strip() == "Y" else "N"
    if str(seq).strip() == modality and dcm_fs == fatsat and dcm_ce == contrast:
        return "#44dd44"
    return "#ff4444"


# ---------------------------------------------------------------------------
# DICOM loading
# ---------------------------------------------------------------------------

def read_pixel_array(ds):
    try:
        return ds.pixel_array
    except Exception:
        ds.PixelRepresentation = 0
        return ds.pixel_array


def _normalize_img(img):
    if img.ndim == 3:
        img = img[0]
    lo, hi = float(img.min()), float(img.max())
    if hi > lo:
        return ((img.astype(float) - lo) / (hi - lo) * 255).astype(np.uint8)
    return np.zeros_like(img, dtype=np.uint8)


def load_image_cached(dcm_path):
    safe_name = dcm_path.replace("\\", "_").replace("/", "_").replace(":", "")
    cache_path = os.path.join(cache_folder, safe_name + ".jpg")
    if os.path.exists(cache_path):
        return np.array(Image.open(cache_path).convert("L"))
    ds  = pydicom.dcmread(dcm_path)
    img = read_pixel_array(ds)
    img_norm = _normalize_img(img)
    Image.fromarray(img_norm, mode="L").save(cache_path, quality=90)
    return img_norm


def pad_image(img, target_height, target_width):
    if img.ndim == 3:
        img = img[0]
    height, width = img.shape
    if height == 0 or width == 0 or target_height == 0 or target_width == 0:
        return np.zeros((max(target_height, 1), max(target_width, 1)))
    scale_h = target_height / height
    scale_w = target_width  / width
    scale   = min(scale_h, scale_w)
    if not np.isfinite(scale):
        return np.zeros((target_height, target_width))
    new_h = int(height * scale)
    new_w = int(width  * scale)
    img = resize(img, (new_h, new_w), anti_aliasing=True)
    pad_h = max(0, target_height - new_h)
    pad_w = max(0, target_width  - new_w)
    padded = np.pad(img,
                    ((pad_h // 2, pad_h - pad_h // 2),
                     (pad_w // 2, pad_w - pad_w // 2)),
                    mode="constant", constant_values=0)
    return padded[:target_height, :target_width]


# ---------------------------------------------------------------------------
# Batch navigation
# ---------------------------------------------------------------------------

def load_batch(start_index):
    return df_to_analyse.iloc[start_index:start_index + batch_size]


def on_button_click(label, dcm_path):
    global selections
    selections[dcm_path] = label


def save_to_csv():
    df_output = df_filtered.copy()

    def to_local(x):
        return os.path.normpath(
            str(x).replace("/Project", r"Z:\mnt\rimp\PROJECTS\BONE-AI")
        )

    df_output["seleccion"] = df_output["Nombre DICOM"].apply(
        lambda x: selections.get(to_local(x), "")
    )
    df_output["viewed"] = df_output["Nombre DICOM"].apply(
        lambda x: "X" if to_local(x) in viewed_images else ""
    )
    try:
        old_sel  = df_filtered.get("seleccion", pd.Series([""] * len(df_filtered), dtype=str)).fillna("")
        old_view = df_filtered.get("viewed",    pd.Series([""] * len(df_filtered), dtype=str)).fillna("")
        df_output["seleccion"] = df_output["seleccion"].where(df_output["seleccion"] != "", old_sel)
        df_output["viewed"]    = df_output["viewed"].where(   df_output["viewed"]    != "", old_view)
    except Exception:
        pass

    df_output.to_csv(output_csv,      index=False)
    df_output.to_csv(output_csv_traz, index=False)
    print("Saved to", output_csv)


def on_click(event):
    if event.inaxes in axes_list:
        idx = axes_list.tolist().index(event.inaxes)
        row = df_to_analyse.iloc[current_index + idx]
        print(f"Selected image: {row['Nombre DICOM']}")


def _update_nav_buttons():
    prev_button.config(state=tk.NORMAL if current_index > 0 else tk.DISABLED)
    next_button.config(
        state=tk.NORMAL
        if current_index + batch_size < len(df_to_analyse)
        else tk.DISABLED
    )


def prev_batch():
    global current_index
    if current_index > 0:
        current_index -= batch_size
        update_view()


def next_batch():
    global current_index
    save_to_csv()
    if current_index + batch_size < len(df_to_analyse):
        current_index += batch_size
        update_view()
    else:
        next_button.config(state=tk.DISABLED)
        save_button.config(state=tk.DISABLED)
        print("No more batches. Closing shortly.")
        root.after(2000, root.quit)


# ---------------------------------------------------------------------------
# View renderer
# ---------------------------------------------------------------------------

def update_view():
    global current_index, axes_list, button_containers

    for container in button_containers:
        container.destroy()
    button_containers = []

    batch      = load_batch(current_index)
    fig_width  = max(root.winfo_width(),  int(min_fig_width  * 100)) / 100
    fig_height = max(root.winfo_height(), int(min_fig_height * 100)) / 100 * 0.9

    fig, axes = plt.subplots(n_row, n_columns, figsize=(fig_width, fig_height))
    fig.suptitle(
        f"{sequence_name}  {current_index // batch_size + 1} / "
        f"{len(df_to_analyse) // batch_size + 1}",
        color="white", fontsize=12,
    )
    fig.patch.set_facecolor("black")
    axes = axes.flatten()
    axes_list = axes

    win_w = root.winfo_width()
    win_h = root.winfo_height()
    target_width  = int(win_w / n_columns)
    target_height = int(win_h / n_row)

    resol = win_w / win_h
    if   1.7 < resol < 1.8:  mov_y = 0.082 * win_h
    elif 1.55 < resol < 1.61: mov_y = 0.092 * win_h
    elif 1.3  < resol < 1.4:  mov_y = 0.085 * win_h
    else:                      mov_y = 0.089 * win_h

    valid_images = 0

    for i, (_, row) in enumerate(batch.iterrows()):
        dcm_path = row["Nombre DICOM"]
        dcm_path = os.path.normpath(
            dcm_path.replace("/Project", r"Z:\mnt\rimp\PROJECTS\BONE-AI")
        )
        num_img     = row.get("Num ImageType", row.get("Num_img", 0))
        folder_name = os.path.basename(os.path.dirname(dcm_path))
        acq         = str(row.get("phys_acquisition", "")).strip()
        viewed_images.add(dcm_path)

        try:
            img        = load_image_cached(dcm_path)
            padded_img = pad_image(img, target_height, target_width)

            axes[i].imshow(padded_img, cmap="gray")
            axes[i].axis("off")
            axes[i].set_facecolor("black")
            valid_images += 1

            col_i   = i % n_columns
            row_i   = i // n_columns
            x_pos   = int(win_w / n_columns * col_i)
            cell_h  = (fig.get_figheight() * fig.dpi) / n_row
            y_pos   = int(cell_h * row_i * 0.9 + mov_y)

            button_frame = tk.Frame(root, bg="black")
            button_frame.place(x=x_pos, y=y_pos)

            selected_option = tk.StringVar(value="Choose")

            def on_select(selection, path=dcm_path):
                on_button_click(selection, path)

            dropdown = tk.OptionMenu(button_frame, selected_option,
                                     *button_labels, command=on_select)
            dropdown.config(bg="gray", fg="white", font=("Arial", 9))
            dropdown.pack(side="left")

            # Image count
            img_color = "orange" if num_img < 10 else "#555555"
            tk.Label(button_frame, text=f"Img:{num_img}",
                     fg=img_color, bg="black", font=("Arial", 9)
                     ).pack(side="left", padx=(4, 2))

            # Acquisition type info label (phys_acquisition)
            if acq:
                tk.Label(button_frame, text=f"[{acq}]",
                         fg="#aaaaff", bg="black", font=("Arial", 9)
                         ).pack(side="left", padx=(0, 2))

            # DCM classifier reference (optional)
            if ref_csv_path:
                ref_text  = format_ref_label(row)
                ref_color = ref_label_color(row,
                                            _args.modality,
                                            _args.fatsat,
                                            _args.contrast)
                tk.Label(button_frame, text=ref_text,
                         fg=ref_color, bg="black", font=("Arial", 9, "bold")
                         ).pack(side="left", padx=(2, 4))

            button_containers.append(button_frame)

            already_viewed = str(row.get("viewed", "")).strip() == "X"
            axes[i].set_title(
                ("✓ " if already_viewed else "") + folder_name,
                fontsize=9,
                color="#44dd44" if already_viewed else "white",
                loc="center",
            )

        except Exception as e:
            print(f"Error loading {dcm_path}: {e}")

    # Blank unused cells
    for i in range(len(batch), len(axes)):
        axes[i].imshow(np.zeros((target_height, target_width)), cmap="gray")
        axes[i].axis("off")
        axes[i].set_facecolor("black")

    plt.subplots_adjust(left=0, right=1, top=0.9, bottom=0, wspace=0, hspace=0)
    canvas.figure = fig
    canvas.draw()
    _update_nav_buttons()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

_parser = argparse.ArgumentParser(description="Physics-label DICOM sequence reviewer")
_parser.add_argument("--modality", default="T1W",
                     choices=["T1W", "T2W", "T2*", "PD", "DWI", "Localizer", "Unknown"],
                     help="phys_sequence value to review (default: T1W)")
_parser.add_argument("--fatsat",   default="N", choices=["Y", "N"],
                     help="Fat saturation filter — Y (FS or STIR present) / N (default: N)")
_parser.add_argument("--contrast", default="N", choices=["Y", "N"],
                     help="Contrast filter — Y (Contrast present) / N (default: N)")
_parser.add_argument("--excel",
                     default=r"Z:\home\ext_xinwan\Bone_AI\output\DCM_PHYSICS\physics_labels.csv",
                     help="Path to physics-label CSV (label_physics.py output)")
_parser.add_argument("--output",
                     default=r"c:\Users\E78357656\Documents\output_viewer\physics",
                     help="Folder for saving review results")
_parser.add_argument("--ref", default="",
                     help="Path to DCM-classifier CSV for agreement overlay (optional)")
_args = _parser.parse_args()

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

excel_path    = _args.excel
output_folder = _args.output
ref_csv_path  = _args.ref if _args.ref else None
cache_folder  = os.path.join(output_folder, "cache")
os.makedirs(cache_folder, exist_ok=True)

# Build human-readable sequence name for window title / file names
sequence_name = _args.modality
sequence_name += "_FS"  if _args.fatsat   == "Y" else "_nFS"
sequence_name += "_CE"  if _args.contrast == "Y" else "_nCE"

n_row      = 3
n_columns  = 8
batch_size = n_row * n_columns
timestamp  = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

output_csv      = os.path.join(output_folder, f"Review_{sequence_name}.csv")
output_traz_dir = os.path.join(output_folder, "Backup")
os.makedirs(output_traz_dir, exist_ok=True)
output_csv_traz = os.path.join(output_traz_dir, f"Review_{sequence_name}_{timestamp}.csv")

button_labels = [
    "T1W  noFS  noCE", "T1W  FS  noCE", "T1W  noFS  CE", "T1W  FS  CE",
    "T2W  noFS",       "T2W  FS",
    "T2*", "PD", "DWI",
    "OTHERS", "To_review",
]
button_containers = []
selections    = {}
viewed_images = set()

# ---------------------------------------------------------------------------
# Load and filter data
# ---------------------------------------------------------------------------

try:
    df = pd.read_csv(output_csv)
except Exception:
    df = pd.read_csv(excel_path)

df_filtered = df.copy()

# Normalise physics columns to Y/N for filtering
df_filtered["_fs_yn"] = df_filtered["phys_fat_sat"].apply(_yn_fat)
df_filtered["_ce_yn"] = df_filtered["phys_contrast"].apply(_yn_contrast)

df_filtered = df_filtered[
    (df_filtered["phys_sequence"] == _args.modality) &
    (df_filtered["_fs_yn"]        == _args.fatsat)   &
    (df_filtered["_ce_yn"]        == _args.contrast)
].reset_index(drop=True)

# Merge optional DCM-classifier reference
_dcm_cols = ["dcm_modality", "dcm_fat_sat", "dcm_contrast"]
df_filtered = df_filtered.drop(columns=[c for c in _dcm_cols if c in df_filtered.columns])

if ref_csv_path:
    df_ref = pd.read_csv(ref_csv_path).rename(columns={
        "Predicción Clases W":  "dcm_modality",
        "Predicción Clases FS": "dcm_fat_sat",
        "Predicción Clases C":  "dcm_contrast",
        "Paciente": "scan",
        "Estudio":  "session",
        "Serie":    "subject",
    })
    df_filtered = df_filtered.merge(
        df_ref[["scan", "session", "subject"] + _dcm_cols],
        on=["scan", "session", "subject"], how="left"
    )

# Resume from last unreviewed image
df_to_analyse = df_filtered.copy()
try:
    viewed_col    = df_to_analyse.get("viewed", pd.Series([""] * len(df_to_analyse)))
    unreviewed    = (viewed_col != "X").values.nonzero()[0]
    first_pos     = int(unreviewed[0]) if len(unreviewed) > 0 else 0
    current_index = (first_pos // batch_size) * batch_size
except Exception:
    current_index = 0

print(f"Filtered to {len(df_to_analyse)} rows for: {sequence_name}")

# ---------------------------------------------------------------------------
# Build main window
# ---------------------------------------------------------------------------

root = tk.Tk()
root.title(f"Physics viewer — {sequence_name}")
root.configure(bg="black")
root.state("zoomed")

frame = tk.Frame(root, bg="black")
frame.pack(fill=tk.BOTH, expand=True)

min_fig_width  = root.winfo_screenwidth()  / 100
min_fig_height = root.winfo_screenheight() / 100 * 0.9

canvas = FigureCanvasTkAgg(
    plt.figure(figsize=(min_fig_width, min_fig_height)), master=frame
)
canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
canvas.mpl_connect("button_press_event", on_click)

nav_frame = tk.Frame(root, bg="black")
nav_frame.pack(pady=6)

prev_button = tk.Button(nav_frame, text="◀ Previous", command=prev_batch,
                        fg="white", bg="#333333", state=tk.DISABLED)
prev_button.pack(side="left", padx=8)

next_button = tk.Button(nav_frame, text="Next ▶", command=next_batch,
                        fg="white", bg="#333333")
next_button.pack(side="left", padx=8)

save_button = tk.Button(nav_frame, text="Save & Exit",
                        command=lambda: [save_to_csv(), root.quit()],
                        fg="white", bg="#333333")
save_button.pack(side="left", padx=8)

axes_list = np.array([])

_resize_id = None
def on_resize(event):
    global _resize_id
    if event.widget is root:
        if _resize_id:
            root.after_cancel(_resize_id)
        _resize_id = root.after(300, update_view)

root.bind("<Configure>", on_resize)

update_view()
root.mainloop()
