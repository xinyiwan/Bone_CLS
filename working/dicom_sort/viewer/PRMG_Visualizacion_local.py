import os
import pandas as pd
import pydicom
import matplotlib.pyplot as plt
import tkinter as tk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from skimage.transform import resize
import numpy as np
from datetime import datetime


def format_ref_label(row):
    """Format reference sequence info as e.g. 'T1W   FS  CE' or 'T2W   noFS  noCE'.
    Returns an empty string when no reference data is available."""
    seq = row.get("sequence_type", "")
    fs  = row.get("fat_sat",       "")
    ce  = row.get("contrast",      "")
    if pd.isna(seq) or str(seq).strip() == "":
        return "no matching"
    fs_str = "noFS" if (pd.isna(fs) or str(fs).strip() == "") else "FS"
    ce_str = "noCE" if (pd.isna(ce) or str(ce).strip() == "") else "CE"
    return f"{seq}   {fs_str}  {ce_str}"


def read_pixel_array(ds):
    """Read pixel data from a pydicom dataset.

    Some GE scanners incorrectly set PixelRepresentation=1 (signed int16)
    even when pixel values exceed the int16 range (e.g. 32768).  pydicom
    raises an overflow error when decoding such files.  The fix is to
    override the tag to unsigned (0) so pydicom decodes as uint16 instead.
    """
    try:
        return ds.pixel_array
    except Exception:
        ds.PixelRepresentation = 0
        return ds.pixel_array


def load_batch(start_index):
    """Return the next slice of rows from df_to_analyse."""
    return df_to_analyse.iloc[start_index:start_index + batch_size]


def on_button_click(label, dcm_path):
    """Store the user's label selection for a given DICOM path."""
    global selections
    selections[dcm_path] = label


def save_to_excel():
    """Write the current selections and viewed flags to both the main output
    file and a timestamped traceability backup."""
    df_output = df_filtered.copy()
    def to_local(x):
        return os.path.normpath(str(x).replace("/Project", r"Z:\mnt\rimp\PROJECTS\BONE-AI"))
    df_output["seleccion"] = df_output["Nombre DICOM"].apply(lambda x: selections.get(to_local(x), ""))
    df_output["viewed"] = df_output["Nombre DICOM"].apply(lambda x: "X" if to_local(x) in viewed_images else "")
    # Preserve any labels/viewed marks that were loaded from a previous session
    try:
        df_output["seleccion"] = df_filtered["seleccion"].where(df_filtered["seleccion"] == "X", df_output["seleccion"])
        df_output["viewed"] = df_filtered["viewed"].where(df_filtered["viewed"] == "X", df_output["viewed"])
    except Exception:
        pass
    df_output.to_csv(output_excel, index=False)
    df_output.to_csv(output_excel_traz, index=False)
    print("Saved to", output_excel)


def on_click(event):
    """Print the DICOM path when the user clicks on an image panel."""
    if event.inaxes in axes_list:
        idx = axes_list.tolist().index(event.inaxes)
        row = df_to_analyse.iloc[current_index + idx]
        dcm_path = row["Nombre DICOM"]
        print(f"Selected image: {dcm_path}")


def next_batch():
    """Save progress and advance to the next batch of images.

    Disables navigation buttons and schedules window close when the last
    batch has been reached.
    """
    global current_index
    save_to_excel()
    if current_index + batch_size < len(df_to_analyse):
        current_index += batch_size
        update_view()
    else:
        next_button.config(state=tk.DISABLED)
        save_button.config(state=tk.DISABLED)
        print("No more batches. The application will close shortly.")
        root.after(2000, root.quit)


def pad_image(img, target_height, target_width):
    """Resize *img* to fit within (target_height, target_width) while
    preserving aspect ratio, then centre-pad with zeros to exact target size."""
    # Flatten to 2D if multi-frame (take first frame)
    if img.ndim == 3:
        img = img[0]
    height, width = img.shape

    # Guard against degenerate sizes (window not yet rendered, or bad DICOM)
    if height == 0 or width == 0 or target_height == 0 or target_width == 0:
        return np.zeros((max(target_height, 1), max(target_width, 1)))

    scale_factor_h = target_height / height
    scale_factor_w = target_width / width
    scale_factor = scale_factor_w if scale_factor_h > scale_factor_w else scale_factor_h

    if not np.isfinite(scale_factor):
        return np.zeros((target_height, target_width))

    new_height = int(height * scale_factor)
    new_width = int(width * scale_factor)
    img = resize(img, (new_height, new_width), anti_aliasing=True)
    pad_height = max(0, target_height - new_height)
    pad_width = max(0, target_width - new_width)
    padded_img = np.pad(img, ((pad_height // 2, pad_height - pad_height // 2),
                              (pad_width // 2, pad_width - pad_width // 2)),
                        mode="constant", constant_values=0)
    return padded_img[:target_height, :target_width]


def update_view():
    """Render the current batch of DICOM images with their labelling dropdowns."""
    global current_index, axes_list, button_containers

    # Remove dropdown widgets from the previous batch
    for container in button_containers:
        container.destroy()
    button_containers = []

    batch = load_batch(current_index)
    fig_width  = max(root.winfo_width(),  int(min_fig_width  * 100)) / 100
    fig_height = max(root.winfo_height(), int(min_fig_height * 100)) / 100 * 0.9
    fig, axes = plt.subplots(n_row, n_columns, figsize=(fig_width, fig_height))
    fig.suptitle(f"{sequence_name} {current_index // batch_size + 1} / {len(df_to_analyse) // batch_size + 1}",
                 color="white", fontsize=12)
    fig.patch.set_facecolor("black")
    axes = axes.flatten()
    axes_list = axes

    # Use current window size so layout tracks the window when resized
    win_w = root.winfo_width()
    win_h = root.winfo_height()

    # Target cell size in pixels (used for resizing and widget placement)
    target_width  = int(win_w / n_columns)
    target_height = int(win_h / n_row)
    valid_images = 0

    # Adjust dropdown vertical offset based on current window aspect ratio
    resol = win_w / win_h
    if 1.7 < resol < 1.8:
        mov_y = 0.082 * win_h
    elif 1.55 < resol < 1.61:
        mov_y = 0.092 * win_h
    elif 1.3 < resol < 1.4:
        mov_y = 0.085 * win_h
    else:
        mov_y = 0.089 * win_h

    for i, (idx, row) in enumerate(batch.iterrows()):
        dcm_path = row["Nombre DICOM"]
        dcm_path = os.path.normpath(dcm_path.replace("/Project", r"Z:\mnt\rimp\PROJECTS\BONE-AI"))
        num_img = row.get("Num ImageType", row.get("Num_img", 0))
        folder_name = os.path.basename(os.path.dirname(dcm_path))
        viewed_images.add(dcm_path)

        try:
            ds = pydicom.dcmread(dcm_path)
            img = read_pixel_array(ds)
            padded_img = pad_image(img, target_height, target_width)

            axes[i].imshow(padded_img, cmap="gray")
            axes[i].axis("off")
            axes[i].set_facecolor("black")
            valid_images += 1

            # Horizontal position: left edge of each grid column
            col = i % n_columns
            x_pos = int(win_w / n_columns * col)

            # Vertical position: based on new figure pixel height
            row_idx = i // n_columns
            fig_height_px = fig.get_figheight() * fig.dpi
            y_pos = int((fig_height_px / n_row) * row_idx * 0.9 + mov_y)

            # Dropdown widget for sequence labelling
            button_frame = tk.Frame(root, bg="black")
            button_frame.place(x=int(x_pos), y=int(y_pos))

            selected_option = tk.StringVar()
            selected_option.set("Choose")

            # Highlight series with few images in red as a quality-control cue
            color = "red" if num_img < 10 else "white"
            label_img = tk.Label(button_frame, text=f"Img: {num_img}",
                                 fg=color, bg="black", font=("Arial", 10))
            label_img.pack(side="right", padx=20)

            # Reference label (sequence_type / fat_sat / contrast from external CSV)
            if ref_csv_path:
                ref_text = format_ref_label(row)
                ref_label = tk.Label(button_frame, text=ref_text,
                                     fg="cyan", bg="black", font=("Arial", 9))
                ref_label.pack(side="right", padx=5)

            def on_select(selection, path=dcm_path):
                on_button_click(selection, path)

            dropdown = tk.OptionMenu(button_frame, selected_option, *button_labels, command=on_select)
            dropdown.config(bg="gray", fg="white")
            dropdown.pack()

            # Colour-code dropdown entries by sequence type for quick identification
            menu = dropdown["menu"]
            color_mapping = {
                "T1W   noFS  noCE": "black",
                "T1W     FS    noCE": "black",
                "T1W   noFS     CE": "red",
                "T1W   FS     CE": "red",
                "T2W   noFS": "green",
                "T2W   FS": "green",
                "DWI": "black",
                "OTHERS": "black",
                "To_review": "black",
            }
            for idx, label in enumerate(button_labels):
                menu.entryconfigure(idx, foreground=color_mapping.get(label, "black"))

            button_containers.append(button_frame)
            axes[i].set_title(folder_name, fontsize=10, color="white", loc="center")

        except Exception as e:
            print(f"Error loading {dcm_path}: {e}")

    # Fill unused grid cells with blank panels
    for i in range(len(batch), len(axes)):
        blank_img = np.zeros((target_height, target_width))
        axes[i].imshow(blank_img, cmap="gray")
        axes[i].axis("off")
        axes[i].set_facecolor("black")

    for i in range(valid_images, len(axes)):
        axes[i].axis("off")
        axes[i].set_facecolor("black")

    plt.subplots_adjust(left=0, right=1, top=0.9, bottom=0, wspace=0, hspace=0)
    canvas.figure = fig
    canvas.draw()


# ── Configuration ─────────────────────────────────────────────────────────────
excel_path = r"Z:\home\ext_xinwan\Bone_AI\output\DCM_CLF\Results\Sequence_Classifier_test.csv"
output_folder = r"c:\Users\E78357656\Documents\output_viewer"

# Optional reference CSV with ground-truth labels.
# Columns: subject (=Paciente), session (=Estudio), scan (=Serie),
#          sequence_type, fat_sat, contrast
# Set to None to disable.
ref_csv_path = r"Z:\home\ext_xinwan\Bone_AI\output\DCM_DICT\dicom_header_labelled_Mar10.csv"  # e.g. r"Z:\path\to\reference.csv"

# Columns and values used to filter the sequence type to review
secuencia_colum = ["Predicción Clases W", "Predicción Clases FS", "Predicción Clases C"]
sequence_filter = ["T1W", "N", "Y"]

# Build a human-readable name for the sequence being reviewed
sequence_name = sequence_filter[0]
for ind, name_seq in enumerate(sequence_filter):
    if ind == 1:
        sequence_name += "_FS" if name_seq == "Y" else "_nFS"
    elif ind == 2:
        sequence_name += "_CE" if name_seq == "Y" else "_nCE"

n_row = 3
n_columns = 8
batch_size = n_row * n_columns
timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

output_excel = os.path.join(output_folder, "Resultados_revision_" + sequence_name + ".csv")
output_folder_trazability = os.path.join(output_folder, "Backup")
os.makedirs(output_folder_trazability, exist_ok=True)
output_excel_traz = os.path.join(output_folder_trazability, f"Review_{sequence_name}_{timestamp}.csv")

button_containers = []
button_labels = [
    "T1W   noFS  noCE", "T1W     FS    noCE", "T1W   noFS     CE", "T1W   FS     CE",
    "T2W   noFS", "T2W   FS", "DWI", "OTHERS", "To_review",
]
selections = {}

# Load from the latest backup if available, otherwise start from the classifier output
try:
    df = pd.read_csv(output_excel_traz)
except Exception:
    df = pd.read_csv(excel_path)

df_filtered = df.copy()
# Keep only rows matching the target sequence
for index, col in enumerate(secuencia_colum):
    df_filtered = df_filtered[df_filtered[col] == sequence_filter[index]].reset_index(drop=True)

# Load and merge optional reference labels
if ref_csv_path:
    df_ref = pd.read_csv(ref_csv_path).rename(columns={
        "subject": "Serie", "session": "Estudio", "scan": "Paciente"
    })
    df_filtered = df_filtered.merge(
        df_ref[["Paciente", "Estudio", "Serie", "sequence_type", "fat_sat", "contrast"]],
        on=["Paciente", "Estudio", "Serie"], how="left"
    )

# Skip images already marked as viewed in a previous session
try:
    df_to_analyse = df_filtered[df_filtered["viewed"] != "X"].reset_index(drop=True)
except Exception:
    df_to_analyse = df_filtered

viewed_images = set()
current_index = 0

# ── Build main window ─────────────────────────────────────────────────────────
root = tk.Tk()
root.title("DICOM image viewer")
root.configure(bg="black")
root.state("zoomed")

frame = tk.Frame(root, bg="black")
frame.pack(fill=tk.BOTH, expand=True)

# Persistent bottom button bar (lives outside update_view so it is never recreated)
button_frame = tk.Frame(root, bg="black")
button_frame.pack(pady=20)

# Minimum figure size (fallback before the window is fully rendered)
min_fig_width  = root.winfo_screenwidth()  / 100
min_fig_height = root.winfo_screenheight() / 100 * 0.9

canvas = FigureCanvasTkAgg(plt.figure(figsize=(min_fig_width, min_fig_height)), master=frame)
canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
canvas.mpl_connect("button_press_event", on_click)

next_button = tk.Button(root, text="Next Batch", command=next_batch, fg="white", bg="black")
next_button.pack()

save_button = tk.Button(root, text="Save and exit",
                        command=lambda: [save_to_excel(), root.quit()],
                        fg="white", bg="black")
save_button.pack()

# Redraw when the window is resized (debounced to avoid rapid redraws)
_resize_id = None
def on_resize(event):
    global _resize_id
    if event.widget is root:
        if _resize_id:
            root.after_cancel(_resize_id)
        _resize_id = root.after(300, update_view)

root.bind("<Configure>", on_resize)

# Render the first batch and start the event loop
update_view()
root.mainloop()
