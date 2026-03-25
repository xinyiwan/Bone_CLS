"""
Web viewer — DCM classifier with physics-label reference overlay.

Main CSV  (uploaded) : DCM-classifier output CSV  (Predicción Clases W/FS/C)
Ref CSV   (optional) : physics-label CSV from label_physics.py
                       (phys_sequence / phys_fat_sat / phys_contrast / phys_acquisition)

Physics label is shown below each image, colour-coded:
  green  – physics agrees with the active DCM-clf filter
  red    – physics disagrees
  grey   – no physics data for this row
"""

import os
import subprocess
import pandas as pd
import numpy as np
import streamlit as st
from datetime import datetime
from PIL import Image
from utils import pad_image, load_dicom_dataframe_csv

st.set_page_config(layout="wide", page_title="DICOM — Physics Reference")

n_cols = 4
n_rows = 3

# ---------------------------------------------------------------------------
# Label helpers
# ---------------------------------------------------------------------------

BUTTON_LABELS = [
    "T1W noFS noCE", "T1W FS noCE", "T1W noFS CE", "T1W FS CE",
    "T2W noFS",       "T2W FS",      "T2W noFS CE", "T2W FS CE",
    "T2* noFS noCE",  "T2* other",
    "PD noFS noCE",   "PD FS noCE",  "PD noFS CE",  "PD FS CE",
    "DWI", "Localizer", "Other", "To_review",
]

_DECODE_MAP = {
    "T1W noFS noCE":  ("T1W",  "N", "N"),
    "T1W FS noCE":    ("T1W",  "Y", "N"),
    "T1W noFS CE":    ("T1W",  "N", "Y"),
    "T1W FS CE":      ("T1W",  "Y", "Y"),
    "T2W noFS":       ("T2W",  "N", "-"),
    "T2W FS":         ("T2W",  "Y", "-"),
    "T2W noFS CE":    ("T2W",  "N", "Y"),
    "T2W FS CE":      ("T2W",  "Y", "Y"),
    "T2* noFS noCE":  ("T2*",  "N", "N"),
    "T2* other":      ("T2*",  "-", "-"),
    "PD noFS noCE":   ("PD",   "N", "N"),
    "PD FS noCE":     ("PD",   "Y", "N"),
    "PD noFS CE":     ("PD",   "N", "Y"),
    "PD FS CE":       ("PD",   "Y", "Y"),
    "DWI":            ("DW",   "-", "-"),
    "Localizer":      ("Localizer", "-", "-"),
    "Other":          ("Other", "-", "-"),
    "To_review":      ("To_review", "To_review", "To_review"),
}


def decode_selection(sel, orig_w, orig_fs, orig_c):
    return _DECODE_MAP.get(sel, (orig_w, orig_fs, orig_c))


def default_label(w, fs, c):
    """Derive the default pill label from DCM-classifier columns."""
    w, fs, c = str(w).strip(), str(fs).strip(), str(c).strip()
    if w == "T1W":
        return f"T1W {'FS' if fs=='Y' else 'noFS'} {'CE' if c=='Y' else 'noCE'}"
    if w == "T2W":
        return f"T2W {'FS' if fs=='Y' else 'noFS'}"
    if w == "DW":
        return "DWI"
    if w == "Other":
        return "Other"
    return "To_review"


# ---------------------------------------------------------------------------
# Physics reference helpers
# ---------------------------------------------------------------------------

def _phys_fat_yn(val) -> str:
    return "Y" if str(val).strip() in ("FS", "STIR") else "N"


def _phys_contrast_yn(val) -> str:
    return "Y" if str(val).strip() == "Contrast" else "N"


def phys_ref_label(row) -> str:
    """Format physics label as seq_fatsat_CE_acq with 'nFS'/'nCE' for absent fields."""
    def _v(key, empty="x"):
        v = str(row.get(key, "") or "").strip()
        return v if v and v.lower() not in ("nan", "none") else empty

    ce  = "CE"  if str(row.get("phys_contrast",    "") or "").strip() == "Contrast" else "nCE"
    fs  = _v("phys_fat_sat", "nFS")
    seq = _v("phys_sequence")
    acq = _v("phys_acquisition")
    return f"{seq}_{fs}_{ce}_{acq}"


def phys_ref_color(row, filter_w: str, filter_fs: str, filter_c: str) -> str:
    """
    Green  – physics agrees with the active DCM-clf filter.
    Red    – physics disagrees.
    Grey   – no physics data.
    """
    seq = str(row.get("phys_sequence", "") or "").strip()
    if not seq or seq.lower() in ("nan", "none", "x", ""):
        return "#888888"

    # Map DCM filter → physics equivalents
    dcm_to_phys_mod = {
        "T1W": "T1W", "T2W": "T2W", "T2*": "T2*",
        "PD": "PD", "DW": "DWI", "Other": None,
    }
    expected_mod = dcm_to_phys_mod.get(filter_w)
    if expected_mod is None:
        return "#888888"

    mod_ok = (seq == expected_mod)
    fs_ok  = (_phys_fat_yn(row.get("phys_fat_sat",  "")) == filter_fs) if filter_fs != "-" else True
    ce_ok  = (_phys_contrast_yn(row.get("phys_contrast", "")) == filter_c) if filter_c != "-" else True

    return "#44dd44" if (mod_ok and fs_ok and ce_ok) else "#ff4444"


def phys_badge(row, filter_w, filter_fs, filter_c) -> str:
    """Return an HTML span with the physics label, coloured by agreement."""
    label = phys_ref_label(row)
    color = phys_ref_color(row, filter_w, filter_fs, filter_c)
    return (
        f'<span style="color:{color}; font-size:0.72em; '
        f'font-family:monospace; font-weight:bold">{label}</span>'
    )


# ---------------------------------------------------------------------------
# Image generation (delegates to img_generator.py)
# ---------------------------------------------------------------------------

def run_img_generator(csv_path: str, img_base: str) -> bool:
    script = os.path.join(os.path.dirname(__file__), "img_generator.py")
    if not os.path.exists(script):
        st.error(f"img_generator.py not found at: {script}")
        return False
    try:
        st.info("Generating missing images… please wait.")
        res = subprocess.run(
            ["python3", script, "--excel", csv_path, "--out_dir", img_base],
            capture_output=True, text=True, check=True,
        )
        st.success("Images generated successfully.")
        return True
    except subprocess.CalledProcessError as e:
        st.error(f"img_generator.py failed:\n{e.stderr}")
        return False


# ---------------------------------------------------------------------------
# Main app
# ---------------------------------------------------------------------------

def main():
    st.sidebar.title("Configuration")

    path_results  = st.sidebar.text_input("1. Output folder",      value="/Proyecto/Results")
    path_img_base = st.sidebar.text_input("2. Image base folder",  value="/Proyecto/IMG")
    uploaded_main = st.sidebar.file_uploader("3. DCM-classifier CSV", type=["csv"])
    uploaded_ref  = st.sidebar.file_uploader("4. Physics-label CSV (optional)", type=["csv"])

    if uploaded_main is None:
        st.title("DICOM Classifier — Physics Reference")
        st.info("Upload the DCM-classifier CSV from the sidebar to start.")
        return

    # ── Image generator button ─────────────────────────────────────────────
    st.sidebar.divider()
    if st.sidebar.button("Generate missing images", help="Runs img_generator.py on the uploaded CSV"):
        os.makedirs(path_results, exist_ok=True)
        tmp_csv = os.path.join(path_results, "_tmp_upload.csv")
        with open(tmp_csv, "wb") as f:
            f.write(uploaded_main.getbuffer())
        run_img_generator(tmp_csv, path_img_base)
        if os.path.exists(tmp_csv):
            os.remove(tmp_csv)

    # ── Resume detection ────────────────────────────────────────────────────
    master_filename = f"Review_{uploaded_main.name.replace('.csv','')}.csv"
    master_filepath = os.path.join(path_results, master_filename)

    usar_progreso = False
    if os.path.exists(master_filepath):
        st.sidebar.warning(f"Previous progress found:\n**{master_filename}**")
        usar_progreso = st.sidebar.checkbox("Resume from progress", value=True)

    # ── Load data ───────────────────────────────────────────────────────────
    if "df_master" not in st.session_state or st.sidebar.button("Reload"):
        if usar_progreso and os.path.exists(master_filepath):
            df_temp = pd.read_csv(master_filepath, dtype=str).fillna("")
            st.toast("Previous progress loaded", icon="📁")
        else:
            df_temp = load_dicom_dataframe_csv(uploaded_main)
            for col, default in [
                ("viewed",        ""),
                ("Clase W Final",  df_temp.get("Predicción Clases W",  pd.Series()).fillna("")),
                ("Clase FS Final", df_temp.get("Predicción Clases FS", pd.Series()).fillna("")),
                ("Clase C Final",  df_temp.get("Predicción Clases C",  pd.Series()).fillna("")),
            ]:
                if col not in df_temp.columns:
                    df_temp[col] = default if isinstance(default, str) else default.values
            st.toast("CSV loaded from scratch", icon="📄")

        # Merge physics reference if supplied
        _phys_cols = ["phys_sequence", "phys_acquisition", "phys_fat_sat", "phys_contrast"]
        df_temp = df_temp.drop(columns=[c for c in _phys_cols if c in df_temp.columns])
        if uploaded_ref is not None:
            df_ref = pd.read_csv(uploaded_ref, dtype=str).fillna("")
            # Normalise key columns to match DCM clf naming
            df_ref = df_ref.rename(columns={
                "subject": "Serie", "session": "Estudio", "scan": "Paciente"
            })
            join_keys = [k for k in ["Paciente", "Estudio", "Serie"] if k in df_temp.columns and k in df_ref.columns]
            if join_keys:
                df_temp = df_temp.merge(
                    df_ref[join_keys + [c for c in _phys_cols if c in df_ref.columns]],
                    on=join_keys, how="left"
                )
        for c in _phys_cols:
            if c not in df_temp.columns:
                df_temp[c] = ""

        st.session_state.df_master        = df_temp
        st.session_state.master_filepath  = master_filepath
        st.session_state.has_physics_ref  = uploaded_ref is not None

    df = st.session_state.df_master
    has_phys = st.session_state.get("has_physics_ref", False)

    # ── Summary table ───────────────────────────────────────────────────────
    with st.expander("Summary table", expanded=False):
        grp_cols = ["Predicción Clases W", "Predicción Clases FS", "Predicción Clases C"]
        grp_cols = [c for c in grp_cols if c in df.columns]
        resumen  = df.groupby(grp_cols).size().reset_index(name="Total")
        revisado = df[df["viewed"] == "X"].groupby(grp_cols).size().reset_index(name="Reviewed")
        tabla    = pd.merge(resumen, revisado, on=grp_cols, how="left").fillna(0)
        tabla["Reviewed"]  = tabla["Reviewed"].astype(int)
        tabla["Remaining"] = tabla["Total"] - tabla["Reviewed"]
        st.dataframe(tabla, use_container_width=True, hide_index=True)

    # ── Filters ─────────────────────────────────────────────────────────────
    st.sidebar.divider()
    st.sidebar.subheader("Sequence filters")

    val_w  = st.sidebar.selectbox("Modality (W)",  ["T1W", "T2W", "T2*", "PD", "DW", "Other"])
    mask   = df["Predicción Clases W"] == val_w

    if val_w in ("DW", "Other"):
        val_fs, val_c = "-", "-"
    else:
        val_fs = st.sidebar.selectbox("Fat sat (FS)", ["N", "Y"])
        mask   = mask & (df["Predicción Clases FS"] == val_fs)
        if val_w == "T1W":
            val_c = st.sidebar.selectbox("Contrast (C)", ["N", "Y"])
            mask  = mask & (df["Predicción Clases C"] == val_c)
        else:
            val_c = "-"

    show_pending = st.sidebar.checkbox("Show only pending", value=True)
    if show_pending:
        mask = mask & (df["viewed"] != "X")

    df_filtered = df[mask].reset_index()

    # ── Page header ─────────────────────────────────────────────────────────
    st.title(f"Review: {val_w}  |  FS: {val_fs}  |  CE: {val_c}")
    if has_phys:
        st.caption("Physics reference label shown below each image  •  green = agrees  •  red = disagrees  •  grey = no data")

    if df_filtered.empty:
        st.success("All reviewed for this filter, or no cases found.")
        return

    batch       = df_filtered.head(n_cols * n_rows)
    user_actions = {}

    with st.form("batch_form"):
        cols = st.columns(n_cols)
        for i, (_, row) in enumerate(batch.iterrows()):
            orig_idx = row["index"]
            with cols[i % n_cols]:
                paciente = str(row.get("Paciente", ""))
                estudio  = str(row.get("Estudio",  ""))
                serie    = str(row.get("Serie",    ""))

                img_path = os.path.join(path_img_base, paciente, estudio, serie, "Img.png")
                try:
                    img_pil    = Image.open(img_path).convert("L")
                    img_padded = pad_image(np.array(img_pil))
                    st.image(img_padded, use_container_width=True, clamp=True)
                except Exception as e:
                    st.error(f"Image not found: {serie[:15]}…\n{e}")

                # Physics reference label
                if has_phys:
                    badge_html = phys_badge(row, val_w, val_fs, val_c)
                    st.markdown(badge_html, unsafe_allow_html=True)

                orig_w  = row.get("Predicción Clases W",  "")
                orig_fs = row.get("Predicción Clases FS", "")
                orig_c  = row.get("Predicción Clases C",  "")
                initial = default_label(orig_w, orig_fs, orig_c)

                sel = st.pills(
                    "Class", BUTTON_LABELS,
                    default=initial,
                    key=f"p_{orig_idx}",
                    label_visibility="collapsed",
                )
                user_actions[orig_idx] = sel

        if st.form_submit_button("Save & Next", use_container_width=True):
            for idx, sel in user_actions.items():
                fila = st.session_state.df_master.loc[idx]
                orig_w  = fila.get("Predicción Clases W",  "")
                orig_fs = fila.get("Predicción Clases FS", "")
                orig_c  = fila.get("Predicción Clases C",  "")

                if not sel:
                    sel = default_label(orig_w, orig_fs, orig_c)

                w_f, fs_f, c_f = decode_selection(sel, orig_w, orig_fs, orig_c)

                st.session_state.df_master.at[idx, "viewed"]        = "X"
                st.session_state.df_master.at[idx, "Clase W Final"]  = w_f
                st.session_state.df_master.at[idx, "Clase FS Final"] = fs_f
                st.session_state.df_master.at[idx, "Clase C Final"]  = c_f

            os.makedirs(path_results, exist_ok=True)

            # Master file (overwritten each time)
            st.session_state.df_master.to_csv(
                st.session_state.master_filepath, index=False
            )

            # Timestamped backup
            backup_dir  = os.path.join(path_results, "Backups")
            os.makedirs(backup_dir, exist_ok=True)
            backup_path = os.path.join(
                backup_dir,
                f"Review_{val_w}_{datetime.now().strftime('%Y%m%d-%H%M')}.csv",
            )
            st.session_state.df_master.to_csv(backup_path, index=False)

            st.rerun()


if __name__ == "__main__":
    main()
