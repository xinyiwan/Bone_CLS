import streamlit as st
import streamlit.components.v1 as components
import os
import pandas as pd
import numpy as np
import subprocess
from datetime import datetime
from PIL import Image
from utils import pad_image, load_dicom_dataframe, load_dicom_dataframe_csv

st.set_page_config(layout="wide", page_title="DICOM Classifier")

n_cols = 4
n_rows = 3

# ---------------------------------------------------------------------------
# Label helpers
# ---------------------------------------------------------------------------

BUTTON_GROUPS = [
    ("T1W", ["T1W noFS noCE", "T1W FS noCE",  "T1W noFS CE",  "T1W FS CE"]),
    ("T2W", ["T2W noFS noCE", "T2W FS noCE",  "T2W noFS CE",  "T2W FS CE",
              "T2W STIR noCE", "T2W STIR CE"]),
    ("T2*", ["T2* noFS noCE", "T2* FS noCE",  "T2* noFS CE",  "T2* FS CE"]),
    ("PD",  ["PD noFS noCE",  "PD FS noCE",   "PD noFS CE",   "PD FS CE"]),
    ("Sp",  ["DWI", "Localizer", "Other", "To_review", "Zip/JPG"]),
]
BUTTON_LABELS = [lbl for _, grp in BUTTON_GROUPS for lbl in grp]

_DECODE_MAP = {
    "T1W noFS noCE":  ("T1W",       "N",         "N"),
    "T1W FS noCE":    ("T1W",       "Y",         "N"),
    "T1W noFS CE":    ("T1W",       "N",         "Y"),
    "T1W FS CE":      ("T1W",       "Y",         "Y"),
    "T2W noFS":       ("T2W",       "N",         "-"),
    "T2W FS":         ("T2W",       "Y",         "-"),
    "T2W noFS CE":    ("T2W",       "N",         "Y"),
    "T2W FS CE":      ("T2W",       "Y",         "Y"),
    "T2W STIR noCE":  ("T2W",  "Y-STIR",         "N"),
    "T2W STIR CE":    ("T2W",  "Y-STIR",         "Y"),
    "T2* noFS noCE":  ("T2*",       "N",         "N"),
    "T2* FS noCE":    ("T2*",       "Y",         "N"),
    "T2* noFS CE":    ("T2*",       "N",         "Y"),
    "T2* FS CE":      ("T2*",       "Y",         "Y"),
    "PD noFS noCE":   ("PD",        "N",         "N"),
    "PD FS noCE":     ("PD",        "Y",         "N"),
    "PD noFS CE":     ("PD",        "N",         "Y"),
    "PD FS CE":       ("PD",        "Y",         "Y"),
    "DWI":            ("DW",        "-",         "-"),
    "Localizer":      ("Localizer", "-",         "-"),
    "Other":          ("Other",     "-",         "-"),
    "To_review":      ("To_review", "To_review", "To_review"),
    "Zip/JPG":        ("Zip/JPG",   "-",         "-"),
}


def decode_selection(sel, orig_w, orig_fs, orig_c):
    return _DECODE_MAP.get(sel, (orig_w, orig_fs, orig_c))


def get_default_label(w, fs, c):
    w, fs, c = str(w).strip(), str(fs).strip(), str(c).strip()
    fs_str = "FS"   if fs == "Y" else "noFS"
    ce_str = "CE"   if c  == "Y" else "noCE"
    if w == "T1W":
        return f"T1W {fs_str} {ce_str}"
    if w == "T2W":
        return f"T2W {fs_str} {ce_str}"
    if w == "T2*":
        return f"T2* {fs_str} {ce_str}" if f"T2* {fs_str} {ce_str}" in BUTTON_LABELS else "T2* other"
    if w == "PD":
        return f"PD {fs_str} {ce_str}"
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
    """Format physics label as seq_fatsat_CE_acq, with nFS/nCE for absent fields."""
    def _v(key, empty="x"):
        v = str(row.get(key, "") or "").strip()
        return v if v and v.lower() not in ("nan", "none") else empty
    ce = "CE" if str(row.get("phys_contrast", "") or "").strip() == "Contrast" else "nCE"
    return f"{_v('phys_sequence')}_{_v('phys_fat_sat','nFS')}_{ce}_{_v('phys_acquisition')}"


def phys_ref_color(row, filter_w, filter_fs, filter_c) -> str:
    seq = str(row.get("phys_sequence", "") or "").strip()
    if not seq or seq.lower() in ("nan", "none", "x", ""):
        return "#888888"
    dcm_to_phys = {"T1W": "T1W", "T2W": "T2W", "T2*": "T2*", "PD": "PD", "DW": "DWI", "Other": "Localizer"}
    expected = dcm_to_phys.get(filter_w)
    if expected is None:
        return "#888888"
    mod_ok = seq == expected
    fs_ok  = (_phys_fat_yn(row.get("phys_fat_sat",  "")) == filter_fs) if filter_fs != "-" else True
    ce_ok  = (_phys_contrast_yn(row.get("phys_contrast", "")) == filter_c) if filter_c != "-" else True
    return "#44dd44" if (mod_ok and fs_ok and ce_ok) else "#ff4444"


def phys_badge(row, filter_w, filter_fs, filter_c) -> str:
    label = phys_ref_label(row)
    color = phys_ref_color(row, filter_w, filter_fs, filter_c)
    return (f'<span style="color:{color}; font-size:0.72em; '
            f'font-family:monospace; font-weight:bold">{label}</span>')


# ---------------------------------------------------------------------------
# Pill styling
# ---------------------------------------------------------------------------


def inject_pill_styles() -> None:
    st.markdown("""
<style>
button[data-testid="stBaseButton-pills"] {
    font-size: 0.5em !important;
    padding: 1px 6px !important;
    line-height: 1.2 !important;
    border-width: 1px !important;
}
</style>
""", unsafe_allow_html=True)
    components.html("""
<script>
(function() {
  function applyColors() {
    var d = window.parent.document;
    d.querySelectorAll('button[data-testid="stBaseButton-pills"]').forEach(function(btn) {
      var t = btn.textContent.trim();
      var c = t.startsWith('T1W') ? '#1565c0'
            : t.startsWith('T2W') ? '#2e7d32'
            : t.startsWith('T2*') ? '#e65100'
            : t.startsWith('PD')  ? '#6a1b9a'
            : t === 'DWI'         ? '#01579b'
            : '#455a64';
      btn.style.setProperty('border-color', c, 'important');
      btn.style.setProperty('color',        c, 'important');
    });
  }
  applyColors();
  new MutationObserver(applyColors).observe(
    window.parent.document.body, {childList: true, subtree: true}
  );
})();
</script>
""", height=0)


# ---------------------------------------------------------------------------
# Image generator
# ---------------------------------------------------------------------------

def run_img_generator(excel_path, img_base, dcm_root="", dcm_orig="/Project"):
    script = os.path.join(os.path.dirname(__file__), "img_generator.py")
    if not os.path.exists(script):
        st.error(f"img_generator.py not found: {script}")
        return False
    try:
        st.info("Generating missing images… please wait.")
        cmd = ["python3", script, "--excel", excel_path, "--out_dir", img_base]
        if dcm_root:
            cmd += ["--dcm_root", dcm_root, "--dcm_orig", dcm_orig]
        subprocess.run(cmd, capture_output=True, text=True, check=True)
        st.success("Images generated successfully.")
        return True
    except subprocess.CalledProcessError as e:
        st.error(f"Image generation failed:\n{e.stderr}")
        return False


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():

    st.sidebar.title("Configuration")

    path_results  = st.sidebar.text_input("1. Output folder",     value="/Proyecto/Results")
    path_img_base = st.sidebar.text_input("2. Image base folder", value="/Proyecto/IMG")
    uploaded_file = st.sidebar.file_uploader("3. Classifier CSV / Excel", type=["csv", "xlsx"])
    uploaded_ref  = st.sidebar.file_uploader("4. Physics-label CSV (optional)", type=["csv"])
    st.sidebar.divider()
    st.sidebar.subheader("DICOM path mapping")
    dcm_orig = st.sidebar.text_input("Original path prefix (in CSV)",  value="/Project",
                                     help="Prefix found in 'Nombre DICOM' column")
    dcm_root = st.sidebar.text_input("Actual path prefix (on server)", value="",
                                     help="Replace the prefix above with this real mount path")

    if uploaded_file is None:
        st.title("DICOM Classifier")
        st.info("Upload the classifier file from the sidebar to start.")
        return

    # ── Image generator ────────────────────────────────────────────────────
    st.sidebar.divider()
    if st.sidebar.button("Generate missing images",
                         help="Runs img_generator.py to create .png previews"):
        os.makedirs(path_results, exist_ok=True)
        ext     = ".csv" if uploaded_file.name.endswith(".csv") else ".xlsx"
        tmp     = os.path.join(path_results, f"_tmp_upload{ext}")
        with open(tmp, "wb") as f:
            f.write(uploaded_file.getbuffer())
        run_img_generator(tmp, path_img_base, dcm_root=dcm_root, dcm_orig=dcm_orig)
        if os.path.exists(tmp):
            os.remove(tmp)

    # ── Resume detection ───────────────────────────────────────────────────
    master_filename = f"Review_{uploaded_file.name.rsplit('.',1)[0]}.csv"
    master_filepath = os.path.join(path_results, master_filename)

    usar_progreso = False
    if os.path.exists(master_filepath):
        st.sidebar.warning(f"Previous progress found:\n**{master_filename}**")
        usar_progreso = st.sidebar.checkbox("Resume from progress", value=True)

    # ── Load data ──────────────────────────────────────────────────────────
    if "df_master" not in st.session_state or st.sidebar.button("Reload"):
        if usar_progreso and os.path.exists(master_filepath):
            df_temp = pd.read_csv(master_filepath, dtype=str).fillna("")
            st.toast("Previous progress loaded", icon="📁")
        else:
            if uploaded_file.name.endswith(".csv"):
                df_temp = load_dicom_dataframe_csv(uploaded_file)
            else:
                df_temp = load_dicom_dataframe(uploaded_file)

            for col, src in [
                ("viewed",        ""),
                ("Clase W Final",  "Predicción Clases W"),
                ("Clase FS Final", "Predicción Clases FS"),
                ("Clase C Final",  "Predicción Clases C"),
            ]:
                if col not in df_temp.columns:
                    df_temp[col] = df_temp[src].fillna("") if isinstance(src, str) and src in df_temp.columns else ""
            st.toast("File loaded from scratch", icon="📄")

        # Merge physics reference
        # columns to match the CLF prediction
        _phys_cols = ["phys_sequence", "phys_acquisition", "phys_fat_sat", "phys_contrast"]
        df_temp = df_temp.drop(columns=[c for c in _phys_cols if c in df_temp.columns])
        if uploaded_ref is not None:
            df_ref = pd.read_csv(uploaded_ref, dtype=str).fillna("").rename(columns={
                "subject": "Serie", "session": "Estudio", "scan": "Paciente"
            })
            join_keys = [k for k in ["Paciente", "Estudio", "Serie"]
                         if k in df_temp.columns and k in df_ref.columns]
            if join_keys:
                df_temp = df_temp.merge(
                    df_ref[join_keys + [c for c in _phys_cols if c in df_ref.columns]],
                    on=join_keys, how="left",
                )
        for c in _phys_cols:
            if c not in df_temp.columns:
                df_temp[c] = ""

        st.session_state.df_master       = df_temp
        st.session_state.master_filepath = master_filepath
        st.session_state.has_phys        = uploaded_ref is not None

    df       = st.session_state.df_master
    has_phys = st.session_state.get("has_phys", False)

    # ── Summary table ──────────────────────────────────────────────────────
    with st.expander("Summary table", expanded=False):
        grp      = ["Predicción Clases W", "Predicción Clases FS", "Predicción Clases C"]
        grp      = [c for c in grp if c in df.columns]
        resumen  = df.groupby(grp).size().reset_index(name="Total")
        reviewed = df[df["viewed"] == "X"].groupby(grp).size().reset_index(name="Reviewed")
        tabla    = pd.merge(resumen, reviewed, on=grp, how="left").fillna(0)
        tabla["Reviewed"]  = tabla["Reviewed"].astype(int)
        tabla["Remaining"] = tabla["Total"] - tabla["Reviewed"]
        st.dataframe(tabla, use_container_width=True, hide_index=True)

    # ── Filters ────────────────────────────────────────────────────────────
    st.sidebar.divider()
    st.sidebar.subheader("Sequence filters")

    second_review = st.sidebar.checkbox("Second review", value=False,
                                        help="Show only images previously marked as To_review")

    if second_review:
        val_w, val_fs, val_c = "To_review", "To_review", "To_review"
        mask = (
            (df["Clase W Final"]  == "To_review") &
            (df["Clase FS Final"] == "To_review") &
            (df["Clase C Final"]  == "To_review") &
            (df["viewed"]         == "X")
        )
    else:
        val_w = st.sidebar.selectbox("Modality (W)", ["T1W", "T2W", "T2*", "PD", "DW", "Other"])
        mask  = df["Predicción Clases W"] == val_w

        if val_w in ("DW", "Other"):
            val_fs, val_c = "-", "-"
        else:
            val_fs = st.sidebar.selectbox("Fat sat (FS)", ["N", "Y"])
            mask   = mask & (df["Predicción Clases FS"] == val_fs)
            val_c  = st.sidebar.selectbox("Contrast (C)", ["N", "Y"]) if val_w == "T1W" else "-"
            if val_c != "-":
                mask = mask & (df["Predicción Clases C"] == val_c)

        show_pending = st.sidebar.checkbox("Show only pending", value=True)
        if show_pending:
            mask = mask & (df["viewed"] != "X")

    df_filtered = df[mask].reset_index()

    # ── Main view ──────────────────────────────────────────────────────────
    if second_review:
        st.title("Second Review: To_review cases")
    else:
        st.title(f"Review: {val_w}  |  FS: {val_fs}  |  CE: {val_c}")
    if has_phys:
        st.caption("Physics ref shown below each image  •  green = agrees  •  red = disagrees  •  grey = no data")

    if df_filtered.empty:
        st.success("All reviewed for this filter, or no cases found.")
        return

    batch        = df_filtered.head(n_cols * n_rows)
    user_actions = {}

    inject_pill_styles()
    with st.form("batch_form"):

        cols = st.columns(n_cols)
        for i, (_, row) in enumerate(batch.iterrows()):
            orig_idx = row["index"]
            with cols[i % n_cols]:
                paciente = str(row.get("Paciente", ""))
                estudio  = str(row.get("Estudio",  ""))
                serie    = str(row.get("Serie",    ""))

                # here the order is only to adapt with current structure
                st.caption(paciente)

                img_path = os.path.join(path_img_base, serie, estudio, paciente, "Img.png")
                try:
                    img_pil    = Image.open(img_path).convert("L")
                    img_padded = pad_image(np.array(img_pil))
                    st.image(img_padded, use_container_width=True, clamp=True)
                except Exception as e:
                    st.error(f"Image not found:\n{serie[:15]}…\n{e}")

                if has_phys:
                    st.markdown(phys_badge(row, val_w, val_fs, val_c), unsafe_allow_html=True)

                orig_w  = row.get("Predicción Clases W",  "")
                orig_fs = row.get("Predicción Clases FS", "")
                orig_c  = row.get("Predicción Clases C",  "")
                if second_review:
                    initial = "To_review"
                else:
                    initial = get_default_label(orig_w, orig_fs, orig_c)

                grp_sels = [
                    st.pills(gname, glabels,
                             default=initial if initial in glabels else None,
                             key=f"p_{orig_idx}_{gname}",
                             label_visibility="collapsed")
                    for gname, glabels in BUTTON_GROUPS
                ]
                user_actions[orig_idx] = next(
                    (v for v in grp_sels if v is not None), None
                )

        if st.form_submit_button("Save & Next", use_container_width=True):
            for idx, sel in user_actions.items():
                fila    = st.session_state.df_master.loc[idx]
                orig_w  = fila.get("Predicción Clases W",  "")
                orig_fs = fila.get("Predicción Clases FS", "")
                orig_c  = fila.get("Predicción Clases C",  "")

                if not sel:
                    sel = get_default_label(orig_w, orig_fs, orig_c)

                w_f, fs_f, c_f = decode_selection(sel, orig_w, orig_fs, orig_c)

                st.session_state.df_master.at[idx, "viewed"]        = "X"
                st.session_state.df_master.at[idx, "Clase W Final"]  = w_f
                st.session_state.df_master.at[idx, "Clase FS Final"] = fs_f
                st.session_state.df_master.at[idx, "Clase C Final"]  = c_f

            def _save_csv(path: str) -> None:
                st.session_state.df_master.to_csv(path, index=False)
                os.chmod(path, 0o777)

            os.makedirs(path_results, exist_ok=True)
            os.chmod(path_results, 0o777)
            _save_csv(st.session_state.master_filepath)

            backup_dir = os.path.join(path_results, "Backups")
            os.makedirs(backup_dir, exist_ok=True)
            os.chmod(backup_dir, 0o777)
            _save_csv(os.path.join(
                backup_dir, f"Review_{val_w}_{datetime.now().strftime('%Y%m%d-%H%M')}.csv"
            ))
            st.session_state.do_scroll_top = True
            st.rerun()
    
        # Handle scroll after rerun - using components.html instead of markdown
        if st.session_state.get('do_scroll_top', False):
            # Use components.html instead of st.markdown
            components.html("""
                <script>
                // Try to find the anchor first
                var anchor = window.parent.document.getElementById('form-top');
                if (anchor) {
                    anchor.scrollIntoView({ behavior: 'smooth', block: 'start' });
                } else {
                    window.parent.scrollTo({ top: 0, behavior: 'smooth' });
                }
                </script>
            """, height=0)
            # Reset the flag
            st.session_state.do_scroll_top = False

if __name__ == "__main__":
    main()
