import os
import json
import pandas as pd
import logging
from tqdm import tqdm
import numpy as np
import argparse
import concurrent.futures
from functools import partial
import pydicom
from PIL import Image


def save_img(imgs_folder: str, dcm_file: str) -> dict | None:
    """
    Generate Img.png for one DICOM file.

    dcm_file    – full path to the .dcm file (from 'Nombre DICOM' column)
    imgs_folder – base output directory; output is imgs_folder/patient/study/series/Img.png
    """
    try:
        # Derive patient/study/series from the last 3 path components before the filename
        parts = os.path.normpath(dcm_file).split(os.sep)
        patient, study, serie = parts[-4], parts[-3], parts[-2]
        img_path = os.path.join(imgs_folder, patient, study, serie)

        # Skip if already generated
        img_file = os.path.join(img_path, "Img.png")
        if os.path.exists(img_file):
            return {"status": "skipped", "path": img_file}

        ds   = pydicom.dcmread(dcm_file)
        data = ds.pixel_array.astype(np.float32)

        slope     = float(getattr(ds, "RescaleSlope",     1))
        intercept = float(getattr(ds, "RescaleIntercept", 0))
        data = data * slope + intercept

        val_low, val_high = np.percentile(data, 1), np.percentile(data, 99)
        data = np.clip(data, val_low, val_high)

        if val_high != val_low:
            img_uint8 = ((data - val_low) / (val_high - val_low) * 255).astype(np.uint8)
        else:
            img_uint8 = np.zeros(data.shape, dtype=np.uint8)

        os.makedirs(img_path, exist_ok=True)
        Image.fromarray(img_uint8).save(img_file)
        return {"status": "success", "path": img_file}

    except Exception as e:
        logging.error(f"Error processing {dcm_file}: {e}")
        return None


def main():
    parser = argparse.ArgumentParser(description="Generate PNG previews from DICOM files")
    parser.add_argument("--excel",    required=True,  help="Path to classifier CSV or Excel file")
    parser.add_argument("--out_dir",  required=True,  help="Base directory for saving generated images")
    parser.add_argument("--dcm_col",  default="Nombre DICOM",
                        help="Column containing the DICOM file path (default: 'Nombre DICOM')")
    parser.add_argument("--dcm_root", default="",
                        help="Replace the path prefix in dcm_col with this root "
                             "(e.g. /mnt/rimp/PROJECTS replaces /Project)")
    parser.add_argument("--dcm_orig", default="/Project",
                        help="Original prefix to replace in dcm_col paths (default: /Project)")
    args = parser.parse_args()

    path_excel = args.excel
    path_out   = args.out_dir
    dcm_col    = args.dcm_col
    dcm_root   = args.dcm_root.rstrip("/")
    dcm_orig   = args.dcm_orig.rstrip("/")

    # Column names fallback (patient/study/series used only when dcm_col absent)
    config_file = os.path.join(
        os.environ.get("CONFIG_PATH", "/Parameters_config"),
        os.environ.get("CONFIG_FILE",  "parameter_configuration.json"),
    )
    cfg = {}
    try:
        with open(config_file) as f:
            cfg = json.load(f).get("VIEWER", {})
    except Exception:
        pass

    patient_col = cfg.get("patient_column", "Paciente")
    study_col   = cfg.get("study_column",   "Estudio")
    serie_col   = cfg.get("serie_column",   "Serie")

    print(f"Loading file: {path_excel}")
    if path_excel.lower().endswith(".csv"):
        df = pd.read_csv(path_excel, dtype=str).fillna("")
    else:
        df = pd.read_excel(path_excel, dtype=str).fillna("")

    def build_path(row):
        # Prefer the full DICOM path column when available
        if dcm_col in df.columns:
            p = str(row[dcm_col]).strip()
            # Replace original prefix with the real mount root if provided
            if dcm_root and p.startswith(dcm_orig):
                p = dcm_root + p[len(dcm_orig):]
            return p
        # Fallback: build from patient / study / series columns
        for col in [patient_col, study_col, serie_col]:
            if col not in df.columns:
                raise ValueError(f"Column '{col}' not found. Available: {df.columns.tolist()}")
        base = os.path.join(
            os.environ.get("INPUT_PATH", "/Proyecto"),
            str(row[patient_col]), str(row[study_col]), str(row[serie_col]),
        )
        return base  # save_img will find the first .dcm inside

    df["_path"] = df.apply(build_path, axis=1)
    paths = df["_path"].unique().tolist()
    print(f"Unique series to process: {len(paths)}")

    worker = partial(save_img, path_out)
    results = {"success": 0, "skipped": 0, "failed": 0}

    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        futures = {executor.submit(worker, p): p for p in paths}
        for future in tqdm(concurrent.futures.as_completed(futures),
                           total=len(paths), desc="Processing DICOM"):
            try:
                r = future.result()
                if r:
                    results[r["status"]] = results.get(r["status"], 0) + 1
                else:
                    results["failed"] += 1
            except Exception as exc:
                print(f"Worker failed: {exc}")
                results["failed"] += 1

    print(f"Done — success: {results['success']}  "
          f"skipped: {results['skipped']}  failed: {results['failed']}")


if __name__ == "__main__":
    main()
