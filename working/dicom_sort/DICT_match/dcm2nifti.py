"""
Convert DICOM scans to NIfTI format.

Conversion is attempted with three tools in order:
    1. dcm2niix       (preferred – pip install dcm2niix)
    2. dicom2nifti    (fallback  – pip install dicom2nifti)
    3. SimpleITK      (last resort – pip install SimpleITK)

Expected directory structure (same as extract_headers.py):
    DATADIR/
    └── <subject>/
        └── <session>/
            └── <scan>/
                └── *.dcm

NIfTI files are written under OUTDIR mirroring the input hierarchy:
    OUTDIR/<subject>/<session>/<scan>.nii.gz

Set DATADIR and OUTDIR before running, then:
    python dcm2nifti.py
"""

from __future__ import annotations

import logging
import os
import shutil
from pathlib import Path
from Bone_CLS.working.dicom_sort.DICT_match.utils import find_scan_dirs

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

DATADIR = Path("/data")
OUTDIR = Path("/results")


# ---------------------------------------------------------------------------
# Conversion tool wrappers (adapted from xnattools/dcm2nifti.py)
# ---------------------------------------------------------------------------

def run_dcm2niix(dicom_dir, nifti_path, overwrite=False):
    # type: (Path, Path, bool) -> bool
    """Convert *dicom_dir* with dcm2niix.

    Returns True on success.
    """
    try:
        import dcm2niix as _dcm2niix
    except ImportError:
        logger.warning("dcm2niix not installed – skipping.")
        return False

    out_dir = str(nifti_path.parent)
    filename = nifti_path.name
    if filename.endswith(".nii.gz"):
        filename = filename[:-7]
        compress = "y"
    elif filename.endswith(".nii"):
        filename = filename[:-4]
        compress = "n"
    else:
        compress = "y"

    cmd = [
        "-9",          # maximum gz compression
        "-b", "n",     # no BIDS sidecar
        "-m", "y",     # merge 2D slices
        "-s", "n",     # single file output
        "-t", "n",     # no text output
        "-x", "n",     # no reorientation crop
        "-z", compress,
        "-o", out_dir,
        "-f", filename,
        str(dicom_dir),
    ]
    try:
        result = _dcm2niix.main(args=cmd, check=True, capture_output=True, text=True)
        if result != 0:
            logger.warning("dcm2niix returned non-zero for %s.", dicom_dir)
            return False
        for f in nifti_path.parent.glob("*.nii*"):
            os.chmod(f, 0o664)
        return True
    except Exception as exc:
        logger.warning("dcm2niix error: %s", exc)
        return False


def run_dicom2nifti(dicom_dir, nifti_path, overwrite=False):
    # type: (Path, Path, bool) -> bool
    """Convert *dicom_dir* with the dicom2nifti package.

    Returns True on success.
    """
    try:
        import dicom2nifti as _d2n
    except ImportError:
        logger.warning("dicom2nifti not installed – skipping.")
        return False

    out_dir = nifti_path.parent
    if out_dir.exists():
        shutil.rmtree(str(out_dir))
    out_dir.mkdir(parents=True)

    try:
        _d2n.convert_directory(
            str(dicom_dir), str(out_dir), compression=True, reorient=True
        )
    except Exception as exc:
        logger.warning("dicom2nifti error: %s", exc)
        return False

    output_files = sorted(out_dir.glob("*.nii.gz"))
    if not output_files:
        return False

    # Rename outputs to match the expected nifti_path name
    stem = nifti_path.name[:-7] if nifti_path.name.endswith(".nii.gz") else nifti_path.stem
    for n, f in enumerate(output_files):
        suffix = "" if len(output_files) == 1 else f"_{n}"
        dest = out_dir / f"{stem}{suffix}.nii.gz"
        f.rename(dest)
        os.chmod(dest, 0o664)

    return True


def run_manual_dcm2nifti(dicom_dir, nifti_path, overwrite=False):
    # type: (Path, Path, bool) -> bool
    """Convert *dicom_dir* to NIfTI using SimpleITK (last-resort fallback).

    Reads the DICOM series with SimpleITK's GDCM reader.  A Y/Z flip is
    applied to correct orientation issues that typically trigger this path.

    Returns True on success.
    """
    try:
        import SimpleITK as sitk
    except ImportError:
        logger.warning("SimpleITK not installed – skipping.")
        return False

    try:
        reader = sitk.ImageSeriesReader()
        series_ids = reader.GetGDCMSeriesIDs(str(dicom_dir))
        if not series_ids:
            logger.warning("SimpleITK: no DICOM series found in %s", dicom_dir)
            return False
        reader.SetFileNames(
            reader.GetGDCMSeriesFileNames(str(dicom_dir), series_ids[0])
        )
        image = reader.Execute()
        image = sitk.Flip(image, [False, True, True])
    except RuntimeError as exc:
        logger.warning("SimpleITK RuntimeError: %s", exc)
        return False
    except Exception as exc:
        logger.warning("SimpleITK error: %s", exc)
        return False

    nifti_path.parent.mkdir(parents=True, exist_ok=True)
    sitk.WriteImage(image, str(nifti_path))
    os.chmod(nifti_path, 0o664)
    return True


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

_TOOLS = (run_dcm2niix, run_dicom2nifti, run_manual_dcm2nifti)


def convert_scan(dicom_dir, nifti_path, overwrite=False):
    # type: (Path, Path, bool) -> bool
    """Try each conversion tool in sequence until one succeeds.

    Args:
        dicom_dir:  Directory containing .dcm files.
        nifti_path: Desired output NIfTI file path.
        overwrite:  If False, skip when *nifti_path* already exists.

    Returns:
        True on success or if the output exists and overwrite=False.
    """
    if not overwrite and nifti_path.exists():
        logger.info("Skipping %s (output already exists).", nifti_path)
        return True

    nifti_path.parent.mkdir(parents=True, exist_ok=True)

    for tool in _TOOLS:
        logger.info("  Trying %s …", tool.__name__)
        if tool(dicom_dir, nifti_path, overwrite=overwrite):
            logger.info("  → success with %s", tool.__name__)
            return True

    logger.error("All conversion tools failed for %s", dicom_dir)
    return False


def convert_all(datadir, outdir, overwrite=False):
    # type: (Path, Path, bool) -> None
    """Convert every DICOM scan under *datadir* to NIfTI under *outdir*.

    Output hierarchy mirrors input:
        ``outdir/<subject>/<session>/<scan>.nii.gz``

    Args:
        datadir:   Root directory containing subject folders.
        outdir:    Root output directory for NIfTI files.
        overwrite: Re-convert even when output files already exist.
    """
    n_ok, n_failed, n_skipped = 0, 0, 0

    for subject, session, scan, scan_dir in find_scan_dirs(datadir):
        # Flatten any sub-folder separators in the scan name
        scan_stem = scan.replace("/", "_")
        nifti_path = outdir / subject / session / scan_stem / "images.nii.gz"

        if not overwrite and nifti_path.exists():
            logger.info("Skipping %s/%s/%s (exists).", subject, session, scan)
            n_skipped += 1
            continue

        logger.info("Converting  %s / %s / %s …", subject, session, scan)
        if convert_scan(scan_dir, nifti_path, overwrite=overwrite):
            n_ok += 1
        else:
            n_failed += 1

    print(
        "\nDone: %d converted, %d skipped, %d failed." % (n_ok, n_skipped, n_failed)
    )


if __name__ == "__main__":
    convert_all(DATADIR, OUTDIR)
