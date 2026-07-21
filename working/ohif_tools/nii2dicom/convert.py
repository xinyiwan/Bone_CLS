"""
Convert the BONE-AI nii.gz dataset into an organized tree of DICOM MR series +
DICOM-SEG, ready to upload to Orthanc (local or server).

USAGE
-----
    python convert.py --input  /path/to/data-bt-tmp \
                      --output /path/to/dicom_out \
                      [--subjects BONE_AI_51 BONE_AI_52]   # optional filter

INPUT layout (current dataset)
    <input>/<SUBJECT>/<SESSION>/<N_desc>/images.nii.gz
    <input>/<SUBJECT>/<SESSION>/segmentation_history/segs/<N_desc>_seg.nii.gz

OUTPUT layout (mirrors the DICOM hierarchy -> easy to browse / upload)
    <output>/<SUBJECT>/<SESSION>/<NNN_desc>/0001.dcm ...      (MR series)
    <output>/<SUBJECT>/<SESSION>/<NNN_desc>_SEG/SEG.dcm       (DICOM-SEG)
    <output>/manifest.csv                                     (index of everything)

WHY THIS IS "ORTHANC-COMPATIBLE"
    Orthanc files data by DICOM UID, not folder path. So the important part is the
    tags we set, not the tree:
      * DETERMINISTIC UIDs (hashed from subject/session/series) -> re-running is
        idempotent; re-upload overwrites instead of duplicating, and never orphans
        review data keyed to a StudyInstanceUID.
      * One Study per subject-session, one SeriesInstanceUID per series, a SEG series
        per segmented image, all sharing ONE FrameOfReferenceUID per subject-session
        so segmentations align and cross-series tools work.
    The folder tree just makes the same data pleasant for humans and upload tools.

GEOMETRY NOTES
    NIfTI is RAS+, DICOM is LPS: FLIP = diag(-1,-1,1). Array [i,j,k] -> frame k is
    arr[:,:,k].T (i->column, j->row). A seg may be stored on a different slice
    ordering than its image; we reindex the mask onto the image voxel grid by world
    coordinates (handles arbitrary flips/permutations).
"""
import os
import re
import csv
import argparse
import numpy as np
import nibabel as nib
import highdicom as hd
from pydicom import Dataset
from pydicom.dataset import FileMetaDataset
from pydicom.uid import (
    generate_uid, ExplicitVRLittleEndian, MRImageStorage,
    PYDICOM_IMPLEMENTATION_UID,
)
from pydicom.sr.coding import Code

FLIP = np.diag([-1.0, -1.0, 1.0, 1.0])  # RAS -> LPS
# Stable namespace prefix for our deterministic UIDs (valid pydicom root prefix).
UID_PREFIX = "1.2.826.0.1.3680043.10.1338."


def det_uid(*parts):
    """Deterministic DICOM UID from string parts (idempotent re-ingestion)."""
    return generate_uid(prefix=UID_PREFIX, entropy_srcs=[str(p) for p in parts])


def series_number_of(name):
    m = re.match(r"^(\d+)", name)
    return int(m.group(1)) if m else 0


def clean_desc(name):
    """'3_AxPD-fatsat' -> 'AxPD-fatsat' (drop the leading 'N_')."""
    return re.sub(r"^\d+_", "", name)


def study_date_of(session):
    m = re.search(r"(\d{8})", session)
    return m.group(1) if m else ""


def session_has_segmentation(subj_sess, seg_dir_rel="segmentation_history/segs"):
    """True if the session has at least one <N_desc>_seg.nii.gz to review."""
    seg_dir = os.path.join(subj_sess, seg_dir_rel)
    if not os.path.isdir(seg_dir):
        return False
    return any(f.endswith("_seg.nii.gz") for f in os.listdir(seg_dir))


def reindex_seg_to_image(seg_arr, A_seg, img_shape, A_img):
    """Nearest-neighbour resample seg onto the image voxel grid via world coords."""
    M = np.linalg.inv(A_img) @ A_seg
    Minv = np.linalg.inv(M)
    ni, nj, nk = img_shape
    ii, jj, kk = np.meshgrid(np.arange(ni), np.arange(nj), np.arange(nk), indexing="ij")
    coords = np.stack([ii, jj, kk, np.ones_like(ii)], 0).reshape(4, -1)
    sc = np.rint(Minv @ coords).astype(int)[:3]
    sni, snj, snk = seg_arr.shape
    valid = (
        (sc[0] >= 0) & (sc[0] < sni)
        & (sc[1] >= 0) & (sc[1] < snj)
        & (sc[2] >= 0) & (sc[2] < snk)
    )
    out = np.zeros(ni * nj * nk, dtype=seg_arr.dtype)
    out[valid] = seg_arr[sc[0, valid], sc[1, valid], sc[2, valid]]
    return out.reshape(ni, nj, nk)


def load_image_volumes(img_path):
    """Return a list of (volume3d, nii_affine, suffix).

    dcm2niix writes 4-D NIfTI (X, Y, Z, T) for multi-volume series (dual-echo,
    in/opposed-phase Dixon, dynamics, ...). We collapse trailing singleton dims
    and split any real 4th (volume) axis into separate 3-D volumes, each sharing
    the same spatial affine — so nothing is dropped and geometry stays correct.
    """
    img = nib.load(img_path)
    arr = np.asanyarray(img.dataobj)
    while arr.ndim > 3 and arr.shape[-1] == 1:
        arr = arr[..., 0]
    if arr.ndim == 3:
        return [(arr, img.affine, "")]
    if arr.ndim == 4:
        return [(arr[..., t], img.affine, f"_v{t}") for t in range(arr.shape[3])]
    raise ValueError(f"Unsupported image ndim={arr.ndim} ({arr.shape}) for {img_path}")


def build_mr_series(arr, nii_affine, *, patient_id, study_uid, study_date, study_desc,
                    series_uid, series_number, series_desc, frame_of_reference_uid):
    if arr.dtype != np.uint16:
        arr = np.clip(arr, 0, None).astype(np.uint16)
    ni, nj, nk = arr.shape
    A = FLIP @ nii_affine

    col_i = A[:3, 0]; sp_i = float(np.linalg.norm(col_i)); e_i = col_i / sp_i
    col_j = A[:3, 1]; sp_j = float(np.linalg.norm(col_j)); e_j = col_j / sp_j
    sp_k = float(np.linalg.norm(A[:3, 2]))
    iop = [*e_i.tolist(), *e_j.tolist()]
    pixel_spacing = [round(sp_j, 6), round(sp_i, 6)]

    lo, hi = np.percentile(arr[arr > 0], [1, 99]) if (arr > 0).any() else (0, 1)
    wc = float((hi + lo) / 2); ww = float(max(hi - lo, 1))

    datasets = []
    for k in range(nk):
        frame = np.ascontiguousarray(arr[:, :, k].T)
        ipp = (A @ np.array([0, 0, k, 1.0]))[:3]
        sop_uid = det_uid(series_uid, "instance", k)

        fm = FileMetaDataset()
        fm.MediaStorageSOPClassUID = MRImageStorage
        fm.MediaStorageSOPInstanceUID = sop_uid
        fm.TransferSyntaxUID = ExplicitVRLittleEndian
        fm.ImplementationClassUID = PYDICOM_IMPLEMENTATION_UID

        ds = Dataset()
        ds.file_meta = fm
        ds.is_little_endian = True
        ds.is_implicit_VR = False

        ds.SOPClassUID = MRImageStorage
        ds.SOPInstanceUID = sop_uid
        ds.Modality = "MR"
        ds.PatientName = patient_id
        ds.PatientID = patient_id
        ds.PatientBirthDate = ""
        ds.PatientSex = ""
        ds.StudyInstanceUID = study_uid
        ds.StudyDate = study_date
        ds.StudyTime = "000000"
        ds.StudyID = "1"
        ds.AccessionNumber = ""
        ds.StudyDescription = study_desc
        ds.SeriesInstanceUID = series_uid
        ds.SeriesNumber = series_number
        ds.SeriesDescription = series_desc
        ds.FrameOfReferenceUID = frame_of_reference_uid
        ds.InstanceNumber = k + 1

        ds.ImageOrientationPatient = [round(v, 8) for v in iop]
        ds.ImagePositionPatient = [round(float(v), 6) for v in ipp]
        ds.SliceLocation = round(float(np.dot(ipp, np.cross(e_i, e_j))), 6)
        ds.PixelSpacing = pixel_spacing
        ds.SliceThickness = round(sp_k, 6)
        ds.SpacingBetweenSlices = round(sp_k, 6)

        ds.SamplesPerPixel = 1
        ds.PhotometricInterpretation = "MONOCHROME2"
        ds.Rows = nj
        ds.Columns = ni
        ds.BitsAllocated = 16
        ds.BitsStored = 16
        ds.HighBit = 15
        ds.PixelRepresentation = 0
        ds.WindowCenter = round(wc, 2)
        ds.WindowWidth = round(ww, 2)
        ds.PixelData = frame.tobytes()

        datasets.append(ds)
    return datasets


def build_seg(source_datasets, seg_path, img_shape, img_affine, *, series_uid,
              series_number, series_desc, segment_label):
    seg = nib.load(seg_path)
    seg_arr = np.asanyarray(seg.dataobj)
    while seg_arr.ndim > 3 and seg_arr.shape[-1] == 1:
        seg_arr = seg_arr[..., 0]
    if seg_arr.ndim != 3:
        print(f"    ! skipping SEG (unexpected ndim={seg_arr.ndim}): {seg_path}")
        return None, 0
    seg_on_img = reindex_seg_to_image(seg_arr, seg.affine, img_shape, img_affine)
    seg_on_img = (seg_on_img > 0).astype(np.uint8)
    if seg_on_img.sum() == 0:
        return None, 0

    nk = img_shape[2]
    frames = np.stack([seg_on_img[:, :, k].T for k in range(nk)], axis=0)

    seg_desc = hd.seg.SegmentDescription(
        segment_number=1,
        segment_label=segment_label,
        segmented_property_category=Code("91723000", "SCT", "Anatomical Structure"),
        segmented_property_type=Code("272673000", "SCT", "Bone"),
        algorithm_type=hd.seg.SegmentAlgorithmTypeValues.SEMIAUTOMATIC,
        algorithm_identification=hd.AlgorithmIdentificationSequence(
            name="BoneAI", version="0.1", family=Code("113026", "DCM", "Region growing")
        ),
    )

    seg_ds = hd.seg.Segmentation(
        source_images=source_datasets,
        pixel_array=frames,
        segmentation_type=hd.seg.SegmentationTypeValues.BINARY,
        segment_descriptions=[seg_desc],
        series_instance_uid=series_uid,
        series_number=series_number,
        sop_instance_uid=det_uid(series_uid, "seg-instance"),
        instance_number=1,
        manufacturer="BoneAI",
        manufacturer_model_name="nii2dicom",
        software_versions="0.1",
        device_serial_number="0001",
        series_description=series_desc,
    )
    return seg_ds, int(frames.sum())


def convert_subject(input_root, output_root, subject, session, manifest,
                    seg_dir_rel="segmentation_history/segs", volumes_mode="first"):
    subj_sess = os.path.join(input_root, subject, session)
    study_uid = det_uid(subject, session, "study")
    for_uid = det_uid(subject, session, "frameofref")
    study_date = study_date_of(session)

    seg_dir = os.path.join(subj_sess, seg_dir_rel)
    series_dirs = sorted(
        d for d in os.listdir(subj_sess)
        if os.path.isdir(os.path.join(subj_sess, d))
        and os.path.exists(os.path.join(subj_sess, d, "images.nii.gz"))
    )

    n_series = n_seg = 0
    for name in series_dirs:
        img_path = os.path.join(subj_sess, name, "images.nii.gz")
        snum = series_number_of(name)
        desc = clean_desc(name)
        volumes = load_image_volumes(img_path)
        if len(volumes) > 1:
            if volumes_mode == "first":
                print(f"    * {name}: 4D ({len(volumes)} vols) -> keeping first only "
                      f"(the segmented/visible one; use --volumes all to keep all)")
                volumes = [(volumes[0][0], volumes[0][1], "")]  # clean suffix, looks like a normal series
            else:
                print(f"    * {name}: 4D image -> {len(volumes)} volumes, split into separate series")
        seg_path = os.path.join(seg_dir, f"{name}_seg.nii.gz")
        has_seg = os.path.exists(seg_path)

        for t, (vol, aff, suffix) in enumerate(volumes):
            vseries_num = snum if suffix == "" else snum * 100 + t
            series_uid = det_uid(subject, session, name, "image", suffix)
            datasets = build_mr_series(
                vol, aff, patient_id=subject, study_uid=study_uid, study_date=study_date,
                study_desc=session, series_uid=series_uid, series_number=vseries_num,
                series_desc=f"{name}{suffix}", frame_of_reference_uid=for_uid,
            )
            series_folder = os.path.join(output_root, subject, session, f"{snum:03d}_{desc}{suffix}")
            os.makedirs(series_folder, exist_ok=True)
            for ds in datasets:
                ds.save_as(os.path.join(series_folder, f"{ds.InstanceNumber:04d}.dcm"),
                           enforce_file_format=True)
            n_series += 1
            manifest.append([subject, session, "MR", vseries_num, f"{name}{suffix}",
                             os.path.relpath(series_folder, output_root),
                             study_uid, series_uid, len(datasets)])

            # Attach the segmentation to the first volume only (mask is one 3D volume).
            if has_seg and t == 0:
                seg_series_uid = det_uid(subject, session, name, "seg")
                seg_ds, _ = build_seg(
                    datasets, seg_path, vol.shape, aff, series_uid=seg_series_uid,
                    series_number=vseries_num + 2000, series_desc=f"{name} SEG",
                    segment_label="Lesion",
                )
                if seg_ds is not None:
                    seg_folder = os.path.join(output_root, subject, session, f"{snum:03d}_{desc}_SEG")
                    os.makedirs(seg_folder, exist_ok=True)
                    seg_ds.save_as(os.path.join(seg_folder, "SEG.dcm"), enforce_file_format=True)
                    n_seg += 1
                    manifest.append([subject, session, "SEG", vseries_num + 1000, f"{name} SEG",
                                     os.path.relpath(seg_folder, output_root),
                                     study_uid, seg_series_uid, 1])
    return study_uid, n_series, n_seg


def main():
    ap = argparse.ArgumentParser(description="Convert BONE-AI nii.gz -> DICOM for Orthanc.")
    ap.add_argument("--input", required=True, help="dataset root (contains BONE_AI_* subjects)")
    ap.add_argument("--output", default=os.path.join(os.path.dirname(__file__), "dicom_out"),
                    help="output DICOM tree (default: ./dicom_out)")
    ap.add_argument("--subjects", nargs="*", default=None,
                    help="optional subject filter, e.g. --subjects BONE_AI_51 BONE_AI_52")
    ap.add_argument("--volumes", choices=["first", "all"], default="first",
                    help="for 4D (multi-echo/Dixon) series: 'first' keeps only the "
                         "segmented/visible volume (default), 'all' emits every volume "
                         "as a separate series")
    ap.add_argument("--sessions", choices=["segmented", "all"], default="segmented",
                    help="'segmented' (default) converts only sessions that have a "
                         "segmentation to review; 'all' converts every session")
    args = ap.parse_args()

    input_root = os.path.abspath(args.input)
    output_root = os.path.abspath(args.output)
    os.makedirs(output_root, exist_ok=True)

    subjects = sorted(
        s for s in os.listdir(input_root)
        if os.path.isdir(os.path.join(input_root, s)) and s.startswith("BONE_AI")
        and (args.subjects is None or s in args.subjects)
    )

    manifest = []
    n_converted = n_skipped = 0
    print(f"input : {input_root}\noutput: {output_root}\nsessions: {args.sessions}\n")
    for subject in subjects:
        subp = os.path.join(input_root, subject)
        for session in sorted(d for d in os.listdir(subp) if os.path.isdir(os.path.join(subp, d))):
            subj_sess = os.path.join(subp, session)
            if args.sessions == "segmented" and not session_has_segmentation(subj_sess):
                print(f"{subject}/{session}: skipped (no segmentation to review)")
                n_skipped += 1
                continue
            study_uid, n_series, n_seg = convert_subject(
                input_root, output_root, subject, session, manifest,
                volumes_mode=args.volumes)
            print(f"{subject}/{session}: {n_series} MR series, {n_seg} SEG   study={study_uid}")
            n_converted += 1

    manifest_path = os.path.join(output_root, "manifest.csv")
    with open(manifest_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["subject", "session", "modality", "series_number", "series_desc",
                    "folder", "study_instance_uid", "series_instance_uid", "num_instances"])
        w.writerows(manifest)
    print(f"\nConverted {n_converted} session(s), skipped {n_skipped}.")
    print(f"Wrote manifest: {manifest_path}  ({len(manifest)} series rows)")


if __name__ == "__main__":
    main()
