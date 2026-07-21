# nii2dicom — BONE-AI nii.gz → DICOM → Orthanc

Tooling to ingest the BONE-AI dataset (NIfTI images + segmentations) into an
Orthanc PACS so the OHIF **Bone-AI** mode can display and edit it.

## What it produces

`convert.py` turns each subject into one DICOM **Study** (subject-session), with
one **Series** per image and a **DICOM-SEG** series per segmentation. Key
properties that make it safe for a database:

- **Deterministic UIDs** (hashed from subject/session/series) → re-running is
  idempotent; re-upload overwrites instead of duplicating, and never orphans
  review data keyed to a StudyInstanceUID.
- **One shared FrameOfReferenceUID** per subject-session → segmentations align
  and cross-series tools work.
- Segmentation is reindexed onto the image voxel grid by world coordinates, so a
  seg stored with a flipped/permuted axis still overlays correctly.

Input layout expected:

```
<input>/<SUBJECT>/<SESSION>/<N_desc>/images.nii.gz
<input>/<SUBJECT>/<SESSION>/segmentation_history/segs/<N_desc>_seg.nii.gz
```

Output layout (mirrors the DICOM hierarchy + a manifest.csv index):

```
<output>/<SUBJECT>/<SESSION>/<NNN_desc>/0001.dcm ...     # MR series
<output>/<SUBJECT>/<SESSION>/<NNN_desc>_SEG/SEG.dcm      # DICOM-SEG
<output>/manifest.csv
```

## Setup (once)

```bash
python3 -m venv venv
./venv/bin/pip install numpy nibabel pydicom highdicom
```

## Scan shapes first (recommended for large datasets)

Before converting a big dataset, check which series are non-3-D (4-D multi-echo /
Dixon, or odd cases). Reads only NIfTI headers, so it's fast over thousands of files:

```bash
./venv/bin/python scan_shapes.py --input /path/to/filtered_nifti
# scan segmentations too:
./venv/bin/python scan_shapes.py --input /path/to/filtered_nifti --pattern '*.nii.gz'
```

It prints a summary (counts by dimensionality, 4-D volume counts) and writes
`shapes_report.csv`. 4-D series are handled by `convert.py`:

- default `--volumes first`: only the segmented/visible volume (frame 0) becomes a
  series — matches what you segmented in Slicer, no seg-less duplicates.
- `--volumes all`: every volume becomes a separate series (`_v0`, `_v1`, …), SEG on `_v0`.

## Convert

```bash
./venv/bin/python convert.py --input /path/to/data-bt-tmp --output ./dicom_out
# optional subject filter:
./venv/bin/python convert.py --input /path/to/data-bt-tmp --output ./dicom_out \
    --subjects BONE_AI_51 BONE_AI_52
```

By default (`--sessions segmented`) only sessions that have a segmentation to review
are converted — sessions with no `segmentation_history/segs/*_seg.nii.gz` are skipped
(they don't need a second check in OHIF). Use `--sessions all` to convert everything.

## Upload to Orthanc

```bash
# local Orthanc
./venv/bin/python upload.py --input ./dicom_out

# server Orthanc behind nginx + basic auth
./venv/bin/python upload.py --input ./dicom_out \
    --host YOUR_SERVER --port 443 --scheme https --base /orthanc \
    --user USER --password PASS
```

You can also drag `dicom_out/` into **Orthanc Explorer 2** (`/ui/app/`) instead of
using `upload.py`.

## Running a local Orthanc for dev (macOS)

```bash
colima start                         # provides the Docker engine
docker run -d --name ohif-orthanc \
  -p 8042:8042 -p 4242:4242 \
  -v "$PWD/orthanc.json:/etc/orthanc/orthanc.json:ro" \
  -v "$PWD/orthanc-db:/var/lib/orthanc/db" \
  jodogne/orthanc-plugins
```

The Orthanc config must include `"Plugins": ["/usr/share/orthanc/plugins",
"/usr/local/share/orthanc/plugins"]` or the DICOMweb endpoint returns 404.
