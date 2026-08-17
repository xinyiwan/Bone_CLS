# Bone_CLS

Research scripts for bone imaging: DICOM sorting, segmentation, and classification.

## Environments

This repo is **not** a single installable package — `working/` holds task scripts whose
dependencies genuinely conflict (TensorFlow 2.10 vs. torch, pandas 2.0 vs. 2.3,
pydicom 2.4 vs. 3.0). Rather than force one environment, it uses three tiers.

### 1. Root environment — the shared analysis stack

Covers `working/analysis`, `working/data_ov`, `working/clinical_info`,
`working/seg_model`, and `working/ohif_tools`. These share a compatible stack, so
one environment serves them all (and keeps cross-directory imports working).

```bash
uv sync                                        # create .venv from uv.lock
uv run python working/analysis/compare.py --help
```

Add a dependency with `uv add <pkg>`; it updates `pyproject.toml` and `uv.lock`.

### 2. Self-contained scripts — inline metadata (PEP 723)

Scripts with heavy or incompatible dependencies declare them in a `# /// script`
header instead of joining the root environment. `uv run` builds an isolated,
cached environment for the script automatically — no activation, no `uv sync`.

```bash
uv run working/segmentation/run_totalseg.py --help   # pulls TotalSegmentator, not into .venv
```

Pinned in `<script>.py.lock` next to the script. Refresh with `uv lock --script <path>`.

### 3. Subprojects — separate apps with their own containers

Each has its own `pyproject.toml`, `uv.lock`, and `.venv`. Run uv commands from
inside the subproject directory.

```bash
cd working/dicom_sort/viewer_web
uv sync
uv run streamlit run main.py
```

| Subproject | Python | Status |
| :--- | :--- | :--- |
| `working/dicom_sort/viewer_web` | 3.11 | migrated to uv |
| `working/dicom_sort/Classifier_final` | 3.9–3.10 (TensorFlow 2.10) | still `requirements.txt` |
| `working/vision_model/medgemma_pilot` | — | still `requirements.txt` |
| `working/nnInteractive` | — | still `requirements.txt` |

Where a subproject has a `Dockerfile`, keep its `requires-python` in sync with the
base image.
