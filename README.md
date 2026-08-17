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
| `working/vision_model/medgemma_pilot` | 3.11 | migrated to uv (GPU, see below) |
| `working/dicom_sort/Classifier_final` | 3.9–3.10 (TensorFlow 2.10) | still `requirements.txt` |
| `working/nnInteractive` | — | still `requirements.txt` |

### medgemma_pilot and CUDA

`torch` is pinned to `2.6.0` — the last cu124 build — because the GPU host runs
NVIDIA driver 550.144.03. Bumping to 2.7+ drops cu124 and targets CUDA 12.6/12.8,
which need driver >=560. The constraint is documented in `Dockerfile.hf`; keep the
two in sync.

The CUDA build comes from PyTorch's own index rather than PyPI, selected per
platform in `[tool.uv.sources]`:

| Platform | Resolves to |
| :--- | :--- |
| Linux x86_64 | `torch==2.6.0+cu124` from `download.pytorch.org/whl/cu124` |
| macOS arm64 | `torch==2.6.0` from PyPI (CPU/MPS, for local development) |

One lockfile covers both. `[tool.uv] environments` restricts resolution to those
two targets, since no matching torch build exists for Windows.

Downloading MedGemma weights needs a Hugging Face token with accepted access to
the gated [HAI-DEF license](https://huggingface.co/google/medgemma-1.5-4b-it).
Pass it via `HF_TOKEN` at build time — never commit it.

Where a subproject has a `Dockerfile`, keep its `requires-python` in sync with the
base image.

## What is committed

| Path | Committed | Why |
| :--- | :--- | :--- |
| `pyproject.toml` | yes | declares dependencies |
| `uv.lock` | yes | exact versions; makes environments reproducible across machines |
| `.python-version` | yes | interpreter pin |
| `<script>.py.lock` | yes | lock for a PEP 723 script |
| `.venv/` | **no** | machine-specific, rebuilt by `uv sync` |

`uv lock` resolves *universally* — one lockfile covers macOS and Linux, x86_64 and
arm64. A lock generated on a laptop reproduces identically on a Linux cluster with
no re-resolution. So moving to another machine is just:

```bash
git clone https://github.com/xinyiwan/Bone_CLS.git
cd Bone_CLS && uv sync
```

## Running on HPC (Snellius)

Install uv without root — it also downloads its own CPython, so no `module load python`:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh    # installs to ~/.local/bin
```

### Keep the cache off `$HOME`

The uv cache reaches tens of GB and hundreds of thousands of files once torch or
TensorFlow is involved. On HPC the *inode* quota usually bites before the size
quota. Add to `~/.bashrc`:

```bash
export UV_CACHE_DIR=/scratch-shared/$USER/uv-cache
```

Verify with `uv cache dir` — it must print the scratch path. uv silently ignores
misspelled variables, so a typo fails invisibly.

### Keep each `.venv` off `$HOME` -- one per project

Do **not** set `UV_PROJECT_ENVIRONMENT` globally in `~/.bashrc`. It is a single
absolute path applied to *every* project, so syncing a second project uninstalls
the first one's packages into the same directory. They cannot coexist.

Instead, symlink each project's `.venv` to its own scratch directory:

```bash
mkdir -p /scratch-shared/$USER/envs/bone-cls
ln -s /scratch-shared/$USER/envs/bone-cls .venv

mkdir -p /scratch-shared/$USER/envs/viewer-web
ln -s /scratch-shared/$USER/envs/viewer-web working/dicom_sort/viewer_web/.venv
```

Then run `uv sync` in each project directory as normal.

In `ln -s TARGET LINK`, the scratch **target** must exist (that is the `mkdir -p`);
the `.venv` **link** must *not* exist. Two ways this goes wrong silently:

> **Create the target directory before the symlink.** If the symlink dangles, uv
> deletes it and creates a real directory in its place -- putting the environment
> back on `$HOME`, which is what this avoids. An empty directory is enough.

> **Remove any existing `.venv` first.** If `.venv` is already a directory, `ln -s`
> does not fail -- it exits 0 and creates the link *inside* it
> (`.venv/viewer-web -> ...`). Run `rm -rf .venv` first; it is always rebuildable
> with `uv sync`.

Check the result with `ls -ld .venv`: it should start with `l` and show an arrow to
scratch. A leading `d` means the link did not take.

Scratch is purged periodically. Both the cache and the environments are
disposable; `uv sync` rebuilds them from `uv.lock`.

### SLURM jobs

Run `uv sync` on the **login node** before submitting, so jobs only *use* the
environment. This avoids array tasks hammering PyPI in parallel and does not
depend on compute nodes having outbound network access.

```bash
#!/bin/bash
#SBATCH --job-name=bone-cls
#SBATCH --time=01:00:00

export UV_CACHE_DIR=/scratch-shared/$USER/uv-cache

uv run --no-sync python working/analysis/compare.py
```

`--no-sync` makes the job fail immediately if the environment is missing, rather
than silently attempting a download mid-job.

Setting the exports in the job script (rather than relying on `~/.bashrc`) is
deliberate: many `.bashrc` files return early for non-interactive shells, so
anything below that guard never runs under SLURM.

### GPU builds

`working/vision_model/` and `working/segmentation/dot_sanity` need torch. A bare
`torch` dependency resolves to the default (often CPU) build. For the GPU nodes,
pin an explicit CUDA index with `[[tool.uv.index]]` plus platform markers rather
than relying on the default resolution. Not yet configured -- see the migration
table above.
