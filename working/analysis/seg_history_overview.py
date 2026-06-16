import os
import glob
from pathlib import Path
from datetime import datetime
import shutil

base_path = "/home/ext_xinwan/Bone_AI/tmp_sorted_data"
pattern = os.path.join(base_path, "BONE_AI_*", "*", "segmentation_history")


def is_valid_seg_history(seg_dir):
    """
    Decide whether a segmentation_history folder is an *actual* history.

    A folder is considered NOT a real history when it has neither:
      - a non-empty 'segs' folder anywhere inside it, nor
      - a final segmentation file (*.nii or *.nii.gz) anywhere inside it.

    Args:
        seg_dir: Path to a segmentation_history folder.

    Returns:
        (is_valid, reason): bool and a short human-readable explanation.
    """
    seg_path = Path(seg_dir)

    has_segs = any(
        p.name == "segs" and p.is_dir() and any(p.iterdir())
        for p in seg_path.rglob("segs")
    )
    has_final = any(
        p.is_file() and (p.name.endswith(".nii") or p.name.endswith(".nii.gz"))
        for p in seg_path.rglob("*")
    )

    if not has_segs and not has_final:
        return False, "no 'segs' and no .nii/.nii.gz"
    reasons = []
    if has_segs:
        reasons.append("has segs")
    if has_final:
        reasons.append("has .nii")
    return True, ", ".join(reasons)


def copy_segmentation_histories(source_base_path, dest_base_path, dry_run=True):
    """
    Copy segmentation_history folders from source to destination maintaining the same structure.
    
    Args:
        source_base_path: Source directory path (e.g., ADQUISICIONES)
        dest_base_path: Destination directory path (e.g., tmp_sorted_data)
        dry_run: If True, only print what would be copied without actually copying
    """
    pattern = os.path.join(source_base_path, "BONE_AI_*", "*", "segmentation_history")
    seg_history_dirs = sorted(glob.glob(pattern))
    
    if not seg_history_dirs:
        print(f"No segmentation_history folders found under {source_base_path}")
        return 0
    
    copied_count = 0
    skipped_count = 0
    invalid_count = 0

    for src_seg_dir in seg_history_dirs:
        # Get the relative path from source_base_path
        src_path = Path(src_seg_dir)
        rel_path = src_path.relative_to(source_base_path)

        # Skip folders that are not actual segmentation histories
        valid, reason = is_valid_seg_history(src_path)
        if not valid:
            print(f"⊘ Skipping (not a real history: {reason}): {src_path}")
            invalid_count += 1
            continue

        # Build destination path
        dest_path = Path(dest_base_path) / rel_path

        # Check if destination already exists
        if dest_path.exists():
            print(f"⚠ Skipping (already exists): {dest_path}")
            skipped_count += 1
            continue
        
        if dry_run:
            print(f"[DRY RUN] Would copy: {src_path} -> {dest_path}")
            copied_count += 1
        else:
            try:
                # Create parent directories if they don't exist
                dest_path.parent.mkdir(parents=True, exist_ok=True)
                
                # Copy the entire directory
                shutil.copytree(src_path, dest_path)
                print(f"✓ Copied: {src_path} -> {dest_path}")
                copied_count += 1
            except Exception as e:
                print(f"✗ Error copying {src_path}: {e}")
    
    print(
        f"\nSummary: {copied_count} folders copied, "
        f"{skipped_count} skipped (already exist), "
        f"{invalid_count} skipped (not real histories)"
    )
    return copied_count

seg_history_dirs = sorted(glob.glob(pattern))

if not seg_history_dirs:
    print(f"No segmentation_history folders found under {base_path}")
    exit()

rows = []
for seg_dir in seg_history_dirs:
    p = Path(seg_dir)
    subject = p.parent.name
    bone_ai_folder = p.parent.parent.name

    valid, reason = is_valid_seg_history(p)
    status = "valid" if valid else "EMPTY"
    rows.append((bone_ai_folder, subject, status, reason))

# Print table
col_widths = [
    max(len("Scanner"), max(len(r[0]) for r in rows)),
    max(len("Subject"), max(len(r[1]) for r in rows)),
    max(len("Status"), max(len(r[2]) for r in rows)),
    max(len("Reason"), max(len(r[3]) for r in rows)),
]


sep = "  ".join("-" * w for w in col_widths)

n_valid = sum(1 for r in rows if r[2] == "valid")
n_invalid = len(rows) - n_valid

print(f"\nSegmentation history overview  ({len(rows)} folders)\n")
print(
    f"{'Scanner':<{col_widths[0]}}  "
    f"{'Subject':<{col_widths[1]}}  "
    f"{'Status':<{col_widths[2]}}  "
    f"{'Reason':<{col_widths[3]}}"
)
print(sep)
for r in rows:
    print(
        f"{r[0]:<{col_widths[0]}}  "
        f"{r[1]:<{col_widths[1]}}  "
        f"{r[2]:<{col_widths[2]}}  "
        f"{r[3]:<{col_widths[3]}}"
    )

print(
    f"\nTotal folders: {len(rows)}  |  "
    f"real histories: {n_valid}  |  empty/not real: {n_invalid}"
)


# Ask user if they want to copy files
print("\n" + "="*60)
copy_choice = input("Do you want to copy segmentation_history folders to tmp_sorted_data? (yes/no): ").lower()

if copy_choice in ['yes', 'y']:
    dest_base = "/working/Bone_AI/tmp_sorted_data"
    
    # First do a dry run to show what would be copied
    print("\n--- DRY RUN (preview) ---")
    copy_segmentation_histories(base_path, dest_base, dry_run=True)
    
    # Ask for confirmation
    confirm = input("\nProceed with actual copy? (yes/no): ").lower()
    if confirm in ['yes', 'y']:
        print("\n--- ACTUAL COPY ---")
        copy_segmentation_histories(base_path, dest_base, dry_run=False)
    else:
        print("Copy cancelled.")
else:
    print("Copy skipped.")