import os
import glob
from pathlib import Path
from datetime import datetime
import shutil

base_path = "/home/ext_xinwan/Bone_AI/tmp_sorted_data"
pattern = os.path.join(base_path, "BONE_AI_*", "*", "segmentation_history")


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
    
    for src_seg_dir in seg_history_dirs:
        # Get the relative path from source_base_path
        src_path = Path(src_seg_dir)
        rel_path = src_path.relative_to(source_base_path)
        
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
    
    print(f"\nSummary: {copied_count} folders copied, {skipped_count} skipped")
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

    rows.append((bone_ai_folder, subject))

# Print table
col_widths = [
    max(len("Scanner"), max(len(r[0]) for r in rows)),
    max(len("Subject"), max(len(r[1]) for r in rows)),
]


sep = "  ".join("-" * w for w in col_widths)

print(f"\nSegmentation history overview  ({len(rows)} subjects)\n")
print(sep)
for r in rows:
    print(
        f"{r[0]:<{col_widths[0]}}  "
        f"{r[1]:<{col_widths[1]}}  "
    )

print(f"\nTotal subjects with segmentation history: {len(rows)}")


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