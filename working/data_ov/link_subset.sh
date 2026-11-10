#!/bin/bash
# Make a folder of symlinks to the subset subjects, to use as "Data root" in SlicerNNInteractive.
# Slicer then lists only the subset; masks are saved into the source tree through the links.
# Usage: link_subset.sh <subset.csv> <source_root> <link_dir> [all]   (default: only subjects without a mask)
set -euo pipefail
csv=$1; src=$2; out=$3; mode=${4:-todo}
mkdir -p "$out"
python3 - "$csv" "$mode" <<'P' | while read -r s; do
import csv, sys
for r in csv.DictReader(open(sys.argv[1])):
    if sys.argv[2] == "all" or r["has_seg"] == "False": print(r["subject_code"].strip())
P
  [ -d "$src/$s" ] && ln -sfn "$src/$s" "$out/$s" || echo "missing: $s" >&2
done
echo "$(ls "$out" | wc -l) subjects linked in $out"
