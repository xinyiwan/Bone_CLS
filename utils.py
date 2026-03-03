from pathlib import Path

def find_scan_dirs(datadir: Path):
    """Yield ``(subject, session, scan, scan_dir)`` for every folder with .dcm files.

    The path components relative to *datadir* are interpreted as:
        3+ levels deep : parts[0]=subject, parts[1]=session, parts[2]=scan
        2 levels deep  : parts[0]=subject, parts[1]=scan (session = scan name)
        1 level deep   : subject = scan name, session = ""
    """
    scan_dirs = sorted({f.parent for f in datadir.rglob("*.dcm") if f.is_file()})
    for scan_dir in scan_dirs:
        parts = scan_dir.relative_to(datadir).parts
        # if parts[0] is ADQUISICIONES, we skip this part
        if "ADQUISICIONES" in parts[0] or "test" in parts[0]:
            parts = parts[1:]
            
        if len(parts) >= 3:
            subject, session, scan = parts[0], parts[1], "/".join(parts[2:])
        elif len(parts) == 2:
            subject, session, scan = parts[0], "", parts[1]
        elif len(parts) == 1:
            subject, session, scan = parts[0], "", parts[0]
        else:
            subject, session, scan = "", "", ""
        yield subject, session, scan, scan_dir