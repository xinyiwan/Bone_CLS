"""Repair shap.png files corrupted by the old NaN->null replacement in bti_predict.py.

Each 'NaN' inside the base64 image became 'null' (+1 char, shifts all later bytes) and the string
was then cut to a multiple of 4. Here: re-encode the saved bytes, try turning each 'null' back into
'NaN' (2^m combinations, m is small) and keep the combination whose PNG chunk CRCs and zlib stream
are all valid; the lost last bytes belong to the constant IEND chunk, which is re-appended.
Unrepairable files are left as they are.   python fix_shap.py output/BTI/prediction [more dirs]
"""
import base64, itertools, struct, sys, zlib
from pathlib import Path

IEND = b"\x00\x00\x00\x00IEND\xaeB`\x82"


def parse(d):
    """Return the PNG rebuilt up to the last valid IDAT + IEND, or None if any chunk is bad."""
    if d[:8] != b"\x89PNG\r\n\x1a\n":
        return None
    pos, idat, end, wh = 8, b"", 8, None
    while pos + 8 <= len(d):
        n, t = struct.unpack(">I4s", d[pos:pos + 8])
        if t == b"IEND" or pos + 12 + n > len(d):
            break  # IEND reached, or the (cut) tail
        body = d[pos + 8:pos + 8 + n]
        if zlib.crc32(t + body) != struct.unpack(">I", d[pos + 8 + n:pos + 12 + n])[0]:
            return None
        if t == b"IHDR":
            wh = struct.unpack(">II", body[:8])
        if t == b"IDAT":
            idat += body
            end = pos + 12 + n
        pos += 12 + n
    try:
        raw = zlib.decompress(idat)
    except zlib.error:
        return None
    return d[:end] + IEND if wh and len(raw) == wh[1] * (wh[0] * 4 + 1) else None  # RGBA 8-bit


def repair(path):
    s = base64.b64encode(path.read_bytes()).decode()
    parts = s.split("null")
    for mask in itertools.product((False, True), repeat=len(parts) - 1):
        cand = parts[0] + "".join(("NaN" if m else "null") + p for m, p in zip(mask, parts[1:]))
        cand = cand[: len(cand) // 4 * 4]
        png = parse(base64.b64decode(cand))
        if png:
            return png


if __name__ == "__main__":
    ok = fixed = bad = 0
    for root in sys.argv[1:]:
        for f in sorted(Path(root).glob("**/shap.png")):
            d = f.read_bytes()
            if parse(d) == d:
                ok += 1; continue
            png = repair(f)
            if png:
                f.write_bytes(png); fixed += 1
            else:
                bad += 1
                if bad <= 5: print("could not repair", f)
    print(f"already fine {ok} | repaired {fixed} | unrepairable {bad}")
