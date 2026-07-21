"""
Upload a DICOM tree to Orthanc via its REST API (POST /instances).

USAGE
-----
    # local Orthanc (default)
    python upload.py --input ./dicom_out

    # server Orthanc behind a base path + basic auth
    python upload.py --input ./dicom_out \
        --host my.server.org --port 443 --scheme https \
        --base /orthanc --user alice --password secret

Re-uploading is safe: the converter uses deterministic UIDs and the Orthanc config
has OverwriteInstances=true, so repeats overwrite rather than duplicate.
"""
import os, sys, glob, time, base64, argparse
import http.client


def main():
    ap = argparse.ArgumentParser(description="Upload a DICOM tree to Orthanc.")
    ap.add_argument("--input", default="./dicom_out", help="DICOM tree to upload")
    ap.add_argument("--host", default="localhost")
    ap.add_argument("--port", type=int, default=8042)
    ap.add_argument("--scheme", choices=["http", "https"], default="http")
    ap.add_argument("--base", default="", help="base path prefix, e.g. /orthanc")
    ap.add_argument("--user", default=None)
    ap.add_argument("--password", default=None)
    args = ap.parse_args()

    files = sorted(glob.glob(os.path.join(args.input, "**", "*.dcm"), recursive=True))
    if not files:
        print(f"No .dcm files under {args.input}"); sys.exit(1)

    headers = {"Content-Type": "application/dicom"}
    if args.user is not None:
        token = base64.b64encode(f"{args.user}:{args.password or ''}".encode()).decode()
        headers["Authorization"] = f"Basic {token}"

    def connect():
        if args.scheme == "https":
            return http.client.HTTPSConnection(args.host, args.port, timeout=60)
        return http.client.HTTPConnection(args.host, args.port, timeout=60)

    conn = connect()
    url = f"{args.base}/instances"
    ok = fail = 0
    t0 = time.time()
    print(f"Uploading {len(files)} files -> {args.scheme}://{args.host}:{args.port}{url}")
    for i, f in enumerate(files):
        with open(f, "rb") as fh:
            data = fh.read()
        try:
            conn.request("POST", url, body=data,
                         headers={**headers, "Content-Length": str(len(data))})
            r = conn.getresponse(); r.read()
            if r.status == 200:
                ok += 1
            else:
                fail += 1
                if fail <= 3:
                    print(f"  {r.status} on {f}")
        except Exception:
            fail += 1
            conn.close(); conn = connect()
        if (i + 1) % 1000 == 0:
            print(f"  {i+1}/{len(files)}  ok={ok} fail={fail}  ({time.time()-t0:.0f}s)", flush=True)
    print(f"DONE {ok} ok, {fail} fail in {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
