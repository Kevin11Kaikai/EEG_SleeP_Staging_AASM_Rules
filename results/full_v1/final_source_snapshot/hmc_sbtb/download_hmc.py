"""Explicit official-manifest download, resume, SHA256, and persistent status."""
import argparse
import concurrent.futures
import json
import time
import urllib.request
from pathlib import Path
from .io_utils import sha256, write_json

BASE = "https://physionet.org/files/hmc-sleep-staging/1.1/"


def fetch(url, path, expected=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and (not expected or sha256(path) == expected):
        return {"file": str(path), "status": "VERIFIED", "cached": True,
                "bytes": path.stat().st_size, "sha256": sha256(path)}
    part = path.with_suffix(path.suffix + ".part")
    last = None
    for attempt in range(4):
        try:
            size = part.stat().st_size if part.exists() else 0
            req = urllib.request.Request(url, headers={"Range": f"bytes={size}-"} if size else {})
            with urllib.request.urlopen(req, timeout=90) as response:
                append = size and response.status == 206
                with part.open("ab" if append else "wb") as out:
                    while block := response.read(4 * 1024 * 1024):
                        out.write(block)
            actual = sha256(part)
            if expected and actual != expected:
                part.unlink()
                raise ValueError(f"checksum mismatch {actual} != {expected}")
            part.replace(path)
            return {"file": str(path), "status": "VERIFIED", "cached": False,
                    "bytes": path.stat().st_size, "sha256": actual}
        except Exception as exc:
            last = repr(exc)
            time.sleep(min(2 ** attempt, 8))
    return {"file": str(path), "status": "MISSING", "error": last}


def official_manifest(root):
    root = Path(root)
    # Fetch checksum first; all other metadata receive official validation.
    result = fetch(BASE + "SHA256SUMS.txt", root / "SHA256SUMS.txt")
    if result["status"] != "VERIFIED":
        raise RuntimeError(result)
    sums = {}
    for line in (root / "SHA256SUMS.txt").read_text().splitlines():
        digest, name = line.split(maxsplit=1)
        sums[name.strip().lstrip("*")] = digest
    for name in ["RECORDS", "LICENSE.txt", "Subjects_info_aggregated.txt"]:
        result = fetch(BASE + name, root / name, sums.get(name))
        if result["status"] != "VERIFIED":
            raise RuntimeError(result)
    records = sorted(Path(x.strip()).stem for x in
                     (root / "RECORDS").read_text().splitlines() if x.strip())
    write_json(root / "pilot_selection.json", {
        "algorithm": "first 10 lexicographically sorted IDs in official v1.1 RECORDS",
        "pilot": records[:10], "smoke": records[:1], "official_count": len(records),
        "subject_policy": "record ID as subject proxy; one-night-per-patient assumption audited separately",
        "untouched": records[10:],
    })
    return records, sums


def download(root, subset="smoke", workers=2):
    root = Path(root)
    records, sums = official_manifest(root)
    chosen = records if subset == "all" else records[:10 if subset == "pilot" else 1]
    status_path = root / "download_status.json"
    status = json.loads(status_path.read_text()) if status_path.exists() else {}

    def one(record):
        rows = []
        for suffix in [".edf", "_sleepscoring.edf", "_sleepscoring.txt"]:
            name = f"recordings/{record}{suffix}"
            rows.append(fetch(BASE + name, root / name, sums.get(name)))
        return record, rows

    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(one, r): r for r in chosen}
        for future in concurrent.futures.as_completed(futures):
            record, rows = future.result()
            status[record] = rows
            write_json(status_path, status)
            print(record, [r["status"] for r in rows], sum(r.get("bytes", 0) for r in rows), flush=True)
    return status


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--subset", choices=["smoke", "pilot", "all"], default="smoke")
    parser.add_argument("--workers", type=int, default=2)
    args = parser.parse_args()
    download(args.data_root, args.subset, args.workers)
