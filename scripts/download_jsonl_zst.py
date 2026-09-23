# Copyright (C) Michael Lee (李登淳) 2026. All rights reserved.
# Open-source under the MIT License. See LICENSE for details.
"""Download a directory of HF dataset files matching a suffix (e.g. .jsonl.zst).

Like scripts/download_parquets.py but keeps original filenames and supports
arbitrary suffixes (used for DCLM's .jsonl.zst shards).

Usage:
    python scripts/download_jsonl_zst.py \
        --repo mlfoundations/dclm-baseline-1.0 \
        --subdir global-shard_01_of_10/local-shard_0_of_10 \
        --output data/raw/r5_web --suffix .jsonl.zst --first 10
"""

import argparse
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import requests


def list_files(session, repo, subdir, suffix):
    url = f"https://huggingface.co/api/datasets/{repo}/tree/main/{subdir}"
    for attempt in range(1, 11):
        try:
            r = session.get(url, timeout=60)
            r.raise_for_status()
            break
        except Exception as e:
            if attempt == 10:
                raise
            time.sleep(min(30, 2 ** attempt))
    files = [f for f in r.json() if f["path"].endswith(suffix)]
    files.sort(key=lambda f: f["path"])
    return [(f["path"], f["size"]) for f in files]


def download_one(session, repo, path, size, outdir, verbose=True):
    out = outdir / Path(path).name
    if out.exists() and out.stat().st_size == size:
        if verbose:
            print(f"  [skip] {out.name} ({size // 10**6}MB)", flush=True)
        return True
    part = out.with_suffix(out.suffix + ".part")
    base_url = f"https://huggingface.co/datasets/{repo}/resolve/main/{path}"
    for attempt in range(1, 11):
        have = part.stat().st_size if part.exists() else 0
        headers = {"Range": f"bytes={have}-"} if have else {}
        try:
            with session.get(base_url, stream=True, headers=headers, timeout=120) as resp:
                if have and resp.status_code == 200:
                    have = 0
                    part.unlink(missing_ok=True)
                if resp.status_code not in (200, 206):
                    raise RuntimeError(f"HTTP {resp.status_code}")
                with open(part, "ab") as f:
                    for chunk in resp.iter_content(1 << 20):
                        f.write(chunk)
            if part.stat().st_size == size:
                part.replace(out)
                if verbose:
                    print(f"  [ok] {out.name} ({size // 10**6}MB)", flush=True)
                return True
            raise RuntimeError(f"size mismatch {part.stat().st_size}/{size}")
        except Exception as e:
            if attempt == 10:
                print(f"  [FAIL] {out.name}: {e}", flush=True)
                return False
            time.sleep(min(30, 2 ** attempt))
    return False


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--repo", required=True)
    p.add_argument("--subdir", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--suffix", default=".jsonl.zst")
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--first", type=int, default=None)
    args = p.parse_args()

    outdir = Path(args.output)
    outdir.mkdir(parents=True, exist_ok=True)
    session = requests.Session()
    print(f"Listing {args.repo}/{args.subdir} [{args.suffix}] ...", flush=True)
    files = list_files(session, args.repo, args.subdir, args.suffix)
    if args.first:
        files = files[: args.first]
    print(f"{len(files)} files, {sum(s for _, s in files) / 1e9:.1f} GB -> {outdir}", flush=True)

    lock = threading.Lock()
    results = {}

    def worker(item):
        path, size = item
        ok = download_one(session, args.repo, path, size, outdir)
        with lock:
            results[path] = ok

    t0 = time.time()
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for _ in pool.map(worker, files):
            pass
    failed = [p for p, ok in results.items() if not ok]
    got = sum(f.stat().st_size for f in outdir.glob(f"*{args.suffix}")) / 1e9
    print(f"Done in {(time.time() - t0) / 60:.1f} min: {len(files) - len(failed)}/{len(files)}, "
          f"{got:.1f} GB", flush=True)
    if failed:
        print("FAILED:", *failed, sep="\n  ", flush=True)


if __name__ == "__main__":
    main()
