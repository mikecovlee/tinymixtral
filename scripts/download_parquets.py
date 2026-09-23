#!/usr/bin/env python3
# Copyright (C) Michael Lee (李登淳) 2026. All rights reserved.
# Open-source under the MIT License. See LICENSE for details.
"""并行下载 HF dataset 仓库中某个目录下的全部 parquet 文件到本地。

用于网络代理下 `datasets` 流式读取（xet/range 请求）效率低时的替代方案：
先把整文件拉下来，再用 scripts/prepare_data_local.py 本地 tokenize。

用法:
    python scripts/download_parquets.py \
        --repo HuggingFaceFW/fineweb-edu --subdir sample/10BT \
        --output data/raw/fineweb3 --workers 4

    python scripts/download_parquets.py \
        --repo HuggingFaceTB/cosmopedia-v2 --subdir cosmopedia-v2 \
        --output data/raw/cosmopedia3 --first 8
"""

import argparse
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import requests


def list_parquets(session, repo, subdir):
    url = f"https://huggingface.co/api/datasets/{repo}/tree/main/{subdir}"
    for attempt in range(1, 11):
        try:
            r = session.get(url, timeout=60)
            r.raise_for_status()
            break
        except Exception as e:
            if attempt == 10:
                raise
            wait = min(30, 2 ** attempt)
            print(f"[retry {attempt}/10] list: {e} (wait {wait}s)", flush=True)
            time.sleep(wait)
    files = [f for f in r.json() if f["path"].endswith(".parquet")]
    files.sort(key=lambda f: f["path"])
    return [(f["path"], f["size"]) for f in files]


def download_one(session, repo, path, size, outdir, verbose=True):
    out = outdir / f"{Path(path).name}.parquet"
    if out.exists() and out.stat().st_size == size:
        if verbose:
            print(f"  [skip] {Path(path).name} already complete ({size//1e6}MB)", flush=True)
        return True
    part = out.with_suffix(".parquet.part")
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
                expected = size - have if resp.status_code == 206 else size
                with open(part, "ab") as f:
                    for chunk in resp.iter_content(1 << 20):
                        f.write(chunk)
            if part.stat().st_size == size:
                part.replace(out)
                if verbose:
                    print(f"  [ok] {Path(path).name} ({size//1e6}MB)", flush=True)
                return True
            raise RuntimeError(f"size mismatch: {part.stat().st_size}/{size}")
        except Exception as e:
            if attempt == 10:
                print(f"  [FAIL] {Path(path).name}: {e}", flush=True)
                return False
            wait = min(30, 2 ** attempt)
            print(f"  [retry {attempt}/10] {Path(path).name}: {e} (resume at {have//1e6}MB, wait {wait}s)", flush=True)
            time.sleep(wait)
    return False


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--repo", required=True, help="HF dataset id, e.g. HuggingFaceFW/fineweb-edu")
    p.add_argument("--subdir", required=True, help="目录路径, e.g. sample/10BT")
    p.add_argument("--output", required=True, help="本地输出目录")
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--first", type=int, default=None, help="只下载排序后的前 N 个文件")
    p.add_argument("--last", type=int, default=None, help="只下载排序后的后 N 个文件")
    args = p.parse_args()

    outdir = Path(args.output)
    outdir.mkdir(parents=True, exist_ok=True)

    session = requests.Session()
    print(f"Listing {args.repo}/{args.subdir} ...", flush=True)
    files = list_parquets(session, args.repo, args.subdir)
    if args.first:
        files = files[: args.first]
    if args.last:
        files = files[-args.last:]
    total_gb = sum(s for _, s in files) / 1e9
    print(f"{len(files)} files, {total_gb:.1f} GB -> {outdir}", flush=True)

    results = {}
    lock = threading.Lock()
    t0 = time.time()

    def worker(item):
        path, size = item
        ok = download_one(session, args.repo, path, size, outdir)
        with lock:
            results[path] = ok

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for _ in pool.map(worker, files):
            pass

    failed = [p for p, ok in results.items() if not ok]
    got_gb = sum(
        f.stat().st_size for f in outdir.glob("*.parquet")
    ) / 1e9
    print(f"\nDone in {(time.time()-t0)/60:.1f} min: {len(files)-len(failed)}/{len(files)} files, "
          f"{got_gb:.1f} GB on disk", flush=True)
    if failed:
        print("FAILED:", *failed, sep="\n  ", flush=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
