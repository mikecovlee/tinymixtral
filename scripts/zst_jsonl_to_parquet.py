# Copyright (C) Michael Lee (李登淳) 2026. All rights reserved.
# Open-source under the MIT License. See LICENSE for details.
"""Convert .jsonl.zst shards (e.g. DCLM) to parquet with a single `text` column.

Output is named `<stem>.parquet.parquet` so scripts/prepare_data_local.py picks
it up unchanged.

Usage:
    python scripts/zst_jsonl_to_parquet.py --input data/raw/r5_web --output data/raw/r5_web_pq
"""

import argparse
import io
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import zstandard as zstd

SCHEMA = pa.schema([("text", pa.string())])


def convert_one(job):
    zst_path, out_dir, text_key = job
    zst_path, out_dir = Path(zst_path), Path(out_dir)
    name = zst_path.name
    stem = name[: -len(".jsonl.zst")] if name.endswith(".jsonl.zst") else zst_path.stem
    out = out_dir / f"{stem}.parquet.parquet"
    if out.exists():
        return name, -1
    dctx = zstd.ZstdDecompressor()
    n = 0
    writer = pq.ParquetWriter(out, SCHEMA, compression="zstd")
    buf = []

    def flush():
        if buf:
            writer.write_table(pa.table({"text": pa.array(buf, type=pa.string())}))
            buf.clear()

    try:
        with open(zst_path, "rb") as fh, dctx.stream_reader(fh) as reader:
            tw = io.TextIOWrapper(reader, encoding="utf-8")
            for line in tw:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except Exception:
                    continue
                t = obj.get(text_key)
                if t and len(t) >= 10:
                    buf.append(t)
                    n += 1
                if len(buf) >= 20000:
                    flush()
        flush()
    finally:
        writer.close()
    return name, n


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--text-key", default="text")
    p.add_argument("--workers", type=int, default=4)
    args = p.parse_args()

    outdir = Path(args.output)
    outdir.mkdir(parents=True, exist_ok=True)
    files = sorted(Path(args.input).glob("*.jsonl.zst"))
    print(f"{len(files)} .jsonl.zst files -> {outdir}", flush=True)
    jobs = [(str(f), str(outdir), args.text_key) for f in files]
    total = 0
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        for i, (name, n) in enumerate(ex.map(convert_one, jobs), 1):
            if n < 0:
                print(f"  [{i}/{len(files)}] skip {name}", flush=True)
            else:
                total += n
                print(f"  [{i}/{len(files)}] {name}: {n:,} docs", flush=True)
    print(f"done, {total:,} docs", flush=True)


if __name__ == "__main__":
    main()
