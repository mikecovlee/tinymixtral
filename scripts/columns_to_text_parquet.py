# Copyright (C) Michael Lee (李登淳) 2026. All rights reserved.
# Open-source under the MIT License. See LICENSE for details.
"""Build a `text`-column parquet from a dataset whose text is split across columns.

Used for OpenCodeInstruct (columns input/output, no `text`).

Usage:
    python scripts/columns_to_text_parquet.py \
        --input data/raw/r5_code --output data/raw/r5_code_text \
        --columns input output --sep "\n\n"
"""

import argparse
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

SCHEMA = pa.schema([("text", pa.string())])


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--columns", nargs="+", required=True)
    p.add_argument("--sep", default="\n\n")
    p.add_argument("--min-chars", type=int, default=10)
    args = p.parse_args()

    outdir = Path(args.output)
    outdir.mkdir(parents=True, exist_ok=True)
    files = sorted(Path(args.input).glob("*.parquet*"))
    print(f"{len(files)} files -> {outdir}", flush=True)
    for i, f in enumerate(files, 1):
        out = outdir / f"{f.name}"
        if out.exists():
            print(f"  [{i}/{len(files)}] skip {f.name}", flush=True)
            continue
        writer = pq.ParquetWriter(out, SCHEMA, compression="zstd")
        n = 0
        pf = pq.ParquetFile(f)
        for batch in pf.iter_batches(batch_size=5000, columns=args.columns):
            d = batch.to_pydict()
            cols = [d[c] for c in args.columns]
            buf = []
            for vals in zip(*cols):
                t = args.sep.join(v or "" for v in vals)
                if len(t) >= args.min_chars:
                    buf.append(t)
                    n += 1
            if buf:
                writer.write_table(pa.table({"text": pa.array(buf, type=pa.string())}))
        writer.close()
        print(f"  [{i}/{len(files)}] {f.name}: {n:,} docs", flush=True)
    print("done", flush=True)


if __name__ == "__main__":
    main()
