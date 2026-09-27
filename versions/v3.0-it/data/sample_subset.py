"""Build the 50k CoT-polish SFT subset by stratified sampling from a 3M train set.

The polish tier is a small reasoning-heavy slice trained at low lr (5e-6) on top of
a finished ladder run. This carves a 50k slice out of the already-deduped /
decontaminated 3M train parquet, so no re-download or re-filtering is needed.
"""
import argparse
import json
import time
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

DEFAULT_SRC = "data/sft_3m/train.parquet"
DEFAULT_OUT = "data/sft_polish"
QUOTA = {"metamath": 12000, "orcamath": 12000, "omi2": 8000, "tulu3": 10000, "slimorca": 8000}
SEED = 42


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=DEFAULT_SRC)
    ap.add_argument("--out", default=DEFAULT_OUT)
    ap.add_argument("--batch-size", type=int, default=2048)
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    remaining = dict(QUOTA)
    got = {k: 0 for k in QUOTA}

    pf = pq.ParquetFile(args.src)
    names = set(pf.schema_arrow.names)
    missing = [c for c in ("conversations", "source") if c not in names]
    if missing:
        raise SystemExit(
            f"--src is missing column(s) {missing}; expected the output of data/build_dataset.py"
        )
    cols = [c for c in ["conversations", "source", "category", "lang", "n_turns"] if c in names]
    writer = None
    t0 = time.time()
    for batch in pf.iter_batches(batch_size=args.batch_size, columns=cols):
        d = batch.to_pydict()
        keep = []
        for i, s in enumerate(d["source"]):
            if remaining.get(s, 0) > 0:
                remaining[s] -= 1
                got[s] += 1
                keep.append(i)
        if not keep:
            continue
        tbl = pa.Table.from_pydict({k: [v[i] for i in keep] for k, v in d.items()})
        if writer is None:
            writer = pq.ParquetWriter(out / "train.parquet", tbl.schema)
        writer.write_table(tbl)
        if all(v <= 0 for v in remaining.values()):
            break
    if writer is not None:
        writer.close()

    stats = {
        "source_set": args.src,
        "seed": SEED,
        "quota": QUOTA,
        "rows": sum(got.values()),
        "by_source": got,
        "elapsed_s": round(time.time() - t0, 1),
    }
    (out / "stats.json").write_text(json.dumps(stats, indent=2))
    print(json.dumps(stats, indent=2), flush=True)


if __name__ == "__main__":
    main()
