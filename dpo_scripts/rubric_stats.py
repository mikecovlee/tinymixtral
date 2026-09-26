"""Aggregate rubric judge outputs: per-arm means +- se, and PAIRED per-item
deltas vs a reference arm (default sft) on shared ids.

Usage: python dpo_scripts/rubric_stats.py data/dpo/rubric2_*.jsonl
       python dpo_scripts/rubric_stats.py --ref data/dpo/rubric2_sft.jsonl data/dpo/rubric2_*.jsonl
"""
import argparse
import glob
import json
import math
from pathlib import Path

DIMS = ("correctness", "completeness", "reasoning", "instruction_following")


def load(path):
    rows = {}
    for line in open(path, encoding="utf-8"):
        line = line.strip()
        if not line:
            continue
        d = json.loads(line)
        rows[d["id"]] = d
    return rows


def mean_se(vals):
    n = len(vals)
    if n == 0:
        return 0.0, 0.0
    m = sum(vals) / n
    var = sum((v - m) ** 2 for v in vals) / max(n - 1, 1)
    return m, math.sqrt(var / n)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", default=None)
    ap.add_argument("files", nargs="+")
    a = ap.parse_args()

    paths = []
    for f in a.files:
        paths.extend(glob.glob(f))
    paths = sorted(set(paths))
    data = {Path(p).stem.replace("rubric2_", "").replace("rubric_", ""): load(p)
            for p in paths}

    print(f"{'arm':12s} {'n':>4s} " + " ".join(f"{d[:9]:>9s}" for d in DIMS) + "  overall")
    for arm, rows in data.items():
        vals = {d: [r[d] for r in rows.values() if d in r] for d in DIMS}
        overall = [sum(r[d] for d in DIMS if d in r) / len([d for d in DIMS if d in r])
                   for r in rows.values() if all(d in r for d in DIMS)]
        line = f"{arm:12s} {len(rows):4d} "
        for d in DIMS:
            m, _ = mean_se(vals[d])
            line += f"{m:9.1f}"
        m, se = mean_se(overall)
        print(line + f"  {m:5.1f}+-{se:.2f}")

    ref = a.ref or str(Path(paths[0]).with_name(Path(paths[0]).name))
    ref_arm = "sft" if "sft" in data else list(data)[0]
    if ref_arm in data:
        print(f"\nPAIRED vs {ref_arm} (shared ids): delta +- se (t)")
        for arm, rows in data.items():
            if arm == ref_arm:
                continue
            shared = set(rows) & set(data[ref_arm])
            if not shared:
                continue
            dd = []
            for i in shared:
                a1 = [rows[i][d] for d in DIMS if d in rows[i]]
                b1 = [data[ref_arm][i][d] for d in DIMS if d in data[ref_arm][i]]
                dd.append(sum(a1) / len(a1) - sum(b1) / len(b1))
            m, se = mean_se(dd)
            t = m / se if se else 0.0
            print(f"  {arm:12s} n={len(shared):4d}  {m:+.2f} +- {se:.2f}  (t={t:+.1f})")


if __name__ == "__main__":
    main()
