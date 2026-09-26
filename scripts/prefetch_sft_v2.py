"""Pre-download SFT-v2 source files to local disk.

Local-file loading is ~100x faster than HTTP streaming through the proxy.
Files land in data/sft_v2_src/<name>/ preserving the repo sub-path.
"""
import argparse
import fnmatch
import json
from pathlib import Path

from huggingface_hub import HfApi, hf_hub_download

SPECS = {
    "tulu3": ("allenai/tulu-3-sft-mixture", ["data/*.parquet"], 6),
    "openhermes": ("teknium/OpenHermes-2.5", ["*.json"], 1),
    "slimorca": ("Open-Orca/SlimOrca", ["*.jsonl"], 1),
    "openorca": ("Open-Orca/OpenOrca", ["*.parquet"], 1),
    "ultrachat": ("HuggingFaceH4/ultrachat_200k", ["data/train_sft-*.parquet"], 4),
    "metamath": ("meta-math/MetaMathQA", ["*.json"], 1),
    "orcamath": ("microsoft/orca-math-word-problems-200k", ["*.parquet"], 1),
    "omi2": ("nvidia/OpenMathInstruct-2", ["data/train-*.parquet"], 4),
    "squad2": ("rajpurkar/squad_v2", ["squad_v2/train-*.parquet"], 1),
    "trivia": ("mandarjoshi/trivia_qa", ["rc.nocontext/train-*.parquet"], 2),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/sft_v2_src")
    ap.add_argument("--sources", default="")
    args = ap.parse_args()
    only = {s.strip() for s in args.sources.split(",") if s.strip()}
    api = HfApi()
    out_root = Path(args.out)
    manifest = {}
    for name, (repo, patterns, cap) in SPECS.items():
        if only and name not in only:
            continue
        d = out_root / name
        d.mkdir(parents=True, exist_ok=True)
        try:
            files = api.list_repo_files(repo, repo_type="dataset")
        except Exception as e:
            print(f"{name}: list failed: {str(e)[:120]}", flush=True)
            continue
        sel = []
        for f in files:
            if any(fnmatch.fnmatch(f, p) for p in patterns):
                sel.append(f)
        sel = sorted(sel)[:cap]
        got = []
        for f in sel:
            try:
                p = hf_hub_download(repo_id=repo, filename=f, repo_type="dataset",
                                    local_dir=str(d))
                got.append({"file": f, "path": p, "size": Path(p).stat().st_size})
                print(f"{name}: {f} {Path(p).stat().st_size/1e6:.1f}MB", flush=True)
            except Exception as e:
                print(f"{name}: {f} FAILED {str(e)[:100]}", flush=True)
        manifest[name] = {"repo": repo, "files": got}
        print(f"{name}: {len(got)}/{len(sel)} files, "
              f"{sum(x['size'] for x in got)/1e9:.2f}GB", flush=True)
    (out_root / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print("PREFETCH_DONE")


if __name__ == "__main__":
    main()
