#!/usr/bin/env python3
# Copyright (C) Michael Lee (李登淳) 2026. All rights reserved.
# Open-source under the MIT License. See LICENSE for details.
"""对任意 checkpoint（final 或周期 step_*）测一次 val loss/PPL。

用于补回只在终端 pane 打印、未落盘的 [eval] 终点，或对非分段产物的
checkpoint 做同口径评估（与训练内 val evaluator 完全相同的批次/autocast）。
"""
import math, sys, argparse, glob
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from model.modeling import TinyMixtralForCausalLM
from scripts.train_utils import make_val_evaluator


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True,
                   help="checkpoint 目录（含 config.json + pytorch_model.bin）")
    p.add_argument("--val-dir", required=True, help="含 val_*.pt 的目录")
    p.add_argument("--batch-size", type=int, default=48)
    p.add_argument("--seq-len", type=int, default=1024)
    p.add_argument("--max-tokens", type=int, default=2_000_000)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    if args.batch_size <= 0 or args.seq_len <= 0:
        p.error("batch-size and seq-len must be positive")
    val_files = sorted(glob.glob(f"{args.val_dir}/val_*.pt"))
    if not val_files:
        sys.exit(f"ERROR: no val_*.pt in {args.val_dir}")

    model = TinyMixtralForCausalLM.from_pretrained(args.checkpoint)
    model = model.to(args.device).to(torch.bfloat16)
    model.eval()
    eval_fn = make_val_evaluator(val_files, args.batch_size, args.seq_len,
                                 args.max_tokens, device=args.device)
    result = eval_fn(model)
    if result is None:
        sys.exit("ERROR: evaluator saw zero batches")
    name = Path(args.checkpoint).name
    print(f"VAL name={name} val_loss={result['val_loss']:.4f} "
          f"val_ppl={result['val_ppl']:.2f} batches={result['val_batches']} "
          f"device={args.device}", flush=True)


if __name__ == "__main__":
    main()
