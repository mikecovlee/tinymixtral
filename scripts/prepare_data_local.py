#!/usr/bin/env python3
# Copyright (C) Michael Lee (李登淳) 2026. All rights reserved.
# Open-source under the MIT License. See LICENSE for details.
"""从本地 parquet 文件 tokenize 成 .pt shards（网络代理下 datasets 流式读取低效时的替代方案）。

与 scripts/prepare_data.py 的语义完全一致：
  - 每个文档 `encode(text, add_special_tokens=False)` 后追加 eos
  - 文档 `len(text) < 10` 跳过
  - `--skip-tokens` 按「编码后 token 流」精确跳过（可落在文档中间，取该文档尾部）
  - `--max-tokens` 截断时把最后一个 token 强制为 eos
  - 输出 100M-token 的 int64 shard（train_NNNN.pt），staging -> 原子替换

用法:
    # 1) 先并行下载整文件（见 scripts/download_parquets.py）
    # 2) 本地 tokenize（多进程绕开 GIL）
    python scripts/prepare_data_local.py \
        --input data/raw/fineweb3 --tokenizer tokenizer/ \
        --output data/pretrain/fineweb3 \
        --skip-tokens 7120000000 --max-tokens 3560000000 --force --workers 8
"""

import argparse
import glob
import os
import shutil
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import torch
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).parent.parent))


_TOK = None


def _init(tokenizer_dir, expect_vocab):
    global _TOK
    from transformers import AutoTokenizer

    _TOK = AutoTokenizer.from_pretrained(tokenizer_dir, legacy=False)
    if len(_TOK) != expect_vocab:
        raise ValueError(f"vocab {len(_TOK)} != expected {expect_vocab}")
    if _TOK.eos_token_id is None:
        raise ValueError("tokenizer must define eos_token_id")


def _iter_texts(path):
    pf = pq.ParquetFile(path)
    for batch in pf.iter_batches(batch_size=20000, columns=["text"]):
        yield from batch.column("text").to_pylist()


def _count(args):
    idx, path = args
    n = 0
    for text in _iter_texts(path):
        if len(text) < 10:
            continue
        n += len(_TOK.encode(text, add_special_tokens=False)) + 1  # + eos
    return idx, n


def _encode(args):
    idx, path, count, out = args
    import numpy as np

    eos = _TOK.eos_token_id
    arr = np.memmap(out, dtype=np.int64, mode="w+", shape=(count,))
    pos = 0
    for text in _iter_texts(path):
        if len(text) < 10:
            continue
        ids = _TOK.encode(text, add_special_tokens=False)
        n = len(ids)
        arr[pos:pos + n] = ids
        arr[pos + n] = eos
        pos += n + 1
    arr.flush()
    del arr
    assert pos == count, f"encode count mismatch: {pos} != {count}"
    return idx, count


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True, help="本地 parquet 目录（按文件名排序）")
    p.add_argument("--tokenizer", default="tokenizer/")
    p.add_argument("--output", required=True, help="输出 shard 目录")
    p.add_argument("--skip-tokens", type=int, default=0)
    p.add_argument("--max-tokens", type=int, default=None)
    p.add_argument("--shard-size", type=int, default=100_000_000)
    p.add_argument("--workers", type=int, default=min(8, os.cpu_count() or 1))
    p.add_argument("--force", action="store_true")
    p.add_argument("--counts-json", default=None,
                   help="可选：从 JSON 文件读取每文件 token 计数，跳过 Phase 1")
    args = p.parse_args()

    from model.config import TinyMixtralConfig
    expect_vocab = TinyMixtralConfig().vocab_size

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists() and any(output.iterdir()) and not args.force:
        print(f"ERROR: output not empty: {output} (use --force)", file=sys.stderr)
        sys.exit(1)
    stale = list(output.parent.glob(f".{output.name}.tmp-*")) + list(
        output.parent.glob(f".{output.name}.backup-*")
    )
    if stale:
        print("ERROR: stale staging dirs found:", *stale, sep="\n  ", file=sys.stderr)
        sys.exit(1)

    files = sorted(glob.glob(str(Path(args.input) / "*.parquet.parquet")))
    if not files:
        print(f"ERROR: no *.parquet.parquet in {args.input}", file=sys.stderr)
        sys.exit(1)
    print(f"Input: {len(files)} files in {args.input}", flush=True)

    staging = output.with_name(f".{output.name}.tmp-{os.getpid()}")
    encdir = staging / "_enc"
    encdir.mkdir(parents=True)

    # ---- Phase 1: 并行计数（定位 skip 边界 + 总 token 数）----
    counts = None
    if args.counts_json and os.path.exists(args.counts_json):
        import json
        with open(args.counts_json) as f:
            counts = json.load(f)
        if len(counts) != len(files):
            p.error(f"--counts-json 有 {len(counts)} 个计数，但输入有 {len(files)} 个文件")
        print(f"Loaded {len(counts)} counts from {args.counts_json} (skip Phase 1)", flush=True)
    if counts is None:
        print(f"Phase 1: counting tokens ({args.workers} procs) ...", flush=True)
        t0 = time.time()
        counts = [0] * len(files)
        with ProcessPoolExecutor(max_workers=args.workers, initializer=_init,
                                 initargs=(args.tokenizer, expect_vocab)) as ex:
            for idx, c in ex.map(_count, [(i, f) for i, f in enumerate(files)]):
                counts[idx] = c
                print(f"  file {idx:02d}/{len(files)-1}: {c:,} tokens "
                      f"({time.time()-t0:.0f}s)", flush=True)
        print(f"Phase 1 done: total={sum(counts):,} tokens in {time.time()-t0:.0f}s", flush=True)
    total = sum(counts)
    cum = []
    s = 0
    for c in counts:
        cum.append(s)
        s += c

    # ---- 定位写窗口 [skip, skip+max_tokens) 覆盖的文件 ----
    hi = total
    if args.max_tokens is not None:
        hi = min(hi, args.skip_tokens + args.max_tokens)
    window = [i for i in range(len(files)) if cum[i] + counts[i] > args.skip_tokens and cum[i] < hi]
    if not window:
        print(f"WARNING: skip={args.skip_tokens:,} 超出数据范围 (total={total:,})，无数据可写", flush=True)
        shutil.rmtree(staging, ignore_errors=True)
        sys.exit(1)
    offset = args.skip_tokens - cum[window[0]]  # 首个窗口文件内需丢弃的 token 数
    window_tokens = sum(counts[i] for i in window) - offset
    print(f"Window: files {window[0]}..{window[-1]} ({len(window)}), "
          f"write {window_tokens:,} tokens (offset={offset:,})", flush=True)

    # ---- Phase 2: 并行编码窗口文件 ----
    print(f"Phase 2: encoding window files ({args.workers} procs) ...", flush=True)
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=args.workers, initializer=_init,
                             initargs=(args.tokenizer, expect_vocab)) as ex:
        jobs = [(i, files[i], counts[i], str(encdir / f"enc_{i:04d}.bin")) for i in window]
        for idx, n in ex.map(_encode, jobs):
            print(f"  encoded file {idx:02d}: {n:,} tokens ({time.time()-t0:.0f}s)", flush=True)
    print(f"Phase 2 done in {time.time()-t0:.0f}s", flush=True)

    # ---- Phase 3: 顺序装配 -> shard（丢弃 offset、按 max_tokens 截断、末 token 强制 eos）----
    print("Phase 3: assembling shards ...", flush=True)
    eos = _load_eos(args.tokenizer)
    cap = None
    exhausted = False
    if args.max_tokens is not None:
        cap = min(args.max_tokens, window_tokens)
        exhausted = window_tokens < args.max_tokens
    written = 0
    shard_idx = 0
    buf = []
    bufn = 0
    abs_pos = 0  # 已写出的绝对 token 位置（从写窗口起点 0 计）

    def flush_shard():
        nonlocal buf, bufn, shard_idx
        if bufn == 0:
            return
        t = torch.cat(buf) if len(buf) > 1 else buf[0].clone()  # clone 切断 view，避免保存整块底层存储
        path = staging / f"train_{shard_idx:04d}.pt"
        torch.save(t, path)
        print(f"  {path.name}: {bufn:,} tokens", flush=True)
        buf, bufn = [], 0
        shard_idx += 1

    import numpy as np

    for i in window:
        mm = np.memmap(str(encdir / f"enc_{i:04d}.bin"), dtype=np.int64,
                       mode="r", shape=(counts[i],))
        t = torch.from_numpy(mm)
        off = offset if i == window[0] else 0
        L = t.numel() - off
        take = L if cap is None else min(L, cap - written)
        if take <= 0:
            del t, mm
            break
        pos = off
        end = off + take
        while pos < end:
            space = args.shard_size - bufn
            chunk = t[pos:pos + min(space, end - pos)]
            # 若截断点 cap-1 落在此 chunk 内，把该位置强制为 eos（与原逻辑一致）
            buf.append(chunk)
            bufn += int(chunk.numel())
            pos += int(chunk.numel())
            if bufn >= args.shard_size:
                flush_shard()
        written += int(take)
        del t, mm
        chunk = None
        if cap is not None and written >= cap:
            break

    # 末 token 处理：写满 cap 且数据未耗尽 -> 截断点强制 eos；数据耗尽 -> 告警（末 token 本就是 eos）
    flush_shard()
    if cap is not None and not exhausted:
        last = staging / f"train_{shard_idx-1:04d}.pt"
        t = torch.load(last)  # 常规加载并显式 del，避免 Windows 下 mmap 句柄挡住 os.replace
        t[-1] = eos
        tmp = last.with_suffix(".fix")
        torch.save(t, tmp)
        del t
        tmp.replace(last)
        print(f"  forced last token -> eos in {last.name}", flush=True)
    elif cap is not None and exhausted:
        print(f"WARNING: data ended after {written:,} tokens, target was {args.max_tokens:,} "
              f"({written/args.max_tokens*100:.1f}%)", flush=True)

    # ---- 原子替换 ----
    import gc
    import time as _time
    gc.collect()
    # 先删除中间 _enc 文件（不随输出一起发布）；Windows 下 memmap 句柄可能延迟释放，重试
    for _ in range(40):
        shutil.rmtree(encdir, ignore_errors=True)
        if not encdir.exists():
            break
        gc.collect()
        _time.sleep(0.5)
    backup = output.with_name(f".{output.name}.backup-{os.getpid()}")
    if output.exists():
        os.replace(output, backup)
    try:
        for attempt in range(40):
            try:
                os.replace(staging, output)
                break
            except PermissionError:
                if attempt == 39:
                    raise
                gc.collect()
                _time.sleep(0.5)
    except BaseException:
        if backup.exists():
            os.replace(backup, output)
        raise
    if backup.exists():
        shutil.rmtree(backup)
    print(f"Done: {written:,} tokens -> {shard_idx} shards -> {output}", flush=True)


def _load_eos(tokenizer_dir):
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(tokenizer_dir, legacy=False).eos_token_id


if __name__ == "__main__":
    main()
