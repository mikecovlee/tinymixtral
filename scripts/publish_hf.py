#!/usr/bin/env python3
# Copyright (C) Michael Lee (李登淳) 2026. All rights reserved.
# Open-source under the MIT License. See LICENSE for details.
"""将训练好的 checkpoint 打包为 HuggingFace 兼容格式。

用法:
    python scripts/publish_hf.py --checkpoint checkpoints/run/step_0177557_final --output publish/
    python scripts/publish_hf.py --checkpoint checkpoints/run/step_0177557_final --output publish/ --tokenizer tokenizer/

产出 publish/ 目录，可以用 AutoModelForCausalLM.from_pretrained("publish/", trust_remote_code=True) 直接加载。
"""

import argparse
import json
import shutil
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from hf.configuration_tinymixtral import TinyMixtralConfig
from hf.modeling_tinymixtral import TinyMixtralForCausalLM


def main():
    p = argparse.ArgumentParser(description="Package TinyMixtral for HF Hub")
    p.add_argument("--checkpoint", required=True, help="源 checkpoint 路径")
    p.add_argument("--output", default="publish", help="输出目录")
    p.add_argument("--tokenizer", default=None, help="可选：Tokenizer 目录")
    args = p.parse_args()

    ckpt = Path(args.checkpoint)
    bin_file = ckpt / "pytorch_model.bin"
    if not bin_file.exists():
        print(f"ERROR: {bin_file} not found"); sys.exit(1)

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)

    # 1. 从旧 config 读取参数，创建 HF 兼容 config
    cfg_file = ckpt / "config.json"
    if cfg_file.exists():
        with open(cfg_file) as f:
            old = json.load(f)
    else:
        old = {}
    valid = {k: v for k, v in old.items() if k in TinyMixtralConfig().__dict__}
    config = TinyMixtralConfig(**valid)

    # 2. 加载权重
    print(f"Loading {bin_file} ...")
    model = TinyMixtralForCausalLM(config)
    state_dict = torch.load(bin_file, map_location="cpu", weights_only=True)
    model.load_state_dict(state_dict, strict=True)
    nM = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"Loaded {nM:.0f}M params, {len(state_dict)} keys")

    # 3. save_pretrained + 补充 auto_map
    model.save_pretrained(str(output), safe_serialization=False)
    cfg_path = output / "config.json"
    cfg = json.loads(cfg_path.read_text())
    cfg["auto_map"] = {
        "AutoConfig": "configuration_tinymixtral.TinyMixtralConfig",
        "AutoModelForCausalLM": "modeling_tinymixtral.TinyMixtralForCausalLM",
    }
    cfg_path.write_text(json.dumps(cfg, indent=2))
    print(f"Saved {output}/pytorch_model.bin + config.json (auto_map)")

    # 4. 复制兼容层代码 + LICENSE 到 publish
    root = Path(__file__).parent.parent
    hf_dir = root / "hf"
    for name in ("configuration_tinymixtral.py", "modeling_tinymixtral.py"):
        shutil.copy(hf_dir / name, output / name)
    shutil.copy(root / "LICENSE", output / "LICENSE")
    print("Copied configuration_tinymixtral.py + modeling_tinymixtral.py + LICENSE")

    # 5. tokenizer
    if args.tokenizer:
        for f in Path(args.tokenizer).iterdir():
            if f.is_file():
                shutil.copy(f, output / f.name)
        print(f"Copied tokenizer from {args.tokenizer}")

    # 6. 验证
    print("\nVerifying AutoModelForCausalLM.from_pretrained() ...")
    from transformers import AutoModelForCausalLM
    m = AutoModelForCausalLM.from_pretrained(str(output), trust_remote_code=True)
    nM2 = sum(p.numel() for p in m.parameters()) / 1e6
    assert abs(nM - nM2) < 1, f"Mismatch: {nM} vs {nM2}"
    print(f"OK: {nM2:.0f}M params via AutoModelForCausalLM")

    x = torch.randint(0, 1000, (2, 64))
    out = m(x[:, :-1], labels=x[:, 1:])
    print(f"OK: forward loss={out.loss.item():.4f}")

    print(f"\n{'='*60}")
    print(f"Ready: {output}/")
    print(f"  from transformers import AutoModelForCausalLM")
    print(f"  model = AutoModelForCausalLM.from_pretrained('{output}/', trust_remote_code=True)")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
