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

from model import peek_mechanism


def main():
    p = argparse.ArgumentParser(description="Package TinyMixtral for HF Hub")
    p.add_argument("--checkpoint", required=True, help="源 checkpoint 路径")
    p.add_argument("--output", default="publish", help="输出目录")
    p.add_argument("--tokenizer", default=None, help="可选：Tokenizer 目录")
    args = p.parse_args()

    ckpt = Path(args.checkpoint)
    bin_file = ckpt / "pytorch_model.bin"
    if not bin_file.exists():
        print(f"ERROR: {bin_file} not found")
        sys.exit(1)

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)

    # 1. 从旧 config 读取参数，创建 HF 兼容 config（按机制方言选择对应的 hf 镜像）
    cfg_file = ckpt / "config.json"
    if cfg_file.exists():
        with open(cfg_file) as f:
            old = json.load(f)
    else:
        old = {}
    mech = peek_mechanism(old)
    if mech == "dense":
        from model.dense.hf.configuration_tinymistral import TinyMistralConfig as HFConfig
        from model.dense.hf.modeling_tinymistral import TinyMistralForCausalLM as HFModel
        hf_pkg, hf_cfg_file, hf_model_file = "dense", "configuration_tinymistral.py", "modeling_tinymistral.py"
        auto_cfg, auto_model = "configuration_tinymistral.TinyMistralConfig", "modeling_tinymistral.TinyMistralForCausalLM"
    elif mech == "shared_topk":
        from model.shared_topk.hf.configuration_tinymixtral import TinyMixtralConfig as HFConfig
        from model.shared_topk.hf.modeling_tinymixtral import TinyMixtralForCausalLM as HFModel
        hf_pkg, hf_cfg_file, hf_model_file = "shared_topk", "configuration_tinymixtral.py", "modeling_tinymixtral.py"
        auto_cfg, auto_model = "configuration_tinymixtral.TinyMixtralConfig", "modeling_tinymixtral.TinyMixtralForCausalLM"
    else:
        from model.topk.hf.configuration_tinymixtral import TinyMixtralConfig as HFConfig
        from model.topk.hf.modeling_tinymixtral import TinyMixtralForCausalLM as HFModel
        hf_pkg, hf_cfg_file, hf_model_file = "topk", "configuration_tinymixtral.py", "modeling_tinymixtral.py"
        auto_cfg, auto_model = "configuration_tinymixtral.TinyMixtralConfig", "modeling_tinymixtral.TinyMixtralForCausalLM"
    print(f"Mechanism: {mech} -> model/{hf_pkg}/hf/")
    valid = {k: v for k, v in old.items() if k in HFConfig().__dict__}
    config = HFConfig(**valid)

    # 2. 加载权重
    print(f"Loading {bin_file} ...")
    model = HFModel(config)
    state_dict = torch.load(bin_file, map_location="cpu", weights_only=True)
    model.load_state_dict(state_dict, strict=True)
    nM = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"Loaded {nM:.0f}M params, {len(state_dict)} keys")

    # 3. save_pretrained + 补充 auto_map
    eos_id = config.eos_token_id if config.eos_token_id is not None else 2
    pad_id = config.pad_token_id if config.pad_token_id is not None else eos_id
    model.generation_config.eos_token_id = eos_id
    model.generation_config.pad_token_id = pad_id
    model.save_pretrained(str(output), safe_serialization=False)
    cfg_path = output / "config.json"
    cfg = json.loads(cfg_path.read_text())
    cfg["auto_map"] = {
        "AutoConfig": auto_cfg,
        "AutoModelForCausalLM": auto_model,
    }
    cfg_path.write_text(json.dumps(cfg, indent=2))

    # generation_config.json 里只写 _from_model_config 时，加载后 eos_token_id 会丢失，
    # 导致 generate() 无法在 </s> 处停止；这里显式写入 eos/pad。
    gen_path = output / "generation_config.json"
    gen = json.loads(gen_path.read_text()) if gen_path.exists() else {}
    gen.pop("_from_model_config", None)
    gen["eos_token_id"] = eos_id
    gen["pad_token_id"] = pad_id
    gen_path.write_text(json.dumps(gen, indent=2))
    print(f"Saved {output}/pytorch_model.bin + config.json (auto_map) + generation_config.json")

    # 4. 复制兼容层代码 + LICENSE 到 publish
    root = Path(__file__).parent.parent
    hf_dir = root / "model" / hf_pkg / "hf"
    for name in (hf_cfg_file, hf_model_file):
        shutil.copy(hf_dir / name, output / name)
    shutil.copy(root / "LICENSE", output / "LICENSE")
    print(f"Copied {hf_cfg_file} + {hf_model_file} + LICENSE")

    # 5. tokenizer
    if args.tokenizer:
        keep = ("tokenizer", "vocab", "merges", "special_tokens_map.json", "chat_template.jinja")
        for f in Path(args.tokenizer).iterdir():
            if f.is_file() and any(f.name.startswith(k) or f.name == k for k in keep):
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
    print("  from transformers import AutoModelForCausalLM")
    print(f"  model = AutoModelForCausalLM.from_pretrained('{output}/', trust_remote_code=True)")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
