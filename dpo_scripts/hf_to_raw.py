"""Convert an HF safetensors snapshot into the raw layout the local
TinyMixtralForCausalLM.from_pretrained expects (config.json + pytorch_model.bin).
"""
import argparse
import os
import shutil

import torch
from safetensors.torch import load_file


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--src", required=True)
    p.add_argument("--out", required=True)
    a = p.parse_args()

    os.makedirs(a.out, exist_ok=True)
    sd = load_file(os.path.join(a.src, "model.safetensors"))
    if "lm_head.weight" not in sd and "embed_tokens.weight" in sd:
        sd["lm_head.weight"] = sd["embed_tokens.weight"]
    torch.save(sd, os.path.join(a.out, "pytorch_model.bin"))
    shutil.copy(os.path.join(a.src, "config.json"),
                os.path.join(a.out, "config.json"))
    print("wrote", a.out, len(sd), "keys")


if __name__ == "__main__":
    main()
