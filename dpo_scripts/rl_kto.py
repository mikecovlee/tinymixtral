"""KTO (Kahneman-Tversky Optimization) on the local preference pairs.

Unpaired objective: each completion is labelled desirable/undesirable.
Runs full fine-tuning of the TinyMixtral v3.0 MoE on publish/imp-sft.
"""
import argparse
from pathlib import Path

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import KTOConfig, KTOTrainer


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--base", default="publish/imp-sft")
    p.add_argument("--data", default="data/dpo/kto_pairs.parquet")
    p.add_argument("--output-dir", default="dpo_local2/kto")
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--beta", type=float, default=0.1)
    p.add_argument("--epochs", type=float, default=3.0)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--grad-accum", type=int, default=4)
    p.add_argument("--max-length", type=int, default=1024)
    args = p.parse_args()

    tok = AutoTokenizer.from_pretrained(args.base, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.base, trust_remote_code=True, dtype=torch.bfloat16)
    ds = load_dataset("parquet", data_files=args.data, split="train")

    cfg = KTOConfig(
        output_dir=args.output_dir,
        learning_rate=args.lr,
        beta=args.beta,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        max_length=args.max_length,
        gradient_checkpointing=True,
        bf16=True,
        logging_steps=10,
        save_steps=250,
        save_total_limit=2,
        save_safetensors=False,
        report_to="none",
        seed=42,
        trust_remote_code=True,
        model_init_kwargs={"trust_remote_code": True},
    )
    trainer = KTOTrainer(model=model, args=cfg, train_dataset=ds,
                         processing_class=tok)
    trainer.train()
    trainer.save_model(str(Path(args.output_dir) / "final"))


if __name__ == "__main__":
    main()
