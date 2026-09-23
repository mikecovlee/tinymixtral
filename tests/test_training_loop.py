# Copyright (C) Michael Lee (李登淳) 2026. All rights reserved.
# Open-source under the MIT License. See LICENSE for details.

import math

import pytest
import torch

from model.config import TinyMixtralConfig
from model.modeling import TinyMixtralForCausalLM
from scripts.train_utils import make_adamw, make_cosine_schedule, make_val_evaluator, training_loop


def tiny_model(seed: int = 3) -> TinyMixtralForCausalLM:
    cfg = TinyMixtralConfig(
        hidden_size=32, num_hidden_layers=2, num_attention_heads=4,
        num_key_value_heads=2, head_dim=8, vocab_size=64,
        max_position_embeddings=64, num_local_experts=4,
        num_experts_per_tok=2, expert_intermediate_size=48,
        router_aux_loss_coef=0.01, router_jitter_noise=0.0,
        use_qk_norm=True,
    )
    torch.manual_seed(seed)
    return TinyMixtralForCausalLM(cfg)


def write_shards(tmp_path, n_shards: int, tokens_per_shard: int, prefix="train"):
    paths = []
    for i in range(n_shards):
        t = torch.randint(0, 64, (tokens_per_shard,))
        p = tmp_path / f"{prefix}_{i}.pt"
        torch.save(t, p)
        paths.append(p)
    return sorted(tmp_path.glob(f"{prefix}_*.pt"))


BS, SEQ = 2, 8
CHUNK = (SEQ + 1) * BS


def run_loop(tmp_path, files, max_steps, **kw):
    model = tiny_model().train()
    opt = make_adamw(model, lr=1e-3, weight_decay=0.0)
    sched = make_cosine_schedule(opt, warmup_steps=1, total_steps=max_steps)
    return training_loop(
        model, opt, sched, files, fi=0, ptr=0, total_tok=0,
        bs=BS, seq=SEQ, chunk=CHUNK,
        output_dir=str(tmp_path / "out"), max_steps=max_steps,
        save_every_min=10_000, log_every=1, step_start=0,
        schedule_args={"warmup_steps": 1, "total_steps": max_steps},
        keep_last_checkpoints=1, device="cpu", **kw,
    )


def test_data_exhaustion_stops_without_rewind(tmp_path, capsys):
    files = write_shards(tmp_path, n_shards=2, tokens_per_shard=CHUNK * 3)
    step, total_tok, fi, ptr, elapsed = run_loop(tmp_path, files, max_steps=10_000)
    out = capsys.readouterr().out
    assert step < 10_000
    assert "exhausted" in out.lower() or "耗尽" in out
    assert fi == len(files) - 1
    assert step == 6


def test_eval_fn_invoked_and_logs_val_ppl(tmp_path, capsys):
    files = write_shards(tmp_path, n_shards=1, tokens_per_shard=CHUNK * 2)
    calls = []

    def eval_fn(model):
        calls.append(model.training)
        return {"val_loss": 1.0, "val_ppl": math.e, "val_batches": 4}

    run_loop(tmp_path, files, max_steps=10_000, eval_fn=eval_fn, eval_every=1)
    out = capsys.readouterr().out
    assert calls and all(c for c in calls)
    assert "val_ppl=2.72" in out


def test_logs_ce_aux_and_utilization(tmp_path, capsys):
    files = write_shards(tmp_path, n_shards=1, tokens_per_shard=CHUNK * 2)
    run_loop(tmp_path, files, max_steps=3)
    out = capsys.readouterr().out
    assert "ce=" in out and "aux=" in out
    assert "util%=" in out and "min=" in out


def test_make_val_evaluator_restores_train_mode(tmp_path):
    files = write_shards(tmp_path, n_shards=1, tokens_per_shard=CHUNK * 4, prefix="val")
    model = tiny_model().train()
    eval_fn = make_val_evaluator(files, BS, SEQ, max_tokens=CHUNK * 2, device="cpu")
    metrics = eval_fn(model)
    assert metrics is not None and model.training
    assert metrics["val_batches"] == 2
    assert metrics["val_ppl"] > 1.0
