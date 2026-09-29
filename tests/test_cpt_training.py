"""Native CPT optimizer/transaction integration using bounded synthetic data."""
from pathlib import Path
from unittest.mock import patch

import pytest
import torch

from model.config import TinyMixtralConfig
from model.modeling import TinyMixtralForCausalLM
from scripts.train_utils import make_adamw, make_cosine_schedule, save_checkpoint, training_loop


def model_and_optimizer():
    torch.manual_seed(7)
    config = TinyMixtralConfig(
        vocab_size=32, hidden_size=16, num_hidden_layers=1,
        num_attention_heads=2, num_key_value_heads=1, head_dim=8,
        num_local_experts=4, expert_intermediate_size=24,
        cpt_state_chunk_size=2, use_qk_norm=True,
    )
    model = TinyMixtralForCausalLM(config)
    model.gradient_checkpointing_enable()
    optimizer = make_adamw(model, lr=1e-3, weight_decay=0.0)
    scheduler = make_cosine_schedule(optimizer, warmup_steps=0, total_steps=3)
    return model, optimizer, scheduler


def run_steps(model, optimizer, scheduler, shard, directory, *, max_steps, step=0, total=0, fi=0, ptr=0):
    return training_loop(
        model, optimizer, scheduler, [shard], fi=fi, ptr=ptr, total_tok=total,
        bs=1, seq=4, chunk=5, output_dir=str(directory), max_steps=max_steps,
        save_every_min=10000, log_every=100, step_start=step,
        schedule_args={'warmup_steps': 0, 'total_steps': 3}, device='cpu',
    )


def test_optimizer_commit_checkpoint_and_resume(tmp_path):
    torch.set_num_threads(1)
    shard = tmp_path / 'train_0.pt'
    torch.save(torch.arange(20) % 32, shard)
    model, optimizer, scheduler = model_and_optimizer()
    step, total, fi, ptr, _ = run_steps(model, optimizer, scheduler, shard, tmp_path/'run', max_steps=2)
    assert step == model.get_cpt_optimizer_step() == model.get_cpt_state_version() == 2
    checkpoint = save_checkpoint(model, optimizer, scheduler, str(tmp_path/'run'),
                                 step, total, 0, 3, fi, ptr, 1, 4)
    restored = TinyMixtralForCausalLM.from_pretrained(str(checkpoint))
    for name, tensor in model.state_dict().items():
        torch.testing.assert_close(restored.state_dict()[name], tensor, rtol=0, atol=0)
    saved = torch.load(Path(checkpoint)/'training_state.pt', weights_only=True)
    resumed_optimizer = make_adamw(restored, lr=1e-3, weight_decay=0.0)
    resumed_scheduler = make_cosine_schedule(resumed_optimizer, warmup_steps=0, total_steps=3)
    resumed_optimizer.load_state_dict(saved['opt'])
    resumed_scheduler.load_state_dict(saved['sched'])
    result = run_steps(restored, resumed_optimizer, resumed_scheduler, shard,
                       tmp_path/'resumed', max_steps=3, step=step, total=total, fi=fi, ptr=ptr)
    assert result[0] == restored.get_cpt_optimizer_step() == restored.get_cpt_state_version() == 3


def test_failed_optimizer_does_not_commit_cpt_state(tmp_path):
    torch.set_num_threads(1)
    shard = tmp_path/'train_0.pt'
    torch.save(torch.arange(10) % 32, shard)
    model, optimizer, scheduler = model_and_optimizer()
    with patch.object(optimizer, 'step', side_effect=RuntimeError('optimizer failed')):
        with pytest.raises(RuntimeError, match='optimizer failed'):
            run_steps(model, optimizer, scheduler, shard, tmp_path/'run', max_steps=1)
    assert model.get_cpt_optimizer_step() == model.get_cpt_state_version() == 0
