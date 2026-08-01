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
import atexit
import json
import shutil
import stat
import sys
import tempfile
import uuid
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from hf.configuration_tinymixtral import TinyMixtralConfig
from hf.modeling_tinymixtral import TinyMixtralForCausalLM


CONFIG_FIELDS = (
    "vocab_size",
    "hidden_size",
    "num_hidden_layers",
    "num_attention_heads",
    "num_key_value_heads",
    "head_dim",
    "max_position_embeddings",
    "num_local_experts",
    "num_experts_per_tok",
    "expert_intermediate_size",
    "cpt_router_version",
    "cpt_router_recompute",
    "cpt_num_prototypes",
    "cpt_projection_dim",
    "cpt_rho_beta",
    "cpt_beta_max",
    "cpt_expert_temperature",
    "cpt_state_radius",
    "cpt_eps_z",
    "cpt_eps_m",
    "cpt_eps_init",
    "cpt_capacity_factor",
    "cpt_init_seed",
    "rms_norm_eps",
    "rope_theta",
    "attention_dropout",
    "tie_word_embeddings",
    "initializer_range",
)


def _resolved(path):
    return Path(path).expanduser().resolve(strict=False)


def _paths_overlap(first, second):
    first = _resolved(first)
    second = _resolved(second)
    return (
        first == second
        or first in second.parents
        or second in first.parents
    )


def _validate_publication_paths(checkpoint, output, tokenizer, overwrite):
    raw_checkpoint = Path(checkpoint).expanduser()
    raw_output = Path(output).expanduser()
    raw_tokenizer = Path(tokenizer).expanduser() if tokenizer is not None else None
    for label, path in (
        ("checkpoint", raw_checkpoint),
        ("output", raw_output),
        ("tokenizer", raw_tokenizer),
    ):
        if path is not None and path.exists() and _is_link_or_reparse_point(path):
            raise RuntimeError(
                f"Publication {label} path must not be a symbolic link, "
                f"junction, or other reparse point: {path}"
            )

    checkpoint = _resolved(raw_checkpoint)
    output = _resolved(raw_output)
    tokenizer = _resolved(raw_tokenizer) if raw_tokenizer is not None else None

    if not checkpoint.is_dir():
        raise RuntimeError(f"Checkpoint directory does not exist: {checkpoint}")
    if _paths_overlap(checkpoint, output):
        raise RuntimeError(
            "Publication output must not equal, contain, or be contained by "
            f"the checkpoint directory: checkpoint={checkpoint}, output={output}"
        )
    if tokenizer is not None:
        if not tokenizer.is_dir():
            raise RuntimeError(f"Tokenizer directory does not exist: {tokenizer}")
        if _paths_overlap(tokenizer, output):
            raise RuntimeError(
                "Publication output must not equal, contain, or be contained "
                f"by the tokenizer directory: tokenizer={tokenizer}, "
                f"output={output}"
            )
    if output.exists():
        if not output.is_dir():
            raise RuntimeError(f"Publication output exists and is not a directory: {output}")
        if not overwrite:
            raise RuntimeError(
                f"Publication output already exists: {output}. Pass "
                "--overwrite to replace it only after staging verification succeeds."
            )
    return checkpoint, output, tokenizer


def _create_staging_directory(output):
    output.parent.mkdir(parents=True, exist_ok=True)
    return Path(
        tempfile.mkdtemp(
            prefix=f".{output.name}.staging-",
            dir=output.parent,
        )
    )


def _cleanup_tree(path):
    path = Path(path)
    if path.is_symlink():
        path.unlink()
    elif path.exists():
        shutil.rmtree(path)


def _is_link_or_reparse_point(path):
    """Return whether a tokenizer entry can redirect file access elsewhere."""
    path = Path(path)
    if path.is_symlink():
        return True
    is_junction = getattr(path, "is_junction", None)
    if is_junction is not None and is_junction():
        return True
    attributes = getattr(path.lstat(), "st_file_attributes", 0)
    reparse_flag = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0)
    return bool(attributes & reparse_flag)


def _copy_tokenizer_files(tokenizer, output):
    existing_names = {
        child.name.casefold(): child.name
        for child in output.iterdir()
    }
    copied = []
    for source in sorted(tokenizer.iterdir(), key=lambda path: path.name.casefold()):
        if _is_link_or_reparse_point(source):
            raise RuntimeError(
                "Tokenizer publication refuses symbolic links, junctions, "
                f"and other reparse points: {source}"
            )
        if not source.is_file():
            raise RuntimeError(
                "Tokenizer publication currently requires a flat directory of "
                f"regular files; unsupported entry: {source}"
            )
        collision = existing_names.get(source.name.casefold())
        if collision is not None:
            raise RuntimeError(
                "Tokenizer file would overwrite a model publication artifact: "
                f"{source.name} conflicts with {collision}"
            )
        shutil.copy2(source, output / source.name)
        existing_names[source.name.casefold()] = source.name
        copied.append(source.name)
    return tuple(copied)


def _remove_path(path):
    """Remove one exact path without following a symbolic-link target."""
    path = Path(path)
    if path.is_symlink() or path.is_file():
        path.unlink()
    elif path.exists():
        shutil.rmtree(path)


def _publication_lock_path(output):
    output = Path(output)
    return output.with_name(f".{output.name}.publish-lock")


def _acquire_publication_lock(output):
    lock = _publication_lock_path(output)
    try:
        lock.mkdir()
    except FileExistsError as error:
        raise RuntimeError(
            "Another publication or an interrupted promotion owns the output "
            f"lock: {lock}. Refusing to modify the current output."
        ) from error
    return lock


def _promote_staging_directory(staging, output, overwrite):
    staging = Path(staging)
    output = Path(output)
    if not staging.is_dir():
        raise RuntimeError(f"Publication staging directory is missing: {staging}")
    lock = _acquire_publication_lock(output)
    retain_lock = False
    try:
        if not output.exists():
            staging.replace(output)
            return
        if not overwrite:
            raise RuntimeError(f"Publication output already exists: {output}")

        backup = output.with_name(f".{output.name}.backup-{uuid.uuid4().hex}")
        output.replace(backup)
        try:
            staging.replace(output)
        except BaseException as promotion_error:
            try:
                _remove_path(output)
                backup.replace(output)
            except BaseException as rollback_error:
                retain_lock = True
                raise RuntimeError(
                    "Publication promotion failed and the previous output "
                    "could not be restored automatically. The backup is "
                    f"retained at {backup}"
                ) from rollback_error
            add_note = getattr(promotion_error, "add_note", None)
            if add_note is not None:
                add_note(
                    "The previous publication output was restored successfully."
                )
            raise
        try:
            shutil.rmtree(backup)
        except BaseException as error:
            raise RuntimeError(
                "Publication succeeded but the previous-output backup could "
                f"not be removed: {backup}"
            ) from error
    finally:
        if not retain_lock:
            try:
                lock.rmdir()
            except FileNotFoundError:
                pass

def _reject_incompatible_checkpoint(state_dict):
    legacy_router_keys = sorted(
        key for key in state_dict if key.endswith(".moe.router.weight")
    )
    has_cpt_router = any(".moe.cpt_router." in key for key in state_dict)
    if legacy_router_keys or not has_cpt_router:
        detail = (
            f"; found legacy keys such as {legacy_router_keys[0]}"
            if legacy_router_keys
            else "; no CPT router state was found"
        )
        raise RuntimeError(
            "Legacy Linear-router checkpoints cannot be exported as CPT-MoE "
            "models. CPT router v1 requires a native CPT checkpoint and does "
            f"not permit partial or strict=False loading{detail}."
        )


def _require_complete_config(payload):
    """Reject an incomplete canonical config before publication output."""
    missing = [name for name in CONFIG_FIELDS if name not in payload]
    if missing:
        raise RuntimeError(
            "CPT checkpoint config is incomplete; missing required canonical "
            "fields: " + ", ".join(missing)
        )


def _normalize_source_config(payload):
    """Add only the execution-policy default needed by older CPT configs."""
    normalized = dict(payload)
    normalized.setdefault("cpt_router_recompute", "global")
    return normalized


def _canonical_config(config):
    """Return the exact published config surface used for parity."""
    return {name: getattr(config, name) for name in CONFIG_FIELDS}


def _assert_canonical_config_equal(source, candidate, context):
    source_values = _canonical_config(source)
    candidate_values = _canonical_config(candidate)
    mismatches = {
        name: (source_values[name], candidate_values[name])
        for name in CONFIG_FIELDS
        if source_values[name] != candidate_values[name]
    }
    if mismatches:
        raise RuntimeError(
            f"{context} changed canonical config fields: {mismatches}"
        )


def _assert_state_dict_exact(source, candidate, context):
    """Require an exact serialization round-trip for every tensor and buffer."""
    source_keys = set(source)
    candidate_keys = set(candidate)
    if source_keys != candidate_keys:
        raise RuntimeError(
            f"{context} state-dict key mismatch: "
            f"missing={sorted(source_keys - candidate_keys)}, "
            f"unexpected={sorted(candidate_keys - source_keys)}"
        )
    for key in sorted(source_keys):
        source_value = source[key]
        candidate_value = candidate[key]
        if not isinstance(source_value, torch.Tensor) or not isinstance(
            candidate_value,
            torch.Tensor,
        ):
            raise RuntimeError(
                f"{context} state entry is not a tensor: {key}"
            )
        if (
            source_value.shape != candidate_value.shape
            or source_value.dtype != candidate_value.dtype
            or source_value.layout != candidate_value.layout
            or not torch.equal(source_value, candidate_value)
        ):
            raise RuntimeError(
                f"{context} changed serialized state tensor: {key}; "
                f"source(shape={tuple(source_value.shape)}, "
                f"dtype={source_value.dtype}, layout={source_value.layout}), "
                f"candidate(shape={tuple(candidate_value.shape)}, "
                f"dtype={candidate_value.dtype}, "
                f"layout={candidate_value.layout})"
            )


def _load_state_dict_preserving_source_dtype(model, state_dict, context):
    """Load every source tensor without silently casting checkpoint dtypes."""
    model.load_state_dict(state_dict, strict=True, assign=True)
    model.tie_weights()
    _assert_state_dict_exact(state_dict, model.state_dict(), context)
    return model


def _require_consistent_cpt_state_version(model, context):
    """Require one materialized persistent CPT version across every layer."""
    try:
        return model.get_cpt_state_version()
    except RuntimeError as error:
        raise RuntimeError(
            f"{context} has inconsistent CPT layer state versions"
        ) from error


def main():
    p = argparse.ArgumentParser(description="Package TinyMixtral for HF Hub")
    p.add_argument("--checkpoint", required=True, help="源 checkpoint 路径")
    p.add_argument("--output", default="publish", help="输出目录")
    p.add_argument("--tokenizer", default=None, help="可选：Tokenizer 目录")
    p.add_argument(
        "--overwrite",
        action="store_true",
        help=(
            "Replace an existing output only after a fresh staging directory "
            "passes all publication verification"
        ),
    )
    args = p.parse_args()

    ckpt, final_output, tokenizer = _validate_publication_paths(
        args.checkpoint,
        args.output,
        args.tokenizer,
        args.overwrite,
    )
    bin_file = ckpt / "pytorch_model.bin"
    if not bin_file.exists():
        print(f"ERROR: {bin_file} not found"); sys.exit(1)

    # 1. 从旧 config 读取参数，创建 HF 兼容 config
    cfg_file = ckpt / "config.json"
    if cfg_file.exists():
        with open(cfg_file) as f:
            old = _normalize_source_config(json.load(f))
    else:
        raise RuntimeError(f"CPT checkpoint config is missing: {cfg_file}")

    # 2. 加载权重
    print(f"Loading {bin_file} ...")
    state_dict = torch.load(bin_file, map_location="cpu", weights_only=True)
    _reject_incompatible_checkpoint(state_dict)

    _require_complete_config(old)
    valid = {key: old[key] for key in CONFIG_FIELDS}
    config = TinyMixtralConfig(**valid)
    source_canonical = _canonical_config(config)

    model = TinyMixtralForCausalLM(config)
    _load_state_dict_preserving_source_dtype(
        model,
        state_dict,
        "source checkpoint load",
    )
    source_state_version = _require_consistent_cpt_state_version(
        model,
        "source checkpoint",
    )
    model.eval()
    nM = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"Loaded {nM:.0f}M params, {len(state_dict)} keys")

    output = _create_staging_directory(final_output)
    atexit.register(_cleanup_tree, output)

    # 3. save_pretrained + 补充 auto_map
    model.save_pretrained(str(output), safe_serialization=False)
    cfg_path = output / "config.json"
    cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    # ``PretrainedConfig.save_pretrained`` serializes a diff and may omit a
    # behavior-critical value when it equals a Transformers default (for
    # example ``tie_word_embeddings=True``).  Publish the complete canonical
    # surface explicitly so remote code never depends on library defaults.
    cfg.update(source_canonical)
    cfg["auto_map"] = {
        "AutoConfig": "configuration_tinymixtral.TinyMixtralConfig",
        "AutoModelForCausalLM": "modeling_tinymixtral.TinyMixtralForCausalLM",
    }
    cfg_path.write_text(json.dumps(cfg, indent=2), encoding="utf-8")
    _require_complete_config(cfg)
    published_config = TinyMixtralConfig(
        **{name: cfg[name] for name in CONFIG_FIELDS}
    )
    _assert_canonical_config_equal(
        config,
        published_config,
        "serialized Hugging Face config",
    )
    if _canonical_config(published_config) != source_canonical:
        raise RuntimeError(
            "Serialized Hugging Face config did not preserve the canonical "
            "source configuration"
        )
    print(f"Saved {output}/pytorch_model.bin + config.json (auto_map)")

    # 4. 复制兼容层代码 + LICENSE 到 publish
    root = Path(__file__).parent.parent
    hf_dir = root / "hf"
    for name in ("configuration_tinymixtral.py", "modeling_tinymixtral.py"):
        shutil.copy(hf_dir / name, output / name)
    shutil.copy(root / "model" / "cpt_router.py", output / "cpt_router.py")
    shutil.copy(root / "LICENSE", output / "LICENSE")
    print(
        "Copied configuration_tinymixtral.py + modeling_tinymixtral.py + "
        "cpt_router.py + LICENSE"
    )

    # 5. tokenizer
    if tokenizer is not None:
        copied_tokenizer_files = _copy_tokenizer_files(tokenizer, output)
        print(
            f"Copied {len(copied_tokenizer_files)} tokenizer files from "
            f"{tokenizer}"
        )

    # 6. 验证
    print("\nVerifying AutoModelForCausalLM.from_pretrained() ...")
    from transformers import AutoModelForCausalLM
    m = AutoModelForCausalLM.from_pretrained(
        str(output),
        trust_remote_code=True,
        dtype="auto",
    )
    m.eval()
    reloaded_state_version = _require_consistent_cpt_state_version(
        m,
        "reloaded Hugging Face model",
    )
    if reloaded_state_version != source_state_version:
        raise RuntimeError(
            "Reloaded CPT state version differs from the source checkpoint: "
            f"source={source_state_version}, reloaded={reloaded_state_version}"
        )
    _assert_canonical_config_equal(
        config,
        m.config,
        "reloaded Hugging Face config",
    )
    nM2 = sum(p.numel() for p in m.parameters()) / 1e6
    if abs(nM - nM2) >= 1:
        raise RuntimeError(
            f"Reloaded parameter-count mismatch: source={nM}M, reloaded={nM2}M"
        )
    print(f"OK: {nM2:.0f}M params via AutoModelForCausalLM")

    source_state = model.state_dict()
    reloaded_state = m.state_dict()
    _assert_state_dict_exact(
        source_state,
        reloaded_state,
        "reloaded Hugging Face model",
    )

    generator = torch.Generator(device="cpu").manual_seed(20260730)
    length = min(16, config.max_position_embeddings)
    x = torch.randint(
        0,
        config.vocab_size,
        (2, length),
        generator=generator,
    )
    attention_mask = torch.ones_like(x)
    if length >= 4:
        attention_mask[0, :2] = 0
        x[0, :2] = 0
    segment_ids = torch.zeros_like(x)
    if length >= 4:
        segment_ids[:, length // 2 :] = 1
    with torch.inference_mode():
        expected = model(
            input_ids=x,
            attention_mask=attention_mask,
            cpt_segment_ids=segment_ids,
        )
        actual = m(
            input_ids=x,
            attention_mask=attention_mask,
            cpt_segment_ids=segment_ids,
        )
    torch.testing.assert_close(actual.logits, expected.logits, rtol=0, atol=0)
    torch.testing.assert_close(actual["logits"], actual.logits, rtol=0, atol=0)
    for expected_state, actual_state in zip(
        expected.cpt_sequence_states,
        actual.cpt_sequence_states,
    ):
        torch.testing.assert_close(
            actual_state.state_s,
            expected_state.state_s,
            rtol=0,
            atol=0,
        )
        torch.testing.assert_close(
            actual_state.state_nu,
            expected_state.state_nu,
            rtol=0,
            atol=0,
        )
        if not torch.equal(actual_state.initialized, expected_state.initialized):
            raise RuntimeError(
                "Reloaded continuation initialized flags differ from source"
            )
        if not torch.equal(
            actual_state.state_version,
            expected_state.state_version,
        ):
            raise RuntimeError(
                "Reloaded continuation state versions differ from source"
            )

    continuation_length = min(3, config.max_position_embeddings)
    continuation_input = torch.randint(
        0,
        config.vocab_size,
        (2, continuation_length),
        generator=generator,
    )
    with torch.inference_mode():
        expected_continuation = model(
            input_ids=continuation_input,
            cpt_sequence_states=expected.cpt_sequence_states,
        )
        actual_continuation = m(
            input_ids=continuation_input,
            cpt_sequence_states=actual.cpt_sequence_states,
        )
    torch.testing.assert_close(
        actual_continuation.logits,
        expected_continuation.logits,
        rtol=0,
        atol=0,
    )
    for expected_state, actual_state in zip(
        expected_continuation.cpt_sequence_states,
        actual_continuation.cpt_sequence_states,
    ):
        torch.testing.assert_close(
            actual_state.state_s,
            expected_state.state_s,
            rtol=0,
            atol=0,
        )
        torch.testing.assert_close(
            actual_state.state_nu,
            expected_state.state_nu,
            rtol=0,
            atol=0,
        )

    if config.max_position_embeddings >= 2:
        prompt_length = min(3, config.max_position_embeddings - 1)
        generation_steps = min(
            2,
            config.max_position_embeddings - prompt_length,
        )
        generation_input = x[1:2, :prompt_length]
        with torch.inference_mode():
            generated = m.generate(
                generation_input,
                max_new_tokens=generation_steps,
                do_sample=False,
                pad_token_id=0,
                eos_token_id=None,
            )
            manual = generation_input.clone()
            for _ in range(generation_steps):
                next_token = m(manual).logits[:, -1].argmax(
                    dim=-1,
                    keepdim=True,
                )
                manual = torch.cat((manual, next_token), dim=1)
        if not torch.equal(generated, manual):
            raise RuntimeError(
                "Reloaded greedy generation differs from manual greedy decoding"
            )

    persistent_cpt_keys = sorted(
        key
        for key in source_state
        if ".moe.cpt_router." in key
        and ("price" in key or "version" in key)
    )
    for key in persistent_cpt_keys:
        torch.testing.assert_close(reloaded_state[key], source_state[key])
        if "price" in key and source_state[key].is_floating_point():
            if reloaded_state[key].dtype != torch.float32:
                raise RuntimeError(
                    f"Persistent congestion price must remain FP32: {key} is "
                    f"{reloaded_state[key].dtype}"
                )
    print(
        "OK: strict state keys + packed/masked forward + continuation + "
        "generate + CPT persistent state round-trip"
    )

    _promote_staging_directory(output, final_output, args.overwrite)
    atexit.unregister(_cleanup_tree)
    output = final_output

    print(f"\n{'='*60}")
    print(f"Ready: {output}/")
    print(f"  from transformers import AutoModelForCausalLM")
    print(f"  model = AutoModelForCausalLM.from_pretrained('{output}/', trust_remote_code=True)")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
