from pathlib import Path

import pytest
import torch

from scripts import publish_hf


def test_publication_paths_reject_checkpoint_overlap(tmp_path: Path) -> None:
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()

    with pytest.raises(RuntimeError, match="must not equal, contain"):
        publish_hf._validate_publication_paths(
            checkpoint,
            checkpoint / "publish",
            None,
            overwrite=False,
        )

    with pytest.raises(RuntimeError, match="must not equal, contain"):
        publish_hf._validate_publication_paths(
            checkpoint,
            tmp_path,
            None,
            overwrite=False,
        )


def test_publication_paths_require_explicit_overwrite(tmp_path: Path) -> None:
    checkpoint = tmp_path / "checkpoint"
    output = tmp_path / "publish"
    checkpoint.mkdir()
    output.mkdir()

    with pytest.raises(RuntimeError, match="--overwrite"):
        publish_hf._validate_publication_paths(
            checkpoint,
            output,
            None,
            overwrite=False,
        )

    actual_checkpoint, actual_output, actual_tokenizer = (
        publish_hf._validate_publication_paths(
            checkpoint,
            output,
            None,
            overwrite=True,
        )
    )
    assert actual_checkpoint == checkpoint.resolve()
    assert actual_output == output.resolve()
    assert actual_tokenizer is None


def test_tokenizer_copy_rejects_case_insensitive_artifact_collision(
    tmp_path: Path,
) -> None:
    output = tmp_path / "staging"
    tokenizer = tmp_path / "tokenizer"
    output.mkdir()
    tokenizer.mkdir()
    (output / "config.json").write_text("model", encoding="utf-8")
    (tokenizer / "CONFIG.JSON").write_text("tokenizer", encoding="utf-8")

    with pytest.raises(RuntimeError, match="would overwrite"):
        publish_hf._copy_tokenizer_files(tokenizer, output)

    assert (output / "config.json").read_text(encoding="utf-8") == "model"


def test_tokenizer_copy_preserves_noncolliding_files(tmp_path: Path) -> None:
    output = tmp_path / "staging"
    tokenizer = tmp_path / "tokenizer"
    output.mkdir()
    tokenizer.mkdir()
    (output / "config.json").write_text("model", encoding="utf-8")
    (tokenizer / "tokenizer.json").write_text("tokens", encoding="utf-8")

    copied = publish_hf._copy_tokenizer_files(tokenizer, output)

    assert copied == ("tokenizer.json",)
    assert (output / "tokenizer.json").read_text(encoding="utf-8") == "tokens"


def test_tokenizer_copy_rejects_reparse_like_entry(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output = tmp_path / "staging"
    tokenizer = tmp_path / "tokenizer"
    output.mkdir()
    tokenizer.mkdir()
    source = tokenizer / "tokenizer.json"
    source.write_text("tokens", encoding="utf-8")

    original = publish_hf._is_link_or_reparse_point
    monkeypatch.setattr(
        publish_hf,
        "_is_link_or_reparse_point",
        lambda path: Path(path) == source or original(path),
    )
    with pytest.raises(RuntimeError, match="reparse points"):
        publish_hf._copy_tokenizer_files(tokenizer, output)

    assert not (output / "tokenizer.json").exists()


def test_publication_paths_reject_reparse_root(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    checkpoint = tmp_path / "checkpoint"
    tokenizer = tmp_path / "tokenizer"
    checkpoint.mkdir()
    tokenizer.mkdir()
    original = publish_hf._is_link_or_reparse_point
    monkeypatch.setattr(
        publish_hf,
        "_is_link_or_reparse_point",
        lambda path: Path(path) == tokenizer or original(path),
    )

    with pytest.raises(RuntimeError, match="tokenizer path.*reparse point"):
        publish_hf._validate_publication_paths(
            checkpoint,
            tmp_path / "publish",
            tokenizer,
            overwrite=False,
        )


def test_tokenizer_copy_rejects_silently_ignored_subdirectories(
    tmp_path: Path,
) -> None:
    output = tmp_path / "staging"
    tokenizer = tmp_path / "tokenizer"
    output.mkdir()
    tokenizer.mkdir()
    (tokenizer / "nested").mkdir()

    with pytest.raises(RuntimeError, match="flat directory"):
        publish_hf._copy_tokenizer_files(tokenizer, output)


def test_verified_staging_atomically_replaces_existing_output(
    tmp_path: Path,
) -> None:
    output = tmp_path / "publish"
    staging = tmp_path / ".publish.staging-test"
    output.mkdir()
    staging.mkdir()
    (output / "stale.bin").write_text("old", encoding="utf-8")
    (staging / "pytorch_model.bin").write_text("new", encoding="utf-8")

    publish_hf._promote_staging_directory(
        staging,
        output,
        overwrite=True,
    )

    assert not staging.exists()
    assert not (output / "stale.bin").exists()
    assert (output / "pytorch_model.bin").read_text(encoding="utf-8") == "new"
    assert not list(tmp_path.glob(".publish.backup-*"))
    assert not publish_hf._publication_lock_path(output).exists()


def test_failed_promotion_restores_existing_output(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output = tmp_path / "publish"
    staging = tmp_path / ".publish.staging-test"
    output.mkdir()
    staging.mkdir()
    (output / "stale.bin").write_text("old", encoding="utf-8")
    (staging / "pytorch_model.bin").write_text("new", encoding="utf-8")

    original_replace = Path.replace

    def fail_staging_replace(path: Path, target: Path):
        if path == staging:
            raise OSError("injected staging promotion failure")
        return original_replace(path, target)

    monkeypatch.setattr(Path, "replace", fail_staging_replace)
    with pytest.raises(OSError, match="injected staging promotion failure"):
        publish_hf._promote_staging_directory(
            staging,
            output,
            overwrite=True,
        )

    assert staging.is_dir()
    assert (staging / "pytorch_model.bin").read_text(encoding="utf-8") == "new"
    assert (output / "stale.bin").read_text(encoding="utf-8") == "old"
    assert not list(tmp_path.glob(".publish.backup-*"))
    assert not publish_hf._publication_lock_path(output).exists()


def test_existing_publication_lock_prevents_output_mutation(
    tmp_path: Path,
) -> None:
    output = tmp_path / "publish"
    staging = tmp_path / ".publish.staging-test"
    output.mkdir()
    staging.mkdir()
    (output / "stale.bin").write_text("old", encoding="utf-8")
    (staging / "pytorch_model.bin").write_text("new", encoding="utf-8")
    lock = publish_hf._publication_lock_path(output)
    lock.mkdir()

    with pytest.raises(RuntimeError, match="output lock"):
        publish_hf._promote_staging_directory(
            staging,
            output,
            overwrite=True,
        )

    assert (output / "stale.bin").read_text(encoding="utf-8") == "old"
    assert (staging / "pytorch_model.bin").read_text(encoding="utf-8") == "new"
    assert lock.is_dir()


def test_failed_rollback_retains_backup_and_lock_for_manual_recovery(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output = tmp_path / "publish"
    staging = tmp_path / ".publish.staging-test"
    output.mkdir()
    staging.mkdir()
    (output / "stale.bin").write_text("old", encoding="utf-8")
    (staging / "pytorch_model.bin").write_text("new", encoding="utf-8")

    original_replace = Path.replace

    def fail_promotion_and_rollback(path: Path, target: Path):
        if path == staging or ".backup-" in path.name:
            raise OSError("injected rename failure")
        return original_replace(path, target)

    monkeypatch.setattr(Path, "replace", fail_promotion_and_rollback)
    with pytest.raises(RuntimeError, match="retained at"):
        publish_hf._promote_staging_directory(
            staging,
            output,
            overwrite=True,
        )

    backups = list(tmp_path.glob(".publish.backup-*"))
    assert len(backups) == 1
    assert (backups[0] / "stale.bin").read_text(encoding="utf-8") == "old"
    assert (staging / "pytorch_model.bin").read_text(encoding="utf-8") == "new"
    assert publish_hf._publication_lock_path(output).is_dir()


def test_state_dict_round_trip_requires_every_tensor_to_match() -> None:
    source = {
        "used.weight": torch.tensor([1.0, 2.0]),
        "unused_expert.weight": torch.tensor([3.0, 4.0]),
    }
    publish_hf._assert_state_dict_exact(
        source,
        {key: value.clone() for key, value in source.items()},
        "round-trip",
    )

    changed = {key: value.clone() for key, value in source.items()}
    changed["unused_expert.weight"][0] = -3.0
    with pytest.raises(RuntimeError, match="unused_expert.weight"):
        publish_hf._assert_state_dict_exact(source, changed, "round-trip")

    wrong_dtype = {key: value.clone() for key, value in source.items()}
    wrong_dtype["unused_expert.weight"] = wrong_dtype[
        "unused_expert.weight"
    ].to(torch.float64)
    with pytest.raises(RuntimeError, match="unused_expert.weight"):
        publish_hf._assert_state_dict_exact(source, wrong_dtype, "round-trip")


def test_source_checkpoint_load_preserves_mixed_tensor_dtypes_and_weight_tying() -> None:
    config = publish_hf.TinyMixtralConfig(
        vocab_size=32,
        hidden_size=16,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=8,
        max_position_embeddings=16,
        num_local_experts=2,
        num_experts_per_tok=1,
        expert_intermediate_size=24,
        cpt_num_prototypes=2,
        cpt_projection_dim=8,
    )
    source = publish_hf.TinyMixtralForCausalLM(config).to(torch.bfloat16)
    source_state = {
        key: value.detach().clone()
        for key, value in source.state_dict().items()
    }
    target = publish_hf.TinyMixtralForCausalLM(config)

    publish_hf._load_state_dict_preserving_source_dtype(
        target,
        source_state,
        "source checkpoint",
    )

    assert target.lm_head.weight is target.embed_tokens.weight
    publish_hf._assert_state_dict_exact(
        source_state,
        target.state_dict(),
        "source checkpoint",
    )
    assert target.embed_tokens.weight.dtype == torch.bfloat16
    router = target.layers[0].moe.cpt_router
    assert router.projection.dtype == torch.float32
    assert router.anchors.dtype == torch.float32
    assert router.energy.dtype == torch.float32
    assert router.congestion_price.dtype == torch.float32
