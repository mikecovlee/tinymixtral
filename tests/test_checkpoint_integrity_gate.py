import copy
import shutil
import tempfile
from pathlib import Path

import pytest
from safetensors.torch import load_file, save_file
import torch

from hf.configuration_tinymixtral import TinyMixtralConfig as HFTinyMixtralConfig
from hf.modeling_tinymixtral import TinyMixtralForCausalLM as HFTinyMixtralForCausalLM
from model.config import TinyMixtralConfig as NativeTinyMixtralConfig
from model.modeling import TinyMixtralForCausalLM as NativeTinyMixtralForCausalLM


_CPT_STATE_NAMES = (
    "projection",
    "anchors",
    "energy",
    "congestion_price",
    "state_version",
    "router_algorithm_version",
)


def _config_kwargs() -> dict:
    return {
        "vocab_size": 64,
        "hidden_size": 16,
        "num_hidden_layers": 2,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "head_dim": 4,
        "max_position_embeddings": 16,
        "num_local_experts": 4,
        "num_experts_per_tok": 2,
        "expert_intermediate_size": 24,
        "cpt_router_version": 1,
        "cpt_num_prototypes": 4,
        "cpt_projection_dim": 8,
        "cpt_rho_beta": 0.95,
        "cpt_beta_max": 0.45,
        "cpt_expert_temperature": 1.0,
        "cpt_state_radius": 1.0,
        "cpt_eps_z": 1e-6,
        "cpt_eps_m": 1e-6,
        "cpt_eps_init": 1e-8,
        "cpt_capacity_factor": 1.25,
        "cpt_init_seed": 17,
        "rms_norm_eps": 1e-6,
        "rope_theta": 10_000.0,
        "attention_dropout": 0.0,
        "tie_word_embeddings": True,
        "initializer_range": 0.02,
    }


@pytest.fixture
def hf_checkpoint_dir():
    tests_dir = Path(__file__).resolve().parent
    temp_root = tests_dir / ".tmp_checkpoint_integrity_gate"
    temp_root.mkdir(exist_ok=True)
    checkpoint_dir = Path(
        tempfile.mkdtemp(prefix="checkpoint_", dir=temp_root)
    )
    try:
        torch.manual_seed(20260731)
        model = HFTinyMixtralForCausalLM(
            HFTinyMixtralConfig(**_config_kwargs())
        )
        model.save_pretrained(checkpoint_dir, safe_serialization=False)
        yield checkpoint_dir
    finally:
        shutil.rmtree(checkpoint_dir, ignore_errors=True)
        try:
            temp_root.rmdir()
        except OSError:
            pass


def _checkpoint_state(checkpoint_dir: Path) -> dict:
    return torch.load(
        checkpoint_dir / "pytorch_model.bin",
        map_location="cpu",
        weights_only=True,
    )


def _replace_checkpoint_state(checkpoint_dir: Path, state: dict) -> None:
    torch.save(state, checkpoint_dir / "pytorch_model.bin")


@pytest.mark.parametrize("state_name", _CPT_STATE_NAMES)
def test_hf_from_pretrained_rejects_each_missing_cpt_state_item(
    hf_checkpoint_dir,
    state_name,
) -> None:
    state = _checkpoint_state(hf_checkpoint_dir)
    state.pop(f"layers.0.moe.cpt_router.{state_name}")
    _replace_checkpoint_state(hf_checkpoint_dir, state)

    with pytest.raises(
        RuntimeError,
        match="CPT checkpoint integrity validation failed",
    ):
        HFTinyMixtralForCausalLM.from_pretrained(
            hf_checkpoint_dir,
            local_files_only=True,
        )


def test_hf_from_pretrained_rejects_all_missing_cpt_state(
    hf_checkpoint_dir,
) -> None:
    state = _checkpoint_state(hf_checkpoint_dir)
    for key in tuple(state):
        if ".moe.cpt_router." in key:
            state.pop(key)
    _replace_checkpoint_state(hf_checkpoint_dir, state)

    with pytest.raises(
        RuntimeError,
        match="CPT checkpoint integrity validation failed",
    ):
        HFTinyMixtralForCausalLM.from_pretrained(
            hf_checkpoint_dir,
            local_files_only=True,
        )


def test_hf_from_pretrained_rejects_cpt_shape_mismatch_even_if_ignored(
    hf_checkpoint_dir,
) -> None:
    state = _checkpoint_state(hf_checkpoint_dir)
    key = "layers.0.moe.cpt_router.energy"
    state[key] = state[key][:-1].clone()
    _replace_checkpoint_state(hf_checkpoint_dir, state)

    with pytest.raises(RuntimeError, match="shape-mismatched CPT keys"):
        HFTinyMixtralForCausalLM.from_pretrained(
            hf_checkpoint_dir,
            local_files_only=True,
            ignore_mismatched_sizes=True,
        )


def test_hf_from_pretrained_rejects_unexpected_cpt_state(
    hf_checkpoint_dir,
) -> None:
    state = _checkpoint_state(hf_checkpoint_dir)
    state["layers.0.moe.cpt_router.obsolete_state"] = torch.zeros(1)
    _replace_checkpoint_state(hf_checkpoint_dir, state)

    with pytest.raises(RuntimeError, match="unexpected CPT keys"):
        HFTinyMixtralForCausalLM.from_pretrained(
            hf_checkpoint_dir,
            local_files_only=True,
        )


def test_hf_from_pretrained_rejects_legacy_linear_router_state(
    hf_checkpoint_dir,
) -> None:
    state = _checkpoint_state(hf_checkpoint_dir)
    state["layers.0.moe.router.weight"] = torch.zeros(4, 16)
    _replace_checkpoint_state(hf_checkpoint_dir, state)

    with pytest.raises(RuntimeError, match="Legacy Linear-router"):
        HFTinyMixtralForCausalLM.from_pretrained(
            hf_checkpoint_dir,
            local_files_only=True,
        )


def test_hf_from_pretrained_complete_checkpoint_preserves_return_semantics(
    hf_checkpoint_dir,
) -> None:
    model_only = HFTinyMixtralForCausalLM.from_pretrained(
        hf_checkpoint_dir,
        local_files_only=True,
    )
    assert isinstance(model_only, HFTinyMixtralForCausalLM)

    model, loading_info = HFTinyMixtralForCausalLM.from_pretrained(
        hf_checkpoint_dir,
        local_files_only=True,
        output_loading_info=True,
    )
    assert isinstance(model, HFTinyMixtralForCausalLM)
    assert loading_info["missing_keys"] == []
    assert loading_info["unexpected_keys"] == []
    assert loading_info["mismatched_keys"] == []


def test_native_low_level_load_rejects_strict_false() -> None:
    model = NativeTinyMixtralForCausalLM(
        NativeTinyMixtralConfig(**_config_kwargs())
    )
    state = copy.deepcopy(model.state_dict())
    state.pop("layers.0.moe.cpt_router.projection")

    with pytest.raises(RuntimeError, match="forbids strict=False"):
        model.load_state_dict(state, strict=False)


def test_hf_low_level_staged_partial_load_remains_available() -> None:
    model = HFTinyMixtralForCausalLM(
        HFTinyMixtralConfig(**_config_kwargs())
    )
    partial_state = {
        "embed_tokens.weight": model.embed_tokens.weight.detach().clone()
    }
    result = model.load_state_dict(partial_state, strict=False)
    assert result.missing_keys
    assert result.unexpected_keys == []


@pytest.mark.parametrize("model_kind", ["native", "hf"])
@pytest.mark.parametrize("assign", [False, True])
@pytest.mark.parametrize(
    "serialized_version",
    [
        pytest.param(torch.tensor(2, dtype=torch.int64), id="wrong-int"),
        pytest.param(torch.tensor(1.5, dtype=torch.float32), id="float"),
        pytest.param(torch.tensor(True, dtype=torch.bool), id="bool"),
        pytest.param(torch.tensor([1], dtype=torch.int64), id="vector"),
    ],
)
def test_low_level_load_rejects_invalid_router_algorithm_version(
    model_kind,
    assign,
    serialized_version,
) -> None:
    if model_kind == "native":
        model = NativeTinyMixtralForCausalLM(
            NativeTinyMixtralConfig(**_config_kwargs())
        )
    else:
        model = HFTinyMixtralForCausalLM(
            HFTinyMixtralConfig(**_config_kwargs())
        )
    state = copy.deepcopy(model.state_dict())
    state[
        "layers.0.moe.cpt_router.router_algorithm_version"
    ] = serialized_version.clone()

    with pytest.raises(
        RuntimeError,
        match="router_algorithm_version.*materialized int64 scalar",
    ):
        model.load_state_dict(state, strict=True, assign=assign)


@pytest.mark.parametrize(
    "serialized_version",
    [
        pytest.param(torch.tensor(2, dtype=torch.int64), id="wrong-int"),
        pytest.param(torch.tensor(1.5, dtype=torch.float32), id="float"),
        pytest.param(torch.tensor(True, dtype=torch.bool), id="bool"),
        pytest.param(torch.tensor([1], dtype=torch.int64), id="vector"),
    ],
)
def test_hf_from_pretrained_rejects_invalid_router_algorithm_version(
    hf_checkpoint_dir,
    serialized_version,
) -> None:
    state = _checkpoint_state(hf_checkpoint_dir)
    state[
        "layers.0.moe.cpt_router.router_algorithm_version"
    ] = serialized_version.clone()
    _replace_checkpoint_state(hf_checkpoint_dir, state)

    with pytest.raises(
        RuntimeError,
        match="router_algorithm_version.*materialized int64 scalar",
    ):
        HFTinyMixtralForCausalLM.from_pretrained(
            hf_checkpoint_dir,
            local_files_only=True,
        )


@pytest.mark.parametrize(
    "serialized_version",
    [
        pytest.param(torch.tensor(1.5, dtype=torch.float32), id="float"),
        pytest.param(torch.tensor(True, dtype=torch.bool), id="bool"),
    ],
)
def test_hf_safetensors_preflight_rejects_dtype_smuggled_algorithm_version(
    tmp_path,
    serialized_version,
) -> None:
    checkpoint_dir = tmp_path / "safetensors_identity"
    model = HFTinyMixtralForCausalLM(
        HFTinyMixtralConfig(**_config_kwargs())
    )
    model.save_pretrained(checkpoint_dir, safe_serialization=True)
    weights_path = checkpoint_dir / "model.safetensors"
    state = load_file(weights_path, device="cpu")
    state[
        "layers.0.moe.cpt_router.router_algorithm_version"
    ] = serialized_version.clone()
    save_file(state, weights_path, metadata={"format": "pt"})

    with pytest.raises(
        RuntimeError,
        match="router_algorithm_version.*materialized int64 scalar",
    ):
        HFTinyMixtralForCausalLM.from_pretrained(
            checkpoint_dir,
            local_files_only=True,
        )
