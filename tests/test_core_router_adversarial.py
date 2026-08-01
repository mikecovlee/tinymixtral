import copy
from dataclasses import replace

import pytest
import torch

from hf.configuration_tinymixtral import TinyMixtralConfig as HFConfig
from hf.modeling_tinymixtral import TinyMixtralForCausalLM as HFModel
from model.config import TinyMixtralConfig as NativeConfig
from model.cpt_router import CPTRouter


CONFIG_KWARGS = {
    "vocab_size": 32,
    "hidden_size": 16,
    "num_hidden_layers": 1,
    "num_attention_heads": 4,
    "num_key_value_heads": 2,
    "head_dim": 4,
    "max_position_embeddings": 16,
    "num_local_experts": 4,
    "num_experts_per_tok": 2,
    "expert_intermediate_size": 24,
    "cpt_num_prototypes": 4,
    "cpt_projection_dim": 8,
    "cpt_init_seed": 41,
}


def test_sequence_state_rejects_future_router_version_provenance() -> None:
    router = CPTRouter(NativeConfig(**CONFIG_KWARGS), layer_index=0).eval()
    state = router(torch.randn(1, 2, 16)).sequence_state
    future_state = replace(
        state,
        state_version=state.state_version + 1,
    )

    with pytest.raises(ValueError, match="cannot be newer"):
        router(torch.randn(1, 1, 16), sequence_state=future_state)


@pytest.mark.parametrize("tie_word_embeddings", [True, False])
def test_hf_default_safe_serialization_round_trip(
    tmp_path,
    tie_word_embeddings,
) -> None:
    config = HFConfig(
        **dict(CONFIG_KWARGS, tie_word_embeddings=tie_word_embeddings)
    )
    source = HFModel(config).eval()
    output_dir = tmp_path / "safe_checkpoint"

    source.save_pretrained(output_dir)
    assert (output_dir / "model.safetensors").is_file()
    restored, loading_info = HFModel.from_pretrained(
        output_dir,
        local_files_only=True,
        output_loading_info=True,
    )

    assert loading_info["missing_keys"] == []
    assert loading_info["unexpected_keys"] == []
    assert loading_info["mismatched_keys"] == []
    assert loading_info["error_msgs"] == []
    assert set(source.state_dict()) == set(restored.state_dict())
    for key, value in source.state_dict().items():
        torch.testing.assert_close(value, restored.state_dict()[key], rtol=0, atol=0)


@pytest.mark.parametrize(
    ("loading_info", "message"),
    [
        (
            {
                "missing_keys": ["layers.0.self_attn.q_proj.weight"],
                "unexpected_keys": [],
                "mismatched_keys": [],
                "error_msgs": [],
            },
            "missing model keys",
        ),
        (
            {
                "missing_keys": [],
                "unexpected_keys": ["layers.0.obsolete.weight"],
                "mismatched_keys": [],
                "error_msgs": [],
            },
            "unexpected model keys",
        ),
        (
            {
                "missing_keys": [],
                "unexpected_keys": [],
                "mismatched_keys": [
                    ("layers.0.self_attn.q_proj.weight", (1,), (2,))
                ],
                "error_msgs": [],
            },
            "shape-mismatched model keys",
        ),
        (
            {
                "missing_keys": [],
                "unexpected_keys": [],
                "mismatched_keys": [],
                "error_msgs": ["non-CPT loader failure"],
            },
            "model loading errors",
        ),
    ],
)
def test_hf_loading_gate_rejects_non_cpt_integrity_violations(
    loading_info,
    message,
) -> None:
    model = HFModel(HFConfig(**CONFIG_KWARGS))
    with pytest.raises(RuntimeError, match=message):
        HFModel._validate_cpt_loading_info(model, copy.deepcopy(loading_info))


def test_hf_beam_generation_ignores_unneeded_sequence_identity_metadata() -> None:
    torch.manual_seed(17)
    model = HFModel(HFConfig(**CONFIG_KWARGS)).eval()
    input_ids = torch.tensor([[1, 2, 3], [4, 5, 6]])
    sequence_ids = torch.tensor([101, 202])

    with torch.inference_mode():
        reference = model.generate(
            input_ids,
            max_new_tokens=2,
            num_beams=2,
            do_sample=False,
            pad_token_id=0,
        )
        identity_annotated = model.generate(
            input_ids,
            max_new_tokens=2,
            num_beams=2,
            do_sample=False,
            pad_token_id=0,
            cpt_sequence_ids=sequence_ids,
        )

    assert torch.equal(identity_annotated, reference)
