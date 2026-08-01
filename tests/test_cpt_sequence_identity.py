import copy
from dataclasses import replace

import pytest
import torch

from hf.configuration_tinymixtral import TinyMixtralConfig as HFConfig
from hf.modeling_tinymixtral import TinyMixtralForCausalLM as HFModel
from model.config import TinyMixtralConfig as NativeConfig
from model.cpt_router import CPTRouter
from model.modeling import TinyMixtralForCausalLM as NativeModel


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
    "cpt_init_seed": 29,
    "rms_norm_eps": 1e-6,
    "rope_theta": 10_000.0,
    "attention_dropout": 0.0,
    "tie_word_embeddings": True,
    "initializer_range": 0.02,
}


def _router() -> CPTRouter:
    return CPTRouter(NativeConfig(**CONFIG_KWARGS), layer_index=0).eval()


def _model(kind, mode="global"):
    kwargs = dict(CONFIG_KWARGS, cpt_router_recompute=mode)
    if kind == "native":
        return NativeModel(NativeConfig(**kwargs))
    if kind == "hf":
        return HFModel(HFConfig(**kwargs))
    raise AssertionError(kind)


def _states(kind, output):
    if kind == "native":
        return output["cpt_sequence_states"]
    return output.cpt_sequence_states


def _loss(kind, output):
    if kind == "native":
        return output["loss"]
    return output.loss


def test_router_identity_aware_full_chunk_continuation_is_exact() -> None:
    torch.manual_seed(1)
    router = _router()
    hidden = torch.randn(2, 7, 16)
    ids = torch.tensor([101, -7], dtype=torch.int32)

    with torch.inference_mode():
        full = router(hidden, cpt_sequence_ids=ids)
        first = router(hidden[:, :3], cpt_sequence_ids=ids)
        second = router(
            hidden[:, 3:],
            sequence_state=first.sequence_state,
            cpt_sequence_ids=ids,
        )

    torch.testing.assert_close(
        full.q_probabilities.reshape(2, 7, -1),
        torch.cat(
            (
                first.q_probabilities.reshape(2, 3, -1),
                second.q_probabilities.reshape(2, 4, -1),
            ),
            dim=1,
        ),
        rtol=0,
        atol=0,
    )
    torch.testing.assert_close(full.sequence_state.state_s, second.sequence_state.state_s)
    torch.testing.assert_close(full.sequence_state.state_nu, second.sequence_state.state_nu)
    assert full.sequence_state.sequence_ids.dtype == torch.int64
    assert torch.equal(full.sequence_state.sequence_ids, ids.to(torch.int64))
    assert torch.equal(second.sequence_state.sequence_ids, ids.to(torch.int64))


def test_router_rejects_silent_row_reorder_but_accepts_explicit_state_reorder() -> None:
    torch.manual_seed(2)
    router = _router()
    hidden = torch.randn(2, 5, 16)
    ids = torch.tensor([10, 20])
    first = router(hidden[:, :2], cpt_sequence_ids=ids)

    with pytest.raises(ValueError, match="do not match continuation state"):
        router(
            hidden.flip(0)[:, 2:],
            sequence_state=first.sequence_state,
            cpt_sequence_ids=ids.flip(0),
        )

    order = torch.tensor([1, 0])
    explicitly_reordered = replace(
        first.sequence_state,
        state_s=first.sequence_state.state_s.index_select(0, order),
        state_nu=first.sequence_state.state_nu.index_select(0, order),
        initialized=first.sequence_state.initialized.index_select(0, order),
        sequence_ids=first.sequence_state.sequence_ids.index_select(0, order),
    )
    accepted = router(
        hidden.flip(0)[:, 2:],
        sequence_state=explicitly_reordered,
        cpt_sequence_ids=ids.flip(0),
    )
    assert torch.equal(accepted.sequence_state.sequence_ids, ids.flip(0))


def test_explicit_first_valid_reset_rebinds_only_changed_logical_row() -> None:
    torch.manual_seed(3)
    router = _router()
    first = router(
        torch.randn(2, 2, 16),
        cpt_sequence_ids=torch.tensor([10, 20]),
    )
    reset = torch.tensor([[True, False, False], [False, False, False]])
    continued = router(
        torch.randn(2, 3, 16),
        reset_mask=reset,
        sequence_state=first.sequence_state,
        cpt_sequence_ids=torch.tensor([30, 20]),
    )
    assert torch.equal(
        continued.sequence_state.sequence_ids,
        torch.tensor([30, 20]),
    )

    late_reset = torch.tensor([[False, True, False], [False, False, False]])
    with pytest.raises(ValueError, match="first valid token"):
        router(
            torch.randn(2, 3, 16),
            reset_mask=late_reset,
            sequence_state=first.sequence_state,
            cpt_sequence_ids=torch.tensor([30, 20]),
        )


def test_all_padding_preserves_identity_and_cannot_silently_replace_it() -> None:
    router = _router()
    ids = torch.tensor([10, 20])
    first = router(torch.randn(2, 2, 16), cpt_sequence_ids=ids)
    invalid = torch.zeros(2, 3, dtype=torch.bool)
    same = router(
        torch.randn(2, 3, 16),
        route_valid_mask=invalid,
        sequence_state=first.sequence_state,
        cpt_sequence_ids=ids,
    )
    torch.testing.assert_close(same.sequence_state.state_s, first.sequence_state.state_s)
    torch.testing.assert_close(same.sequence_state.state_nu, first.sequence_state.state_nu)
    assert torch.equal(same.sequence_state.sequence_ids, ids)

    with pytest.raises(ValueError, match="do not match continuation state"):
        router(
            torch.randn(2, 3, 16),
            route_valid_mask=invalid,
            sequence_state=first.sequence_state,
            cpt_sequence_ids=torch.tensor([30, 20]),
        )


@pytest.mark.parametrize(
    ("ids", "error_type", "message"),
    [
        (torch.tensor([[1, 2]]), ValueError, "shape"),
        (torch.tensor([1.0, 2.0]), TypeError, "integer dtype"),
        (torch.tensor([True, False]), TypeError, "integer dtype"),
        (torch.tensor([1, 1]), ValueError, "unique within the batch"),
    ],
)
def test_sequence_id_validation_is_strict(ids, error_type, message) -> None:
    router = _router()
    with pytest.raises(error_type, match=message):
        router(torch.randn(2, 2, 16), cpt_sequence_ids=ids)


def test_uint64_sequence_ids_are_range_checked_before_int64_conversion() -> None:
    router = _router()
    valid = torch.tensor([1, 2], dtype=torch.uint64)
    output = router(torch.randn(2, 2, 16), cpt_sequence_ids=valid)

    assert output.sequence_state.sequence_ids.dtype == torch.int64
    assert torch.equal(output.sequence_state.sequence_ids, valid.to(torch.int64))

    overflow = torch.tensor([1, 2**63], dtype=torch.uint64)
    with pytest.raises(ValueError, match="representable as int64"):
        router(torch.randn(2, 2, 16), cpt_sequence_ids=overflow)


def test_identity_aware_state_requires_ids_but_legacy_state_remains_compatible() -> None:
    router = _router()
    hidden = torch.randn(2, 2, 16)
    aware = router(hidden, cpt_sequence_ids=torch.tensor([1, 2])).sequence_state
    with pytest.raises(ValueError, match="is required"):
        router(hidden, sequence_state=aware)

    legacy = router(hidden).sequence_state
    assert legacy.sequence_ids is None
    continued = router(hidden, sequence_state=legacy)
    assert continued.sequence_state.sequence_ids is None
    bound = router(
        hidden,
        sequence_state=legacy,
        cpt_sequence_ids=torch.tensor([1, 2]),
    )
    assert torch.equal(bound.sequence_state.sequence_ids, torch.tensor([1, 2]))


def test_native_hf_identity_continuation_parity() -> None:
    torch.manual_seed(4)
    native = _model("native").eval()
    hf = _model("hf").eval()
    hf.load_state_dict(copy.deepcopy(native.state_dict()), strict=True)
    ids = torch.tensor([111, 222])
    prefix = torch.tensor([[1, 2, 3], [4, 5, 6]])
    suffix = torch.tensor([[7, 8], [9, 10]])
    with torch.inference_mode():
        native_first = native(prefix, cpt_sequence_ids=ids)
        hf_first = hf(prefix, cpt_sequence_ids=ids)
        native_second = native(
            suffix,
            cpt_sequence_states=_states("native", native_first),
            cpt_sequence_ids=ids,
        )
        hf_second = hf(
            suffix,
            cpt_sequence_states=_states("hf", hf_first),
            cpt_sequence_ids=ids,
        )
    torch.testing.assert_close(native_first["logits"], hf_first.logits)
    torch.testing.assert_close(native_second["logits"], hf_second.logits)
    for native_state, hf_state in zip(
        _states("native", native_second),
        _states("hf", hf_second),
    ):
        torch.testing.assert_close(native_state.state_s, hf_state.state_s)
        torch.testing.assert_close(native_state.state_nu, hf_state.state_nu)
        assert torch.equal(native_state.sequence_ids, ids)
        assert torch.equal(hf_state.sequence_ids, ids)


@pytest.mark.parametrize("kind", ["native", "hf"])
def test_public_model_rejects_duplicate_sequence_ids(kind) -> None:
    model = _model(kind).eval()
    with pytest.raises(ValueError, match="unique within the batch"):
        model(
            torch.tensor([[1, 2], [3, 4]]),
            cpt_sequence_ids=torch.tensor([7, 7]),
        )


@pytest.mark.parametrize("kind", ["native", "hf"])
@pytest.mark.parametrize(
    ("global_recompute", "mode"),
    [
        (False, "global"),
        (False, "on"),
        (False, "off"),
        (True, "global"),
        (True, "on"),
        (True, "off"),
    ],
)
def test_sequence_ids_survive_every_recompute_policy_and_caller_mutation(
    kind,
    global_recompute,
    mode,
) -> None:
    torch.manual_seed(5)
    model = _model(kind, mode=mode).train()
    if global_recompute:
        model.gradient_checkpointing_enable()
    ids = torch.tensor([101, 202])
    prefix = torch.tensor([[1, 2, 3], [4, 5, 6]])
    suffix = torch.tensor([[7, 8, 9], [10, 11, 12]])

    with torch.no_grad():
        first = model(prefix, cpt_sequence_ids=ids)
    continuation_states = _states(kind, first)
    output = model(
        suffix,
        labels=suffix,
        cpt_sequence_states=continuation_states,
        cpt_sequence_ids=ids,
    )
    ids.fill_(999)
    for state in continuation_states:
        state.sequence_ids.fill_(777)
    _loss(kind, output).backward()
    for state in _states(kind, output):
        assert torch.equal(state.sequence_ids, torch.tensor([101, 202]))
