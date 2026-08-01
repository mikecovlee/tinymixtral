import copy
from dataclasses import asdict

import pytest
import torch

from hf.configuration_tinymixtral import TinyMixtralConfig as HFTinyMixtralConfig
from hf.modeling_tinymixtral import TinyMixtralForCausalLM as HFTinyMixtralForCausalLM
from model.cpt_router import CPTRouter
from model.modeling import TinyMixtralForCausalLM
from scripts.train_utils import BF16AdamW, execute_training_step, make_adamw


_FP32_ROUTER_NAMES = ("projection", "anchors", "energy")


def _assert_router_fp32(router: CPTRouter) -> None:
    for name in _FP32_ROUTER_NAMES:
        parameter = getattr(router, name)
        assert parameter.dtype == torch.float32
        if parameter.grad is not None:
            assert parameter.grad.dtype == torch.float32
    assert router.congestion_price.dtype == torch.float32
    assert router.state_version.dtype == torch.int64
    assert router.router_algorithm_version.dtype == torch.int64
    assert (
        router.router_algorithm_version.item()
        == router.expected_router_algorithm_version
    )


def _assert_state_dict_exact(source, restored) -> None:
    source_state = source.state_dict()
    restored_state = restored.state_dict()
    assert source_state.keys() == restored_state.keys()
    for key, source_value in source_state.items():
        restored_value = restored_state[key]
        assert restored_value.dtype == source_value.dtype, key
        assert restored_value.shape == source_value.shape, key
        assert torch.equal(restored_value, source_value), key


def _assert_nested_exact(actual, expected) -> None:
    if isinstance(expected, torch.Tensor):
        assert isinstance(actual, torch.Tensor)
        assert actual.dtype == expected.dtype
        assert actual.shape == expected.shape
        assert torch.equal(actual.cpu(), expected.cpu())
    elif isinstance(expected, dict):
        assert isinstance(actual, dict)
        assert set(actual) == set(expected)
        for key in expected:
            _assert_nested_exact(actual[key], expected[key])
    elif isinstance(expected, (list, tuple)):
        assert isinstance(actual, type(expected))
        assert len(actual) == len(expected)
        for actual_item, expected_item in zip(actual, expected):
            _assert_nested_exact(actual_item, expected_item)
    else:
        assert actual == expected


@pytest.mark.parametrize("default_dtype", [torch.bfloat16, torch.float64])
def test_router_construction_ignores_non_fp32_default_dtype(
    tiny_config,
    default_dtype,
) -> None:
    original_dtype = torch.get_default_dtype()
    try:
        torch.set_default_dtype(default_dtype)
        router = CPTRouter(tiny_config, layer_index=0)
    finally:
        torch.set_default_dtype(original_dtype)

    _assert_router_fp32(router)


@pytest.mark.parametrize(
    ("method_name", "body_dtype"),
    [
        ("half", torch.float16),
        ("bfloat16", torch.bfloat16),
        ("float", torch.float32),
        ("double", torch.float64),
    ],
)
def test_parent_dtype_conversion_is_bitwise_lossless_for_cpt_state_and_gradients(
    tiny_config,
    method_name,
    body_dtype,
) -> None:
    model = TinyMixtralForCausalLM(tiny_config)
    protected_values = {}
    protected_gradients = {}
    protected_identities = {}
    for layer_index, layer in enumerate(model.layers):
        router = layer.moe.cpt_router
        for name in _FP32_ROUTER_NAMES:
            parameter = getattr(router, name)
            # Include values that do not survive an FP32 -> BF16 -> FP32 cycle.
            parameter.data.add_(torch.finfo(torch.float32).eps * 3.0)
            parameter.grad = torch.randn_like(parameter)
            key = (layer_index, name)
            protected_values[key] = parameter.detach().clone()
            protected_gradients[key] = parameter.grad.detach().clone()
            protected_identities[key] = id(parameter)

    getattr(model, method_name)()
    assert model.embed_tokens.weight.dtype == body_dtype
    for layer_index, layer in enumerate(model.layers):
        router = layer.moe.cpt_router
        _assert_router_fp32(router)
        for name in _FP32_ROUTER_NAMES:
            parameter = getattr(router, name)
            key = (layer_index, name)
            assert id(parameter) == protected_identities[key]
            assert torch.equal(parameter, protected_values[key])
            assert torch.equal(parameter.grad, protected_gradients[key])


def test_meta_to_empty_and_share_memory_preserve_fp32_policy(tiny_config) -> None:
    meta_router = CPTRouter(tiny_config, layer_index=0).to(device="meta")
    identities = {
        name: id(getattr(meta_router, name)) for name in _FP32_ROUTER_NAMES
    }
    meta_router.to_empty(device="cpu")
    _assert_router_fp32(meta_router)
    for name in _FP32_ROUTER_NAMES:
        parameter = getattr(meta_router, name)
        assert parameter.device.type == "cpu"
        assert id(parameter) == identities[name]

    shared_router = CPTRouter(tiny_config, layer_index=0)
    shared_router.share_memory()
    _assert_router_fp32(shared_router)
    for name in _FP32_ROUTER_NAMES:
        assert getattr(shared_router, name).is_shared()
    assert shared_router.congestion_price.is_shared()
    assert shared_router.router_algorithm_version.is_shared()


@pytest.mark.parametrize("model_kind", ["native", "hf"])
def test_cpu_bfloat16_model_without_autocast_uses_fp32_router_master_parameters(
    tiny_config,
    fixed_batch,
    model_kind,
) -> None:
    if model_kind == "native":
        model = TinyMixtralForCausalLM(tiny_config)
    else:
        model = HFTinyMixtralForCausalLM(
            HFTinyMixtralConfig(**asdict(tiny_config))
        )
    model.to(dtype=torch.bfloat16).train()
    input_ids, labels, attention_mask = fixed_batch

    output = model(
        input_ids,
        labels=labels,
        attention_mask=attention_mask,
    )
    loss = output["loss"] if isinstance(output, dict) else output.loss
    assert torch.isfinite(loss)
    loss.backward()

    assert model.embed_tokens.weight.dtype == torch.bfloat16
    assert model.embed_tokens.weight.grad is not None
    assert model.embed_tokens.weight.grad.dtype == torch.bfloat16
    for layer in model.layers:
        router = layer.moe.cpt_router
        _assert_router_fp32(router)
        for name in _FP32_ROUTER_NAMES:
            gradient = getattr(router, name).grad
            assert gradient is not None
            assert torch.isfinite(gradient).all()


def test_adamw_moments_follow_mixed_parameter_dtypes(
    tiny_config,
    fixed_batch,
) -> None:
    model = TinyMixtralForCausalLM(tiny_config).to(torch.bfloat16).train()
    optimizer = make_adamw(model, lr=3e-4, weight_decay=0.1)
    input_ids, labels, attention_mask = fixed_batch

    with torch.autocast("cpu", dtype=torch.bfloat16):
        output = model(
            input_ids,
            labels=labels,
            attention_mask=attention_mask,
        )
    output["loss"].backward()
    optimizer.step()

    body_state = optimizer.state[model.embed_tokens.weight]
    assert body_state["exp_avg"].dtype == torch.bfloat16
    assert body_state["exp_avg_sq"].dtype == torch.bfloat16
    for layer in model.layers:
        router = layer.moe.cpt_router
        for name in _FP32_ROUTER_NAMES:
            parameter = getattr(router, name)
            state = optimizer.state[parameter]
            assert state["exp_avg"].dtype == torch.float32
            assert state["exp_avg_sq"].dtype == torch.float32


def test_bf16_adamw_preserves_fp32_cpt_moments_and_exact_reload(
    tiny_config,
    fixed_batch,
) -> None:
    torch.manual_seed(991)
    model = TinyMixtralForCausalLM(tiny_config).to(torch.bfloat16).train()
    optimizer = make_adamw(
        model,
        lr=3e-4,
        weight_decay=0.1,
        bf16_states=True,
    )
    assert type(optimizer) is BF16AdamW
    input_ids, labels, attention_mask = fixed_batch

    def strict_step(current_model, current_optimizer):
        with torch.autocast("cpu", dtype=torch.bfloat16):
            output = current_model(
                input_ids,
                labels=labels,
                attention_mask=attention_mask,
            )
        execute_training_step(current_model, output, current_optimizer)

    strict_step(model, optimizer)

    body_state = optimizer.state[model.embed_tokens.weight]
    assert body_state["exp_avg"].dtype == torch.bfloat16
    assert body_state["exp_avg_sq"].dtype == torch.bfloat16
    optimized_parameters = {
        id(parameter)
        for group in optimizer.param_groups
        for parameter in group["params"]
    }
    for parameter, state in optimizer.state.items():
        assert state["step"].dtype == torch.float32
        assert state["step"].shape == ()
        assert torch.isfinite(state["exp_avg"]).all()
        assert torch.isfinite(state["exp_avg_sq"]).all()
        assert (state["exp_avg_sq"] >= 0).all()
        assert state["exp_avg"].dtype == parameter.dtype
        assert state["exp_avg_sq"].dtype == parameter.dtype

    for layer in model.layers:
        router = layer.moe.cpt_router
        _assert_router_fp32(router)
        assert id(router.congestion_price) not in optimized_parameters
        assert router.congestion_price.grad is None
        for name in _FP32_ROUTER_NAMES:
            parameter = getattr(router, name)
            state = optimizer.state[parameter]
            assert parameter.grad is None
            assert state["exp_avg"].dtype == torch.float32
            assert state["exp_avg_sq"].dtype == torch.float32

    restored = TinyMixtralForCausalLM(tiny_config).to(torch.bfloat16).train()
    restored.load_state_dict(copy.deepcopy(model.state_dict()))
    restored_optimizer = make_adamw(
        restored,
        lr=3e-4,
        weight_decay=0.1,
        bf16_states=True,
    )
    restored_optimizer.load_state_dict(copy.deepcopy(optimizer.state_dict()))
    assert type(restored_optimizer) is BF16AdamW
    _assert_nested_exact(restored_optimizer.state_dict(), optimizer.state_dict())

    strict_step(model, optimizer)
    strict_step(restored, restored_optimizer)
    _assert_state_dict_exact(model, restored)
    _assert_nested_exact(restored_optimizer.state_dict(), optimizer.state_dict())


@pytest.mark.parametrize("model_kind", ["native", "hf"])
def test_training_mode_no_grad_does_not_create_committable_transaction(
    tiny_config,
    fixed_batch,
    model_kind,
) -> None:
    if model_kind == "native":
        model = TinyMixtralForCausalLM(tiny_config).train()
    else:
        model = HFTinyMixtralForCausalLM(
            HFTinyMixtralConfig(**asdict(tiny_config))
        ).train()
    input_ids, _, attention_mask = fixed_batch

    with torch.no_grad():
        output = model(input_ids, attention_mask=attention_mask)
    transaction = (
        output.get("cpt_transaction")
        if type(output) is dict
        else output.cpt_transaction
    )
    assert transaction is None
    assert model.get_cpt_state_version() == 0


@pytest.mark.parametrize("parameter_name", _FP32_ROUTER_NAMES)
def test_assign_load_rejects_non_fp32_router_parameter(
    tiny_config,
    parameter_name,
) -> None:
    source = TinyMixtralForCausalLM(tiny_config)
    state = {
        key: value.detach().clone() for key, value in source.state_dict().items()
    }
    key = f"layers.0.moe.cpt_router.{parameter_name}"
    state[key] = state[key].to(torch.bfloat16)

    target = TinyMixtralForCausalLM(tiny_config)
    with pytest.raises(RuntimeError, match=rf"CPT {parameter_name} must remain FP32"):
        target.load_state_dict(state, strict=True, assign=True)


def test_native_mixed_dtype_save_load_is_exact(
    tiny_config,
    tmp_path,
) -> None:
    source = TinyMixtralForCausalLM(tiny_config).to(torch.bfloat16).eval()
    output = tmp_path / "native_mixed"
    source.save_pretrained(str(output))

    restored = TinyMixtralForCausalLM.from_pretrained(str(output)).eval()
    assert restored.embed_tokens.weight.dtype == torch.bfloat16
    assert restored.lm_head.weight is restored.embed_tokens.weight
    for layer in restored.layers:
        _assert_router_fp32(layer.moe.cpt_router)
    _assert_state_dict_exact(source, restored)


def test_hf_mixed_dtype_safetensors_default_and_explicit_load_are_exact(
    tiny_config,
    tmp_path,
) -> None:
    config = HFTinyMixtralConfig(**asdict(tiny_config))
    source = HFTinyMixtralForCausalLM(config).to(torch.bfloat16).eval()
    output = tmp_path / "hf_mixed"
    source.save_pretrained(output)

    default_restored = HFTinyMixtralForCausalLM.from_pretrained(output).eval()
    explicit_restored = HFTinyMixtralForCausalLM.from_pretrained(
        output,
        dtype=torch.bfloat16,
    ).eval()
    for restored in (default_restored, explicit_restored):
        assert restored.embed_tokens.weight.dtype == torch.bfloat16
        assert restored.lm_head.weight is restored.embed_tokens.weight
        for layer in restored.layers:
            _assert_router_fp32(layer.moe.cpt_router)
        _assert_state_dict_exact(source, restored)
