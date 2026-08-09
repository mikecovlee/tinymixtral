import copy
import json
from collections import OrderedDict

import pytest
import torch

from model.config import TinyMixtralConfig
from model.modeling import TinyMixtralForCausalLM
from scripts.resume import (
    restore_cpt_optimizer_state,
    validate_cpt_resume_state,
)
from scripts.train_utils import (
    BF16AdamW,
    _execute_cpt_optimizer_step,
    get_cpt_optimizer_state_dtype,
    make_adamw,
    make_cosine_schedule,
    make_wsd_schedule,
    save_checkpoint,
    save_training_state,
    training_loop,
    validate_cpt_optimizer_state,
    validate_serialized_cpt_optimizer_state,
)


class TransparentModuleWrapper(torch.nn.Module):
    """Minimal module wrapper that forwards Native CPT APIs transparently."""

    def __init__(self, module):
        super().__init__()
        self.module = module

    def forward(self, *args, **kwargs):
        return self.module(*args, **kwargs)

    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            module = super().__getattr__("module")
            return getattr(module, name)


def tiny_config(**overrides):
    values = {
        "vocab_size": 43,
        "hidden_size": 8,
        "num_hidden_layers": 2,
        "num_attention_heads": 2,
        "num_key_value_heads": 1,
        "head_dim": 4,
        "max_position_embeddings": 24,
        "num_local_experts": 3,
        "num_experts_per_tok": 2,
        "expert_intermediate_size": 16,
        "cpt_projection_dim": 4,
        "cpt_init_seed": 211,
    }
    values.update(overrides)
    return TinyMixtralConfig(**values)


def clone_state_dict(model):
    return OrderedDict((key, value.detach().clone()) for key, value in model.state_dict().items())


def training_step(model, optimizer, scheduler, input_ids):
    output = model(input_ids, labels=input_ids)
    output["loss"].backward()
    _execute_cpt_optimizer_step(model, optimizer, scheduler, output["cpt_transaction"])
    optimizer.zero_grad(set_to_none=True)
    return output


def rebuild_scheduler(optimizer, state, schedule_kind, decay_ratio=0.1):
    saved_lrs = [group["lr"] for group in optimizer.param_groups]
    if schedule_kind == "wsd":
        scheduler = make_wsd_schedule(
            optimizer,
            state["warmup_steps"],
            state["total_steps"],
            decay_ratio=decay_ratio,
        )
    else:
        scheduler = make_cosine_schedule(optimizer, state["warmup_steps"], state["total_steps"])
    for group, lr in zip(optimizer.param_groups, saved_lrs):
        group["lr"] = lr
    scheduler.load_state_dict(state["sched"])
    return scheduler


def serialized_parameter_id_for(optimizer, optimizer_state, target_parameter):
    for live_group, saved_group in zip(
        optimizer.param_groups,
        optimizer_state["param_groups"],
    ):
        for parameter, parameter_id in zip(
            live_group["params"],
            saved_group["params"],
        ):
            if parameter is target_parameter:
                return parameter_id
    raise AssertionError("target parameter is absent from serialized optimizer")


def cpt_parameter_bindings_for(model, optimizer, optimizer_state):
    cpt_parameter_ids = {id(parameter) for parameter in model.cpt_trainable_parameters()}
    return tuple(
        (
            name,
            serialized_parameter_id_for(optimizer, optimizer_state, parameter),
        )
        for name, parameter in model.named_parameters()
        if id(parameter) in cpt_parameter_ids
    )


def make_resume_case(
    tmp_path,
    *,
    bf16_states=False,
    legacy_single_group=False,
    steps=1,
):
    config = tiny_config()
    model = TinyMixtralForCausalLM(config)
    if legacy_single_group:
        optimizer_class = BF16AdamW if bf16_states else torch.optim.AdamW
        optimizer = optimizer_class(
            model.parameters(),
            lr=1e-3,
            weight_decay=0.0,
            betas=(0.9, 0.95),
        )
    else:
        optimizer = make_adamw(
            model,
            lr=1e-3,
            weight_decay=0.0,
            bf16_states=bf16_states,
        )
    total_steps = max(steps + 1, 2)
    scheduler = make_cosine_schedule(optimizer, 0, total_steps)
    for _ in range(steps):
        input_ids = torch.randint(0, model.config.vocab_size, (2, 5))
        training_step(model, optimizer, scheduler, input_ids)

    state_path = tmp_path / "training_state.pt"
    save_training_state(
        state_path,
        model,
        optimizer,
        scheduler,
        step=steps,
        total_tok=steps * 10,
        warmup_steps=0,
        total_steps=total_steps,
        fi=0,
        ptr=steps * 12,
        batch_size=2,
        seq_len=5,
    )
    state = torch.load(state_path, weights_only=True)
    resumed_model = TinyMixtralForCausalLM(tiny_config())
    resumed_model.load_state_dict(model.state_dict())
    resumed_optimizer = make_adamw(
        resumed_model,
        lr=1e-3,
        weight_decay=0.0,
        bf16_states=bf16_states,
    )
    return (
        model,
        optimizer,
        scheduler,
        state,
        resumed_model,
        resumed_optimizer,
    )


def assert_restore_rejected_before_load(
    state,
    model,
    optimizer,
    *,
    bf16_states=False,
    match,
):
    load_calls = []
    original_load_state_dict = optimizer.load_state_dict

    def tracked_load_state_dict(optimizer_state):
        load_calls.append(optimizer_state)
        return original_load_state_dict(optimizer_state)

    optimizer.load_state_dict = tracked_load_state_dict
    assert not optimizer.state
    with pytest.raises(RuntimeError, match=match):
        restore_cpt_optimizer_state(
            state,
            model,
            optimizer,
            lr=1e-3,
            weight_decay=0.0,
            bf16_states=bf16_states,
        )
    assert load_calls == []
    assert not optimizer.state


def test_model_checkpoint_round_trip_contains_all_cpt_parameters_and_state(tmp_path):
    torch.manual_seed(1)
    model = TinyMixtralForCausalLM(tiny_config())
    optimizer = make_adamw(model, lr=1e-3, weight_decay=0.1)
    scheduler = make_cosine_schedule(optimizer, warmup_steps=0, total_steps=3)
    input_ids = torch.randint(0, model.config.vocab_size, (2, 5))
    training_step(model, optimizer, scheduler, input_ids)

    checkpoint = save_checkpoint(
        model,
        optimizer,
        scheduler,
        tmp_path,
        step=1,
        total_tok=10,
        warmup_steps=0,
        total_steps=3,
        fi=2,
        ptr=120,
        batch_size=2,
        seq_len=5,
    )
    loaded = TinyMixtralForCausalLM.from_pretrained(str(checkpoint))
    assert loaded.get_cpt_state_version() == 1
    assert loaded.get_cpt_optimizer_step() == 1
    for key, value in model.state_dict().items():
        assert torch.equal(loaded.state_dict()[key], value)
    keys = set(loaded.state_dict())
    assert any(key.endswith(".projection") for key in keys)
    assert any(key.endswith(".anchors") for key in keys)
    assert any(key.endswith(".energy") for key in keys)
    assert any(key.endswith(".congestion_price") for key in keys)
    assert any(key.endswith(".optimizer_step") for key in keys)
    assert any(key.endswith(".state_version") for key in keys)
    assert any(key.endswith(".router_algorithm_version") for key in keys)
    assert all("transaction" not in key and "load_sum" not in key for key in keys)

    state = torch.load(checkpoint / "training_state.pt", weights_only=True)
    loaded_optimizer = make_adamw(loaded, lr=1e-3, weight_decay=0.1)
    assert validate_cpt_resume_state(state, loaded, loaded_optimizer) == 1
    assert state["cpt_state_version"] == 1
    assert state["cpt_optimizer_state_dtype"] == "float32"
    cpt_parameter_ids = {id(parameter) for parameter in model.cpt_trainable_parameters()}
    expected_presence = tuple(
        name
        for name, parameter in model.named_parameters()
        if id(parameter) in cpt_parameter_ids and parameter in optimizer.state
    )
    assert state["cpt_optimizer_state_presence"] == expected_presence
    assert state["fi"] == 2 and state["ptr"] == 120


def test_training_state_adds_only_minimal_cpt_fields_to_upstream_schema(tmp_path):
    model = TinyMixtralForCausalLM(tiny_config(num_hidden_layers=1))
    optimizer = make_adamw(model, lr=2e-4, weight_decay=0.05)
    scheduler = make_cosine_schedule(optimizer, 1, 10)
    path = tmp_path / "training_state.pt"
    save_training_state(
        path,
        model,
        optimizer,
        scheduler,
        step=0,
        total_tok=0,
        warmup_steps=1,
        total_steps=10,
        fi=0,
        ptr=0,
        batch_size=2,
        seq_len=5,
    )
    state = torch.load(path, weights_only=True)
    assert set(state) == {
        "opt",
        "sched",
        "step",
        "total_tok",
        "warmup_steps",
        "total_steps",
        "fi",
        "ptr",
        "batch_size",
        "seq_len",
        "cpt_state_version",
        "cpt_optimizer_state_presence",
        "cpt_optimizer_parameter_bindings",
        "cpt_optimizer_state_dtype",
    }
    assert state["cpt_optimizer_state_dtype"] == "float32"
    assert validate_cpt_resume_state(state, model, optimizer) == 0


@pytest.mark.parametrize("bf16_states", [False, True])
@pytest.mark.parametrize("schedule_kind", ["cosine", "wsd"])
def test_uninterrupted_and_resumed_next_step_match(tmp_path, bf16_states, schedule_kind):
    torch.manual_seed(7)
    model = TinyMixtralForCausalLM(tiny_config())
    optimizer = make_adamw(model, lr=3e-4, weight_decay=0.01, bf16_states=bf16_states)
    scheduler = (
        make_wsd_schedule(optimizer, 1, 4, decay_ratio=0.5) if schedule_kind == "wsd" else make_cosine_schedule(optimizer, 1, 4)
    )
    first_batch = torch.randint(0, model.config.vocab_size, (2, 5))
    training_step(model, optimizer, scheduler, first_batch)
    checkpoint = save_checkpoint(model, optimizer, scheduler, tmp_path, 1, 10, 1, 4, 0, 12, 2, 5)

    resumed = TinyMixtralForCausalLM.from_pretrained(str(checkpoint))
    state = torch.load(checkpoint / "training_state.pt", weights_only=True)
    resumed_optimizer = make_adamw(
        resumed,
        lr=3e-4,
        weight_decay=0.01,
        bf16_states=bf16_states,
    )
    validate_cpt_resume_state(state, resumed, resumed_optimizer)
    resumed_optimizer.load_state_dict(state["opt"])
    validate_cpt_optimizer_state(
        resumed,
        resumed_optimizer,
        expected_presence=state["cpt_optimizer_state_presence"],
        expected_state_dtype=state["cpt_optimizer_state_dtype"],
    )
    resumed_scheduler = rebuild_scheduler(resumed_optimizer, state, schedule_kind, decay_ratio=0.5)

    second_batch = torch.randint(0, model.config.vocab_size, (2, 5))
    original_output = model(second_batch, labels=second_batch)
    resumed_output = resumed(second_batch, labels=second_batch)
    torch.testing.assert_close(resumed_output["logits"], original_output["logits"], atol=0, rtol=0)
    torch.testing.assert_close(resumed_output["loss"], original_output["loss"], atol=0, rtol=0)
    for original, restored in zip(
        original_output["cpt_transaction"].proposals,
        resumed_output["cpt_transaction"].proposals,
    ):
        assert torch.equal(restored.load_sum, original.load_sum)
        assert torch.equal(restored.token_count, original.token_count)
        assert torch.equal(restored.state_version, original.state_version)

    original_output["loss"].backward()
    resumed_output["loss"].backward()
    for original_parameter, resumed_parameter in zip(model.parameters(), resumed.parameters()):
        if original_parameter.grad is None:
            assert resumed_parameter.grad is None
        else:
            torch.testing.assert_close(resumed_parameter.grad, original_parameter.grad, atol=0, rtol=0)
    _execute_cpt_optimizer_step(model, optimizer, scheduler, original_output["cpt_transaction"])
    _execute_cpt_optimizer_step(
        resumed,
        resumed_optimizer,
        resumed_scheduler,
        resumed_output["cpt_transaction"],
    )
    for key, original in model.state_dict().items():
        torch.testing.assert_close(resumed.state_dict()[key], original, atol=0, rtol=0)
    assert resumed_scheduler.state_dict() == scheduler.state_dict()


def mutation_cases(model):
    base = clone_state_dict(model)
    cpt_keys = [key for key in base if ".moe.cpt_router." in key]

    missing = OrderedDict(base)
    missing.pop(cpt_keys[0])
    yield "missing CPT keys", missing

    unknown = OrderedDict(base)
    unknown["layers.0.moe.cpt_router.unknown"] = torch.zeros(1)
    yield "unknown CPT keys", unknown

    shape = OrderedDict(base)
    projection_key = next(key for key in cpt_keys if key.endswith(".projection"))
    shape[projection_key] = shape[projection_key][:-1]
    yield "shape mismatch", shape

    dtype = OrderedDict(base)
    dtype[projection_key] = dtype[projection_key].to(torch.bfloat16)
    yield "dtype mismatch", dtype

    nonfinite = OrderedDict(base)
    nonfinite[projection_key] = nonfinite[projection_key].clone()
    nonfinite[projection_key].view(-1)[0] = float("nan")
    yield "non-finite", nonfinite

    bad_algorithm = OrderedDict(base)
    algorithm_key = next(key for key in cpt_keys if key.endswith(".router_algorithm_version"))
    bad_algorithm[algorithm_key] = torch.tensor(2, dtype=torch.int64)
    yield "unsupported CPT algorithm", bad_algorithm

    negative_price = OrderedDict(base)
    price_key = next(key for key in cpt_keys if key.endswith(".congestion_price"))
    negative_price[price_key] = negative_price[price_key].clone()
    negative_price[price_key][0] = -1.0
    yield "price is negative", negative_price

    bad_anchors = OrderedDict(base)
    anchor_key = next(key for key in cpt_keys if key.endswith(".anchors"))
    bad_anchors[anchor_key] = bad_anchors[anchor_key] * 2
    yield "anchors are not unit", bad_anchors

    mismatched_versions = OrderedDict(base)
    version_keys = [key for key in cpt_keys if key.endswith(".state_version")]
    mismatched_versions[version_keys[-1]] = torch.tensor(1, dtype=torch.int64)
    yield "versions disagree", mismatched_versions

    negative_optimizer_step = OrderedDict(base)
    optimizer_step_keys = [key for key in cpt_keys if key.endswith(".optimizer_step")]
    negative_optimizer_step[optimizer_step_keys[0]] = torch.tensor(
        -1,
        dtype=torch.int64,
    )
    yield "optimizer_step is negative", negative_optimizer_step

    mismatched_optimizer_steps = OrderedDict(base)
    mismatched_optimizer_steps[optimizer_step_keys[-1]] = torch.tensor(
        1,
        dtype=torch.int64,
    )
    yield "optimizer steps disagree", mismatched_optimizer_steps

    optimizer_step_ahead = OrderedDict(base)
    for optimizer_step_key in optimizer_step_keys:
        optimizer_step_ahead[optimizer_step_key] = torch.tensor(
            1,
            dtype=torch.int64,
        )
    yield "optimizer_step exceeds state_version", optimizer_step_ahead

    legacy = OrderedDict(base)
    legacy["layers.0.moe.router.weight"] = torch.zeros(model.config.num_local_experts, model.config.hidden_size)
    yield "legacy Linear Router", legacy


def test_strict_cpt_load_rejects_missing_unknown_malformed_and_legacy_state():
    source = TinyMixtralForCausalLM(tiny_config())
    for message, state in mutation_cases(source):
        target = TinyMixtralForCausalLM(tiny_config())
        with pytest.raises(RuntimeError, match=message):
            target.load_state_dict(state, strict=True)


def test_strict_false_allows_non_cpt_partial_load_but_keeps_cpt_strict():
    source = TinyMixtralForCausalLM(tiny_config())
    partial = clone_state_dict(source)
    partial.pop("norm.weight")
    target = TinyMixtralForCausalLM(tiny_config())
    result = target.load_state_dict(partial, strict=False)
    assert result.missing_keys == ["norm.weight"]

    missing_cpt = clone_state_dict(source)
    cpt_key = next(key for key in missing_cpt if ".moe.cpt_router." in key)
    missing_cpt.pop(cpt_key)
    with pytest.raises(RuntimeError, match="missing CPT keys"):
        target.load_state_dict(missing_cpt, strict=False)


def test_load_rejects_mutated_live_config_before_copying_weights():
    source = TinyMixtralForCausalLM(tiny_config())
    target = TinyMixtralForCausalLM(tiny_config())
    before = clone_state_dict(target)
    target.config.cpt_router_version = True

    with pytest.raises(RuntimeError, match="config changed after construction"):
        target.load_state_dict(source.state_dict(), strict=True)
    for key, value in target.state_dict().items():
        assert torch.equal(value, before[key])


def test_explicit_checkpoint_config_mismatch_is_rejected(tmp_path):
    model = TinyMixtralForCausalLM(tiny_config())
    model.save_pretrained(tmp_path)
    mismatched = tiny_config(cpt_prototype_temperature=0.75)
    with pytest.raises(RuntimeError, match="disagrees"):
        TinyMixtralForCausalLM.from_pretrained(str(tmp_path), config=mismatched)


def test_explicit_checkpoint_config_rejects_cross_type_equality(tmp_path):
    model = TinyMixtralForCausalLM(tiny_config())
    model.save_pretrained(tmp_path)
    mutated = tiny_config()
    mutated.cpt_router_version = True
    with pytest.raises(RuntimeError, match="disagrees"):
        TinyMixtralForCausalLM.from_pretrained(str(tmp_path), config=mutated)


def test_explicit_checkpoint_config_rejects_top_k_mismatch(tmp_path):
    model = TinyMixtralForCausalLM(tiny_config())
    model.save_pretrained(tmp_path)
    top_one = tiny_config()
    top_one.num_experts_per_tok = 1
    with pytest.raises(RuntimeError, match="disagrees"):
        TinyMixtralForCausalLM.from_pretrained(str(tmp_path), config=top_one)


@pytest.mark.parametrize("mutation", ["missing", "unknown"])
def test_checkpoint_config_requires_known_complete_cpt_schema(tmp_path, mutation):
    model = TinyMixtralForCausalLM(tiny_config())
    model.save_pretrained(tmp_path)
    config_path = tmp_path / "config.json"
    with config_path.open(encoding="utf-8") as config_file:
        payload = json.load(config_file)
    if mutation == "missing":
        payload.pop("cpt_prototype_temperature")
        message = "missing CPT fields"
    else:
        payload["cpt_future_semantics"] = 1
        message = "unknown CPT fields"
    with config_path.open("w", encoding="utf-8") as config_file:
        json.dump(payload, config_file)
    with pytest.raises(RuntimeError, match=message):
        TinyMixtralForCausalLM.from_pretrained(str(tmp_path))


@pytest.mark.parametrize(
    "field",
    [
        "hidden_size",
        "num_hidden_layers",
        "num_local_experts",
        "num_experts_per_tok",
    ],
)
def test_checkpoint_config_requires_routing_architecture_field(field, tmp_path):
    model = TinyMixtralForCausalLM(tiny_config())
    model.save_pretrained(tmp_path)
    config_path = tmp_path / "config.json"
    with config_path.open(encoding="utf-8") as config_file:
        payload = json.load(config_file)
    payload.pop(field)
    with config_path.open("w", encoding="utf-8") as config_file:
        json.dump(payload, config_file)
    with pytest.raises(
        RuntimeError,
        match=rf"missing CPT routing architecture fields: {field}",
    ):
        TinyMixtralForCausalLM.from_pretrained(str(tmp_path))


@pytest.mark.parametrize(
    "field, invalid_value",
    [
        ("hidden_size", 8.0),
        ("hidden_size", True),
        ("num_hidden_layers", 2.0),
        ("num_hidden_layers", True),
        ("num_local_experts", 3.0),
        ("num_local_experts", True),
        ("num_experts_per_tok", 2.0),
        ("num_experts_per_tok", True),
    ],
)
def test_checkpoint_config_rejects_non_integer_routing_architecture(field, invalid_value, tmp_path):
    model = TinyMixtralForCausalLM(tiny_config())
    model.save_pretrained(tmp_path)
    config_path = tmp_path / "config.json"
    with config_path.open(encoding="utf-8") as config_file:
        payload = json.load(config_file)
    payload[field] = invalid_value
    with config_path.open("w", encoding="utf-8") as config_file:
        json.dump(payload, config_file)
    with pytest.raises(ValueError, match="positive integer"):
        TinyMixtralForCausalLM.from_pretrained(str(tmp_path))


def test_checkpoint_config_rejects_non_top2_routing(tmp_path):
    model = TinyMixtralForCausalLM(tiny_config())
    model.save_pretrained(tmp_path)
    config_path = tmp_path / "config.json"
    with config_path.open(encoding="utf-8") as config_file:
        payload = json.load(config_file)
    payload["num_experts_per_tok"] = 1
    with config_path.open("w", encoding="utf-8") as config_file:
        json.dump(payload, config_file)
    with pytest.raises(ValueError, match="must remain 2"):
        TinyMixtralForCausalLM.from_pretrained(str(tmp_path))


@pytest.mark.parametrize(
    "field",
    [
        "cpt_num_prototypes",
        "cpt_kappa_beta",
        "cpt_lambda_sa",
        "cpt_prototype_temperature",
        "cpt_state_step_size",
        "cpt_energy_init_scale",
        "cpt_price_learning_rate",
    ],
)
def test_checkpoint_config_rejects_unresolved_cpt_null(field, tmp_path):
    model = TinyMixtralForCausalLM(tiny_config())
    model.save_pretrained(tmp_path)
    config_path = tmp_path / "config.json"
    with config_path.open(encoding="utf-8") as config_file:
        payload = json.load(config_file)
    payload[field] = None
    with config_path.open("w", encoding="utf-8") as config_file:
        json.dump(payload, config_file)
    with pytest.raises(RuntimeError):
        TinyMixtralForCausalLM.from_pretrained(str(tmp_path))


def test_checkpoint_config_allows_unknown_non_cpt_metadata(tmp_path):
    model = TinyMixtralForCausalLM(tiny_config())
    model.save_pretrained(tmp_path)
    config_path = tmp_path / "config.json"
    with config_path.open(encoding="utf-8") as config_file:
        payload = json.load(config_file)
    payload["future_non_cpt_metadata"] = {"producer": "test"}
    with config_path.open("w", encoding="utf-8") as config_file:
        json.dump(payload, config_file)
    loaded = TinyMixtralForCausalLM.from_pretrained(str(tmp_path))
    assert loaded.config.cpt_config_dict() == model.config.cpt_config_dict()


def test_checkpoint_rejects_optimizer_missing_this_models_cpt_parameters(tmp_path):
    model = TinyMixtralForCausalLM(tiny_config())
    foreign_model = TinyMixtralForCausalLM(tiny_config())
    optimizer = make_adamw(foreign_model, lr=1e-3, weight_decay=0.0)
    scheduler = make_cosine_schedule(optimizer, 0, 2)
    with pytest.raises(RuntimeError, match="every CPT learnable parameter"):
        save_checkpoint(model, optimizer, scheduler, tmp_path, 0, 0, 0, 2, 0, 0, 2, 5)


def test_checkpoint_rejects_cpt_config_mutation_after_model_construction(tmp_path):
    model = TinyMixtralForCausalLM(tiny_config())
    optimizer = make_adamw(model, lr=1e-3, weight_decay=0.0)
    scheduler = make_cosine_schedule(optimizer, 0, 2)
    model.config.cpt_prototype_temperature = 0.75
    with pytest.raises(RuntimeError, match="config changed after construction"):
        save_checkpoint(model, optimizer, scheduler, tmp_path, 0, 0, 0, 2, 0, 0, 2, 5)


@pytest.mark.parametrize(
    "field, equal_value_with_wrong_type",
    [
        ("cpt_router_version", True),
        ("cpt_projection_dim", 4.0),
        ("num_hidden_layers", 2.0),
        ("hidden_size", 8.0),
        ("num_experts_per_tok", 2.0),
    ],
)
def test_router_config_binding_rejects_cross_type_equality(field, equal_value_with_wrong_type):
    model = TinyMixtralForCausalLM(tiny_config())
    setattr(model.config, field, equal_value_with_wrong_type)
    with pytest.raises(RuntimeError, match="config changed after construction"):
        model.validate_persistent_cpt_state()


def test_checkpoint_rejects_top_k_mutation_after_model_construction(tmp_path):
    model = TinyMixtralForCausalLM(tiny_config())
    optimizer = make_adamw(model, lr=1e-3, weight_decay=0.0)
    scheduler = make_cosine_schedule(optimizer, 0, 2)
    model.config.num_experts_per_tok = 1
    with pytest.raises(RuntimeError, match="num_experts_per_tok"):
        save_checkpoint(model, optimizer, scheduler, tmp_path, 0, 0, 0, 2, 0, 0, 2, 5)


def test_checkpoint_rejects_live_top_k_mutation(tmp_path):
    model = TinyMixtralForCausalLM(tiny_config())
    optimizer = make_adamw(model, lr=1e-3, weight_decay=0.0)
    scheduler = make_cosine_schedule(optimizer, 0, 2)
    model.layers[0].moe.top_k = 1
    with pytest.raises(RuntimeError, match="Top-k"):
        save_checkpoint(model, optimizer, scheduler, tmp_path, 0, 0, 0, 2, 0, 0, 2, 5)


def test_nonfinite_optimizer_moment_is_rejected_before_checkpoint(tmp_path):
    model = TinyMixtralForCausalLM(tiny_config())
    optimizer = make_adamw(model, lr=1e-3, weight_decay=0.0)
    scheduler = make_cosine_schedule(optimizer, 0, 2)
    input_ids = torch.randint(0, model.config.vocab_size, (2, 5))
    training_step(model, optimizer, scheduler, input_ids)
    parameter = model.cpt_trainable_parameters()[0]
    optimizer.state[parameter]["exp_avg"].view(-1)[0] = float("nan")
    with pytest.raises(RuntimeError, match="non-finite"):
        save_checkpoint(model, optimizer, scheduler, tmp_path, 1, 10, 0, 2, 0, 12, 2, 5)


def test_incomplete_initialized_cpt_optimizer_state_is_rejected():
    model = TinyMixtralForCausalLM(tiny_config())
    optimizer = make_adamw(model, lr=1e-3, weight_decay=0.0)
    scheduler = make_cosine_schedule(optimizer, 0, 2)
    input_ids = torch.randint(0, model.config.vocab_size, (2, 5))
    training_step(model, optimizer, scheduler, input_ids)
    state = optimizer.state[model.cpt_trainable_parameters()[0]]
    state.pop("exp_avg")
    with pytest.raises(RuntimeError, match="incomplete"):
        validate_cpt_optimizer_state(model, optimizer)


def test_resume_rejects_deleted_cpt_optimizer_state_entry(tmp_path):
    model = TinyMixtralForCausalLM(tiny_config())
    optimizer = make_adamw(model, lr=1e-3, weight_decay=0.0)
    scheduler = make_cosine_schedule(optimizer, 0, 2)
    input_ids = torch.randint(0, model.config.vocab_size, (2, 5))
    training_step(model, optimizer, scheduler, input_ids)
    checkpoint = save_checkpoint(model, optimizer, scheduler, tmp_path, 1, 10, 0, 2, 0, 12, 2, 5)
    state = torch.load(checkpoint / "training_state.pt", weights_only=True)
    target = model.cpt_trainable_parameters()[0]
    serialized_parameter_id = serialized_parameter_id_for(
        optimizer,
        state["opt"],
        target,
    )
    state["opt"]["state"].pop(serialized_parameter_id)

    resumed = TinyMixtralForCausalLM.from_pretrained(str(checkpoint))
    resumed_optimizer = make_adamw(resumed, lr=1e-3, weight_decay=0.0)
    assert not resumed_optimizer.state
    with pytest.raises(RuntimeError, match="empty or contain every"):
        validate_cpt_resume_state(state, resumed, resumed_optimizer)
    assert not resumed_optimizer.state


def test_optimizer_presence_manifest_is_exact_and_canonical():
    model = TinyMixtralForCausalLM(tiny_config(num_hidden_layers=1))
    optimizer = make_adamw(model, lr=1e-3, weight_decay=0.0)
    scheduler = make_cosine_schedule(optimizer, 0, 2)
    input_ids = torch.randint(0, model.config.vocab_size, (2, 5))
    training_step(model, optimizer, scheduler, input_ids)
    presence = validate_cpt_optimizer_state(model, optimizer)
    assert presence == (
        "layers.0.moe.cpt_router.projection",
        "layers.0.moe.cpt_router.anchors",
        "layers.0.moe.cpt_router.energy",
    )

    invalid_manifests = (
        presence + (presence[0],),
        presence + ("layers.unknown.moe.cpt_router.projection",),
        tuple(reversed(presence)),
        presence[:-1],
    )
    for invalid_manifest in invalid_manifests:
        with pytest.raises(RuntimeError):
            validate_cpt_optimizer_state(
                model,
                optimizer,
                expected_presence=invalid_manifest,
            )


def test_transparent_wrapper_keeps_canonical_checkpoint_names(tmp_path):
    model = TinyMixtralForCausalLM(tiny_config(num_hidden_layers=1))
    optimizer = make_adamw(model, lr=1e-3, weight_decay=0.0)
    scheduler = make_cosine_schedule(optimizer, 0, 2)
    input_ids = torch.randint(0, model.config.vocab_size, (2, 5))
    training_step(model, optimizer, scheduler, input_ids)

    expected_presence = validate_cpt_optimizer_state(model, optimizer)
    wrapped = TransparentModuleWrapper(model)
    assert validate_cpt_optimizer_state(wrapped, optimizer) == expected_presence

    checkpoint = save_checkpoint(wrapped, optimizer, scheduler, tmp_path, 1, 10, 0, 2, 0, 12, 2, 5)
    serialized_model = torch.load(
        checkpoint / "pytorch_model.bin",
        weights_only=True,
    )
    assert not any(name.startswith("module.") for name in serialized_model)
    reloaded = TinyMixtralForCausalLM.from_pretrained(str(checkpoint))
    state = torch.load(checkpoint / "training_state.pt", weights_only=True)
    reloaded_optimizer = make_adamw(reloaded, lr=1e-3, weight_decay=0.0)
    assert (
        validate_cpt_resume_state(
            state,
            reloaded,
            reloaded_optimizer,
        )
        == 1
    )
    assert state["cpt_optimizer_state_presence"] == expected_presence


def test_all_padding_step_preserves_lazy_cpt_optimizer_resume(tmp_path):
    torch.manual_seed(23)
    model = TinyMixtralForCausalLM(tiny_config())
    optimizer = make_adamw(model, lr=3e-4, weight_decay=0.01)
    scheduler = make_cosine_schedule(optimizer, 0, 3)
    input_ids = torch.randint(0, model.config.vocab_size, (2, 5))
    attention_mask = torch.zeros_like(input_ids, dtype=torch.bool)

    output = model(
        input_ids,
        attention_mask=attention_mask,
        labels=input_ids,
    )
    output["loss"].backward()
    cpt_parameters = model.cpt_trainable_parameters()
    assert all(parameter.grad is None for parameter in cpt_parameters)
    _execute_cpt_optimizer_step(
        model,
        optimizer,
        scheduler,
        output["cpt_transaction"],
    )
    optimizer.zero_grad(set_to_none=True)
    assert model.get_cpt_state_version() == 1
    assert model.get_cpt_optimizer_step() == 0
    assert all(parameter not in optimizer.state for parameter in cpt_parameters)

    checkpoint = save_checkpoint(model, optimizer, scheduler, tmp_path, 1, 10, 0, 3, 0, 12, 2, 5)
    resumed = TinyMixtralForCausalLM.from_pretrained(str(checkpoint))
    state = torch.load(checkpoint / "training_state.pt", weights_only=True)
    assert state["cpt_optimizer_state_presence"] == ()
    assert state["cpt_optimizer_state_dtype"] == "float32"
    for required_field in (
        "opt",
        "sched",
        "cpt_optimizer_parameter_bindings",
        "cpt_optimizer_state_dtype",
    ):
        incomplete = dict(state)
        incomplete.pop(required_field)
        with pytest.raises(RuntimeError, match="incomplete"):
            validate_cpt_resume_state(incomplete, resumed, optimizer)
    mismatched_optimizer = make_adamw(
        resumed,
        lr=3e-4,
        weight_decay=0.01,
        bf16_states=True,
    )
    with pytest.raises(RuntimeError, match="dtype disagrees"):
        validate_cpt_resume_state(state, resumed, mismatched_optimizer)
    resumed_optimizer = make_adamw(
        resumed,
        lr=3e-4,
        weight_decay=0.01,
    )
    validate_cpt_resume_state(state, resumed, resumed_optimizer)
    resumed_optimizer.load_state_dict(state["opt"])
    validate_cpt_optimizer_state(
        resumed,
        resumed_optimizer,
        expected_presence=state["cpt_optimizer_state_presence"],
        expected_state_dtype=state["cpt_optimizer_state_dtype"],
    )
    assert all(parameter not in resumed_optimizer.state for parameter in resumed.cpt_trainable_parameters())
    assert resumed.get_cpt_optimizer_step() == 0
    resumed_scheduler = rebuild_scheduler(
        resumed_optimizer,
        state,
        "cosine",
    )

    next_input_ids = torch.randint(0, model.config.vocab_size, (2, 5))
    original_output = model(next_input_ids, labels=next_input_ids)
    resumed_output = resumed(next_input_ids, labels=next_input_ids)
    torch.testing.assert_close(
        resumed_output["logits"],
        original_output["logits"],
        atol=0,
        rtol=0,
    )
    original_output["loss"].backward()
    resumed_output["loss"].backward()
    _execute_cpt_optimizer_step(
        model,
        optimizer,
        scheduler,
        original_output["cpt_transaction"],
    )
    _execute_cpt_optimizer_step(
        resumed,
        resumed_optimizer,
        resumed_scheduler,
        resumed_output["cpt_transaction"],
    )
    for key, value in model.state_dict().items():
        torch.testing.assert_close(
            resumed.state_dict()[key],
            value,
            atol=0,
            rtol=0,
        )
    assert resumed_scheduler.state_dict() == scheduler.state_dict()


def test_cpt_optimizer_step_tracks_padding_then_valid_then_padding():
    torch.manual_seed(29)
    model = TinyMixtralForCausalLM(tiny_config())
    optimizer = make_adamw(model, lr=3e-4, weight_decay=0.01)
    scheduler = make_cosine_schedule(optimizer, 0, 4)
    input_ids = torch.randint(0, model.config.vocab_size, (2, 5))
    padding_mask = torch.zeros_like(input_ids, dtype=torch.bool)

    for expected_version, use_padding in ((1, True), (2, False), (3, True)):
        output = model(
            input_ids,
            attention_mask=padding_mask if use_padding else None,
            labels=input_ids,
        )
        output["loss"].backward()
        _execute_cpt_optimizer_step(
            model,
            optimizer,
            scheduler,
            output["cpt_transaction"],
        )
        optimizer.zero_grad(set_to_none=True)
        assert model.get_cpt_state_version() == expected_version
        assert model.get_cpt_optimizer_step() == (0 if expected_version == 1 else 1)

    presence = validate_cpt_optimizer_state(model, optimizer)
    assert presence
    for parameter in model.cpt_trainable_parameters():
        assert int(optimizer.state[parameter]["step"].item()) == 1


def test_missing_ordinary_expert_optimizer_state_is_not_a_cpt_error():
    model = TinyMixtralForCausalLM(tiny_config())
    optimizer = make_adamw(model, lr=1e-3, weight_decay=0.0)
    scheduler = make_cosine_schedule(optimizer, 0, 2)
    input_ids = torch.randint(0, model.config.vocab_size, (2, 5))
    training_step(model, optimizer, scheduler, input_ids)
    ordinary_expert = next(parameter for name, parameter in model.named_parameters() if name.endswith(".moe.gate_proj"))
    optimizer.state.pop(ordinary_expert, None)
    validate_cpt_optimizer_state(model, optimizer)


def test_bf16_optimizer_load_restores_bf16_moment_storage():
    model = TinyMixtralForCausalLM(tiny_config(num_hidden_layers=1))
    optimizer = make_adamw(model, lr=1e-3, weight_decay=0.0, bf16_states=True)
    scheduler = make_cosine_schedule(optimizer, 0, 2)
    input_ids = torch.randint(0, model.config.vocab_size, (2, 5))
    training_step(model, optimizer, scheduler, input_ids)
    presence = validate_cpt_optimizer_state(model, optimizer)
    payload = optimizer.state_dict()

    restored_model = TinyMixtralForCausalLM(tiny_config(num_hidden_layers=1))
    restored_model.load_state_dict(model.state_dict())
    restored = make_adamw(restored_model, lr=1e-3, weight_decay=0.0, bf16_states=True)
    restored.load_state_dict(payload)
    validate_cpt_optimizer_state(
        restored_model,
        restored,
        expected_presence=presence,
        expected_state_dtype="bfloat16",
    )
    for state in restored.state.values():
        assert state["exp_avg"].dtype == torch.bfloat16
        assert state["exp_avg_sq"].dtype == torch.bfloat16


def test_bf16_optimizer_load_preserves_large_steps_with_sparse_ids():
    source_parameters = [
        torch.nn.Parameter(torch.ones(2)),
        torch.nn.Parameter(torch.ones(3)),
    ]
    source_optimizer = BF16AdamW(
        [
            {"params": [source_parameters[0]]},
            {"params": [source_parameters[1]]},
        ],
        lr=1e-3,
    )
    for parameter in source_parameters:
        parameter.grad = torch.ones_like(parameter)
    source_optimizer.step()
    payload = source_optimizer.state_dict()

    old_ids = [serialized_id for group in payload["param_groups"] for serialized_id in group["params"]]
    new_ids = (11, 29)
    id_remap = dict(zip(old_ids, new_ids))
    payload["state"] = {id_remap[serialized_id]: state for serialized_id, state in payload["state"].items()}
    for group in payload["param_groups"]:
        group["params"] = [id_remap[serialized_id] for serialized_id in group["params"]]
    exact_steps = (16_777_217, 16_777_219)
    for serialized_id, exact_step in zip(new_ids, exact_steps):
        payload["state"][serialized_id]["step"] = exact_step

    restored_parameters = [
        torch.nn.Parameter(torch.ones(2)),
        torch.nn.Parameter(torch.ones(3)),
    ]
    restored_optimizer = BF16AdamW(
        [
            {"params": [restored_parameters[0]]},
            {"params": [restored_parameters[1]]},
        ],
        lr=1e-3,
    )
    restored_optimizer.load_state_dict(payload)

    for parameter, exact_step in zip(restored_parameters, exact_steps):
        restored_step = restored_optimizer.state[parameter]["step"]
        assert type(restored_step) is int
        assert restored_step == exact_step


@pytest.mark.parametrize(
    ("saved_bf16", "resume_bf16", "saved_dtype"),
    (
        (False, True, "float32"),
        (True, False, "bfloat16"),
    ),
)
def test_resume_rejects_optimizer_state_dtype_change_before_load(
    tmp_path,
    saved_bf16,
    resume_bf16,
    saved_dtype,
):
    model = TinyMixtralForCausalLM(tiny_config(num_hidden_layers=1))
    optimizer = make_adamw(
        model,
        lr=1e-3,
        weight_decay=0.0,
        bf16_states=saved_bf16,
    )
    scheduler = make_cosine_schedule(optimizer, 0, 2)
    input_ids = torch.randint(0, model.config.vocab_size, (2, 5))
    training_step(model, optimizer, scheduler, input_ids)
    checkpoint = save_checkpoint(model, optimizer, scheduler, tmp_path, 1, 10, 0, 2, 0, 12, 2, 5)

    state = torch.load(checkpoint / "training_state.pt", weights_only=True)
    assert state["cpt_optimizer_state_dtype"] == saved_dtype
    assert get_cpt_optimizer_state_dtype(optimizer) == saved_dtype
    resumed = TinyMixtralForCausalLM.from_pretrained(str(checkpoint))
    mismatched_optimizer = make_adamw(
        resumed,
        lr=1e-3,
        weight_decay=0.0,
        bf16_states=resume_bf16,
    )
    assert not mismatched_optimizer.state
    with pytest.raises(RuntimeError, match="dtype disagrees"):
        validate_cpt_resume_state(state, resumed, mismatched_optimizer)
    assert not mismatched_optimizer.state


@pytest.mark.parametrize(
    ("bf16_states", "tampered_dtype"),
    (
        (False, torch.bfloat16),
        (True, torch.float32),
    ),
)
def test_resume_rejects_raw_cpt_moment_dtype_mismatch_before_load(
    tmp_path,
    bf16_states,
    tampered_dtype,
):
    model = TinyMixtralForCausalLM(tiny_config(num_hidden_layers=1))
    optimizer = make_adamw(
        model,
        lr=1e-3,
        weight_decay=0.0,
        bf16_states=bf16_states,
    )
    scheduler = make_cosine_schedule(optimizer, 0, 2)
    input_ids = torch.randint(0, model.config.vocab_size, (2, 5))
    training_step(model, optimizer, scheduler, input_ids)
    checkpoint = save_checkpoint(model, optimizer, scheduler, tmp_path, 1, 10, 0, 2, 0, 12, 2, 5)

    state = torch.load(checkpoint / "training_state.pt", weights_only=True)
    target = model.cpt_trainable_parameters()[0]
    target_id = serialized_parameter_id_for(optimizer, state["opt"], target)
    target_state = state["opt"]["state"][target_id]
    target_state["exp_avg"] = target_state["exp_avg"].to(tampered_dtype)

    resumed = TinyMixtralForCausalLM.from_pretrained(str(checkpoint))
    resumed_optimizer = make_adamw(
        resumed,
        lr=1e-3,
        weight_decay=0.0,
        bf16_states=bf16_states,
    )
    assert not resumed_optimizer.state
    with pytest.raises(RuntimeError, match="dtype mismatch"):
        validate_cpt_resume_state(state, resumed, resumed_optimizer)
    assert not resumed_optimizer.state


@pytest.mark.parametrize("bf16_states", [False, True])
@pytest.mark.parametrize("legacy_single_group", [False, True])
def test_production_restore_helper_supports_grouped_and_legacy_adamw(
    tmp_path,
    bf16_states,
    legacy_single_group,
):
    (
        _,
        _,
        _,
        state,
        resumed_model,
        resumed_optimizer,
    ) = make_resume_case(
        tmp_path,
        bf16_states=bf16_states,
        legacy_single_group=legacy_single_group,
    )
    restored_optimizer, used_legacy_fallback = restore_cpt_optimizer_state(
        state,
        resumed_model,
        resumed_optimizer,
        lr=1e-3,
        weight_decay=0.0,
        bf16_states=bf16_states,
    )
    assert used_legacy_fallback is legacy_single_group
    assert (restored_optimizer is resumed_optimizer) is not legacy_single_group
    assert type(restored_optimizer) is (BF16AdamW if bf16_states else torch.optim.AdamW)
    assert (
        validate_cpt_optimizer_state(
            resumed_model,
            restored_optimizer,
            expected_presence=state["cpt_optimizer_state_presence"],
            expected_state_dtype=state["cpt_optimizer_state_dtype"],
        )
        == state["cpt_optimizer_state_presence"]
    )


def test_resume_rejects_permuted_cpt_serialized_ids_before_load(tmp_path):
    _, _, _, state, resumed_model, resumed_optimizer = make_resume_case(tmp_path)
    projection_ids = [
        serialized_id for name, serialized_id in state["cpt_optimizer_parameter_bindings"] if name.endswith(".projection")
    ]
    assert len(projection_ids) == 2
    first_id, second_id = projection_ids
    for group in state["opt"]["param_groups"]:
        group["params"] = [
            second_id if serialized_id == first_id else first_id if serialized_id == second_id else serialized_id
            for serialized_id in group["params"]
        ]
    assert_restore_rejected_before_load(
        state,
        resumed_model,
        resumed_optimizer,
        match="canonical order",
    )


def test_resume_rejects_cpt_parameter_binding_drift_before_load(tmp_path):
    _, _, _, state, resumed_model, resumed_optimizer = make_resume_case(tmp_path)
    bindings = list(state["cpt_optimizer_parameter_bindings"])
    first_name, first_id = bindings[0]
    second_name, second_id = bindings[1]
    bindings[0] = (first_name, second_id)
    bindings[1] = (second_name, first_id)
    state["cpt_optimizer_parameter_bindings"] = tuple(bindings)
    assert_restore_rejected_before_load(
        state,
        resumed_model,
        resumed_optimizer,
        match="bindings disagree",
    )


def test_resume_rejects_cpt_entry_and_manifest_deleted_together(tmp_path):
    _, _, _, state, resumed_model, resumed_optimizer = make_resume_case(tmp_path)
    target_name, target_id = state["cpt_optimizer_parameter_bindings"][0]
    state["opt"]["state"].pop(target_id)
    state["cpt_optimizer_state_presence"] = tuple(name for name in state["cpt_optimizer_state_presence"] if name != target_name)
    assert_restore_rejected_before_load(
        state,
        resumed_model,
        resumed_optimizer,
        match="empty or contain every",
    )


def test_resume_rejects_all_cpt_entries_and_manifest_deleted_together(tmp_path):
    _, _, _, state, resumed_model, resumed_optimizer = make_resume_case(tmp_path)
    for _, serialized_id in state["cpt_optimizer_parameter_bindings"]:
        state["opt"]["state"].pop(serialized_id)
    state["cpt_optimizer_state_presence"] = ()

    assert resumed_model.get_cpt_optimizer_step() == 1
    assert_restore_rejected_before_load(
        state,
        resumed_model,
        resumed_optimizer,
        match="missing after initialization",
    )


def test_resume_rejects_disagreeing_cpt_optimizer_steps_before_load(tmp_path):
    _, _, _, state, resumed_model, resumed_optimizer = make_resume_case(
        tmp_path,
        steps=2,
    )
    _, target_id = state["cpt_optimizer_parameter_bindings"][0]
    state["opt"]["state"][target_id]["step"] = torch.tensor(
        1.0,
        dtype=torch.float32,
    )
    assert_restore_rejected_before_load(
        state,
        resumed_model,
        resumed_optimizer,
        match="steps disagree",
    )


def test_resume_rejects_all_cpt_steps_drifted_from_model_step(tmp_path):
    _, _, _, state, resumed_model, resumed_optimizer = make_resume_case(tmp_path)
    for router in resumed_model._cpt_routers():
        router.state_version.fill_(2)
    state["cpt_state_version"] = 2
    for _, serialized_id in state["cpt_optimizer_parameter_bindings"]:
        state["opt"]["state"][serialized_id]["step"] = torch.tensor(
            2.0,
            dtype=torch.float32,
        )

    assert resumed_model.get_cpt_optimizer_step() == 1
    assert_restore_rejected_before_load(
        state,
        resumed_model,
        resumed_optimizer,
        match="disagrees with the model checkpoint",
    )


@pytest.mark.parametrize(
    ("saved_step", "match"),
    (
        (0.0, "at least one"),
        (2.0, "exceeds the model CPT state version"),
    ),
)
def test_resume_rejects_cpt_steps_outside_state_version(
    tmp_path,
    saved_step,
    match,
):
    _, _, _, state, resumed_model, resumed_optimizer = make_resume_case(tmp_path)
    for _, serialized_id in state["cpt_optimizer_parameter_bindings"]:
        state["opt"]["state"][serialized_id]["step"] = torch.tensor(
            saved_step,
            dtype=torch.float32,
        )
    assert_restore_rejected_before_load(
        state,
        resumed_model,
        resumed_optimizer,
        match=match,
    )


@pytest.mark.parametrize(
    ("bf16_states", "invalid_step", "match"),
    (
        (False, "python_int", "FP32 scalar tensor"),
        (False, "float64_tensor", "FP32 scalar tensor"),
        (False, "attached_tensor", "must be detached"),
        (True, "tensor", "Python integer"),
        (True, "bool", "Python integer"),
    ),
)
def test_resume_rejects_noncanonical_cpt_step_representation_before_load(
    tmp_path,
    bf16_states,
    invalid_step,
    match,
):
    _, _, _, state, resumed_model, resumed_optimizer = make_resume_case(
        tmp_path,
        bf16_states=bf16_states,
    )
    _, target_id = state["cpt_optimizer_parameter_bindings"][0]
    if invalid_step == "python_int":
        value = 1
    elif invalid_step == "float64_tensor":
        value = torch.tensor(1.0, dtype=torch.float64)
    elif invalid_step == "attached_tensor":
        value = torch.tensor(1.0, dtype=torch.float32, requires_grad=True)
    elif invalid_step == "tensor":
        value = torch.tensor(1.0, dtype=torch.float32)
    else:
        value = True
    state["opt"]["state"][target_id]["step"] = value
    assert_restore_rejected_before_load(
        state,
        resumed_model,
        resumed_optimizer,
        bf16_states=bf16_states,
        match=match,
    )


def test_resume_rejects_unknown_cpt_training_metadata_before_load(tmp_path):
    _, _, _, state, resumed_model, resumed_optimizer = make_resume_case(tmp_path)
    state["cpt_future_schema"] = 1
    assert_restore_rejected_before_load(
        state,
        resumed_model,
        resumed_optimizer,
        match="unknown CPT fields",
    )


def test_resume_allows_ordinary_future_training_metadata(tmp_path):
    _, _, _, state, resumed_model, resumed_optimizer = make_resume_case(tmp_path)
    state["future_upstream_metadata"] = {"schema": 1}
    restored_optimizer, used_legacy_fallback = restore_cpt_optimizer_state(
        state,
        resumed_model,
        resumed_optimizer,
        lr=1e-3,
        weight_decay=0.0,
        bf16_states=False,
    )
    assert restored_optimizer is resumed_optimizer
    assert used_legacy_fallback is False


@pytest.mark.parametrize("legacy_single_group", [False, True])
@pytest.mark.parametrize(
    ("field", "invalid_value"),
    (
        ("lr", float("nan")),
        ("initial_lr", float("inf")),
        ("weight_decay", -1.0),
    ),
)
def test_resume_rejects_invalid_optimizer_group_scalar_before_load(
    tmp_path,
    monkeypatch,
    legacy_single_group,
    field,
    invalid_value,
):
    _, _, _, state, resumed_model, resumed_optimizer = make_resume_case(
        tmp_path,
        legacy_single_group=legacy_single_group,
    )
    state["opt"]["param_groups"][0][field] = invalid_value
    load_calls = []
    original_load_state_dict = torch.optim.AdamW.load_state_dict

    def tracked_load_state_dict(optimizer, optimizer_state):
        load_calls.append((optimizer, optimizer_state))
        return original_load_state_dict(optimizer, optimizer_state)

    monkeypatch.setattr(
        torch.optim.AdamW,
        "load_state_dict",
        tracked_load_state_dict,
    )
    assert not resumed_optimizer.state
    with pytest.raises(RuntimeError, match="finite non-negative float"):
        restore_cpt_optimizer_state(
            state,
            resumed_model,
            resumed_optimizer,
            lr=1e-3,
            weight_decay=0.0,
            bf16_states=False,
        )
    assert load_calls == []
    assert not resumed_optimizer.state


def test_bf16_production_restore_preserves_large_exact_cpt_step(tmp_path):
    _, _, _, state, resumed_model, resumed_optimizer = make_resume_case(
        tmp_path,
        bf16_states=True,
    )
    exact_step = 16_777_217
    for router in resumed_model._cpt_routers():
        router.optimizer_step.fill_(exact_step)
        router.state_version.fill_(exact_step)
    state["cpt_state_version"] = exact_step
    for _, serialized_id in state["cpt_optimizer_parameter_bindings"]:
        state["opt"]["state"][serialized_id]["step"] = exact_step

    restored_optimizer, used_legacy_fallback = restore_cpt_optimizer_state(
        state,
        resumed_model,
        resumed_optimizer,
        lr=1e-3,
        weight_decay=0.0,
        bf16_states=True,
    )

    assert restored_optimizer is resumed_optimizer
    assert used_legacy_fallback is False
    for parameter in resumed_model.cpt_trainable_parameters():
        restored_step = restored_optimizer.state[parameter]["step"]
        assert type(restored_step) is int
        assert restored_step == exact_step
    assert resumed_model.get_cpt_optimizer_step() == exact_step


@pytest.mark.parametrize(
    ("mutation", "match"),
    (
        ("optimizer_top_level", "unexpected or missing fields"),
        ("orphan_state", "orphan entry"),
        ("group_unknown", "unexpected or missing fields"),
        ("group_missing", "unexpected or missing fields"),
        ("algorithm_value", "changes AdamW algorithm field betas"),
        ("algorithm_type", "changes AdamW algorithm field capturable"),
        ("invalid_parameter_id", "parameter id is invalid"),
        ("duplicate_parameter_id", "canonical order"),
        ("extra_parameter_group", "parameter-group structure mismatch"),
        ("cpt_entry_extra", "unexpected fields"),
        ("cpt_entry_missing", "incomplete"),
    ),
)
def test_resume_rejects_malformed_optimizer_payload_before_load(
    tmp_path,
    mutation,
    match,
):
    _, _, _, state, resumed_model, resumed_optimizer = make_resume_case(tmp_path)
    optimizer_state = state["opt"]
    first_group = optimizer_state["param_groups"][0]
    _, first_cpt_id = state["cpt_optimizer_parameter_bindings"][0]
    if mutation == "optimizer_top_level":
        optimizer_state["future"] = None
    elif mutation == "orphan_state":
        orphan_id = max(serialized_id for group in optimizer_state["param_groups"] for serialized_id in group["params"]) + 1
        optimizer_state["state"][orphan_id] = copy.deepcopy(optimizer_state["state"][first_cpt_id])
    elif mutation == "group_unknown":
        first_group["future"] = None
    elif mutation == "group_missing":
        first_group.pop("eps")
    elif mutation == "algorithm_value":
        first_group["betas"] = (0.8, 0.95)
    elif mutation == "algorithm_type":
        first_group["capturable"] = 0
    elif mutation == "invalid_parameter_id":
        first_group["params"][0] = -1
    elif mutation == "duplicate_parameter_id":
        first_group["params"][1] = first_group["params"][0]
    elif mutation == "extra_parameter_group":
        optimizer_state["param_groups"].append(copy.deepcopy(optimizer_state["param_groups"][-1]))
    elif mutation == "cpt_entry_extra":
        optimizer_state["state"][first_cpt_id]["future"] = None
    else:
        optimizer_state["state"][first_cpt_id].pop("exp_avg")
    assert_restore_rejected_before_load(
        state,
        resumed_model,
        resumed_optimizer,
        match=match,
    )


def test_save_training_state_rejects_partial_live_cpt_state(tmp_path):
    (
        model,
        optimizer,
        scheduler,
        _,
        _,
        _,
    ) = make_resume_case(tmp_path)
    optimizer.state.pop(model.cpt_trainable_parameters()[0])
    invalid_path = tmp_path / "invalid_training_state.pt"
    with pytest.raises(RuntimeError, match="empty or contain every"):
        save_training_state(
            invalid_path,
            model,
            optimizer,
            scheduler,
            step=1,
            total_tok=10,
            warmup_steps=0,
            total_steps=2,
        )
    assert not invalid_path.exists()


@pytest.mark.parametrize("bf16_states", [False, True])
def test_legacy_single_group_optimizer_preserves_cpt_manifest(bf16_states):
    model = TinyMixtralForCausalLM(tiny_config(num_hidden_layers=1))
    optimizer_class = BF16AdamW if bf16_states else torch.optim.AdamW
    optimizer = optimizer_class(
        model.parameters(),
        lr=1e-3,
        weight_decay=0.0,
        betas=(0.9, 0.95),
    )
    scheduler = make_cosine_schedule(optimizer, 0, 2)
    input_ids = torch.randint(0, model.config.vocab_size, (2, 5))
    training_step(model, optimizer, scheduler, input_ids)
    presence = validate_cpt_optimizer_state(model, optimizer)
    payload = optimizer.state_dict()

    restored_model = TinyMixtralForCausalLM(tiny_config(num_hidden_layers=1))
    restored_model.load_state_dict(model.state_dict())
    grouped_optimizer = make_adamw(
        restored_model,
        lr=1e-3,
        weight_decay=0.0,
        bf16_states=bf16_states,
    )
    with pytest.raises(ValueError):
        grouped_optimizer.load_state_dict(payload)

    restored_optimizer = optimizer_class(
        restored_model.parameters(),
        lr=1e-3,
        weight_decay=0.0,
        betas=(0.9, 0.95),
    )
    expected_state_dtype = "bfloat16" if bf16_states else "float32"
    validate_serialized_cpt_optimizer_state(
        restored_model,
        restored_optimizer,
        payload,
        expected_presence=presence,
        expected_parameter_bindings=cpt_parameter_bindings_for(
            model,
            optimizer,
            payload,
        ),
        expected_state_dtype=expected_state_dtype,
    )
    restored_optimizer.load_state_dict(payload)
    validate_cpt_optimizer_state(
        restored_model,
        restored_optimizer,
        expected_presence=presence,
        expected_state_dtype=expected_state_dtype,
    )


def test_resume_rejects_invalid_cpt_training_state_identity():
    model = TinyMixtralForCausalLM(tiny_config())
    optimizer = make_adamw(model, lr=1e-3, weight_decay=0.0)
    optimizer_state = optimizer.state_dict()
    parameter_bindings = cpt_parameter_bindings_for(
        model,
        optimizer,
        optimizer_state,
    )
    with pytest.raises(RuntimeError, match="incomplete"):
        validate_cpt_resume_state({}, model, optimizer)
    for invalid in (True, -1, 1.5):
        with pytest.raises(RuntimeError, match="invalid"):
            validate_cpt_resume_state(
                {
                    "opt": {},
                    "sched": {},
                    "cpt_state_version": invalid,
                    "cpt_optimizer_state_presence": (),
                    "cpt_optimizer_parameter_bindings": parameter_bindings,
                    "cpt_optimizer_state_dtype": "float32",
                },
                model,
                optimizer,
            )
    with pytest.raises(RuntimeError, match="disagrees"):
        validate_cpt_resume_state(
            {
                "opt": {},
                "sched": {},
                "cpt_state_version": 1,
                "cpt_optimizer_state_presence": (),
                "cpt_optimizer_parameter_bindings": parameter_bindings,
                "cpt_optimizer_state_dtype": "float32",
            },
            model,
            optimizer,
        )
    for invalid_dtype in (None, True, "fp32", "float16"):
        with pytest.raises(RuntimeError, match="cpt_optimizer_state_dtype"):
            validate_cpt_resume_state(
                {
                    "opt": {},
                    "sched": {},
                    "cpt_state_version": 0,
                    "cpt_optimizer_state_presence": (),
                    "cpt_optimizer_parameter_bindings": parameter_bindings,
                    "cpt_optimizer_state_dtype": invalid_dtype,
                },
                model,
                optimizer,
            )


def test_cpt_state_version_is_independent_of_post_training_run_step():
    model = TinyMixtralForCausalLM(tiny_config())
    optimizer = make_adamw(model, lr=1e-3, weight_decay=0.0)
    scheduler = make_cosine_schedule(optimizer, 0, 2)
    input_ids = torch.randint(0, model.config.vocab_size, (2, 5))
    training_step(model, optimizer, scheduler, input_ids)
    state = {
        "opt": optimizer.state_dict(),
        "sched": scheduler.state_dict(),
        "cpt_state_version": 1,
        "cpt_optimizer_state_presence": validate_cpt_optimizer_state(
            model,
            optimizer,
        ),
        "cpt_optimizer_parameter_bindings": cpt_parameter_bindings_for(
            model,
            optimizer,
            optimizer.state_dict(),
        ),
        "cpt_optimizer_state_dtype": "float32",
        "step": 0,
    }
    assert state["cpt_optimizer_state_presence"]
    assert validate_cpt_resume_state(state, model, optimizer) == 1


def test_price_and_learnable_router_tensors_remain_fp32_after_bfloat16():
    model = TinyMixtralForCausalLM(tiny_config()).to(torch.bfloat16)
    for router in model._cpt_routers():
        assert router.projection.dtype == torch.float32
        assert router.anchors.dtype == torch.float32
        assert router.energy.dtype == torch.float32
        assert router.congestion_price.dtype == torch.float32


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_native_training_loop_runs_one_cpt_step(tmp_path):
    model = TinyMixtralForCausalLM(tiny_config(num_hidden_layers=1))
    model = model.to("cuda").to(torch.bfloat16)
    model.gradient_checkpointing_enable()
    optimizer = make_adamw(model, lr=1e-3, weight_decay=0.0)
    scheduler = make_cosine_schedule(optimizer, 0, 1)
    shard_path = tmp_path / "train_00000.pt"
    torch.save(
        torch.randint(0, model.config.vocab_size, (12,), dtype=torch.long),
        shard_path,
    )

    step, total_tok, fi, ptr, _ = training_loop(
        model,
        optimizer,
        scheduler,
        [str(shard_path)],
        fi=0,
        ptr=0,
        total_tok=0,
        bs=2,
        seq=5,
        chunk=12,
        output_dir=tmp_path / "output",
        max_steps=1,
        save_every_min=10_000,
        log_every=1,
        step_start=0,
        schedule_args={"warmup_steps": 0, "total_steps": 1},
    )
    assert (step, total_tok, fi, ptr) == (1, 10, 0, 12)
    assert model.get_cpt_state_version() == 1
    validate_cpt_optimizer_state(model, optimizer)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_cuda_bf16_native_step_checkpoint_and_reload(tmp_path):
    torch.manual_seed(17)
    model = TinyMixtralForCausalLM(tiny_config()).to("cuda").to(torch.bfloat16)
    model.gradient_checkpointing_enable()
    optimizer = make_adamw(model, lr=2e-4, weight_decay=0.01, bf16_states=True)
    scheduler = make_cosine_schedule(optimizer, 0, 3)
    input_ids = torch.randint(0, model.config.vocab_size, (2, 6), device="cuda")
    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
        output = model(input_ids, labels=input_ids)
    assert torch.isfinite(output["loss"])
    output["loss"].backward()
    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    assert torch.isfinite(grad_norm)
    _execute_cpt_optimizer_step(model, optimizer, scheduler, output["cpt_transaction"])
    optimizer.zero_grad(set_to_none=True)
    assert model.get_cpt_state_version() == 1
    for router in model._cpt_routers():
        assert router.congestion_price.dtype == torch.float32
        for parameter in router.trainable_parameters():
            assert parameter.dtype == torch.float32
            assert optimizer.state[parameter]["exp_avg"].dtype == torch.bfloat16
            assert optimizer.state[parameter]["exp_avg_sq"].dtype == torch.bfloat16

    checkpoint = save_checkpoint(model, optimizer, scheduler, tmp_path, 1, 12, 0, 3, 0, 14, 2, 6)
    reloaded = TinyMixtralForCausalLM.from_pretrained(str(checkpoint))
    assert reloaded.get_cpt_state_version() == 1
    for key, value in model.state_dict().items():
        assert torch.equal(reloaded.state_dict()[key], value.detach().cpu())
