# Copyright (C) Michael Lee (李登淳) 2026. All rights reserved.
# Open-source under the MIT License. See LICENSE for details.
"""train.py 和 resume.py 共享的训练逻辑。"""

import copy, math, sys, time, os, shutil, subprocess, json, signal
from pathlib import Path

import torch


# ============================================================
# 共享工具
# ============================================================

class BF16AdamW(torch.optim.AdamW):
    """AdamW that stores optimizer states in bfloat16 to save VRAM.

    States (exp_avg, exp_avg_sq) are kept in bf16 between steps and
    cast to fp32 only during the update computation, saving ~50% of
    optimizer state memory at the cost of minor precision loss.
    """

    def __init__(self, *args, **kwargs):
        kwargs.setdefault("foreach", False)
        kwargs.setdefault("fused", False)
        super().__init__(*args, **kwargs)

    def load_state_dict(self, state_dict):
        """Restore the optimizer's declared BF16 storage policy exactly."""
        exact_steps = {}
        if isinstance(state_dict, dict):
            serialized_state = state_dict.get("state")
            serialized_groups = state_dict.get("param_groups")
            if (
                isinstance(serialized_state, dict)
                and isinstance(serialized_groups, list)
                and len(serialized_groups) == len(self.param_groups)
            ):
                for live_group, serialized_group in zip(
                    self.param_groups,
                    serialized_groups,
                ):
                    live_parameters = live_group.get("params")
                    serialized_ids = (
                        serialized_group.get("params")
                        if isinstance(serialized_group, dict)
                        else None
                    )
                    if (
                        not isinstance(live_parameters, list)
                        or not isinstance(serialized_ids, list)
                        or len(live_parameters) != len(serialized_ids)
                    ):
                        continue
                    for parameter, serialized_id in zip(
                        live_parameters,
                        serialized_ids,
                    ):
                        serialized_entry = serialized_state.get(serialized_id)
                        if not isinstance(serialized_entry, dict):
                            continue
                        serialized_step = serialized_entry.get("step")
                        if type(serialized_step) is int:
                            exact_steps[parameter] = _optimizer_step_value(
                                serialized_step
                            )
        result = super().load_state_dict(state_dict)
        for parameter, state in self.state.items():
            if not state:
                continue
            exact_step = exact_steps.get(parameter)
            if exact_step is not None:
                state["step"] = exact_step
            else:
                step = state.get("step")
                if step is not None:
                    state["step"] = _optimizer_step_value(step)
            for name in ("exp_avg", "exp_avg_sq"):
                value = state.get(name)
                if isinstance(value, torch.Tensor):
                    state[name] = value.to(dtype=torch.bfloat16)
        return result

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            beta1, beta2 = group["betas"]
            lr = group["lr"]
            wd = group["weight_decay"]
            eps = group["eps"]

            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad.float()

                state = self.state[p]
                if len(state) == 0:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros(p.shape, dtype=torch.bfloat16, device=p.device)
                    state["exp_avg_sq"] = torch.zeros(p.shape, dtype=torch.bfloat16, device=p.device)

                state["step"] = int(state["step"]) + 1
                t = state["step"]

                exp_avg = state["exp_avg"].float()
                exp_avg_sq = state["exp_avg_sq"].float()

                exp_avg.lerp_(grad, 1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                bias_correction1 = 1 - beta1 ** t
                bias_correction2 = 1 - beta2 ** t
                step_size = lr / bias_correction1
                denom = (exp_avg_sq.sqrt() / (bias_correction2 ** 0.5)).add_(eps)

                if wd > 0:
                    p.mul_(1 - lr * wd)
                p.addcdiv_(exp_avg, denom, value=-step_size)

                state["exp_avg"] = exp_avg.to(torch.bfloat16)
                state["exp_avg_sq"] = exp_avg_sq.to(torch.bfloat16)

        return loss


def make_adamw(model, lr, weight_decay, betas=(0.9, 0.95), bf16_states=False):
    """构建 AdamW：矩阵权重衰减，RMSNorm 等 1D 参数不衰减。

    bf16_states=True 时使用 BF16AdamW，优化器状态存储为 bf16，
    节省约 50% 优化器显存。
    """
    decay = []
    no_decay = []
    for parameter in model.parameters():
        if not parameter.requires_grad:
            continue
        (decay if parameter.ndim >= 2 else no_decay).append(parameter)
    cls = BF16AdamW if bf16_states else torch.optim.AdamW
    return cls(
        [
            {"params": decay, "weight_decay": weight_decay},
            {"params": no_decay, "weight_decay": 0.0},
        ],
        lr=lr,
        betas=betas,
    )


_CPT_OPTIMIZER_STATE_DTYPES = {
    "float32": torch.float32,
    "bfloat16": torch.bfloat16,
}
_ADAMW_OPTIMIZER_STATE_FIELDS = frozenset({"state", "param_groups"})
_ADAMW_PARAMETER_GROUP_FIELDS = frozenset(
    {
        "params",
        "lr",
        "betas",
        "eps",
        "weight_decay",
        "amsgrad",
        "maximize",
        "foreach",
        "capturable",
        "differentiable",
        "fused",
        "decoupled_weight_decay",
    }
)
_ADAMW_SCHEDULER_GROUP_FIELD = "initial_lr"
_MAX_EXACT_FP32_INTEGER = 1 << 24


def _validate_adamw_group_float(name, value, expected_value) -> None:
    if type(expected_value) is not float:
        raise RuntimeError(
            "live AdamW parameter group uses an unsupported "
            f"{name} scalar type"
        )
    if type(value) is not float:
        raise RuntimeError(
            "serialized optimizer parameter group has an invalid "
            f"{name} type"
        )
    if not math.isfinite(value) or value < 0.0:
        raise RuntimeError(
            "serialized optimizer parameter group "
            f"{name} must be a finite non-negative float"
        )


def _normalize_cpt_optimizer_state_dtype(value: str) -> str:
    if not isinstance(value, str) or value not in _CPT_OPTIMIZER_STATE_DTYPES:
        raise RuntimeError(
            "cpt_optimizer_state_dtype must be 'float32' or 'bfloat16'"
        )
    return value


def get_cpt_optimizer_state_dtype(optimizer) -> str:
    """Return the checkpointed storage dtype declared by the optimizer type."""
    optimizer_type = type(optimizer)
    if optimizer_type is BF16AdamW:
        return "bfloat16"
    if optimizer_type is torch.optim.AdamW:
        return "float32"
    raise TypeError(
        "CPT checkpointing supports torch.optim.AdamW and BF16AdamW only"
    )


def validate_cpt_optimizer_state_dtype(optimizer, expected: str) -> str:
    """Fail closed when resume would change CPT AdamW moment storage."""
    expected = _normalize_cpt_optimizer_state_dtype(expected)
    actual = get_cpt_optimizer_state_dtype(optimizer)
    if actual != expected:
        raise RuntimeError(
            "CPT optimizer state dtype disagrees with checkpoint: "
            f"checkpoint={expected}, live={actual}; resume with --bf16-optim "
            "exactly as used when the checkpoint was saved"
        )
    return actual


def _unwrap_cpt_model(model):
    module = getattr(model, "module", None)
    if module is not None and hasattr(module, "validate_cpt_transaction"):
        return module
    if hasattr(model, "validate_cpt_transaction"):
        return model
    raise TypeError("training requires a Native TinyMixtral CPT model")


def _assert_not_poisoned(*objects) -> None:
    if any(bool(getattr(obj, "_tinymixtral_fail_stop", False)) for obj in objects):
        raise RuntimeError(
            "training state is fail-stop poisoned; restore the last "
            "successful checkpoint"
        )


def _mark_fail_stop(*objects) -> None:
    errors = []
    for obj in objects:
        try:
            setattr(obj, "_tinymixtral_fail_stop", True)
        except BaseException as error:
            errors.append(error)
    if errors:
        raise RuntimeError(
            "failed to mark every training object as fail-stop"
        ) from errors[0]


def _assert_cpt_optimizer_binding(model, optimizer):
    """Require every Native CPT learnable parameter exactly once."""
    cpt_model = _unwrap_cpt_model(model)
    required = tuple(cpt_model.cpt_trainable_parameters())
    counts = {}
    for group in optimizer.param_groups:
        for parameter in group.get("params", ()):
            identity = id(parameter)
            counts[identity] = counts.get(identity, 0) + 1
    for parameter in required:
        count = counts.get(id(parameter), 0)
        if count != 1:
            raise RuntimeError(
                "optimizer must contain every CPT learnable parameter exactly once"
            )
    return required


def _named_cpt_optimizer_parameters(model, optimizer):
    cpt_model = _unwrap_cpt_model(model)
    parameters = _assert_cpt_optimizer_binding(model, optimizer)
    names_by_identity = {
        id(parameter): name
        for name, parameter in cpt_model.named_parameters()
    }
    named_parameters = []
    for parameter in parameters:
        name = names_by_identity.get(id(parameter))
        if name is None:
            raise RuntimeError("CPT optimizer parameter has no stable model name")
        named_parameters.append((name, parameter))
    return tuple(named_parameters)


def _normalize_cpt_optimizer_state_presence(value, known_names):
    if not isinstance(value, (list, tuple)) or any(
        not isinstance(name, str) for name in value
    ):
        raise RuntimeError(
            "cpt_optimizer_state_presence must be a sequence of parameter names"
        )
    presence = tuple(value)
    if len(set(presence)) != len(presence):
        raise RuntimeError("cpt_optimizer_state_presence contains duplicates")
    known = tuple(known_names)
    present = set(presence)
    if not present.issubset(known):
        raise RuntimeError(
            "cpt_optimizer_state_presence contains unknown parameter names"
        )
    canonical = tuple(name for name in known if name in present)
    if presence != canonical:
        raise RuntimeError(
            "cpt_optimizer_state_presence is not in canonical parameter order"
        )
    if presence not in ((), known):
        raise RuntimeError(
            "cpt_optimizer_state_presence must be empty or contain every "
            "CPT learnable parameter"
        )
    return presence


def _normalize_cpt_optimizer_parameter_bindings(value, known_names):
    if not isinstance(value, (list, tuple)):
        raise RuntimeError(
            "cpt_optimizer_parameter_bindings must be an ordered sequence"
        )
    bindings = []
    for item in value:
        if not isinstance(item, (list, tuple)) or len(item) != 2:
            raise RuntimeError(
                "cpt_optimizer_parameter_bindings contains a malformed binding"
            )
        name, serialized_id = item
        if not isinstance(name, str):
            raise RuntimeError(
                "cpt_optimizer_parameter_bindings contains an invalid name"
            )
        if (
            isinstance(serialized_id, bool)
            or not isinstance(serialized_id, int)
            or serialized_id < 0
        ):
            raise RuntimeError(
                "cpt_optimizer_parameter_bindings contains an invalid parameter id"
            )
        bindings.append((name, serialized_id))

    bindings = tuple(bindings)
    known = tuple(known_names)
    if tuple(name for name, _ in bindings) != known:
        raise RuntimeError(
            "cpt_optimizer_parameter_bindings must bind every CPT parameter "
            "in canonical name order"
        )
    serialized_ids = tuple(serialized_id for _, serialized_id in bindings)
    if len(set(serialized_ids)) != len(serialized_ids):
        raise RuntimeError(
            "cpt_optimizer_parameter_bindings contains duplicate parameter ids"
        )
    return bindings


def _optimizer_step_value(value) -> int:
    if type(value) is int:
        if value < 0:
            raise RuntimeError(
                "CPT optimizer step must be a non-negative integer"
            )
        return value
    if isinstance(value, torch.Tensor):
        if value.is_meta:
            raise RuntimeError("CPT optimizer step must be materialized")
        if value.shape != torch.Size([]):
            raise RuntimeError("CPT optimizer step must be scalar")
        if value.requires_grad:
            raise RuntimeError("CPT optimizer step must be detached")
        scalar = value.item()
        if isinstance(scalar, bool) or not isinstance(scalar, (int, float)):
            raise RuntimeError(
                "CPT optimizer step must be a non-negative integer"
            )
        numeric = float(scalar)
        if (
            not math.isfinite(numeric)
            or numeric < 0.0
            or not numeric.is_integer()
        ):
            raise RuntimeError(
                "CPT optimizer step must be a non-negative integer"
            )
        return int(numeric)
    raise RuntimeError("CPT optimizer step must be a non-negative integer")


def _validate_cpt_optimizer_step(value, optimizer_type) -> int:
    if optimizer_type is BF16AdamW:
        if type(value) is not int:
            raise RuntimeError(
                "BF16AdamW CPT optimizer step must be a Python integer"
            )
    elif optimizer_type is torch.optim.AdamW:
        if not isinstance(value, torch.Tensor) or value.dtype != torch.float32:
            raise RuntimeError(
                "AdamW CPT optimizer step must be an FP32 scalar tensor"
            )
    else:
        raise TypeError(
            "CPT checkpointing supports torch.optim.AdamW and BF16AdamW only"
        )
    return _optimizer_step_value(value)


def _validate_cpt_optimizer_entry(
    state,
    parameter,
    expected_dtype,
    *,
    expected_device,
    optimizer_type,
) -> int:
    if not isinstance(state, dict):
        raise RuntimeError("CPT optimizer state is incomplete")
    required_fields = {"step", "exp_avg", "exp_avg_sq"}
    missing = required_fields - set(state)
    if missing:
        raise RuntimeError("CPT optimizer state is incomplete")
    unexpected = set(state) - required_fields
    if unexpected:
        raise RuntimeError("CPT optimizer state contains unexpected fields")
    step = _validate_cpt_optimizer_step(state["step"], optimizer_type)
    for name in ("exp_avg", "exp_avg_sq"):
        value = state[name]
        if not isinstance(value, torch.Tensor):
            raise RuntimeError(f"CPT optimizer {name} must be a tensor")
        if value.is_meta:
            raise RuntimeError(f"CPT optimizer {name} must be materialized")
        if value.shape != parameter.shape:
            raise RuntimeError(f"CPT optimizer {name} shape mismatch")
        if value.dtype != expected_dtype:
            raise RuntimeError(f"CPT optimizer {name} dtype mismatch")
        if expected_device is not None and value.device != expected_device:
            raise RuntimeError(f"CPT optimizer {name} device mismatch")
        if value.requires_grad:
            raise RuntimeError(f"CPT optimizer {name} must be detached")
        if not bool(torch.isfinite(value).all()):
            raise RuntimeError(f"CPT optimizer {name} is non-finite")
    if bool((state["exp_avg_sq"] < 0).any()):
        raise RuntimeError("CPT optimizer exp_avg_sq must be non-negative")
    return step


def _validate_cpt_optimizer_steps(
    steps,
    state_version,
    presence,
    expected_optimizer_step,
) -> int:
    if not presence:
        if steps:
            raise RuntimeError(
                "CPT optimizer steps exist without initialized CPT state"
            )
        if expected_optimizer_step != 0:
            raise RuntimeError(
                "CPT optimizer state is missing after initialization"
            )
        return 0
    if len(steps) != len(presence) or len(set(steps)) != 1:
        raise RuntimeError("CPT optimizer steps disagree across parameters")
    step = steps[0]
    if step < 1:
        raise RuntimeError(
            "initialized CPT optimizer state must have step at least one"
        )
    if step > state_version:
        raise RuntimeError(
            "CPT optimizer step exceeds the model CPT state version"
        )
    if step != expected_optimizer_step:
        raise RuntimeError(
            "CPT optimizer step disagrees with the model checkpoint"
        )
    return step


def _validate_live_cpt_optimizer_state(
    model,
    optimizer,
    *,
    expected_presence=None,
    expected_state_dtype=None,
    expected_optimizer_step=None,
    maximum_state_version=None,
):
    cpt_model = _unwrap_cpt_model(model)
    _assert_not_poisoned(model, cpt_model, optimizer)
    state_dtype = get_cpt_optimizer_state_dtype(optimizer)
    if expected_state_dtype is not None:
        state_dtype = validate_cpt_optimizer_state_dtype(
            optimizer,
            expected_state_dtype,
        )
    named_parameters = _named_cpt_optimizer_parameters(model, optimizer)
    known_names = tuple(name for name, _ in named_parameters)
    actual_presence = tuple(
        name
        for name, parameter in named_parameters
        if parameter in optimizer.state
    )
    actual_presence = _normalize_cpt_optimizer_state_presence(
        actual_presence,
        known_names,
    )
    if expected_presence is not None:
        normalized_presence = _normalize_cpt_optimizer_state_presence(
            expected_presence,
            known_names,
        )
        if actual_presence != normalized_presence:
            raise RuntimeError(
                "CPT optimizer state presence disagrees with checkpoint manifest"
            )
    expected_dtype = _CPT_OPTIMIZER_STATE_DTYPES[state_dtype]
    optimizer_type = type(optimizer)
    steps = []
    for _, parameter in named_parameters:
        if parameter not in optimizer.state:
            # AdamW initializes moments lazily.  A successful all-padding step
            # advances CPT state_version but produces no CPT gradients or moments.
            continue
        steps.append(
            _validate_cpt_optimizer_entry(
                optimizer.state[parameter],
                parameter,
                expected_dtype,
                expected_device=parameter.device,
                optimizer_type=optimizer_type,
            )
        )
    if expected_optimizer_step is None:
        expected_optimizer_step = cpt_model.get_cpt_optimizer_step()
    if maximum_state_version is None:
        maximum_state_version = cpt_model.get_cpt_state_version()
    optimizer_step = _validate_cpt_optimizer_steps(
        steps,
        maximum_state_version,
        actual_presence,
        expected_optimizer_step,
    )
    return actual_presence, optimizer_step


def validate_cpt_optimizer_state(
    model,
    optimizer,
    *,
    expected_presence=None,
    expected_state_dtype=None,
):
    """Validate every initialized Native CPT optimizer state exactly."""
    presence, _ = _validate_live_cpt_optimizer_state(
        model,
        optimizer,
        expected_presence=expected_presence,
        expected_state_dtype=expected_state_dtype,
    )
    return presence


def _preflight_cpt_optimizer_step(model, optimizer, transaction) -> int:
    """Bind the next optimizer delta to gradients before any parameter write."""
    cpt_model = _unwrap_cpt_model(model)
    cpt_model.validate_persistent_cpt_state(require_unit_anchors=True)
    parameters = _assert_cpt_optimizer_binding(model, optimizer)
    optimizer_type = type(optimizer)
    supported_optimizer = optimizer_type in (torch.optim.AdamW, BF16AdamW)
    if supported_optimizer:
        _validate_live_cpt_optimizer_state(model, optimizer)
    else:
        if any(parameter in optimizer.state for parameter in parameters):
            raise TypeError(
                "CPT transactional step validation supports AdamW only"
            )
        if cpt_model.get_cpt_optimizer_step() != 0:
            raise TypeError(
                "an unsupported optimizer cannot resume initialized CPT state"
            )

    gradient_presence = tuple(
        parameter.grad is not None for parameter in parameters
    )
    if any(gradient_presence) and not all(gradient_presence):
        raise RuntimeError(
            "CPT gradients must be present for every learnable Router "
            "parameter or for none of them"
        )
    has_gradients = bool(gradient_presence and gradient_presence[0])
    token_count = int(transaction.proposals[0].token_count.item())
    if token_count > 0 and not has_gradients:
        raise RuntimeError(
            "a CPT transaction with valid tokens requires gradients from "
            "a successful backward pass"
        )
    if token_count == 0 and has_gradients:
        raise RuntimeError(
            "an all-padding CPT transaction requires Router gradients "
            "to be absent"
        )

    state_version = cpt_model.get_cpt_state_version()
    if state_version == torch.iinfo(torch.int64).max:
        raise RuntimeError("CPT state_version overflow")
    current_optimizer_step = cpt_model.get_cpt_optimizer_step()
    if not has_gradients:
        return current_optimizer_step
    if current_optimizer_step == torch.iinfo(torch.int64).max:
        raise RuntimeError("CPT optimizer_step overflow")
    if (
        optimizer_type is torch.optim.AdamW
        and current_optimizer_step >= _MAX_EXACT_FP32_INTEGER
    ):
        raise RuntimeError(
            "standard AdamW cannot represent the next CPT optimizer step "
            "exactly; use a fresh supported training run"
        )
    return current_optimizer_step + 1


def _validate_post_step_cpt_optimizer_state(
    model,
    optimizer,
    expected_optimizer_step,
) -> None:
    """Require optimizer.step to realize exactly the preflight CPT delta."""
    if type(optimizer) in (torch.optim.AdamW, BF16AdamW):
        cpt_model = _unwrap_cpt_model(model)
        _validate_live_cpt_optimizer_state(
            model,
            optimizer,
            expected_optimizer_step=expected_optimizer_step,
            maximum_state_version=cpt_model.get_cpt_state_version() + 1,
        )
        return
    cpt_model = _unwrap_cpt_model(model)
    parameters = _assert_cpt_optimizer_binding(model, optimizer)
    if expected_optimizer_step != cpt_model.get_cpt_optimizer_step() or any(
        parameter in optimizer.state for parameter in parameters
    ):
        raise TypeError(
            "CPT transactional step validation supports AdamW only"
        )


def _serialized_optimizer_parameter_ids(optimizer, optimizer_state):
    if not isinstance(optimizer_state, dict):
        raise RuntimeError("serialized optimizer state must be a dictionary")
    canonical_state = optimizer.state_dict()
    if set(canonical_state) != _ADAMW_OPTIMIZER_STATE_FIELDS:
        raise RuntimeError("live AdamW optimizer state schema is unsupported")
    if set(optimizer_state) != _ADAMW_OPTIMIZER_STATE_FIELDS:
        raise RuntimeError(
            "serialized optimizer state contains unexpected or missing fields"
        )
    state = optimizer_state["state"]
    serialized_groups = optimizer_state["param_groups"]
    canonical_groups = canonical_state["param_groups"]
    if not isinstance(state, dict) or not isinstance(serialized_groups, list):
        raise RuntimeError("serialized optimizer state is malformed")
    if len(serialized_groups) != len(canonical_groups):
        raise RuntimeError(
            "serialized optimizer parameter-group structure mismatch"
        )

    parameter_ids = {}
    seen_serialized_ids = set()
    for live_group, canonical_group, serialized_group in zip(
        optimizer.param_groups,
        canonical_groups,
        serialized_groups,
    ):
        if not isinstance(serialized_group, dict):
            raise RuntimeError("serialized optimizer parameter group is malformed")
        canonical_group_fields = frozenset(canonical_group)
        supported_group_fields = (
            _ADAMW_PARAMETER_GROUP_FIELDS,
            _ADAMW_PARAMETER_GROUP_FIELDS
            | {_ADAMW_SCHEDULER_GROUP_FIELD},
        )
        if canonical_group_fields not in supported_group_fields:
            raise RuntimeError("live AdamW parameter-group schema is unsupported")
        serialized_group_fields = frozenset(serialized_group)
        allowed_serialized_fields = {canonical_group_fields}
        if _ADAMW_SCHEDULER_GROUP_FIELD not in canonical_group_fields:
            allowed_serialized_fields.add(
                _ADAMW_PARAMETER_GROUP_FIELDS
                | {_ADAMW_SCHEDULER_GROUP_FIELD}
            )
        if serialized_group_fields not in allowed_serialized_fields:
            raise RuntimeError(
                "serialized optimizer parameter group contains unexpected or "
                "missing fields"
            )
        if (
            _ADAMW_SCHEDULER_GROUP_FIELD in canonical_group
            and _ADAMW_SCHEDULER_GROUP_FIELD not in serialized_group
        ):
            raise RuntimeError(
                "serialized optimizer parameter group is missing initial_lr"
            )
        for name in canonical_group_fields - {
            "params",
            "lr",
            "weight_decay",
            _ADAMW_SCHEDULER_GROUP_FIELD,
        }:
            saved_value = serialized_group[name]
            canonical_value = canonical_group[name]
            if type(saved_value) is not type(canonical_value) or (
                isinstance(saved_value, torch.Tensor)
                and not torch.equal(saved_value, canonical_value)
            ) or (
                not isinstance(saved_value, torch.Tensor)
                and saved_value != canonical_value
            ):
                raise RuntimeError(
                    "serialized optimizer parameter group changes AdamW "
                    f"algorithm field {name}"
                )
        for name in ("lr", "weight_decay"):
            _validate_adamw_group_float(
                name,
                serialized_group[name],
                canonical_group[name],
            )
        if _ADAMW_SCHEDULER_GROUP_FIELD in serialized_group:
            canonical_initial_lr = canonical_group.get(
                _ADAMW_SCHEDULER_GROUP_FIELD,
                canonical_group["lr"],
            )
            _validate_adamw_group_float(
                _ADAMW_SCHEDULER_GROUP_FIELD,
                serialized_group[_ADAMW_SCHEDULER_GROUP_FIELD],
                canonical_initial_lr,
            )
        live_parameters = live_group.get("params")
        canonical_parameters = canonical_group.get("params")
        serialized_parameters = serialized_group.get("params")
        if not isinstance(live_parameters, list) or not isinstance(
            canonical_parameters,
            list,
        ) or not isinstance(
            serialized_parameters,
            list,
        ):
            raise RuntimeError("serialized optimizer parameter group is malformed")
        if (
            len(live_parameters) != len(canonical_parameters)
            or len(live_parameters) != len(serialized_parameters)
        ):
            raise RuntimeError(
                "serialized optimizer parameter-group structure mismatch"
            )
        for serialized_id in serialized_parameters:
            if (
                isinstance(serialized_id, bool)
                or not isinstance(serialized_id, int)
                or serialized_id < 0
            ):
                raise RuntimeError("serialized optimizer parameter id is invalid")
        if serialized_parameters != canonical_parameters:
            raise RuntimeError(
                "serialized optimizer parameter ids disagree with the live "
                "optimizer's canonical order"
            )
        for parameter, serialized_id in zip(
            live_parameters,
            serialized_parameters,
        ):
            if serialized_id in seen_serialized_ids:
                raise RuntimeError("serialized optimizer parameter ids are duplicated")
            parameter_identity = id(parameter)
            if parameter_identity in parameter_ids:
                raise RuntimeError("live optimizer parameters are duplicated")
            seen_serialized_ids.add(serialized_id)
            parameter_ids[parameter_identity] = serialized_id
    for serialized_id in state:
        if (
            isinstance(serialized_id, bool)
            or not isinstance(serialized_id, int)
            or serialized_id < 0
        ):
            raise RuntimeError("serialized optimizer state id is invalid")
        if serialized_id not in seen_serialized_ids:
            raise RuntimeError("serialized optimizer state contains an orphan entry")
    return state, parameter_ids


def _cpt_optimizer_parameter_bindings(named_parameters, parameter_ids):
    bindings = []
    for name, parameter in named_parameters:
        serialized_id = parameter_ids.get(id(parameter))
        if serialized_id is None:
            raise RuntimeError(
                "serialized optimizer is missing a bound CPT parameter"
            )
        bindings.append((name, serialized_id))
    return tuple(bindings)


def _serialized_cpt_optimizer_parameter_bindings(
    model,
    optimizer,
    optimizer_state,
):
    named_parameters = _named_cpt_optimizer_parameters(model, optimizer)
    _, parameter_ids = _serialized_optimizer_parameter_ids(
        optimizer,
        optimizer_state,
    )
    return _cpt_optimizer_parameter_bindings(named_parameters, parameter_ids)


def validate_serialized_cpt_optimizer_state(
    model,
    optimizer,
    optimizer_state,
    *,
    expected_presence,
    expected_parameter_bindings,
    expected_state_dtype,
):
    """Validate raw CPT AdamW entries before PyTorch can cast them on load."""
    cpt_model = _unwrap_cpt_model(model)
    _assert_not_poisoned(model, cpt_model, optimizer)
    state_dtype = validate_cpt_optimizer_state_dtype(
        optimizer,
        expected_state_dtype,
    )
    expected_dtype = _CPT_OPTIMIZER_STATE_DTYPES[state_dtype]
    named_parameters = _named_cpt_optimizer_parameters(model, optimizer)
    known_names = tuple(name for name, _ in named_parameters)
    expected_presence = _normalize_cpt_optimizer_state_presence(
        expected_presence,
        known_names,
    )
    expected_parameter_bindings = (
        _normalize_cpt_optimizer_parameter_bindings(
            expected_parameter_bindings,
            known_names,
        )
    )
    serialized_state, parameter_ids = _serialized_optimizer_parameter_ids(
        optimizer,
        optimizer_state,
    )
    actual_parameter_bindings = _cpt_optimizer_parameter_bindings(
        named_parameters,
        parameter_ids,
    )
    if actual_parameter_bindings != expected_parameter_bindings:
        raise RuntimeError(
            "CPT optimizer parameter bindings disagree with the checkpoint manifest"
        )

    actual_presence = []
    initialized_entries = []
    for name, parameter in named_parameters:
        serialized_id = parameter_ids[id(parameter)]
        if serialized_id in serialized_state:
            actual_presence.append(name)
            initialized_entries.append(
                (serialized_state[serialized_id], parameter)
            )
    actual_presence = _normalize_cpt_optimizer_state_presence(
        actual_presence,
        known_names,
    )
    if actual_presence != expected_presence:
        raise RuntimeError(
            "CPT optimizer state presence disagrees with checkpoint manifest"
        )
    optimizer_type = type(optimizer)
    steps = []
    for entry, parameter in initialized_entries:
        steps.append(
            _validate_cpt_optimizer_entry(
                entry,
                parameter,
                expected_dtype,
                expected_device=None,
                optimizer_type=optimizer_type,
            )
        )
    _validate_cpt_optimizer_steps(
        steps,
        cpt_model.get_cpt_state_version(),
        actual_presence,
        cpt_model.get_cpt_optimizer_step(),
    )
    return actual_presence


def _clone_optimizer_value(value):
    if isinstance(value, torch.Tensor):
        return value.detach().clone()
    if isinstance(value, dict):
        return {key: _clone_optimizer_value(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_clone_optimizer_value(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_clone_optimizer_value(item) for item in value)
    return copy.deepcopy(value)


_MISSING_OPTIMIZER_STATE = object()


def _snapshot_cpt_optimizer_state(model, optimizer):
    """Snapshot only CPT learnable tensors and their optimizer moments."""
    cpt_parameters = _assert_cpt_optimizer_binding(model, optimizer)

    snapshot = []
    for parameter in cpt_parameters:
        state = (
            _clone_optimizer_value(optimizer.state[parameter])
            if parameter in optimizer.state
            else _MISSING_OPTIMIZER_STATE
        )
        snapshot.append((parameter, parameter.detach().clone(), state))
    return tuple(snapshot)


@torch.no_grad()
def _restore_cpt_optimizer_state(optimizer, snapshot) -> None:
    for parameter, value, state in snapshot:
        parameter.copy_(value)
        if state is _MISSING_OPTIMIZER_STATE:
            optimizer.state.pop(parameter, None)
        else:
            optimizer.state[parameter] = _clone_optimizer_value(state)


def _execute_cpt_optimizer_step(model, optimizer, scheduler, transaction) -> int:
    """Apply one step, with CPT rollback and fail-stop on post-step failure."""
    cpt_model = _unwrap_cpt_model(model)
    _assert_not_poisoned(model, cpt_model, optimizer, scheduler)
    try:
        cpt_model.validate_cpt_transaction(transaction)
        expected_optimizer_step = _preflight_cpt_optimizer_step(
            model,
            optimizer,
            transaction,
        )
    except BaseException:
        try:
            cpt_model.abort_cpt_transaction(transaction)
        except BaseException:
            pass
        raise
    snapshot = _snapshot_cpt_optimizer_state(model, optimizer)
    persistent_snapshot = tuple(
        router.commit_snapshot() for router in cpt_model._cpt_routers()
    )
    try:
        optimizer.step()
        _validate_post_step_cpt_optimizer_state(
            model,
            optimizer,
            expected_optimizer_step,
        )
        version = cpt_model.commit_cpt_transaction(
            transaction,
            optimizer_step=expected_optimizer_step,
        )
        scheduler.step()
    except BaseException:
        recovery_errors = []
        try:
            cpt_model.abort_cpt_transaction(transaction)
        except BaseException as error:
            recovery_errors.append(error)
        try:
            _mark_fail_stop(cpt_model, model, optimizer, scheduler)
        except BaseException as error:
            recovery_errors.append(error)
        try:
            _restore_cpt_optimizer_state(optimizer, snapshot)
        except BaseException as error:
            recovery_errors.append(error)
        for router, router_snapshot in zip(
            cpt_model._cpt_routers(), persistent_snapshot
        ):
            try:
                router.restore_commit_snapshot(router_snapshot)
            except BaseException as error:
                recovery_errors.append(error)
        # A real optimizer may already have partially changed ordinary model
        # parameters.  Do not allow this live process to continue or save.
        if recovery_errors:
            failure_types = ", ".join(
                type(error).__name__ for error in recovery_errors
            )
            raise RuntimeError(
                "CPT step failed and recovery was incomplete: " + failure_types
            ) from recovery_errors[0]
        raise
    return version


def save_training_state(
    path,
    model,
    opt,
    sched,
    step,
    total_tok,
    warmup_steps,
    total_steps,
    fi=None,
    ptr=None,
    batch_size=None,
    seq_len=None,
):
    """保存 optimizer + scheduler + 数据位置状态到文件。"""
    cpt_model = _unwrap_cpt_model(model)
    _assert_not_poisoned(model, cpt_model, opt, sched)
    cpt_model.validate_persistent_cpt_state()
    cpt_state_version = cpt_model.get_cpt_state_version()
    cpt_optimizer_state_presence = validate_cpt_optimizer_state(model, opt)
    optimizer_state = opt.state_dict()
    cpt_optimizer_parameter_bindings = (
        _serialized_cpt_optimizer_parameter_bindings(
            model,
            opt,
            optimizer_state,
        )
    )
    cpt_optimizer_state_dtype = get_cpt_optimizer_state_dtype(opt)
    validate_serialized_cpt_optimizer_state(
        model,
        opt,
        optimizer_state,
        expected_presence=cpt_optimizer_state_presence,
        expected_parameter_bindings=cpt_optimizer_parameter_bindings,
        expected_state_dtype=cpt_optimizer_state_dtype,
    )
    state = {
        "opt": optimizer_state,
        "sched": sched.state_dict(),
        "step": step,
        "total_tok": total_tok,
        "warmup_steps": warmup_steps,
        "total_steps": total_steps,
        "cpt_state_version": cpt_state_version,
        "cpt_optimizer_state_presence": cpt_optimizer_state_presence,
        "cpt_optimizer_parameter_bindings": (
            cpt_optimizer_parameter_bindings
        ),
        "cpt_optimizer_state_dtype": cpt_optimizer_state_dtype,
    }
    if fi is not None:
        state["fi"] = fi
        state["ptr"] = ptr
    if batch_size is not None:
        state["batch_size"] = batch_size
        state["seq_len"] = seq_len
    torch.save(state, path)


def save_checkpoint(model, opt, sched, output_dir, step, total_tok,
                    warmup_steps, total_steps, fi, ptr,
                    batch_size=None, seq_len=None, final=False):
    """完整写入临时目录后原子发布 checkpoint。"""
    cpt_model = _unwrap_cpt_model(model)
    _assert_not_poisoned(model, cpt_model, opt, sched)
    cpt_model.validate_persistent_cpt_state()
    suffix = "_final" if final else ""
    target = Path(output_dir) / f"step_{step:07d}{suffix}"
    temp = target.with_name(f".{target.name}.tmp-{os.getpid()}")
    if target.exists():
        raise FileExistsError(f"Checkpoint already exists: {target}")
    if temp.exists():
        shutil.rmtree(temp)
    temp.mkdir(parents=True)
    try:
        torch.save(cpt_model.state_dict(), temp / "pytorch_model.bin")
        save_training_state(
            temp / "training_state.pt", model, opt, sched, step, total_tok,
            warmup_steps, total_steps, fi, ptr, batch_size, seq_len,
        )
        cpt_model.config.save_pretrained(str(temp))
        os.replace(temp, target)
    except BaseException:
        shutil.rmtree(temp, ignore_errors=True)
        raise
    return target


def prune_periodic_checkpoints(output_dir, keep_last):
    """仅保留最近的周期 checkpoint；final checkpoint 永不删除。"""
    checkpoints = sorted(
        path for path in Path(output_dir).glob("step_*")
        if path.is_dir() and not path.name.endswith("_final")
    )
    for path in checkpoints[:-keep_last]:
        shutil.rmtree(path)


def check_checkpoint_disk_space(model, output_dir, keep_last):
    """确认磁盘可容纳保留的周期 checkpoint、final 和一次原子临时写入。"""
    output_path = Path(output_dir)
    probe_path = output_path
    while not probe_path.exists():
        probe_path = probe_path.parent

    parameter_bytes = sum(
        parameter.numel() * parameter.element_size()
        for parameter in model.parameters() if parameter.requires_grad
    )
    buffer_bytes = sum(buffer.numel() * buffer.element_size() for buffer in model.buffers())
    checkpoint_bytes = int((3 * parameter_bytes + buffer_bytes) * 1.15)
    checkpoint_bytes = max(checkpoint_bytes, 64 * 1024**2)
    target_bytes = checkpoint_bytes * (keep_last + 2)
    existing_bytes = sum(
        file.stat().st_size
        for checkpoint in output_path.glob("step_*") if checkpoint.is_dir()
        for file in checkpoint.rglob("*") if file.is_file()
    ) if output_path.exists() else 0
    additional_bytes = max(checkpoint_bytes, target_bytes - existing_bytes)
    free_bytes = shutil.disk_usage(probe_path).free
    if free_bytes < additional_bytes:
        raise RuntimeError(
            f"Insufficient checkpoint disk space: need about "
            f"{additional_bytes / 1024**3:.1f} GiB more, "
            f"only {free_bytes / 1024**3:.1f} GiB free at {probe_path}"
        )
    print(
        f"Checkpoint disk preflight: ~{checkpoint_bytes / 1024**3:.1f} GiB each, "
        f"{free_bytes / 1024**3:.1f} GiB free",
        flush=True,
    )


def make_cosine_schedule(opt, warmup_steps, total_steps):
    """创建 cosine LR schedule with linear warmup，返回 LambdaLR。"""
    if total_steps <= 0:
        raise ValueError("total_steps must be positive")
    warmup_steps = min(max(warmup_steps, 0), max(total_steps - 1, 0))

    def lr_lambda(s):
        if warmup_steps > 0 and s < warmup_steps:
            return (s + 1) / warmup_steps
        if warmup_steps == 0:
            progress = s / max(1, total_steps - 1)
        else:
            progress = (s - warmup_steps + 1) / max(1, total_steps - warmup_steps)
        progress = min(max(progress, 0.0), 1.0)
        return 0.5 * (1.0 + math.cos(math.pi * progress))
    return torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)


def make_wsd_schedule(opt, warmup_steps, total_steps, decay_ratio=0.1):
    """创建 WSD (Warmup-Stable-Decay) LR schedule。

    warmup:  linear 0 → peak  (warmup_steps)
    stable:  constant peak     (warmup_steps → decay_start)
    decay:   linear peak → 0  (decay_start → total_steps)

    Args:
        decay_ratio: fraction of steps in decay phase (default 0.1 = 10%)
    """
    if total_steps <= 0:
        raise ValueError("total_steps must be positive")
    warmup_steps = min(max(warmup_steps, 0), max(total_steps - 2, 0))
    decay_steps = max(1, int(total_steps * decay_ratio))
    decay_start = max(warmup_steps + 1, total_steps - decay_steps)

    def lr_lambda(s):
        if s < warmup_steps:
            return (s + 1) / max(1, warmup_steps)
        if s < decay_start:
            return 1.0
        progress = (s - decay_start + 1) / max(1, total_steps - decay_start)
        progress = min(max(progress, 0.0), 1.0)
        return 1.0 - progress
    return torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)


# ============================================================
# CPU eval 子进程
# ============================================================

def run_cpu_eval(checkpoint_path, eval_dir):
    """子进程 CPU GLUE eval。"""
    script = Path(__file__).parent / "eval_glue.py"
    output = Path(eval_dir) / "summary.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    tokenizer_path = Path(__file__).parent.parent / "tokenizer"
    # 删除旧结果，防止子进程失败时误读
    if output.exists():
        output.unlink()
    cmd = [sys.executable, str(script),
           "--checkpoint", checkpoint_path, "--tokenizer", str(tokenizer_path),
           "--tasks", "sst2,mrpc,qnli,rte,cola", "--limit", "200",
           "--batch-size", "2", "--max-length", "256",
           "--device", "cpu", "--precision", "fp32",
           "--output", str(output), "--seed", "1234"]
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)
        if r.returncode != 0:
            stderr_tail = r.stderr.strip()[-500:] if r.stderr else "(empty)"
            print(f"  [eval err] exit={r.returncode} stderr={stderr_tail}", flush=True)
            return None, None  # 失败后不读旧结果
        if output.exists():
            with open(output) as f: d = json.load(f)
            return d.get("aggregate", {}).get("mean_score"), d.get("results", {})
    except Exception as e:
        print(f"  [eval err] {e}", flush=True)
    return None, None


def training_loop(model, opt, sched, files, fi, ptr, total_tok, bs, seq, chunk,
                  output_dir, max_steps, save_every_min, log_every, step_start=0,
                  schedule_args=None, eval_on_save=False,
                  keep_last_checkpoints=5):
    """按绝对 step 目标训练。

    schedule_args: 可选 dict with warmup_steps, total_steps，用于 checkpoint 恢复。
    """
    os.makedirs(output_dir, exist_ok=True)
    cpt_model = _unwrap_cpt_model(model)
    _assert_not_poisoned(model, cpt_model, opt, sched)
    validate_cpt_optimizer_state(model, opt)
    shard = torch.load(files[fi], weights_only=True)
    tok_base = total_tok
    sa = schedule_args or {}
    t0 = time.time()
    last_save = t0
    last_saved_step = None
    step = step_start
    stop_signal = None
    previous_handlers = {}

    def request_stop(signum, _frame):
        nonlocal stop_signal
        if stop_signal is not None:
            raise KeyboardInterrupt
        stop_signal = signum
        print(f"\n  Signal {signum} received; saving after current step...", flush=True)

    for signum in (signal.SIGINT, signal.SIGTERM):
        previous_handlers[signum] = signal.getsignal(signum)
        signal.signal(signum, request_stop)

    try:
        while step < max_steps:
            elapsed = time.time() - t0
            hours_elapsed = elapsed / 3600

            # ---- 分片循环 ----
            if ptr + chunk > len(shard):
                ptr = 0
                for _ in range(len(files)):
                    fi = (fi + 1) % len(files)
                    del shard
                    shard = torch.load(files[fi], weights_only=True)
                    if len(shard) >= chunk:
                        break
                else:
                    raise RuntimeError(f"No shard contains at least {chunk} tokens")

            batch = shard[ptr:ptr + chunk]
            if batch.numel() != chunk:
                print(f"  WARN: short read {batch.numel()}/{chunk} shard={fi}", flush=True)
                ptr = 0
                for _ in range(len(files)):
                    fi = (fi + 1) % len(files)
                    del shard
                    shard = torch.load(files[fi], weights_only=True)
                    if len(shard) >= chunk:
                        break
                else:
                    raise RuntimeError(f"No shard contains at least {chunk} tokens")
                continue

            batch = batch.view(bs, seq + 1).to("cuda", non_blocking=True)
            ptr += chunk
            total_tok += bs * seq

            transaction = None
            try:
                with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                    out = model(batch[:, :-1], labels=batch[:, 1:])
                transaction = out.get("cpt_transaction")
                if not torch.isfinite(out["loss"]):
                    raise FloatingPointError(
                        f"Non-finite loss at step {step + 1}: {out['loss'].item()}"
                    )
                out["loss"].backward()
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                if not torch.isfinite(grad_norm):
                    raise FloatingPointError(
                        f"Non-finite gradient norm at step {step + 1}: "
                        f"{grad_norm.item()}"
                    )
                # Preserve upstream optimizer/scheduler mathematics and
                # ordering; the only insertion is the CPT commit between them.
                _execute_cpt_optimizer_step(model, opt, sched, transaction)
            except BaseException:
                cpt_model.abort_cpt_transaction(transaction)
                opt.zero_grad(set_to_none=True)
                raise
            opt.zero_grad(set_to_none=True)
            step += 1

            if step % log_every == 0:
                elapsed = time.time() - t0
                hours_elapsed = elapsed / 3600
                print(f"  step {step:7d}: loss={out['loss'].item():.4f} "
                      f"aux={out['aux_loss'].item():.1f} "
                      f"tok/s={(total_tok - tok_base) / elapsed:.0f} "
                      f"lr={sched.get_last_lr()[0]:.2e} "
                      f"shard={fi}/{len(files)} "
                      f"[{hours_elapsed:.1f}h]", flush=True)

            # ---- 保存 + eval ----
            if time.time() - last_save > save_every_min * 60:
                d = save_checkpoint(
                    model, opt, sched, output_dir, step, total_tok,
                    sa.get("warmup_steps", 0), sa.get("total_steps", 0), fi, ptr,
                    bs, seq,
                )
                print(f"  -> Saved {d}", flush=True)
                prune_periodic_checkpoints(output_dir, keep_last_checkpoints)
                last_save = time.time()
                last_saved_step = step

                if eval_on_save:
                    eval_dir = f"evals/{Path(output_dir).name}/step_{step:07d}"
                    mean, res = run_cpu_eval(d, eval_dir)
                    if mean is not None and res:
                        parts = [f'{t}={r.get("accuracy", r.get("matthews_correlation", r.get("f1", float("nan")))):.3f}'
                                 for t, r in sorted(res.items())]
                        print(f"  [eval] mean={mean:.4f} | {' '.join(parts)}", flush=True)

            if stop_signal is not None:
                if last_saved_step == step:
                    print("  -> Current step was already checkpointed", flush=True)
                else:
                    d = save_checkpoint(
                        model, opt, sched, output_dir, step, total_tok,
                        sa.get("warmup_steps", 0), sa.get("total_steps", 0), fi, ptr,
                        bs, seq,
                    )
                    print(f"  -> Emergency checkpoint saved: {d}", flush=True)
                    prune_periodic_checkpoints(output_dir, keep_last_checkpoints)
                raise SystemExit(128 + stop_signal)
    finally:
        for signum, handler in previous_handlers.items():
            signal.signal(signum, handler)

    elapsed = time.time() - t0
    return step, total_tok, fi, ptr, elapsed


def final_save(model, opt, sched, output_dir, step, total_tok, elapsed,
               fi, ptr, batch_size, seq_len, schedule_args=None):
    """保存最终 checkpoint。"""
    sa = schedule_args or {}
    d = save_checkpoint(
        model, opt, sched, output_dir, step, total_tok,
        sa.get("warmup_steps", 0), sa.get("total_steps", 0), fi, ptr,
        batch_size, seq_len, final=True,
    )
    print(f"\nDone: {step} steps {total_tok / 1e9:.3f}B tokens "
          f"session={elapsed / 3600:.1f}h → {d}", flush=True)
