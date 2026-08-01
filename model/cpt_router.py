# Copyright (C) Michael Lee (李登淳) 2026. All rights reserved.
# Open-source under the MIT License. See LICENSE for details.

"""CPT-MoE probability router and optimizer-step transaction support.

Mathematical notation in this module follows the column-vector convention:

    q_t in R^{K x 1}, Q=[q_1,...,q_T] in R^{K x T},
    B in R^{K x N}, Pi=B^T Q in R^{N x T}.

The implementation stores tokens as rows. Therefore the final probability
matrix consumed by TinyMixtral is ``Pi.T = Q.T @ B`` with shape ``[T, N]``.
Only route-valid tokens appear in that matrix.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Hashable, Iterable, Optional, Sequence
import weakref

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F


LEGACY_ROUTER_KEY_SUFFIX = ".moe.router.weight"
FP32_ROUTER_PARAMETER_NAMES = ("projection", "anchors", "energy")
CPT_ROUTER_ALGORITHM_VERSION = 1


def validate_serialized_router_algorithm_version(
    serialized,
    *,
    key: str,
    expected: int = CPT_ROUTER_ALGORITHM_VERSION,
) -> None:
    """Reject a serialized Router identity before any dtype conversion."""
    valid = (
        isinstance(serialized, torch.Tensor)
        and not serialized.is_meta
        and serialized.ndim == 0
        and serialized.dtype == torch.int64
        and int(serialized.item()) == expected
    )
    if valid:
        return
    if isinstance(serialized, torch.Tensor):
        observed = (
            f"shape={tuple(serialized.shape)}, dtype={serialized.dtype}"
        )
        if not serialized.is_meta and serialized.ndim == 0:
            observed += f", value={serialized.item()!r}"
    else:
        observed = f"type={type(serialized).__name__}"
    raise RuntimeError(
        f"{key} must be a materialized int64 scalar equal to {expected}; "
        f"observed {observed}"
    )


def _device_predicates_all(predicates: Iterable[torch.Tensor]) -> bool:
    """Resolve many device-side scalar predicates with one sync per device.

    Callers build all ordinary-success validation predicates on the device.
    Only the final reduced scalar is transferred to the host.  When that gate
    fails, callers deliberately run their original ordered slow diagnostic so
    externally visible error precedence and messages remain unchanged.
    """
    grouped: dict[torch.device, list[torch.Tensor]] = {}
    for predicate in predicates:
        if not isinstance(predicate, torch.Tensor):
            raise TypeError("CPT validation predicates must be tensors")
        scalar = predicate.detach()
        if scalar.ndim != 0:
            scalar = scalar.all()
        if scalar.dtype != torch.bool:
            scalar = scalar.to(torch.bool)
        grouped.setdefault(scalar.device, []).append(scalar)

    for values in grouped.values():
        if not bool(torch.stack(values).all().item()):
            return False
    return True


def _read_int64_scalars_once(values: Sequence[torch.Tensor]) -> tuple[int, ...]:
    """Read ordered scalar tensors with one device-to-host transfer per device."""
    if not values:
        return ()

    grouped: dict[torch.device, list[tuple[int, torch.Tensor]]] = {}
    for index, value in enumerate(values):
        grouped.setdefault(value.device, []).append((index, value.detach()))

    result: list[Optional[int]] = [None] * len(values)
    for indexed_values in grouped.values():
        stacked = torch.stack([value for _, value in indexed_values])
        host_values = stacked.to(device="cpu", dtype=torch.int64).tolist()
        for (index, _), host_value in zip(indexed_values, host_values):
            result[index] = int(host_value)

    if any(value is None for value in result):
        raise RuntimeError("failed to materialize CPT scalar values")
    return tuple(int(value) for value in result if value is not None)


def _autocast_is_enabled(device_type: str) -> bool:
    """Query device autocast across the supported PyTorch 2.x API variants."""
    try:
        return bool(torch.is_autocast_enabled(device_type))
    except TypeError:
        if device_type == "cuda":
            return bool(torch.is_autocast_enabled())
        if device_type == "cpu":
            cpu_checker = getattr(torch, "is_autocast_cpu_enabled", None)
            return bool(cpu_checker()) if callable(cpu_checker) else False
        return False


def normalize_binary_mask(
    mask: torch.Tensor,
    expected_shape: tuple[int, ...],
    device: torch.device,
    *,
    name: str,
) -> torch.Tensor:
    """Validate a discrete ``{0, 1}`` control and return a bool tensor.

    Boolean masks are already canonical. Integer and real floating masks are
    accepted only when every entry is finite and exactly zero or one.  This
    intentionally rejects the ordinary ``Tensor.to(bool)`` behavior that
    would silently interpret NaN, negative values, or values such as ``2`` as
    active routing/reset controls.
    """
    if not isinstance(mask, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if tuple(mask.shape) != expected_shape:
        raise ValueError(
            f"{name} must have shape {expected_shape}, got "
            f"{tuple(mask.shape)}"
        )
    if mask.is_meta:
        raise TypeError(f"{name} must be materialized")
    if mask.dtype == torch.bool:
        return mask.to(device=device)
    if mask.dtype.is_complex:
        raise TypeError(f"{name} must use a boolean, integer, or real dtype")
    if not mask.dtype.is_floating_point:
        try:
            torch.iinfo(mask.dtype)
        except TypeError as error:
            raise TypeError(
                f"{name} must use a boolean, integer, or real dtype"
            ) from error

    canonical = mask.to(device=device)
    if canonical.dtype.is_floating_point and not torch.isfinite(canonical).all():
        raise ValueError(f"{name} must contain only finite binary values 0 or 1")
    if not ((canonical == 0) | (canonical == 1)).all():
        raise ValueError(f"{name} must contain only binary values 0 or 1")
    return canonical.to(dtype=torch.bool)


def normalize_sequence_ids(
    sequence_ids: torch.Tensor,
    batch_size: int,
    device: torch.device,
    *,
    name: str = "cpt_sequence_ids",
) -> torch.Tensor:
    """Return a private canonical snapshot of stable logical row identities."""
    if not isinstance(sequence_ids, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if tuple(sequence_ids.shape) != (batch_size,):
        raise ValueError(
            f"{name} must have shape ({batch_size},), got "
            f"{tuple(sequence_ids.shape)}"
        )
    if sequence_ids.is_meta:
        raise TypeError(f"{name} must be materialized")
    if (
        sequence_ids.dtype == torch.bool
        or sequence_ids.dtype.is_floating_point
        or sequence_ids.dtype.is_complex
    ):
        raise TypeError(f"{name} must use an integer dtype")
    try:
        dtype_info = torch.iinfo(sequence_ids.dtype)
    except TypeError as error:
        raise TypeError(f"{name} must use an integer dtype") from error

    if dtype_info.max > torch.iinfo(torch.int64).max:
        # PyTorch does not implement uint64 comparison kernels on either CPU
        # or CUDA. Sequence IDs are only one scalar per batch row, so perform
        # the representability check with Python integers before canonicalizing.
        python_values = sequence_ids.detach().cpu().tolist()
        if any(value > torch.iinfo(torch.int64).max for value in python_values):
            raise ValueError(f"{name} values must be representable as int64")
        canonical = torch.tensor(
            python_values,
            device=device,
            dtype=torch.int64,
        )
    else:
        canonical = sequence_ids.to(device=device, dtype=torch.int64)
    if torch.unique(canonical).numel() != batch_size:
        raise ValueError(f"{name} values must be unique within the batch")
    return canonical.detach().clone()


@dataclass
class CPTLayerProposal:
    """Detached optimizer-step statistics for one CPT router layer."""

    layer_index: int
    load_sum: torch.Tensor
    token_count: torch.Tensor
    state_version: torch.Tensor
    valid: torch.Tensor


@dataclass(frozen=True)
class CPTSequenceState:
    r"""Detached sequence-local CPT state for one router layer.

    The mathematical objects follow the PDF's column-vector convention:

    * ``state_s[b]`` stores :math:`S_b\in\mathbb R^{d_p\times K}`;
    * ``state_nu[b]`` stores :math:`\nu_b\in\mathbb R^{K\times1}` as ``[K]``;
    * ``initialized[b]`` records whether row ``b`` continues a logical
      sequence rather than starting a new one in the next forward call.
    * optional ``sequence_ids[b]`` is the stable logical identity of batch row
      ``b``. IDs are unique within a batch and protect continuation from
      silently inheriting state after dynamic row replacement or reordering.

    State tensors are always detached FP32 values. ``state_version`` is
    provenance metadata recording the router version that produced the state;
    it is deliberately not a compatibility lock because the PDF gives
    :math:`S,\nu` no optimizer-step subscript and permits their continuation
    across a successful optimizer step. ``layer_index`` and
    ``router_identity`` are transient ownership metadata: a continuation may
    return only to the exact router instance and layer that produced it. This
    prevents equal-shaped states from being silently swapped between layers or
    models. Replay safety comes from treating this object as immutable
    state-out and adopting it only after the caller's surrounding logical
    transaction succeeds.
    """

    state_s: torch.Tensor
    state_nu: torch.Tensor
    initialized: torch.Tensor
    state_version: torch.Tensor
    layer_index: int
    router_identity: object = field(repr=False, compare=False)
    sequence_ids: Optional[torch.Tensor] = field(
        default=None,
        repr=False,
        compare=False,
    )


@dataclass
class CPTTransaction:
    """All-layer proposal bundle owned by the caller, never by a module."""

    proposals: tuple[CPTLayerProposal, ...]
    closed: bool = False
    aborted: bool = False
    _prepared_prices: Optional[tuple[torch.Tensor, ...]] = field(
        default=None,
        repr=False,
    )
    _microbatch_records: dict[
        Hashable,
        tuple[CPTLayerProposal, ...],
    ] = field(default_factory=dict, repr=False)
    _prepared_base_prices: Optional[tuple[torch.Tensor, ...]] = field(
        default=None,
        repr=False,
    )
    _prepared_base_versions: Optional[tuple[int, ...]] = field(
        default=None,
        repr=False,
    )
    _prepared_global_loads: Optional[torch.Tensor] = field(
        default=None,
        repr=False,
    )
    _prepared_global_counts: Optional[torch.Tensor] = field(
        default=None,
        repr=False,
    )
    _prepared_microbatch_ids: Optional[tuple[Hashable, ...]] = field(
        default=None,
        repr=False,
    )

    def __post_init__(self) -> None:
        self.proposals = tuple(self.proposals)
        if not self._microbatch_records:
            self._microbatch_records = {("single", 0): self.proposals}

    @property
    def microbatch_ids(self) -> tuple[Hashable, ...]:
        """Stable identities whose raw statistics appear exactly once."""
        return tuple(self._microbatch_records)



@dataclass
class CPTRouterOutput:
    """CPT probabilities and diagnostics for route-valid tokens only."""

    probabilities: torch.Tensor
    flat_valid_indices: torch.Tensor
    q_probabilities: torch.Tensor
    expert_kernel: torch.Tensor
    proposal: CPTLayerProposal
    sequence_state: Optional[CPTSequenceState] = None


class CPTRouter(nn.Module):
    r"""Per-layer CPT probability router.

    Learnable objects use the PDF shapes

    * P in R^{d_p x d}
    * A in R^{d_p x K}
    * Theta_C in R^{K x N}

    while the persistent congestion price is the detached column vector
    lambda in R^{N x 1}. In code it is stored as a one-dimensional tensor
    with shape ``[N]``.
    """

    def __init__(self, config, layer_index: int):
        super().__init__()
        self.layer_index = int(layer_index)
        self.hidden_size = int(config.hidden_size)
        self.num_experts = int(config.num_local_experts)
        self.num_prototypes = int(config.cpt_num_prototypes)
        configured_projection_dim = getattr(
            config,
            "cpt_projection_dim",
            None,
        )
        if (
            isinstance(configured_projection_dim, bool)
            or not isinstance(configured_projection_dim, int)
            or configured_projection_dim < 1
        ):
            raise ValueError(
                "this strict-TeX Router implementation requires "
                "integer cpt_projection_dim >= 1"
            )
        self.projection_dim = configured_projection_dim
        if self.projection_dim == 1 and self.num_prototypes > 2:
            raise ValueError(
                "cpt_projection_dim=1 supports at most 2 distinct unit "
                "prototype anchors"
            )

        self.rho_beta = float(config.cpt_rho_beta_effective)
        if not 0.0 <= self.rho_beta < 1.0:
            raise ValueError(
                "cpt_rho_beta must remain in [0, 1) in FP32 router state"
            )
        self.state_nu_upper_bound = 1.0 / (1.0 - self.rho_beta)
        # ``nu`` is accumulated and persisted in FP32.  Build a tight numerical
        # envelope from the FP32 representation error of the exact bound plus a
        # fixed ULP budget.  Eight ULPs cover the multiply/add recurrence; the
        # row-mass audit adds one ULP per stored prototype component.  Unlike a
        # U^2 error formula, this remains a tiny relative envelope as rho -> 1.
        upper_fp32_tensor = torch.tensor(
            self.state_nu_upper_bound,
            dtype=torch.float32,
            device="cpu",
        )
        upper_fp32 = float(upper_fp32_tensor.item())
        upper_next_fp32 = float(
            torch.nextafter(
                upper_fp32_tensor,
                torch.tensor(
                    float("inf"),
                    dtype=torch.float32,
                    device="cpu",
                ),
            ).item()
        )
        upper_ulp = upper_next_fp32 - upper_fp32
        upper_representation_error = abs(
            upper_fp32 - self.state_nu_upper_bound
        )
        self.state_nu_element_tolerance = max(
            1e-6,
            upper_representation_error + 8.0 * upper_ulp,
        )
        self.state_nu_mass_tolerance = max(
            1e-6,
            upper_representation_error
            + (8.0 + self.num_prototypes) * upper_ulp,
        )
        # Backward-compatible diagnostic name used by existing callers/tests.
        self.state_nu_tolerance = self.state_nu_mass_tolerance
        self.beta_max = float(config.cpt_beta_max)
        self.kappa_beta = 1.0 / (
            self.num_prototypes * (1.0 - self.rho_beta)
        )
        self.lambda_sa = float(config.cpt_lambda_sa)
        self.projection_temperature = float(
            config.cpt_projection_temperature
        )
        self.expert_temperature = float(config.cpt_expert_temperature)
        self.state_step_size = float(config.cpt_state_step_size)
        self.state_radius = float(config.cpt_state_radius)
        self.eps_z = float(config.cpt_eps_z)
        self.eps_m = float(config.cpt_eps_m)
        self.eps_init = float(config.cpt_eps_init)
        self.energy_init_scale = float(config.cpt_energy_init_scale)
        self.capacity_factor = float(config.cpt_capacity_factor)
        self.price_learning_rate = float(
            config.cpt_price_learning_rate
        )
        self.init_seed = int(config.cpt_init_seed)
        configured_router_version = getattr(config, "cpt_router_version", None)
        if (
            isinstance(configured_router_version, bool)
            or not isinstance(configured_router_version, int)
            or configured_router_version != CPT_ROUTER_ALGORITHM_VERSION
        ):
            raise ValueError(
                "this strict-TeX Router implementation requires "
                f"cpt_router_version={CPT_ROUTER_ALGORITHM_VERSION}"
            )
        self.expected_router_algorithm_version = (
            CPT_ROUTER_ALGORITHM_VERSION
        )

        # Raw parameters are used instead of nn.Linear so model-wide generic
        # Linear initialization cannot overwrite the PDF-specific scheme. The
        # explicit dtype also protects construction under a temporary HF
        # BF16/FP16 default-dtype context.
        self.projection = nn.Parameter(
            torch.empty(
                self.projection_dim,
                self.hidden_size,
                dtype=torch.float32,
            )
        )
        self.anchors = nn.Parameter(
            torch.empty(
                self.projection_dim,
                self.num_prototypes,
                dtype=torch.float32,
            )
        )
        self.energy = nn.Parameter(
            torch.empty(
                self.num_prototypes,
                self.num_experts,
                dtype=torch.float32,
            )
        )

        self.register_buffer(
            "congestion_price",
            torch.zeros(self.num_experts, dtype=torch.float32),
            persistent=True,
        )
        self.register_buffer(
            "state_version",
            torch.zeros((), dtype=torch.int64),
            persistent=True,
        )
        self.register_buffer(
            "router_algorithm_version",
            torch.tensor(
                self.expected_router_algorithm_version,
                dtype=torch.int64,
            ),
            persistent=True,
        )

        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Initialize P, A and Theta_C using the PDF's explicit scheme."""
        seed = self.init_seed + 104_729 * self.layer_index
        generator = torch.Generator(device="cpu").manual_seed(seed)

        with torch.no_grad():
            if self.projection_dim <= self.hidden_size:
                # Q has orthonormal columns, so P=Q^T gives P P^T=I.
                raw = torch.randn(
                    self.hidden_size,
                    self.projection_dim,
                    generator=generator,
                    dtype=torch.float32,
                )
                q_matrix, _ = torch.linalg.qr(raw, mode="reduced")
                projection = q_matrix.T.contiguous()
            else:
                std = (2.0 / (self.hidden_size + self.projection_dim)) ** 0.5
                projection = torch.randn(
                    self.projection_dim,
                    self.hidden_size,
                    generator=generator,
                    dtype=torch.float32,
                ) * std
            self.projection.copy_(projection)

            if self.num_prototypes <= self.projection_dim:
                raw_anchors = torch.randn(
                    self.projection_dim,
                    self.num_prototypes,
                    generator=generator,
                    dtype=torch.float32,
                )
                anchors, _ = torch.linalg.qr(raw_anchors, mode="reduced")
            else:
                # Random sphere directions are appropriate when K>d_p, but
                # finite precision must still satisfy the PDF's requirement
                # that no two anchor columns have the same direction.  Retry
                # deterministically from the layer seed if a zero/near-equal
                # candidate set is encountered. Opposite unit vectors remain
                # distinct directions and are intentionally permitted.
                direction_tolerance = 8.0 * torch.finfo(torch.float32).eps
                anchors = None
                for _ in range(64):
                    candidate = torch.randn(
                        self.projection_dim,
                        self.num_prototypes,
                        generator=generator,
                        dtype=torch.float32,
                    )
                    norms = candidate.norm(dim=0, keepdim=True)
                    if (
                        not torch.isfinite(candidate).all()
                        or not torch.isfinite(norms).all()
                        or (norms <= self.eps_init).any()
                    ):
                        continue
                    candidate = candidate / norms
                    gram = candidate.T @ candidate
                    duplicate_mask = ~torch.eye(
                        self.num_prototypes,
                        dtype=torch.bool,
                        device=gram.device,
                    )
                    if not (
                        gram[duplicate_mask] > 1.0 - direction_tolerance
                    ).any():
                        anchors = candidate
                        break
                if anchors is None:
                    raise RuntimeError(
                        "failed to initialize distinct CPT anchor directions "
                        "for cpt_num_prototypes > cpt_projection_dim"
                    )
            self.anchors.copy_(anchors)

            # H_K Xi_C H_N is implemented by explicit double centering.
            # The normalization is over the maximum absolute matrix entry.
            initial_energy = None
            for _ in range(8):
                xi = torch.randn(
                    self.num_prototypes,
                    self.num_experts,
                    generator=generator,
                    dtype=torch.float32,
                )
                centered = (
                    xi
                    - xi.mean(dim=0, keepdim=True)
                    - xi.mean(dim=1, keepdim=True)
                    + xi.mean()
                )
                max_abs = centered.abs().amax()
                if not torch.isfinite(max_abs) or max_abs <= self.eps_init:
                    continue
                normalized = centered / max(max_abs.item(), self.eps_init)
                candidate = (self.energy_init_scale * normalized).float()
                if self._valid_initial_energy(candidate):
                    initial_energy = candidate
                    break
            if initial_energy is None:
                raise RuntimeError("failed to initialize non-degenerate CPT energy")

            self.energy.copy_(initial_energy)
            self.congestion_price.zero_()
            self.state_version.zero_()
            self.router_algorithm_version.fill_(
                self.expected_router_algorithm_version
            )

    @staticmethod
    def _valid_initial_energy(candidate: torch.Tensor) -> bool:
        """Check the PDF's finite-precision non-degeneracy requirement."""
        if candidate.dtype != torch.float32 or not torch.isfinite(candidate).all():
            return False
        if candidate.numel() == 0 or candidate.abs().amax() == 0:
            return False
        if candidate.shape[0] > 1 and torch.equal(
            candidate[1:],
            candidate[:1].expand_as(candidate[1:]),
        ):
            return False
        return True

    @staticmethod
    def _apply_tensor_preserving_fp32(
        tensor: torch.Tensor,
        fn,
    ) -> torch.Tensor:
        """Apply a module conversion without quantizing protected FP32 data."""
        with torch.no_grad():
            applied = fn(tensor)
        if applied.dtype == torch.float32:
            return applied
        if tensor.is_meta:
            return torch.empty_like(applied, dtype=torch.float32)
        return tensor.to(
            device=applied.device,
            dtype=torch.float32,
            copy=True,
        )

    @classmethod
    def _apply_parameter_preserving_fp32(
        cls,
        parameter: nn.Parameter,
        fn,
    ) -> nn.Parameter:
        """Apply ``fn`` while retaining the Parameter object's identity."""
        original_data = parameter.data
        applied_data = cls._apply_tensor_preserving_fp32(original_data, fn)
        original_grad = parameter.grad

        if applied_data is not original_data:
            parameter.grad = None
            replacement = nn.Parameter(
                applied_data,
                requires_grad=parameter.requires_grad,
            )
            swap_tensors = getattr(torch.utils, "swap_tensors", None)
            if callable(swap_tensors):
                swap_tensors(parameter, replacement)
            else:
                try:
                    parameter.data = applied_data
                except RuntimeError:
                    # Older PyTorch releases cannot install a materialized
                    # tensor into a meta Parameter in place. This fallback is
                    # safe only before optimizer/DDP construction, matching
                    # ordinary meta materialization requirements.
                    parameter = replacement

        if original_grad is not None:
            applied_grad = cls._apply_tensor_preserving_fp32(
                original_grad,
                fn,
            )
            parameter.grad = applied_grad.requires_grad_(
                original_grad.requires_grad
            )
        return parameter

    def _apply(self, fn, recurse: bool = True):
        """Follow parent device moves while preserving all CPT state in FP32.

        Converting an already-quantized BF16 result back to FP32 would preserve
        only the dtype label, not the original low bits. Protected tensors are
        therefore hidden from ``Module._apply`` and moved directly from their
        original FP32 values. Parameter identity and optimizer references stay
        unchanged; existing gradients follow the same rule.
        """
        protected_parameters = {
            name: self._parameters[name]
            for name in FP32_ROUTER_PARAMETER_NAMES
        }
        protected_price = self._buffers["congestion_price"]
        protected_algorithm_version = self._buffers[
            "router_algorithm_version"
        ]
        if (
            protected_algorithm_version is not None
            and not protected_algorithm_version.is_meta
            and (
                protected_algorithm_version.ndim != 0
                or protected_algorithm_version.dtype != torch.int64
                or int(protected_algorithm_version.item())
                != self.expected_router_algorithm_version
            )
        ):
            raise RuntimeError(
                "CPT router_algorithm_version is invalid before module "
                "device/dtype conversion"
            )
        for name in protected_parameters:
            self._parameters[name] = None
        self._buffers["congestion_price"] = None
        self._buffers["router_algorithm_version"] = None

        try:
            result = super()._apply(fn, recurse=recurse)
        except BaseException:
            self._parameters.update(protected_parameters)
            self._buffers["congestion_price"] = protected_price
            self._buffers[
                "router_algorithm_version"
            ] = protected_algorithm_version
            raise

        try:
            for name, parameter in protected_parameters.items():
                if parameter is None:
                    raise RuntimeError(f"missing protected CPT parameter {name}")
                parameter = self._apply_parameter_preserving_fp32(
                    parameter,
                    fn,
                )
                self._parameters[name] = parameter
            if protected_price is None:
                raise RuntimeError("missing protected CPT congestion_price")
            self._buffers["congestion_price"] = (
                self._apply_tensor_preserving_fp32(protected_price, fn)
            )
            if protected_algorithm_version is None:
                raise RuntimeError(
                    "missing protected CPT router_algorithm_version"
                )
            with torch.no_grad():
                applied_algorithm_version = fn(
                    protected_algorithm_version
                )
                if not applied_algorithm_version.is_meta:
                    applied_algorithm_version.fill_(
                        self.expected_router_algorithm_version
                    )
            self._buffers[
                "router_algorithm_version"
            ] = applied_algorithm_version
        except BaseException:
            self._parameters.update(protected_parameters)
            self._buffers["congestion_price"] = protected_price
            self._buffers[
                "router_algorithm_version"
            ] = protected_algorithm_version
            raise
        return result

    @staticmethod
    def _normalize_columns(matrix: torch.Tensor, eps: float) -> torch.Tensor:
        return matrix / matrix.norm(
            dim=-2,
            keepdim=True,
        ).clamp_min(eps)

    @staticmethod
    def _first_valid_mask(route_valid_mask: torch.Tensor) -> torch.Tensor:
        return route_valid_mask & (
            route_valid_mask.to(torch.int64).cumsum(dim=1) == 1
        )

    @staticmethod
    def _all_routes_valid(route_valid_mask: torch.Tensor) -> bool:
        """Whether the common dense training path can skip row gather/scatter."""
        return bool(route_valid_mask.all().item())

    def _new_sequence_state(
        self,
        batch_size: int,
        device: torch.device,
        sequence_ids: Optional[torch.Tensor] = None,
    ) -> CPTSequenceState:
        return self._make_sequence_state(
            state_s=torch.zeros(
                batch_size,
                self.projection_dim,
                self.num_prototypes,
                device=device,
                dtype=torch.float32,
            ),
            state_nu=torch.zeros(
                batch_size,
                self.num_prototypes,
                device=device,
                dtype=torch.float32,
            ),
            initialized=torch.zeros(
                batch_size,
                device=device,
                dtype=torch.bool,
            ),
            state_version=self.state_version.detach().clone(),
            sequence_ids=sequence_ids,
        )

    def _make_sequence_state(
        self,
        *,
        state_s: torch.Tensor,
        state_nu: torch.Tensor,
        initialized: torch.Tensor,
        state_version: torch.Tensor,
        sequence_ids: Optional[torch.Tensor] = None,
    ) -> CPTSequenceState:
        """Bind detached state tensors to this exact layer/router instance."""
        return CPTSequenceState(
            state_s=state_s.detach().clone(),
            state_nu=state_nu.detach().clone(),
            initialized=initialized.detach().clone(),
            state_version=state_version.detach().clone(),
            layer_index=self.layer_index,
            # A weak reference survives deepcopy(state) while still pointing
            # to the original router; deepcopy(model) creates a new router and
            # therefore cannot accidentally accept the original state.
            router_identity=weakref.ref(self),
            sequence_ids=(
                None
                if sequence_ids is None
                else sequence_ids.detach().clone()
            ),
        )

    def _validate_sequence_state(
        self,
        sequence_state: CPTSequenceState,
        batch_size: int,
        device: torch.device,
    ) -> CPTSequenceState:
        if not isinstance(sequence_state, CPTSequenceState):
            raise TypeError("sequence_state must be a CPTSequenceState")

        if (
            isinstance(sequence_state.layer_index, bool)
            or not isinstance(sequence_state.layer_index, int)
        ):
            raise TypeError("sequence_state.layer_index must be an integer")
        if sequence_state.layer_index != self.layer_index:
            raise ValueError(
                "sequence_state layer identity does not match this CPT router: "
                f"got layer {sequence_state.layer_index}, expected "
                f"{self.layer_index}"
            )
        if (
            not isinstance(sequence_state.router_identity, weakref.ReferenceType)
            or sequence_state.router_identity() is not self
        ):
            raise ValueError(
                "sequence_state belongs to a different CPT router instance"
            )

        expected_s = (
            batch_size,
            self.projection_dim,
            self.num_prototypes,
        )
        expected_nu = (batch_size, self.num_prototypes)
        expected_initialized = (batch_size,)
        values = (
            ("state_s", sequence_state.state_s, expected_s, torch.float32),
            ("state_nu", sequence_state.state_nu, expected_nu, torch.float32),
            (
                "initialized",
                sequence_state.initialized,
                expected_initialized,
                torch.bool,
            ),
        )
        for name, value, expected_shape, expected_dtype in values:
            if not isinstance(value, torch.Tensor):
                raise TypeError(f"sequence_state.{name} must be a tensor")
            if tuple(value.shape) != expected_shape:
                raise ValueError(
                    f"sequence_state.{name} must have shape {expected_shape}, "
                    f"got {tuple(value.shape)}"
                )
            if value.device != device:
                raise ValueError(
                    f"sequence_state.{name} is on {value.device}, expected {device}"
                )
            if value.dtype != expected_dtype:
                raise ValueError(
                    f"sequence_state.{name} must have dtype {expected_dtype}, "
                    f"got {value.dtype}"
                )
            if value.requires_grad:
                raise ValueError(
                    f"sequence_state.{name} must be detached from autograd"
                )

        state_version = sequence_state.state_version
        if not isinstance(state_version, torch.Tensor):
            raise TypeError("sequence_state.state_version must be a tensor")
        if state_version.ndim != 0:
            raise ValueError("sequence_state.state_version must be a scalar")
        if state_version.device != device:
            raise ValueError(
                "sequence_state.state_version is on "
                f"{state_version.device}, expected {device}"
            )
        if state_version.dtype != torch.int64:
            raise ValueError(
                "sequence_state.state_version must have dtype torch.int64"
            )
        if state_version.requires_grad:
            raise ValueError("sequence_state.state_version must be detached")
        supplied_version = int(state_version.item())
        if supplied_version < 0:
            raise ValueError(
                "sequence_state.state_version provenance must be non-negative"
            )
        active_version = int(self.state_version.item())
        if supplied_version > active_version:
            raise ValueError(
                "sequence_state.state_version provenance cannot be newer than "
                "the active CPT router state_version"
            )

        sequence_ids = sequence_state.sequence_ids
        if sequence_ids is not None:
            sequence_ids = normalize_sequence_ids(
                sequence_ids,
                batch_size,
                device,
                name="sequence_state.sequence_ids",
            )

        state_s = sequence_state.state_s
        state_nu = sequence_state.state_nu
        if not torch.isfinite(state_s).all():
            raise ValueError("sequence_state.state_s must be finite")
        if not torch.isfinite(state_nu).all():
            raise ValueError("sequence_state.state_nu must be finite")
        if (state_nu < 0).any():
            raise ValueError("sequence_state.state_nu must be non-negative")
        if (
            state_nu
            > self.state_nu_upper_bound + self.state_nu_element_tolerance
        ).any():
            raise ValueError(
                "sequence_state.state_nu exceeds the theoretical upper "
                "bound 1 / (1 - cpt_rho_beta)"
            )
        state_nu_mass = state_nu.sum(dim=-1, dtype=torch.float64)
        if (
            state_nu_mass
            > self.state_nu_upper_bound + self.state_nu_mass_tolerance
        ).any():
            raise ValueError(
                "sequence_state.state_nu row mass exceeds the theoretical "
                "upper bound 1 / (1 - cpt_rho_beta)"
            )
        state_norms = state_s.norm(dim=1)
        radius_tolerance = 1e-5 * max(1.0, self.state_radius)
        if (state_norms > self.state_radius + radius_tolerance).any():
            raise ValueError(
                "sequence_state.state_s violates the CPT state radius"
            )

        uninitialized = ~sequence_state.initialized
        if uninitialized.any():
            if state_s[uninitialized].abs().amax() != 0:
                raise ValueError(
                    "uninitialized sequence rows must have zero state_s"
                )
            if state_nu[uninitialized].abs().amax() != 0:
                raise ValueError(
                    "uninitialized sequence rows must have zero state_nu"
                )

        return self._make_sequence_state(
            state_s=state_s,
            state_nu=state_nu,
            initialized=sequence_state.initialized,
            state_version=state_version,
            sequence_ids=sequence_ids,
        )

    def _resolve_sequence_ids(
        self,
        sequence_state: CPTSequenceState,
        sequence_ids: Optional[torch.Tensor],
        route_valid_mask: torch.Tensor,
        reset_mask: torch.Tensor,
    ) -> Optional[torch.Tensor]:
        """Validate row identity continuity and return state-out identities."""
        previous_ids = sequence_state.sequence_ids
        if previous_ids is not None and sequence_ids is None:
            raise ValueError(
                "cpt_sequence_ids is required when continuation state carries "
                "logical sequence identities"
            )
        if sequence_ids is None:
            return None
        if previous_ids is None:
            # Legacy live states carried no row identity metadata.  The first
            # identity-aware continuation binds them without changing the
            # behavior of callers that continue to omit IDs.
            return sequence_ids

        mismatched = previous_ids != sequence_ids
        if not mismatched.any():
            return sequence_ids

        first_valid = self._first_valid_mask(route_valid_mask)
        explicitly_replaced = (first_valid & reset_mask).any(dim=1)
        allowed = ~sequence_state.initialized | explicitly_replaced
        rejected = mismatched & ~allowed
        if rejected.any():
            rows = rejected.nonzero(as_tuple=False).flatten().tolist()
            raise ValueError(
                "cpt_sequence_ids do not match continuation state for batch "
                f"rows {rows}; explicitly reset each changed row at its "
                "first valid token or reorder the continuation state"
            )
        return sequence_ids

    def _validate_masks(
        self,
        hidden_states: torch.Tensor,
        route_valid_mask: Optional[torch.Tensor],
        reset_mask: Optional[torch.Tensor],
        continuation_initialized: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size, seq_len, _ = hidden_states.shape
        expected_shape = (batch_size, seq_len)
        if route_valid_mask is None:
            route_valid_mask = torch.ones(
                expected_shape,
                device=hidden_states.device,
                dtype=torch.bool,
            )
        else:
            route_valid_mask = normalize_binary_mask(
                route_valid_mask,
                expected_shape,
                hidden_states.device,
                name="route_valid_mask",
            )

        first_valid = self._first_valid_mask(route_valid_mask)
        required_reset = first_valid & ~continuation_initialized[:, None]
        if reset_mask is None:
            reset_mask = required_reset
        else:
            reset_mask = normalize_binary_mask(
                reset_mask,
                expected_shape,
                hidden_states.device,
                name="reset_mask",
            )
            if torch.any(reset_mask & ~route_valid_mask):
                raise ValueError("reset_mask cannot mark an invalid route position")
            # Backward-compatible enforcement: callers may omit an explicit
            # first-token reset for a new row, but the router never permits a
            # new logical sequence to inherit an uninitialized state.
            reset_mask = reset_mask | required_reset
        return route_valid_mask, reset_mask

    def _expert_kernel(self) -> torch.Tensor:
        # C=Theta_C H_N: multiplying by H_N is row centering.
        centered_energy = self.energy.float()
        centered_energy = centered_energy - centered_energy.mean(
            dim=-1,
            keepdim=True,
        )
        effective_energy = centered_energy - self.congestion_price.detach()[
            None, :
        ]
        return torch.softmax(
            effective_energy / self.expert_temperature,
            dim=-1,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        route_valid_mask: Optional[torch.Tensor] = None,
        reset_mask: Optional[torch.Tensor] = None,
        sequence_state: Optional[CPTSequenceState] = None,
        cpt_sequence_ids: Optional[torch.Tensor] = None,
    ) -> CPTRouterOutput:
        """Compute ``Pi.T=Q.T @ B`` for valid tokens without side effects."""
        if hidden_states.ndim != 3:
            raise ValueError("hidden_states must have shape [B, S, d]")
        batch_size, seq_len, hidden_size = hidden_states.shape
        if hidden_size != self.hidden_size:
            raise ValueError(
                f"expected hidden size {self.hidden_size}, got {hidden_size}"
            )

        # ``None`` is the canonical trusted provenance for the ordinary dense
        # model path: the caller did not supply padding/packing controls, so all
        # routes are valid and the only implicit reset can occur at position 0.
        # Explicit masks still pass through the full value validation below.
        trusted_dense_route = route_valid_mask is None
        default_reset_schedule = reset_mask is None

        canonical_sequence_ids = (
            None
            if cpt_sequence_ids is None
            else normalize_sequence_ids(
                cpt_sequence_ids,
                batch_size,
                hidden_states.device,
            )
        )

        if sequence_state is None:
            active_sequence_state = self._new_sequence_state(
                batch_size,
                hidden_states.device,
            )
        else:
            active_sequence_state = self._validate_sequence_state(
                sequence_state,
                batch_size,
                hidden_states.device,
            )

        route_valid_mask, reset_mask = self._validate_masks(
            hidden_states,
            route_valid_mask,
            reset_mask,
            active_sequence_state.initialized,
        )
        output_sequence_ids = self._resolve_sequence_ids(
            active_sequence_state,
            canonical_sequence_ids,
            route_valid_mask,
            reset_mask,
        )

        # Large projection GEMM may use the model compute dtype. The trainable
        # projection itself stays FP32; ambient autocast supplies the efficient
        # cached compute cast during normal training. A direct BF16/FP16 forward
        # without autocast needs an explicit differentiable dtype bridge because
        # linear does not accept mixed input/weight dtypes. All following
        # normalization, softmax and state arithmetic are explicitly FP32.
        # Invalid positions are replaced before the differentiable projection,
        # not after softmax.  This implements the PDF's v=0 direct bypass and
        # prevents NaN/Inf padding values from creating 0*NaN router gradients.
        routing_hidden_states = torch.where(
            route_valid_mask[:, :, None],
            hidden_states,
            torch.zeros_like(hidden_states),
        )
        projection_for_compute = self.projection
        if (
            routing_hidden_states.dtype != projection_for_compute.dtype
            and not _autocast_is_enabled(hidden_states.device.type)
        ):
            projection_for_compute = projection_for_compute.to(
                dtype=routing_hidden_states.dtype
            )
        projected = F.linear(
            routing_hidden_states,
            projection_for_compute,
        ).float()
        z_rows = projected / projected.norm(
            dim=-1,
            keepdim=True,
        ).clamp_min(self.eps_z)

        state_s = active_sequence_state.state_s
        state_nu = active_sequence_state.state_nu
        initialized = active_sequence_state.initialized
        anchors = self.anchors.float()
        q_steps: list[torch.Tensor] = []

        # The default pretraining path contains no padding, so every batch row
        # is route-valid at every position.  Detect that common case once and
        # avoid per-position ``nonzero``/gather/scatter operations.  The causal
        # recurrence and its stop-gradient boundary remain identical; masked
        # evaluation/packing continues to use the general sparse-row path.
        dense_route_valid = (
            True
            if trusted_dense_route
            else self._all_routes_valid(route_valid_mask)
        )
        dense_first_position_reset = (
            dense_route_valid and default_reset_schedule
        )
        dense_reset_positions: Optional[list[bool]] = None
        if dense_route_valid and not dense_first_position_reset:
            # Reset is normally needed only at position zero.  Materialize one
            # compact position-level schedule so the dense path does not launch
            # two no-op ``where`` kernels at every later token.  Arbitrary
            # packed/reset positions are still honored exactly.
            dense_reset_positions = (
                reset_mask.any(dim=0).detach().to(device="cpu").tolist()
            )

        for position in range(seq_len):
            valid = route_valid_mask[:, position]
            reset = reset_mask[:, position]

            apply_reset = not dense_route_valid
            if dense_route_valid:
                apply_reset = (
                    position == 0
                    if dense_first_position_reset
                    else bool(dense_reset_positions[position])
                )
            if apply_reset:
                with torch.no_grad():
                    state_s = torch.where(
                        reset[:, None, None],
                        torch.zeros_like(state_s),
                        state_s,
                    )
                    state_nu = torch.where(
                        reset[:, None],
                        torch.zeros_like(state_nu),
                        state_nu,
                    )

            valid_rows: Optional[torch.Tensor]
            if dense_route_valid:
                valid_rows = None
                state_s_valid = state_s
                state_nu_valid = state_nu
                z_valid = z_rows[:, position, :]
            else:
                valid_rows = valid.nonzero(as_tuple=False).flatten()
                if valid_rows.numel() > 0:
                    state_s_valid = state_s.index_select(0, valid_rows)
                    state_nu_valid = state_nu.index_select(0, valid_rows)
                    z_valid = z_rows[:, position, :].index_select(0, valid_rows)

            if dense_route_valid or (
                valid_rows is not None and valid_rows.numel() > 0
            ):
                beta_valid = self.beta_max * state_nu_valid / (
                    state_nu_valid + self.kappa_beta
                )
                mixed_valid = (
                    anchors[None, :, :]
                    * (1.0 - beta_valid.detach()[:, None, :])
                    + state_s_valid.detach()
                    * beta_valid.detach()[:, None, :]
                )
                prototypes_valid = self._normalize_columns(
                    mixed_valid,
                    self.eps_m,
                )
                # The prototype dot products feed a required FP32 softmax.
                # CPU autocast otherwise lowers einsum and softmax to BF16,
                # which can violate the exact probability-mass transaction
                # check even though q is later copied into an FP32 tensor.
                with torch.autocast(
                    device_type=hidden_states.device.type,
                    enabled=False,
                ):
                    logits_valid = torch.einsum(
                        "bd,bdk->bk",
                        z_valid.float(),
                        prototypes_valid.float(),
                    ) / self.projection_temperature
                    q_valid_at_position = torch.softmax(
                        logits_valid,
                        dim=-1,
                        dtype=torch.float32,
                    )
                if dense_route_valid:
                    q_current = q_valid_at_position
                else:
                    q_current = torch.zeros(
                        batch_size,
                        self.num_prototypes,
                        device=hidden_states.device,
                        dtype=torch.float32,
                    ).index_copy(0, valid_rows, q_valid_at_position)
            else:
                q_current = torch.zeros(
                    batch_size,
                    self.num_prototypes,
                    device=hidden_states.device,
                    dtype=torch.float32,
                )
            q_steps.append(q_current)

            # State writes use stop-gradient and never become module state.
            with torch.no_grad():
                if dense_route_valid:
                    q_state = q_current.detach()
                    z_state = z_rows[:, position, :].detach()
                    previous_s = state_s
                    previous_nu = state_nu
                elif valid_rows is not None and valid_rows.numel() > 0:
                    q_state = q_current.index_select(0, valid_rows).detach()
                    z_state = z_valid.detach()
                    previous_s = state_s.index_select(0, valid_rows)
                    previous_nu = state_nu.index_select(0, valid_rows)
                else:
                    q_state = None

                if q_state is not None:
                    anchor_state = anchors.detach()
                    gradient_s = (
                        (previous_s - z_state[:, :, None])
                        * q_state[:, None, :]
                        + self.lambda_sa
                        * (previous_s - anchor_state[None, :, :])
                    )
                    candidate_s = (
                        previous_s - self.state_step_size * gradient_s
                    )
                    scale = (
                        candidate_s.norm(dim=1, keepdim=True)
                        .div(self.state_radius)
                        .clamp_min(1.0)
                    )
                    candidate_s = candidate_s / scale
                    candidate_nu = self.rho_beta * previous_nu + q_state
                    if dense_route_valid:
                        state_s = candidate_s
                        state_nu = candidate_nu
                    else:
                        state_s = state_s.index_copy(
                            0,
                            valid_rows,
                            candidate_s,
                        )
                        state_nu = state_nu.index_copy(
                            0,
                            valid_rows,
                            candidate_nu,
                        )
                if not dense_route_valid:
                    initialized = initialized | valid

        if dense_route_valid and seq_len > 0:
            with torch.no_grad():
                initialized = torch.ones_like(initialized)

        if q_steps:
            q_all_rows = torch.stack(q_steps, dim=1)
        else:
            q_all_rows = torch.zeros(
                batch_size,
                0,
                self.num_prototypes,
                device=hidden_states.device,
                dtype=torch.float32,
            )

        q_flat_rows = q_all_rows.reshape(-1, self.num_prototypes)
        if dense_route_valid:
            q_valid_rows = q_flat_rows
            flat_valid_indices = torch.arange(
                batch_size * seq_len,
                device=hidden_states.device,
                dtype=torch.long,
            )
        else:
            flat_valid_indices = route_valid_mask.reshape(-1).nonzero(
                as_tuple=False
            ).flatten()
            q_valid_rows = q_flat_rows.index_select(0, flat_valid_indices)
        expert_kernel = self._expert_kernel()

        # Token-major code equivalent of the column-vector equation
        # Pi=B^T Q: Pi.T=Q.T @ B.
        # Keep the small K-by-N probability composition in FP32. Under CUDA
        # autocast, ``matmul`` would otherwise return BF16 even though both
        # operands were promoted to FP32 above, which weakens the probability
        # simplex invariant and can make its validation dtype-dependent.
        with torch.autocast(device_type=hidden_states.device.type, enabled=False):
            probabilities = q_valid_rows.float() @ expert_kernel.float()
        token_count = torch.tensor(
            flat_valid_indices.numel(),
            device=hidden_states.device,
            dtype=torch.int64,
        )
        load_sum = probabilities.detach().float().sum(dim=0)

        sequence_state_out = self._make_sequence_state(
            state_s=state_s,
            state_nu=state_nu,
            initialized=initialized,
            state_version=self.state_version,
            sequence_ids=output_sequence_ids,
        )

        with torch.no_grad():
            finite = (
                torch.isfinite(q_valid_rows).all()
                & torch.isfinite(expert_kernel).all()
                & torch.isfinite(probabilities).all()
                & torch.isfinite(sequence_state_out.state_s).all()
                & torch.isfinite(sequence_state_out.state_nu).all()
                & (sequence_state_out.state_nu >= 0).all()
                & (
                    sequence_state_out.state_nu
                    <= self.state_nu_upper_bound
                    + self.state_nu_element_tolerance
                ).all()
                & (
                    sequence_state_out.state_nu.sum(
                        dim=-1,
                        dtype=torch.float64,
                    )
                    <= self.state_nu_upper_bound
                    + self.state_nu_mass_tolerance
                ).all()
                & (
                    sequence_state_out.state_s.norm(dim=1)
                    <= self.state_radius + 1e-5 * max(1.0, self.state_radius)
                ).all()
            )
            if flat_valid_indices.numel() > 0:
                q_sums = q_valid_rows.float().sum(dim=-1)
                probability_sums = probabilities.float().sum(dim=-1)
                finite = (
                    finite
                    & (q_valid_rows >= 0).all()
                    & (expert_kernel >= 0).all()
                    & (probabilities >= 0).all()
                    & torch.isclose(
                        q_sums,
                        torch.ones_like(q_sums),
                        atol=1e-5,
                        rtol=1e-5,
                    ).all()
                    & torch.isclose(
                        probability_sums,
                        torch.ones_like(probability_sums),
                        atol=1e-5,
                        rtol=1e-5,
                    ).all()
                )

        proposal = CPTLayerProposal(
            layer_index=self.layer_index,
            load_sum=load_sum,
            token_count=token_count.detach(),
            state_version=self.state_version.detach().clone(),
            valid=torch.as_tensor(
                finite,
                device=hidden_states.device,
                dtype=torch.bool,
            ).detach(),
        )
        return CPTRouterOutput(
            probabilities=probabilities,
            flat_valid_indices=flat_valid_indices,
            q_probabilities=q_valid_rows,
            expert_kernel=expert_kernel,
            proposal=proposal,
            sequence_state=sequence_state_out,
        )

    def validate_persistent_invariants(
        self,
        *,
        layer_index: Optional[int] = None,
    ) -> None:
        """Reject a loaded CPT state that violates persistent invariants."""
        label = self.layer_index if layer_index is None else int(layer_index)
        for name in FP32_ROUTER_PARAMETER_NAMES:
            parameter = getattr(self, name)
            value = parameter.detach()
            if value.is_meta:
                raise RuntimeError(
                    f"layer {label} CPT {name} is still on the meta device; "
                    "persistent invariants can only be validated after the "
                    "checkpoint is fully materialized"
                )
            if value.dtype != torch.float32:
                raise RuntimeError(
                    f"layer {label} CPT {name} must remain FP32"
                )
            if parameter.grad is not None and parameter.grad.dtype != torch.float32:
                raise RuntimeError(
                    f"layer {label} CPT {name} gradient must remain FP32"
                )
            if not torch.isfinite(value).all():
                raise RuntimeError(
                    f"layer {label} CPT {name} is non-finite after loading"
                )

        price = self.congestion_price.detach()
        if price.is_meta:
            raise RuntimeError(
                f"layer {label} CPT congestion_price is still on the meta "
                "device after checkpoint loading"
            )
        if price.dtype != torch.float32:
            raise RuntimeError(
                f"layer {label} CPT congestion_price must remain FP32"
            )
        if not torch.isfinite(price).all() or (price < 0).any():
            raise RuntimeError(
                f"layer {label} CPT congestion_price must be finite and "
                "non-negative"
            )

        version = self.state_version.detach()
        if version.is_meta:
            raise RuntimeError(
                f"layer {label} CPT state_version is still on the meta device "
                "after checkpoint loading"
            )
        if version.ndim != 0 or version.dtype != torch.int64:
            raise RuntimeError(
                f"layer {label} CPT state_version must be an int64 scalar"
            )
        if int(version.item()) < 0:
            raise RuntimeError(
                f"layer {label} CPT state_version must be non-negative"
            )

        algorithm_version = self.router_algorithm_version.detach()
        if algorithm_version.is_meta:
            raise RuntimeError(
                f"layer {label} CPT router_algorithm_version is still on "
                "the meta device after checkpoint loading"
            )
        if algorithm_version.ndim != 0 or algorithm_version.dtype != torch.int64:
            raise RuntimeError(
                f"layer {label} CPT router_algorithm_version must be an "
                "int64 scalar"
            )
        loaded_algorithm_version = int(algorithm_version.item())
        if loaded_algorithm_version != self.expected_router_algorithm_version:
            raise RuntimeError(
                f"layer {label} CPT router_algorithm_version="
                f"{loaded_algorithm_version} is incompatible with this "
                f"Router implementation, which requires "
                f"{self.expected_router_algorithm_version}"
            )

        anchor_norms = self.anchors.detach().float().norm(dim=0)
        if not torch.isfinite(anchor_norms).all():
            raise RuntimeError(
                f"layer {label} CPT anchor norms are non-finite"
            )
        if not torch.allclose(
            anchor_norms,
            torch.ones_like(anchor_norms),
            rtol=0.0,
            atol=1e-4,
        ):
            raise RuntimeError(
                f"layer {label} CPT anchors are not column-normalized"
            )

    def _persistent_state_is_materialized(self) -> bool:
        """Whether all tensors needed by the load-time audit are non-meta.

        Transformers may call ``load_state_dict(assign=True)`` repeatedly while
        a low-memory model is only partially materialized.  Auditing inside one
        of those intermediate calls would either inspect unrelated meta tensors
        or fail with ``Tensor.item() cannot be called on meta tensors``.  The
        final model-level/HF load path still calls
        :meth:`validate_persistent_invariants` after every tensor is assigned.
        """
        tensors = (
            self.projection,
            self.anchors,
            self.energy,
            self.congestion_price,
            self.state_version,
            self.router_algorithm_version,
        )
        return all(not value.is_meta for value in tensors)

    def load_state_dict(
        self,
        state_dict,
        strict: bool = True,
        assign: bool = False,
    ):
        result = super().load_state_dict(
            state_dict,
            strict=strict,
            assign=assign,
        )
        if self._persistent_state_is_materialized():
            self.validate_persistent_invariants()
        return result

    def _load_from_state_dict(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ):
        """Validate immutable Router identity before PyTorch casts tensors.

        ``Module.load_state_dict(assign=False)`` normally converts checkpoint
        tensors to the destination dtype before post-load validation.  Without
        this pre-copy gate, a float or bool value can be silently converted to
        an integer and impersonate the implementation's expected Router
        version.  The recursive hook covers direct Router loads, Native model
        loads, and Hugging Face staged loads.
        """
        algorithm_key = prefix + "router_algorithm_version"
        if algorithm_key in state_dict:
            validate_serialized_router_algorithm_version(
                state_dict[algorithm_key],
                key=algorithm_key,
                expected=self.expected_router_algorithm_version,
            )
        super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )


class CPTModelTransactionMixin:
    """Model-level all-or-none CPT transaction operations."""

    def _assert_not_training_poisoned(self, *, operation: str) -> None:
        """Reject computational reuse after a fail-stop training error.

        The training helper marks the live model when an optimizer or another
        transaction phase may already have performed a partial in-place write.
        Training and checkpoint entrypoints reject that instance; model forward
        must reject it too so corrupted parameters cannot be evaluated or used
        for generation as if they were an accepted state.
        """
        reason = getattr(self, "_cpt_training_poison_reason", None)
        if reason is not None:
            raise RuntimeError(
                f"Cannot {operation}: this live model instance is poisoned by a "
                f"prior fail-stop training error ({reason}). Discard it and "
                "reload the latest successfully published checkpoint."
            )

    def _cpt_routers(self) -> tuple[CPTRouter, ...]:
        return tuple(layer.moe.cpt_router for layer in self.layers)

    def prepare_cpt_transaction(
        self,
        proposals: Iterable[CPTLayerProposal],
        microbatch_id: Optional[Hashable] = None,
    ) -> CPTTransaction:
        """Create a pure transaction for one logical micro-batch.

        ``microbatch_id`` must be stable across a replay.  Reusing an identity
        with identical raw statistics is idempotent; reusing it with different
        statistics is rejected.  The default is intentionally suitable only
        for the legacy one-forward-per-step path.  Callers that accumulate
        multiple micro-batches must provide explicit identities.
        """
        identity = self._normalize_microbatch_id(microbatch_id)
        cloned = self._clone_proposal_tuple(proposals)
        return CPTTransaction(
            cloned,
            _microbatch_records={identity: cloned},
        )

    @staticmethod
    def _normalize_microbatch_id(
        microbatch_id: Optional[Hashable],
    ) -> Hashable:
        identity: Hashable = (
            ("single", 0) if microbatch_id is None else microbatch_id
        )
        if isinstance(identity, torch.Tensor):
            raise TypeError("microbatch_id must not be a tensor")
        try:
            hash(identity)
        except TypeError as exc:
            raise TypeError("microbatch_id must be hashable") from exc
        return identity

    @staticmethod
    def _clone_proposal_tuple(
        proposals: Iterable[CPTLayerProposal],
    ) -> tuple[CPTLayerProposal, ...]:
        cloned = []
        for proposal in proposals:
            if not isinstance(proposal, CPTLayerProposal):
                raise TypeError("every CPT proposal must be a CPTLayerProposal")
            tensor_fields = (
                proposal.load_sum,
                proposal.token_count,
                proposal.state_version,
                proposal.valid,
            )
            if not all(isinstance(value, torch.Tensor) for value in tensor_fields):
                raise TypeError("all CPT proposal statistics must be tensors")
            cloned.append(
                CPTLayerProposal(
                    layer_index=int(proposal.layer_index),
                    load_sum=proposal.load_sum.detach().clone(),
                    token_count=proposal.token_count.detach().clone(),
                    state_version=proposal.state_version.detach().clone(),
                    valid=proposal.valid.detach().clone(),
                )
            )
        return tuple(cloned)

    @staticmethod
    def _proposal_tuples_equal(
        left: Sequence[CPTLayerProposal],
        right: Sequence[CPTLayerProposal],
    ) -> bool:
        if len(left) != len(right):
            return False
        predicates = []
        for lhs, rhs in zip(left, right):
            if lhs.layer_index != rhs.layer_index:
                return False
            for name in ("load_sum", "token_count", "state_version", "valid"):
                lhs_value = getattr(lhs, name)
                rhs_value = getattr(rhs, name)
                if (
                    not isinstance(lhs_value, torch.Tensor)
                    or not isinstance(rhs_value, torch.Tensor)
                    or lhs_value.shape != rhs_value.shape
                    or lhs_value.dtype != rhs_value.dtype
                    or lhs_value.device != rhs_value.device
                ):
                    return False
                predicates.append(torch.eq(lhs_value, rhs_value).all())
        return _device_predicates_all(predicates)

    @staticmethod
    def _aggregate_microbatch_records(
        records: dict[Hashable, tuple[CPTLayerProposal, ...]],
    ) -> tuple[CPTLayerProposal, ...]:
        if not records:
            raise RuntimeError("CPT transaction has no micro-batch records")
        proposal_groups = tuple(records.values())
        layer_count = len(proposal_groups[0])
        if any(len(group) != layer_count for group in proposal_groups):
            raise RuntimeError("CPT micro-batches disagree on layer count")

        aggregated = []
        for layer_index in range(layer_count):
            layer_proposals = tuple(
                group[layer_index] for group in proposal_groups
            )
            first = layer_proposals[0]
            aggregated.append(
                CPTLayerProposal(
                    layer_index=first.layer_index,
                    load_sum=torch.stack(
                        [proposal.load_sum.detach().float() for proposal in layer_proposals]
                    ).sum(dim=0),
                    token_count=torch.stack(
                        [
                            proposal.token_count.detach().to(torch.int64)
                            for proposal in layer_proposals
                        ]
                    ).sum(),
                    state_version=first.state_version.detach().clone(),
                    valid=torch.stack(
                        [proposal.valid.detach().to(torch.bool) for proposal in layer_proposals]
                    ).all(),
                )
            )
        return tuple(aggregated)

    def accumulate_cpt_transaction(
        self,
        transaction: CPTTransaction,
        proposals: Iterable[CPTLayerProposal],
        *,
        microbatch_id: Hashable,
    ) -> CPTTransaction:
        """Accumulate detached raw sums/counts exactly once per identity."""
        if transaction.closed:
            state = "aborted" if transaction.aborted else "committed"
            raise RuntimeError(f"CPT transaction is already {state}")
        if transaction._prepared_prices is not None:
            raise RuntimeError(
                "cannot add a CPT micro-batch after price preparation"
            )

        identity = self._normalize_microbatch_id(microbatch_id)
        cloned = self._clone_proposal_tuple(proposals)
        routers = self._cpt_routers()
        self._validate_proposal_tuple(cloned, routers, context=str(identity))

        existing = transaction._microbatch_records.get(identity)
        if existing is not None:
            if self._proposal_tuples_equal(existing, cloned):
                return transaction
            raise RuntimeError(
                f"CPT microbatch_id {identity!r} was reused with different "
                "raw statistics"
            )

        records = dict(transaction._microbatch_records)
        records[identity] = cloned
        aggregate = self._aggregate_microbatch_records(records)
        transaction._microbatch_records = records
        transaction.proposals = aggregate
        return transaction

    def get_cpt_state_version(self) -> int:
        versions = _read_int64_scalars_once(
            [router.state_version for router in self._cpt_routers()]
        )
        if not versions:
            return 0
        if len(set(versions)) != 1:
            raise RuntimeError(f"CPT layer versions disagree: {list(versions)}")
        return versions[0]

    @staticmethod
    def _proposal_tuple_structure_is_valid(
        proposals: Sequence[CPTLayerProposal],
        routers: Sequence[CPTRouter],
    ) -> bool:
        if len(proposals) != len(routers):
            return False
        for layer_index, (router, proposal) in enumerate(zip(routers, proposals)):
            if proposal.layer_index != layer_index:
                return False
            tensor_fields = (
                proposal.load_sum,
                proposal.token_count,
                proposal.state_version,
                proposal.valid,
            )
            if not all(isinstance(value, torch.Tensor) for value in tensor_fields):
                return False
            if (
                proposal.load_sum.shape != (router.num_experts,)
                or proposal.load_sum.device != router.congestion_price.device
                or proposal.load_sum.dtype != torch.float32
                or proposal.token_count.ndim != 0
                or proposal.token_count.device != router.congestion_price.device
                or proposal.token_count.dtype != torch.int64
                or proposal.state_version.ndim != 0
                or proposal.state_version.device != router.congestion_price.device
                or proposal.state_version.dtype != torch.int64
                or proposal.valid.ndim != 0
                or proposal.valid.device != router.congestion_price.device
                or proposal.valid.dtype != torch.bool
            ):
                return False
        return True

    def _validate_proposal_tuple_slow(
        self,
        proposals: Sequence[CPTLayerProposal],
        routers: Sequence[CPTRouter],
        *,
        context: str,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Original ordered diagnostic path used only after a fast-gate miss."""
        current_versions = [
            int(router.state_version.item()) for router in routers
        ]
        if len(set(current_versions)) != 1:
            raise RuntimeError(
                f"CPT layer versions disagree before commit: {current_versions}"
            )

        loads = []
        counts = []
        for layer_index, (router, proposal) in enumerate(
            zip(routers, proposals)
        ):
            if proposal.layer_index != layer_index:
                raise RuntimeError(
                    f"proposal layer index {proposal.layer_index} does not "
                    f"match expected {layer_index}"
                )
            if proposal.load_sum.shape != (router.num_experts,):
                raise RuntimeError(
                    f"layer {layer_index} load_sum has shape "
                    f"{tuple(proposal.load_sum.shape)}, expected "
                    f"({router.num_experts},)"
                )
            if proposal.load_sum.device != router.congestion_price.device:
                raise RuntimeError(
                    f"layer {layer_index} load_sum is on "
                    f"{proposal.load_sum.device}, expected "
                    f"{router.congestion_price.device}"
                )
            if proposal.load_sum.dtype != torch.float32:
                raise RuntimeError("CPT load_sum must use FP32")
            if proposal.token_count.ndim != 0:
                raise RuntimeError("CPT token_count must be a scalar")
            if proposal.token_count.device != router.congestion_price.device:
                raise RuntimeError("CPT token_count is on the wrong device")
            if proposal.token_count.dtype != torch.int64:
                raise RuntimeError("CPT token_count must use int64")
            if proposal.state_version.ndim != 0:
                raise RuntimeError("CPT state_version must be a scalar")
            if proposal.state_version.device != router.congestion_price.device:
                raise RuntimeError("CPT state_version is on the wrong device")
            if proposal.state_version.dtype != torch.int64:
                raise RuntimeError("CPT state_version must use int64")
            if proposal.valid.ndim != 0:
                raise RuntimeError("CPT proposal.valid must be a scalar")
            if proposal.valid.device != router.congestion_price.device:
                raise RuntimeError("CPT proposal.valid is on the wrong device")
            if proposal.valid.dtype != torch.bool:
                raise RuntimeError("CPT proposal.valid must use bool")
            if not bool(proposal.valid.item()):
                raise RuntimeError(f"layer {layer_index} CPT proposal is invalid")

            proposal_version = int(proposal.state_version.item())
            if proposal_version != current_versions[layer_index]:
                raise RuntimeError(
                    f"layer {layer_index} proposal version "
                    f"{proposal_version} does not match active version "
                    f"{current_versions[layer_index]}"
                )

            load = proposal.load_sum.detach().float()
            count = proposal.token_count.detach().to(torch.int64)
            if not torch.isfinite(load).all():
                raise RuntimeError(f"layer {layer_index} CPT load is non-finite")
            if (load < 0).any():
                raise RuntimeError(f"layer {layer_index} CPT load is negative")
            if int(count.item()) < 0:
                raise RuntimeError(f"layer {layer_index} CPT count is negative")
            tolerance = max(1e-4, 1e-4 * max(int(count.item()), 1))
            if abs(float(load.sum().item()) - float(count.item())) > tolerance:
                raise RuntimeError(
                    f"layer {layer_index} CPT probability mass does not "
                    "match token_count"
                )
            loads.append(load)
            counts.append(count)

        stacked_loads = torch.stack(loads)
        stacked_counts = torch.stack(counts)
        if stacked_counts.numel() > 1 and not torch.equal(
            stacked_counts,
            stacked_counts[0].expand_as(stacked_counts),
        ):
            raise RuntimeError("CPT layers disagree on route-valid token_count")
        return stacked_loads, stacked_counts

    def _validate_proposal_tuple(
        self,
        proposals: Sequence[CPTLayerProposal],
        routers: Sequence[CPTRouter],
        *,
        context: str,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if len(proposals) != len(routers):
            raise RuntimeError(
                f"expected {len(routers)} CPT layer proposals, got "
                f"{len(proposals)} in {context}"
            )

        if not self._proposal_tuple_structure_is_valid(proposals, routers):
            return self._validate_proposal_tuple_slow(
                proposals,
                routers,
                context=context,
            )

        stacked_loads = torch.stack(
            [proposal.load_sum.detach().float() for proposal in proposals]
        )
        stacked_counts = torch.stack(
            [proposal.token_count.detach().to(torch.int64) for proposal in proposals]
        )
        current_versions = torch.stack(
            [router.state_version.detach() for router in routers]
        )
        proposal_versions = torch.stack(
            [proposal.state_version.detach() for proposal in proposals]
        )
        proposal_validity = torch.stack(
            [proposal.valid.detach() for proposal in proposals]
        )
        count_values = stacked_counts.to(torch.float64)
        mass_tolerance = 1e-4 * count_values.clamp_min(1.0)
        mass_matches = (
            (
                stacked_loads.sum(dim=1).to(torch.float64)
                - count_values
            ).abs()
            <= mass_tolerance
        ).all()

        predicates = (
            (current_versions == current_versions[0]).all(),
            proposal_validity.all(),
            (proposal_versions == current_versions).all(),
            torch.isfinite(stacked_loads).all(),
            (stacked_loads >= 0).all(),
            (stacked_counts >= 0).all(),
            mass_matches,
            (stacked_counts == stacked_counts[0]).all(),
        )
        if _device_predicates_all(predicates):
            return stacked_loads, stacked_counts
        return self._validate_proposal_tuple_slow(
            proposals,
            routers,
            context=context,
        )

    def _validate_local_proposals(
        self,
        transaction: CPTTransaction,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if transaction.closed:
            state = "aborted" if transaction.aborted else "committed"
            raise RuntimeError(f"CPT transaction is already {state}")
        if not transaction._microbatch_records:
            raise RuntimeError("CPT transaction has no micro-batch records")

        routers = self._cpt_routers()
        for identity, proposals in transaction._microbatch_records.items():
            self._normalize_microbatch_id(identity)
            self._validate_proposal_tuple(
                proposals,
                routers,
                context=f"micro-batch {identity!r}",
            )

        expected_aggregate = self._aggregate_microbatch_records(
            transaction._microbatch_records
        )
        if not self._proposal_tuples_equal(
            transaction.proposals,
            expected_aggregate,
        ):
            raise RuntimeError(
                "CPT aggregate proposals were modified outside the "
                "micro-batch accumulator"
            )
        return self._validate_proposal_tuple(
            transaction.proposals,
            routers,
            context="aggregate transaction",
        )

    @staticmethod
    def _validate_prepared_prices_slow(
        prices: Sequence[torch.Tensor],
        routers: Sequence[CPTRouter],
    ) -> None:
        for layer_index, (router, price) in enumerate(zip(routers, prices)):
            if not isinstance(price, torch.Tensor):
                raise RuntimeError(
                    f"layer {layer_index} prepared CPT price must be a tensor"
                )
            if price.shape != (router.num_experts,):
                raise RuntimeError(
                    f"layer {layer_index} prepared CPT price has shape "
                    f"{tuple(price.shape)}, expected ({router.num_experts},)"
                )
            if price.device != router.congestion_price.device:
                raise RuntimeError(
                    f"layer {layer_index} prepared CPT price is on "
                    f"{price.device}, expected {router.congestion_price.device}"
                )
            if price.dtype != torch.float32:
                raise RuntimeError(
                    f"layer {layer_index} prepared CPT price must use FP32"
                )
            if not torch.isfinite(price).all() or (price < 0).any():
                raise RuntimeError(
                    f"layer {layer_index} prepared CPT price is invalid"
                )

    @staticmethod
    def _prepared_prices_structure_is_valid(
        prices: Sequence[torch.Tensor],
        routers: Sequence[CPTRouter],
    ) -> bool:
        return all(
            isinstance(price, torch.Tensor)
            and price.shape == (router.num_experts,)
            and price.device == router.congestion_price.device
            and price.dtype == torch.float32
            for router, price in zip(routers, prices)
        )

    @staticmethod
    def _validate_prepared_prices(
        transaction: CPTTransaction,
        routers: Sequence[CPTRouter],
    ) -> tuple[torch.Tensor, ...]:
        prices = transaction._prepared_prices
        if prices is None:
            raise RuntimeError("CPT transaction has no prepared prices")
        if len(prices) != len(routers):
            raise RuntimeError(
                f"expected {len(routers)} prepared CPT prices, got {len(prices)}"
            )
        if not CPTModelTransactionMixin._prepared_prices_structure_is_valid(
            prices,
            routers,
        ):
            CPTModelTransactionMixin._validate_prepared_prices_slow(
                prices,
                routers,
            )
        predicates = (
            torch.isfinite(price).all() & (price >= 0).all()
            for price in prices
        )
        if not _device_predicates_all(predicates):
            CPTModelTransactionMixin._validate_prepared_prices_slow(
                prices,
                routers,
            )
        return prices

    @staticmethod
    def _validate_learnable_router_state_slow(
        routers: Sequence[CPTRouter],
    ) -> None:
        for layer_index, router in enumerate(routers):
            for name in FP32_ROUTER_PARAMETER_NAMES:
                value = getattr(router, name).detach()
                if value.dtype != torch.float32:
                    raise RuntimeError(
                        f"layer {layer_index} CPT {name} must remain FP32 after "
                        "optimizer step"
                    )
                if not torch.isfinite(value).all():
                    raise RuntimeError(
                        f"layer {layer_index} CPT {name} is non-finite after "
                        "optimizer step"
                    )

    @staticmethod
    def _validate_learnable_router_state(
        routers: Sequence[CPTRouter],
    ) -> None:
        values = [
            getattr(router, name).detach()
            for router in routers
            for name in FP32_ROUTER_PARAMETER_NAMES
        ]
        if not all(value.dtype == torch.float32 for value in values):
            CPTModelTransactionMixin._validate_learnable_router_state_slow(
                routers
            )
        if not _device_predicates_all(
            torch.isfinite(value).all() for value in values
        ):
            CPTModelTransactionMixin._validate_learnable_router_state_slow(
                routers
            )

    @staticmethod
    def _validated_normalized_anchors(
        routers: Sequence[CPTRouter],
    ) -> list[torch.Tensor]:
        anchor_values = [router.anchors.detach().float() for router in routers]
        norms = [
            anchors.norm(dim=0, keepdim=True) for anchors in anchor_values
        ]
        predicates = []
        for router, anchors, layer_norms in zip(
            routers,
            anchor_values,
            norms,
        ):
            predicates.extend(
                (
                    torch.isfinite(anchors).all()
                    & torch.isfinite(layer_norms).all(),
                    (layer_norms > router.eps_init).all(),
                )
            )
        if not _device_predicates_all(predicates):
            for layer_index, (router, anchors, layer_norms) in enumerate(
                zip(routers, anchor_values, norms)
            ):
                if (
                    not torch.isfinite(anchors).all()
                    or not torch.isfinite(layer_norms).all()
                ):
                    raise RuntimeError(
                        f"layer {layer_index} CPT anchors are non-finite"
                    )
                if (layer_norms <= router.eps_init).any():
                    raise RuntimeError(
                        f"layer {layer_index} CPT anchor norm is zero"
                    )
            raise RuntimeError("CPT anchor validation failed without a diagnosis")
        return [
            anchors / layer_norms
            for anchors, layer_norms in zip(anchor_values, norms)
        ]

    @staticmethod
    def _distributed_active() -> bool:
        return (
            dist.is_available()
            and dist.is_initialized()
            and dist.get_world_size() > 1
        )

    def _raise_after_validation_consensus(
        self,
        local_error: Optional[Exception],
        *,
        context: str,
    ) -> None:
        if not self._distributed_active():
            if local_error is not None:
                raise local_error
            return

        routers = self._cpt_routers()
        device = (
            routers[0].congestion_price.device
            if routers
            else torch.device("cpu")
        )
        valid_flag = torch.tensor(
            0 if local_error is not None else 1,
            device=device,
            dtype=torch.int32,
        )
        dist.all_reduce(valid_flag, op=dist.ReduceOp.MIN)
        if int(valid_flag.item()) != 1:
            if local_error is not None:
                raise local_error
            raise RuntimeError(
                f"another distributed rank rejected CPT {context}"
            )
        if local_error is not None:
            raise local_error

    @staticmethod
    def _validate_global_statistics_slow(
        loads: torch.Tensor,
        counts: torch.Tensor,
    ) -> None:
        if not torch.isfinite(loads).all():
            raise RuntimeError("global CPT load is non-finite")
        if (loads < 0).any():
            raise RuntimeError("global CPT load is negative")
        if (counts < 0).any():
            raise RuntimeError("global CPT token_count is negative")
        for layer_index, (load, count) in enumerate(zip(loads, counts)):
            count_value = int(count.item())
            tolerance = max(1e-4, 1e-4 * max(count_value, 1))
            if abs(float(load.sum().item()) - float(count_value)) > tolerance:
                raise RuntimeError(
                    f"global layer {layer_index} CPT probability mass does not "
                    "match token_count"
                )
        if counts.numel() > 1 and not torch.equal(
            counts,
            counts[0].expand_as(counts),
        ):
            raise RuntimeError(
                "global CPT layers disagree on route-valid token_count"
            )

    @staticmethod
    def _validate_global_statistics(
        loads: torch.Tensor,
        counts: torch.Tensor,
    ) -> None:
        count_values = counts.to(torch.float64)
        mass_tolerance = 1e-4 * count_values.clamp_min(1.0)
        mass_matches = (
            (loads.sum(dim=1).to(torch.float64) - count_values).abs()
            <= mass_tolerance
        ).all()
        predicates = (
            torch.isfinite(loads).all(),
            (loads >= 0).all(),
            (counts >= 0).all(),
            mass_matches,
            (counts == counts[0]).all(),
        )
        if not _device_predicates_all(predicates):
            CPTModelTransactionMixin._validate_global_statistics_slow(
                loads,
                counts,
            )

    def _validated_global_statistics(
        self,
        transaction: CPTTransaction,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        local_error: Optional[Exception] = None
        loads: Optional[torch.Tensor] = None
        counts: Optional[torch.Tensor] = None
        try:
            loads, counts = self._validate_local_proposals(transaction)
        except Exception as error:
            local_error = error
        self._raise_after_validation_consensus(
            local_error,
            context="local proposals",
        )
        assert loads is not None and counts is not None

        global_loads = loads.detach().float().clone()
        global_counts = counts.detach().to(torch.int64).clone()
        if self._distributed_active():
            # Sum raw probability mass and raw token counts.  Dividing only
            # after these collectives gives the required token-weighted global
            # mean; rank-local means are never averaged.
            dist.all_reduce(global_loads, op=dist.ReduceOp.SUM)
            dist.all_reduce(global_counts, op=dist.ReduceOp.SUM)

        global_error = None
        try:
            self._validate_global_statistics(global_loads, global_counts)
        except Exception as error:
            global_error = error
        self._raise_after_validation_consensus(
            global_error,
            context="global raw statistics",
        )
        return global_loads, global_counts

    @staticmethod
    def _compute_price_candidates(
        routers: Sequence[CPTRouter],
        loads: torch.Tensor,
        counts: torch.Tensor,
        base_prices: Sequence[torch.Tensor],
    ) -> tuple[torch.Tensor, ...]:
        prices = []
        for layer_index, (router, load, count, base_price) in enumerate(
            zip(routers, loads, counts, base_prices)
        ):
            base_price_fp32 = base_price.detach().float()
            safe_count = count.to(torch.float32).clamp_min(1.0)
            mean_probability = load.float() / safe_count
            updated = torch.clamp_min(
                base_price_fp32
                + router.price_learning_rate
                * (
                    mean_probability
                    - router.capacity_factor / router.num_experts
                ),
                0.0,
            )
            candidate = torch.where(
                count > 0,
                updated,
                base_price_fp32,
            )
            prices.append(candidate)
        if all(candidate.dtype == torch.float32 for candidate in prices):
            predicates = (
                torch.isfinite(candidate).all() & (candidate >= 0).all()
                for candidate in prices
            )
            if _device_predicates_all(predicates):
                return tuple(prices)
        for layer_index, candidate in enumerate(prices):
            if candidate.dtype != torch.float32:
                raise RuntimeError(
                    f"layer {layer_index} CPT price candidate is not FP32"
                )
            if not torch.isfinite(candidate).all() or (candidate < 0).any():
                raise RuntimeError(
                    f"layer {layer_index} CPT price candidate is invalid"
                )
        return tuple(prices)

    @staticmethod
    def _prepared_snapshot_structure_is_valid(
        transaction: CPTTransaction,
        routers: Sequence[CPTRouter],
        global_loads: torch.Tensor,
        global_counts: torch.Tensor,
    ) -> bool:
        base_prices = transaction._prepared_base_prices
        base_versions = transaction._prepared_base_versions
        prepared_loads = transaction._prepared_global_loads
        prepared_counts = transaction._prepared_global_counts
        if (
            base_prices is None
            or base_versions is None
            or prepared_loads is None
            or prepared_counts is None
            or len(base_prices) != len(routers)
            or len(base_versions) != len(routers)
            or not isinstance(prepared_loads, torch.Tensor)
            or not isinstance(prepared_counts, torch.Tensor)
            or prepared_loads.shape != global_loads.shape
            or prepared_loads.dtype != global_loads.dtype
            or prepared_loads.device != global_loads.device
            or prepared_counts.shape != global_counts.shape
            or prepared_counts.dtype != global_counts.dtype
            or prepared_counts.device != global_counts.device
        ):
            return False
        if not all(isinstance(version, int) for version in base_versions):
            return False
        return all(
            isinstance(base_price, torch.Tensor)
            and base_price.shape == router.congestion_price.shape
            and base_price.dtype == torch.float32
            and base_price.device == router.congestion_price.device
            for router, base_price in zip(routers, base_prices)
        )

    def _validate_prepared_snapshot_slow(
        self,
        transaction: CPTTransaction,
        routers: Sequence[CPTRouter],
        global_loads: torch.Tensor,
        global_counts: torch.Tensor,
        prepared_prices: Sequence[torch.Tensor],
    ) -> None:
        """Original ordered snapshot diagnostics after a device gate fails."""
        current_versions = tuple(
            int(router.state_version.item()) for router in routers
        )
        if transaction._prepared_base_versions != current_versions:
            raise RuntimeError(
                "CPT router versions changed after price preparation"
            )
        if len(transaction._prepared_base_prices) != len(routers):
            raise RuntimeError("CPT prepared base-price layer count is invalid")
        for layer_index, (router, base_price) in enumerate(
            zip(routers, transaction._prepared_base_prices)
        ):
            if (
                base_price.dtype != torch.float32
                or base_price.device != router.congestion_price.device
                or not torch.equal(base_price, router.congestion_price)
            ):
                raise RuntimeError(
                    f"layer {layer_index} active CPT price changed after preparation"
                )

        if not torch.equal(
            transaction._prepared_global_counts,
            global_counts,
        ):
            raise RuntimeError("CPT global token counts changed after preparation")
        if not torch.allclose(
            transaction._prepared_global_loads,
            global_loads,
            rtol=0.0,
            atol=1e-7,
        ):
            raise RuntimeError("CPT global raw loads changed after preparation")

        expected_prices = self._compute_price_candidates(
            routers,
            global_loads,
            global_counts,
            transaction._prepared_base_prices,
        )
        for layer_index, (actual, expected) in enumerate(
            zip(prepared_prices, expected_prices)
        ):
            if not torch.allclose(actual, expected, rtol=0.0, atol=1e-7):
                raise RuntimeError(
                    f"layer {layer_index} prepared CPT price does not match "
                    "its immutable raw-statistics formula"
                )

    def _validate_prepared_snapshot(
        self,
        transaction: CPTTransaction,
        routers: Sequence[CPTRouter],
        global_loads: torch.Tensor,
        global_counts: torch.Tensor,
    ) -> None:
        prepared_prices = self._validate_prepared_prices(transaction, routers)
        if transaction._prepared_base_prices is None:
            raise RuntimeError("CPT transaction has no prepared base prices")
        if transaction._prepared_base_versions is None:
            raise RuntimeError("CPT transaction has no prepared base versions")
        if transaction._prepared_global_loads is None:
            raise RuntimeError("CPT transaction has no prepared global loads")
        if transaction._prepared_global_counts is None:
            raise RuntimeError("CPT transaction has no prepared global counts")
        if transaction._prepared_microbatch_ids is None:
            raise RuntimeError("CPT transaction has no prepared micro-batch IDs")
        if transaction._prepared_microbatch_ids != transaction.microbatch_ids:
            raise RuntimeError("CPT micro-batch identities changed after preparation")

        if not self._prepared_snapshot_structure_is_valid(
            transaction,
            routers,
            global_loads,
            global_counts,
        ):
            self._validate_prepared_snapshot_slow(
                transaction,
                routers,
                global_loads,
                global_counts,
                prepared_prices,
            )

        current_versions = torch.stack(
            [router.state_version.detach() for router in routers]
        )
        prepared_versions = torch.tensor(
            transaction._prepared_base_versions,
            device=current_versions.device,
            dtype=current_versions.dtype,
        )
        snapshot_predicates = [
            torch.eq(current_versions, prepared_versions).all(),
            torch.eq(
                transaction._prepared_global_counts,
                global_counts,
            ).all(),
            torch.isclose(
                transaction._prepared_global_loads,
                global_loads,
                rtol=0.0,
                atol=1e-7,
            ).all(),
        ]
        snapshot_predicates.extend(
            torch.eq(base_price, router.congestion_price).all()
            for router, base_price in zip(
                routers,
                transaction._prepared_base_prices,
            )
        )
        if not _device_predicates_all(snapshot_predicates):
            self._validate_prepared_snapshot_slow(
                transaction,
                routers,
                global_loads,
                global_counts,
                prepared_prices,
            )

        expected_prices = self._compute_price_candidates(
            routers,
            global_loads,
            global_counts,
            transaction._prepared_base_prices,
        )
        formula_predicates = (
            torch.isclose(actual, expected, rtol=0.0, atol=1e-7).all()
            for actual, expected in zip(prepared_prices, expected_prices)
        )
        if not _device_predicates_all(formula_predicates):
            self._validate_prepared_snapshot_slow(
                transaction,
                routers,
                global_loads,
                global_counts,
                prepared_prices,
            )

    @staticmethod
    def _clear_prepared_transaction(transaction: CPTTransaction) -> None:
        transaction._prepared_prices = None
        transaction._prepared_base_prices = None
        transaction._prepared_base_versions = None
        transaction._prepared_global_loads = None
        transaction._prepared_global_counts = None
        transaction._prepared_microbatch_ids = None

    def validate_cpt_transaction(self, transaction: CPTTransaction) -> None:
        """Validate proposals and precompute all lambda shadow candidates."""
        routers = self._cpt_routers()
        global_loads, global_counts = self._validated_global_statistics(
            transaction
        )

        if transaction._prepared_prices is None:
            base_prices = tuple(
                router.congestion_price.detach().float().clone()
                for router in routers
            )
            local_error = None
            prices: Optional[tuple[torch.Tensor, ...]] = None
            try:
                prices = self._compute_price_candidates(
                    routers,
                    global_loads,
                    global_counts,
                    base_prices,
                )
            except Exception as error:
                local_error = error
            self._raise_after_validation_consensus(
                local_error,
                context="price candidates",
            )
            assert prices is not None
            transaction._prepared_prices = tuple(
                price.detach().float().clone() for price in prices
            )
            transaction._prepared_base_prices = base_prices
            transaction._prepared_base_versions = _read_int64_scalars_once(
                [router.state_version for router in routers]
            )
            transaction._prepared_global_loads = global_loads.detach().clone()
            transaction._prepared_global_counts = global_counts.detach().clone()
            transaction._prepared_microbatch_ids = transaction.microbatch_ids
            self._validate_prepared_prices(transaction, routers)
            return

        prepared_error = None
        try:
            self._validate_prepared_snapshot(
                transaction,
                routers,
                global_loads,
                global_counts,
            )
        except Exception as error:
            prepared_error = error
        self._raise_after_validation_consensus(
            prepared_error,
            context="prepared price snapshot",
        )

    def commit_cpt_transaction(self, transaction: CPTTransaction) -> None:
        """Jointly commit all layers after optimizer and scheduler succeed."""
        self.validate_cpt_transaction(transaction)
        routers = self._cpt_routers()
        post_optimizer_error = None
        prepared_prices: Optional[tuple[torch.Tensor, ...]] = None
        normalized_anchors: list[torch.Tensor] = []
        try:
            self._validate_learnable_router_state(routers)
            prepared_prices = self._validate_prepared_prices(transaction, routers)
            normalized_anchors = self._validated_normalized_anchors(routers)
        except Exception as error:
            post_optimizer_error = error
        self._raise_after_validation_consensus(
            post_optimizer_error,
            context="post-optimizer router state",
        )
        assert prepared_prices is not None

        anchor_backups = [router.anchors.detach().clone() for router in routers]
        price_backups = [
            router.congestion_price.detach().clone() for router in routers
        ]
        version_backups = [router.state_version.detach().clone() for router in routers]

        try:
            with torch.no_grad():
                for router, normalized, price in zip(
                    routers,
                    normalized_anchors,
                    prepared_prices,
                ):
                    router.anchors.copy_(normalized.to(router.anchors.dtype))
                    router.congestion_price.copy_(price.float())
                for router in routers:
                    router.state_version.add_(1)
        except BaseException:
            with torch.no_grad():
                for router, anchor, price, version in zip(
                    routers,
                    anchor_backups,
                    price_backups,
                    version_backups,
                ):
                    router.anchors.copy_(anchor)
                    router.congestion_price.copy_(price)
                    router.state_version.copy_(version)
            raise

        transaction.closed = True
        self._clear_prepared_transaction(transaction)

    def abort_cpt_transaction(self, transaction: Optional[CPTTransaction]) -> None:
        """Close a proposal without changing any persistent CPT state."""
        if transaction is None or transaction.closed:
            return
        transaction.closed = True
        transaction.aborted = True
        self._clear_prepared_transaction(transaction)

    def load_state_dict(self, state_dict, strict: bool = True, assign: bool = False):
        legacy_keys = [
            key for key in state_dict if key.endswith(LEGACY_ROUTER_KEY_SUFFIX)
        ]
        if legacy_keys:
            raise RuntimeError(
                "Legacy Linear-router checkpoint is incompatible with CPT "
                "probability routing. CPT v1 must be trained from scratch; "
                "silent partial loading is forbidden."
            )
        result = super().load_state_dict(
            state_dict,
            strict=strict,
            assign=assign,
        )
        routers = self._cpt_routers()
        materialized = tuple(
            router._persistent_state_is_materialized() for router in routers
        )
        for layer_index, (router, is_materialized) in enumerate(
            zip(routers, materialized)
        ):
            if is_materialized:
                router.validate_persistent_invariants(layer_index=layer_index)
        # Transformers' low-memory loader may call this method while only a
        # subset of layers has left the meta device.  Cross-layer agreement is
        # meaningful only after every router is fully materialized; at that
        # final boundary mixed-version checkpoints are rejected atomically.
        if all(materialized):
            self.get_cpt_state_version()
        return result
