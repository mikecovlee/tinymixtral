# Copyright (C) Michael Lee (李登淳) 2026. All rights reserved.
# Open-source under the MIT License. See LICENSE for details.

"""Native CPT-MoE probability Router with blockwise sequence state.

Theory uses column vectors.  For one token ``x_t in R^{d x 1}``:

    z_t = StableL2(P x_t)
    q_t = softmax(M_{t-1}^T z_t / tau_p)
    B   = row_softmax((Theta_C H_N - 1 lambda^T) / tau_e)
    pi_t = B^T q_t

The implementation stores tokens row-major, hence the final equivalent
operation is ``pi_t.T = q_t.T @ B``.  There is deliberately no softmax
after this product. Tokens in one state chunk share its entry state and are
routed in parallel; setting ``cpt_state_chunk_size=1`` recovers strict v1.

With ``cpt_state_corrector=True`` an optional predictor-corrector pass
refines the chunk probabilities: pass one routes from the entry state, then a
first-order per-position trajectory of ``(S, nu)`` built from the pass-one
responsibilities (exclusive prefix sums, all parallel) yields corrected
per-position prototypes for pass two.  The trajectory correction vanishes for
single-token chunks, so ``cpt_state_chunk_size=1`` still recovers strict v1
exactly.  The default keeps the frozen blockwise semantics.
"""

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import CPT_ROUTING_ARCHITECTURE_FIELDS, TinyMixtralConfig


CPT_ROUTER_ALGORITHM_VERSION = 1

# Full-forward compilation unrolls the chunk loop; compile time grows with the
# number of chunks, so only compile when the unroll stays moderate.  Larger
# sequences should raise ``cpt_state_chunk_size`` rather than drop to eager.
_MAX_COMPILED_CHUNKS = 128


@dataclass(frozen=True)
class CPTLayerProposal:
    """Detached soft-load proposal produced by one Router forward."""

    layer_index: int
    load_sum: torch.Tensor
    token_count: torch.Tensor
    state_version: torch.Tensor


@dataclass
class CPTTransaction:
    """All-layer transient proposal for one model forward."""

    proposals: tuple[CPTLayerProposal, ...]
    consumed: bool = False


@dataclass(frozen=True)
class CPTRouterOutput:
    """FP32 expert probabilities and the detached state proposal."""

    probabilities: torch.Tensor
    proposal: CPTLayerProposal


class CPTRouter(nn.Module):
    """CPT v1 Router for a single MoE layer.

    ``projection``, ``anchors`` and ``energy`` (the code name for
    ``Theta_C``) are learnable.  The congestion price, optimizer step and
    Router state version are persistent but never optimized.  Sequence state
    ``S, nu`` is local to one forward and always starts from zero for every
    batch row.
    """

    def __init__(self, config: TinyMixtralConfig, layer_index: int):
        super().__init__()
        if isinstance(layer_index, bool) or not isinstance(layer_index, int):
            raise ValueError("layer_index must be an integer")
        if not 0 <= layer_index < config.num_hidden_layers:
            raise ValueError("layer_index must identify an existing model layer")

        self.layer_index = layer_index
        self._cpt_config_snapshot = config.cpt_config_dict()
        self._architecture_snapshot = {
            field: getattr(config, field)
            for field in CPT_ROUTING_ARCHITECTURE_FIELDS
        }
        self.hidden_size = config.hidden_size
        self.projection_dim = config.cpt_projection_dim
        self.num_prototypes = config.cpt_num_prototypes
        self.num_experts = config.num_local_experts

        self.rho_beta = config.cpt_rho_beta
        self.beta_max = config.cpt_beta_max
        self.kappa_beta = config.cpt_kappa_beta
        self.lambda_sa = config.cpt_lambda_sa
        self.prototype_temperature = config.cpt_prototype_temperature
        self.expert_temperature = config.cpt_expert_temperature
        self.state_step_size = config.cpt_state_step_size
        self.state_radius = config.cpt_state_radius
        self.state_chunk_size = config.cpt_state_chunk_size
        self.state_corrector = config.cpt_state_corrector
        self.eps_z = config.cpt_eps_z
        self.eps_m = config.cpt_eps_m
        self.eps_init = config.cpt_eps_init
        self.energy_init_scale = config.cpt_energy_init_scale
        self.capacity_factor = config.cpt_capacity_factor
        self.price_learning_rate = config.cpt_price_learning_rate
        self.init_seed = config.cpt_init_seed + 104_729 * layer_index

        self.projection = nn.Parameter(
            torch.empty(self.projection_dim, self.hidden_size, dtype=torch.float32)
        )
        self.anchors = nn.Parameter(
            torch.empty(self.projection_dim, self.num_prototypes, dtype=torch.float32)
        )
        self.energy = nn.Parameter(
            torch.empty(self.num_prototypes, self.num_experts, dtype=torch.float32)
        )
        self.register_buffer(
            "congestion_price",
            torch.zeros(self.num_experts, dtype=torch.float32),
        )
        self.register_buffer("optimizer_step", torch.zeros((), dtype=torch.int64))
        self.register_buffer("state_version", torch.zeros((), dtype=torch.int64))
        self.register_buffer(
            "router_algorithm_version",
            torch.tensor(CPT_ROUTER_ALGORITHM_VERSION, dtype=torch.int64),
        )
        self.reset_parameters()
        self._compiled_forward_impl = None

    def _apply(self, fn, recurse: bool = True):
        """Move devices normally while retaining the CPT FP32 precision island."""
        self._compiled_forward_impl = None
        parameter_state = {
            name: (
                parameter.detach().clone(),
                None
                if parameter.grad is None
                else parameter.grad.detach().clone(),
            )
            for name, parameter in (
                ("projection", self.projection),
                ("anchors", self.anchors),
                ("energy", self.energy),
            )
        }
        congestion_price = self.congestion_price.detach().clone()
        super()._apply(fn, recurse=recurse)
        for name, (value, gradient) in parameter_state.items():
            parameter = getattr(self, name)
            if parameter.dtype != torch.float32:
                parameter.data = value.to(
                    device=parameter.device,
                    dtype=torch.float32,
                )
            if parameter.grad is not None and parameter.grad.dtype != torch.float32:
                parameter.grad.data = gradient.to(
                    device=parameter.device,
                    dtype=torch.float32,
                )
        if self.congestion_price.dtype != torch.float32:
            self.congestion_price.data = congestion_price.to(
                device=self.congestion_price.device,
                dtype=torch.float32,
            )
        return self

    @torch.no_grad()
    def reset_parameters(self) -> None:
        generator = torch.Generator(device="cpu")
        generator.manual_seed(self.init_seed)

        if self.projection_dim <= self.hidden_size:
            candidate = torch.randn(
                self.hidden_size,
                self.projection_dim,
                generator=generator,
                dtype=torch.float32,
            )
            q, _ = torch.linalg.qr(candidate, mode="reduced")
            projection = q.T.contiguous()
        else:
            projection = torch.randn(
                self.projection_dim,
                self.hidden_size,
                generator=generator,
                dtype=torch.float32,
            ) / self.hidden_size ** 0.5
        self.projection.copy_(projection)

        if self.num_prototypes <= self.projection_dim:
            anchor_candidate = torch.randn(
                self.projection_dim,
                self.num_prototypes,
                generator=generator,
                dtype=torch.float32,
            )
            anchors, _ = torch.linalg.qr(anchor_candidate, mode="reduced")
        else:
            anchors = None
            for _ in range(16):
                anchor_candidate = torch.randn(
                    self.projection_dim,
                    self.num_prototypes,
                    generator=generator,
                    dtype=torch.float32,
                )
                candidate_norms = torch.linalg.vector_norm(
                    anchor_candidate,
                    dim=0,
                    keepdim=True,
                )
                if (
                    not bool(torch.isfinite(candidate_norms).all())
                    or bool((candidate_norms <= 0).any())
                ):
                    continue
                candidate = anchor_candidate / candidate_norms
                unique_directions = torch.unique(candidate.T, dim=0).shape[0]
                if (
                    bool(torch.isfinite(candidate).all())
                    and unique_directions == self.num_prototypes
                ):
                    anchors = candidate
                    break
            if anchors is None:
                raise RuntimeError("failed to initialize distinct CPT anchors")
        self.anchors.copy_(anchors)

        initialized = False
        for _ in range(16):
            xi = torch.randn(
                self.num_prototypes,
                self.num_experts,
                generator=generator,
                dtype=torch.float32,
            )
            direction = xi - xi.mean(dim=0, keepdim=True)
            direction = direction - direction.mean(dim=1, keepdim=True)
            scale = direction.abs().amax()
            if not bool(torch.isfinite(scale)):
                continue
            direction = direction / scale.clamp_min(self.eps_init)
            energy = self.energy_init_scale * direction
            centered = energy - energy.mean(dim=-1, keepdim=True)
            rows_identical = torch.equal(
                centered,
                centered[:1].expand_as(centered),
            )
            if bool(torch.isfinite(centered).all()) and not rows_identical:
                self.energy.copy_(energy)
                initialized = True
                break
        if not initialized:
            raise RuntimeError("failed to initialize non-degenerate CPT energy")

        self.congestion_price.zero_()
        self.optimizer_step.zero_()
        self.state_version.zero_()
        self.router_algorithm_version.fill_(CPT_ROUTER_ALGORITHM_VERSION)
        self.validate_persistent_state(require_unit_anchors=True)

    @staticmethod
    def _stable_l2(tensor: torch.Tensor, dim: int, eps: float) -> torch.Tensor:
        norm = torch.linalg.vector_norm(tensor, dim=dim, keepdim=True)
        return tensor / norm.clamp_min(eps)

    def _route_chunk(
        self,
        z_chunk: torch.Tensor,
        valid_chunk: torch.Tensor,
        state_old: torch.Tensor,
        nu_old: torch.Tensor,
        rho: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Route one chunk from the entry state and advance ``(S, nu)``.

        With ``cpt_state_corrector`` enabled, a second pass refines the chunk
        probabilities using a first-order estimate of the intra-chunk state
        and responsibility trajectories; the estimate is exact for
        single-token chunks.
        """
        valid_rows = valid_chunk.any(dim=1, keepdim=True)

        beta = self.beta_max * nu_old / (nu_old + self.kappa_beta)
        mixed = (
            self.anchors.unsqueeze(0)
            * (1.0 - beta.detach().unsqueeze(1))
            + state_old.detach() * beta.detach().unsqueeze(1)
        )
        prototypes = self._stable_l2(mixed, dim=1, eps=self.eps_m)
        prototype_logits = torch.bmm(z_chunk, prototypes) / self.prototype_temperature
        q_chunk = F.softmax(prototype_logits, dim=-1, dtype=torch.float32)
        q_chunk = q_chunk * valid_chunk.unsqueeze(-1)

        if self.state_corrector:
            q_chunk = self._correct_chunk(
                z_chunk, valid_chunk, state_old, nu_old, rho, q_chunk
            )

        with torch.no_grad():
            q_detached = q_chunk.detach()
            valid_count_int = valid_chunk.sum(dim=1)
            valid_count = valid_count_int.to(torch.float32)
            routing_mass = q_detached.sum(dim=1)
            weighted_projection = torch.bmm(
                z_chunk.detach().transpose(1, 2),
                q_detached,
            )
            state_gradient = (
                state_old * routing_mass.unsqueeze(1)
                - weighted_projection
                + self.lambda_sa
                * valid_count.view(-1, 1, 1)
                * (state_old - self.anchors.detach().unsqueeze(0))
            )
            state_candidate = state_old - self.state_step_size * state_gradient
            state_norm = torch.linalg.vector_norm(
                state_candidate, dim=1, keepdim=True
            )
            state_candidate = state_candidate / torch.clamp_min(
                state_norm / self.state_radius,
                1.0,
            )
            valid_after = valid_count_int.unsqueeze(-1) - valid_chunk.cumsum(
                dim=1,
            )
            decay_weights = rho.pow(valid_after).unsqueeze(-1)
            retained_responsibility = rho.pow(valid_count).unsqueeze(-1)
            nu_candidate = (
                retained_responsibility * nu_old
                + (q_detached * decay_weights).sum(dim=1)
            )
            next_state = torch.where(
                valid_rows.unsqueeze(-1), state_candidate, state_old
            )
            next_responsibility = torch.where(
                valid_rows, nu_candidate, nu_old
            )
        return q_chunk, next_state, next_responsibility

    def _correct_chunk(
        self,
        z_chunk: torch.Tensor,
        valid_chunk: torch.Tensor,
        state_old: torch.Tensor,
        nu_old: torch.Tensor,
        rho: torch.Tensor,
        q_chunk: torch.Tensor,
    ) -> torch.Tensor:
        """Refine chunk probabilities with first-order state trajectories.

        Pass 1 routed every token from the entry state.  This pass estimates
        the state ``S_t`` and responsibility ``nu_t`` that the strict
        sequential update would have held at each position (using pass-1
        responsibilities, via exclusive prefix sums) and re-routes each token
        from its own corrected prototype.  Single-token chunks have an empty
        prefix and therefore reproduce pass 1 exactly.
        """
        with torch.no_grad():
            token_gradients = (
                (state_old.unsqueeze(1) - z_chunk.unsqueeze(-1))
                * q_chunk.unsqueeze(2)
                + self.lambda_sa
                * (state_old.unsqueeze(1) - self.anchors.detach().unsqueeze(0).unsqueeze(0))
            ) * valid_chunk.unsqueeze(-1).unsqueeze(-1)
            exclusive_gradients = token_gradients.cumsum(dim=1) - token_gradients
            state_trajectory = (
                state_old.unsqueeze(1) - self.state_step_size * exclusive_gradients
            )
            trajectory_norm = torch.linalg.vector_norm(
                state_trajectory, dim=2, keepdim=True
            )
            state_trajectory = state_trajectory / torch.clamp_min(
                trajectory_norm / self.state_radius,
                1.0,
            )

            valid_float = valid_chunk.to(torch.float32)
            valid_before = valid_float.cumsum(dim=1) - valid_float
            inverse_decay = rho.pow(-(valid_before + 1.0)).unsqueeze(-1)
            weighted_probabilities = q_chunk.detach() * inverse_decay
            exclusive_weighted = (
                weighted_probabilities.cumsum(dim=1) - weighted_probabilities
            )
            responsibility_trajectory = rho.pow(valid_before).unsqueeze(-1) * (
                nu_old.unsqueeze(1) + exclusive_weighted
            )
            beta_trajectory = (
                self.beta_max
                * responsibility_trajectory
                / (responsibility_trajectory + self.kappa_beta)
            )

        corrected_mixed = (
            self.anchors.unsqueeze(0).unsqueeze(0)
            * (1.0 - beta_trajectory.detach().unsqueeze(2))
            + state_trajectory.detach() * beta_trajectory.detach().unsqueeze(2)
        )
        corrected_prototypes = self._stable_l2(
            corrected_mixed, dim=2, eps=self.eps_m
        )
        corrected_logits = torch.einsum(
            "btd,btdk->btk", z_chunk, corrected_prototypes
        ) / self.prototype_temperature
        q_corrected = F.softmax(corrected_logits, dim=-1, dtype=torch.float32)
        return q_corrected * valid_chunk.unsqueeze(-1)

    def expert_kernel(self) -> torch.Tensor:
        """Return B in R^{K x N}, row-normalized over experts."""
        # ``energy`` stores Theta_C. Right-centering implements C = Theta_C H_N.
        centered_energy = self.energy - self.energy.mean(dim=-1, keepdim=True)
        logits = (
            centered_energy
            - self.congestion_price.detach().unsqueeze(0)
        ) / self.expert_temperature
        return F.softmax(logits, dim=-1, dtype=torch.float32)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> CPTRouterOutput:
        """Route states with parallel token routing inside each state chunk.

        ``attention_mask`` is ``[batch, sequence]`` with True marking a valid
        (routable) token and False marking padding.

        On CUDA during training the implementation is compiled on first use so
        the chunk loop runs as fused kernels.  Compilation unrolls the loop, so
        it is skipped when the chunk count exceeds ``_MAX_COMPILED_CHUNKS``;
        raise ``cpt_state_chunk_size`` to keep long sequences on the fast path.
        CPU and eval paths stay eager for determinism and to avoid recompiling
        on variable lengths.
        """
        if (
            hidden_states.is_cuda
            and self.training
            and self._should_compile(hidden_states.shape[1])
        ):
            if self._compiled_forward_impl is None:
                self._compiled_forward_impl = torch.compile(self._forward_impl)
            return self._compiled_forward_impl(hidden_states, attention_mask)
        return self._forward_impl(hidden_states, attention_mask)

    def _should_compile(self, sequence_length: int) -> bool:
        num_chunks = (sequence_length + self.state_chunk_size - 1) // (
            self.state_chunk_size
        )
        return num_chunks <= _MAX_COMPILED_CHUNKS

    def _forward_impl(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> CPTRouterOutput:
        if hidden_states.ndim != 3:
            raise ValueError("hidden_states must have shape [batch, sequence, hidden]")
        batch_size, sequence_length, hidden_size = hidden_states.shape
        if hidden_size != self.hidden_size:
            raise ValueError(
                f"hidden size mismatch: expected {self.hidden_size}, got {hidden_size}"
            )
        if sequence_length <= 0:
            raise ValueError("sequence length must be positive")
        if attention_mask is None:
            valid_mask = torch.ones(
                batch_size,
                sequence_length,
                device=hidden_states.device,
                dtype=torch.bool,
            )
        else:
            if not isinstance(attention_mask, torch.Tensor):
                raise TypeError("attention_mask must be a tensor")
            if tuple(attention_mask.shape) != (batch_size, sequence_length):
                raise ValueError(
                    "attention_mask must have shape "
                    f"{(batch_size, sequence_length)}"
                )
            valid_mask = attention_mask.to(
                device=hidden_states.device,
                dtype=torch.bool,
            )

        with torch.autocast(device_type=hidden_states.device.type, enabled=False):
            hidden_fp32 = torch.where(
                valid_mask.unsqueeze(-1),
                hidden_states.float(),
                torch.zeros_like(hidden_states, dtype=torch.float32),
            )
            # Row-major code: x_t^T P^T, equivalent to column-vector P x_t.
            projected = F.linear(hidden_fp32, self.projection)
            projected = self._stable_l2(projected, dim=-1, eps=self.eps_z)
            kernel = self.expert_kernel()

            short_state = torch.zeros(
                batch_size,
                self.projection_dim,
                self.num_prototypes,
                device=hidden_states.device,
                dtype=torch.float32,
            )
            responsibility = torch.zeros(
                batch_size,
                self.num_prototypes,
                device=hidden_states.device,
                dtype=torch.float32,
            )
            prototype_probability_chunks: list[torch.Tensor] = []
            rho = responsibility.new_tensor(self.rho_beta)

            for chunk_start in range(0, sequence_length, self.state_chunk_size):
                chunk_end = min(chunk_start + self.state_chunk_size, sequence_length)
                q_chunk, short_state, responsibility = self._route_chunk(
                    projected[:, chunk_start:chunk_end],
                    valid_mask[:, chunk_start:chunk_end],
                    short_state,
                    responsibility,
                    rho,
                )
                prototype_probability_chunks.append(q_chunk)

            nominal_prototype_probabilities = torch.cat(
                prototype_probability_chunks,
                dim=1,
            )
            flat_prototype_probabilities = nominal_prototype_probabilities.reshape(
                -1,
                self.num_prototypes,
            )
            valid_token_indices = torch.nonzero(
                valid_mask.reshape(-1),
                as_tuple=False,
            ).flatten()
            # Stable gather forms Q^T:[T_valid,K].  This single 2D product is
            # exactly the transpose of the column-vector formula Pi = B^T Q.
            prototype_probabilities = flat_prototype_probabilities.index_select(
                0,
                valid_token_indices,
            )
            valid_probabilities = prototype_probabilities @ kernel
            flat_probabilities = torch.zeros(
                batch_size * sequence_length,
                self.num_experts,
                device=hidden_states.device,
                dtype=torch.float32,
            )
            probabilities = flat_probabilities.index_copy(
                0,
                valid_token_indices,
                valid_probabilities,
            ).view(batch_size, sequence_length, self.num_experts)
            load_sum = valid_probabilities.detach().sum(dim=0)
            token_count = valid_mask.sum(dtype=torch.int64)

        proposal = CPTLayerProposal(
            layer_index=self.layer_index,
            load_sum=load_sum.detach().clone(),
            token_count=token_count.detach().clone(),
            state_version=self.state_version.detach().clone(),
        )
        return CPTRouterOutput(probabilities=probabilities, proposal=proposal)

    def validate_proposal(self, proposal: CPTLayerProposal) -> None:
        if not isinstance(proposal, CPTLayerProposal):
            raise TypeError("CPT proposal has the wrong type")
        if isinstance(proposal.layer_index, bool) or not isinstance(
            proposal.layer_index,
            int,
        ):
            raise TypeError("CPT proposal layer_index must be an integer")
        if proposal.layer_index != self.layer_index:
            raise RuntimeError("CPT proposal layer index mismatch")
        if not isinstance(proposal.load_sum, torch.Tensor):
            raise TypeError("CPT load_sum must be a tensor")
        if tuple(proposal.load_sum.shape) != (self.num_experts,):
            raise RuntimeError("CPT load_sum has the wrong shape")
        if proposal.load_sum.dtype != torch.float32:
            raise RuntimeError("CPT load_sum must be FP32")
        if proposal.load_sum.device != self.congestion_price.device:
            raise RuntimeError("CPT load_sum is on the wrong device")
        if proposal.load_sum.requires_grad:
            raise RuntimeError("CPT load_sum must be detached")
        if not bool(torch.isfinite(proposal.load_sum).all()):
            raise RuntimeError("CPT load_sum is non-finite")
        if bool((proposal.load_sum < 0).any()):
            raise RuntimeError("CPT load_sum must be non-negative")

        for name, value in (
            ("token_count", proposal.token_count),
            ("state_version", proposal.state_version),
        ):
            if not isinstance(value, torch.Tensor) or value.shape != torch.Size([]):
                raise RuntimeError(f"CPT {name} must be a scalar tensor")
            if value.dtype != torch.int64:
                raise RuntimeError(f"CPT {name} must be int64")
            if value.device != self.state_version.device:
                raise RuntimeError(f"CPT {name} is on the wrong device")
            if value.requires_grad:
                raise RuntimeError(f"CPT {name} must be detached")
        count = int(proposal.token_count.item())
        if count < 0:
            raise RuntimeError("CPT token_count must be non-negative")
        if not torch.equal(proposal.state_version, self.state_version):
            raise RuntimeError("stale or mismatched CPT state version")
        expected_mass = float(count)
        actual_mass = float(proposal.load_sum.sum().item())
        tolerance = max(1e-5, 2e-5 * max(1, count))
        if abs(actual_mass - expected_mass) > tolerance:
            raise RuntimeError("CPT soft-load mass does not equal token_count")

    @torch.no_grad()
    def prepare_commit(
        self,
        proposal: CPTLayerProposal,
        optimizer_step: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        self.validate_proposal(proposal)
        if type(optimizer_step) is not int or optimizer_step < 0:
            raise TypeError("CPT optimizer_step must be a non-negative integer")
        current_optimizer_step = int(self.optimizer_step.item())
        if optimizer_step not in (
            current_optimizer_step,
            current_optimizer_step + 1,
        ):
            raise RuntimeError(
                "CPT optimizer_step must stay unchanged or advance by one"
            )
        for name, parameter in (
            ("projection", self.projection),
            ("anchors", self.anchors),
            ("energy", self.energy),
        ):
            if parameter.dtype != torch.float32:
                raise RuntimeError(f"CPT {name} must remain FP32")
            if not bool(torch.isfinite(parameter).all()):
                raise RuntimeError(f"CPT {name} is non-finite after optimizer.step")

        anchor_norms = torch.linalg.vector_norm(self.anchors, dim=0)
        if not bool(torch.isfinite(anchor_norms).all()) or bool(
            (anchor_norms <= 0).any()
        ):
            raise RuntimeError("CPT anchor columns must be finite and non-zero")
        normalized_anchors = self.anchors / anchor_norms.unsqueeze(0)

        count = int(proposal.token_count.item())
        if count == 0:
            next_price = self.congestion_price.clone()
        else:
            expert_probability = proposal.load_sum / float(count)
            capacity = self.capacity_factor / self.num_experts
            next_price = torch.clamp_min(
                self.congestion_price
                + self.price_learning_rate * (expert_probability - capacity),
                0.0,
            )
        if not bool(torch.isfinite(next_price).all()) or bool((next_price < 0).any()):
            raise RuntimeError("invalid CPT congestion-price candidate")
        if int(self.state_version.item()) == torch.iinfo(torch.int64).max:
            raise RuntimeError("CPT state_version overflow")
        next_version = self.state_version + 1
        if optimizer_step > int(next_version.item()):
            raise RuntimeError("CPT optimizer_step exceeds state_version")
        next_optimizer_step = self.optimizer_step.new_tensor(optimizer_step)
        return (
            normalized_anchors.detach().clone(),
            next_price.detach().clone(),
            next_optimizer_step.detach().clone(),
            next_version.detach().clone(),
        )

    @torch.no_grad()
    def apply_commit(
        self,
        normalized_anchors: torch.Tensor,
        next_price: torch.Tensor,
        next_optimizer_step: torch.Tensor,
        next_version: torch.Tensor,
    ) -> None:
        self.anchors.copy_(normalized_anchors)
        self.congestion_price.copy_(next_price)
        self.optimizer_step.copy_(next_optimizer_step)
        self.state_version.copy_(next_version)

    @torch.no_grad()
    def commit_snapshot(
        self,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        return (
            self.anchors.detach().clone(),
            self.congestion_price.detach().clone(),
            self.optimizer_step.detach().clone(),
            self.state_version.detach().clone(),
        )

    @torch.no_grad()
    def restore_commit_snapshot(
        self,
        snapshot: tuple[
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
        ],
    ) -> None:
        anchors, price, optimizer_step, version = snapshot
        self.anchors.copy_(anchors)
        self.congestion_price.copy_(price)
        self.optimizer_step.copy_(optimizer_step)
        self.state_version.copy_(version)

    @torch.no_grad()
    def validate_persistent_state(self, require_unit_anchors: bool) -> None:
        expected_parameters = {
            "projection": (self.projection_dim, self.hidden_size),
            "anchors": (self.projection_dim, self.num_prototypes),
            "energy": (self.num_prototypes, self.num_experts),
        }
        for name, shape in expected_parameters.items():
            value = getattr(self, name)
            if tuple(value.shape) != shape or value.dtype != torch.float32:
                raise RuntimeError(f"invalid CPT {name} shape or dtype")
            if not bool(torch.isfinite(value).all()):
                raise RuntimeError(f"CPT {name} is non-finite")
        anchor_norms = torch.linalg.vector_norm(self.anchors, dim=0)
        if bool((anchor_norms <= 0).any()) or not bool(
            torch.isfinite(anchor_norms).all()
        ):
            raise RuntimeError("CPT anchor columns must be finite and non-zero")
        if require_unit_anchors and not torch.allclose(
            anchor_norms,
            torch.ones_like(anchor_norms),
            atol=5e-5,
            rtol=5e-5,
        ):
            raise RuntimeError("CPT anchor columns must have unit L2 norm")
        if (
            tuple(self.congestion_price.shape) != (self.num_experts,)
            or self.congestion_price.dtype != torch.float32
            or self.congestion_price.requires_grad
            or not bool(torch.isfinite(self.congestion_price).all())
            or bool((self.congestion_price < 0).any())
        ):
            raise RuntimeError("invalid CPT congestion_price")
        for name in (
            "optimizer_step",
            "state_version",
            "router_algorithm_version",
        ):
            value = getattr(self, name)
            if value.shape != torch.Size([]) or value.dtype != torch.int64:
                raise RuntimeError(f"invalid CPT {name}")
        optimizer_step = int(self.optimizer_step.item())
        if optimizer_step < 0:
            raise RuntimeError("CPT optimizer_step must be non-negative")
        if int(self.state_version.item()) < 0:
            raise RuntimeError("CPT state_version must be non-negative")
        if optimizer_step > int(self.state_version.item()):
            raise RuntimeError("CPT optimizer_step exceeds state_version")
        if int(self.router_algorithm_version.item()) != CPT_ROUTER_ALGORITHM_VERSION:
            raise RuntimeError("unsupported CPT Router algorithm version")

    def validate_config_binding(self, config: TinyMixtralConfig) -> None:
        """Reject saving a config that no longer describes this live Router."""
        architecture = {
            field: getattr(config, field)
            for field in CPT_ROUTING_ARCHITECTURE_FIELDS
        }
        changed_architecture = sorted(
            field
            for field in self._architecture_snapshot.keys() | architecture.keys()
            if field not in self._architecture_snapshot
            or field not in architecture
            or type(architecture[field]) is not type(
                self._architecture_snapshot[field]
            )
            or architecture[field] != self._architecture_snapshot[field]
        )
        if changed_architecture:
            raise RuntimeError(
                f"CPT Router layer {self.layer_index} architecture config changed "
                "after construction: " + ", ".join(changed_architecture)
            )

        current = config.cpt_config_dict()
        changed = sorted(
            key
            for key in self._cpt_config_snapshot.keys() | current.keys()
            if key not in self._cpt_config_snapshot
            or key not in current
            or type(current[key]) is not type(self._cpt_config_snapshot[key])
            or current[key] != self._cpt_config_snapshot[key]
        )
        if changed:
            raise RuntimeError(
                f"CPT Router layer {self.layer_index} config changed after "
                "construction: " + ", ".join(changed)
            )

    def trainable_parameters(self) -> tuple[nn.Parameter, nn.Parameter, nn.Parameter]:
        return self.projection, self.anchors, self.energy
