# Copyright (C) Michael Lee (李登淳) 2026. All rights reserved.
# Open-source under the MIT License. See LICENSE for details.

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Optional

from safetensors import safe_open
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from transformers import PreTrainedModel
from transformers.generation import GenerationMixin
from transformers.modeling_utils import load_state_dict as _load_checkpoint_state_dict
from transformers.utils import ModelOutput
from transformers.utils import (
    SAFE_WEIGHTS_INDEX_NAME,
    SAFE_WEIGHTS_NAME,
    WEIGHTS_INDEX_NAME,
    WEIGHTS_NAME,
)

from .configuration_tinymixtral import TinyMixtralConfig

try:
    from .cpt_router import (
        CPT_ROUTER_ALGORITHM_VERSION,
        CPTLayerProposal,
        CPTModelTransactionMixin,
        CPTRouter,
        CPTSequenceState,
        CPTTransaction,
        normalize_binary_mask,
        normalize_sequence_ids,
        validate_serialized_router_algorithm_version,
    )
except ModuleNotFoundError as error:
    expected_local_module = f"{__package__}.cpt_router"
    if error.name != expected_local_module:
        raise
    from model.cpt_router import (
        CPT_ROUTER_ALGORITHM_VERSION,
        CPTLayerProposal,
        CPTModelTransactionMixin,
        CPTRouter,
        CPTSequenceState,
        CPTTransaction,
        normalize_binary_mask,
        normalize_sequence_ids,
        validate_serialized_router_algorithm_version,
    )


_INTEGER_DTYPES = {
    torch.uint8,
    torch.int8,
    torch.int16,
    torch.int32,
    torch.int64,
}

_CPT_PERSISTENT_STATE_NAMES = (
    "projection",
    "anchors",
    "energy",
    "congestion_price",
    "state_version",
    "router_algorithm_version",
)
_LEGACY_ROUTER_KEY_SUFFIX = ".moe.router.weight"
_ROUTER_ALGORITHM_VERSION_SUFFIX = (
    ".moe.cpt_router.router_algorithm_version"
)


def _validate_router_algorithm_entries(state_dict, *, context: str) -> None:
    if not isinstance(state_dict, dict):
        raise RuntimeError(f"{context} is not a tensor state dictionary")
    for key, value in state_dict.items():
        if str(key).endswith(_ROUTER_ALGORITHM_VERSION_SUFFIX):
            validate_serialized_router_algorithm_version(
                value,
                key=str(key),
                expected=CPT_ROUTER_ALGORITHM_VERSION,
            )


def _variant_filename(filename: str, variant: Optional[str]) -> str:
    if not variant:
        return filename
    stem, suffix = filename.rsplit(".", 1)
    return f"{stem}.{variant}.{suffix}"


def _validated_local_shard_path(root: Path, shard_name: str) -> Path:
    relative = Path(shard_name)
    if relative.is_absolute() or ".." in relative.parts:
        raise RuntimeError(
            f"invalid local checkpoint shard path: {shard_name!r}"
        )
    candidate = (root / relative).resolve()
    try:
        candidate.relative_to(root.resolve())
    except ValueError as error:
        raise RuntimeError(
            f"checkpoint shard escapes the local model directory: {shard_name!r}"
        ) from error
    return candidate


def _local_algorithm_checkpoint_files(args, kwargs) -> tuple[Path, ...]:
    source = (
        args[0]
        if args
        else kwargs.get("pretrained_model_name_or_path")
    )
    if source is None:
        return ()
    try:
        source_path = Path(source)
    except TypeError:
        return ()
    if not source_path.exists():
        return ()
    if source_path.is_file():
        return (source_path.resolve(),)

    root = source_path
    subfolder = kwargs.get("subfolder", "")
    if subfolder:
        relative_subfolder = Path(subfolder)
        if relative_subfolder.is_absolute() or ".." in relative_subfolder.parts:
            raise RuntimeError("invalid local checkpoint subfolder")
        root = root / relative_subfolder
    root = root.resolve()
    variant = kwargs.get("variant")
    use_safetensors = kwargs.get("use_safetensors")
    if use_safetensors is False:
        index_names = (WEIGHTS_INDEX_NAME,)
        weight_names = (WEIGHTS_NAME,)
    elif use_safetensors is True:
        index_names = (SAFE_WEIGHTS_INDEX_NAME,)
        weight_names = (SAFE_WEIGHTS_NAME,)
    else:
        index_names = (SAFE_WEIGHTS_INDEX_NAME, WEIGHTS_INDEX_NAME)
        weight_names = (SAFE_WEIGHTS_NAME, WEIGHTS_NAME)

    for base_name in index_names:
        index_path = root / _variant_filename(base_name, variant)
        if not index_path.is_file():
            continue
        payload = json.loads(index_path.read_text(encoding="utf-8"))
        weight_map = payload.get("weight_map")
        if not isinstance(weight_map, dict):
            raise RuntimeError(
                f"invalid checkpoint index weight_map: {index_path}"
            )
        shard_names = sorted(
            {
                str(shard_name)
                for key, shard_name in weight_map.items()
                if str(key).endswith(_ROUTER_ALGORITHM_VERSION_SUFFIX)
            }
        )
        return tuple(
            _validated_local_shard_path(root, name)
            for name in shard_names
        )

    for base_name in weight_names:
        weight_path = root / _variant_filename(base_name, variant)
        if weight_path.is_file():
            return (weight_path.resolve(),)
    return ()


def _preflight_local_router_algorithm_identity(args, kwargs) -> None:
    supplied_state_dict = kwargs.get("state_dict")
    if supplied_state_dict is not None:
        _validate_router_algorithm_entries(
            supplied_state_dict,
            context="supplied Hugging Face state_dict",
        )

    for checkpoint_file in _local_algorithm_checkpoint_files(args, kwargs):
        if checkpoint_file.suffix == ".safetensors":
            with safe_open(
                str(checkpoint_file),
                framework="pt",
                device="cpu",
            ) as handle:
                for key in handle.keys():
                    if key.endswith(_ROUTER_ALGORITHM_VERSION_SUFFIX):
                        validate_serialized_router_algorithm_version(
                            handle.get_tensor(key),
                            key=key,
                            expected=CPT_ROUTER_ALGORITHM_VERSION,
                        )
        else:
            state_dict = _load_checkpoint_state_dict(
                str(checkpoint_file),
                map_location="cpu",
                weights_only=True,
            )
            _validate_router_algorithm_entries(
                state_dict,
                context=f"local checkpoint {checkpoint_file}",
            )


def _resolve_router_recompute(
    mode: str,
    global_recompute: bool,
) -> bool:
    """Resolve the Router checkpoint policy without changing global state."""
    if mode == "global":
        return global_recompute
    if mode == "on":
        return True
    if mode == "off":
        return False
    raise ValueError(
        "cpt_router_recompute must be one of: global, on, off"
    )


def _normalize_segment_ids(
    segment_ids: torch.Tensor,
    expected_shape: tuple[int, int],
    device: torch.device,
    *,
    name: str = "cpt_segment_ids",
) -> torch.Tensor:
    """Validate public packed-sequence ids and canonicalize them to int64."""
    if not isinstance(segment_ids, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if tuple(segment_ids.shape) != expected_shape:
        raise ValueError(
            f"{name} must have shape {expected_shape}, got "
            f"{tuple(segment_ids.shape)}"
        )
    if segment_ids.dtype not in _INTEGER_DTYPES:
        raise TypeError(f"{name} must use an integer dtype")
    if segment_ids.is_meta:
        raise TypeError(f"{name} must be materialized")
    return segment_ids.to(device=device, dtype=torch.int64)


def _segment_transition_mask(
    route_valid_mask: torch.Tensor,
    segment_ids: torch.Tensor,
) -> torch.Tensor:
    """Mark in-micro-batch packed-segment transitions after the first token."""
    batch_size, seq_len = route_valid_mask.shape
    if seq_len == 0:
        return torch.zeros_like(route_valid_mask)

    positions = torch.arange(
        seq_len,
        device=route_valid_mask.device,
        dtype=torch.int64,
    ).expand(batch_size, -1)
    valid_positions = torch.where(
        route_valid_mask,
        positions,
        torch.full_like(positions, -1),
    )
    last_valid_at_or_before = valid_positions.cummax(dim=1).values
    previous_valid_index = torch.cat(
        (
            torch.full(
                (batch_size, 1),
                -1,
                device=route_valid_mask.device,
                dtype=torch.int64,
            ),
            last_valid_at_or_before[:, :-1],
        ),
        dim=1,
    )
    has_previous = previous_valid_index >= 0
    previous_segment = segment_ids.gather(
        1,
        previous_valid_index.clamp_min(0),
    )
    invalid_order = (
        route_valid_mask
        & has_previous
        & (segment_ids < previous_segment)
    )
    if torch.any(invalid_order):
        raise ValueError(
            "cpt_segment_ids must be nondecreasing across valid tokens "
            "within each batch row"
        )
    return (
        route_valid_mask
        & has_previous
        & (segment_ids != previous_segment)
    )


def _prepare_causal_lm_loss_tensors(
    logits: torch.Tensor,
    labels: torch.Tensor,
    route_valid_mask: torch.Tensor,
    segment_ids: Optional[torch.Tensor],
    label_segment_ids: Optional[torch.Tensor],
    *,
    labels_are_pre_shifted: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply the explicit label contract and mask invalid/cross-segment targets."""
    if not isinstance(labels_are_pre_shifted, bool):
        raise TypeError("labels_are_pre_shifted must be a boolean")
    if not isinstance(labels, torch.Tensor):
        raise TypeError("labels must be a torch.Tensor")
    expected_shape = tuple(logits.shape[:2])
    if tuple(labels.shape) != expected_shape:
        raise ValueError(
            f"labels must have shape {expected_shape}, got {tuple(labels.shape)}"
        )
    if labels.dtype not in _INTEGER_DTYPES:
        raise TypeError("labels must use an integer dtype")
    if labels.is_meta:
        raise TypeError("labels must be materialized")
    labels = labels.to(device=logits.device, dtype=torch.int64)
    invalid_label_values = (labels != -100) & (
        (labels < 0) | (labels >= logits.shape[-1])
    )
    if torch.any(invalid_label_values):
        raise ValueError(
            "labels must contain only -100 or token ids in "
            f"[0, {logits.shape[-1]})"
        )

    normalized_label_segments = None
    if label_segment_ids is not None:
        if segment_ids is None:
            raise ValueError(
                "cpt_label_segment_ids requires cpt_segment_ids"
            )
        normalized_label_segments = _normalize_segment_ids(
            label_segment_ids,
            expected_shape,
            logits.device,
            name="cpt_label_segment_ids",
        )

    if labels_are_pre_shifted:
        loss_logits = logits
        effective_labels = labels.clone()
        valid_targets = route_valid_mask.clone()
        if segment_ids is not None:
            if normalized_label_segments is not None:
                valid_targets &= segment_ids == normalized_label_segments
            else:
                inferred = torch.zeros_like(route_valid_mask)
                if route_valid_mask.shape[1] > 1:
                    inferred[:, :-1] = (
                        route_valid_mask[:, :-1]
                        & route_valid_mask[:, 1:]
                        & (segment_ids[:, :-1] == segment_ids[:, 1:])
                    )
                valid_targets &= inferred
        effective_labels.masked_fill_(~valid_targets, -100)
    else:
        if logits.shape[1] < 2:
            raise ValueError(
                "standard causal-LM labels require at least two token positions"
            )
        loss_logits = logits[:, :-1, :].contiguous()
        effective_labels = labels[:, 1:].contiguous().clone()
        valid_targets = (
            route_valid_mask[:, :-1]
            & route_valid_mask[:, 1:]
        )
        if segment_ids is not None:
            target_segments = (
                normalized_label_segments[:, 1:]
                if normalized_label_segments is not None
                else segment_ids[:, 1:]
            )
            valid_targets &= segment_ids[:, :-1] == target_segments
        effective_labels.masked_fill_(~valid_targets, -100)

    if not torch.any(effective_labels != -100):
        raise ValueError("labels contain no valid next-token targets")
    return loss_logits, effective_labels


# ============================================================
# Layers
# ============================================================

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x):
        dtype = x.dtype
        x = x.float()
        norm = x.pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(norm + self.eps)
        return (x * self.weight).to(dtype)


class RotaryEmbedding(nn.Module):
    def __init__(self, dim, max_position_embeddings=2048, theta=10000.0):
        super().__init__()
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.theta = theta
        self._build_cache()

    def _build_cache(self):
        inv_freq = 1.0 / (self.theta ** (torch.arange(0, self.dim, 2).float() / self.dim))
        t = torch.arange(self.max_position_embeddings).float()
        freqs = torch.outer(t, inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos(), persistent=False)
        self.register_buffer("sin_cached", emb.sin(), persistent=False)

    def forward(self, x, position_ids):
        cos = self.cos_cached[position_ids].unsqueeze(1)
        sin = self.sin_cached[position_ids].unsqueeze(1)
        x_rot = x.float()
        x1, x2 = x_rot.chunk(2, dim=-1)
        rotated = torch.cat((-x2, x1), dim=-1)
        return (x_rot * cos + rotated * sin).to(x.dtype)


class GQAAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.head_dim = config.head_dim
        self.num_groups = self.num_heads // self.num_kv_heads
        assert self.num_heads % self.num_kv_heads == 0

        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(self.hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(self.hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False)
        self.rotary_emb = RotaryEmbedding(self.head_dim, config.max_position_embeddings, config.rope_theta)
        self.attention_dropout = config.attention_dropout

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        position_ids=None,
        segment_ids=None,
    ):
        B, S, _ = hidden_states.shape
        q = self.q_proj(hidden_states).view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(hidden_states).view(B, S, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(hidden_states).view(B, S, self.num_kv_heads, self.head_dim).transpose(1, 2)
        if position_ids is None:
            position_ids = torch.arange(S, device=hidden_states.device).unsqueeze(0).expand(B, -1)
        q, k = self.rotary_emb(q, position_ids), self.rotary_emb(k, position_ids)

        if attention_mask is not None or segment_ids is not None:
            k_exp = (
                k.unsqueeze(2)
                .expand(-1, -1, self.num_groups, -1, -1)
                .reshape(B, self.num_heads, S, self.head_dim)
            )
            v_exp = (
                v.unsqueeze(2)
                .expand(-1, -1, self.num_groups, -1, -1)
                .reshape(B, self.num_heads, S, self.head_dim)
            )
            causal = torch.tril(torch.ones(S, S, device=hidden_states.device, dtype=torch.bool))
            combined = causal[None, None, :, :]
            if attention_mask is not None:
                if tuple(attention_mask.shape) != (B, S):
                    raise ValueError(
                        f"attention_mask must have shape {(B, S)}, got "
                        f"{tuple(attention_mask.shape)}"
                    )
                attention_mask = attention_mask.to(
                    device=hidden_states.device,
                    dtype=torch.bool,
                )
                combined = combined & attention_mask[:, None, None, :]
            if segment_ids is not None:
                if tuple(segment_ids.shape) != (B, S):
                    raise ValueError(
                        f"segment_ids must have shape {(B, S)}, got "
                        f"{tuple(segment_ids.shape)}"
                    )
                segment_ids = segment_ids.to(hidden_states.device)
                same_segment = (
                    segment_ids[:, None, :, None]
                    == segment_ids[:, None, None, :]
                )
                combined = combined & same_segment
            attn = F.scaled_dot_product_attention(
                q, k_exp, v_exp, attn_mask=combined,
                dropout_p=self.attention_dropout if self.training else 0.0,
                is_causal=False,
            )
        else:
            attn = F.scaled_dot_product_attention(
                q, k, v, attn_mask=None,
                dropout_p=self.attention_dropout if self.training else 0.0,
                is_causal=True,
                enable_gqa=True,
            )
        return self.o_proj(attn.transpose(1, 2).reshape(B, S, -1))


class SparseMoE(nn.Module):
    """Mixtral experts fed by CPT probabilities for route-valid tokens only."""

    def __init__(self, config, layer_index: int):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_experts = config.num_local_experts
        self.top_k = config.num_experts_per_tok
        self.expert_intermediate = config.expert_intermediate_size
        self.cpt_router = CPTRouter(config, layer_index=layer_index)
        self.gate_proj = nn.Parameter(
            torch.empty(
                self.num_experts,
                self.expert_intermediate,
                self.hidden_size,
            )
        )
        self.up_proj = nn.Parameter(
            torch.empty(
                self.num_experts,
                self.expert_intermediate,
                self.hidden_size,
            )
        )
        self.down_proj = nn.Parameter(
            torch.empty(
                self.num_experts,
                self.hidden_size,
                self.expert_intermediate,
            )
        )
        self._init_weights(config.initializer_range)

    def _init_weights(self, std=0.02):
        nn.init.normal_(self.gate_proj, std=std)
        nn.init.normal_(self.up_proj, std=std)
        nn.init.normal_(self.down_proj, std=std)

    def _router_forward_tensors(
        self,
        x,
        route_valid_mask,
        reset_mask,
        sequence_state,
        cpt_sequence_ids,
    ):
        """Return the Router result as a checkpoint-safe tensor tuple."""
        batch_size = x.shape[0]
        router_kwargs = {
            "route_valid_mask": route_valid_mask,
            "reset_mask": reset_mask,
        }
        if sequence_state is not None:
            router_kwargs["sequence_state"] = sequence_state
        if cpt_sequence_ids is not None:
            router_kwargs["cpt_sequence_ids"] = cpt_sequence_ids
        router_output = self.cpt_router(x, **router_kwargs)
        sequence_state_out = router_output.sequence_state
        if sequence_state_out is None:
            sequence_state_out = self.cpt_router._new_sequence_state(
                batch_size,
                x.device,
                sequence_ids=cpt_sequence_ids,
            )
        proposal = router_output.proposal
        return (
            router_output.probabilities,
            router_output.flat_valid_indices,
            proposal.load_sum,
            proposal.token_count,
            proposal.state_version,
            proposal.valid,
            sequence_state_out.state_s,
            sequence_state_out.state_nu,
            sequence_state_out.initialized,
            sequence_state_out.state_version,
            (
                sequence_state_out.sequence_ids
                if sequence_state_out.sequence_ids is not None
                else torch.empty(0, device=x.device, dtype=torch.int64)
            ),
        )

    def _expert_forward(self, x, probabilities, flat_valid_indices):
        """Apply Top-k dispatch and experts without calling the Router."""
        batch_size, seq_len, hidden_size = x.shape
        if flat_valid_indices.numel() == 0:
            return torch.zeros_like(x)

        x_flat = x.reshape(-1, hidden_size)

        selected_weights, selected_experts = torch.topk(
            probabilities,
            self.top_k,
            dim=-1,
        )
        # Renormalize only selected combine weights; keep full Pi unchanged for
        # the detached optimizer-step price transaction.
        selected_weights = selected_weights / selected_weights.sum(
            dim=-1,
            keepdim=True,
        )

        flat_experts = selected_experts.reshape(-1)
        flat_weights = selected_weights.reshape(-1)
        flat_token_indices = (
            flat_valid_indices[:, None]
            .expand(-1, self.top_k)
            .reshape(-1)
        )
        sorted_indices = flat_experts.argsort(stable=True)
        sorted_token_indices = flat_token_indices[sorted_indices]
        sorted_weights = flat_weights[sorted_indices]
        sorted_experts = flat_experts[sorted_indices]
        expert_counts = torch.bincount(
            sorted_experts,
            minlength=self.num_experts,
        ).tolist()

        output_flat = torch.zeros_like(x_flat)
        start = 0
        for expert in range(self.num_experts):
            count = expert_counts[expert]
            if count == 0:
                continue
            end = start + count
            token_indices = sorted_token_indices[start:end]
            weight = sorted_weights[start:end]
            token_states = x_flat.index_select(0, token_indices)
            gate = F.silu(token_states @ self.gate_proj[expert].T)
            up = token_states @ self.up_proj[expert].T
            expert_output = (gate * up) @ self.down_proj[expert].T
            weighted_output = expert_output * weight.to(
                expert_output.dtype
            ).unsqueeze(-1)
            output_flat.index_add_(
                0,
                token_indices,
                weighted_output.to(output_flat.dtype),
            )
            start = end

        return output_flat.reshape(batch_size, seq_len, hidden_size)

    def forward(
        self,
        x,
        route_valid_mask=None,
        reset_mask=None,
        sequence_state=None,
        cpt_sequence_ids=None,
        return_sequence_state=False,
        checkpoint_router=False,
        checkpoint_experts=False,
    ):
        checkpoint_active = self.training and torch.is_grad_enabled()
        router_sequence_ids = (
            None
            if cpt_sequence_ids is None
            else normalize_sequence_ids(
                cpt_sequence_ids,
                x.shape[0],
                x.device,
            )
        )
        router_sequence_state = sequence_state
        if checkpoint_router and checkpoint_active:
            if sequence_state is not None:
                # Non-Tensor checkpoint arguments are not protected by
                # autograd version counters. Keep a private canonical snapshot
                # so caller mutation between forward and backward cannot alter
                # Router recomputation.
                router_sequence_state = (
                    self.cpt_router._validate_sequence_state(
                        sequence_state,
                        batch_size=x.shape[0],
                        device=x.device,
                    )
                )
            router_tensors = checkpoint(
                self._router_forward_tensors,
                x,
                route_valid_mask,
                reset_mask,
                router_sequence_state,
                router_sequence_ids,
                use_reentrant=False,
                preserve_rng_state=True,
            )
        else:
            router_tensors = self._router_forward_tensors(
                x,
                route_valid_mask,
                reset_mask,
                sequence_state,
                router_sequence_ids,
            )
        (
            probabilities,
            flat_valid_indices,
            load_sum,
            token_count,
            state_version,
            proposal_valid,
            state_s,
            state_nu,
            state_initialized,
            sequence_state_version,
            sequence_state_ids,
        ) = router_tensors

        if (
            checkpoint_experts
            and checkpoint_active
            and flat_valid_indices.numel() > 0
        ):
            expert_output = checkpoint(
                self._expert_forward,
                x,
                probabilities,
                flat_valid_indices,
                use_reentrant=False,
                preserve_rng_state=True,
            )
        else:
            expert_output = self._expert_forward(
                x,
                probabilities,
                flat_valid_indices,
            )
        result = (
            expert_output,
            load_sum,
            token_count,
            state_version,
            proposal_valid,
            state_s,
            state_nu,
            state_initialized,
            sequence_state_version,
            sequence_state_ids,
        )
        return result if return_sequence_state else result[:5]


class MoETransformerBlock(nn.Module):
    def __init__(self, config, layer_index: int):
        super().__init__()
        self.input_layernorm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.self_attn = GQAAttention(config)
        self.moe = SparseMoE(config, layer_index=layer_index)

    def _attention_forward(
        self,
        hidden_states,
        attention_mask,
        position_ids,
        segment_ids,
    ):
        return self.self_attn(
            self.input_layernorm(hidden_states),
            attention_mask,
            position_ids,
            segment_ids,
        )

    def forward(
        self,
        x,
        attention_mask=None,
        position_ids=None,
        route_valid_mask=None,
        reset_mask=None,
        segment_ids=None,
        sequence_state=None,
        cpt_sequence_ids=None,
        return_sequence_state=False,
        checkpoint_attention=False,
        checkpoint_router=False,
        checkpoint_experts=False,
    ):
        residual = x
        checkpoint_active = self.training and torch.is_grad_enabled()
        if checkpoint_attention and checkpoint_active:
            attention_output = checkpoint(
                self._attention_forward,
                x,
                attention_mask,
                position_ids,
                segment_ids,
                use_reentrant=False,
                preserve_rng_state=True,
            )
        else:
            attention_output = self._attention_forward(
                x,
                attention_mask,
                position_ids,
                segment_ids,
            )
        x = residual + attention_output
        (
            h,
            load_sum,
            token_count,
            state_version,
            proposal_valid,
            state_s,
            state_nu,
            state_initialized,
            sequence_state_version,
            sequence_state_ids,
        ) = self.moe(
            self.post_attention_layernorm(x),
            route_valid_mask=route_valid_mask,
            reset_mask=reset_mask,
            sequence_state=sequence_state,
            cpt_sequence_ids=cpt_sequence_ids,
            return_sequence_state=True,
            checkpoint_router=checkpoint_router,
            checkpoint_experts=checkpoint_experts,
        )
        result = (
            x + h,
            load_sum,
            token_count,
            state_version,
            proposal_valid,
            state_s,
            state_nu,
            state_initialized,
            sequence_state_version,
            sequence_state_ids,
        )
        return result if return_sequence_state else result[:5]


# ============================================================
# Causal LM
# ============================================================

@dataclass
class TinyMixtralCausalLMOutput(ModelOutput):
    loss: Optional[torch.Tensor] = None
    logits: Optional[torch.Tensor] = None
    cpt_transaction: Optional[CPTTransaction] = None
    cpt_sequence_states: Optional[tuple[CPTSequenceState, ...]] = None


class TinyMixtralForCausalLM(
    CPTModelTransactionMixin,
    PreTrainedModel,
    GenerationMixin,
):
    config_class = TinyMixtralConfig
    base_model_prefix = "tinymixtral"
    supports_gradient_checkpointing = True
    _no_split_modules = ["MoETransformerBlock"]
    # Tell Transformers/safetensors that the duplicate state-dict entry is an
    # intentional embedding tie rather than an undeclared shared tensor.
    _tied_weights_keys = ["lm_head.weight"]
    # The model body follows the requested/checkpoint compute dtype, while the
    # complete CPT Router state is a fixed FP32 numerical island.
    _keep_in_fp32_modules_strict = ["cpt_router"]

    def __init__(self, config):
        super().__init__(config)
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList(
            [
                MoETransformerBlock(config, layer_index=layer_index)
                for layer_index in range(config.num_hidden_layers)
            ]
        )
        self.norm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        if config.tie_word_embeddings:
            self.lm_head.weight = self.embed_tokens.weight
        self._use_activation_checkpointing = False
        self.post_init()

    def save_pretrained(self, *args, **kwargs):
        """Refuse to publish a live model poisoned by a fail-stop error."""
        self._assert_not_training_poisoned(operation="save model")
        return super().save_pretrained(*args, **kwargs)

    def _init_weights(self, module):
        std = self.config.initializer_range
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)

    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs=None):
        self._use_activation_checkpointing = True

    def gradient_checkpointing_disable(self):
        self._use_activation_checkpointing = False

    def set_router_recompute(self, mode: str) -> None:
        """Set the Router recompute policy without changing the global flag."""
        if not isinstance(mode, str):
            raise ValueError("cpt_router_recompute must be a string")
        _resolve_router_recompute(mode, False)
        self.config.cpt_router_recompute = mode

    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        """Load a complete CPT checkpoint and audit its loading report.

        Transformers normally treats missing and unexpected weights as
        warnings.  That behavior is unsafe for CPT persistent state because a
        missing tensor would retain a valid-looking fresh initialization.
        Always request the loading report internally, reject every CPT
        integrity violation, and then preserve the caller's requested return
        shape.
        """
        _preflight_local_router_algorithm_identity(args, kwargs)
        requested_loading_info = bool(kwargs.get("output_loading_info", False))
        if "dtype" not in kwargs and "torch_dtype" not in kwargs:
            # Transformers otherwise defaults to FP32 even for a saved BF16
            # checkpoint. Preserve the checkpoint body dtype automatically;
            # ``_keep_in_fp32_modules_strict`` retains CPT Router FP32 state.
            kwargs["dtype"] = "auto"
        kwargs["output_loading_info"] = True
        try:
            loaded = super().from_pretrained(*args, **kwargs)
        except RuntimeError as error:
            message = str(error)
            if (
                "CPT" in message
                or ".moe.cpt_router." in message
                or "Legacy Linear-router" in message
            ):
                raise RuntimeError(
                    "CPT checkpoint integrity validation failed during "
                    f"Transformers staged loading: {message}"
                ) from error
            raise
        if not isinstance(loaded, tuple) or len(loaded) != 2:
            raise RuntimeError(
                "Transformers did not return the required checkpoint "
                "loading information"
            )
        model, loading_info = loaded
        cls._validate_cpt_loading_info(model, loading_info)
        for layer_index, router in enumerate(model._cpt_routers()):
            router.validate_persistent_invariants(layer_index=layer_index)
        model.get_cpt_state_version()
        if requested_loading_info:
            return model, loading_info
        return model

    @classmethod
    def _validate_cpt_loading_info(cls, model, loading_info):
        """Reject incomplete, mismatched, or foreign model checkpoint state."""
        if not isinstance(loading_info, dict):
            raise RuntimeError("invalid Transformers checkpoint loading information")
        required_loading_fields = {
            "missing_keys",
            "unexpected_keys",
            "mismatched_keys",
            "error_msgs",
        }
        absent_loading_fields = sorted(
            required_loading_fields.difference(loading_info)
        )
        if absent_loading_fields:
            raise RuntimeError(
                "Transformers checkpoint loading information is incomplete: "
                + ", ".join(absent_loading_fields)
            )

        expected_cpt_keys = {
            key
            for key in model.state_dict()
            if any(
                key.endswith(f".moe.cpt_router.{state_name}")
                for state_name in _CPT_PERSISTENT_STATE_NAMES
            )
        }
        expected_count = (
            int(model.config.num_hidden_layers)
            * len(_CPT_PERSISTENT_STATE_NAMES)
        )
        if len(expected_cpt_keys) != expected_count:
            raise RuntimeError(
                "model CPT state schema is incomplete: expected "
                f"{expected_count} persistent tensors, found "
                f"{len(expected_cpt_keys)}"
            )

        prefix = f"{getattr(model, 'base_model_prefix', '')}."

        def canonical_key(key):
            key = str(key)
            if prefix != "." and key.startswith(prefix):
                return key[len(prefix):]
            return key

        missing_keys = tuple(
            canonical_key(key)
            for key in loading_info.get("missing_keys", ())
        )
        missing_cpt_keys = sorted(expected_cpt_keys.intersection(missing_keys))
        missing_model_keys = sorted(
            set(missing_keys).difference(expected_cpt_keys)
        )

        mismatched_entries = tuple(
            loading_info.get("mismatched_keys", ())
        )
        mismatched_cpt_keys = []
        mismatched_model_keys = []
        for entry in mismatched_entries:
            key = entry[0] if isinstance(entry, (tuple, list)) else entry
            normalized = canonical_key(key)
            if (
                normalized in expected_cpt_keys
                or ".moe.cpt_router." in normalized
            ):
                mismatched_cpt_keys.append(str(key))
            else:
                mismatched_model_keys.append(str(key))

        unexpected_entries = tuple(
            str(key) for key in loading_info.get("unexpected_keys", ())
        )
        unexpected_cpt_keys = sorted(
            key for key in unexpected_entries if ".moe.cpt_router." in key
        )
        legacy_router_keys = sorted(
            key
            for key in unexpected_entries
            if key.endswith(_LEGACY_ROUTER_KEY_SUFFIX)
        )
        unexpected_model_keys = sorted(
            key
            for key in unexpected_entries
            if key not in unexpected_cpt_keys and key not in legacy_router_keys
        )

        error_messages = tuple(
            str(message) for message in loading_info.get("error_msgs", ())
        )
        cpt_error_messages = tuple(
            message
            for message in error_messages
            if "cpt_router" in message
            or _LEGACY_ROUTER_KEY_SUFFIX in message
        )
        other_error_messages = tuple(
            message
            for message in error_messages
            if message not in cpt_error_messages
        )

        violations = []
        if missing_cpt_keys:
            violations.append(
                "missing CPT keys: " + ", ".join(missing_cpt_keys)
            )
        if missing_model_keys:
            violations.append(
                "missing model keys: " + ", ".join(missing_model_keys)
            )
        if mismatched_cpt_keys:
            violations.append(
                "shape-mismatched CPT keys: "
                + ", ".join(sorted(mismatched_cpt_keys))
            )
        if mismatched_model_keys:
            violations.append(
                "shape-mismatched model keys: "
                + ", ".join(sorted(mismatched_model_keys))
            )
        if unexpected_cpt_keys:
            violations.append(
                "unexpected CPT keys: " + ", ".join(unexpected_cpt_keys)
            )
        if legacy_router_keys:
            violations.append(
                "Legacy Linear-router checkpoint keys: "
                + ", ".join(legacy_router_keys)
            )
        if unexpected_model_keys:
            violations.append(
                "unexpected model keys: " + ", ".join(unexpected_model_keys)
            )
        if cpt_error_messages:
            violations.append(
                "CPT loading errors: " + " | ".join(cpt_error_messages)
            )
        if other_error_messages:
            violations.append(
                "model loading errors: " + " | ".join(other_error_messages)
            )
        if violations:
            raise RuntimeError(
                "CPT checkpoint integrity validation failed; "
                + "; ".join(violations)
            )

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value):
        self.embed_tokens = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, value):
        self.lm_head = value

    @staticmethod
    def _extend_generation_control(
        control,
        target_length,
        *,
        repeat_last,
    ):
        if control is None or control.shape[1] >= target_length:
            return control
        extension_length = target_length - control.shape[1]
        if repeat_last:
            fill = control[:, -1:].expand(-1, extension_length)
        else:
            fill = torch.zeros(
                control.shape[0],
                extension_length,
                device=control.device,
                dtype=control.dtype,
            )
        return torch.cat((control, fill), dim=1)

    def prepare_inputs_for_generation(
        self,
        input_ids,
        attention_mask=None,
        **kwargs,
    ):
        """Recompute the full prefix because this model has no KV cache."""
        # GenerationMixin may provision a cache object before the first call.
        # It is deliberately ignored: returning no cache forces every decoding
        # step to run on the complete, growing input prefix.
        segment_ids = self._extend_generation_control(
            kwargs.get("cpt_segment_ids"),
            input_ids.shape[1],
            repeat_last=True,
        )
        reset_mask = self._extend_generation_control(
            kwargs.get("cpt_reset_mask"),
            input_ids.shape[1],
            repeat_last=False,
        )
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "cpt_reset_mask": reset_mask,
            "cpt_segment_ids": segment_ids,
            # Generation always replays the complete prefix and deliberately
            # does not consume continuation state. Stable row IDs therefore
            # provide no safety here; forwarding them would break beam search
            # because GenerationMixin duplicates each ID across beam rows.
            "cpt_sequence_ids": None,
            "use_cache": False,
        }

    def forward(
        self,
        input_ids,
        attention_mask=None,
        labels=None,
        cpt_reset_mask=None,
        cpt_segment_ids=None,
        cpt_label_segment_ids=None,
        cpt_sequence_states=None,
        cpt_sequence_ids=None,
        labels_are_pre_shifted=False,
        return_dict=True,
        **kwargs,
    ):
        """Run the decoder with standard HF CausalLM labels by default.

        ``labels_are_pre_shifted=False`` computes logits at positions
        ``0..S-2`` against labels at ``1..S-1``. Set it to ``True`` only for
        repository-style batches whose labels have already been shifted.
        Packed inputs must provide ``cpt_segment_ids``; optional
        ``cpt_label_segment_ids`` identifies pre-shifted target segments and
        permits a same-segment final target to remain supervised.
        ``cpt_sequence_ids`` may provide one unique stable integer identity per
        batch row. Once identity-aware continuation state is returned, later
        chunks must provide matching IDs unless a changed row explicitly
        resets at its first valid token.
        """
        self._assert_not_training_poisoned(operation="execute model forward")
        if not isinstance(input_ids, torch.Tensor):
            raise TypeError("input_ids must be a torch.Tensor")
        if input_ids.is_meta:
            raise TypeError("input_ids must be materialized")
        if input_ids.ndim != 2:
            raise ValueError("input_ids must have shape [batch_size, seq_len]")
        if input_ids.dtype not in (torch.int32, torch.int64):
            raise TypeError("input_ids must use torch.int32 or torch.int64")
        batch_size, seq_len = input_ids.shape
        if batch_size <= 0:
            raise ValueError("input_ids batch size must be positive")
        if seq_len <= 0:
            raise ValueError("input_ids sequence length must be positive")
        if seq_len > self.config.max_position_embeddings:
            raise ValueError(
                f"sequence length {seq_len} exceeds max_position_embeddings="
                f"{self.config.max_position_embeddings}"
            )
        if cpt_label_segment_ids is not None and labels is None:
            raise ValueError("cpt_label_segment_ids requires labels")
        position_ids = torch.arange(
            seq_len,
            device=input_ids.device,
        ).unsqueeze(0).expand(batch_size, -1)

        router_uses_default_dense_controls = (
            attention_mask is None
            and cpt_segment_ids is None
            and cpt_reset_mask is None
        )
        causal_mask = None
        if attention_mask is not None:
            causal_mask = normalize_binary_mask(
                attention_mask,
                (batch_size, seq_len),
                input_ids.device,
                name="attention_mask",
            )

        route_valid_mask = (
            causal_mask
            if causal_mask is not None
            else torch.ones_like(input_ids, dtype=torch.bool)
        )
        canonical_sequence_ids = (
            None
            if cpt_sequence_ids is None
            else normalize_sequence_ids(
                cpt_sequence_ids,
                batch_size,
                input_ids.device,
            )
        )
        segment_ids = None
        if cpt_segment_ids is not None:
            segment_ids = _normalize_segment_ids(
                cpt_segment_ids,
                (batch_size, seq_len),
                input_ids.device,
            )
            segment_reset_mask = _segment_transition_mask(
                route_valid_mask,
                segment_ids,
            )
        else:
            segment_reset_mask = torch.zeros_like(route_valid_mask)
        if cpt_reset_mask is None:
            reset_mask = segment_reset_mask
        else:
            reset_mask = normalize_binary_mask(
                cpt_reset_mask,
                (batch_size, seq_len),
                input_ids.device,
                name="cpt_reset_mask",
            )
            if torch.any(reset_mask & ~route_valid_mask):
                raise ValueError(
                    "cpt_reset_mask cannot mark an invalid route position"
                )
            reset_mask = reset_mask | segment_reset_mask

        # Preserve the full route mask for loss construction, while passing
        # ``None`` to the Router only when the public inputs prove the canonical
        # dense/no-explicit-reset layout.  Explicit attention, packed-segment,
        # or reset controls retain their strict tensor validation path.
        router_route_valid_mask = (
            None if router_uses_default_dense_controls else route_valid_mask
        )
        router_reset_mask = (
            None if router_uses_default_dense_controls else reset_mask
        )

        if cpt_sequence_states is None:
            input_sequence_states = (None,) * len(self.layers)
        else:
            input_sequence_states = tuple(cpt_sequence_states)
            if len(input_sequence_states) != len(self.layers):
                raise ValueError(
                    f"expected {len(self.layers)} cpt_sequence_states, got "
                    f"{len(input_sequence_states)}"
                )

        checkpoint_active = self.training and torch.is_grad_enabled()
        global_recompute = bool(
            self._use_activation_checkpointing and checkpoint_active
        )
        router_recompute = bool(
            checkpoint_active
            and _resolve_router_recompute(
                self.config.cpt_router_recompute,
                global_recompute,
            )
        )

        hidden_states = self.embed_tokens(input_ids)
        layer_proposals = []
        output_sequence_states = []
        for layer_index, (layer, input_sequence_state) in enumerate(
            zip(self.layers, input_sequence_states)
        ):
            if global_recompute and router_recompute:
                checkpoint_sequence_state = input_sequence_state
                if input_sequence_state is not None:
                    # The whole-layer checkpoint receives sequence state as a
                    # Python object, so snapshot its tensors before checkpoint
                    # can retain the external container for backward replay.
                    checkpoint_sequence_state = (
                        layer.moe.cpt_router._validate_sequence_state(
                            input_sequence_state,
                            batch_size=batch_size,
                            device=hidden_states.device,
                        )
                    )
                (
                    hidden_states,
                    load_sum,
                    token_count,
                    state_version,
                    proposal_valid,
                    state_s,
                    state_nu,
                    state_initialized,
                    sequence_state_version,
                    sequence_state_ids,
                ) = checkpoint(
                    layer,
                    hidden_states,
                    causal_mask,
                    position_ids,
                    router_route_valid_mask,
                    router_reset_mask,
                    segment_ids,
                    checkpoint_sequence_state,
                    canonical_sequence_ids,
                    True,
                    use_reentrant=False,
                    preserve_rng_state=True,
                )
            elif not global_recompute and not router_recompute:
                (
                    hidden_states,
                    load_sum,
                    token_count,
                    state_version,
                    proposal_valid,
                    state_s,
                    state_nu,
                    state_initialized,
                    sequence_state_version,
                    sequence_state_ids,
                ) = layer(
                    hidden_states,
                    causal_mask,
                    position_ids,
                    router_route_valid_mask,
                    router_reset_mask,
                    segment_ids,
                    input_sequence_state,
                    canonical_sequence_ids,
                    True,
                )
            else:
                (
                    hidden_states,
                    load_sum,
                    token_count,
                    state_version,
                    proposal_valid,
                    state_s,
                    state_nu,
                    state_initialized,
                    sequence_state_version,
                    sequence_state_ids,
                ) = layer(
                    hidden_states,
                    causal_mask,
                    position_ids,
                    router_route_valid_mask,
                    router_reset_mask,
                    segment_ids,
                    input_sequence_state,
                    canonical_sequence_ids,
                    True,
                    checkpoint_attention=global_recompute,
                    checkpoint_router=router_recompute,
                    checkpoint_experts=global_recompute,
                )
            layer_proposals.append(
                CPTLayerProposal(
                    layer_index=layer_index,
                    load_sum=load_sum.detach(),
                    token_count=token_count.detach(),
                    state_version=state_version.detach(),
                    valid=proposal_valid.detach(),
                )
            )
            output_sequence_states.append(
                layer.moe.cpt_router._make_sequence_state(
                    state_s=state_s,
                    state_nu=state_nu,
                    initialized=state_initialized,
                    state_version=sequence_state_version,
                    sequence_ids=(
                        None
                        if sequence_state_ids.numel() == 0
                        else sequence_state_ids
                    ),
                )
            )

        logits = self.lm_head(self.norm(hidden_states)).float()

        loss = None
        if labels is not None:
            loss_logits, effective_labels = _prepare_causal_lm_loss_tensors(
                logits,
                labels,
                route_valid_mask,
                segment_ids,
                cpt_label_segment_ids,
                labels_are_pre_shifted=labels_are_pre_shifted,
            )
            loss = F.cross_entropy(
                loss_logits.reshape(-1, loss_logits.size(-1)),
                effective_labels.reshape(-1),
                ignore_index=-100,
            )
        transaction = (
            self.prepare_cpt_transaction(layer_proposals)
            if self.training and torch.is_grad_enabled()
            else None
        )
        if not return_dict:
            output = (
                logits,
                transaction,
                tuple(output_sequence_states),
            )
            return ((loss,) + output) if loss is not None else output
        return TinyMixtralCausalLMOutput(
            loss=loss,
            logits=logits,
            cpt_transaction=transaction,
            cpt_sequence_states=tuple(output_sequence_states),
        )
