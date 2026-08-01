# Copyright (C) Michael Lee (李登淳) 2026. All rights reserved.
# Open-source under the MIT License. See LICENSE for details.

"""TinyMixtral——小型 Mixtral 风格 MoE 因果语言模型。

架构：
- decoder-only, RMSNorm, RoPE, GQA
- Mixtral-style sparse MoE FFN (top-k routing, SwiGLU experts)
- 支持 activation checkpointing, FlashAttention (sdpa)
"""

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from .config import TinyMixtralConfig
from .cpt_router import (
    CPTLayerProposal,
    CPTModelTransactionMixin,
    CPTRouter,
    CPTSequenceState,
    normalize_binary_mask,
    normalize_sequence_ids,
)


_INTEGER_DTYPES = {
    torch.uint8,
    torch.int8,
    torch.int16,
    torch.int32,
    torch.int64,
}


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
                # Pre-shifted labels describe the next token, while public
                # segment ids describe the current input token. Infer every
                # in-window target from the next input position. The final
                # target has no observable segment id, so mask it safely.
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
# RMSNorm
# ============================================================

class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization."""

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        x = x.float()
        norm = x.pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(norm + self.eps)
        return (x * self.weight).to(dtype)


# ============================================================
# RoPE
# ============================================================

class RotaryEmbedding(nn.Module):
    """RoPE 位置编码，使用复数旋转。"""

    def __init__(self, dim: int, max_position_embeddings: int = 2048, theta: float = 10000.0):
        super().__init__()
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.theta = theta
        self._build_cache()

    def _build_cache(self):
        inv_freq = 1.0 / (self.theta ** (torch.arange(0, self.dim, 2).float() / self.dim))
        t = torch.arange(self.max_position_embeddings).float()
        freqs = torch.outer(t, inv_freq)  # [seq_len, dim/2]
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos(), persistent=False)
        self.register_buffer("sin_cached", emb.sin(), persistent=False)

    def forward(self, x: torch.Tensor, position_ids: torch.Tensor):
        """x: [batch, num_heads, seq_len, head_dim]"""
        cos = self.cos_cached[position_ids].unsqueeze(1)  # [B, 1, S, D]
        sin = self.sin_cached[position_ids].unsqueeze(1)
        x_rot = x.float()
        x1, x2 = x_rot.chunk(2, dim=-1)
        rotated = torch.cat((-x2, x1), dim=-1)
        return (x_rot * cos + rotated * sin).to(x.dtype)


# ============================================================
# GQA Attention
# ============================================================

class GQAAttention(nn.Module):
    """Grouped Query Attention with RoPE and FlashAttention (sdpa)."""

    def __init__(self, config: TinyMixtralConfig):
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

        self.rotary_emb = RotaryEmbedding(
            self.head_dim, config.max_position_embeddings, config.rope_theta
        )
        self.attention_dropout = config.attention_dropout

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        segment_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        B, S, _ = hidden_states.shape

        q = self.q_proj(hidden_states).view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(hidden_states).view(B, S, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(hidden_states).view(B, S, self.num_kv_heads, self.head_dim).transpose(1, 2)

        # RoPE
        if position_ids is None:
            position_ids = torch.arange(S, device=hidden_states.device).unsqueeze(0).expand(B, -1)
        q = self.rotary_emb(q, position_ids)
        k = self.rotary_emb(k, position_ids)

        # FlashAttention via sdpa — 必须显式合并 causal + padding mask
        # PyTorch 2.x 不允许 attn_mask 和 is_causal 同时设置
        if attention_mask is not None or segment_ids is not None:
            # Keep compact KV heads until an explicit mask is required. SDPA's
            # masked path does not accept enable_gqa, so expand only here.
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
            # attention_mask: [B, S] bool, True=valid token
            # 构造 4D causal mask 并与 padding 合并
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
                # padding: [B, 1, 1, S] → 控制哪些 key 可见
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
            attn_output = F.scaled_dot_product_attention(
                q, k_exp, v_exp,
                attn_mask=combined,
                dropout_p=self.attention_dropout if self.training else 0.0,
                is_causal=False,
            )
        else:
            attn_output = F.scaled_dot_product_attention(
                q, k, v,
                attn_mask=None,
                dropout_p=self.attention_dropout if self.training else 0.0,
                is_causal=True,
                enable_gqa=True,
            )

        attn_output = attn_output.transpose(1, 2).reshape(B, S, -1)
        return self.o_proj(attn_output)


# ============================================================
# MoE FFN
# ============================================================

class SparseMoE(nn.Module):
    """Mixtral-style Sparse Mixture of Experts FFN。

    每个 token 通过 top-k gating 路由到 k 个 expert。
    Expert 使用 SwiGLU 激活。
    """

    def __init__(self, config: TinyMixtralConfig, layer_index: int):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_experts = config.num_local_experts
        self.top_k = config.num_experts_per_tok
        self.expert_intermediate = config.expert_intermediate_size

        # CPT probability router. The legacy Linear router, jitter and final
        # softmax are fully replaced; Top-2 and all downstream expert logic
        # remain below.
        self.cpt_router = CPTRouter(config, layer_index=layer_index)

        # Expert 参数：每个 expert 有 gate_proj, up_proj, down_proj
        # 使用 3D 权重 [num_experts, intermediate, hidden] 方便实现
        self.gate_proj = nn.Parameter(
            torch.empty(self.num_experts, self.expert_intermediate, self.hidden_size)
        )
        self.up_proj = nn.Parameter(
            torch.empty(self.num_experts, self.expert_intermediate, self.hidden_size)
        )
        self.down_proj = nn.Parameter(
            torch.empty(self.num_experts, self.hidden_size, self.expert_intermediate)
        )

        self._init_weights(config.initializer_range)

    def _init_weights(self, initializer_range=0.02):
        nn.init.normal_(self.gate_proj, std=initializer_range)
        nn.init.normal_(self.up_proj, std=initializer_range)
        nn.init.normal_(self.down_proj, std=initializer_range)

    def _router_forward_tensors(
        self,
        x: torch.Tensor,
        route_valid_mask: Optional[torch.Tensor],
        reset_mask: Optional[torch.Tensor],
        sequence_state: Optional[CPTSequenceState],
        cpt_sequence_ids: Optional[torch.Tensor],
    ) -> tuple[torch.Tensor, ...]:
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
            # Compatibility for test doubles created before continuation state
            # became part of CPTRouterOutput. Production CPTRouter always
            # returns a detached state candidate.
            sequence_state_out = self.cpt_router._new_sequence_state(
                batch_size,
                x.device,
                sequence_ids=cpt_sequence_ids,
            )
        proposal = router_output.proposal
        return (
            # Full Pi.T remains differentiable for the selected expert combine
            # path. Only the optimizer-step load statistics are detached.
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

    def _expert_forward(
        self,
        x: torch.Tensor,
        routing_weights: torch.Tensor,
        valid_indices: torch.Tensor,
    ) -> torch.Tensor:
        """Apply Top-k dispatch and experts without calling the Router."""
        batch_size, seq_len, hidden_size = x.shape
        if valid_indices.numel() == 0:
            return torch.zeros_like(x)

        x_flat = x.reshape(-1, hidden_size)
        routing_weights_topk, selected_experts = torch.topk(
            routing_weights,
            self.top_k,
            dim=-1,
        )
        # Top-k chooses the sparse expert set.  Renormalize only the selected
        # combine weights used to mix expert outputs; full Pi above is kept.
        routing_weights_topk = routing_weights_topk / routing_weights_topk.sum(
            dim=-1,
            keepdim=True,
        )

        # Preserve CPT's valid-token mapping while adopting upstream's
        # sort-based expert dispatch. Each valid global token index is repeated
        # once per selected Top-k expert.
        flat_experts = selected_experts.reshape(-1)
        flat_weights = routing_weights_topk.reshape(-1)
        flat_token_idx = (
            valid_indices[:, None]
            .expand(-1, self.top_k)
            .reshape(-1)
        )
        sorted_indices = flat_experts.argsort(stable=True)
        sorted_token_idx = flat_token_idx[sorted_indices]
        sorted_weights = flat_weights[sorted_indices]
        sorted_experts = flat_experts[sorted_indices]
        expert_counts = torch.bincount(
            sorted_experts,
            minlength=self.num_experts,
        ).tolist()

        final_out = torch.zeros_like(x_flat)
        start = 0
        for expert in range(self.num_experts):
            count = expert_counts[expert]
            if count == 0:
                continue
            end = start + count
            idx = sorted_token_idx[start:end]
            weight = sorted_weights[start:end]
            token_states = x_flat.index_select(0, idx)
            gate = F.silu(token_states @ self.gate_proj[expert].T)
            up = token_states @ self.up_proj[expert].T
            expert_output = (gate * up) @ self.down_proj[expert].T
            weighted_output = expert_output * weight.to(
                expert_output.dtype
            ).unsqueeze(-1)
            final_out.index_add_(
                0,
                idx,
                weighted_output.to(final_out.dtype),
            )
            start = end
        return final_out.view(batch_size, seq_len, hidden_size)

    def forward(
        self,
        x: torch.Tensor,
        route_valid_mask: Optional[torch.Tensor] = None,
        reset_mask: Optional[torch.Tensor] = None,
        sequence_state: Optional[CPTSequenceState] = None,
        cpt_sequence_ids: Optional[torch.Tensor] = None,
        return_sequence_state: bool = False,
        checkpoint_router: bool = False,
        checkpoint_experts: bool = False,
    ) -> tuple[torch.Tensor, ...]:
        """Run Router and experts with independently selectable checkpoints."""
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
            routing_weights,
            valid_indices,
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
            and valid_indices.numel() > 0
        ):
            expert_output = checkpoint(
                self._expert_forward,
                x,
                routing_weights,
                valid_indices,
                use_reentrant=False,
                preserve_rng_state=True,
            )
        else:
            expert_output = self._expert_forward(
                x,
                routing_weights,
                valid_indices,
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


# ============================================================
# Transformer Block
# ============================================================

class MoETransformerBlock(nn.Module):
    """一个 Transformer 层：GQA Attention + MoE FFN。"""

    def __init__(self, config: TinyMixtralConfig, layer_index: int):
        super().__init__()
        self.input_layernorm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.self_attn = GQAAttention(config)
        self.moe = SparseMoE(config, layer_index=layer_index)

    def _attention_forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        position_ids: Optional[torch.Tensor],
        segment_ids: Optional[torch.Tensor],
    ) -> torch.Tensor:
        return self.self_attn(
            self.input_layernorm(hidden_states),
            attention_mask,
            position_ids,
            segment_ids,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        route_valid_mask: Optional[torch.Tensor] = None,
        reset_mask: Optional[torch.Tensor] = None,
        segment_ids: Optional[torch.Tensor] = None,
        sequence_state: Optional[CPTSequenceState] = None,
        cpt_sequence_ids: Optional[torch.Tensor] = None,
        return_sequence_state: bool = False,
        checkpoint_attention: bool = False,
        checkpoint_router: bool = False,
        checkpoint_experts: bool = False,
    ) -> tuple[torch.Tensor, ...]:
        residual = hidden_states
        checkpoint_active = self.training and torch.is_grad_enabled()
        if checkpoint_attention and checkpoint_active:
            attention_output = checkpoint(
                self._attention_forward,
                hidden_states,
                attention_mask,
                position_ids,
                segment_ids,
                use_reentrant=False,
                preserve_rng_state=True,
            )
        else:
            attention_output = self._attention_forward(
                hidden_states,
                attention_mask,
                position_ids,
                segment_ids,
            )
        hidden_states = residual + attention_output

        # MoE FFN
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
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
        ) = self.moe(
            hidden_states,
            route_valid_mask=route_valid_mask,
            reset_mask=reset_mask,
            sequence_state=sequence_state,
            cpt_sequence_ids=cpt_sequence_ids,
            return_sequence_state=True,
            checkpoint_router=checkpoint_router,
            checkpoint_experts=checkpoint_experts,
        )
        hidden_states = residual + hidden_states

        result = (
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
        )
        return result if return_sequence_state else result[:5]


# ============================================================
# TinyMixtralForCausalLM
# ============================================================

class TinyMixtralForCausalLM(CPTModelTransactionMixin, nn.Module):
    """TinyMixtral 因果语言模型。

    支持：
    - activation checkpointing（省显存）
    - FlashAttention via F.scaled_dot_product_attention
    - 与 HuggingFace transformers 兼容的 save/load 接口
    """

    def __init__(self, config: TinyMixtralConfig):
        super().__init__()
        self.config = config

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList([
            MoETransformerBlock(config, layer_index=layer_index)
            for layer_index in range(config.num_hidden_layers)
        ])
        self.norm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Tie embeddings
        if config.tie_word_embeddings:
            self.lm_head.weight = self.embed_tokens.weight

        self._use_activation_checkpointing = False
        self._init_weights()

    def _init_weights(self):
        std = self.config.initializer_range
        for module in self.modules():
            if isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, std=std)
            elif isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, std=std)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    @property
    def device(self):
        return next(self.parameters()).device

    def gradient_checkpointing_enable(self):
        self._use_activation_checkpointing = True

    def gradient_checkpointing_disable(self):
        self._use_activation_checkpointing = False

    def set_router_recompute(self, mode: str) -> None:
        """Set the Router recompute policy without changing the global flag."""
        if not isinstance(mode, str):
            raise ValueError("cpt_router_recompute must be a string")
        _resolve_router_recompute(mode, False)
        self.config.cpt_router_recompute = mode

    def load_state_dict(
        self,
        state_dict,
        strict: bool = True,
        assign: bool = False,
    ):
        """Load a complete native checkpoint only.

        The Hugging Face implementation intentionally keeps the mixin's
        staged ``strict=False`` behavior because Transformers materializes
        checkpoints incrementally.  The native model has no staged loader,
        so accepting ``strict=False`` here would silently admit a checkpoint
        with missing CPT router state and replace it with fresh initialization.
        """
        if strict is not True:
            raise RuntimeError(
                "Native TinyMixtral forbids strict=False checkpoint loading; "
                "a complete strict CPT state_dict is required."
            )
        return super().load_state_dict(
            state_dict,
            strict=True,
            assign=assign,
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        cpt_reset_mask: Optional[torch.Tensor] = None,
        cpt_segment_ids: Optional[torch.Tensor] = None,
        cpt_label_segment_ids: Optional[torch.Tensor] = None,
        cpt_sequence_states: Optional[Tuple[CPTSequenceState, ...]] = None,
        cpt_sequence_ids: Optional[torch.Tensor] = None,
        labels_are_pre_shifted: bool = True,
        return_dict: bool = True,
    ) -> dict:
        """
        Args:
            input_ids: [batch_size, seq_len]
            attention_mask: [batch_size, seq_len], 1=valid, 0=pad
            labels: [batch_size, seq_len], -100=masked. By default the
                native training API expects labels already shifted to the
                next token, matching the repository training loop. Set
                labels_are_pre_shifted=False for standard CausalLM labels.
            cpt_reset_mask: router-only reset points; this does not isolate
                self-attention by itself.
            cpt_segment_ids: integer packed-sequence ids. Tokens attend only
                to causal tokens with the same id, and every id transition
                resets the CPT sequence-local state. IDs must be
                nondecreasing across valid tokens in each batch row.
            cpt_label_segment_ids: optional segment ids for pre-shifted label
                positions. Without them, packed pre-shifted loss safely masks
                the final target because its segment is not observable from
                the current input window.
            cpt_sequence_states: optional detached per-layer continuation
                states from an earlier logical micro-batch. The returned
                state candidates must only be adopted after the caller's
                surrounding transaction succeeds. If a continued row starts
                a new segment at this micro-batch boundary, mark its first
                valid token in cpt_reset_mask.
            cpt_sequence_ids: optional unique integer logical sequence ID for
                each batch row. Identity-aware continuation rejects silent
                row replacement/reordering unless a changed row explicitly
                resets at its first valid token.
        Returns:
            dict with logits, loss, CPT transaction, and sequence-state candidates
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
        B, S = input_ids.shape
        if B <= 0:
            raise ValueError("input_ids batch size must be positive")
        if S <= 0:
            raise ValueError("input_ids sequence length must be positive")
        if S > self.config.max_position_embeddings:
            raise ValueError(
                f"sequence length {S} exceeds max_position_embeddings="
                f"{self.config.max_position_embeddings}"
            )
        if cpt_label_segment_ids is not None and labels is None:
            raise ValueError("cpt_label_segment_ids requires labels")
        position_ids = torch.arange(S, device=input_ids.device).unsqueeze(0).expand(B, -1)

        # 如果有 attention_mask，生成 causal mask 的复合 mask
        router_uses_default_dense_controls = (
            attention_mask is None
            and cpt_segment_ids is None
            and cpt_reset_mask is None
        )
        causal_mask = None
        if attention_mask is not None:
            # sdpa 需要 boolean mask: True = keep
            causal_mask = normalize_binary_mask(
                attention_mask,
                (B, S),
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
                B,
                input_ids.device,
            )
        )
        segment_ids = None
        if cpt_segment_ids is not None:
            segment_ids = _normalize_segment_ids(
                cpt_segment_ids,
                (B, S),
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
                (B, S),
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
                            batch_size=B,
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

        hidden_states = self.norm(hidden_states)
        logits = self.lm_head(hidden_states).float()  # fp32 for logits

        loss = None
        if labels is not None:
            # 训练循环已做 input/label 对齐（input=batch[:,:-1], labels=batch[:,1:]）
            # 此处无需再次 shift，直接用 logits 和 labels 计算 loss
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
        return {
            "logits": logits,
            "loss": loss,
            "cpt_transaction": transaction,
            "cpt_sequence_states": tuple(output_sequence_states),
        }

    def save_pretrained(self, path: str):
        """保存为 HuggingFace 兼容格式。"""
        self._assert_not_training_poisoned(operation="save model")
        import os
        os.makedirs(path, exist_ok=True)
        self.config.save_pretrained(path)
        state_dict = self.state_dict()
        torch.save(state_dict, f"{path}/pytorch_model.bin")

    @classmethod
    def from_pretrained(cls, path: str, config: Optional[TinyMixtralConfig] = None) -> "TinyMixtralForCausalLM":
        """从 HF 格式加载模型。"""
        if config is None:
            config = TinyMixtralConfig.from_json_file(f"{path}/config.json")
        model = cls(config)
        state_dict = torch.load(f"{path}/pytorch_model.bin", map_location="cpu", weights_only=True)
        model.load_state_dict(state_dict, strict=True, assign=True)
        if model.config.tie_word_embeddings:
            model.lm_head.weight = model.embed_tokens.weight
        # ``load_state_dict`` performs the same check once all layers are
        # materialized; retain an explicit final load boundary here so future
        # staged-loading changes cannot admit a mixed-version checkpoint.
        model.get_cpt_state_version()
        return model

    @property
    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())

    @property
    def num_parameters_trainable(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
