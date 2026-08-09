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

from .config import CPT_ROUTING_ARCHITECTURE_FIELDS, TinyMixtralConfig
from .cpt_router import (
    CPT_ROUTER_ALGORITHM_VERSION,
    CPTLayerProposal,
    CPTTransaction,
    CPTRouter,
)


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

        if attention_mask is not None:
            k_exp = k.unsqueeze(2).expand(-1, -1, self.num_groups, -1, -1).reshape(B, self.num_heads, S, self.head_dim)
            v_exp = v.unsqueeze(2).expand(-1, -1, self.num_groups, -1, -1).reshape(B, self.num_heads, S, self.head_dim)
            causal = torch.tril(torch.ones(S, S, device=hidden_states.device, dtype=torch.bool))
            pad_4d = attention_mask[:, None, None, :]
            combined = causal[None, None, :, :] & pad_4d
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

        # Strict CPT v1 probability Router.  The legacy Linear gate and its
        # jitter are intentionally absent.
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

        self._init_weights()

    def _init_weights(self, initializer_range=0.02):
        nn.init.normal_(self.gate_proj, std=initializer_range)
        nn.init.normal_(self.up_proj, std=initializer_range)
        nn.init.normal_(self.down_proj, std=initializer_range)

    def forward(
        self,
        x: torch.Tensor,
        route_valid_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        """
        Args:
            x: [batch_size, seq_len, hidden_size]
            route_valid_mask: [batch_size, seq_len], True=route, False=padding
        Returns:
            out: [batch_size, seq_len, hidden_size]
            aux_loss: FP32 scalar zero retained for interface compatibility
            load_sum: detached FP32 soft expert load
            token_count: detached int64 count of routed tokens
            state_version: detached int64 CPT state version
        """
        B, S, D = x.shape
        x_flat = x.view(-1, D)  # [B*S, D]
        N = B * S

        router_output = self.cpt_router(x, route_valid_mask=route_valid_mask)
        # CPT already returns final expert probabilities Pi^T = Q^T B.
        # There is no post-Pi softmax; upstream Top-2 consumes them directly.
        all_routing_weights = router_output.probabilities.view(-1, self.num_experts)
        if route_valid_mask is None:
            valid_token_idx = torch.arange(N, device=x.device)
        else:
            valid_token_idx = torch.nonzero(
                route_valid_mask.to(device=x.device, dtype=torch.bool).reshape(-1),
                as_tuple=False,
            ).flatten()
        routing_weights = all_routing_weights.index_select(
            0, valid_token_idx
        ).to(x.dtype)
        routing_weights_topk, selected_experts = torch.topk(
            routing_weights,
            self.top_k,
            dim=-1,
        )
        routing_weights_topk = routing_weights_topk / routing_weights_topk.sum(
            dim=-1,
            keepdim=True,
        )

        # Interface-compatible placeholder: the old differentiable load
        # balancing loss is not part of CPT and is no longer computed.
        aux_loss = torch.zeros((), device=x.device, dtype=torch.float32)

        flat_experts = selected_experts.view(-1)
        flat_weights = routing_weights_topk.view(-1)
        flat_token_idx = (
            valid_token_idx.unsqueeze(1).expand(-1, self.top_k).reshape(-1)
        )

        sorted_indices = flat_experts.argsort(stable=True)
        sorted_token_idx = flat_token_idx[sorted_indices]
        sorted_weights = flat_weights[sorted_indices]
        sorted_experts = flat_experts[sorted_indices]

        expert_counts = torch.bincount(sorted_experts, minlength=self.num_experts).tolist()

        final_out = torch.zeros(N, D, device=x.device, dtype=x.dtype)
        start = 0
        for e in range(self.num_experts):
            count = expert_counts[e]
            if count == 0:
                continue
            end = start + count
            idx = sorted_token_idx[start:end]
            w = sorted_weights[start:end]
            token_states = x_flat[idx]

            gate = F.silu(torch.matmul(token_states, self.gate_proj[e].T))
            up = torch.matmul(token_states, self.up_proj[e].T)
            expert_out = torch.matmul(gate * up, self.down_proj[e].T)

            final_out.index_add_(0, idx, (expert_out * w.unsqueeze(-1)).to(x.dtype))
            start = end

        proposal = router_output.proposal
        return (
            final_out.view(B, S, D),
            aux_loss,
            proposal.load_sum,
            proposal.token_count,
            proposal.state_version,
        )


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

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
    ) -> Tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        # Self-attention
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(hidden_states, attention_mask, position_ids)
        hidden_states = residual + hidden_states

        # MoE FFN
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states, aux_loss, load_sum, token_count, state_version = self.moe(
            hidden_states, route_valid_mask=attention_mask
        )
        hidden_states = residual + hidden_states

        return hidden_states, aux_loss, load_sum, token_count, state_version


# ============================================================
# TinyMixtralForCausalLM
# ============================================================

class TinyMixtralForCausalLM(nn.Module):
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

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        return_dict: bool = True,
    ) -> dict:
        """
        Args:
            input_ids: [batch_size, seq_len]
            attention_mask: [batch_size, seq_len], 1=valid, 0=pad
            labels: [batch_size, seq_len], -100=masked
        Returns:
            dict with keys: logits, loss, aux_loss, cpt_transaction
        """
        B, S = input_ids.shape
        position_ids = torch.arange(S, device=input_ids.device).unsqueeze(0).expand(B, -1)

        # 如果有 attention_mask，生成 causal mask 的复合 mask
        causal_mask = None
        if attention_mask is not None:
            # sdpa 需要 boolean mask: True = keep
            causal_mask = attention_mask.bool()

        hidden_states = self.embed_tokens(input_ids)
        total_aux_loss = torch.tensor(0.0, device=input_ids.device, dtype=torch.float32)
        layer_proposals: list[CPTLayerProposal] = []

        for layer_index, layer in enumerate(self.layers):
            if self._use_activation_checkpointing and self.training:
                (
                    hidden_states,
                    aux_loss,
                    load_sum,
                    token_count,
                    state_version,
                ) = checkpoint(
                    layer,
                    hidden_states,
                    causal_mask,
                    position_ids,
                    use_reentrant=False,
                )
            else:
                hidden_states, aux_loss, load_sum, token_count, state_version = layer(
                    hidden_states, causal_mask, position_ids
                )
            total_aux_loss = total_aux_loss + aux_loss
            layer_proposals.append(
                CPTLayerProposal(
                    layer_index=layer_index,
                    load_sum=load_sum,
                    token_count=token_count,
                    state_version=state_version,
                )
            )

        total_aux_loss = total_aux_loss / len(self.layers)

        hidden_states = self.norm(hidden_states)
        logits = self.lm_head(hidden_states).float()  # fp32 for logits

        loss = None
        if labels is not None:
            # 训练循环已做 input/label 对齐（input=batch[:,:-1], labels=batch[:,1:]）
            # 此处无需再次 shift，直接用 logits 和 labels 计算 loss
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                labels.reshape(-1),
                ignore_index=-100,
            )

        return {
            "logits": logits,
            "loss": loss,
            "aux_loss": total_aux_loss.detach(),
            "cpt_transaction": CPTTransaction(tuple(layer_proposals)),
        }

    def _cpt_routers(self) -> Tuple[CPTRouter, ...]:
        return tuple(layer.moe.cpt_router for layer in self.layers)

    def cpt_trainable_parameters(self) -> Tuple[nn.Parameter, ...]:
        return tuple(
            parameter
            for router in self._cpt_routers()
            for parameter in router.trainable_parameters()
        )

    def _validate_cpt_config_binding(self) -> None:
        for layer_index, (layer, router) in enumerate(
            zip(self.layers, self._cpt_routers())
        ):
            router.validate_config_binding(self.config)
            if (
                type(layer.moe.top_k) is not type(self.config.num_experts_per_tok)
                or layer.moe.top_k != self.config.num_experts_per_tok
            ):
                raise RuntimeError(
                    f"MoE layer {layer_index} Top-k no longer matches model config"
                )

    @torch.no_grad()
    def validate_persistent_cpt_state(
        self,
        *,
        require_unit_anchors: bool = True,
    ) -> None:
        self._validate_cpt_config_binding()
        versions: list[int] = []
        optimizer_steps: list[int] = []
        for router in self._cpt_routers():
            router.validate_persistent_state(
                require_unit_anchors=require_unit_anchors,
            )
            versions.append(int(router.state_version.item()))
            optimizer_steps.append(int(router.optimizer_step.item()))
        if versions and len(set(versions)) != 1:
            raise RuntimeError(f"CPT layer state versions disagree: {versions}")
        if optimizer_steps and len(set(optimizer_steps)) != 1:
            raise RuntimeError(
                "CPT layer optimizer steps disagree: "
                f"{optimizer_steps}"
            )

    def get_cpt_state_version(self) -> int:
        versions = [
            int(router.state_version.item())
            for router in self._cpt_routers()
        ]
        if not versions:
            raise RuntimeError("model has no CPT Routers")
        if len(set(versions)) != 1:
            raise RuntimeError(f"CPT layer state versions disagree: {versions}")
        return versions[0]

    def get_cpt_optimizer_step(self) -> int:
        optimizer_steps = [
            int(router.optimizer_step.item())
            for router in self._cpt_routers()
        ]
        if not optimizer_steps:
            raise RuntimeError("model has no CPT Routers")
        if len(set(optimizer_steps)) != 1:
            raise RuntimeError(
                "CPT layer optimizer steps disagree: "
                f"{optimizer_steps}"
            )
        return optimizer_steps[0]

    def validate_cpt_transaction(self, transaction: CPTTransaction) -> None:
        if not isinstance(transaction, CPTTransaction):
            raise TypeError("model output is missing a valid CPT transaction")
        if transaction.consumed:
            raise RuntimeError("CPT transaction was already consumed")
        self._validate_cpt_config_binding()
        routers = self._cpt_routers()
        if not isinstance(transaction.proposals, tuple):
            raise TypeError("CPT transaction proposals must be a tuple")
        if len(transaction.proposals) != len(routers):
            raise RuntimeError("CPT transaction does not cover every MoE layer")
        self.get_cpt_state_version()
        token_counts: list[int] = []
        for router, proposal in zip(routers, transaction.proposals):
            router.validate_proposal(proposal)
            token_counts.append(int(proposal.token_count.item()))
        if token_counts and len(set(token_counts)) != 1:
            raise RuntimeError(f"CPT layer token counts disagree: {token_counts}")

    @torch.no_grad()
    def commit_cpt_transaction(
        self,
        transaction: CPTTransaction,
        *,
        optimizer_step: Optional[int] = None,
    ) -> int:
        """Commit all Router prices/anchors/versions, or none of them."""
        if bool(getattr(self, "_tinymixtral_fail_stop", False)):
            raise RuntimeError(
                "training state is fail-stop poisoned; restore the last "
                "successful checkpoint"
            )
        try:
            self.validate_cpt_transaction(transaction)
            current_optimizer_step = self.get_cpt_optimizer_step()
            if optimizer_step is None:
                optimizer_step = current_optimizer_step
            elif type(optimizer_step) is not int or optimizer_step < 0:
                raise TypeError(
                    "CPT optimizer_step must be a non-negative integer"
                )
            if optimizer_step not in (
                current_optimizer_step,
                current_optimizer_step + 1,
            ):
                raise RuntimeError(
                    "CPT optimizer_step must stay unchanged or advance by one"
                )
        except BaseException:
            if isinstance(transaction, CPTTransaction):
                transaction.consumed = True
            raise
        # A commit attempt is single-use.  Any later preparation, write, or
        # validation failure discards this proposal even when rollback succeeds.
        transaction.consumed = True
        routers = self._cpt_routers()
        prepared = [
            router.prepare_commit(
                proposal,
                optimizer_step=optimizer_step,
            )
            for router, proposal in zip(routers, transaction.proposals)
        ]
        snapshots = [router.commit_snapshot() for router in routers]
        try:
            for router, candidate in zip(routers, prepared):
                router.apply_commit(*candidate)
            self.validate_persistent_cpt_state()
        except BaseException:
            recovery_errors = []
            for router, snapshot in zip(routers, snapshots):
                try:
                    router.restore_commit_snapshot(snapshot)
                except BaseException as error:
                    recovery_errors.append(error)
            if recovery_errors:
                self._tinymixtral_fail_stop = True
                failure_types = ", ".join(
                    type(error).__name__ for error in recovery_errors
                )
                raise RuntimeError(
                    "CPT commit failed and recovery was incomplete: "
                    + failure_types
                ) from recovery_errors[0]
            raise
        return self.get_cpt_state_version()

    @staticmethod
    def abort_cpt_transaction(transaction: Optional[CPTTransaction]) -> None:
        if transaction is None:
            return
        if not isinstance(transaction, CPTTransaction):
            raise TypeError("cannot abort an invalid CPT transaction")
        transaction.consumed = True

    def _validate_serialized_cpt_state(self, state_dict) -> None:
        if not hasattr(state_dict, "keys"):
            raise TypeError("state_dict must be a mapping")
        keys = tuple(state_dict.keys())
        legacy = [key for key in keys if key.endswith(".moe.router.weight")]
        if legacy:
            raise RuntimeError(
                "legacy Linear Router checkpoints are incompatible with CPT v1: "
                + ", ".join(legacy)
            )

        marker = ".moe.cpt_router."
        expected_state = super().state_dict()
        expected_keys = {key for key in expected_state if marker in key}
        supplied_keys = {key for key in keys if marker in key}
        missing = sorted(expected_keys - supplied_keys)
        unexpected = sorted(supplied_keys - expected_keys)
        if missing:
            raise RuntimeError("checkpoint is missing CPT keys: " + ", ".join(missing))
        if unexpected:
            raise RuntimeError(
                "checkpoint contains unknown CPT keys: " + ", ".join(unexpected)
            )

        serialized_versions: list[int] = []
        serialized_optimizer_steps: list[int] = []
        for key in sorted(expected_keys):
            value = state_dict[key]
            expected = expected_state[key]
            if not isinstance(value, torch.Tensor):
                raise RuntimeError(f"checkpoint CPT value is not a tensor: {key}")
            if value.shape != expected.shape:
                raise RuntimeError(
                    f"checkpoint CPT shape mismatch for {key}: "
                    f"expected {tuple(expected.shape)}, got {tuple(value.shape)}"
                )
            if value.dtype != expected.dtype:
                raise RuntimeError(
                    f"checkpoint CPT dtype mismatch for {key}: "
                    f"expected {expected.dtype}, got {value.dtype}"
                )
            if value.is_floating_point() and not bool(torch.isfinite(value).all()):
                raise RuntimeError(f"checkpoint CPT value is non-finite: {key}")
            if key.endswith(".anchors"):
                norms = torch.linalg.vector_norm(value, dim=0)
                if not torch.allclose(
                    norms,
                    torch.ones_like(norms),
                    atol=5e-5,
                    rtol=5e-5,
                ):
                    raise RuntimeError("checkpoint CPT anchors are not unit-normalized")
            elif key.endswith(".congestion_price"):
                if bool((value < 0).any()):
                    raise RuntimeError("checkpoint CPT congestion price is negative")
            elif key.endswith(".router_algorithm_version"):
                if int(value.item()) != CPT_ROUTER_ALGORITHM_VERSION:
                    raise RuntimeError(
                        "checkpoint uses an unsupported CPT algorithm version"
                    )
            elif key.endswith(".optimizer_step"):
                optimizer_step = int(value.item())
                if optimizer_step < 0:
                    raise RuntimeError("checkpoint CPT optimizer_step is negative")
                serialized_optimizer_steps.append(optimizer_step)
            elif key.endswith(".state_version"):
                version = int(value.item())
                if version < 0:
                    raise RuntimeError("checkpoint CPT state_version is negative")
                serialized_versions.append(version)
        if serialized_versions and len(set(serialized_versions)) != 1:
            raise RuntimeError(
                f"checkpoint CPT layer state versions disagree: {serialized_versions}"
            )
        if serialized_optimizer_steps and len(set(serialized_optimizer_steps)) != 1:
            raise RuntimeError(
                "checkpoint CPT layer optimizer steps disagree: "
                f"{serialized_optimizer_steps}"
            )
        if (
            serialized_versions
            and serialized_optimizer_steps
            and serialized_optimizer_steps[0] > serialized_versions[0]
        ):
            raise RuntimeError(
                "checkpoint CPT optimizer_step exceeds state_version"
            )

    def load_state_dict(self, state_dict, strict: bool = True, assign: bool = False):
        self._validate_cpt_config_binding()
        self._validate_serialized_cpt_state(state_dict)
        result = super().load_state_dict(
            state_dict,
            strict=strict,
            assign=assign,
        )
        self.validate_persistent_cpt_state()
        return result

    def save_pretrained(self, path: str):
        """保存为 HuggingFace 兼容格式。"""
        import os
        if bool(getattr(self, "_tinymixtral_fail_stop", False)):
            raise RuntimeError(
                "training state is fail-stop poisoned; restore the last "
                "successful checkpoint"
            )
        self.validate_persistent_cpt_state()
        os.makedirs(path, exist_ok=True)
        self.config.save_pretrained(path)
        state_dict = self.state_dict()
        torch.save(state_dict, f"{path}/pytorch_model.bin")

    @classmethod
    def from_pretrained(cls, path: str, config: Optional[TinyMixtralConfig] = None) -> "TinyMixtralForCausalLM":
        """从 HF 格式加载模型。"""
        import json

        with open(f"{path}/config.json", encoding="utf-8") as config_file:
            raw_config = json.load(config_file)
        if not isinstance(raw_config, dict):
            raise RuntimeError("checkpoint config.json must contain an object")
        missing_routing_architecture = sorted(
            field
            for field in CPT_ROUTING_ARCHITECTURE_FIELDS
            if field not in raw_config
        )
        if missing_routing_architecture:
            raise RuntimeError(
                "checkpoint config is missing CPT routing architecture fields: "
                + ", ".join(missing_routing_architecture)
            )
        expected_cpt_keys = {
            key
            for key in TinyMixtralConfig.__dataclass_fields__
            if key.startswith("cpt_")
        }
        supplied_cpt_keys = {
            key
            for key in raw_config
            if key.startswith("cpt_")
        }
        missing_cpt = sorted(expected_cpt_keys - supplied_cpt_keys)
        unknown_cpt = sorted(supplied_cpt_keys - expected_cpt_keys)
        if missing_cpt:
            raise RuntimeError(
                "checkpoint config is missing CPT fields: " + ", ".join(missing_cpt)
            )
        if unknown_cpt:
            raise RuntimeError(
                "checkpoint config contains unknown CPT fields: "
                + ", ".join(unknown_cpt)
            )
        unresolved_cpt = sorted(
            key for key in expected_cpt_keys if raw_config[key] is None
        )
        if unresolved_cpt:
            raise RuntimeError(
                "checkpoint config contains unresolved CPT fields: "
                + ", ".join(unresolved_cpt)
            )
        saved_config = TinyMixtralConfig.from_dict(raw_config)
        if config is None:
            config = saved_config
        else:
            explicit_router_config = config.cpt_config_dict()
            saved_router_config = saved_config.cpt_config_dict()
            for field in CPT_ROUTING_ARCHITECTURE_FIELDS:
                explicit_router_config[field] = getattr(config, field)
                saved_router_config[field] = getattr(saved_config, field)
            router_config_disagrees = (
                explicit_router_config.keys() != saved_router_config.keys()
                or any(
                    type(explicit_router_config[key]) is not type(
                        saved_router_config[key]
                    )
                    or explicit_router_config[key] != saved_router_config[key]
                    for key in (
                        explicit_router_config.keys()
                        & saved_router_config.keys()
                    )
                )
            )
            if router_config_disagrees:
                raise RuntimeError(
                    "explicit CPT routing config disagrees with checkpoint "
                    "config.json"
                )
        model = cls(config)
        state_dict = torch.load(f"{path}/pytorch_model.bin", map_location="cpu", weights_only=True)
        model.load_state_dict(state_dict, strict=True)
        return model

    @property
    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())

    @property
    def num_parameters_trainable(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
