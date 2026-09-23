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

        self.use_qk_norm = config.use_qk_norm
        if self.use_qk_norm:
            self.q_norm = RMSNorm(self.head_dim, config.rms_norm_eps)
            self.k_norm = RMSNorm(self.head_dim, config.rms_norm_eps)
        else:
            self.q_norm = None
            self.k_norm = None

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

        # QK-Norm（投影后、RoPE 前，per-head RMSNorm）
        if self.q_norm is not None:
            q = self.q_norm(q)
            k = self.k_norm(k)

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

    def __init__(self, config: TinyMixtralConfig):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_experts = config.num_local_experts
        self.top_k = config.num_experts_per_tok
        self.expert_intermediate = config.expert_intermediate_size
        self.jitter_noise = config.router_jitter_noise
        self.aux_loss_coef = config.router_aux_loss_coef
        self.last_expert_counts: Optional[torch.Tensor] = None

        # Router
        self.router = nn.Linear(self.hidden_size, self.num_experts, bias=False)

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

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: [batch_size, seq_len, hidden_size]
        Returns:
            out: [batch_size, seq_len, hidden_size]
            aux_loss: scalar tensor
        """
        B, S, D = x.shape
        x_flat = x.view(-1, D)  # [B*S, D]
        N = B * S

        router_logits = self.router(x_flat)  # [N, num_experts]

        if self.training and self.jitter_noise > 0:
            router_logits = router_logits * (1 + torch.randn_like(router_logits) * self.jitter_noise)

        routing_weights = F.softmax(router_logits.float(), dim=-1).to(x.dtype)
        routing_weights_topk, selected_experts = torch.topk(routing_weights, self.top_k, dim=-1)
        routing_weights_topk = routing_weights_topk / routing_weights_topk.sum(dim=-1, keepdim=True)

        aux_loss = torch.tensor(0.0, device=x.device, dtype=x.dtype)
        if self.training and self.aux_loss_coef > 0:
            with torch.no_grad():
                expert_mask = F.one_hot(selected_experts, num_classes=self.num_experts).float()
                f_i = expert_mask.mean(dim=(0, 1))
            P_i = routing_weights.mean(dim=0)
            aux_loss = (f_i.detach() * P_i).sum() * self.num_experts

        if self.training:
            with torch.no_grad():
                self.last_expert_counts = torch.bincount(
                    selected_experts.view(-1), minlength=self.num_experts
                )

        flat_experts = selected_experts.view(-1)
        flat_weights = routing_weights_topk.view(-1)
        flat_token_idx = torch.arange(N, device=x.device).unsqueeze(1).expand(-1, self.top_k).reshape(-1)

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

        if self.training:
            # 零 token 专家梯度保活：0 × Σ(all expert weights) 挂在计算图上，
            # 保证未命中专家也收到（零）梯度，避免 DDP unused-parameter 报错、
            # 并保持权重衰减对其一致生效。对输出数值零扰动。
            keepalive = (self.gate_proj.sum() + self.up_proj.sum() + self.down_proj.sum()) * 0.0
            final_out = final_out + keepalive.to(final_out.dtype)

        return final_out.view(B, S, D), aux_loss


# ============================================================
# Transformer Block
# ============================================================

class MoETransformerBlock(nn.Module):
    """一个 Transformer 层：GQA Attention + MoE FFN。"""

    def __init__(self, config: TinyMixtralConfig):
        super().__init__()
        self.input_layernorm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.self_attn = GQAAttention(config)
        self.moe = SparseMoE(config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # Self-attention
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(hidden_states, attention_mask, position_ids)
        hidden_states = residual + hidden_states

        # MoE FFN
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states, aux_loss = self.moe(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states, aux_loss


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
            MoETransformerBlock(config) for _ in range(config.num_hidden_layers)
        ])
        self.norm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Tie embeddings
        if config.tie_word_embeddings:
            self.lm_head.weight = self.embed_tokens.weight

        self._use_activation_checkpointing = False
        self.use_chunked_ce = False
        self.ce_chunk_size = 512
        self._init_weights()

    def _chunked_cross_entropy(self, hidden_states, labels):
        def _ce(hc, lc):
            lg = self.lm_head(hc).float()
            return F.cross_entropy(
                lg.reshape(-1, lg.size(-1)), lc.reshape(-1),
                ignore_index=-100, reduction="sum",
            )

        total = torch.zeros((), device=hidden_states.device, dtype=torch.float32)
        n_valid = 0
        for i in range(0, hidden_states.size(1), self.ce_chunk_size):
            hc = hidden_states[:, i:i + self.ce_chunk_size]
            lc = labels[:, i:i + self.ce_chunk_size]
            if self.training and hc.requires_grad:
                s = checkpoint(_ce, hc, lc, use_reentrant=False)
            else:
                s = _ce(hc, lc)
            total = total + s
            n_valid += int((lc != -100).sum())
        return total / max(n_valid, 1)

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
            dict with keys: logits, loss, aux_loss
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

        for layer in self.layers:
            if self._use_activation_checkpointing and self.training:
                hidden_states, aux_loss = checkpoint(
                    layer, hidden_states, causal_mask, position_ids,
                    use_reentrant=False,
                )
            else:
                hidden_states, aux_loss = layer(hidden_states, causal_mask, position_ids)
            total_aux_loss = total_aux_loss + aux_loss

        total_aux_loss = total_aux_loss / len(self.layers)

        hidden_states = self.norm(hidden_states)

        loss = None
        ce_loss = None
        logits = None
        if labels is not None:
            if self.use_chunked_ce:
                ce_loss = self._chunked_cross_entropy(hidden_states, labels)
            else:
                logits = self.lm_head(hidden_states).float()
                ce_loss = F.cross_entropy(
                    logits.reshape(-1, logits.size(-1)),
                    labels.reshape(-1),
                    ignore_index=-100,
                )
            loss = ce_loss + self.config.router_aux_loss_coef * total_aux_loss
        else:
            logits = self.lm_head(hidden_states).float()

        return {
            "logits": logits,
            "loss": loss,
            "ce_loss": ce_loss.detach() if ce_loss is not None else None,
            "aux_loss": total_aux_loss.detach(),
        }

    def expert_utilization(self) -> Optional[list]:
        """各专家在最近一次训练 forward 中被选中的 slot 占比（跨层聚合）。

        返回长度 num_local_experts 的浮点列表（和为 1），无统计时返回 None。
        """
        counts = [
            layer.moe.last_expert_counts
            for layer in self.layers
            if layer.moe.last_expert_counts is not None
        ]
        if not counts:
            return None
        total = torch.stack(counts).sum(dim=0).float()
        return (total / total.sum()).tolist()

    def save_pretrained(self, path: str):
        """保存为 HuggingFace 兼容格式。"""
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
        model.load_state_dict(state_dict, strict=True)
        return model

    @property
    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())

    @property
    def num_parameters_trainable(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
