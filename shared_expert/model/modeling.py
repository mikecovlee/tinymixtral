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

        # GQA: expand KV heads
        k = k.unsqueeze(2).expand(-1, -1, self.num_groups, -1, -1).reshape(B, self.num_heads, S, self.head_dim)
        v = v.unsqueeze(2).expand(-1, -1, self.num_groups, -1, -1).reshape(B, self.num_heads, S, self.head_dim)

        # RoPE
        if position_ids is None:
            position_ids = torch.arange(S, device=hidden_states.device).unsqueeze(0).expand(B, -1)
        q = self.rotary_emb(q, position_ids)
        k = self.rotary_emb(k, position_ids)

        # FlashAttention via sdpa — 必须显式合并 causal + padding mask
        # PyTorch 2.x 不允许 attn_mask 和 is_causal 同时设置
        if attention_mask is not None:
            # attention_mask: [B, S] bool, True=valid token
            # 构造 4D causal mask 并与 padding 合并
            causal = torch.tril(torch.ones(S, S, device=hidden_states.device, dtype=torch.bool))
            # padding: [B, 1, 1, S] → 控制哪些 key 可见
            pad_4d = attention_mask[:, None, None, :]  # [B, 1, 1, S]
            combined = causal[None, None, :, :] & pad_4d  # [B, 1, S, S]
            attn_output = F.scaled_dot_product_attention(
                q, k, v,
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
            )

        attn_output = attn_output.transpose(1, 2).reshape(B, S, -1)
        return self.o_proj(attn_output)


# ============================================================
# MoE FFN
# ============================================================

class SparseMoE(nn.Module):
    """DeepSeek-style Sparse MoE: shared experts (always active) + routed experts (top-k).

    Shared experts learn general-purpose features and are always computed.
    Routed experts specialize via top-k gating. Output is the sum of both.
    """

    def __init__(self, config: TinyMixtralConfig):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_shared = config.num_shared_experts     # always-on
        self.num_routed = config.num_routed_experts      # top-k selected
        self.top_k = config.num_experts_per_tok
        self.expert_intermediate = config.expert_intermediate_size
        self.jitter_noise = config.router_jitter_noise
        self.aux_loss_coef = config.router_aux_loss_coef

        # Router — only over routed experts
        if self.num_routed > 0:
            self.router = nn.Linear(self.hidden_size, self.num_routed, bias=False)

        # Shared expert parameters: [num_shared, intermediate, hidden] × 3
        if self.num_shared > 0:
            self.shared_gate_proj = nn.Parameter(
                torch.empty(self.num_shared, self.expert_intermediate, self.hidden_size)
            )
            self.shared_up_proj = nn.Parameter(
                torch.empty(self.num_shared, self.expert_intermediate, self.hidden_size)
            )
            self.shared_down_proj = nn.Parameter(
                torch.empty(self.num_shared, self.hidden_size, self.expert_intermediate)
            )

        # Routed expert parameters: [num_routed, intermediate, hidden] × 3
        if self.num_routed > 0:
            self.gate_proj = nn.Parameter(
                torch.empty(self.num_routed, self.expert_intermediate, self.hidden_size)
            )
            self.up_proj = nn.Parameter(
                torch.empty(self.num_routed, self.expert_intermediate, self.hidden_size)
            )
            self.down_proj = nn.Parameter(
                torch.empty(self.num_routed, self.hidden_size, self.expert_intermediate)
            )

        self._init_weights()

    def _init_weights(self, std=0.02):
        for name in ("shared_gate_proj", "shared_up_proj", "shared_down_proj",
                     "gate_proj", "up_proj", "down_proj"):
            param = getattr(self, name, None)
            if param is not None:
                nn.init.normal_(param, std=std)

    def _forward_expert(self, x, gate_w, up_w, down_w):
        """SwiGLU expert: silu(gate(x)) * up(x) → down."""
        gate = F.silu(torch.matmul(x, gate_w.T))
        up = torch.matmul(x, up_w.T)
        return torch.matmul(gate * up, down_w.T)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        B, S, D = x.shape
        x_flat = x.view(-1, D)

        out = torch.zeros(B * S, D, device=x.device, dtype=x.dtype)

        # 1. Shared experts — always active, equal weight
        for e in range(self.num_shared):
            out += self._forward_expert(
                x_flat, self.shared_gate_proj[e], self.shared_up_proj[e], self.shared_down_proj[e]
            )

        # 2. Routed experts — top-k gating
        aux_loss = torch.tensor(0.0, device=x.device, dtype=x.dtype)
        if self.num_routed == 0:
            return out.view(B, S, D), aux_loss

        router_logits = self.router(x_flat)

        if self.training and self.jitter_noise > 0:
            router_logits = router_logits * (1 + torch.randn_like(router_logits) * self.jitter_noise)

        routing_weights = F.softmax(router_logits.float(), dim=-1).to(x.dtype)
        routing_weights_topk, selected_experts = torch.topk(routing_weights, self.top_k, dim=-1)
        routing_weights_topk = routing_weights_topk / routing_weights_topk.sum(dim=-1, keepdim=True)

        if self.training and self.aux_loss_coef > 0:
            with torch.no_grad():
                expert_mask = F.one_hot(selected_experts, num_classes=self.num_routed).float()
                f_i = expert_mask.mean(dim=(0, 1))
            P_i = routing_weights.mean(dim=0)
            aux_loss = (f_i.detach() * P_i).sum() * self.num_routed

        for k in range(self.top_k):
            expert_idx = selected_experts[:, k]
            weight = routing_weights_topk[:, k]
            for e in range(self.num_routed):
                mask = (expert_idx == e)
                if not mask.any():
                    continue
                expert_out = self._forward_expert(
                    x_flat[mask], self.gate_proj[e], self.up_proj[e], self.down_proj[e]
                )
                out[mask] += expert_out * weight[mask].unsqueeze(-1)

        return out.view(B, S, D), aux_loss


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
            loss = loss + self.config.router_aux_loss_coef * total_aux_loss

        return {
            "logits": logits,
            "loss": loss,
            "aux_loss": total_aux_loss.detach(),
        }

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
