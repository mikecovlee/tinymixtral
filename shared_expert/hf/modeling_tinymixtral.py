# Copyright (C) Michael Lee (李登淳) 2026. All rights reserved.
# Open-source under the MIT License. See LICENSE for details.

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from transformers import PreTrainedModel

from .configuration_tinymixtral import TinyMixtralConfig


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

    def forward(self, hidden_states, attention_mask=None, position_ids=None):
        B, S, _ = hidden_states.shape
        q = self.q_proj(hidden_states).view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(hidden_states).view(B, S, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(hidden_states).view(B, S, self.num_kv_heads, self.head_dim).transpose(1, 2)
        if position_ids is None:
            position_ids = torch.arange(S, device=hidden_states.device).unsqueeze(0).expand(B, -1)
        q, k = self.rotary_emb(q, position_ids), self.rotary_emb(k, position_ids)

        if attention_mask is not None:
            k_exp = k.unsqueeze(2).expand(-1, -1, self.num_groups, -1, -1).reshape(B, self.num_heads, S, self.head_dim)
            v_exp = v.unsqueeze(2).expand(-1, -1, self.num_groups, -1, -1).reshape(B, self.num_heads, S, self.head_dim)
            causal = torch.tril(torch.ones(S, S, device=hidden_states.device, dtype=torch.bool))
            combined = causal[None, None, :, :] & attention_mask[:, None, None, :]
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
    def __init__(self, config):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_shared = config.num_shared_experts
        self.num_routed = config.num_routed_experts
        self.top_k = config.num_experts_per_tok
        self.expert_intermediate = config.expert_intermediate_size
        self.jitter_noise = config.router_jitter_noise
        self.aux_loss_coef = config.router_aux_loss_coef

        if self.num_routed > 0:
            self.router = nn.Linear(self.hidden_size, self.num_routed, bias=False)
        if self.num_shared > 0:
            self.shared_gate_proj = nn.Parameter(torch.empty(self.num_shared, self.expert_intermediate, self.hidden_size))
            self.shared_up_proj = nn.Parameter(torch.empty(self.num_shared, self.expert_intermediate, self.hidden_size))
            self.shared_down_proj = nn.Parameter(torch.empty(self.num_shared, self.hidden_size, self.expert_intermediate))
        if self.num_routed > 0:
            self.gate_proj = nn.Parameter(torch.empty(self.num_routed, self.expert_intermediate, self.hidden_size))
            self.up_proj = nn.Parameter(torch.empty(self.num_routed, self.expert_intermediate, self.hidden_size))
            self.down_proj = nn.Parameter(torch.empty(self.num_routed, self.hidden_size, self.expert_intermediate))
        self._init_weights()

    def _init_weights(self, std=0.02):
        for name in ("shared_gate_proj", "shared_up_proj", "shared_down_proj",
                     "gate_proj", "up_proj", "down_proj"):
            param = getattr(self, name, None)
            if param is not None:
                nn.init.normal_(param, std=std)

    def _forward_expert(self, x, gate_w, up_w, down_w):
        gate = F.silu(x @ gate_w.T)
        up = x @ up_w.T
        return (gate * up) @ down_w.T

    def forward(self, x):
        B, S, D = x.shape
        x_flat = x.view(-1, D)

        out = torch.zeros(B * S, D, device=x.device, dtype=x.dtype)

        # Shared experts: always active
        for e in range(self.num_shared):
            out += self._forward_expert(x_flat, self.shared_gate_proj[e], self.shared_up_proj[e], self.shared_down_proj[e])

        aux = torch.tensor(0.0, device=x.device, dtype=x.dtype)
        if self.num_routed == 0:
            return out.view(B, S, D), aux

        # Routed experts: top-k gating
        logits = self.router(x_flat)
        if self.training and self.jitter_noise > 0:
            logits = logits * (1 + torch.randn_like(logits) * self.jitter_noise)
        weights = F.softmax(logits.float(), dim=-1).to(x.dtype)
        w_topk, experts = torch.topk(weights, self.top_k, dim=-1)
        w_topk = w_topk / w_topk.sum(dim=-1, keepdim=True)

        if self.training and self.aux_loss_coef > 0:
            with torch.no_grad():
                mask = F.one_hot(experts, num_classes=self.num_routed).float()
                f_i = mask.mean(dim=(0, 1))
            P_i = weights.mean(dim=0)
            aux = (f_i.detach() * P_i).sum() * self.num_routed

        N = B * S
        flat_experts = experts.view(-1)
        flat_weights = w_topk.view(-1)
        flat_token_idx = torch.arange(N, device=x.device).unsqueeze(1).expand(-1, self.top_k).reshape(-1)

        sorted_indices = flat_experts.argsort(stable=True)
        sorted_token_idx = flat_token_idx[sorted_indices]
        sorted_weights = flat_weights[sorted_indices]
        sorted_experts = flat_experts[sorted_indices]

        expert_counts = torch.bincount(sorted_experts, minlength=self.num_routed).tolist()

        start = 0
        for e in range(self.num_routed):
            count = expert_counts[e]
            if count == 0:
                continue
            end = start + count
            idx = sorted_token_idx[start:end]
            w = sorted_weights[start:end]
            ts = x_flat[idx]
            expert_out = self._forward_expert(ts, self.gate_proj[e], self.up_proj[e], self.down_proj[e])
            out.index_add_(0, idx, (expert_out * w.unsqueeze(-1)).to(x.dtype))
            start = end
        return out.view(B, S, D), aux


class MoETransformerBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.input_layernorm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.self_attn = GQAAttention(config)
        self.moe = SparseMoE(config)

    def forward(self, x, attention_mask=None, position_ids=None):
        x = x + self.self_attn(self.input_layernorm(x), attention_mask, position_ids)
        h, aux = self.moe(self.post_attention_layernorm(x))
        return x + h, aux


# ============================================================
# Causal LM
# ============================================================

@dataclass
class CausalLMOutputWithPast:
    loss: Optional[torch.Tensor] = None
    logits: torch.Tensor = None


class TinyMixtralForCausalLM(PreTrainedModel):
    config_class = TinyMixtralConfig
    base_model_prefix = "tinymixtral"
    supports_gradient_checkpointing = True
    _no_split_modules = ["MoETransformerBlock"]

    def __init__(self, config):
        super().__init__(config)
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList([MoETransformerBlock(config) for _ in range(config.num_hidden_layers)])
        self.norm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        if config.tie_word_embeddings:
            self.lm_head.weight = self.embed_tokens.weight
        self._use_activation_checkpointing = False
        self.post_init()

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

    def forward(self, input_ids, attention_mask=None, labels=None, return_dict=True, **kwargs):
        B, S = input_ids.shape
        pos = torch.arange(S, device=input_ids.device).unsqueeze(0).expand(B, -1)
        cmask = attention_mask.bool() if attention_mask is not None else None

        h = self.embed_tokens(input_ids)
        total_aux = torch.tensor(0.0, device=input_ids.device, dtype=torch.float32)
        for layer in self.layers:
            if self._use_activation_checkpointing and self.training:
                h, aux = checkpoint(layer, h, cmask, pos, use_reentrant=False)
            else:
                h, aux = layer(h, cmask, pos)
            total_aux = total_aux + aux
        logits = self.lm_head(self.norm(h)).float()

        loss = None
        if labels is not None:
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                labels.reshape(-1),
                ignore_index=-100,
            )
            loss = loss + self.config.router_aux_loss_coef * (total_aux / len(self.layers))

        if not return_dict:
            return (loss, logits) if loss is not None else (logits,)
        return CausalLMOutputWithPast(loss=loss, logits=logits)
