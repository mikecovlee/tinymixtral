# Copyright (C) Michael Lee (李登淳) 2026. All rights reserved.
# Open-source under the MIT License. See LICENSE for details.

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from transformers import PreTrainedModel, GenerationMixin
from transformers.modeling_outputs import ModelOutput

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

    def forward(self, hidden_states, attention_mask=None, position_ids=None, past_key_value=None, use_cache=False):
        B, S, _ = hidden_states.shape
        q = self.q_proj(hidden_states).view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(hidden_states).view(B, S, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(hidden_states).view(B, S, self.num_kv_heads, self.head_dim).transpose(1, 2)

        cache_len = past_key_value[0].shape[2] if past_key_value is not None else 0
        if position_ids is None:
            position_ids = torch.arange(cache_len, cache_len + S, device=hidden_states.device).unsqueeze(0).expand(B, -1)
        q, k = self.rotary_emb(q, position_ids), self.rotary_emb(k, position_ids)

        if past_key_value is not None:
            k = torch.cat([past_key_value[0], k], dim=2)
            v = torch.cat([past_key_value[1], v], dim=2)
        cache = (k, v) if use_cache else None
        total_len = cache_len + S

        if attention_mask is not None or cache_len > 0:
            k_exp = k.unsqueeze(2).expand(-1, -1, self.num_groups, -1, -1).reshape(B, self.num_heads, total_len, self.head_dim)
            v_exp = v.unsqueeze(2).expand(-1, -1, self.num_groups, -1, -1).reshape(B, self.num_heads, total_len, self.head_dim)
            causal = torch.tril(torch.ones(S, total_len, device=hidden_states.device, dtype=torch.bool), diagonal=cache_len)
            if attention_mask is not None:
                mask = causal[None, None, :, :] & attention_mask[:, None, None, :]
            else:
                mask = causal[None, None, :, :]
            attn = F.scaled_dot_product_attention(
                q, k_exp, v_exp, attn_mask=mask,
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
        return self.o_proj(attn.transpose(1, 2).reshape(B, S, -1)), cache


class SparseMoE(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_experts = config.num_local_experts
        self.top_k = config.num_experts_per_tok
        self.expert_intermediate = config.expert_intermediate_size
        self.jitter_noise = config.router_jitter_noise
        self.aux_loss_coef = config.router_aux_loss_coef
        self.router = nn.Linear(self.hidden_size, self.num_experts, bias=False)
        self.gate_proj = nn.Parameter(torch.empty(self.num_experts, self.expert_intermediate, self.hidden_size))
        self.up_proj = nn.Parameter(torch.empty(self.num_experts, self.expert_intermediate, self.hidden_size))
        self.down_proj = nn.Parameter(torch.empty(self.num_experts, self.hidden_size, self.expert_intermediate))
        self._init_weights()

    def _init_weights(self, std=0.02):
        nn.init.normal_(self.gate_proj, std=std)
        nn.init.normal_(self.up_proj, std=std)
        nn.init.normal_(self.down_proj, std=std)

    def forward(self, x):
        B, S, D = x.shape
        x_flat = x.view(-1, D)
        N = B * S
        logits = self.router(x_flat)
        if self.training and self.jitter_noise > 0:
            logits = logits * (1 + torch.randn_like(logits) * self.jitter_noise)
        weights = F.softmax(logits.float(), dim=-1).to(x.dtype)
        w_topk, experts = torch.topk(weights, self.top_k, dim=-1)
        w_topk = w_topk / w_topk.sum(dim=-1, keepdim=True)

        aux = torch.tensor(0.0, device=x.device, dtype=x.dtype)
        if self.training and self.aux_loss_coef > 0:
            with torch.no_grad():
                mask = F.one_hot(experts, num_classes=self.num_experts).float()
                f_i = mask.mean(dim=(0, 1))
            P_i = weights.mean(dim=0)
            aux = (f_i.detach() * P_i).sum() * self.num_experts

        flat_experts = experts.view(-1)
        flat_weights = w_topk.view(-1)
        flat_token_idx = torch.arange(N, device=x.device).unsqueeze(1).expand(-1, self.top_k).reshape(-1)

        sorted_indices = flat_experts.argsort(stable=True)
        sorted_token_idx = flat_token_idx[sorted_indices]
        sorted_weights = flat_weights[sorted_indices]
        sorted_experts = flat_experts[sorted_indices]

        expert_counts = torch.bincount(sorted_experts, minlength=self.num_experts).tolist()

        out = torch.zeros(N, D, device=x.device, dtype=x.dtype)
        start = 0
        for e in range(self.num_experts):
            count = expert_counts[e]
            if count == 0:
                continue
            end = start + count
            idx = sorted_token_idx[start:end]
            w = sorted_weights[start:end]
            ts = x_flat[idx]
            gate = F.silu(ts @ self.gate_proj[e].T)
            up = ts @ self.up_proj[e].T
            out.index_add_(0, idx, ((gate * up @ self.down_proj[e].T) * w.unsqueeze(-1)).to(x.dtype))
            start = end
        return out.view(B, S, D), aux


class MoETransformerBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.input_layernorm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.self_attn = GQAAttention(config)
        self.moe = SparseMoE(config)

    def forward(self, x, attention_mask=None, position_ids=None, past_key_value=None, use_cache=False):
        attn_out, new_cache = self.self_attn(
            self.input_layernorm(x), attention_mask, position_ids, past_key_value, use_cache
        )
        x = x + attn_out
        h, aux = self.moe(self.post_attention_layernorm(x))
        return x + h, aux, new_cache


# ============================================================
# Causal LM
# ============================================================

@dataclass
class CausalLMOutputWithPast(ModelOutput):
    loss: Optional[torch.Tensor] = None
    logits: torch.Tensor = None
    past_key_values: Optional[tuple] = None


class TinyMixtralForCausalLM(PreTrainedModel, GenerationMixin):
    config_class = TinyMixtralConfig
    base_model_prefix = "tinymixtral"
    supports_gradient_checkpointing = True
    _no_split_modules = ["MoETransformerBlock"]
    _supports_cache_class = False
    _supports_static_cache = False

    def _supports_default_dynamic_cache(self):
        return False

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
        if getattr(self.config, "eos_token_id", None) is None:
            self.config.eos_token_id = 2
            self.config.pad_token_id = 2

    def prepare_inputs_for_generation(self, input_ids, attention_mask=None, past_key_values=None, **kwargs):
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "past_key_values": past_key_values,
            "use_cache": kwargs.get("use_cache", True),
        }

    def _reorder_cache(self, past_key_values, beam_idx):
        return tuple(
            tuple(past.index_select(0, beam_idx) for past in layer_past)
            for layer_past in past_key_values
        )

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

    def forward(self, input_ids, attention_mask=None, labels=None, return_dict=True, past_key_values=None, use_cache=False, **kwargs):
        B, S = input_ids.shape
        past_len = past_key_values[0][0].shape[2] if past_key_values is not None else 0
        if past_len > 0 and S > past_len:
            input_ids = input_ids[:, past_len:]
            S = input_ids.shape[1]
        pos = torch.arange(past_len, past_len + S, device=input_ids.device).unsqueeze(0).expand(B, -1)
        cmask = attention_mask.bool() if attention_mask is not None else None

        h = self.embed_tokens(input_ids)
        total_aux = torch.tensor(0.0, device=input_ids.device, dtype=torch.float32)
        new_caches = []
        for i, layer in enumerate(self.layers):
            layer_cache = past_key_values[i] if past_key_values is not None else None
            if self._use_activation_checkpointing and self.training:
                h, aux, _ = checkpoint(layer, h, cmask, pos, None, False, use_reentrant=False)
            else:
                h, aux, layer_new_cache = layer(h, cmask, pos, layer_cache, use_cache)
                new_caches.append(layer_new_cache)
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

        past_key_values_out = tuple(new_caches) if use_cache else None
        if not return_dict:
            return (loss, logits, past_key_values_out) if loss is not None else (logits, past_key_values_out)
        return CausalLMOutputWithPast(loss=loss, logits=logits, past_key_values=past_key_values_out)
