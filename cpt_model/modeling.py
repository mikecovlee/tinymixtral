# Copyright (C) Michael Lee (李登淳) 2026. All rights reserved.
# Open-source under the MIT License. See LICENSE for details.

"""CPT 模型变体：以薄子类叠加在主线模型之上，model/ 与 main 保持零差异。

- CPTSparseMoE 复用主线的 expert 参数/初始化/keepalive，仅替换路由器；
- CPTBlock / CPTForCausalLM 保持主线的 forward 签名与 (out, aux) 形状，
  模型 forward 整体复用主线实现，事务在 forward 返回后从各层 MoE 的
  模块状态收集（梯度检查点重算发生在其后且值相同）。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.modeling import (
    GQAAttention,
    MoETransformerBlock,
    RMSNorm,
    SparseMoE,
    TinyMixtralForCausalLM,
)

from .config import CPTConfig, config_from_json_file
from .protocol import CPTModelMixin
from .router import CPTLayerProposal, CPTRouter, CPTTransaction


class CPTSparseMoE(SparseMoE):
    """Mixtral SparseMoE with the CPT probability router.

    主线的 expert 计算、初始化与零 token 梯度保活语义保持不变；
    padding 感知的 CPT 路由与实际分配统计由本类提供。
    """

    def __init__(self, config: CPTConfig, layer_index: int = 0):
        super().__init__(config)
        del self.router  # 线性路由器不进入 CPT 的 state_dict
        self.cpt_router = CPTRouter(config, layer_index)
        self.last_token_count: torch.Tensor | None = None

    def forward(self, x: torch.Tensor, attention_mask: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: [batch_size, seq_len, hidden_size]
            attention_mask: [batch_size, seq_len]，1=valid；padding token 不参与分配
        Returns:
            out: [batch_size, seq_len, hidden_size]
            aux_loss: scalar tensor（CPT 无 auxiliary loss，恒为 0）
        """
        B, S, D = x.shape
        x_flat = x.view(-1, D)  # [B*S, D]
        N = B * S

        probabilities = self.cpt_router(x, attention_mask=attention_mask).probabilities.reshape(-1, self.num_experts)
        valid = (torch.arange(N, device=x.device) if attention_mask is None else
                 torch.nonzero(attention_mask.reshape(-1).bool(), as_tuple=False).flatten())
        if type(self.top_k) is not int or not 1 <= self.top_k <= self.num_experts:
            raise ValueError("host top_k must be an integer in [1, num_experts]")
        routing_weights_topk, selected_experts = torch.topk(probabilities.index_select(0, valid), self.top_k, dim=-1)
        routing_weights_topk = routing_weights_topk / routing_weights_topk.sum(dim=-1, keepdim=True)
        routing_weights_topk = routing_weights_topk.to(x.dtype)
        aux_loss = torch.zeros((), device=x.device, dtype=torch.float32)
        expert_hits = torch.bincount(selected_experts.reshape(-1), minlength=self.num_experts)
        self.last_expert_counts = expert_hits.detach().clone()
        self.last_token_count = torch.tensor(valid.numel(), device=x.device, dtype=torch.int64)

        flat_experts = selected_experts.view(-1)
        flat_weights = routing_weights_topk.view(-1)
        flat_token_idx = valid.unsqueeze(1).expand(-1, self.top_k).reshape(-1)

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

    def cpt_layer_proposal(self, layer_index: int) -> CPTLayerProposal:
        """Snapshot this forward's actual host dispatch for the CPT transaction."""
        return CPTLayerProposal(
            layer_index,
            self.last_expert_counts.detach().clone(),
            self.last_token_count.detach().clone(),
            self.cpt_router.state_version.detach().clone(),
        )


class CPTBlock(MoETransformerBlock):
    """主线 Transformer 层 + CPT MoE；forward 形状与主线一致。"""

    def __init__(self, config: CPTConfig, layer_index: int = 0):
        # 不调用 super().__init__：避免构建随即丢弃的整套线性 MoE 参数。
        nn.Module.__init__(self)
        self.input_layernorm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.self_attn = GQAAttention(config)
        self.moe = CPTSparseMoE(config, layer_index)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # 与主线逐行一致，仅把 attention_mask 传给 MoE（CPT padding 感知路由）。
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(hidden_states, attention_mask, position_ids)
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states, aux_loss = self.moe(hidden_states, attention_mask)
        hidden_states = residual + hidden_states

        return hidden_states, aux_loss


class CPTForCausalLM(CPTModelMixin, TinyMixtralForCausalLM):
    """CPT 因果语言模型：主线模型 + CPT 事务协议。"""

    is_cpt = True

    def __init__(self, config: CPTConfig):
        # 不调用 super().__init__：避免构建随即丢弃的整套线性 MoE 层。
        nn.Module.__init__(self)
        self.config = config

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList([
            CPTBlock(config, i) for i in range(config.num_hidden_layers)
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

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
        return_dict: bool = True,
    ) -> dict:
        output = super().forward(input_ids, attention_mask=attention_mask, labels=labels, return_dict=return_dict)
        proposals = tuple(layer.moe.cpt_layer_proposal(i) for i, layer in enumerate(self.layers))
        output["cpt_transaction"] = CPTTransaction(proposals, training_forward=bool(self.training))
        return output

    def save_pretrained(self, path: str):
        self.validate_persistent_cpt_state()
        super().save_pretrained(path)

    @classmethod
    def from_pretrained(cls, path: str, config: CPTConfig | None = None) -> "CPTForCausalLM":
        """从 HF 格式加载 CPT 模型（拒绝线性路由器 checkpoint）。"""
        if config is None:
            config = config_from_json_file(f"{path}/config.json")
        if not isinstance(config, CPTConfig):
            raise RuntimeError(
                "checkpoint is not a CPT model; load it with model.modeling.TinyMixtralForCausalLM"
            )
        return super().from_pretrained(path, config=config)
