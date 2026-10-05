# Copyright (C) Michael Lee (李登淳) 2026. All rights reserved.
# Open-source under the MIT License. See LICENSE for details.

"""CPT transaction protocol for models that host CPTRouter layers.

A model forward produces one CPTTransaction carrying each layer's *actual*
host dispatch.  The training loop validates the transaction before an
optimizer update and commits it once afterwards; a commit updates every
Router's prices/anchors/state version atomically or none of them.

Gradient accumulation merges any sequence of transactions produced against
the same state version into a single price update.  Transactions produced by
non-training forwards and any commit under distributed process groups are
rejected: prices are process-local state and DDP/FSDP are unsupported.
"""

from collections.abc import Sequence

import torch
from torch import nn

from .cpt_router import CPT_ROUTER_ALGORITHM_VERSION, CPTLayerProposal, CPTRouter, CPTTransaction


class CPTModelMixin:
    def _require_cpt(self) -> None:
        if not getattr(self, "is_cpt", False):
            raise RuntimeError("this model uses the linear Router; CPT transactions are unavailable")

    def _cpt_routers(self) -> tuple[CPTRouter, ...]:
        return tuple(layer.moe.cpt_router for layer in self.layers)

    def cpt_trainable_parameters(self) -> tuple[nn.Parameter, ...]:
        self._require_cpt()
        return tuple(parameter for router in self._cpt_routers() for parameter in router.trainable_parameters())

    def no_weight_decay_parameters(self) -> tuple[nn.Parameter, ...]:
        """免权重衰减参数：CPT anchors 约束在单位球面，衰减无意义。"""
        if not getattr(self, "is_cpt", False):
            return ()
        return tuple(router.anchors for router in self._cpt_routers())

    def _validate_cpt_config_binding(self) -> None:
        for layer_index, (layer, router) in enumerate(zip(self.layers, self._cpt_routers(), strict=False)):
            router.validate_config_binding(self.config)
            if (
                type(layer.moe.top_k) is not int
                or not 1 <= layer.moe.top_k <= router.num_experts
            ):
                raise RuntimeError(f"MoE layer {layer_index} Top-k must be an integer in [1, num_experts]")

    @torch.no_grad()
    def validate_persistent_cpt_state(
        self,
        *,
        require_unit_anchors: bool = True,
    ) -> None:
        self._require_cpt()
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
            raise RuntimeError("CPT layer optimizer steps disagree: " f"{optimizer_steps}")

    def get_cpt_state_version(self) -> int:
        self._require_cpt()
        versions = [int(router.state_version.item()) for router in self._cpt_routers()]
        if not versions:
            raise RuntimeError("model has no CPT Routers")
        if len(set(versions)) != 1:
            raise RuntimeError(f"CPT layer state versions disagree: {versions}")
        return versions[0]

    def get_cpt_optimizer_step(self) -> int:
        self._require_cpt()
        optimizer_steps = [int(router.optimizer_step.item()) for router in self._cpt_routers()]
        if not optimizer_steps:
            raise RuntimeError("model has no CPT Routers")
        if len(set(optimizer_steps)) != 1:
            raise RuntimeError("CPT layer optimizer steps disagree: " f"{optimizer_steps}")
        return optimizer_steps[0]

    @staticmethod
    def _normalize_transactions(transactions) -> tuple:
        """Accept one transaction or a sequence (gradient accumulation)."""
        if transactions is None:
            raise TypeError("model output is missing a valid CPT transaction")
        if isinstance(transactions, CPTTransaction):
            normalized = (transactions,)
        elif isinstance(transactions, Sequence):
            normalized = tuple(transactions)
        else:
            raise TypeError("CPT transactions must be a CPTTransaction or a sequence of them")
        if not normalized:
            raise RuntimeError("no CPT transactions were provided")
        return normalized

    @staticmethod
    def _merge_proposals(routers: tuple, transactions: tuple) -> list:
        """Merge micro-batch proposals per layer.

        Actual dispatched loads add up across micro-batches; the merged
        proposal drives one price update and one state-version advance.
        Every transaction must already validate against the live state.
        """
        merged = []
        for layer_index, router in enumerate(routers):
            hits = torch.zeros(router.num_experts, dtype=torch.int64, device=router.congestion_price.device)
            token_count = 0
            version = router.state_version
            for transaction in transactions:
                proposal = transaction.proposals[layer_index]
                hits = hits + proposal.expert_hits
                token_count += int(proposal.token_count.item())
                version = proposal.state_version
            merged_proposal = CPTLayerProposal(
                layer_index,
                hits,
                router.state_version.new_tensor(token_count),
                version.detach().clone(),
            )
            router.validate_proposal(merged_proposal)
            merged.append(merged_proposal)
        return merged

    def validate_cpt_transaction(self, transactions) -> None:
        self._require_cpt()
        normalized = self._normalize_transactions(transactions)
        self._validate_cpt_config_binding()
        routers = self._cpt_routers()
        self.get_cpt_state_version()
        for transaction in normalized:
            if not isinstance(transaction, CPTTransaction):
                raise TypeError("model output is missing a valid CPT transaction")
            if transaction.consumed:
                raise RuntimeError("CPT transaction was already consumed")
            if not transaction.training_forward:
                raise RuntimeError("cannot commit CPT state from a non-training forward")
            if not isinstance(transaction.proposals, tuple):
                raise TypeError("CPT transaction proposals must be a tuple")
            if len(transaction.proposals) != len(routers):
                raise RuntimeError("CPT transaction does not cover every MoE layer")
            token_counts: list[int] = []
            for router, proposal in zip(routers, transaction.proposals, strict=False):
                router.validate_proposal(proposal)
                token_counts.append(int(proposal.token_count.item()))
            if token_counts and len(set(token_counts)) != 1:
                raise RuntimeError(f"CPT layer token counts disagree: {token_counts}")

    @torch.no_grad()
    def commit_cpt_transaction(
        self,
        transactions,
        *,
        optimizer_step: int | None = None,
    ) -> int:
        """Commit all Router prices/anchors/versions, or none of them.

        Accepts one transaction or a sequence of transactions produced since
        the last commit (gradient accumulation); their actual dispatched
        loads merge into a single price update.
        """
        self._require_cpt()
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            raise RuntimeError(
                "CPT commits require single-process training; DDP/FSDP are "
                "unsupported because congestion prices are process-local state"
            )
        normalized = self._normalize_transactions(transactions)
        try:
            self.validate_cpt_transaction(normalized)
            current_optimizer_step = self.get_cpt_optimizer_step()
            if optimizer_step is None:
                optimizer_step = current_optimizer_step
            elif type(optimizer_step) is not int or optimizer_step < 0:
                raise TypeError("CPT optimizer_step must be a non-negative integer")
            if optimizer_step not in (
                current_optimizer_step,
                current_optimizer_step + 1,
            ):
                raise RuntimeError("CPT optimizer_step must stay unchanged or advance by one")
        except BaseException:
            for transaction in normalized:
                if isinstance(transaction, CPTTransaction):
                    transaction.consumed = True
            raise
        # A commit attempt is single-use.  Any later preparation, write, or
        # validation failure discards these proposals even when rollback succeeds.
        for transaction in normalized:
            transaction.consumed = True
        routers = self._cpt_routers()
        merged = self._merge_proposals(routers, normalized)
        prepared = [
            router.prepare_commit(
                proposal,
                optimizer_step=optimizer_step,
            )
            for router, proposal in zip(routers, merged, strict=False)
        ]
        snapshots = [router.commit_snapshot() for router in routers]
        try:
            for router, candidate in zip(routers, prepared, strict=False):
                router.apply_commit(*candidate)
            self.validate_persistent_cpt_state()
        except BaseException:
            recovery_errors = []
            for router, snapshot in zip(routers, snapshots, strict=False):
                try:
                    router.restore_commit_snapshot(snapshot)
                except BaseException as error:
                    recovery_errors.append(error)
            if recovery_errors:
                self._tinymixtral_fail_stop = True
                failure_types = ", ".join(type(error).__name__ for error in recovery_errors)
                raise RuntimeError("CPT commit failed and recovery was incomplete: " + failure_types) from recovery_errors[0]
            raise
        return self.get_cpt_state_version()

    @staticmethod
    def abort_cpt_transaction(transactions) -> None:
        if transactions is None:
            return
        if isinstance(transactions, CPTTransaction):
            transactions = (transactions,)
        elif not isinstance(transactions, Sequence):
            raise TypeError("cannot abort an invalid CPT transaction")
        for transaction in transactions:
            if not isinstance(transaction, CPTTransaction):
                raise TypeError("cannot abort an invalid CPT transaction")
            transaction.consumed = True

    def _validate_serialized_cpt_state(self, state_dict) -> None:
        if not hasattr(state_dict, "keys"):
            raise TypeError("state_dict must be a mapping")
        keys = tuple(state_dict.keys())
        legacy = [key for key in keys if key.endswith(".moe.router.weight")]
        if legacy:
            raise RuntimeError("legacy Linear Router checkpoints are incompatible with the CPT Router: " + ", ".join(legacy))

        marker = ".moe.cpt_router."
        expected_state = super().state_dict()
        expected_keys = {key for key in expected_state if marker in key}
        supplied_keys = {key for key in keys if marker in key}
        missing = sorted(expected_keys - supplied_keys)
        unexpected = sorted(supplied_keys - expected_keys)
        if missing:
            raise RuntimeError("checkpoint is missing CPT keys: " + ", ".join(missing))
        if unexpected:
            raise RuntimeError("checkpoint contains unknown CPT keys: " + ", ".join(unexpected))

        serialized_versions: list[int] = []
        serialized_optimizer_steps: list[int] = []
        for key in sorted(expected_keys):
            value = state_dict[key]
            expected = expected_state[key]
            if not isinstance(value, torch.Tensor):
                raise RuntimeError(f"checkpoint CPT value is not a tensor: {key}")
            if value.shape != expected.shape:
                raise RuntimeError(
                    f"checkpoint CPT shape mismatch for {key}: " f"expected {tuple(expected.shape)}, got {tuple(value.shape)}"
                )
            if value.dtype != expected.dtype:
                raise RuntimeError(f"checkpoint CPT dtype mismatch for {key}: " f"expected {expected.dtype}, got {value.dtype}")
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
                price_scale = max(1.0, float(value.abs().amax().item()))
                if abs(float(value.mean().item())) > 2e-6 * price_scale:
                    raise RuntimeError("checkpoint CPT congestion price is not zero-centered")
            elif key.endswith(".router_algorithm_version"):
                if int(value.item()) != CPT_ROUTER_ALGORITHM_VERSION:
                    raise RuntimeError("checkpoint uses an unsupported CPT algorithm version")
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
            raise RuntimeError(f"checkpoint CPT layer state versions disagree: {serialized_versions}")
        if serialized_optimizer_steps and len(set(serialized_optimizer_steps)) != 1:
            raise RuntimeError("checkpoint CPT layer optimizer steps disagree: " f"{serialized_optimizer_steps}")
        if serialized_versions and serialized_optimizer_steps and serialized_optimizer_steps[0] > serialized_versions[0]:
            raise RuntimeError("checkpoint CPT optimizer_step exceeds state_version")

    def load_state_dict(self, state_dict, strict: bool = True, assign: bool = False):
        if not getattr(self, "is_cpt", False):
            return super().load_state_dict(state_dict, strict=strict, assign=assign)
        self._validate_cpt_config_binding()
        self._validate_serialized_cpt_state(state_dict)
        result = super().load_state_dict(
            state_dict,
            strict=strict,
            assign=assign,
        )
        self.validate_persistent_cpt_state()
        return result
