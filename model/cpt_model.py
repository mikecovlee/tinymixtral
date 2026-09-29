from typing import Optional, Tuple
import torch
from torch import nn
from .cpt_router import CPT_ROUTER_ALGORITHM_VERSION, CPTTransaction, CPTRouter


class CPTModelMixin:
    def _cpt_routers(self) -> Tuple[CPTRouter, ...]:
        return tuple(layer.moe.cpt_router for layer in self.layers)

    def cpt_trainable_parameters(self) -> Tuple[nn.Parameter, ...]:
        return tuple(parameter for router in self._cpt_routers() for parameter in router.trainable_parameters())

    def _validate_cpt_config_binding(self) -> None:
        for layer_index, (layer, router) in enumerate(zip(self.layers, self._cpt_routers())):
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
        versions = [int(router.state_version.item()) for router in self._cpt_routers()]
        if not versions:
            raise RuntimeError("model has no CPT Routers")
        if len(set(versions)) != 1:
            raise RuntimeError(f"CPT layer state versions disagree: {versions}")
        return versions[0]

    def get_cpt_optimizer_step(self) -> int:
        optimizer_steps = [int(router.optimizer_step.item()) for router in self._cpt_routers()]
        if not optimizer_steps:
            raise RuntimeError("model has no CPT Routers")
        if len(set(optimizer_steps)) != 1:
            raise RuntimeError("CPT layer optimizer steps disagree: " f"{optimizer_steps}")
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
            raise RuntimeError("training state is fail-stop poisoned; restore the last " "successful checkpoint")
        try:
            self.validate_cpt_transaction(transaction)
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
                failure_types = ", ".join(type(error).__name__ for error in recovery_errors)
                raise RuntimeError("CPT commit failed and recovery was incomplete: " + failure_types) from recovery_errors[0]
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
            raise RuntimeError("legacy Linear Router checkpoints are incompatible with CPT v1: " + ", ".join(legacy))

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
        self._validate_cpt_config_binding()
        self._validate_serialized_cpt_state(state_dict)
        result = super().load_state_dict(
            state_dict,
            strict=strict,
            assign=assign,
        )
        self.validate_persistent_cpt_state()
        return result
