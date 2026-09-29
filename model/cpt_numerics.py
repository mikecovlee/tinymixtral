"""CPT numerical primitives with explicitly bounded reverse-mode derivatives."""
import torch
from torch import Tensor


class _StableL2(torch.autograd.Function):
    """Exact clamped-L2 forward with a nonsingular analytic vector-Jacobian product.

    For a column vector x, d=max(||x||, eps), u=x/d and incoming gradient g:
        J(x)^T g = (g - 1[||x|| >= eps] u (u^T g)) / d.
    At the clamp boundary we retain PyTorch clamp_min's inclusive derivative.
    Every division uses d, including below the floor and at x=0; no raw
    x/||x|| intermediate needs to be repaired by masking after a division.
    """

    @staticmethod
    def forward(ctx, tensor: Tensor, dim: int, eps: float) -> Tensor:
        norm = torch.linalg.vector_norm(tensor, dim=dim, keepdim=True)
        denominator = norm.clamp_min(eps)
        ctx.save_for_backward(tensor, norm, denominator)
        ctx.dim = dim
        ctx.eps = eps
        return tensor / denominator

    @staticmethod
    def backward(ctx, grad_output: Tensor):
        tensor, norm, denominator = ctx.saved_tensors
        # Saved forward intermediates have no autograd history. Recompute only
        # when a caller explicitly requests a differentiable backward graph.
        if torch.is_grad_enabled():
            norm = torch.linalg.vector_norm(tensor, dim=ctx.dim, keepdim=True)
            denominator = norm.clamp_min(ctx.eps)
        unit = tensor / denominator
        radial_gradient = (unit * grad_output).sum(dim=ctx.dim, keepdim=True)
        correction = torch.where(norm >= ctx.eps, unit * radial_gradient, 0.0)
        return (grad_output - correction) / denominator, None, None


def stable_l2(tensor: Tensor, dim: int, eps: float) -> Tensor:
    """Normalize along dim using a positive floor; callers provide validated eps.

    CPT calls this on real FP32 tensors inside its existing autocast-disabled
    region. The forward preserves vector_norm followed by clamp_min and divide.
    """
    return _StableL2.apply(tensor, dim, eps)
