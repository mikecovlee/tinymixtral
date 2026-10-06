"""Numerical contract tests for the exact clamped-L2 mapping used by CPT."""
import unittest

import torch
from torch.fx.experimental.proxy_tensor import make_fx

from cpt_model import stable_l2


def reference(x, dim, eps):
    return x / torch.linalg.vector_norm(x, dim=dim, keepdim=True).clamp_min(eps)


class FiniteInterpreter(torch.fx.Interpreter):
    def run_node(self, node):
        value = super().run_node(node)
        if isinstance(value, torch.Tensor) and value.is_floating_point():
            if not torch.isfinite(value).all():
                raise AssertionError(f'Non-finite intermediate at {node.name}: {node.target}')
        return value


class StableL2Tests(unittest.TestCase):
    def test_forward_and_vjp_match_native_across_shapes_and_scales(self):
        torch.manual_seed(9)
        for shape, dim in (((3, 7), -1), ((2, 7, 8), 1), ((2, 3, 7, 8), 2)):
            for scale in (0.0, 1e-9, 1.0, 1e10):
                with self.subTest(shape=shape, dim=dim, scale=scale):
                    x = (torch.randn(shape) * scale).requires_grad_()
                    g = torch.randn_like(x)
                    expected = reference(x, dim, 1e-6)
                    actual = stable_l2(x, dim, 1e-6)
                    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                    old_grad, = torch.autograd.grad(expected, x, g)
                    new_grad, = torch.autograd.grad(actual, x, g)
                    self.assertTrue(torch.isfinite(new_grad).all())
                    torch.testing.assert_close(new_grad, old_grad, rtol=2e-5, atol=2e-6)

    def test_first_and_second_derivatives(self):
        torch.manual_seed(11)
        for scale in (0.0, 0.01, 1.0):
            x = (torch.randn(2, 7, dtype=torch.double) * scale).requires_grad_()
            def f(value):
                return stable_l2(value, -1, 0.125)
            self.assertTrue(torch.autograd.gradcheck(f, (x,)))
            self.assertTrue(torch.autograd.gradgradcheck(f, (x,)))

    def test_inclusive_clamp_boundary(self):
        eps = 0.125  # exactly representable, avoiding a rounded boundary input
        x = torch.zeros(7, dtype=torch.double)
        x[0] = eps
        actual = torch.autograd.functional.jacobian(lambda value: stable_l2(value, 0, eps), x)
        expected = torch.eye(7, dtype=torch.double) / eps
        expected[0, 0] = 0
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_zero_vector_backward_has_no_nonfinite_intermediates(self):
        x = torch.zeros(2, 7, requires_grad=True)
        g = torch.ones_like(x)
        def backward(value, incoming):
            return torch.autograd.grad(stable_l2(value, -1, 1e-6), value, incoming)[0]
        graph = make_fx(backward)(x, g)
        actual = FiniteInterpreter(graph).run(x, g)
        torch.testing.assert_close(actual, g / 1e-6, rtol=0, atol=0)


if __name__ == '__main__':
    torch.set_num_threads(1)
    unittest.main()
