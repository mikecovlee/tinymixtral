"""CPT compilation, differentiation, cache and activation-checkpoint regressions."""
import os
import unittest
from unittest.mock import patch

import torch
from torch.utils.checkpoint import checkpoint

from model.config import TinyMixtralConfig
from model.cpt_router import CPTRouter, CPTRouterOutput


def router_config(hidden_size=16, chunk_size=2):
    return TinyMixtralConfig(
        hidden_size=hidden_size,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=hidden_size // 2,
        num_local_experts=4,
        num_experts_per_tok=2,
        cpt_state_chunk_size=chunk_size,
    )


def assert_backward_matches(test, eager, candidate, forward, inputs, mask,
                            *, rtol=3e-5, atol=3e-6, autocast_enabled=False):
    eager.zero_grad(set_to_none=True)
    candidate.zero_grad(set_to_none=True)
    reference_input = inputs.detach().clone().requires_grad_(True)
    candidate_input = inputs.detach().clone().requires_grad_(True)
    with torch.autocast(device_type=inputs.device.type, dtype=torch.bfloat16,
                        enabled=autocast_enabled):
        expected = eager._forward_impl(reference_input, mask).probabilities
        actual = forward(candidate_input, mask).probabilities
    test.assertTrue(torch.isfinite(actual).all())
    torch.testing.assert_close(actual, expected, rtol=rtol, atol=atol)

    # Unequal weights avoid the constant sum(probabilities) objective.
    weights = torch.arange(1, eager.num_experts + 1, device=inputs.device, dtype=torch.float32)
    (expected * weights).sum().backward()
    (actual * weights).sum().backward()
    candidate_parameters = dict(candidate.named_parameters())
    gradients = [('input', reference_input.grad, candidate_input.grad)]
    gradients.extend(
        (name, parameter.grad, candidate_parameters[name].grad)
        for name, parameter in eager.named_parameters()
    )
    for name, reference_gradient, candidate_gradient in gradients:
        test.assertIsNotNone(reference_gradient, name)
        test.assertIsNotNone(candidate_gradient, name)
        test.assertTrue(torch.isfinite(reference_gradient).all(), name)
        test.assertTrue(torch.isfinite(candidate_gradient).all(), name)
        if candidate_gradient.dtype == torch.bfloat16:
            # A one-ULP rounding change can fail elementwise relative tolerance
            # near cancellation. Bound both aggregate and tensor-scale error.
            reference_float = reference_gradient.float()
            difference = candidate_gradient.float() - reference_float
            precision = torch.finfo(torch.bfloat16).eps
            relative_l2 = difference.norm() / reference_float.norm().clamp_min(1e-30)
            scaled_max = difference.abs().max() / reference_float.abs().max().clamp_min(1e-30)
            test.assertLessEqual(float(relative_l2), precision / 2, name)
            test.assertLessEqual(float(scaled_max), precision, name)
        else:
            torch.testing.assert_close(candidate_gradient, reference_gradient,
                                       rtol=rtol, atol=atol)


class CodeUpdateTests(unittest.TestCase):
    def test_compile_cache_and_device_reset_preserve_global_policy(self):
        import torch._inductor.config as inductor_config

        original_policy = inductor_config.shape_padding
        router = CPTRouter(router_config(), 0)
        compiled = object()
        with patch('torch.compile', return_value=compiled) as factory:
            self.assertIs(router._get_compiled_forward(), compiled)
            self.assertIs(router._get_compiled_forward(), compiled)
            factory.assert_called_once_with(router._forward_impl)
            self.assertEqual(inductor_config.shape_padding, original_policy)
            router.to('cpu')
            self.assertIsNone(router._compiled_forward_impl)
            self.assertIs(router._get_compiled_forward(), compiled)
            self.assertEqual(factory.call_count, 2)

    def test_dp7_aot_autograd_forward_backward(self):
        torch.manual_seed(17)
        eager = CPTRouter(router_config(), 0)
        candidate = CPTRouter(router_config(), 0)
        candidate.load_state_dict(eager.state_dict())
        compiled = torch.compile(candidate._forward_impl, backend='aot_eager')
        random_input = torch.randn(2, 5, 16)
        full_mask = torch.ones(2, 5, dtype=torch.bool)
        partial_mask = torch.tensor([[1, 1, 0, 1, 0], [0, 0, 0, 0, 0]], dtype=torch.bool)
        cases = (
            (random_input, partial_mask),
            (torch.zeros_like(random_input), full_mask),
            (random_input * 1e-8, full_mask),
            (random_input, torch.zeros_like(full_mask)),
        )
        for inputs, mask in cases:
            with self.subTest(scale=float(inputs.abs().max()), valid=int(mask.sum())):
                assert_backward_matches(self, eager, candidate, compiled, inputs, mask)

    @unittest.skipUnless(os.environ.get('CPT_RUN_CUDA_REGRESSION') == '1',
                         'CUDA regression is explicitly opt-in')
    def test_dp7_cuda_inductor_forward_backward_and_checkpoint(self):
        self.assertTrue(torch.cuda.is_available(), 'CUDA device required')
        torch.manual_seed(17)
        config = router_config(hidden_size=1024, chunk_size=128)
        eager = CPTRouter(config, 0).cuda().train()
        candidate = CPTRouter(config, 0).cuda().train()
        candidate.load_state_dict(eager.state_dict())
        random_input = torch.randn(2, 257, 1024, device='cuda', dtype=torch.bfloat16)
        mask = torch.ones(2, 257, device='cuda', dtype=torch.bool)
        mask[1, 129:] = False
        for inputs in (random_input, torch.zeros_like(random_input), random_input * 1e-8):
            assert_backward_matches(self, eager, candidate, candidate, inputs, mask,
                                    rtol=1e-4, atol=2e-5, autocast_enabled=True)

        def checkpointed(inputs, attention_mask):
            probabilities = checkpoint(
                lambda hidden: candidate(hidden, attention_mask).probabilities,
                inputs,
                use_reentrant=False,
            )
            return CPTRouterOutput(probabilities=probabilities)

        assert_backward_matches(self, eager, candidate, checkpointed, random_input, mask,
                                rtol=1e-4, atol=2e-5, autocast_enabled=True)
        self.assertIsNotNone(candidate._compiled_forward_impl)


if __name__ == '__main__':
    torch.set_num_threads(1)
    unittest.main()
