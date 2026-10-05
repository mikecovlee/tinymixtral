import math
import unittest

import torch

from cpt_model import CPTConfig, CPTForCausalLM, CPTLayerProposal, CPTRouter


def tiny(k=2, n=4, chunk=2, **over):
    return CPTConfig(vocab_size=32, hidden_size=16, num_hidden_layers=1,
        num_attention_heads=2, num_key_value_heads=1, head_dim=8,
        num_local_experts=n, num_experts_per_tok=k, expert_intermediate_size=24,
        cpt_state_chunk_size=chunk, **over)


class AdaptiveTests(unittest.TestCase):
    def test_derived_geometry_and_serialization(self):
        for n in (2, 4, 6):
            cfg = tiny(n=n)
            self.assertEqual(cfg.cpt_projection_dim, 2*n-1)
            self.assertAlmostEqual(cfg.cpt_prototype_temperature, 1/math.sqrt(2*n-1), places=7)
            self.assertEqual(CPTConfig.from_dict(cfg.to_dict()).to_dict(), cfg.to_dict())
            router = CPTRouter(cfg, 0)
            self.assertEqual(tuple(router.projection.shape), (2*n-1, 16))
        for kwargs in ({'cpt_projection_dim': 16}, {'cpt_prototype_temperature': 0.25},
                       {'num_experts_per_tok': True}, {'num_experts_per_tok': 7}):
            with self.assertRaises(ValueError):
                CPTConfig(**kwargs)

    def test_actual_assignment_price(self):
        router = CPTRouter(tiny(), 0)
        # Fixed k=2, fixed k=4, mixed per-token k=[1,3,2], dropped/all-padding.
        for hits, count in (([3,3,0,0], 3), ([3,3,3,3], 3), ([3,2,1,0], 3),
                            ([0,0,0,0], 3), ([0,0,0,0], 0)):
            router.congestion_price.copy_(torch.tensor([0.03, -0.01, -0.01, -0.01]))
            proposal = CPTLayerProposal(0, torch.tensor(hits), torch.tensor(count), router.state_version.clone())
            old = router.congestion_price.clone()
            prepared = router.prepare_commit(proposal, 0)
            if sum(hits):
                shares = torch.tensor(hits, dtype=torch.float32)/sum(hits)
                lower = (2-router.capacity_factor)/4
                upper = router.capacity_factor/4
                controls = torch.tensor([float(v)-lower if v < lower else float(v)-upper if v > upper else 0 for v in shares])
                expected = old + router.price_learning_rate*controls
                expected -= expected.mean()
            else:
                expected = old
            torch.testing.assert_close(prepared[1], expected)
        bad = CPTLayerProposal(0, torch.tensor([4,0,0,0]), torch.tensor(3), router.state_version.clone())
        with self.assertRaises(RuntimeError):
            router.validate_proposal(bad)

    def test_fixed_and_dynamic_host_dispatch(self):
        tokens = torch.tensor([[1,2,3,4]])
        mask = torch.tensor([[1,1,1,0]])
        for initial_k in (2,4):
            model = CPTForCausalLM(tiny(initial_k)).train()
            model.gradient_checkpointing_enable()
            for k in (initial_k, 1, 4, 2):
                model.layers[0].moe.top_k = k
                out = model(tokens, attention_mask=mask, labels=tokens)
                proposal = out['cpt_transaction'].proposals[0]
                self.assertEqual(int(proposal.expert_hits.sum()), 3*k)
                self.assertEqual(int(proposal.token_count), 3)
                self.assertTrue(torch.isfinite(out['loss']))
                out['loss'].backward()
                self.assertTrue(torch.isfinite(model.layers[0].moe.cpt_router.projection.grad).all())
                model.commit_cpt_transaction(out['cpt_transaction'])
                model.zero_grad(set_to_none=True)
            clone = CPTForCausalLM(tiny(initial_k))
            clone.load_state_dict(model.state_dict())
            model.layers[0].moe.top_k = initial_k
            model.eval()
            clone.eval()
            with torch.no_grad():
                torch.testing.assert_close(model(tokens)['logits'], clone(tokens)['logits'], rtol=0, atol=0)



    def test_responsibility_trajectory_stays_finite_for_long_chunks(self):
        # F1 regression: the rho^{-b} factorization overflowed FP32 past ~1700
        # valid tokens per chunk (exclusive prefix then hit inf - inf -> NaN).
        torch.manual_seed(3)
        rows, length, k = 2, 3000, 8
        q = torch.softmax(torch.randn(rows, length, k), dim=-1)
        valid = torch.ones(rows, length, dtype=torch.bool)
        valid[1, 1000:] = False
        nu_old = torch.rand(rows, k)
        rho = torch.tensor(0.95)
        trajectory = CPTRouter._responsibility_trajectory(valid.float(), q, nu_old, rho)
        self.assertTrue(bool(torch.isfinite(trajectory).all()))
        reference = torch.empty_like(trajectory)
        for row in range(rows):
            nu = nu_old[row].clone()
            for t in range(length):
                reference[row, t] = nu
                if valid[row, t]:
                    nu = rho * nu + q[row, t]
        torch.testing.assert_close(trajectory, reference, rtol=1e-3, atol=1e-5)

    def test_long_chunk_router_forward_stays_finite(self):
        # End-to-end: one chunk holding > 1700 valid tokens must not produce NaN.
        cfg = tiny(chunk=3000, cpt_state_step_size="1/2000")
        router = CPTRouter(cfg, 0)
        torch.manual_seed(4)
        x = torch.randn(1, 3000, 16)
        mask = torch.ones(1, 3000, dtype=torch.bool)
        mask[0, 2500:] = False
        with torch.no_grad():
            out = router(x, attention_mask=mask).probabilities
        self.assertTrue(bool(torch.isfinite(out).all()))
        torch.testing.assert_close(out.sum(dim=-1)[mask], torch.ones(2500), rtol=0, atol=1e-5)

    def test_default_chunk_tracks_sequential_routing(self):
        # F2 regression: oversized aggregated state steps saturated the radius
        # ball and distorted routing; the default chunk must stay faithful to
        # the strict sequential (chunk=1) semantics.
        torch.manual_seed(9)
        x = torch.randn(1, 256, 16)
        with torch.no_grad():
            reference = CPTRouter(tiny(chunk=1), 0)(x).probabilities
        for chunk, step in ((16, None), (32, "1/20")):
            over = {} if step is None else {"cpt_state_step_size": step}
            router = CPTRouter(tiny(chunk=chunk, **over), 0)
            router.load_state_dict(CPTRouter(tiny(chunk=1), 0).state_dict())
            with torch.no_grad():
                probabilities = router(x).probabilities
            self.assertLess(float((probabilities - reference).abs().max()), 0.01)

    def test_state_step_bound_includes_chunk_factor(self):
        # F3 regression: the step bound must scale with the chunk size.
        self.assertEqual(CPTConfig().cpt_state_chunk_size, 16)
        with self.assertRaises(ValueError):
            tiny(chunk=128)  # default step exceeds 2 / (chunk * (1 + lambda))
        tiny(chunk=128, cpt_state_step_size="1/100")  # compliant step is accepted


if __name__ == '__main__':
    torch.set_num_threads(1)
    unittest.main()
