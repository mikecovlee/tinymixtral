"""Exact source expressions, one-step binary32 rounding and config persistence."""
import json
import struct
import tempfile
import unittest
from decimal import Decimal, localcontext
from fractions import Fraction
from pathlib import Path

from model.cpt_config import CPTConfig
from model.cpt_constants import exact_scalar, inverse_sqrt_fp32, rational_fp32


def bits(value):
    return struct.unpack('!I', struct.pack('!f', value))[0]


class ExactConstantTests(unittest.TestCase):
    def test_rounding_ties_and_binary64_double_rounding(self):
        midpoint = Fraction(1) + Fraction(1, 2**24)
        self.assertEqual(bits(rational_fp32('x', midpoint)), 0x3F800000)
        self.assertEqual(bits(rational_fp32('x', midpoint + Fraction(1, 2**80))), 0x3F800001)
        self.assertEqual(bits(rational_fp32('x', midpoint - Fraction(1, 2**80))), 0x3F800000)
        odd_midpoint = midpoint + Fraction(1, 2**23)
        self.assertEqual(bits(rational_fp32('x', odd_midpoint)), 0x3F800002)
        self.assertEqual(bits(rational_fp32('x', -midpoint)), 0xBF800000)

    def test_subnormal_and_overflow_boundaries(self):
        self.assertEqual(bits(rational_fp32('x', Fraction(1, 2**150))), 0)
        self.assertEqual(bits(rational_fp32('x', Fraction(3, 2**150))), 2)
        largest = Fraction((2**24 - 1) * 2**104)
        self.assertEqual(bits(rational_fp32('x', largest)), 0x7F7FFFFF)
        with self.assertRaises(ValueError):
            rational_fp32('x', (largest + 2**128) / 2)
        for value in (True, 'nan', 'inf', '1/0', 'sqrt(7)'):
            with self.assertRaises(ValueError):
                exact_scalar('x', value)

    def test_inverse_square_root_rounding(self):
        with localcontext() as context:
            context.prec = 100
            for dimension in (1, 3, 7, 11, 15, 31, 127, 1023):
                expected = float(Decimal(1) / Decimal(dimension).sqrt())
                self.assertEqual(bits(inverse_sqrt_fp32('tau', dimension)), bits(expected))
        self.assertEqual(bits(inverse_sqrt_fp32('tau', 7)), 0x3EC1848F)

    def test_derived_values_use_exact_inputs(self):
        cfg = CPTConfig(num_local_experts=4)
        expected = {
            'cpt_rho_beta': Fraction(19, 20),
            'cpt_beta_max': Fraction(9, 20),
            'cpt_kappa_beta': Fraction(5, 2),
            'cpt_lambda_sa': Fraction(1, 8),
            'cpt_state_step_size': Fraction(4, 45),
            'cpt_price_learning_rate': Fraction(3, 500),
        }
        for field, exact in expected.items():
            self.assertEqual(getattr(cfg, field), rational_fp32(field, exact))
        # This catches the former use of the rounded rho in kappa's denominator.
        self.assertEqual(cfg.cpt_kappa_beta, 2.5)
        custom = CPTConfig(num_local_experts=4, cpt_rho_beta='9/10',
                                   cpt_lambda_sa='1/3', cpt_expert_temperature='2/3',
                                   cpt_energy_init_scale=None)
        self.assertEqual(custom.cpt_kappa_beta, rational_fp32('kappa', Fraction(5, 4)))
        self.assertEqual(custom.cpt_state_step_size, rational_fp32('eta', Fraction(3, 40)))
        self.assertEqual(custom.cpt_price_learning_rate, rational_fp32('price', Fraction(1, 150)))
        self.assertEqual(custom.cpt_energy_init_scale, rational_fp32('energy', Fraction(1, 30)))

    def test_exact_sources_survive_json_and_rederive_for_new_host(self):
        cfg = CPTConfig(num_local_experts=4)
        source = cfg.to_dict()
        self.assertEqual(source['cpt_rho_beta'], '19/20')
        self.assertIsNone(source['cpt_state_step_size'])
        self.assertIsNone(source['cpt_prototype_temperature'])
        with tempfile.TemporaryDirectory() as directory:
            cfg.save_pretrained(directory)
            restored = CPTConfig.from_json_file(str(Path(directory) / 'config.json'))
        self.assertEqual(cfg.to_dict(), restored.to_dict())
        self.assertEqual(cfg.cpt_config_dict(), restored.cpt_config_dict())
        source.update(num_local_experts=6, cpt_num_prototypes=None, cpt_projection_dim=None)
        changed = CPTConfig.from_dict(json.loads(json.dumps(source)))
        self.assertEqual(changed.cpt_kappa_beta, rational_fp32('kappa', Fraction(5, 3)))
        self.assertEqual(changed.cpt_state_step_size, rational_fp32('eta', Fraction(6, 65)))


if __name__ == '__main__':
    unittest.main()
