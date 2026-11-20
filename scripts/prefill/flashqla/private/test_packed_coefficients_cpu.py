"""Admission boundaries must preserve unqualified shapes and state ABIs."""
import importlib.util
from pathlib import Path
import unittest

path = Path(__file__).resolve().parents[4] / 'pla/prefill/flashqla/private/packed_coefficients.py'
spec = importlib.util.spec_from_file_location('_flashqla_packed_policy', path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class PackedCoefficientPolicyTests(unittest.TestCase):
    def mode(self, tokens, heads=8, key_heads=2, **overrides):
        values = dict(batch_size=1, indexed_state=True, full_sequence=True)
        values.update(overrides)
        return module.tc01_packed_coefficients_mode(8, 0, tokens, heads, key_heads, **values)

    def test_short_fused_boundary(self):
        for key_heads in (2, 4):
            for tokens in (64, 65, 508, 512):
                with self.subTest(tokens=tokens, key_heads=key_heads):
                    self.assertEqual(self.mode(tokens, key_heads=key_heads), 1)
            self.assertEqual(self.mode(513, key_heads=key_heads), 0)
            for tokens in (1, 31, 32, 33, 63):
                self.assertEqual(self.mode(tokens, key_heads=key_heads), 0)

    def test_measured_h16_long_region(self):
        for tokens in (1291, 1292, 1312, 1536, 8191, 8192, 8193, 31801, 32768):
            self.assertEqual(self.mode(tokens, 16, 2), 2)
        for tokens in (1, 507, 508, 509, 1260, 1279, 1280, 1281, 1290, 32769):
            self.assertEqual(self.mode(tokens, 16, 2), 0)

    def test_state_metadata_and_cp_guards(self):
        for override in ({'indexed_state': False}, {'batch_size': 2},
                         {'full_sequence': False}, {'force_cp': True},
                         {'group_override': True}, {'available': False}):
            with self.subTest(override=override):
                self.assertEqual(self.mode(508, **override), 0)
                self.assertEqual(self.mode(8192, 16, 2, **override), 0)

    def test_other_architectures_and_topologies_stay_ordinary(self):
        for heads, key_heads in ((8, 1), (16, 1), (16, 4), (24, 8), (32, 4), (128, 16)):
            for tokens in (508, 1291, 32768):
                self.assertEqual(self.mode(tokens, heads, key_heads), 0)
        for major, minor in ((8, 9), (9, 0), (10, 0)):
            for tokens, heads in ((508, 8), (1291, 16), (32768, 16)):
                self.assertEqual(module.tc01_packed_coefficients_mode(
                    major, minor, tokens, heads, 2, batch_size=1,
                    indexed_state=True, full_sequence=True), 0)


if __name__ == '__main__':
    unittest.main()
