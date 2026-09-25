"""Регрессии численной диагностики: не выдавать переполнение за физический ноль."""
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from multilayer_optics import solve_point


class NumericalLimitTests(unittest.TestCase):
    def test_overflow_returns_diagnostics(self):
        cases = [
            (10, [1e200], [.1], 1.5, 30),
            (1, [1], [1e308], 1.5, 0),
            (1, [1-1j]*3, [8e307]*3, 1.5, 0),
        ]
        for args in cases:
            with self.subTest(args=args):
                result = solve_point(*args)
                self.assertFalse(result.s.valid)
                self.assertFalse(result.p.valid)
                self.assertFalse(result.delta_valid)
                self.assertIn('NUMERICAL_FAILURE', [d.code for d in result.diagnostics])


if __name__ == '__main__':
    unittest.main()
