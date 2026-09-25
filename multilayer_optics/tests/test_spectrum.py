"""Проверки спектрального API, единиц, масок, экспорта и разрешения сетки."""
import json
import math
from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from multilayer_optics import (Layer, calculate_spectrum, stream_spectrum, refine_spectrum,
                              make_grid, to_k, unwrap_valid, SolverConfig)


class SpectrumTests(unittest.TestCase):
    def test_units_equivalence(self):
        nu = make_grid(12500, 25000, 17)
        stack = [Layer(1.8-.03j, 120)]
        a = calculate_spectrum(nu, stack, 1.5, 53, kind='spectroscopic_wavenumber', unit='cm^-1')
        b = calculate_spectrum(to_k(nu, 'spectroscopic_wavenumber', 'cm^-1'), stack, 1.5, 53)
        for name in ('Rs', 'Rp', 'Ts', 'Tp', 'r_s', 'r_p', 'rho', 'delta_rad'):
            np.testing.assert_allclose(a.arrays()[name], b.arrays()[name], atol=2e-14)

    def test_stream_chunk_and_dispersion(self):
        grid = make_grid(6, 12, 19)
        material = lambda k: 1.7+.01*k-.02j
        sub = lambda k: 1.4+.002*k
        stack = [Layer(material, 150), Layer(1.3, .07, 'um')]
        a = calculate_spectrum(grid, stack, sub, 60, chunk_size=3)
        b = tuple(stream_spectrum(grid, stack, sub, 60, chunk_size=11))
        np.testing.assert_allclose(a.arrays()['Rs'], [p.s.R for p in b], atol=1e-14)
        self.assertGreater(np.ptp(a.arrays()['Rs']), .001)
        np.testing.assert_allclose(a.metadata['thickness_um'], [.15, .07])

    def test_json_undefined_and_physical_zero(self):
        matched = calculate_spectrum([5, 6], [], 1, 0)
        data = json.loads(json.dumps(matched.to_dict(), allow_nan=False))['data']
        self.assertEqual(data['rho'], [None, None])
        self.assertEqual(data['delta_rad'], [None, None])
        self.assertEqual(data['delta_valid'], [False, False])
        tir = calculate_spectrum([5, 6], [], 1, 60, ambient=1.5).to_dict()['data']
        self.assertEqual(tir['Ts'], [0, 0])
        self.assertEqual(tir['log_T_s'], [None, None])
        self.assertEqual(tir['transmission_status_s'], ['zero_normal_flux']*2)

    def test_opaque_export_preserves_log(self):
        r = calculate_spectrum([5, 6], [Layer(1.5-.5j, 1000, 'um')], 1.5, 60)
        data = r.to_dict()['data']
        self.assertEqual(data['Ts'], [0, 0])
        self.assertTrue(all(math.isfinite(v) and v < -1000 for v in data['log_T_s']))
        self.assertEqual(data['transmission_status_s'], ['power_underflow']*2)

    def test_time_convention_export(self):
        r = calculate_spectrum([5, 6], [Layer(1.7-.1j, 140)], 1.5, 60)
        a, b = r.arrays(), r.arrays('exp(-iwt)')
        np.testing.assert_allclose(a['rho'].conjugate(), b['rho'])
        np.testing.assert_allclose(a['Rs'], b['Rs'])
        np.testing.assert_allclose((-a['delta_rad'])%(2*np.pi), b['delta_rad'])

    def test_unwrap_segments(self):
        delta = np.radians([350, 5, np.nan, 350, 4])
        out = unwrap_valid(delta, [True, True, False, True, True])
        np.testing.assert_allclose(np.degrees(out), [350, 365, np.nan, 350, 364], equal_nan=True)

    def test_invalid_inputs(self):
        for grid in ([1, 1], [2, 1], [1, np.nan], [1]):
            with self.assertRaises(ValueError):
                calculate_spectrum(grid, [], 1.5, 0)
        with self.assertRaises(ValueError):
            calculate_spectrum([1, 2], [], 1.5-.001j, 0)
        with self.assertRaises(ValueError):
            calculate_spectrum([1, 2], [Layer(1.4, -1)], 0.9, 0)
        with self.assertRaises(ValueError):
            calculate_spectrum([1, 2], [], 1.5, 90)

    def test_refinement_constant_and_budget(self):
        a = refine_spectrum([5, 6], [], 1.5, 60)
        self.assertTrue(a.converged)
        self.assertEqual(a.rounds, 1)
        b = refine_spectrum([5, 6], [Layer(1.8, 700)], 1.5, 60, max_points=2)
        self.assertFalse(b.converged)
        self.assertEqual(b.reason, 'max_points')

    def test_refinement_nontrivial(self):
        a = refine_spectrum(make_grid(5, 12, 21), [Layer(1.8-.02j, 150)], 1.5, 60,
                            max_rounds=7, max_points=5121)
        self.assertTrue(a.converged)
        self.assertLessEqual(a.normalized_error, 1)
        self.assertGreater(len(a.spectrum.points), 21)


if __name__ == '__main__':
    unittest.main()
