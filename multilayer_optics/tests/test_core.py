"""Физические регрессии ядра; запуск: python -m unittest discover -s tests -v.

Эталоны Френеля, Эйри, критических матриц и интеграла локальных потерь
записаны здесь независимо от рабочего S-каскада. Все длины — мкм.
"""
import cmath
import math
from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from core import (SolverConfig, convert_time_convention, ellipsometry,
                  matrix_reference, solve_point)


def passive_q(index, alpha):
    value = cmath.sqrt(complex(index) ** 2 - alpha ** 2)
    return -value if value.imag > 0 else value


def admittance(index, q, pol):
    return q if pol == "s" else index ** 2 / q


def airy(k, index, d, substrate, theta, ambient=1.0, pol="s"):
    """Один слой: аналитическая сумма ряда многократных отражений."""
    alpha = ambient * math.sin(math.radians(theta))
    q = [passive_q(n, alpha) for n in (ambient, index, substrate)]
    y0, y1, ys = [admittance(n, qi, pol)
                  for n, qi in zip((ambient, index, substrate), q)]
    f01 = (y0-y1)/(y0+y1)
    f12 = (y1-ys)/(y1+ys)
    u = cmath.exp(-1j*k*q[1]*d)
    den = 1 + f01*f12*u*u
    return ((f01+f12*u*u)/den,
            (2*y0/(y0+y1))*(2*y1/(y1+ys))*u/den)


def independent_single_layer_loss(k, index, d, substrate, theta, pol):
    """Четыре граничных уравнения + интеграл объемных джоулевых потерь.

    Не использует r/t из solve_point. Неизвестные: r, a, b, t, где a/b —
    касательные E-амплитуды вперед/назад у начала слоя. Для p учитывает
    нормальную компоненту электрического поля Ez=alpha/q*(-E+ + E-).
    """
    alpha = math.sin(math.radians(theta))
    q0, q1, qs = [passive_q(n, alpha) for n in (1.0, index, substrate)]
    y0, y1, ys = [admittance(n, q, pol)
                  for n, q in zip((1.0, index, substrate), (q0, q1, qs))]
    u = cmath.exp(-1j*k*q1*d)
    matrix = np.array([
        [1, -1, -1, 0],
        [-y0, -y1, y1, 0],
        [0, u, 1/u, -1],
        [0, y1*u, -y1/u, -ys],
    ], dtype=complex)
    r, a, b, t = np.linalg.solve(matrix, [-1, -y0, 0, 0])
    nodes, weights = np.polynomial.legendre.leggauss(96)
    z = d*(nodes+1)/2
    forward = a*np.exp(-1j*k*q1*z)
    backward = b*np.exp(1j*k*q1*z)
    field_squared = np.abs(forward+backward)**2
    if pol == "p":
        field_squared += np.abs(alpha/q1*(-forward+backward))**2
    integral = d/2 * np.dot(weights, field_squared)
    absorptance = k*(-(index*index).imag)/y0.real * integral
    return float(absorptance), r, t


class CorePhysicsTests(unittest.TestCase):
    def assertComplexClose(self, actual, expected, tol=2e-12):
        self.assertLessEqual(abs(actual-expected), tol*(1+abs(expected)))

    def assertValid(self, result):
        self.assertTrue(result.s.valid, result.diagnostics)
        self.assertTrue(result.p.valid, result.diagnostics)

    def assertPointsEqual(self, left, right, tol=3e-12):
        self.assertValid(left)
        self.assertValid(right)
        for pol in ("s", "p"):
            a, b = getattr(left, pol), getattr(right, pol)
            self.assertComplexClose(a.r_tan, b.r_tan, tol)
            self.assertComplexClose(a.t_tan, b.t_tan, tol)
            for name in ("R", "T", "A"):
                self.assertAlmostEqual(getattr(a, name), getattr(b, name), delta=tol)

    def test_fresnel_no_layers(self):
        for n0, ns, angle in ((1, 1.5, 0), (1, 1.5, 23), (1, 1.5, 60),
                              (1, 1.5, 85), (1.6, 1.1, 30)):
            with self.subTest(ambient=n0, substrate=ns, angle=angle):
                result = solve_point(10, [], [], ns, angle, ambient=n0)
                self.assertValid(result)
                th = math.radians(angle)
                q0 = n0*math.cos(th)
                qs = math.sqrt(ns*ns-(n0*math.sin(th))**2)
                for pol in ("s", "p"):
                    y0 = q0 if pol == "s" else n0*n0/q0
                    ys = qs if pol == "s" else ns*ns/qs
                    r, t = (y0-ys)/(y0+ys), 2*y0/(y0+ys)
                    got = getattr(result, pol)
                    self.assertComplexClose(got.r_tan, r)
                    self.assertComplexClose(got.t_tan, t)
                    self.assertAlmostEqual(got.R, r*r, delta=2e-12)
                    self.assertAlmostEqual(got.T, ys/y0*t*t, delta=2e-12)
                    self.assertAlmostEqual(got.R+got.T, 1, delta=2e-12)

    def test_prompt_numerical_benchmark(self):
        result = solve_point(10, [], [], 1.5, 60)
        for actual, expected in (
            (result.s.R, 0.17657148808284046),
            (result.p.R, 0.0018019375215850254),
            (result.s.T, 0.8234285119171597),
            (result.p.T, 0.9981980624784150),
            (result.psi_deg, 5.768479516407728),
            (result.delta_deg, 0),
        ):
            self.assertAlmostEqual(actual, expected, delta=2e-12)
        self.assertComplexClose(result.rho, 0.10102051443364353)

    def test_airy_absorbing_layer_amplitudes(self):
        args = (2*math.pi/0.55, [2.1-0.17j], [0.137], 1.47, 51)
        result = solve_point(*args)
        for pol in ("s", "p"):
            r, t = airy(args[0], args[1][0], args[2][0], args[3], args[4], pol=pol)
            self.assertComplexClose(getattr(result, pol).r_tan, r)
            self.assertComplexClose(getattr(result, pol).t_tan, t)

    def test_quarter_wave_antireflection_and_undefined_ellipsometry(self):
        wavelength, ns = 0.63, 1.52
        layer = math.sqrt(ns)
        result = solve_point(2*math.pi/wavelength, [layer], [wavelength/(4*layer)], ns, 0)
        for got in (result.s, result.p):
            self.assertLess(got.R, 1e-28)
            self.assertAlmostEqual(got.T, 1, delta=2e-14)
        self.assertFalse(result.rho_valid)
        self.assertFalse(result.psi_valid)
        self.assertFalse(result.delta_valid)
        self.assertTrue(math.isnan(result.delta_deg))

    def test_brewster_angle_masks(self):
        result = solve_point(10, [], [], 1.5, math.degrees(math.atan(1.5)))
        self.assertLess(result.p.R, 1e-28)
        self.assertTrue(result.rho_valid)
        self.assertTrue(result.psi_valid)
        self.assertFalse(result.delta_valid)
        self.assertAlmostEqual(result.psi_deg, 0, delta=1e-12)

    def test_normal_incidence_basis_and_time_convention(self):
        result = solve_point(10, [2.0-0.1j, 1.35], [0.12, 0.07], 1.5, 0)
        self.assertComplexClose(result.r_p, -result.r_s)
        self.assertAlmostEqual(result.s.R, result.p.R, delta=2e-14)
        self.assertAlmostEqual(result.s.T, result.p.T, delta=2e-14)
        self.assertAlmostEqual(result.psi_deg, 45, delta=2e-12)
        self.assertAlmostEqual(result.delta_deg, 180, delta=2e-12)
        result = solve_point(10, [2.0-0.1j, 1.35], [0.12, 0.07], 1.5, 47)
        transformed = convert_time_convention(result, "exp(-iwt)")
        self.assertComplexClose(transformed.rho, result.rho.conjugate())
        for pol in ("s", "p"):
            a, b = getattr(result, pol), getattr(transformed, pol)
            self.assertComplexClose(a.r_tan.conjugate(), b.r_tan)
            self.assertComplexClose(a.t_tan.conjugate(), b.t_tan)
            self.assertEqual(a.R, b.R)
            self.assertEqual(a.T, b.T)
        self.assertAlmostEqual(transformed.delta_rad, (-result.delta_rad)%(2*math.pi), delta=1e-14)
        self.assertPointsEqual(result, convert_time_convention(transformed, "exp(+iwt)"))

    def test_total_internal_reflection_and_lossless_coating(self):
        for indices, thickness in (([], []), ([1.2, 1.8], [0.1, 0.23])):
            result = solve_point(10, indices, thickness, 1, 58, ambient=1.5)
            self.assertValid(result)
            for got in (result.s, result.p):
                self.assertAlmostEqual(got.R, 1, delta=3e-12)
                self.assertEqual(got.T, 0)
                self.assertEqual(got.log_T, -math.inf)
                self.assertEqual(got.transmission_status, "zero_normal_flux")

    def test_evanescent_substrate_with_absorbing_coating(self):
        result = solve_point(10, [1.6-0.18j], [0.15], 1, 60, ambient=1.5)
        self.assertValid(result)
        for got in (result.s, result.p):
            self.assertEqual(got.T, 0)
            self.assertGreater(got.A, 0.01)
            self.assertLess(got.R, 0.99)

    def test_lossless_evanescent_internal_layer(self):
        result = solve_point(12, [0.8, 1.7], [0.12, 0.09], 1.5, 67)
        self.assertValid(result)
        for got in (result.s, result.p):
            self.assertAlmostEqual(got.R+got.T, 1, delta=3e-12)
            self.assertGreater(got.T, 0)

    def test_critical_substrate_exact_and_neighboring_angles(self):
        n0, ns = 1.5, 1.0
        theta = math.degrees(math.asin(ns/n0))
        result = solve_point(11, [], [], ns, theta, ambient=n0)
        self.assertValid(result)
        self.assertComplexClose(result.s.r_tan, 1)
        self.assertComplexClose(result.s.t_tan, 2)
        self.assertComplexClose(result.p.r_tan, -1)
        self.assertEqual(result.p.t_tan, 0j)
        self.assertEqual(result.s.T, 0)
        self.assertEqual(result.p.T, 0)
        for offset in (-1e-5, 1e-5):
            nearby = solve_point(11, [], [], ns, theta+offset, ambient=n0)
            self.assertValid(nearby)
            for got in (nearby.s, nearby.p):
                self.assertAlmostEqual(got.R+got.T, 1, delta=3e-12)
            if offset < 0:
                self.assertGreater(nearby.s.T, 0)
                self.assertGreater(nearby.p.T, 0)
            else:
                self.assertEqual(nearby.s.T, 0)
                self.assertEqual(nearby.p.T, 0)

    def test_critical_finite_layer_against_explicit_limit(self):
        n0, nl, ns, k, d = 1.5, 1.0, 1.7, 11.0, 0.27
        theta = math.degrees(math.asin(nl/n0))
        result = solve_point(k, [nl], [d], ns, theta, ambient=n0)
        self.assertValid(result)
        q0 = n0*math.cos(math.radians(theta))
        qs = math.sqrt(ns*ns-nl*nl)
        for pol in ("s", "p"):
            y0 = q0 if pol == "s" else n0*n0/q0
            ys = qs if pol == "s" else ns*ns/qs
            upper = 1j*k*d if pol == "s" else 0j
            lower = 0j if pol == "s" else 1j*nl*nl*k*d
            B, C = 1+upper*ys, lower+ys
            den = y0*B+C
            got = getattr(result, pol)
            self.assertComplexClose(got.r_tan, (y0*B-C)/den)
            self.assertComplexClose(got.t_tan, 2*y0/den)
            self.assertAlmostEqual(got.R+got.T, 1, delta=3e-12)
        for angle in (theta-1e-6, theta+1e-6):
            nearby = solve_point(k, [nl], [d], ns, angle, ambient=n0)
            for pol in ("s", "p"):
                self.assertLess(abs(getattr(nearby, pol).r_tan-getattr(result, pol).r_tan), 2e-6)

    def test_near_grazing_identical_media(self):
        angle = 89.9999999
        result = solve_point(10, [1.0], [0.4], 1.0, angle)
        self.assertValid(result)
        expected_t = cmath.exp(-1j*10*0.4*math.cos(math.radians(angle)))
        for got in (result.s, result.p):
            self.assertLess(got.R, 1e-28)
            self.assertAlmostEqual(got.T, 1, delta=3e-14)
            self.assertComplexClose(got.t_tan, expected_t)

    def test_zero_thickness_and_splitting_invariance(self):
        original = solve_point(11, [2.0-0.1j, 1.3], [0.13, 0.09], 1.5, 57)
        # Нулевая ENZ-прослойка тоже должна удаляться до особого p-случая.
        with_zero = solve_point(11, [0j, 2.0-0.1j, 1.3], [0, 0.13, 0.09], 1.5, 57)
        split = solve_point(11, [2.0-0.1j, 2.0-0.1j, 1.3], [0.05, 0.08, 0.09], 1.5, 57)
        self.assertPointsEqual(original, with_zero)
        self.assertPointsEqual(original, split)

    def test_opaque_first_layer_and_log_transmission(self):
        index, theta, k = 2.2-0.3j, 47, 10
        result = solve_point(k, [index]+[1.4, 2.0]*100,
                             [1e5]+[0.1, 0.13]*100, 1.5, theta)
        self.assertValid(result)
        alpha = math.sin(math.radians(theta))
        q0, q1 = passive_q(1, alpha), passive_q(index, alpha)
        for pol in ("s", "p"):
            y0, y1 = admittance(1, q0, pol), admittance(index, q1, pol)
            got = getattr(result, pol)
            self.assertComplexClose(got.r_tan, (y0-y1)/(y0+y1))
            self.assertEqual(got.T, 0)
            self.assertTrue(math.isfinite(got.log_T))
            self.assertLess(got.log_T, -1000)
            self.assertEqual(got.transmission_status, "power_underflow")

    def test_thick_evanescent_layer_tunneling_underflow(self):
        result = solve_point(10, [0.5], [10000], 1.5, 60)
        self.assertValid(result)
        for got in (result.s, result.p):
            self.assertEqual(got.T, 0)
            self.assertTrue(math.isfinite(got.log_T))
            self.assertAlmostEqual(got.R, 1, delta=2e-12)

    def test_large_layer_count(self):
        count = 1500
        result = solve_point(10, [1.5-0.1j, 2.2-0.15j]*(count//2),
                             [0.1, 0.15]*(count//2), 1.5, 53)
        self.assertValid(result)
        for got in (result.s, result.p):
            self.assertTrue(math.isfinite(got.log_T))
            self.assertGreaterEqual(got.A, -2e-12)
            self.assertLessEqual(got.R, 1+2e-12)

    def test_enz_normal_and_unsupported_oblique(self):
        normal = solve_point(10, [0j], [0.1], 1.5, 0)
        self.assertValid(normal)
        self.assertComplexClose(normal.s.r_tan, normal.p.r_tan)
        self.assertComplexClose(normal.s.t_tan, normal.p.t_tan)
        for got in (normal.s, normal.p):
            self.assertAlmostEqual(got.R+got.T, 1, delta=2e-12)
        for angle in (35, 1e-7):
            oblique = solve_point(10, [0j], [0.1], 1.5, angle)
            self.assertTrue(oblique.s.valid)
            self.assertFalse(oblique.p.valid)
            self.assertTrue(any(d.code == "UNSUPPORTED_ENZ" for d in oblique.diagnostics))

    def test_random_matrix_comparison_complex_amplitudes(self):
        rng = np.random.default_rng(20260925)
        for number in range(60):
            count = int(rng.integers(0, 9))
            index = rng.uniform(0.7, 2.8, count)-1j*rng.uniform(0, 0.08, count)
            d = rng.uniform(0.001, 0.18, count)
            k, ns, theta = rng.uniform(5, 13), rng.uniform(1.1, 1.9), rng.uniform(0, 76)
            with self.subTest(case=number, layers=count):
                result = solve_point(k, index, d, ns, theta)
                reference = matrix_reference(k, index, d, ns, theta)
                self.assertValid(result)
                for pol in ("s", "p"):
                    got = getattr(result, pol)
                    r, t = reference[pol]
                    self.assertComplexClose(got.r_tan, r, tol=2e-11)
                    self.assertComplexClose(got.t_tan, t, tol=2e-11)
                    self.assertGreaterEqual(got.A, -2e-11)

    def test_independent_volume_absorption_integral_both_polarizations(self):
        k, index, d, ns, theta = 2*math.pi/0.63, 2.1-0.18j, 0.19, 1.5, 53
        result = solve_point(k, [index], [d], ns, theta)
        for pol in ("s", "p"):
            loss, r, t = independent_single_layer_loss(k, index, d, ns, theta, pol)
            got = getattr(result, pol)
            self.assertGreater(loss, 0.1)
            self.assertAlmostEqual(got.A, loss, delta=3e-12)
            self.assertComplexClose(got.r_tan, r)
            self.assertComplexClose(got.t_tan, t)

    def test_ellipsometry_zero_component_masks(self):
        rho, psi, delta, rv, pv, dv = ellipsometry(0j, 1j)
        self.assertFalse(rv)
        self.assertTrue(pv)
        self.assertFalse(dv)
        self.assertAlmostEqual(psi, math.pi/2)
        self.assertTrue(math.isnan(delta))
        rho, psi, delta, rv, pv, dv = ellipsometry(1j, 0j)
        self.assertTrue(rv)
        self.assertTrue(pv)
        self.assertFalse(dv)
        self.assertEqual(rho, 0j)
        self.assertEqual(psi, 0)
        rho, psi, delta, rv, pv, dv = ellipsometry(0j, 0j)
        self.assertFalse(rv or pv or dv)
        self.assertTrue(math.isnan(psi))
        self.assertTrue(math.isnan(delta))

    def test_angle_units(self):
        args = (10, [1.8-0.02j], [0.11], 1.5)
        self.assertPointsEqual(solve_point(*args, 42),
                               solve_point(*args, math.radians(42), angle_unit="rad"))

    def test_input_validation(self):
        invalid = [
            dict(k=0), dict(k=math.nan), dict(indices=[1+0.1j], thickness_um=[0.1]),
            dict(indices=[1.5], thickness_um=[-0.1]),
            dict(indices=[1.5], thickness_um=[]),
            dict(substrate=1.5-1e-20j), dict(substrate=0),
            dict(ambient=1-1e-20j), dict(theta=90), dict(theta=-1),
        ]
        defaults = dict(k=10, indices=[], thickness_um=[], substrate=1.5, theta=30)
        for change in invalid:
            with self.subTest(change=change):
                with self.assertRaises(ValueError):
                    solve_point(**(defaults | change))
        with self.assertRaises(ValueError):
            SolverConfig(energy_tol=-1)


if __name__ == "__main__":
    unittest.main()
