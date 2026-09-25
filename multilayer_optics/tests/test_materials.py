"""Проверки единиц, дисперсии, пассивности и запрещённой экстраполяции."""

import unittest

import numpy as np

try:
    from multilayer_optics.materials import (
        TabulatedMaterial, evaluate_index, from_k, length_to_um,
        make_grid, material_values, to_k, validate_spectral_grid,
    )
except ModuleNotFoundError:
    from materials import (
        TabulatedMaterial, evaluate_index, from_k, length_to_um,
        make_grid, material_values, to_k, validate_spectral_grid,
    )


class SpectralUnitsTests(unittest.TestCase):
    def test_wavenumber_semantics_are_explicit(self):
        self.assertEqual(float(to_k(10000, "angular_wavenumber", "cm^-1")), 1.0)
        self.assertEqual(float(to_k(10000, "spectroscopic_wavenumber", "cm^-1")), 2*np.pi)
        self.assertEqual(float(to_k(1000, "wavelength", "nm")), 2*np.pi)
        self.assertEqual(float(to_k(1, "angular_wavenumber", "rad/um")), 1.0)

    def test_all_units_round_trip_preserve_shape(self):
        k = np.array([[0.01, 2], [15, 400.]])
        for kind in ("angular_wavenumber", "spectroscopic_wavenumber", "wavelength"):
            for length in ("nm", "um", "mm", "cm", "m"):
                unit = length if kind == "wavelength" else f"{length}^-1"
                with self.subTest(kind=kind, unit=unit):
                    restored = to_k(from_k(k, kind, unit), kind, unit)
                    np.testing.assert_allclose(restored, k, rtol=4e-16, atol=0)
                    self.assertEqual(restored.shape, k.shape)

    def test_nm_cm_phase_is_dimensionless(self):
        k = to_k(12000, "spectroscopic_wavenumber", "cm^-1")
        d = length_to_um(250, "nm")
        self.assertAlmostEqual(float(k*d), 2*np.pi*12000*250*1e-7)

    def test_invalid_coordinates_and_units(self):
        for values in ([0, 1], [-1, 2], [np.nan, 1], [np.inf], [], [1+0j], [True]):
            with self.subTest(values=values), self.assertRaises(ValueError):
                to_k(values)
        for kind, unit in (("unknown", "um^-1"), ("wavelength", "cm^-1"),
                           ("angular_wavenumber", "cm"), ("spectroscopic_wavenumber", "rad/um")):
            with self.subTest(kind=kind, unit=unit), self.assertRaises(ValueError):
                to_k(1, kind, unit)

    def test_grid_checks_original_coordinate_and_endpoints(self):
        x = make_grid(400, 900, 51)
        self.assertEqual(x[0], 400)
        self.assertEqual(x[-1], 900)
        np.testing.assert_array_equal(validate_spectral_grid(x), x)
        self.assertTrue(np.all(np.diff(to_k(x, "wavelength", "nm")) < 0))
        for invalid in ([1], [1, 1], [2, 1], [[1, 2]], [1, np.nan]):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                validate_spectral_grid(invalid)
        for count in (1, True, 3.0):
            with self.subTest(count=count), self.assertRaises(ValueError):
                make_grid(1, 2, count)
        with self.assertRaises(ValueError):
            make_grid(1, np.nextafter(1., 2.), 3)

    def test_lengths_validate(self):
        np.testing.assert_array_equal(length_to_um([0, 1000], "nm"), [0, 1])
        for values in ([-1], [np.nan], [1j], [True]):
            with self.subTest(values=values), self.assertRaises(ValueError):
                length_to_um(values)


class MaterialTests(unittest.TestCase):
    def test_constant_callable_and_explicit_time_conversion(self):
        k = np.array([2., 3., 4.])
        np.testing.assert_array_equal(material_values(1.5-0.1j, k), [1.5-0.1j]*3)
        np.testing.assert_array_equal(material_values(1.5+0.1j, k, "n_plus_ik"), [1.5-0.1j]*3)
        np.testing.assert_array_equal(material_values(lambda x: x - 0.2j, k), k - 0.2j)
        np.testing.assert_array_equal(material_values(lambda x: 1.2, k), [1.2]*3)
        self.assertEqual(material_values(1.5, 2.).shape, ())
        for material in (1.5+0.1j, -1.5-0.1j, np.nan, np.inf, [1, 2, 3]):
            with self.subTest(material=material), self.assertRaises(ValueError):
                material_values(material, k)
        with self.assertRaises(ValueError):
            material_values(lambda x: np.ones((3, 1)), k)

    def test_interpolation_is_in_original_wavelength_coordinate(self):
        table = TabulatedMaterial([400, 800], [1-0.1j, 3-0.3j], "wavelength", "nm")
        query = to_k([400, 600, 800], "wavelength", "nm")
        expected = np.array([1-0.1j, 2-0.2j, 3-0.3j])
        np.testing.assert_allclose(material_values(table, query), expected, atol=5e-16, rtol=5e-16)
        self.assertEqual(table(query.reshape(1, 3)).shape, (1, 3))
        self.assertEqual(table(query[0]).shape, ())

    def test_source_convention_and_mutable_input_is_copied(self):
        x = np.array([1., 2.])
        values = np.array([1+0.2j, 2+0.4j])
        table = TabulatedMaterial(x, values, convention="n_plus_ik")
        x[:] = 50
        values[:] = 50
        np.testing.assert_array_equal(table([1., 2.]), [1-0.2j, 2-0.4j])
        with self.assertRaises(ValueError):
            table.x[0] = 7
        with self.assertRaises(ValueError):
            material_values(table, [1.], convention="n_plus_ik")

    def test_extrapolation_including_one_ulp_is_rejected(self):
        table = TabulatedMaterial([1, 2], [1.5, 1.6])
        for k in (np.nextafter(1., 0.), np.nextafter(2., 3.), 0.5, 3.):
            with self.subTest(k=k), self.assertRaises(ValueError):
                table(k)
        wavelength_table = TabulatedMaterial([400, 800], [1.5, 1.6], "wavelength", "nm")
        for wavelength in (399, 801):
            with self.subTest(wavelength=wavelength), self.assertRaises(ValueError):
                wavelength_table(to_k(wavelength, "wavelength", "nm"))

    def test_table_validation(self):
        cases = (([1, 1], [1, 2]), ([2, 1], [1, 2]), ([1, 2], [1]),
                 ([1, 2], [1, np.nan]), ([1, 2], [1, 1+0.1j]), ([1, 2], [-1, 2]))
        for x, values in cases:
            with self.subTest(x=x, values=values), self.assertRaises(ValueError):
                TabulatedMaterial(x, values)

    def test_transparent_environment_is_strict(self):
        np.testing.assert_array_equal(evaluate_index(1.5, [1., 2.], transparent=True), [1.5, 1.5])
        for index in (1.5-1e-300j, 0.):
            with self.subTest(index=index), self.assertRaises(ValueError):
                evaluate_index(index, [1., 2.], transparent=True)
        # Проверяются и узлы таблицы, не попавшие в данную расчётную выборку.
        table = TabulatedMaterial([1, 2, 3], [1.5, 1.5-0.1j, 1.5])
        with self.assertRaises(ValueError):
            evaluate_index(table, [1., 3.], transparent=True)


if __name__ == "__main__":
    unittest.main()
