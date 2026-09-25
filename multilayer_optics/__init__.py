"""Когерентный многослойник на прозрачной полубесконечной подложке."""
from .core import (SolverConfig, Diagnostic, PolarizationResult, PointResult,
                   forward_q, ellipsometry, solve_point, matrix_reference,
                   convert_time_convention)
from .materials import (TabulatedMaterial, evaluate_index, material_values,
                        to_k, from_k, make_grid, validate_spectral_grid, length_to_um)
from .spectrum import (Layer, SpectrumResult, RefinementResult, calculate_spectrum,
                       stream_spectrum, refine_spectrum, unwrap_valid)

__version__ = "0.1.0"

__all__ = [
    "SolverConfig", "Diagnostic", "PolarizationResult", "PointResult", "forward_q",
    "ellipsometry", "solve_point", "matrix_reference", "convert_time_convention",
    "TabulatedMaterial", "evaluate_index", "material_values", "to_k", "from_k",
    "make_grid", "validate_spectral_grid", "length_to_um", "Layer", "SpectrumResult",
    "RefinementResult", "calculate_spectrum", "stream_spectrum", "refine_spectrum", "unwrap_valid",
]
