"""Спектральный API, уточнение сетки и строгий JSON-экспорт.

Ядро не зависит от файловых форматов или графического интерфейса.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
import json
import math
from pathlib import Path
from typing import Any, Iterator, Sequence

import numpy as np

from .core import PointResult, SolverConfig, solve_point, convert_time_convention
from .materials import evaluate_index, length_to_um, to_k, validate_spectral_grid


@dataclass(frozen=True)
class Layer:
    """Однородный слой: material — константа, callable(k) или таблица.

    thickness — скалярная геометрическая толщина; unit — nm/um/mm/cm/m.
    convention описывает входной показатель константы/callable. Таблица
    хранит собственное соглашение и используется с n_minus_ik здесь.
    """
    material: Any
    thickness: float
    unit: str = "nm"
    convention: str = "n_minus_ik"
    name: str = ""

    def thickness_um(self) -> float:
        """Проверить и перевести скалярную толщину в мкм, O(1)."""
        value = length_to_um(self.thickness, self.unit)
        if value.ndim != 0:
            raise ValueError("Толщина одного слоя должна быть скаляром")
        return float(value)


def unwrap_valid(delta: Sequence[float], valid: Sequence[bool]) -> np.ndarray:
    """Развернуть фазу отдельно в каждом непрерывном участке маски, O(M).

    Нет интерполяции через NaN/нулевое отражение. Результат зависит от
    порядка узлов; недостаточно разрешённые фазовые скачки не исправляются.
    """
    a = np.asarray(delta, dtype=float)
    mask = np.asarray(valid, dtype=bool)
    if a.ndim != 1 or mask.shape != a.shape:
        raise ValueError("delta и valid должны быть одномерными массивами одинаковой длины")
    mask = mask & np.isfinite(a)
    out = np.full(a.shape, np.nan)
    edges = np.flatnonzero(np.diff(np.r_[False, mask, False]))
    for start, end in zip(edges[::2], edges[1::2]):
        out[start:end] = np.unwrap(a[start:end])
    return out


@dataclass(frozen=True)
class SpectrumResult:
    """Спектр в исходном порядке координаты, M точек и метаданные.

    points сохраняют индивидуальную диагностику; arrays() формирует
    одномерные массивы. Хранение O(M); временные материал-массивы блочные.
    """
    coordinate: np.ndarray
    points: tuple[PointResult, ...]
    metadata: dict

    def arrays(self, time_convention: str = "exp(+iwt)") -> dict[str, np.ndarray]:
        """Вернуть именованные массивы формы (M,), включая маски, O(M)."""
        pp = tuple(convert_time_convention(p, time_convention) for p in self.points)
        result = {
            "coordinate": self.coordinate.copy(),
            "k_rad_per_um": np.array([p.k for p in pp]),
            "lambda_um": np.array([p.wavelength_um for p in pp]),
        }
        for pol in ('s', 'p'):
            values = [getattr(p, pol) for p in pp]
            for name in ('R', 'T', 'A'):
                result[name+pol] = np.array([getattr(v, name) for v in values])
            for name in ('r_tan', 't_tan', 'log_abs_t', 'phase_t', 'log_T', 'valid', 'transmission_status'):
                result[name+'_'+pol] = np.array([getattr(v, name) for v in values])
        for name in ('r_s', 'r_p', 'rho', 'psi_rad', 'delta_rad', 'psi_deg', 'delta_deg',
                     'rho_valid', 'psi_valid', 'delta_valid'):
            result[name] = np.array([getattr(p, name) for p in pp])
        result['delta_unwrapped_rad'] = unwrap_valid(result['delta_rad'], result['delta_valid'])
        return result

    def to_dict(self, time_convention: str = "exp(+iwt)") -> dict:
        """JSON-совместимый объект: complex->{re,im}, NaN/Inf->null.

        Маски и transmission_status различают сингулярность rho, физический
        T=0 и потерю представимости. Без явных масок null неоднозначен.
        """
        def safe(x):
            if isinstance(x, np.ndarray):
                return [safe(v) for v in x.tolist()]
            if isinstance(x, complex):
                return {"re": safe(x.real), "im": safe(x.imag)} if math.isfinite(x.real) and math.isfinite(x.imag) else None
            if isinstance(x, (float, np.floating)):
                return float(x) if math.isfinite(x) else None
            if isinstance(x, dict):
                return {key: safe(v) for key, v in x.items()}
            if isinstance(x, (list, tuple)):
                return [safe(v) for v in x]
            return x
        meta = dict(self.metadata, time_convention=time_convention,
                    index_convention="n_minus_ik" if time_convention == "exp(+iwt)" else "n_plus_ik")
        return safe({"metadata": meta, "data": self.arrays(time_convention),
                     "diagnostics": [[asdict(d) for d in p.diagnostics] for p in self.points]})

    def write_json(self, path: str | Path, time_convention: str = "exp(+iwt)") -> None:
        """Записать UTF-8 JSON без NaN/Infinity; каталог должен существовать."""
        Path(path).write_text(json.dumps(self.to_dict(time_convention), ensure_ascii=False,
                                        indent=2, allow_nan=False), encoding='utf-8')


def stream_spectrum(coordinate, layers: Sequence[Layer], substrate, theta: float,
                    ambient=1.0, kind="angular_wavenumber", unit="um^-1",
                    angle_unit="deg", config: SolverConfig = SolverConfig(),
                    chunk_size: int = 256) -> Iterator[PointResult]:
    """Рассчитать спектр блоками; исходная координата строго возрастает.

    Материалы получают k [rad/um] и обязаны возвращать константу или массив
    формы запроса. Внешняя среда и подложка строго действительны >0.
    Для произвольной callable прозрачность проверяется только на рассчитанных
    узлах: свойства между узлами гарантирует поставщик материала.
    Время O(N*M) плюс стоимость материалов; временная память O(N*chunk_size).
    Для остановки после отдельной точки можно прервать итератор.
    """
    x = validate_spectral_grid(coordinate)
    if not isinstance(chunk_size, int) or isinstance(chunk_size, bool) or chunk_size < 1:
        raise ValueError("chunk_size должен быть положительным целым")
    layers = tuple(layers)
    if not all(isinstance(layer, Layer) for layer in layers):
        raise ValueError("layers должен содержать объекты Layer")
    d = np.array([layer.thickness_um() for layer in layers])
    k = to_k(x, kind, unit)
    for start in range(0, len(k), chunk_size):
        kk = k[start:start+chunk_size]
        sub = evaluate_index(substrate, kk, transparent=True)
        n0 = evaluate_index(ambient, kk, transparent=True)
        indices = [evaluate_index(layer.material, kk, convention=layer.convention) for layer in layers]
        for i, ki in enumerate(kk):
            yield solve_point(float(ki), [n[i] for n in indices], d, sub[i], theta,
                              ambient=n0[i], angle_unit=angle_unit, config=config)


def calculate_spectrum(coordinate, layers: Sequence[Layer], substrate, theta: float,
                       ambient=1.0, kind="angular_wavenumber", unit="um^-1",
                       angle_unit="deg", config: SolverConfig = SolverConfig(),
                       chunk_size: int = 256) -> SpectrumResult:
    """Собрать спектр и соглашения; параметры совпадают со stream_spectrum.

    Выходы energies — доли, не проценты; r/t комплексные. O(N*M) времени,
    O(M+N*chunk_size) памяти без учёта таблиц исходных материалов.
    """
    x = validate_spectral_grid(coordinate)
    layers = tuple(layers)
    points = tuple(stream_spectrum(x, layers, substrate, theta, ambient, kind, unit,
                                   angle_unit, config, chunk_size))
    x.setflags(write=False)
    metadata = {
        "schema_version": "1.0", "coordinate_kind": kind, "coordinate_unit": unit,
        "internal_length_unit": "um", "internal_k_unit": "rad/um",
        "theta_rad": points[0].theta_rad, "time_convention": "exp(+iwt)",
        "index_convention": "n_minus_ik", "layer_order": "ambient_to_substrate",
        "layer_count": len(layers), "thickness_um": [l.thickness_um() for l in layers],
        "layer_names": [l.name for l in layers],
        "substrate": "semi_infinite_lossless", "ambient": "lossless",
        "r_p_definition": "minus tangential electric reflection",
        "rho_definition": "r_p/r_s = tan(Psi)*exp(i*Delta)",
        "reference_planes": "r at entrance; t from entrance to substrate boundary",
        "delta_range": "[0,2*pi)", "precision": "float64/complex128",
        "method": "scaled common-port scattering recursion",
        "tolerances": asdict(config),
        "material_validation": "tables: all source knots; callables: sampled spectral nodes",
    }
    return SpectrumResult(x, points, metadata)


@dataclass(frozen=True)
class RefinementResult:
    """Результат проверки разрешения; converged относится только к этому критерию."""
    spectrum: SpectrumResult
    converged: bool
    rounds: int
    normalized_error: float
    reason: str


def refine_spectrum(coordinate, layers: Sequence[Layer], substrate, theta: float, *,
                    max_rounds=4, max_points=20001, atol=1e-5, rtol=1e-4,
                    phase_atol_rad=0.02, **kwargs) -> RefinementResult:
    """Уточнить сетку вставкой всех середин, проверяя энергии, амплитуды и фазу.

    На каждом раунде сравниваются фактические значения в серединах с
    линейным предсказанием по концам (Delta сравнивается на единичном круге).
    Максимальная нормированная ошибка <=1 означает локальную сходимость.
    Это не гарантия обнаружения всех узких резонансов. Время O(N*M_final)
    для геометрического роста сетки, память O(M_final+N*chunk_size).
    max_points включает все рассчитанные узлы последнего спектра.
    """
    if not isinstance(max_rounds, int) or isinstance(max_rounds, bool) or max_rounds < 1:
        raise ValueError("max_rounds должен быть целым >=1")
    if not isinstance(max_points, int) or isinstance(max_points, bool) or max_points < 2:
        raise ValueError("max_points должен быть целым >=2")
    if not all(math.isfinite(v) and v > 0 for v in (atol, rtol, phase_atol_rad)):
        raise ValueError("Допуски должны быть конечными положительными")
    x = validate_spectral_grid(coordinate)
    if len(x) > max_points:
        raise ValueError("Исходная сетка превышает max_points")
    layers = tuple(layers)
    current = calculate_spectrum(x, layers, substrate, theta, **kwargs)
    err, rounds = math.inf, 0
    for step in range(1, max_rounds+1):
        if 2*len(x)-1 > max_points:
            return RefinementResult(current, False, rounds, err, "max_points")
        midpoint = x[:-1] + (x[1:]-x[:-1])/2
        if np.any(midpoint == x[:-1]) or np.any(midpoint == x[1:]):
            return RefinementResult(current, False, rounds, err, "float64_grid_resolution")
        dense = np.empty(2*len(x)-1)
        dense[::2], dense[1::2] = x, midpoint
        current = calculate_spectrum(dense, layers, substrate, theta, **kwargs)
        arrays = current.arrays()
        err = 0.0
        for key in ('Rs', 'Rp', 'Ts', 'Tp', 'As', 'Ap', 'r_s', 'r_p'):
            v = arrays[key]
            prediction = (v[:-2:2]+v[2::2])/2
            real = v[1::2]
            error = np.abs(real-prediction)/(atol+rtol*np.maximum(np.abs(real), np.abs(prediction)))
            if not np.all(np.isfinite(error)):
                err = math.inf
            else:
                err = max(err, float(np.max(error)))
        valid = arrays['delta_valid']
        mask = valid[:-2:2] & valid[1::2] & valid[2::2]
        if np.any(mask):
            phase = np.exp(1j*arrays['delta_rad'])
            prediction = phase[:-2:2]+phase[2::2]
            ambiguous = mask & (np.abs(prediction) < 1e-12)
            if np.any(ambiguous):
                err = math.inf
            good = mask & ~ambiguous
            if np.any(good):
                pe = np.abs(np.angle(phase[1::2][good]/prediction[good]))
                err = max(err, float(np.max(pe))/phase_atol_rad)
        rounds = step
        if err <= 1:
            return RefinementResult(current, True, rounds, err, "midpoint_criterion")
        x = dense
    return RefinementResult(current, False, rounds, err, "max_rounds")
