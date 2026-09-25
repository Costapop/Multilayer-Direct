"""Когерентное ядро многослойника: exp(+iωt), n-iκ, длины в мкм.

Расчёт одной точки имеет сложность O(N), память O(N) на входные массивы.
Основной метод объединяет S-блоки в общем фиктивном портовом адмиттансе.
Это эквивалент рекурсии Фурмана–Тихонравова, регулярный при q слоя = 0.
Материалы и спектральная интерполяция находятся в materials.py.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace
import cmath
import math
from typing import Sequence

import numpy as np


@dataclass(frozen=True)
class SolverConfig:
    """Допуски float64: амплитуд, энергетики, знаменателей и фаз.

    critical_roundoff_factor задаёт множитель машинного epsilon при
    компенсации двух противоположных слагаемых q² прозрачного материала.
    Ноль отключает эту обработку округления; это не добавление потерь.
    """
    ellipsometry_atol: float = 1e-12
    energy_tol: float = 2e-10
    denominator_rtol: float = 1e-12
    critical_roundoff_factor: float = 8.0
    phase_error_tol: float = 1e-7

    def __post_init__(self):
        for name, value in vars(self).items():
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} должен быть конечным неотрицательным числом")


@dataclass(frozen=True)
class Diagnostic:
    """Машиночитаемое сообщение, поляризация и исходный номер слоя (1..N)."""
    code: str
    message: str
    polarization: str | None = None
    layer: int | None = None


@dataclass(frozen=True)
class PolarizationResult:
    """Касательные E-амплитуды и энергетические доли одной поляризации.

    log_abs_t и log_T — натуральные логарифмы. -inf для физического нуля;
    при численном обнулении положительного T log_T остаётся конечным.
    valid=False означает отсутствие достоверного результата, а не T=0.
    """
    r_tan: complex
    t_tan: complex
    R: float
    T: float
    A: float
    log_abs_t: float
    phase_t: float
    log_T: float
    valid: bool = True
    transmission_status: str = "propagating"


@dataclass(frozen=True)
class PointResult:
    """Результат одной точки; все углы в радианах, k в rad/мкм.

    rho, Psi и Delta имеют отдельные маски: при исчезновении отражённых
    компонент соответствующие значения представлены NaN до JSON-экспорта.
    """
    k: float
    wavelength_um: float
    theta_rad: float
    s: PolarizationResult
    p: PolarizationResult
    r_s: complex
    r_p: complex
    rho: complex
    psi_rad: float
    delta_rad: float
    rho_valid: bool
    psi_valid: bool
    delta_valid: bool
    diagnostics: tuple[Diagnostic, ...] = field(default_factory=tuple)
    time_convention: str = "exp(+iwt)"

    @property
    def psi_deg(self) -> float:
        return math.degrees(self.psi_rad)

    @property
    def delta_deg(self) -> float:
        return math.degrees(self.delta_rad)


def _finite_complex(z: complex) -> bool:
    return math.isfinite(z.real) and math.isfinite(z.imag)


def _invalid_pol() -> PolarizationResult:
    z = complex(math.nan, math.nan)
    return PolarizationResult(z, z, math.nan, math.nan, math.nan,
                              math.nan, math.nan, math.nan, False, "invalid")


def _failed_point(k, theta, notes, message):
    """Некорректно представимая общая геометрия: оба канала недостоверны."""
    nan = math.nan
    z = complex(nan, nan)
    notes.append(Diagnostic("NUMERICAL_FAILURE", message))
    notes.append(Diagnostic("ELLIPSOMETRY_UNDEFINED", "Численно не рассчитаны отражённые компоненты"))
    return PointResult(k, 2*math.pi/k, theta, _invalid_pol(), _invalid_pol(),
                       z, z, z, nan, nan, False, False, False, tuple(notes))


def _angle(theta: float, unit: str) -> float:
    if unit not in ("deg", "rad"):
        raise ValueError("angle_unit должен быть 'deg' или 'rad'")
    t = math.radians(float(theta)) if unit == "deg" else float(theta)
    if not math.isfinite(t) or not 0 <= t < math.pi / 2:
        raise ValueError("Угол должен удовлетворять 0 <= theta < pi/2")
    return t


def _transparent(value: complex, name: str) -> float:
    z = complex(value)
    if not _finite_complex(z) or z.imag != 0 or z.real <= 0:
        raise ValueError(f"{name}: нужен действительный положительный показатель")
    return z.real


def _prepare(k, indices, thickness_um, substrate, theta, ambient, angle_unit):
    k = float(k)
    if not math.isfinite(k) or k <= 0:
        raise ValueError("k должен быть конечным положительным (rad/мкм)")
    n = np.asarray(indices, dtype=complex)
    d = np.asarray(thickness_um, dtype=float)
    if n.ndim != 1 or d.ndim != 1 or n.shape != d.shape:
        raise ValueError("indices и thickness_um: одномерные массивы одинаковой длины")
    if not np.all(np.isfinite(n)) or not np.all(np.isfinite(d)) or np.any(d < 0):
        raise ValueError("Показатели/толщины должны быть конечны, толщины >= 0")
    if np.any(n.real < 0) or np.any(n.imag > 0):
        raise ValueError("Слои требуют n>=0 и Im(n_tilde)<=0 (соглашение n-iκ)")
    return (k, n, d, _transparent(substrate, "Подложка"),
            _angle(theta, angle_unit), _transparent(ambient, "Внешняя среда"))


def forward_q(index: complex, ambient: float, theta_rad: float,
              config: SolverConfig = SolverConfig()) -> tuple[complex, complex, bool]:
    """Вернуть (q, q², critical_roundoff) для пассивного материала, O(1).

    Между n²-alpha² и (n-n0)(n+n0)+(n0*cos(theta))² выбирается форма
    с меньшей суммой модулей слагаемых; вторая сохраняет точность для
    совпадающих сред при скользящем падении. При Im(q)>0 корень меняет знак.
    Обработка неопределённости округления выполняется только без потерь.
    Внешние проверки входов выполняет solve_point.
    """
    n = complex(index)
    q0 = ambient * math.cos(theta_rad)
    term = (n - ambient) * (n + ambient)
    q2 = term + q0 * q0
    scale = abs(term) + q0*q0
    alpha = ambient*math.sin(theta_rad)
    direct_scale = abs(n*n) + alpha*alpha
    if direct_scale < scale:
        q2 = n*n-alpha*alpha
        scale = direct_scale
    if not _finite_complex(q2):
        raise ArithmeticError("Переполнение при вычислении q²")
    uncertainty = config.critical_roundoff_factor * np.finfo(float).eps * scale
    snapped = bool(n.imag == 0 and q2 != 0 and abs(q2) <= uncertainty)
    if snapped:
        q2 = 0j
    q = cmath.sqrt(q2)
    if q.imag > 0 or (q.imag == 0 and q.real < 0):
        q = -q
    return q, q2, snapped


def _scaled_layer(k: float, n: complex, d: float, q: complex, q2: complex,
                  yref: float, pol: str) -> tuple[complex, complex, complex, complex]:
    """Вернуть A,P,Q,log(u) масштабированного слоя без растущих экспонент.

    F=(1-u²)/(2q), предел F=i*k*d. Для малой фазы используется ряд
    exprel, чтобы не делить два исчезающих числа. Сложность O(1).
    """
    kd = k * d
    phi = kd * q
    if not math.isfinite(kd) or not _finite_complex(phi):
        raise ArithmeticError("Переполнение оптической толщины")
    logu = -1j * phi
    x = 2 * logu
    if not _finite_complex(x):
        raise ArithmeticError("Удвоенная оптическая фаза вне представимости float64")
    if abs(x) < 1e-4:
        # expm1(x)/x = 1+x/2+x²/6+...; погрешность < float64 epsilon.
        exprel = 1 + x * (0.5 + x * (1/6 + x * (1/24 + x * (1/120 + x/720))))
        F = 1j * kd * exprel
        u2 = cmath.exp(x)
    else:
        u2 = cmath.exp(x)
        F = -complex(np.expm1(x)) / (2 * q)
    A = 1 + u2
    if pol == "s":
        P, Q = F * yref, F * q2 / yref
    else:
        eps = n * n
        if eps == 0:
            # При alpha=0 q²/eps имеет непрерывный предел 1.
            if q2 != 0:
                raise NotImplementedError("p: epsilon=0 при ненормальном падении")
            ratio = 1.0
        else:
            ratio = q2 / eps
        P, Q = F * yref * ratio, F * eps / yref
    if not all(_finite_complex(z) for z in (A, P, Q)):
        raise ArithmeticError("Переполнение масштабированного блока слоя")
    return A, P, Q, logu


def _terminal(ns: float, qs: complex, yref: float, pol: str):
    """Граница фиктивного порта с подложкой, включая точное q_sub=0."""
    if pol == "s":
        den = yref + qs
        return (yref - qs) / den, 2 * yref / den
    den = yref * qs + ns * ns
    return ((yref * qs - ns * ns) / den, 2 * yref * qs / den)


def _power(r, log_abs_t, phase_t, qs, ns, yref, pol, cfg, notes):
    R = abs(r) ** 2
    if log_abs_t == -math.inf:
        t = 0j
        phase_t = math.nan
    else:
        modulus = math.exp(log_abs_t)
        t = modulus * cmath.exp(1j * phase_t)
        if modulus == 0:
            notes.append(Diagnostic("AMPLITUDE_UNDERFLOW", "t округлилось до нуля; log|t| сохранён", pol))
    if qs.real == 0:
        T, logT, status = 0.0, -math.inf, "zero_normal_flux"
    else:
        log_y = math.log(qs.real) if pol == "s" else 2*math.log(ns)-math.log(qs.real)
        logT = log_y - math.log(yref) + 2 * log_abs_t
        if not math.isfinite(logT):
            raise ArithmeticError("log(T) вне представимости float64")
        T = math.exp(logT)
        status = "propagating"
        if T == 0 and math.isfinite(logT):
            status = "power_underflow"
            notes.append(Diagnostic("POWER_UNDERFLOW", "Положительное T не представимо; log(T) сохранён", pol))
    A = 1 - R - T
    if R > 1 + cfg.energy_tol or T > 1 + cfg.energy_tol or A < -cfg.energy_tol:
        notes.append(Diagnostic("ENERGY_BALANCE", "Нарушение пассивности выше допуска; значения не обрезаны", pol))
    return PolarizationResult(r, t, R, T, A, log_abs_t, phase_t, logT, True, status)


def ellipsometry(r_s: complex, r_p: complex, atol: float = 1e-12):
    """rho, Psi, Delta, их три маски; r_p уже в отражённом p-базисе.

    atan2 сохраняет Psi при нуле одной компоненты. Разность фаз не
    перемножает малые амплитуды. Delta в [0,2pi); сложность O(1).
    """
    finite = _finite_complex(r_s) and _finite_complex(r_p)
    if not finite:
        return complex(math.nan, math.nan), math.nan, math.nan, False, False, False
    a, b = math.hypot(r_s.real, r_s.imag), math.hypot(r_p.real, r_p.imag)
    if not math.isfinite(a) or not math.isfinite(b):
        return complex(math.nan, math.nan), math.nan, math.nan, False, False, False
    rho_valid = bool(finite and a > atol)
    psi_valid = bool(finite and max(a, b) > atol)
    delta_valid = bool(finite and min(a, b) > atol)
    rho = r_p / r_s if rho_valid else complex(math.nan, math.nan)
    if rho_valid and not _finite_complex(rho):
        rho_valid, rho = False, complex(math.nan, math.nan)
    psi = math.atan2(b, a) if psi_valid else math.nan
    delta = (cmath.phase(r_p) - cmath.phase(r_s)) % (2*math.pi) if delta_valid else math.nan
    return rho, psi, delta, rho_valid, psi_valid, delta_valid


def solve_point(k: float, indices: Sequence[complex], thickness_um: Sequence[float],
                substrate: float, theta: float, ambient: float = 1.0,
                angle_unit: str = "deg", config: SolverConfig = SolverConfig()) -> PointResult:
    """Рассчитать одну точку, O(N) времени/памяти; float64/complex128.

    k: rad/мкм; indices: N показателей n-iκ в порядке от внешней среды;
    thickness_um: N геометрических толщин в мкм. Внешние среды прозрачны.
    theta: угол к нормали (deg/rad). Нулевая толщина удаляется до делений.
    Ошибки входов -> ValueError. Численная/физическая особенность одной
    поляризации -> valid=False и diagnostic, другая сохраняется.
    """
    k, n, d, ns, th, n0 = _prepare(k, indices, thickness_um, substrate, theta, ambient, angle_unit)
    notes: list[Diagnostic] = []
    q0 = n0 * math.cos(th)
    if not math.isfinite(q0) or q0 <= 0:
        return _failed_point(k, th, notes, "Нулевой/непредставимый падающий нормальный поток")
    try:
        qs, _, snap = forward_q(ns, n0, th, config)
        if snap:
            notes.append(Diagnostic("CRITICAL_ROUNDOFF", "Подложка: q² в неопределённости округления; применён предел"))
        active = []
        for j, (nj, dj) in enumerate(zip(n, d), 1):
            if dj == 0:
                continue
            nj, dj = complex(nj), float(dj)
            q, q2, snap = forward_q(nj, n0, th, config)
            if snap:
                notes.append(Diagnostic("CRITICAL_ROUNDOFF", "q² в неопределённости округления; применён предел", layer=j))
            phase_error = float(np.finfo(float).eps) * abs(k * dj * q.real)
            if phase_error > config.phase_error_tol:
                notes.append(Diagnostic("PHASE_PRECISION", f"Оценка ошибки аргумента фазы {phase_error:.3g} rad", layer=j))
            active.append((j, nj, dj, q, q2))
    except (ArithmeticError, ValueError) as exc:
        return _failed_point(k, th, notes, str(exc))
    output = {}
    for pol in ("s", "p"):
        yref = q0 if pol == "s" else n0 / math.cos(th)
        try:
            if not math.isfinite(yref):
                raise ArithmeticError("Непредставимый портовый адмиттанс")
            b, tau = _terminal(ns, qs, yref, pol)
            zero_amplitude = tau == 0
            logabs = math.log(abs(tau)) if tau != 0 else -math.inf
            phase = cmath.phase(tau)
            for j, nj, dj, q, q2 in reversed(active):
                if pol == 'p' and nj == 0 and th != 0:
                    raise NotImplementedError("p: epsilon=0 при ненормальном падении")
                A, P, Q, logu = _scaled_layer(k, nj, dj, q, q2, yref, pol)
                v, w = P*(1-b), Q*(1+b)
                H = A + v + w
                scale = abs(A) + abs(v) + abs(w)
                if H == 0 or not _finite_complex(H):
                    raise ArithmeticError(f"Вырожденный знаменатель в слое {j}")
                if abs(H) < config.denominator_rtol * scale:
                    notes.append(Diagnostic("SMALL_DENOMINATOR", "Потеря относительной точности около резонанса", pol, j))
                b = (A*b + v - w) / H
                logabs += math.log(2.0) + logu.real - math.log(abs(H))
                phase = math.remainder(phase + logu.imag - cmath.phase(H), 2*math.pi)
            if not _finite_complex(b):
                raise ArithmeticError("Неконечная амплитуда отражения")
            if not zero_amplitude and not math.isfinite(logabs):
                raise ArithmeticError("log|t| вне представимости float64")
            output[pol] = _power(b, logabs, phase, qs, ns, yref, pol, config, notes)
        except (ArithmeticError, ValueError, NotImplementedError) as exc:
            code = "UNSUPPORTED_ENZ" if isinstance(exc, NotImplementedError) else "NUMERICAL_FAILURE"
            notes.append(Diagnostic(code, str(exc), pol))
            output[pol] = _invalid_pol()
    rs, rp = output['s'].r_tan, -output['p'].r_tan
    ell = ellipsometry(rs, rp, config.ellipsometry_atol)
    if not ell[-1]:
        notes.append(Diagnostic("ELLIPSOMETRY_UNDEFINED", "Delta недостоверна при малой/невалидной отражённой компоненте"))
    return PointResult(k, 2*math.pi/k, th, output['s'], output['p'], rs, rp,
                       *ell, tuple(notes))


def matrix_reference(k: float, indices: Sequence[complex], thickness_um: Sequence[float],
                     substrate: float, theta: float, ambient: float = 1.0,
                     angle_unit: str = "deg", config: SolverConfig = SolverConfig()):
    """Независимое перемножение матриц книги: dict s/p -> (r_tan,t_tan).

    Только проверочный метод при умеренных фазовых толщинах. При |Im φ|>20
    или переполнении отклоняет расчёт. q=0 обрабатывается через sinc.
    Сложность O(N), входы/единицы совпадают с solve_point.
    """
    k, n, d, ns, th, n0 = _prepare(k, indices, thickness_um, substrate, theta, ambient, angle_unit)
    qs, _, _ = forward_q(ns, n0, th, config)
    output = {}
    for pol in ('s', 'p'):
        y0 = n0*math.cos(th) if pol == 's' else n0/math.cos(th)
        M = np.eye(2, dtype=complex)
        for nj, dj in zip(n, d):
            if dj == 0:
                continue
            if pol == 'p' and nj == 0 and th != 0:
                raise NotImplementedError("p ENZ при ненормальном падении")
            q, q2, _ = forward_q(nj, n0, th, config)
            phi = k*dj*q
            if abs(phi.imag) > 20:
                raise ValueError("Матрица-эталон предназначена для умеренных оптических толщин")
            f = 1j*k*dj*complex(np.sinc(phi/np.pi))
            eps = nj*nj
            if pol == 's':
                upper, lower = f, f*q2
            else:
                if eps == 0 and q2 != 0:
                    raise NotImplementedError("p ENZ при ненормальном падении")
                upper, lower = f*(q2/eps if eps != 0 else 1), f*eps
            c = cmath.cos(phi)
            M = M @ np.array([[c, upper], [lower, c]], dtype=complex)
        if pol == 'p' and qs == 0:
            vec = np.array([0, 1], dtype=complex)
        else:
            ys = qs if pol == 's' else ns*ns/qs
            vec = np.array([1, ys], dtype=complex)
        B, C = M @ vec
        den = y0*B+C
        if den == 0 or not np.all(np.isfinite(M)):
            raise ArithmeticError("Проверочная матрица вырождена/переполнена")
        output[pol] = ((y0*B-C)/den, 2*y0*vec[0]/den)
    return output


def convert_time_convention(result: PointResult, convention: str) -> PointResult:
    """Вернуть копию точки для exp(+iwt) или exp(-iwt), O(1).

    Геометрические базисы сохраняются, амплитуды сопрягаются, Delta
    меняет знак. Энергии, маски, Psi, log|t| и logT остаются прежними.
    """
    if convention not in ('exp(+iwt)', 'exp(-iwt)'):
        raise ValueError("Неизвестное временное соглашение")
    if convention == result.time_convention:
        return result
    def pol(x):
        return replace(x, r_tan=x.r_tan.conjugate(), t_tan=x.t_tan.conjugate(), phase_t=-x.phase_t)
    return replace(result, s=pol(result.s), p=pol(result.p),
                   r_s=result.r_s.conjugate(), r_p=result.r_p.conjugate(),
                   rho=result.rho.conjugate(),
                   delta_rad=(-result.delta_rad) % (2*math.pi) if result.delta_valid else math.nan,
                   time_convention=convention)
