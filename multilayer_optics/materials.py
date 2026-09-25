"""Спектральные единицы и пассивные оптические материалы.

Внутренние единицы пакета: длина в мкм, угловое вакуумное волновое число
``k = 2*pi/lambda_vac`` в рад/мкм. Временное соглашение — ``exp(+i*omega*t)``,
поэтому комплексный показатель имеет вид ``n - 1j*kappa``. В этом модуле нет
физического расчёта отражения; функции пригодны для независимой проверки входа.
"""

from dataclasses import dataclass, field
from typing import Any

import numpy as np


LENGTH_TO_UM = {"nm": 1e-3, "um": 1.0, "mm": 1e3, "cm": 1e4, "m": 1e6}
SPECTRAL_KINDS = (
    "angular_wavenumber", "spectroscopic_wavenumber", "wavelength"
)
INDEX_CONVENTIONS = ("n_minus_ik", "n_plus_ik")


def _unit_name(unit: str) -> str:
    if not isinstance(unit, str):
        raise ValueError("Единица должна быть строкой и задаваться явно.")
    return unit.strip().replace("μ", "u").replace("µ", "u").replace(" ", "")


def _length_factor(unit: str) -> float:
    name = _unit_name(unit)
    if name not in LENGTH_TO_UM:
        raise ValueError("Единица длины должна быть nm, um, mm, cm или m.")
    return LENGTH_TO_UM[name]


def _inverse_factor(unit: str, kind: str) -> float:
    name = _unit_name(unit).replace("⁻¹", "^-1").replace("−", "-")
    if name.startswith("rad/"):
        if kind != "angular_wavenumber":
            raise ValueError("Единица rad/длина разрешена только для angular_wavenumber.")
        return _length_factor(name[4:])
    if name.startswith("1/"):
        return _length_factor(name[2:])
    if name.endswith("^-1"):
        return _length_factor(name[:-3])
    if name.endswith("-1"):
        return _length_factor(name[:-2])
    raise ValueError("Волновое число требует обратной единицы, например cm^-1 или 1/um.")


def _positive_real(values: Any, name: str) -> np.ndarray:
    raw = np.asarray(values)
    if np.iscomplexobj(raw) or raw.dtype.kind == "b":
        raise ValueError(f"{name}: требуются положительные действительные числа.")
    try:
        array = np.asarray(values, dtype=np.float64)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name}: не удалось прочитать действительные числа.") from exc
    if array.size == 0 or not np.all(np.isfinite(array)) or np.any(array <= 0):
        raise ValueError(f"{name}: все значения должны быть конечными и > 0.")
    return array


def _check_kind(kind: str) -> None:
    if kind not in SPECTRAL_KINDS:
        raise ValueError(f"Неизвестный тип спектральной координаты {kind!r}; допустимы {SPECTRAL_KINDS}.")


def length_to_um(values: Any, unit: str = "um") -> np.ndarray:
    """Перевести неотрицательные длины в мкм, сохраняя форму массива.

    ``values`` — скаляр или массив геометрических толщин; ``unit`` — nm, um,
    mm, cm либо m. Возвращается float64 ndarray, включая массив размерности 0
    для скаляра. Нечисловой, комплексный, отрицательный или бесконечный ввод
    вызывает ValueError. Время и дополнительная память O(число значений).
    """
    raw = np.asarray(values)
    if np.iscomplexobj(raw) or raw.dtype.kind == "b":
        raise ValueError("Толщина должна быть действительной неотрицательной длиной.")
    try:
        array = np.asarray(values, dtype=np.float64)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("Не удалось прочитать толщину.") from exc
    if not np.all(np.isfinite(array)) or np.any(array < 0):
        raise ValueError("Все толщины должны быть конечными и >= 0.")
    with np.errstate(over="ignore", under="ignore"):
        result = np.asarray(array * _length_factor(unit))
    if not np.all(np.isfinite(result)) or np.any((array > 0) & (result == 0)):
        raise ValueError("Толщина не представима в float64 после перевода в мкм.")
    return result


def to_k(values: Any, kind: str = "angular_wavenumber", unit: str = "um^-1") -> np.ndarray:
    """Перевести спектральную координату в k [рад/мкм], сохранив форму.

    ``kind`` явно задаёт angular_wavenumber (2*pi/lambda),
    spectroscopic_wavenumber (1/lambda) или wavelength. ``unit`` — единица
    длины для wavelength либо обратная единица для волновых чисел. Например,
    10000 cm^-1 спектроскопического числа даёт 2*pi рад/мкм; 10000 cm^-1
    углового числа даёт 1 рад/мкм. Смысл координаты не выводится из единицы.

    Допускаются скаляр и массив любой формы; порядок не меняется. Требуются
    конечные положительные значения. При недопустимом вводе или выходе за
    представимость float64 возбуждается ValueError. Время/память O(M).
    """
    _check_kind(kind)
    array = _positive_real(values, "Спектральная координата")
    with np.errstate(over="ignore", under="ignore", divide="ignore", invalid="ignore"):
        if kind == "wavelength":
            # Деления по отдельности избегают преждевременного переполнения lambda*factor.
            result = (2.0 * np.pi / array) / _length_factor(unit)
        else:
            result = array / _inverse_factor(unit, kind)
            if kind == "spectroscopic_wavenumber":
                result = result * (2.0 * np.pi)
    return _positive_real(result, "Волновое число после преобразования")


def from_k(k: Any, kind: str = "angular_wavenumber", unit: str = "um^-1") -> np.ndarray:
    """Преобразовать k [рад/мкм] в указанную спектральную координату.

    Обратная операция к to_k; форма и порядок сохраняются. При kind=wavelength
    возрастающему k соответствует убывающая длина волны — сортировка не
    выполняется. Ограничения ввода, единицы и ошибки совпадают с to_k.
    Время и дополнительная память O(M).
    """
    _check_kind(kind)
    array = _positive_real(k, "Внутреннее волновое число k")
    with np.errstate(over="ignore", under="ignore", divide="ignore", invalid="ignore"):
        if kind == "wavelength":
            result = (2.0 * np.pi / array) / _length_factor(unit)
        else:
            value = array if kind == "angular_wavenumber" else array / (2.0 * np.pi)
            result = value * _inverse_factor(unit, kind)
    return _positive_real(result, "Спектральная координата после преобразования")


def validate_spectral_grid(values: Any) -> np.ndarray:
    """Проверить исходную одномерную спектральную сетку с M >= 2.

    Все узлы должны быть конечными, положительными и строго возрастающими.
    Повторные или неупорядоченные узлы отклоняются с ValueError; молчаливая
    сортировка отсутствует. Возвращается отдельная float64 копия. Проверяется
    именно исходная координата, до to_k: возрастающая сетка длин волн даёт
    убывающий внутренний k, что допустимо. Время и память O(M).
    """
    array = _positive_real(values, "Спектральная сетка")
    if array.ndim != 1 or array.size < 2:
        raise ValueError("Спектральная сетка должна быть одномерной и содержать M >= 2 узла.")
    if not np.all(array[1:] > array[:-1]):
        raise ValueError("Узлы исходной спектральной сетки должны строго возрастать без повторов.")
    return array.copy()


def make_grid(start: float, stop: float, count: int) -> np.ndarray:
    """Построить равномерную сетку [start, stop], включающую оба конца.

    Границы относятся к одной явно выбранной вызывающим кодом координате;
    требуются конечные 0 < start < stop и целое count >= 2. После np.linspace
    проверяется строгая монотонность: невозможность разместить столько разных
    чисел float64 внутри слишком узкого интервала вызывает ValueError.
    Время и память O(count).
    """
    if isinstance(count, (bool, np.bool_)) or not isinstance(count, (int, np.integer)) or count < 2:
        raise ValueError("Число узлов должно быть целым count >= 2.")
    bounds = _positive_real([start, stop], "Границы спектрального интервала")
    if bounds.shape != (2,) or bounds[0] >= bounds[1]:
        raise ValueError("Границы должны быть скалярами и удовлетворять 0 < start < stop.")
    return validate_spectral_grid(np.linspace(bounds[0], bounds[1], int(count), endpoint=True))


def _canonical_indices(values: Any, convention: str) -> np.ndarray:
    if convention not in INDEX_CONVENTIONS:
        raise ValueError(f"Соглашение показателя должно быть одним из {INDEX_CONVENTIONS}.")
    try:
        array = np.asarray(values, dtype=np.complex128)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("Не удалось прочитать комплексные показатели материалов.") from exc
    if convention == "n_plus_ik":
        array = np.conjugate(array)
    if not np.all(np.isfinite(array)):
        raise ValueError("Показатели материалов не должны содержать NaN или Inf.")
    if np.any(array.real < 0) or np.any(array.imag > 0):
        raise ValueError("Пассивная модель требует n >= 0 и kappa >= 0 при n - i*kappa.")
    # При n>=0, Im(n_tilde)<=0 имеем Im(epsilon)=2*n*Im(n_tilde)<=0.
    # Не возводим большие показатели в квадрат только ради этой проверки.
    return array


@dataclass(frozen=True)
class TabulatedMaterial:
    """Табличный пассивный материал с линейной интерполяцией n и kappa.

    ``x`` — строго возрастающий одномерный массив M>=2 исходной координаты;
    ``values`` — комплексные показатели той же формы; ``kind``/``unit``
    однозначно описывают x. ``convention`` явно задаёт n_minus_ik или n_plus_ik.
    Таблица копируется и проверяется при создании, исходные массивы сохраняются
    доступными только для чтения. Значения интерполируются именно по x, даже
    когда x — длина волны, а расчёт задан в k.

    Вызов ``table(k)`` принимает k [рад/мкм] любой формы и возвращает
    complex128 той же формы в соглашении n_minus_ik. Экстраполяция запрещена.
    Нарушения координат, пассивности, формы или диапазона вызывают ValueError.
    Создание занимает O(M) времени и памяти, оценка Q узлов — O(Q*log(M))
    времени для общего порядка запроса и O(Q) дополнительной памяти.
    """

    x: Any
    values: Any
    kind: str = "angular_wavenumber"
    unit: str = "um^-1"
    convention: str = "n_minus_ik"
    _indices: np.ndarray = field(init=False, repr=False, compare=False)
    _k_min: float = field(init=False, repr=False)
    _k_max: float = field(init=False, repr=False)

    def __post_init__(self) -> None:
        coordinates = validate_spectral_grid(self.x)
        indices = _canonical_indices(self.values, self.convention)
        if indices.shape != coordinates.shape:
            raise ValueError("x и values таблицы должны быть одномерными массивами одинаковой длины.")
        canonical = indices.copy()
        source = np.asarray(self.values, dtype=np.complex128).copy()
        k_bounds = to_k(coordinates[[0, -1]], self.kind, self.unit)
        for array in (coordinates, canonical, source):
            array.setflags(write=False)
        object.__setattr__(self, "x", coordinates)
        object.__setattr__(self, "values", source)
        object.__setattr__(self, "_indices", canonical)
        object.__setattr__(self, "_k_min", float(np.min(k_bounds)))
        object.__setattr__(self, "_k_max", float(np.max(k_bounds)))

    def __call__(self, k: Any) -> np.ndarray:
        """Оценить n-i*kappa в узлах k [рад/мкм] без экстраполяции."""
        k_array = _positive_real(k, "Узлы k для материала")
        if np.any(k_array < self._k_min) or np.any(k_array > self._k_max):
            raise ValueError(
                "Запрос материала выходит за диапазон таблицы; экстраполяция запрещена. "
                f"Допустимый k: [{self._k_min:.17g}, {self._k_max:.17g}] рад/мкм."
            )
        query = from_k(k_array, self.kind, self.unit)
        # Диапазон уже строго проверен в k. Clip устраняет лишь округление
        # обратного перевода концов интервала, а не разрешает экстраполяцию.
        query = np.clip(query, self.x[0], self.x[-1])
        flat_query = query.reshape(-1)
        n = np.interp(flat_query, self.x, self._indices.real)
        kappa = np.interp(flat_query, self.x, -self._indices.imag)
        return np.asarray(n - 1j * kappa).reshape(k_array.shape)


def material_values(material: Any, k_array: Any, convention: str = "n_minus_ik") -> np.ndarray:
    """Получить комплексные показатели материала в узлах k [рад/мкм].

    ``material`` — числовая константа, функция ``material(k_array)`` либо
    TabulatedMaterial. Функция материала получает ndarray float64 в рад/мкм
    и должна вернуть скаляр или массив точно той же формы. Возвращается
    отдельный complex128 массив формы k_array с n>=0, Im(n_tilde)<=0.

    ``convention`` относится к константе/результату функции; n_plus_ik явно
    преобразуется сопряжением. У TabulatedMaterial соглашение задаётся в его
    конструкторе, поэтому внешнее n_plus_ik для него отклоняется во избежание
    двойного сопряжения. Ошибки формы, пассивности, соглашения и NaN/Inf
    вызывают ValueError; собственные ошибки функции материала передаются
    вызывающему коду. Время/память O(M), кроме стоимости функции/таблицы.
    """
    k = _positive_real(k_array, "Узлы k для материала")
    if isinstance(material, TabulatedMaterial):
        if convention != "n_minus_ik":
            raise ValueError("Для TabulatedMaterial задавайте исходное соглашение в конструкторе таблицы.")
        raw = material(k)
    elif callable(material):
        raw = material(k.copy())
    else:
        raw = np.asarray(material)
        if raw.ndim != 0:
            raise ValueError("Материал должен быть скаляром, функцией k или TabulatedMaterial.")
    array = _canonical_indices(raw, convention)
    if array.ndim == 0:
        return np.full(k.shape, array.item(), dtype=np.complex128)
    if array.shape != k.shape:
        raise ValueError("Функция материала должна вернуть скаляр или массив формы k_array.")
    return array.copy()


def evaluate_index(
    material: Any,
    k_array: Any,
    transparent: bool = False,
    convention: str = "n_minus_ik",
) -> np.ndarray:
    """Получить материал с дополнительной проверкой прозрачной среды.

    Параметры и сложность совпадают с material_values. При transparent=True
    требуется строго действительный показатель n>0 в каждом запрошенном
    узле: даже малая ненулевая мнимая часть вызывает ValueError и никогда
    молча не отбрасывается. Для таблицы дополнительно проверяются ВСЕ её
    исходные значения; линейная интерполяция прозрачность сохраняет. Для
    произвольной функции проверяются только запрошенные узлы — доказать её
    прозрачность между узлами конечной выборкой невозможно.
    """
    indices = material_values(material, k_array, convention)
    if transparent:
        check = material._indices if isinstance(material, TabulatedMaterial) else indices
        if np.any(check.imag != 0) or np.any(check.real <= 0):
            raise ValueError("Внешняя среда и подложка должны быть непоглощающими: n > 0 и Im(n) = 0.")
    return indices
