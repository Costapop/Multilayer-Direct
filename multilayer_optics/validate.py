"""Запустить тесты и сохранить фактический отчёт проверки рядом с пакетом."""
from __future__ import annotations

from datetime import datetime, timezone
import io
import json
import math
from pathlib import Path
import platform
import sys
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent))
sys.path.insert(0, str(ROOT/'tests'))

from multilayer_optics import solve_point, matrix_reference
from test_core import independent_single_layer_loss


def main():
    folder = ROOT/'examples'
    folder.mkdir(exist_ok=True)
    stream = io.StringIO()
    tests = unittest.defaultTestLoader.discover(str(ROOT/'tests'))
    result = unittest.TextTestRunner(stream=stream, verbosity=2).run(tests)
    (folder/'tests.log').write_text(stream.getvalue(), encoding='utf-8')
    if not result.wasSuccessful():
        print(stream.getvalue())
        raise SystemExit(1)

    rng = np.random.default_rng(20260925)
    worst = 0.0
    for _ in range(60):
        count = int(rng.integers(0, 9))
        indices = rng.uniform(.7, 2.8, count)-1j*rng.uniform(0, .08, count)
        d = rng.uniform(.001, .18, count)
        k, ns, th = rng.uniform(5, 13), rng.uniform(1.1, 1.9), rng.uniform(0, 76)
        actual = solve_point(k, indices, d, ns, th)
        reference = matrix_reference(k, indices, d, ns, th)
        for pol in ('s', 'p'):
            value = getattr(actual, pol)
            r, t = reference[pol]
            worst = max(worst, abs(value.r_tan-r), abs(value.t_tan-t))

    k, index, d, ns, theta = 2*math.pi/.63, 2.1-.18j, .19, 1.5, 53
    actual = solve_point(k, [index], [d], ns, theta)
    absorption = {}
    for pol in ('s', 'p'):
        integral, _, _ = independent_single_layer_loss(k, index, d, ns, theta, pol)
        residual = abs(getattr(actual, pol).A-integral)
        absorption[pol] = {"A_from_balance": getattr(actual, pol).A,
                           "A_from_volume_integral": integral, "absolute_residual": residual}
    benchmark = solve_point(10, [], [], 1.5, 60)
    values = {"Rs": benchmark.s.R, "Rp": benchmark.p.R, "Ts": benchmark.s.T,
              "Tp": benchmark.p.T, "Psi_deg": benchmark.psi_deg, "Delta_deg": benchmark.delta_deg,
              "rho": {"re": benchmark.rho.real, "im": benchmark.rho.imag}}
    opaque = solve_point(10, [2.2-.3j], [1e5], 1.5, 47)
    report = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "python": platform.python_version(), "numpy": np.__version__,
        "tests": {"run": result.testsRun, "failures": len(result.failures),
                  "errors": len(result.errors), "skipped": len(result.skipped)},
        "matrix_comparison": {"stacks": 60, "polarizations_per_stack": 2,
                              "seed": 20260925, "max_complex_amplitude_absolute_residual": worst},
        "independent_absorption_integral": absorption,
        "benchmark": values,
        "opaque_layer": {"log_T_s": opaque.s.log_T, "log_T_p": opaque.p.log_T,
                         "Ts": opaque.s.T, "Tp": opaque.p.T},
    }
    (folder/'validation.json').write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    summary_path = folder/'example_summary.json'
    summary = json.loads(summary_path.read_text()) if summary_path.exists() else None
    lines = [
        '# Фактическая проверка расчётного ядра', '',
        f'Отчёт создан запуском `validate.py`: {report["generated_utc"]}.', '',
        f'Среда: Python {report["python"]}, NumPy {report["numpy"]}; float64/complex128.', '',
        f'**Пройдено {result.testsRun} теста; ошибок и падений нет; пропущенных тестов: {len(result.skipped)}.**', '',
        'Полный журнал: [examples/tests.log](examples/tests.log). Машиночитаемые результаты: [examples/validation.json](examples/validation.json).', '',
        '## Физические и численные проверки', '',
        '| Проверка | Полученный результат |', '|---|---|',
        f'| 60 случайных стеков, 0–8 слоёв, обе поляризации, комплексные r и t против матриц книги | Максимальная абсолютная невязка {worst:.6g} |',
        f'| Независимый интеграл потерь, s | A = {absorption["s"]["A_from_volume_integral"]:.15g}; невязка {absorption["s"]["absolute_residual"]:.6g} |',
        f'| Независимый интеграл потерь, p с учётом нормального E | A = {absorption["p"]["A_from_volume_integral"]:.15g}; невязка {absorption["p"]["absolute_residual"]:.6g} |',
        '| Френель, Эйри, четвертьволновое просветление, угол Брюстера | Пройдены аналитические сравнения и проверки масок |',
        '| Критический конечный слой и критическая подложка | Совпадение с аналитическими предельными матрицами |',
        '| Эванесцентная подложка и поглощающий слой | T=0, R<1, поглощение положительно |',
        '| Угол 89.9999999 градуса, совпадающие материалы | R≈0, T≈1, корректная фаза передачи |',
        '| 1500 конечных слоёв | Конечные результаты и log(T), пассивность соблюдена |',
        f'| Непрозрачный поглощающий слой | Машинные Ts=Tp=0; log(Ts)={opaque.s.log_T:.10g}, log(Tp)={opaque.p.log_T:.10g} остаются конечными |',
        '| Очень толстый эванесцентный слой | Конечный log(T), R≈1, отсутствие растущих экспонент |',
        '| ENZ при нормальном падении | Аналитический предел поддержан |',
        '| Точный ENZ при ненормальном падении, p | Явный UNSUPPORTED_ENZ; s-канал сохраняется |',
        '| Переполнение q² и фазы в предельных float64-входах | NUMERICAL_FAILURE, valid=False, без ложного физического T=0 |',
        '| Материалы и интерфейс | Единицы, интерполяция, отсутствие экстраполяции, строгая прозрачность, JSON, смена соглашения и уточнение сетки проверены |', '',
        'Интеграл поглощения рассчитан независимо: решение четырёх граничных уравнений одного слоя, затем 96-точечная квадратура Гаусса по объёмным потерям. Для p учтена нормальная компонента электрического поля. Это не тождественная проверка A=1−R−T.', '',
        '## Контрольная граница', '',
        'N=0; n0=1; n_sub=1.5; theta0=60 градусов; k=10 rad/um (для постоянных индексов энергии не зависят от k).', '',
        '| Величина | Получено |', '|---|---|',
    ]
    for key in ('Rs','Rp','Ts','Tp','Psi_deg','Delta_deg'):
        lines.append(f'| {key} | {values[key]:.16g} |')
    lines += [f'| rho | {benchmark.rho.real:.16g} + {benchmark.rho.imag:.16g}i |', '',
              '## Допуски и область подтверждения', '',
              'Аналитические проверки: типичные абсолютные допуски 2e-12–3e-12; для случайного матричного сравнения 2e-11*(1+|эталон|). Эти допуски применяются к хорошо обусловленным тестовым случаям, а не заявляют гарантированную ошибку для любого покрытия.', '',
              'Проверки полей относятся к одному поглощающему слою. Случайные матричные сравнения относятся к указанной выборке из 60 стеков. Проверки не заменяют независимую аттестацию на реальных оптических константах и экспериментальных спектрах.', '',
              '## Модельный спектр', '']
    if summary:
        check = summary['grid_check']
        lines += [f'Пять синтетических диспергирующих слоёв, угол 60 градусов, прозрачная диспергирующая подложка; {summary["points"]} исходная точка, диапазон 12500–25000 cm^-1 для 1/lambda (400–800 нм). Все каналы рассчитаны; диагностических сообщений: {summary["diagnostic_count"]}.', '',
                  f'Контроль разрешения: {check["rounds"]} раундов, {check["final_points"]} точка, нормированная невязка {check["normalized_midpoint_error"]:.6g} <= 1. Пройден критерий середин интервалов; это не гарантия обнаружения всех возможных узких резонансов.', '',
                  '[График PNG](examples/spectrum.png), [векторный SVG](examples/spectrum.svg), [CSV](example_spectrum.csv), [JSON](examples/example_spectrum.json).', '']
    lines += ['## Воспроизведение', '', '```bash', 'python example.py', 'python validate.py', 'python plot_example.py  # необязательно, нужен matplotlib', '```', '',
              'Критические ограничения: точный ENZ p при наклонном падении диагностируется, высокоточная арифметика не реализована; крайне узкие резонансы и чрезмерные оптические толщины требуют оценки обусловленности. Подробности — в README_RU.md и THEORY_RU.md.', '']
    (ROOT/'VALIDATION_RU.md').write_text('\n'.join(lines), encoding='utf-8')
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
