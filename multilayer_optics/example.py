"""Воспроизводимый модельный спектр; оптические константы синтетические.

Запуск: python -m multilayer_optics.example после установки пакета, либо
python output/multilayer_optics/example.py из рабочего каталога.
Matplotlib не требуется; CSV/JSON отделены от необязательного построения графика.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path
import sys

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from multilayer_optics import Layer, calculate_spectrum, make_grid, refine_spectrum


def model_layers():
    """Возвращает пять слоёв от внешней среды к подложке.

    Гладкая аналитическая дисперсия служит проверке вычислений, не описывает
    конкретные реальные материалы и не является подбором по эксперименту.
    """
    def high(k):
        lam = 2*np.pi/k
        return 2.05 + 0.015/lam**2 - 1j*(0.012+0.004*np.exp(-((lam-0.52)/0.06)**2))
    def low(k):
        lam = 2*np.pi/k
        return 1.42 + 0.008/lam**2
    return [Layer(high, 70, name="H1, model"), Layer(low, 110, name="L1, model"),
            Layer(high, 140, name="H2, model"), Layer(low, 110, name="L2, model"),
            Layer(high, 70, name="H3, model")]


def model_substrate(k):
    """Действительная дисперсия полубесконечной непоглощающей подложки."""
    return 1.50 + 0.004/(2*np.pi/k)**2


def main():
    root = Path(__file__).resolve().parent
    folder = root / 'examples'
    folder.mkdir(exist_ok=True)
    # Спектроскопические cm^-1: 800..400 нм в порядке возрастающего k.
    grid = make_grid(12500, 25000, 501)
    kwargs = dict(kind='spectroscopic_wavenumber', unit='cm^-1')
    result = calculate_spectrum(grid, model_layers(), model_substrate, 60, **kwargs)
    result.write_json(folder/'example_spectrum.json')
    data = result.arrays()
    names = ['coordinate', 'k_rad_per_um', 'lambda_um', 'Rs', 'Rp', 'Ts', 'Tp', 'As', 'Ap',
             'Psi_deg', 'Delta_deg']
    columns = [data[name] if name not in ('Psi_deg', 'Delta_deg') else data[name.lower()] for name in names]
    with (root/'example_spectrum.csv').open('w', encoding='utf-8', newline='') as out:
        writer = csv.writer(out)
        writer.writerow(names)
        writer.writerows(zip(*columns))
    refined = refine_spectrum(make_grid(12500, 25000, 101), model_layers(), model_substrate,
                              60, max_rounds=6, max_points=12801, **kwargs)
    summary = {
        "description": "Five synthetic dispersive layers, absorbing H layers, real dispersive semi-infinite substrate",
        "angle_deg": 60,
        "points": len(grid),
        "all_polarizations_valid": bool(all(p.s.valid and p.p.valid for p in result.points)),
        "diagnostic_count": sum(len(p.diagnostics) for p in result.points),
        "Rs_range": [float(np.min(data['Rs'])), float(np.max(data['Rs']))],
        "Rp_range": [float(np.min(data['Rp'])), float(np.max(data['Rp']))],
        "As_range": [float(np.min(data['As'])), float(np.max(data['As']))],
        "Ap_range": [float(np.min(data['Ap'])), float(np.max(data['Ap']))],
        "grid_check": {"converged": refined.converged, "rounds": refined.rounds,
                       "final_points": len(refined.spectrum.points),
                       "normalized_midpoint_error": refined.normalized_error,
                       "reason": refined.reason,
                       "scope": "midpoint criterion only; no guarantee of detecting every narrow resonance"},
    }
    (folder/'example_summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
