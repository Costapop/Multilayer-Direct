"""Опциональные научные графики воспроизводимого синтетического спектра.

Запуск из любого каталога::

    python plot_example.py

По умолчанию читается example_spectrum.csv рядом с этим файлом, результаты
сохраняются в examples/spectrum.png и examples/spectrum.svg. Зависимости:
NumPy и Matplotlib; расчётное ядро пакета от Matplotlib не зависит.

CSV должен содержать заголовок:
coordinate,k_rad_per_um,lambda_um,Rs,Rp,Ts,Tp,As,Ap,Psi_deg,Delta_deg.
Неопределённые углы допускаются как пустые значения или NaN. Данные CSV
никогда не изменяются; сортировка по длине волны применяется только к рисунку.
"""

import argparse
import os
from pathlib import Path
import tempfile

import numpy as np


REQUIRED_COLUMNS = (
    "coordinate", "k_rad_per_um", "lambda_um", "Rs", "Rp", "Ts", "Tp",
    "As", "Ap", "Psi_deg", "Delta_deg",
)


def read_spectrum(path: Path) -> np.ndarray:
    """Прочитать CSV и упорядочить его копию по lambda_um для изображения.

    Файл должен содержать не менее двух строк и все REQUIRED_COLUMNS.
    Спектральные координаты — конечные положительные числа, длины волн без
    повторов. Ошибки данных вызывают ValueError; ошибки чтения — OSError.
    Время O(M log M), память O(M). Неопределённые оптические данные остаются NaN.
    """
    data = np.genfromtxt(path, delimiter=",", names=True, dtype=float, encoding="utf-8")
    data = np.atleast_1d(data)
    names = data.dtype.names or ()
    missing = [name for name in REQUIRED_COLUMNS if name not in names]
    if missing:
        raise ValueError(f"CSV не содержит обязательные поля: {', '.join(missing)}")
    if len(data) < 2:
        raise ValueError("Для графика требуются не менее двух спектральных узлов.")
    for name in ("coordinate", "k_rad_per_um", "lambda_um"):
        if not np.all(np.isfinite(data[name])) or np.any(data[name] <= 0):
            raise ValueError(f"Поле {name} должно содержать конечные положительные числа.")
    data = data[np.argsort(data["lambda_um"], kind="stable")]
    if np.any(np.diff(data["lambda_um"]) <= 0):
        raise ValueError("Длины волн для изображения должны быть различными.")
    return data


def break_wrapped_phase(x: np.ndarray, delta_deg: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Вставить NaN между соседними конечными фазами со скачком >180 градусов.

    Все исходные точки сохраняются: добавляется разрыв линии, а не удаляется
    один из концов скачка. Уже неопределённые фазы также разделяют участки.
    Это только способ изображения обёрнутой Delta, не развёртка и не
    интерполяция фазы. Время и память O(M).
    """
    x = np.asarray(x, dtype=float)
    phase = np.asarray(delta_deg, dtype=float)
    if x.ndim != 1 or x.shape != phase.shape:
        raise ValueError("x и delta_deg должны быть одномерными массивами одинаковой длины.")
    finite_pair = np.isfinite(phase[1:]) & np.isfinite(phase[:-1])
    jumps = np.flatnonzero(finite_pair & (np.abs(np.diff(phase)) > 180.0)) + 1
    return np.insert(x, jumps, np.nan), np.insert(phase, jumps, np.nan)


def plot_spectrum(csv_path: Path, output_dir: Path) -> tuple[Path, Path]:
    """Построить 2x2 график Rs/Rp, Ts/Tp/As/Ap, Psi и Delta.

    График использует длину волны в мкм по горизонтали, углы в градусах и
    исходные энергетические коэффициенты без обрезания. Возвращает пути к
    PNG (200 dpi) и векторному SVG. Возможны ImportError при отсутствии
    Matplotlib, ValueError при ошибке CSV и OSError при ошибке записи.
    Расчёт и чтение материалов не выполняются. Время O(M log M), память O(M).
    """
    os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "multilayer_matplotlib"))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator, MultipleLocator

    data = read_spectrum(csv_path)
    wavelength = data["lambda_um"]
    blue, orange, green, purple = "#2468A2", "#CC6500", "#087E6B", "#8159A0"
    style = {
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "axes.titlesize": 12,
        "axes.labelsize": 10,
        "axes.titleweight": "bold",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.edgecolor": "#59636B",
        "axes.labelcolor": "#27333D",
        "text.color": "#27333D",
        "xtick.color": "#59636B",
        "ytick.color": "#59636B",
        "grid.color": "#DEE4E8",
        "grid.linewidth": 0.65,
        "legend.frameon": False,
        "svg.fonttype": "none",
    }
    with plt.rc_context(style):
        fig, axes = plt.subplots(2, 2, figsize=(11.6, 7.8), sharex=True)
        fig.patch.set_facecolor("white")
        fig.suptitle("Synthetic multilayer spectrum", fontsize=17, fontweight="bold", y=0.97)
        fig.text(0.5, 0.929, "Model optical constants · semi-infinite transparent substrate",
                 ha="center", fontsize=10, color="#66747F")

        ax = axes[0, 0]
        ax.set_title("Reflectance", loc="left", pad=11)
        ax.plot(wavelength, data["Rs"], color=blue, lw=1.9, label=r"$R_s$")
        ax.plot(wavelength, data["Rp"], color=orange, lw=1.9, label=r"$R_p$")
        ax.set_ylabel("Energy coefficient")
        ax.legend(loc="best", ncol=2)

        ax = axes[0, 1]
        ax.set_title("Transmittance and coating absorptance", loc="left", pad=11)
        ax.plot(wavelength, data["Ts"], color=blue, lw=1.8, label=r"$T_s$")
        ax.plot(wavelength, data["Tp"], color=orange, lw=1.8, label=r"$T_p$")
        ax.plot(wavelength, data["As"], color=blue, lw=1.5, linestyle="--", label=r"$A_s$")
        ax.plot(wavelength, data["Ap"], color=orange, lw=1.5, linestyle="--", label=r"$A_p$")
        ax.set_ylabel("Energy coefficient")
        ax.legend(loc="best", ncol=2)

        ax = axes[1, 0]
        ax.set_title("Ellipsometric amplitude angle", loc="left", pad=11)
        ax.plot(wavelength, data["Psi_deg"], color=green, lw=1.8)
        ax.set_ylabel(r"$\Psi$ (degrees)")
        ax.set_ylim(0, 90)
        ax.yaxis.set_major_locator(MultipleLocator(15))
        if not np.any(np.isfinite(data["Psi_deg"])):
            ax.text(.5, .5, "No valid amplitude angle", ha="center", transform=ax.transAxes)

        ax = axes[1, 1]
        ax.set_title("Ellipsometric phase difference", loc="left", pad=11)
        phase_x, phase_y = break_wrapped_phase(wavelength, data["Delta_deg"])
        ax.plot(phase_x, phase_y, color=purple, lw=1.8)
        ax.set_ylabel(r"$\Delta$ (degrees)")
        ax.set_ylim(-6, 366)
        ax.yaxis.set_major_locator(MultipleLocator(90))
        if not np.any(np.isfinite(data["Delta_deg"])):
            ax.text(.5, .5, "No valid phase", ha="center", transform=ax.transAxes)

        for ax in axes.flat:
            ax.set_xlim(wavelength[0], wavelength[-1])
            ax.grid(True, which="major")
            ax.set_axisbelow(True)
            ax.xaxis.set_major_locator(MaxNLocator(nbins=6))
            ax.tick_params(length=3, width=.6)
        for ax in axes[1]:
            ax.set_xlabel(r"Vacuum wavelength $\lambda$ ($\mu$m)")
        fig.text(0.075, 0.035,
                 r"Convention: $e^{+i\omega t}$, $\tilde{n}=n-i\kappa$, $\rho=r_p/r_s=\tan\Psi\,e^{i\Delta}$.",
                 fontsize=9, color="#66747F")
        fig.text(0.075, 0.012, "Phase lines are broken at 360°/0° wraps and undefined values.",
                 fontsize=9, color="#66747F")
        fig.subplots_adjust(left=.075, right=.976, top=.867, bottom=.13, wspace=.26, hspace=.34)
        output_dir.mkdir(parents=True, exist_ok=True)
        png_path = output_dir / "spectrum.png"
        svg_path = output_dir / "spectrum.svg"
        fig.savefig(png_path, dpi=200, facecolor="white", metadata={"Description": "Synthetic optical multilayer spectrum"})
        fig.savefig(svg_path, facecolor="white", metadata={"Title": "Synthetic multilayer spectrum"})
        plt.close(fig)
    return png_path, svg_path


def main() -> None:
    """Прочитать необязательные пути CLI и записать два графических файла."""
    directory = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, default=directory / "example_spectrum.csv")
    parser.add_argument("--output-dir", type=Path, default=directory / "examples")
    args = parser.parse_args()
    try:
        paths = plot_spectrum(args.csv, args.output_dir)
    except ImportError as exc:
        parser.exit(2, f"Не хватает опциональной зависимости: {exc}. Установите matplotlib.\n")
    for path in paths:
        print(path)


if __name__ == "__main__":
    main()
