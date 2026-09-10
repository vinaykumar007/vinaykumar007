"""Small parameter study for the thermal-diffusion model."""

from pathlib import Path
import sys

import numpy as np

# Allow execution as: python examples/parameter_study.py
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.thermal_diffusion import simulate, stability_limit  # noqa: E402


CASES = {
    "low_diffusivity": 0.05,
    "reference": 0.10,
    "high_diffusivity": 0.20,
}

length = 1.0
nx = 51
total_time = 0.10
left_temperature = 100.0
right_temperature = 20.0
initial_temperature = 20.0

x = np.linspace(0.0, length, nx)
dx = x[1] - x[0]

for name, alpha in CASES.items():
    dt = 0.45 * stability_limit(alpha, dx)
    x, temperature = simulate(
        length=length,
        nx=nx,
        total_time=total_time,
        dt=dt,
        alpha=alpha,
        left_temperature=left_temperature,
        right_temperature=right_temperature,
        initial_temperature=initial_temperature,
    )
    center = temperature[-1, nx // 2]
    print(f"{name:18s} alpha={alpha:6.3f}  center_T={center:8.3f}")

print("\nInterpretation: larger thermal diffusivity produces faster smoothing of temperature gradients.")
