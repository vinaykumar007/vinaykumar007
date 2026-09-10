"""Explicit 1D transient thermal-diffusion solver.

The solver implements

    dT/dt = alpha * d2T/dx2

using a forward-Euler time step and a centered second derivative.
"""

from __future__ import annotations

import numpy as np


def stability_limit(alpha: float, dx: float) -> float:
    """Return the maximum stable time step for the explicit 1D scheme."""
    if alpha <= 0.0:
        raise ValueError("alpha must be positive")
    if dx <= 0.0:
        raise ValueError("dx must be positive")
    return 0.5 * dx**2 / alpha


def simulate(
    length: float,
    nx: int,
    total_time: float,
    dt: float,
    alpha: float,
    left_temperature: float,
    right_temperature: float,
    initial_temperature: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Simulate 1D heat diffusion with fixed boundary temperatures.

    Returns
    -------
    x : ndarray
        Spatial coordinates.
    temperature : ndarray
        Temperature field with shape (nt + 1, nx).
    """
    if length <= 0.0:
        raise ValueError("length must be positive")
    if nx < 3:
        raise ValueError("nx must be at least 3")
    if total_time <= 0.0:
        raise ValueError("total_time must be positive")
    if dt <= 0.0:
        raise ValueError("dt must be positive")
    if alpha <= 0.0:
        raise ValueError("alpha must be positive")

    x = np.linspace(0.0, length, nx)
    dx = x[1] - x[0]
    limit = stability_limit(alpha, dx)
    if dt > limit:
        raise ValueError(
            f"Unstable time step: dt={dt:g} exceeds the explicit stability "
            f"limit {limit:g}."
        )

    n_steps = int(np.ceil(total_time / dt))
    actual_dt = total_time / n_steps
    if actual_dt > limit:
        raise ValueError("Requested discretization violates the stability limit")

    temperature = np.empty((n_steps + 1, nx), dtype=float)
    temperature[0, :] = initial_temperature
    temperature[0, 0] = left_temperature
    temperature[0, -1] = right_temperature

    r = alpha * actual_dt / dx**2
    for n in range(n_steps):
        previous = temperature[n]
        current = previous.copy()
        current[1:-1] = previous[1:-1] + r * (
            previous[2:] - 2.0 * previous[1:-1] + previous[:-2]
        )
        current[0] = left_temperature
        current[-1] = right_temperature
        temperature[n + 1] = current

    return x, temperature
