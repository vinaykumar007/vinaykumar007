# Computational Materials Simulation: 1D Thermal Diffusion

A compact scientific-computing case study that turns a continuum materials problem into a reproducible numerical model in Python.

> **Status:** Initial portfolio study. This repository section is intentionally scoped to a classical continuum model; it does not claim atomistic, DFT, or molecular-dynamics experience.

## Objective

Study transient heat diffusion through a homogeneous engineering material and demonstrate the full modelling workflow:

**Physical problem → governing equation → numerical discretization → Python implementation → convergence/stability checks → parameter study → physical interpretation**

## Physical Model

For one-dimensional conduction with constant thermal diffusivity \(\alpha\):

\[
\frac{\partial T}{\partial t} = \alpha \frac{\partial^2 T}{\partial x^2}
\]

where:

- `T(x,t)` is temperature
- `alpha = k/(rho*c_p)` is thermal diffusivity
- `k` is thermal conductivity
- `rho` is density
- `c_p` is specific heat capacity

### Assumptions

- homogeneous, isotropic material
- constant thermal properties
- one-dimensional heat transfer
- no internal heat generation
- fixed boundary temperatures

These assumptions make the problem small enough to understand completely while still exposing the core numerical issues found in larger simulation workflows.

## Numerical Method

The baseline solver uses the explicit finite-difference approximation:

\[
T_i^{n+1} = T_i^n + Fo\left(T_{i+1}^n - 2T_i^n + T_{i-1}^n\right)
\]

with Fourier number

\[
Fo = \frac{\alpha\Delta t}{\Delta x^2}.
\]

For the standard explicit 1D diffusion scheme, the time step must satisfy the usual stability restriction `Fo <= 0.5`.

The implementation exposes `dx`, `dt`, material properties, and boundary/initial conditions so the numerical behavior can be explored rather than hidden behind a black-box solver.

## Project Structure

```text
computational-materials-simulation/
├── README.md
├── src/
│   └── thermal_diffusion.py
├── tests/
│   └── test_thermal_diffusion.py
└── examples/
    └── parameter_study.py
```

## Reproducibility

Create a Python environment and install NumPy:

```bash
python -m pip install numpy
```

Run the example:

```bash
python examples/parameter_study.py
```

Run tests:

```bash
python -m unittest discover -s tests
```

## What This Demonstrates

- translating a physical transport problem into a PDE
- implementing an explicit numerical scheme directly in Python
- checking a stability condition before simulation
- separating solver logic from the experiment/parameter study
- testing conservation/monotonicity-related expectations for a diffusion problem
- exploring sensitivity to thermal diffusivity and spatial resolution

## Why It Matters for Materials Modelling

Thermal diffusion is a continuum transport problem, but the workflow generalizes to many engineering simulation tasks: formulate the physics, select an appropriate discretization, implement it transparently, verify numerical behavior, and interpret the resulting field in physical terms.

This case study is deliberately modest. The next natural extensions are 2D diffusion, heterogeneous material properties, temperature-dependent conductivity, coupled thermo-mechanical stress, and inverse parameter estimation.

## Author

Vinay Kumar — M.Tech, Geomechanics for Mineral & Energy Resources, IIT Kharagpur.

This study complements existing work in numerical PDEs, CFD, reservoir engineering, geostatistics, computational geophysics, and scientific Python.
