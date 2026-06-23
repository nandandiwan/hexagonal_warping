"""
params.py — all tunable constants for the TI transport calculations.

Edit this file to change material parameters, grid resolution, or
experiment settings.  No computation happens at import time.

Dependency: none (pure configuration).
"""

from pathlib import Path
import scipy.constants as scc

# ── Material parameters ────────────────────────────────────────────────────
# Fu Hamiltonian: H = vF(ẑ×sigma)·k + λ(kx³ - 3kxky²)sigmaz
#   V_K  [eV·Å]   — Dirac velocity  (v_F = V_K·e·1e-10/ħ ≈ 3.87×10⁵ m/s)
#   LAMBDA [eV·Å³] — hexagonal warping strength

V_K    = 2.55      # eV·Å
LAMBDA = 255       # eV·Å³
V_X    = V_K       # allow anisotropic extension: vx ≠ vy
V_Y    = V_K

# ── Computational resources ────────────────────────────────────────────────
N_WORKERS = 32     # parallel processes for Boltzmann/SBCQ batches

# ── Output paths ───────────────────────────────────────────────────────────
OUT_DIR = Path("plots")

def _make_dirs():
    """Create all output subdirectories."""
    for d in [OUT_DIR,
              OUT_DIR / "sbcq",
              OUT_DIR / "spin_current",
              OUT_DIR / "fisher"]:
        d.mkdir(exist_ok=True)

_make_dirs()

# ── k-grid parameters (coarse — CISP and Berry BCD) ───────────────────────
N_K_COARSE  = 1001
K_MAX_COARSE = 4.0         # Å⁻¹
N_TH_COARSE  = 301

# ── k-grid parameters (fine — SBCQ, needs dk < kT/v_F at 100 K ≈ 0.003 Å⁻¹)
N_K_FINE  = 3001
K_MAX_FINE = 0.6           # Å⁻¹
# fine grid shares the coarse theta grid

# ── Sweep resolution ───────────────────────────────────────────────────────
N_EF    = 150
N_TAU   = 100
N_THETA = 120

# ── Physical defaults ──────────────────────────────────────────────────────
DEFAULT_EF  = 0.1    # eV
DEFAULT_T   = 300    # K
DEFAULT_TAU = 1e-11  # s  (scattering time for drift shifts)
DEFAULT_THETA = 0
E_FIELD     = 1e6
THICKNESS = 8e-9
T_VALUES = [100, 200, 300, 400, 500]   # K — temperatures for sweeps

# ── Derived helpers ────────────────────────────────────────────────────────
def kBT_eV(T):
    """Thermal energy kT in eV."""
    return scc.Boltzmann * T / scc.elementary_charge

def drift_k(tau, theta, E_FIELD = E_FIELD):
    """k-space drift vector (Å⁻¹) for given τ, field angle, and E-field.

    dk = e·E·τ/ħ × 10⁻¹⁰  [Å⁻¹],  direction opposite to E-field.
    """
    import numpy as _np
    import scipy.constants as _sc
    mag = _sc.elementary_charge * E_FIELD * tau / _sc.hbar * 1e-10
    return -mag * _np.cos(theta), -mag * _np.sin(theta)

# ── Fischer torque (experiment parameters) ────────────────────────────────
FISHER_J_2D   = 100.0   # A/m   — 2D current density (typical STFMR)
FISHER_DEL_EX = 50.0    # meV   — exchange coupling at TI/FM interface
FISHER_M_S    = 8e5     # A/m   — saturation magnetisation (Permalloy)
FISHER_T_FM   = 8e-9    # m     — FM layer thickness

# Fermi velocity in SI (derived from V_K — recomputed if V_K changes)
def vF_SI():
    """Fermi velocity in m/s from current V_K."""
    import numpy as _np; import scipy.constants as _sc
    return V_K * _sc.elementary_charge * 1e-10 / _sc.hbar

# ── Spin current prefactors ────────────────────────────────────────────────
SC_E_FIELD = E_FIELD
SC_TAU     = DEFAULT_TAU