"""
hamiltonian.py — Fu Hamiltonian eigenfunctions and k-space grids.

H = vF(ẑ×sigma)·k + λ(kx³ - 3kxky²)sigmaz

All functions are pure (no side effects).  Grids are module-level
constants built from params at import time — reimport or call
rebuild_grids() if you change V_K / LAMBDA.

Dependency: params
"""

import numpy as np
import matplotlib.pyplot as plt
import params

# ── Eigenenergy ────────────────────────────────────────────────────────────

def E_polar(k, theta, s=1):
    """Band energy in polar (k, θ) coordinates.  s = ±1 (upper/lower band)."""
    return s * np.sqrt(
        params.V_K**2 * k**2
        + params.LAMBDA**2 * k**6 * np.cos(3.0 * theta)**2
    )

def E_cartesian(kx, ky, s=1):
    """Band energy in Cartesian coordinates."""
    A = kx**3 - 3 * kx * ky**2
    return s * np.sqrt(params.V_X**2 * (kx**2 + ky**2) + params.LAMBDA**2 * A**2)

# ── Spin expectation values ⟨sigma_a⟩  (polar grid) ───────────────────────────

def sigma_x_polar(k, theta, s=1):
    E = E_polar(k, theta, s)
    return np.divide(-params.V_K * k * np.sin(theta), E,
                     out=np.zeros_like(E), where=np.abs(E) > 1e-14)

def sigma_y_polar(k, theta, s=1):
    E = E_polar(k, theta, s)
    return np.divide(params.V_K * k * np.cos(theta), E,
                     out=np.zeros_like(E), where=np.abs(E) > 1e-14)

def sigma_z_polar(k, theta, s=1):
    E = E_polar(k, theta, s)
    return np.divide(params.LAMBDA * k**3 * np.cos(3.0 * theta), E,
                     out=np.zeros_like(E), where=np.abs(E) > 1e-14)

# ── Fermi–Dirac distribution ───────────────────────────────────────────────

def f0(E, Ef, kBT):
    """Fermi–Dirac occupation.  kBT in eV (same units as E, Ef)."""
    return 1.0 / (1.0 + np.exp(np.clip((E - Ef) / kBT, -700, 700)))

# ── k-grids ────────────────────────────────────────────────────────────────

def build_coarse_grid():
    """Coarse (k, θ) mesh for CISP and Berry-BCD calculations."""
    th = np.linspace(0.0, 2.0 * np.pi, params.N_TH_COARSE)
    k  = np.linspace(1e-4, params.K_MAX_COARSE, params.N_K_COARSE)
    TH, KK = np.meshgrid(th, k)
    return TH, KK

def build_fine_grid():
    """Fine (k, θ) mesh for SBCQ (dk ≪ kT/v_F at 100 K)."""
    th = np.linspace(0.0, 2.0 * np.pi, params.N_TH_COARSE)  # same θ points
    k  = np.linspace(1e-5, params.K_MAX_FINE, params.N_K_FINE)
    TH, KK = np.meshgrid(th, k)
    return TH, KK

# Module-level grids (rebuilt by rebuild_grids() if params change)
TH_C, KK_C = build_coarse_grid()   # coarse: shape (N_K_COARSE, N_TH_COARSE)
TH_F, KK_F = build_fine_grid()     # fine:   shape (N_K_FINE,   N_TH_COARSE)

def rebuild_grids():
    """Call after changing params.N_K_* / params.K_MAX_* to refresh grids."""
    global TH_C, KK_C, TH_F, KK_F
    TH_C, KK_C = build_coarse_grid()
    TH_F, KK_F = build_fine_grid()

# ── Plot helpers ───────────────────────────────────────────────────────────

THETA_TICKS  = np.arange(0, 2 * np.pi + 1e-9, np.pi / 6)
THETA_LABELS = [
    '0', r'$\pi/6$', r'$\pi/3$', r'$\pi/2$', r'$2\pi/3$', r'$5\pi/6$',
    r'$\pi$', r'$7\pi/6$', r'$4\pi/3$', r'$3\pi/2$', r'$5\pi/3$',
    r'$11\pi/6$', r'$2\pi$',
]

def style_theta_axis(ax, xlabel=r"Drift angle $\theta$ (rad)"):
    """Apply standard θ-axis formatting to a matplotlib Axes."""
    ax.axhline(0, color="gray", lw=0.8)
    ax.set_xticks(THETA_TICKS)
    ax.set_xticklabels(THETA_LABELS)
    ax.set_xlabel(xlabel)
    ax.set_xlim(0, 2 * np.pi)
    ax.grid(alpha=0.25)

def _grid(val, ref):
    """Broadcast scalar or array val to shape of ref."""
    return np.broadcast_to(np.asarray(val, dtype=float), ref.shape).copy()