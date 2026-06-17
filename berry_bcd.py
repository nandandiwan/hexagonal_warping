"""
berry_bcd.py — Berry curvature, Berry Curvature Dipole (BCD), and nonlinear Hall.

Grids and sympy-derived functions are precomputed at import time.
Call rebuild() after changing params.V_K / params.LAMBDA.

Public API
----------
plot_berry_curvature_map()           — ΩΩ_z(kx, ky) false-colour plot
sweep_bcd_Ef_temperature()           — D_x, D_y vs E_F for each T
nonlinear_hall_response(...)         — 2nd-order Hall current vs E-field angle

Dependency: params, hamiltonian
"""

import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import sympy as sp
import matplotlib.pyplot as plt

import params
import hamiltonian as H

# ── Sympy Berry curvature (built once) ────────────────────────────────────

def _build_berry_fns():
    kx, ky, vx, vy, lam, s = sp.symbols('kx ky vx vy lam s', real=True)
    d      = sp.Matrix([-vy * ky, vx * kx, lam * (kx**3 - 3 * kx * ky**2)])
    dn     = sp.sqrt(d.dot(d))
    triple = d.dot(d.diff(kx).cross(d.diff(ky)))
    Omega  = s * triple / (2 * dn**3)
    E      = s * dn
    a      = (kx, ky, vx, vy, lam, s)
    return {
        'E':   sp.lambdify(a, E,                    'numpy'),
        'Om':  sp.lambdify(a, Omega,                'numpy'),
        'dOx': sp.lambdify(a, sp.diff(Omega, kx),  'numpy'),
        'dOy': sp.lambdify(a, sp.diff(Omega, ky),  'numpy'),
    }

_BERRY_FN = _build_berry_fns()


# ── Precomputed Berry grids ────────────────────────────────────────────────

def _build_bgrid():
    KX = H.KK_C * np.cos(H.TH_C)
    KY = H.KK_C * np.sin(H.TH_C)
    g  = {'KK': H.KK_C, 'k_ax': H.KK_C[:, 0], 'th_ax': H.TH_C[0, :]}
    for s in (1.0, -1.0):
        g[('E',   s)] = H._grid(_BERRY_FN['E']  (KX, KY, params.V_X, params.V_Y, params.LAMBDA, s), H.KK_C)
        g[('Om',  s)] = H._grid(_BERRY_FN['Om'] (KX, KY, params.V_X, params.V_Y, params.LAMBDA, s), H.KK_C)
        g[('dOx', s)] = H._grid(_BERRY_FN['dOx'](KX, KY, params.V_X, params.V_Y, params.LAMBDA, s), H.KK_C)
        g[('dOy', s)] = H._grid(_BERRY_FN['dOy'](KX, KY, params.V_X, params.V_Y, params.LAMBDA, s), H.KK_C)
    return g

_BGRID = _build_bgrid()


def rebuild():
    """Rebuild grids after changing params.V_K / params.LAMBDA."""
    global _BGRID
    H.rebuild_grids()
    _BGRID = _build_bgrid()


# ── 2D integral helper ─────────────────────────────────────────────────────

def _int2d(F):
    return (
        np.trapezoid(
            np.trapezoid(F, x=_BGRID['th_ax'], axis=1),
            x=_BGRID['k_ax'],
        ) / (2 * np.pi) ** 2
    )


# ── BCD worker ─────────────────────────────────────────────────────────────

def _compute_bcd(args):
    """
    Return (D_x, D_y, AHC) [Å²] for given (Ef [eV], kBT [eV]).

    Both bands s = ±1 are included.
    """
    Ef, kBT = args
    Dx = Dy = ahc = 0.0
    for s in (1.0, -1.0):
        occ  = H.f0(_BGRID[('E', s)], Ef, kBT)
        Dx  += _int2d(occ * _BGRID[('dOx', s)] * _BGRID['KK'])
        Dy  += _int2d(occ * _BGRID[('dOy', s)] * _BGRID['KK'])
        ahc += _int2d(occ * _BGRID[('Om',  s)] * _BGRID['KK'])
    return float(Dx), float(Dy), float(ahc)


def run_berry_batch(arg_list):
    """
    Parallelised BCD computation.

    Each element: (Ef, kBT)
    Returns: ndarray (N, 3) — D_x, D_y, AHC [Å²]
    """
    ctx = mp.get_context("fork")
    with ProcessPoolExecutor(max_workers=params.N_WORKERS, mp_context=ctx) as ex:
        return np.array(list(ex.map(_compute_bcd, arg_list)))


# ── Plots ──────────────────────────────────────────────────────────────────

def plot_berry_curvature_map(save=True):
    """False-colour map of Ω_z for the lower band."""
    KX   = H.KK_C * np.cos(H.TH_C)
    KY   = H.KK_C * np.sin(H.TH_C)
    Om   = _BGRID[('Om', -1.0)]
    vmax = float(np.nanpercentile(np.abs(Om), 99)) or 1.0

    fig, ax = plt.subplots(figsize=(6.4, 5.4))
    pc = ax.contourf(KX, KY, np.clip(Om, -vmax, vmax),
                     levels=np.linspace(-vmax, vmax, 41),
                     cmap="RdBu_r", extend="both")
    fig.colorbar(pc, ax=ax, label=r"$\Omega_z^{(-)}$")
    ax.set_xlabel(r"$k_x$ (Å⁻¹)"); ax.set_ylabel(r"$k_y$ (Å⁻¹)")
    ax.set_aspect("equal")
    ax.set_title(rf"Lower-band Berry curvature  ($\lambda={params.LAMBDA}$ eV·Å³)")
    fig.tight_layout()

    if save:
        fig.savefig(params.OUT_DIR / "berry_curvature_map.png", dpi=200)
        plt.close(fig)
    return fig


def sweep_bcd_Ef_temperature(Ef_range=(-0.5, 0.5), save=True):
    """
    BCD components D_x, D_y vs E_F for each T in params.T_VALUES.

    Returns
    -------
    Ef_vals : ndarray (N_EF,)
    data    : dict {T: ndarray(N_EF, 3)}   columns: D_x, D_y, AHC [Å²]
    """
    Ef_vals = np.linspace(*Ef_range, params.N_EF)
    data    = {}
    for T in params.T_VALUES:
        data[T] = run_berry_batch(
            [(Ef, params.kBT_eV(T)) for Ef in Ef_vals]
        )
        print(f"  T = {T} K done")

    if save:
        for comp, idx, fname in [
            (r"$D_x$ (Å²)", 0, "Dx_vs_Ef.png"),
            (r"$D_y$ (Å²)", 1, "Dy_vs_Ef.png"),
        ]:
            fig, ax = plt.subplots(figsize=(7, 5))
            for T, arr in data.items():
                ax.plot(Ef_vals, arr[:, idx], lw=2, label=f"T={T} K")
            ax.axhline(0, color="gray", lw=0.8)
            ax.set_xlabel(r"$E_F$ (eV)"); ax.set_ylabel(comp)
            ax.set_title(f"BCD {comp} vs $E_F$"); ax.grid(alpha=0.25); ax.legend()
            fig.tight_layout()
            fig.savefig(params.OUT_DIR / fname, dpi=200)
            plt.close(fig)

    return Ef_vals, data


def nonlinear_hall_response(Ef=None, T=None, tau=None, omega=0.0, save=True):
    """
    Second-order nonlinear Hall current vs E-field angle φ.

    Returns
    -------
    phi     : ndarray (361,)
    j_perp  : ndarray (361,)  — transverse current / (|C| E₀²)
    j_par   : ndarray (361,)  — longitudinal current / (|C| E₀²)
    """
    import scipy.constants as scc
    if Ef  is None: Ef  = params.DEFAULT_EF
    if T   is None: T   = params.DEFAULT_T
    if tau is None: tau = params.DEFAULT_TAU

    Dx, Dy, _ = _compute_bcd((Ef, params.kBT_eV(T)))
    e  = scc.elementary_charge
    C  = e**3 * tau / (2.0 * (1.0 + 1j * omega * tau))   # noqa (complex)

    phi    = np.linspace(0.0, 2 * np.pi, 361)
    Ex, Ey = np.cos(phi), np.sin(phi)
    DdotE  = Dx * Ex + Dy * Ey
    jx     = -Ey * DdotE
    jy     =  Ex * DdotE
    j_par  =  jx * Ex + jy * Ey
    j_perp = -jx * Ey + jy * Ex

    if save:
        fig, ax = plt.subplots(figsize=(7, 5))
        ax.plot(phi, j_perp, lw=2, color="crimson",   label=r"$j_\perp$")
        ax.plot(phi, j_par,  lw=2, ls="--", color="slategray", label=r"$j_\parallel\approx0$")
        ax.axhline(0, color="gray", lw=0.8); ax.set_xlim(0, 2 * np.pi)
        ax.set_xlabel(r"field angle $\phi$")
        ax.set_ylabel(r"current / ($|C|E_0^2$)")
        ax.set_title(rf"Nonlinear Hall  ($E_F$={Ef} eV, T={T} K)")
        ax.grid(alpha=0.25); ax.legend()
        fig.tight_layout()
        fig.savefig(params.OUT_DIR / "nonlinear_hall_vs_angle.png", dpi=200)
        plt.close(fig)

    return phi, j_perp, j_par