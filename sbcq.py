"""
sbcq.py — Second-order Berry Curvature (SBCQ) and Drude spin currents.

_compute_sbcq returns 23 values:
  [0:9]   Q tensor: (sigma_x, sigma_y, sigma_z) × (xx, xy, yy)   [Å²]
  [9]     SHC_z  = ∫ Ω̃_zz f₀                          [dim-less]
  [10:13] Σ_zx   sigma_z Drude, flow x: xx, xy, yy          [eV·Å]
  [13:16] Σ_zy   sigma_z Drude, flow y: xx, xy, yy          [eV·Å]
  [16]    SHC_y  = ∫ Ω̃_yz f₀  (≈ 0 by C₃ᵥ symmetry)
  [17:20] Σ_yx   sigma_y Drude, flow x: xx, xy, yy          [eV·Å]
  [20:23] Σ_yy   sigma_y Drude, flow y: xx, xy, yy          [eV·Å]

Spin current prefactors:
  J₁ = (e/2) × SHC × E                     [J/m]
  J₂ = ½(eτ/ħ)² × Σ × (e×10⁻¹⁰) × E²     [J/m]
  J₃ = (e/2)(eτ/ħ)² × Q × 10⁻²⁰ × E³     [J/m]

Public API
----------
run_sbcq_batch(arg_list)             — parallelised SBCQ computation
spin_currents(phi, res, E, tau)      — decompose results into J₁,J₂,J₃ [J/m]
print_spin_current_scale(...)        — diagnostic magnitudes
sweep_sbcq_Ef_temperature()          — Q tensors vs E_F
sweep_spin_current_vs_theta(...)     — J_z, J_y vs E-field angle
sweep_spin_tensors_vs_Ef(...)        — response tensors + currents vs E_F

Dependency: params, hamiltonian
"""

import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor
import warnings

import numpy as np
import sympy as sp
import matplotlib.pyplot as plt
import scipy.constants as scc

import params
import hamiltonian as H

# ── Output subdirectories ──────────────────────────────────────────────────
SBCQ_DIR = params.OUT_DIR / "sbcq"
SC_DIR   = params.OUT_DIR / "spin_current"

# ── Sympy functions (built once) ───────────────────────────────────────────

def _build_sbcq_fns():
    kx, ky, vx, vy, lam, s = sp.symbols('kx ky vx vy lam s', real=True)
    d  = sp.Matrix([-vy * ky, vx * kx, lam * (kx**3 - 3 * kx * ky**2)])
    dn = sp.sqrt(d.dot(d))
    Om = s * d.dot(d.diff(kx).cross(d.diff(ky))) / (2 * dn**3)
    E  = s * dn
    a  = (kx, ky, vx, vy, lam, s)
    fns = {key: sp.lambdify(a, expr, 'numpy') for key, expr in [
        ('E',   E),
        ('vx',  sp.diff(E, kx)),
        ('vy',  sp.diff(E, ky)),
        ('Exx', sp.diff(E, kx, 2)),
        ('Exy', sp.diff(sp.diff(E, kx), ky)),
        ('Eyy', sp.diff(E, ky, 2)),
    ]}
    for i, sa in enumerate('xyz'):
        fns[f'sbc_{sa}'] = sp.lambdify(a, (s * d[i] / dn) * Om, 'numpy')
    return fns


def _build_drude_vtx():
    """Drude spin-current vertices sigma_z and sigma_y (band-independent: s² = 1)."""
    kx, ky, vx, vy, lam = sp.symbols('kx ky vx vy lam', real=True)
    d  = sp.Matrix([-vy * ky, vx * kx, lam * (kx**3 - 3 * kx * ky**2)])
    dn = sp.sqrt(d.dot(d))
    a5 = (kx, ky, vx, vy, lam)
    return {
        'dvx':    sp.lambdify(a5, d[2] * sp.diff(dn, kx) / dn, 'numpy'),  # sigma_z, flow x
        'dvy':    sp.lambdify(a5, d[2] * sp.diff(dn, ky) / dn, 'numpy'),  # sigma_z, flow y
        'dy_dvx': sp.lambdify(a5, d[1] * sp.diff(dn, kx) / dn, 'numpy'),  # sigma_y, flow x
        'dy_dvy': sp.lambdify(a5, d[1] * sp.diff(dn, ky) / dn, 'numpy'),  # sigma_y, flow y
    }


_SBCQ_FN   = _build_sbcq_fns()
_DRUDE_VTX = _build_drude_vtx()


# ── Precomputed fine grids ─────────────────────────────────────────────────

def _build_sqgrid():
    KX = H.KK_F * np.cos(H.TH_F)
    KY = H.KK_F * np.sin(H.TH_F)
    g  = {}
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        for s in (1.0, -1.0):
            for key in ('E', 'vx', 'vy', 'Exx', 'Exy', 'Eyy',
                        'sbc_x', 'sbc_y', 'sbc_z'):
                g[(key, s)] = H._grid(
                    _SBCQ_FN[key](KX, KY, params.V_X, params.V_Y, params.LAMBDA, s),
                    H.KK_F,
                )
        for key in ('dvx', 'dvy', 'dy_dvx', 'dy_dvy'):
            g[key] = H._grid(
                _DRUDE_VTX[key](KX, KY, params.V_X, params.V_Y, params.LAMBDA),
                H.KK_F,
            )
    return g


_SQGRID = _build_sqgrid()
_SQ_TH  = H.TH_F[0, :]
_SQ_K   = H.KK_F[:, 0]


def rebuild():
    """Rebuild grids after changing params.V_K / params.LAMBDA."""
    global _SQGRID, _SQ_TH, _SQ_K
    H.rebuild_grids()
    _SQGRID = _build_sqgrid()
    _SQ_TH  = H.TH_F[0, :]
    _SQ_K   = H.KK_F[:, 0]


def _sq_int2d(F):
    return np.trapezoid(np.trapezoid(F, x=_SQ_TH, axis=1), x=_SQ_K) / (2 * np.pi) ** 2


# ── SBCQ worker ────────────────────────────────────────────────────────────

def _compute_sbcq(args):
    """
    Return 23 SBCQ/SHC/Drude integrals for (Ef, kBT).
    Both bands s = ±1 are included.
    """
    Ef, kBT = args
    res = np.zeros(23)
    for s in (1.0, -1.0):
        occ  = H.f0(_SQGRID[('E', s)], Ef, kBT)
        fp   = -occ * (1 - occ) / kBT
        fpp  =  occ * (1 - occ) * (1 - 2 * occ) / kBT**2
        vx_  = _SQGRID[('vx', s)]
        vy_  = _SQGRID[('vy', s)]
        d2f  = [
            fpp * vx_**2    + fp * _SQGRID[('Exx', s)],
            fpp * vx_ * vy_ + fp * _SQGRID[('Exy', s)],
            fpp * vy_**2    + fp * _SQGRID[('Eyy', s)],
        ]
        for i, sa in enumerate('xyz'):
            sbc = _SQGRID[(f'sbc_{sa}', s)]
            for j, d2 in enumerate(d2f):
                res[3 * i + j] += _sq_int2d(sbc * d2 * H.KK_F)

        res[9]  += _sq_int2d(_SQGRID[('sbc_z', s)] * occ * H.KK_F)   # SHC_z
        res[16] += _sq_int2d(_SQGRID[('sbc_y', s)] * occ * H.KK_F)   # SHC_y (≈0)

        for ai, dvk in enumerate(('dvx', 'dvy', 'dy_dvx', 'dy_dvy')):
            for j, d2 in enumerate(d2f):
                res[10 + ai * 3 + j] += _sq_int2d(_SQGRID[dvk] * d2 * H.KK_F)

    return tuple(res)


def run_sbcq_batch(arg_list):
    """
    Parallelised SBCQ computation.

    Each element: (Ef, kBT)
    Returns: ndarray (N, 23)
    """
    ctx = mp.get_context("fork")
    with ProcessPoolExecutor(max_workers=params.N_WORKERS, mp_context=ctx) as ex:
        return np.array(list(ex.map(_compute_sbcq, arg_list)))


# ── Spin current decomposition ─────────────────────────────────────────────

def spin_currents(phi, res, E_field=None, tau=None):
    """
    Decompose SBCQ result into J₁, J₂, J₃ spin current components [J/m].

    Parameters
    ----------
    phi     : array  — E-field angle(s) [rad]
    res     : array shape (23,) — output of _compute_sbcq
    E_field : float [V/m],  default params.SC_E_FIELD
    tau     : float [s],    default params.SC_TAU

    Returns
    -------
    dict with keys J1z, J2z, J3z, Jtot_z, J1y, J2y, J3y, Jtot_y
    Each value: ndarray shape (2, len(phi)) — x- and y-flow components [J/m]
    """
    if E_field is None: E_field = params.SC_E_FIELD
    if tau     is None: tau     = params.SC_TAU

    e, hbar  = scc.elementary_charge, scc.hbar
    Ex, Ey   = np.cos(phi), np.sin(phi)
    eqh2     = (e * tau / hbar) ** 2
    pref3    = e / 2 * eqh2 * 1e-20 * E_field**3
    hall     = np.array([Ey, -Ex])    # ẑ × E direction

    SHC_z = res[9];  SHC_y = res[16]
    Qz_xx, Qz_xy, Qz_yy = res[6], res[7], res[8]
    Qy_xx, Qy_xy, Qy_yy = res[3], res[4], res[5]
    Szx_xx, Szx_xy, Szx_yy = res[10], res[11], res[12]
    Szy_xx, Szy_xy, Szy_yy = res[13], res[14], res[15]
    Syx_xx, Syx_xy, Syx_yy = res[17], res[18], res[19]
    Syy_xx, Syy_xy, Syy_yy = res[20], res[21], res[22]

    # sigma_z
    J1z = (e / 2) * SHC_z * hall * E_field

    J2z_x = Szx_xx * Ex**2 + 2 * Szx_xy * Ex * Ey + Szx_yy * Ey**2
    J2z_y = Szy_xx * Ex**2 + 2 * Szy_xy * Ex * Ey + Szy_yy * Ey**2
    J2z   = (0.5 * eqh2 * e * 1e-10 * E_field**2) * np.array([J2z_x, J2z_y])

    Qcon_z = Qz_xx * Ex**2 + 2 * Qz_xy * Ex * Ey + Qz_yy * Ey**2
    J3z    = pref3 * Qcon_z * hall

    # sigma_y
    J1y = (e / 2) * SHC_y * hall * E_field   # ≈ 0 by C₃ᵥ

    J2y_x = Syx_xx * Ex**2 + 2 * Syx_xy * Ex * Ey + Syx_yy * Ey**2
    J2y_y = Syy_xx * Ex**2 + 2 * Syy_xy * Ex * Ey + Syy_yy * Ey**2
    J2y   = (0.5 * eqh2 * e * 1e-10 * E_field**2) * np.array([J2y_x, J2y_y])

    Qcon_y = Qy_xx * Ex**2 + 2 * Qy_xy * Ex * Ey + Qy_yy * Ey**2
    J3y    = pref3 * Qcon_y * hall

    return {
        'J1z': J1z, 'J2z': J2z, 'J3z': J3z, 'Jtot_z': J1z + J2z + J3z,
        'J1y': J1y, 'J2y': J2y, 'J3y': J3y, 'Jtot_y': J1y + J2y + J3y,
    }


def print_spin_current_scale(Ef=None, T=None):
    """Print diagnostic magnitude table for each spin current order."""
    if Ef is None: Ef = params.DEFAULT_EF
    if T  is None: T  = params.DEFAULT_T
    e, hbar = scc.elementary_charge, scc.hbar
    res  = np.array(_compute_sbcq((Ef, params.kBT_eV(T))))
    eqh  = e * params.SC_TAU / hbar
    kF   = Ef / params.V_K
    k_dr = eqh * params.SC_E_FIELD * 1e-10
    J1z  = (e / 2) * abs(res[9])  * params.SC_E_FIELD
    J2z  = 0.5 * eqh**2 * abs(res[10]) * e * 1e-10 * params.SC_E_FIELD**2
    J3z  = (e / 2) * eqh**2 * abs(res[6])  * 1e-20 * params.SC_E_FIELD**3
    J1y  = (e / 2) * abs(res[16]) * params.SC_E_FIELD
    J2y  = 0.5 * eqh**2 * abs(res[17]) * e * 1e-10 * params.SC_E_FIELD**2
    J3y  = (e / 2) * eqh**2 * abs(res[3])  * 1e-20 * params.SC_E_FIELD**3
    print(f"\n── Spin current scale  (Ef={Ef} eV, T={T} K, "
          f"E={params.SC_E_FIELD:.0e} V/m, τ={params.SC_TAU:.0e} s) ──")
    print(f"  k_F = {kF:.4f} Å⁻¹,  k_drift = {k_dr:.4f} Å⁻¹")
    print(f"  SHC_y = {res[16]:.4e}  (should be ≈ 0 by C₃ᵥ)")
    print(f"  sigma_z:  J1z = {J1z:.3e}  J2z = {J2z:.3e}  J3z = {J3z:.3e}  J/m")
    print(f"  sigma_y:  J1y = {J1y:.3e}  J2y = {J2y:.3e}  J3y = {J3y:.3e}  J/m")


# ── Sweep functions ────────────────────────────────────────────────────────

def sweep_sbcq_Ef_temperature(Ef_range=(-0.5, 0.5), save=True):
    """
    SBCQ Q tensors vs E_F for each T.

    Returns
    -------
    Ef_vals : ndarray (N_EF,)
    data    : dict {T: ndarray(N_EF, 23)}
    """
    Ef_vals = np.linspace(*Ef_range, params.N_EF)
    data = {}
    for T in params.T_VALUES:
        data[T] = run_sbcq_batch([(Ef, params.kBT_eV(T)) for Ef in Ef_vals])
        print(f"  T = {T} K done")

    if save:
        for i, (sa, slabel) in enumerate([
            ('x', r'$\sigma_x$'), ('y', r'$\sigma_y$'), ('z', r'$\sigma_z$')
        ]):
            fig, axes = plt.subplots(1, 3, figsize=(12, 4))
            for j, (ax, comp) in enumerate(zip(axes, ['xx', 'xy', 'yy'])):
                for T, arr in data.items():
                    ax.plot(Ef_vals, arr[:, 3 * i + j], lw=2, label=f"T={T} K")
                ax.axhline(0, color='gray', lw=0.8)
                ax.set_xlabel(r"$E_F$ (eV)")
                ax.set_ylabel(rf"$Q_{{{sa},{comp}}}$ (Å²)")
                ax.set_title(rf"$Q_{{{sa},{comp}}}$")
                ax.grid(alpha=0.25); ax.legend(fontsize=7)
            fig.suptitle(rf"SBCQ — {slabel}", fontsize=13)
            fig.tight_layout()
            fig.savefig(SBCQ_DIR / f"sbcq_{sa}.png", dpi=200)
            plt.close(fig)

    return Ef_vals, data


def sweep_spin_current_vs_theta(Ef=None, E_field=None, tau=None, save=True):
    """
    Spin current components vs E-field angle φ.

    Returns
    -------
    phi   : ndarray (N_THETA,)
    all_J : dict {T: spin_current_dict}
    """
    if Ef      is None: Ef      = params.DEFAULT_EF
    if E_field is None: E_field = params.SC_E_FIELD
    if tau     is None: tau     = params.SC_TAU

    phi    = np.linspace(0, 2 * np.pi, params.N_THETA)
    colors = plt.cm.plasma(np.linspace(0.1, 0.9, len(params.T_VALUES)))
    k_dr   = (scc.elementary_charge * tau / scc.hbar) * E_field * 1e-10

    all_J = {}
    for T in params.T_VALUES:
        res    = np.array(_compute_sbcq((Ef, params.kBT_eV(T))))
        all_J[T] = spin_currents(phi, res, E_field, tau)

    if save:
        def _fig(keys, titles_x, titles_y, fname, sup):
            fig, axes = plt.subplots(2, 4, figsize=(16, 7), sharex=True)
            for c, (tx, ty) in enumerate(zip(titles_x, titles_y)):
                axes[0, c].set_title(tx, fontsize=9)
                axes[1, c].set_title(ty, fontsize=9)
            for T, col in zip(params.T_VALUES, colors):
                for ci, key in enumerate(keys):
                    axes[0, ci].plot(phi, all_J[T][key][0] * 1e15, color=col, lw=1.8,
                                     label=f"T={T} K")
                    axes[1, ci].plot(phi, all_J[T][key][1] * 1e15, color=col, lw=1.8)
            for ax in axes.flat:
                H.style_theta_axis(ax); ax.set_ylabel(r"$J$ (fJ/m)")
            axes[0, 3].legend(fontsize=7, loc='upper right')
            fig.suptitle(sup, fontsize=10)
            fig.tight_layout(); fig.savefig(SC_DIR / fname, dpi=200); plt.close(fig)

        _fig(
            ['J1z', 'J2z', 'J3z', 'Jtot_z'],
            [r"$J^{(1)}_{z,x}$ (anom)", r"$J^{(2)}_{z,x}$ (Drude)",
             r"$J^{(3)}_{z,x}$ (SBCQ)",  r"$J^{tot}_{z,x}$"],
            [r"$J^{(1)}_{z,y}$", r"$J^{(2)}_{z,y}$",
             r"$J^{(3)}_{z,y}$",  r"$J^{tot}_{z,y}$"],
            "jz_vs_theta.png",
            rf"$\sigma_z$ spin current vs field angle  ($E_F$={Ef} eV, "
            rf"$E$={E_field:.0e} V/m, $\tau$={tau:.0e} s, "
            rf"$k_{{drift}}$={k_dr:.4f} Å$^{{-1}}$)",
        )
        _fig(
            ['J1y', 'J2y', 'J3y', 'Jtot_y'],
            [r"$J^{(1)}_{y,x}$ (anom ≈0)", r"$J^{(2)}_{y,x}$ (Drude)",
             r"$J^{(3)}_{y,x}$ (SBCQ)",     r"$J^{tot}_{y,x}$"],
            [r"$J^{(1)}_{y,y}$", r"$J^{(2)}_{y,y}$",
             r"$J^{(3)}_{y,y}$",  r"$J^{tot}_{y,y}$"],
            "jy_vs_theta.png",
            rf"$\sigma_y$ spin current vs field angle  "
            rf"($E_F$={Ef} eV, $E$={E_field:.0e} V/m, $\tau$={tau:.0e} s)",
        )

        # Clover locus + ratio
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        ax_z, ax_ratio, ax_y = axes
        for T, col in zip(params.T_VALUES, colors):
            J = all_J[T]
            Jz, Jy = J['Jtot_z'], J['Jtot_y']
            ax_z.plot(Jz[0] * 1e15, Jz[1] * 1e15, color=col, lw=2, label=f"T={T} K")
            ax_y.plot(Jy[0] * 1e15, Jy[1] * 1e15, color=col, lw=2, label=f"T={T} K")
            mag_z = np.hypot(Jz[0], Jz[1])
            mag_y = np.hypot(Jy[0], Jy[1]) + 1e-100
            ax_ratio.plot(phi / np.pi, mag_z / mag_y, color=col, lw=2)
        for ax in [ax_z, ax_y]:
            ax.axhline(0, color='gray', lw=0.5); ax.axvline(0, color='gray', lw=0.5)
            ax.set_aspect('equal'); ax.grid(alpha=0.2)
            ax.set_xlabel(r"$J_x$ (fJ/m)"); ax.set_ylabel(r"$J_y$ (fJ/m)")
        ax_z.set_title(r"$J^{tot}_z$ locus"); ax_z.legend(fontsize=7)
        ax_y.set_title(r"$J^{tot}_y$ locus"); ax_y.legend(fontsize=7)
        ax_ratio.axhline(1, color='k', lw=1, ls='--')
        ax_ratio.set_xlabel(r"$\phi$ (rad / $\pi$)")
        ax_ratio.set_ylabel(r"$|J^{tot}_z| / |J^{tot}_y|$")
        ax_ratio.set_title("Magnitude ratio"); ax_ratio.grid(alpha=0.25)
        fig.suptitle(rf"Spin current locus  ($E_F$={Ef} eV, $E$={E_field:.0e} V/m)", fontsize=10)
        fig.tight_layout(); fig.savefig(SC_DIR / "jz_clover.png", dpi=200); plt.close(fig)

    return phi, all_J


def sweep_spin_tensors_vs_Ef(Ef_range=(-0.5, 0.5), E_field=None, tau=None, save=True):
    """
    Response tensors and spin currents vs E_F.

    Returns
    -------
    Ef_vals : ndarray (N_EF,)
    data    : dict {T: ndarray(N_EF, 23)}
    """
    if E_field is None: E_field = params.SC_E_FIELD
    if tau     is None: tau     = params.SC_TAU

    Ef_vals = np.linspace(*Ef_range, params.N_EF)
    data    = {}
    for T in params.T_VALUES:
        data[T] = run_sbcq_batch([(Ef, params.kBT_eV(T)) for Ef in Ef_vals])
        print(f"  T = {T} K done")

    if save:
        # Response tensors (material properties, no E/τ dependence)
        fig, axes = plt.subplots(1, 5, figsize=(18, 4))
        specs = [
            ("SHC_z  (1st sigma_z)",               9),
            (r"$Q_{z,xx}$ [Å²]  (3rd sigma_z)",    6),
            (r"$Q_{y,xx}$ [Å²]  (3rd sigma_y)",    3),
            (r"$\Sigma_{zx,xx}$ [eV·Å] (2nd sigma_z)", 10),
            (r"$\Sigma_{yx,xx}$ [eV·Å] (2nd sigma_y)", 17),
        ]
        for ax, (lbl, idx) in zip(axes, specs):
            for T, arr in data.items():
                ax.plot(Ef_vals, arr[:, idx], lw=2, label=f"T={T} K")
            ax.axhline(0, color='gray', lw=0.8); ax.set_xlabel(r"$E_F$ (eV)")
            ax.set_title(lbl, fontsize=8); ax.grid(alpha=0.25); ax.legend(fontsize=6)
        fig.suptitle(r"Response tensors vs $E_F$  (no $E$, $\tau$ dependence)", fontsize=11)
        fig.tight_layout(); fig.savefig(SC_DIR / "spin_tensors_vs_Ef.png", dpi=200)
        plt.close(fig)

        # Spin currents vs Ef at fixed φ = π/6
        phi = np.array([np.pi / 6])
        for spin, keys, fname, suptitle in [
            ('z', ['J1z', 'J2z', 'J3z', 'Jtot_z'], "jz_mag_vs_Ef.png",
             rf"$\sigma_z$ currents vs $E_F$  ($E$={E_field:.0e} V/m, $\tau$={tau:.0e} s, $\phi$=π/6)"),
            ('y', ['J1y', 'J2y', 'J3y', 'Jtot_y'], "jy_vs_Ef.png",
             rf"$\sigma_y$ currents vs $E_F$  ($E$={E_field:.0e} V/m, $\tau$={tau:.0e} s, $\phi$=π/6)"),
        ]:
            lbls = [rf"$J^{{(1)}}_{{{spin},x}}$", rf"$J^{{(2)}}_{{{spin},x}}$",
                    rf"$J^{{(3)}}_{{{spin},x}}$",  rf"$J^{{tot}}_{{{spin},x}}$"]
            fig, axes = plt.subplots(1, 4, figsize=(16, 4))
            for ax, lbl, key in zip(axes, lbls, keys):
                for T, arr in data.items():
                    vals = [spin_currents(phi, r, E_field, tau)[key][0, 0] * 1e15
                            for r in arr]
                    ax.plot(Ef_vals, vals, lw=2, label=f"T={T} K")
                ax.axhline(0, color='gray', lw=0.8); ax.set_xlabel(r"$E_F$ (eV)")
                ax.set_ylabel(lbl + " (fJ/m)"); ax.set_title(lbl, fontsize=8)
                ax.grid(alpha=0.25); ax.legend(fontsize=7)
            fig.suptitle(suptitle, fontsize=10)
            fig.tight_layout(); fig.savefig(SC_DIR / fname, dpi=200); plt.close(fig)

    return Ef_vals, data