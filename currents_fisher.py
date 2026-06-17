"""
currents_fisher.py — Physical 2D current density from the warped TI Hamiltonian.

Velocity operator from H = d·σ:

    ħ vx = ∂H/∂kx = VK σy + λ(3kx² - 3ky²) σz
    ħ vy = ∂H/∂ky = -VK σx - 6λ kx·ky σz

So the current decomposes into two contributions:

    jx = jx_y  +  jx_z
       = −P·VK·⟨σy⟩  −  P·Λ·Wc2

    jy = jy_x  +  jy_z
       = +P·VK·⟨σx⟩  +  P·Λ·Ws2

where  P = e²·10¹⁰/ħ  [A/m per eV·Å²  =  A/m per Å⁻²]

New warping integrals (both [Å⁻⁴] in code units):

    Wc2 = ∫ 3k²·cos(2θ)·σz·f_shifted  k dk dθ/(2π)²
    Ws2 = ∫ 3k²·sin(2θ)·σz·f_shifted  k dk dθ/(2π)²

The equilibrium contributions to Wc2 and Ws2 vanish by symmetry
(cos(2θ)·cos(3θ) and sin(2θ)·cos(3θ) integrate to zero over [0,2π]).

Unit derivation (for jx_y as example):
    jx_y [A/m] = −(e·VK_SI/ħ)·⟨σy⟩_SI
               = −(e²·VK·10¹⁰/ħ)·⟨σy⟩_code     [code: Å⁻², ×10²⁰ → m⁻²]
    Numerically: pref_y = 6.19×10⁶ A/m per Å⁻²

    jx_z [A/m] = −(e·Λ_SI/ħ)·Wc2_SI
               = −(e²·Λ·10¹⁰/ħ)·Wc2_code        [code: Å⁻⁴, ×10⁴⁰ → m⁻⁴]
    Numerically: pref_z = 6.19×10⁸ A/m per Å⁻⁴
    Ratio pref_z/pref_y = Λ/VK = 100 Å²

Plots generated
---------------
  currents_01_jx_vs_Ef.png     — 3-panel jx decomposition vs E_F
  currents_02_jx_vs_theta.png  — 3-panel jx decomposition vs field direction θ
  currents_03_loops.png        — 2D (jx, jy) loop as θ varies, for 3 E_F values

Dependency: params, hamiltonian
"""

import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import matplotlib.pyplot as plt
import scipy.constants as scc

import params
import hamiltonian as H

# ── Output directory ───────────────────────────────────────────────────────
CURR_DIR = params.OUT_DIR / "currents"
CURR_DIR.mkdir(exist_ok=True)

# ── Precomputed angular weights ────────────────────────────────────────────
# These are evaluated on the coarse grid (same k-points as CISP)
_COS2TH = np.cos(2.0 * H.TH_C)   # cos(2θ), shape (N_K, N_TH)
_SIN2TH = np.sin(2.0 * H.TH_C)   # sin(2θ)

# ── Physical prefactors ────────────────────────────────────────────────────
def _prefs():
    e, hbar = scc.elementary_charge, scc.hbar
    # pref_y [A/m per Å⁻²]:  P·VK = e²·VK·10¹⁰/ħ
    pref_y = e**2 * params.V_K  * 1e10 / hbar
    # pref_z [A/m per Å⁻⁴]:  P·Λ  = e²·Λ·10¹⁰/ħ
    pref_z = e**2 * params.LAMBDA * 1e10 / hbar
    return pref_y, pref_z

def currents_from_integrals(sx, sy, W_c2, W_s2):
    """
    Convert code integrals [Å⁻²] and [Å⁻⁴] to physical current densities [A/m].

    Returns dict with keys:
      jx_y   — Dirac contribution to jx   [A/m]
      jx_z   — warping contribution to jx  [A/m]
      jx     — total jx                   [A/m]
      jy_x   — Dirac contribution to jy   [A/m]
      jy_z   — warping contribution to jy  [A/m]
      jy     — total jy                   [A/m]
    """
    pref_y, pref_z = _prefs()
    jx_y = -pref_y * sy
    jx_z = -pref_z * W_c2
    jy_x = +pref_y * sx
    jy_z = +pref_z * W_s2
    return {
        'jx_y': jx_y, 'jx_z': jx_z, 'jx': jx_y + jx_z,
        'jy_x': jy_x, 'jy_z': jy_z, 'jy': jy_x + jy_z,
    }

# ── Multiprocessing worker ─────────────────────────────────────────────────
_W = {}

def _init_worker_j(KK, TH, cos2th, sin2th):
    _W.update({
        'KK': KK, 'TH': TH, 'cos2th': cos2th, 'sin2th': sin2th,
        'KX': KK * np.cos(TH), 'KY': KK * np.sin(TH),
        'k_ax': KK[:, 0], 'th_ax': TH[0, :],
    })

def _compute_j(args):
    """
    Return (sx, sy, sz, Wc2, Ws2) [all in Å⁻² or Å⁻⁴] for the shifted distribution.

    Both bands s = ±1 are summed.
    """
    Ef, kBT, kdx, kdy = args
    kx_s = _W['KX'] - kdx
    ky_s = _W['KY'] - kdy
    sx = sy = sz = Wc2 = Ws2 = 0.0
    KK, TH = _W['KK'], _W['TH']
    k_ax, th_ax = _W['k_ax'], _W['th_ax']

    for s in [1]:
        occ = H.f0(H.E_cartesian(kx_s, ky_s, s), Ef, kBT)

        def _int2d(F):
            return np.trapezoid(
                np.trapezoid(F * occ * KK, x=th_ax, axis=1), x=k_ax
            ) / (2 * np.pi)**2

        sz_vals = H.sigma_z_polar(KK, TH, s)

        sx  += _int2d(H.sigma_x_polar(KK, TH, s))
        sy  += _int2d(H.sigma_y_polar(KK, TH, s))
        sz  += _int2d(sz_vals)
        Wc2 += _int2d(3 * KK**2 * _W['cos2th'] * sz_vals)
        Ws2 += _int2d(3 * KK**2 * _W['sin2th'] * sz_vals)

    return float(sx), float(sy), float(sz), float(Wc2), float(Ws2)


def run_batch_j(arg_list):
    """
    Parallelised current integral computation.

    Each element: (Ef, kBT, kdx, kdy)
    Returns: ndarray shape (N, 5) — columns: sx, sy, sz, Wc2, Ws2
    """
    ctx = mp.get_context("fork")
    with ProcessPoolExecutor(
        max_workers=params.N_WORKERS, mp_context=ctx,
        initializer=_init_worker_j,
        initargs=(H.KK_C, H.TH_C, _COS2TH, _SIN2TH),
    ) as ex:
        return np.array(list(ex.map(_compute_j, arg_list)))


# ── Sweep helpers ──────────────────────────────────────────────────────────

def _batch_to_currents(batch):
    """Convert ndarray (N, 5) → dict of current arrays, each shape (N,)."""
    sx, sy, sz, Wc2, Ws2 = batch.T
    pref_y, pref_z = _prefs()
    return {
        'jx_y': -pref_y * sy,
        'jx_z': -pref_z * Wc2,
        'jx':   -pref_y * sy - pref_z * Wc2,
        'jy_x': +pref_y * sx,
        'jy_z': +pref_z * Ws2,
        'jy':   +pref_y * sx + pref_z * Ws2,
        'sz':    sz,     # spin density [Å⁻²] — kept for diagnostics
    }


def _pref_note():
    """Annotation string showing prefactor values."""
    py, pz = _prefs()
    return (rf"$P_y = {py:.2e}$ A/m per Å$^{{-2}}$,  "
            rf"$P_z = {pz:.2e}$ A/m per Å$^{{-4}}$")


# ── Plot 1: jx decomposition vs E_F ───────────────────────────────────────

def sweep_jx_vs_Ef(Ef_range=(-0.5, 0.5), tau=None, save=True):
    """
    Three-panel plot: jx^(y), jx^(z), jx_total  vs E_F.
    Current is for E-field in x̂ direction.

    Returns
    -------
    Ef_vals : ndarray (N_EF,)
    data    : dict {T: current_dict}
    """
    if tau is None: tau = params.DEFAULT_TAU
    Ef_vals = np.linspace(*Ef_range, params.N_EF)
    kdx, kdy = params.drift_k(tau, 0.0)   # E-field along x̂

    E = params.E_FIELD
    colors   = plt.cm.plasma(np.linspace(0.1, 0.9, len(params.T_VALUES)))

    print("  currents: jx vs Ef …")
    data = {}
    for T in params.T_VALUES:
        batch = run_batch_j([(Ef, params.kBT_eV(T), kdx, kdy) for Ef in Ef_vals])
        data[T] = _batch_to_currents(batch)
        print(f"    T = {T} K done")

    if save:
        fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), sharey=False)
        titles = [
            r"$\sigma_x^{(y)}$   (Dirac, pure iSGE)",
            r"$\sigma_x^{(z)}$   (warping correction)",
            r"$\sigma_x = \sigma_x^{(y)} + \sigma_x^{(z)}$  (total)",
        ]
        keys = ['jx_y', 'jx_z', 'jx']
        for ax, title, key in zip(axes, titles, keys):
            for T, col in zip(params.T_VALUES, colors):
                ax.plot(Ef_vals, data[T][key]/E / params.THICKNESS, color=col, lw=2, label=f"T={T} K")
            ax.axhline(0, color='gray', lw=0.8)
            ax.set_xlabel(r"$E_F$ (eV)")
            ax.set_ylabel(r"$\sigma_x$ (S/m)")
            ax.set_title(title, fontsize=9)
            ax.grid(alpha=0.25)
        axes[0].legend(fontsize=7)

        fig.suptitle(
            rf"2D conductivity: E-field along $\hat{{x}}$,  "
            rf"$E={params.E_FIELD:.0e}$ V/m,  $\tau={params.DEFAULT_TAU:.0e}$ s",
            fontsize=9,
        )
        fig.tight_layout()
        fig.savefig(CURR_DIR / "conductivity_01_jx_vs_Ef.png", dpi=200)
        plt.close(fig)
        print("  saved conductivity_01_jx_vs_Ef.png")

    return Ef_vals, data


# ── Plot 2: jx decomposition vs field direction θ ─────────────────────────

def sweep_jx_vs_theta(Ef=None, T=None, tau=None, save=True):
    """
    Three-panel plot: jx^(y), jx^(z), jx_total  vs E-field angle θ.

    Returns
    -------
    theta_vals : ndarray (N_THETA,)
    J          : current_dict  (each array shape N_THETA)
    """
    if Ef  is None: Ef  = params.DEFAULT_EF
    if T   is None: T   = params.DEFAULT_T
    if tau is None: tau = params.DEFAULT_TAU

    theta_vals = np.linspace(0, 2 * np.pi, params.N_THETA)
    kBT        = params.kBT_eV(T)

    print("  currents: jx vs theta …")
    batch = run_batch_j(
        [(Ef, kBT, *params.drift_k(tau, th)) for th in theta_vals]
    )
    J = _batch_to_currents(batch)

    if save:
        fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
        panels = [
            (r"$j_x^{(y)}$",     'jx_y', 'royalblue'),
            (r"$j_x^{(z)}$",                'jx_z', 'crimson'),
            (r"$j_x$ total",                                      'jx',   'darkorange'),
        ]
        for ax, (title, key, color) in zip(axes, panels):
            ax.plot(theta_vals, J[key], color=color, lw=2)
            ax.axhline(0, color='gray', lw=0.8)
            ax.set_ylabel(r"$j_x$ (A/m)")
            ax.set_title(title, fontsize=9)
            H.style_theta_axis(ax)

        fig.suptitle(
            rf"$j_x$ vs E-field angle:  $E_F={Ef}$ eV,  T={T} K,"
            rf"  $E={params.E_FIELD:.0e}$ V/m,  $\tau={tau:.0e}$ s",
            fontsize=10,
        )
        fig.tight_layout()
        fig.savefig(CURR_DIR / "currents_02_jx_vs_theta.png", dpi=200)
        plt.close(fig)
        print("  saved currents_02_jx_vs_theta.png")

    return theta_vals, J


# ── Plot 3: 2D current loops (jx, jy) for several E_F values ──────────────

def plot_current_loops(
    Ef_list=(-0.2, 0.0, 0.2),
    T=None,
    tau=None,
    save=True,
):
    """
    Parametric plot of (jx(θ), jy(θ)) as θ ∈ [0, 2π] varies,
    for each E_F in Ef_list.

    For an isotropic material the loop is a perfect circle.
    Hexagonal warping deforms it into a shape with C₃ᵥ symmetry.
    Marks θ = 0 with a dot on each curve.

    Returns
    -------
    loops : dict {Ef: {'jx': ..., 'jy': ...}}
    """
    if T   is None: T   = params.DEFAULT_T
    if tau is None: tau = params.DEFAULT_TAU

    theta_vals = np.linspace(0, 2 * np.pi, params.N_THETA + 1)  # closed loop
    kBT        = params.kBT_eV(T)

    print("  currents: 2D loops …")
    loop_colors = plt.cm.coolwarm(np.linspace(0.1, 0.9, len(Ef_list)))
    loops = {}
    for Ef in Ef_list:
        batch = run_batch_j(
            [(Ef, kBT, *params.drift_k(tau, th)) for th in theta_vals]
        )
        J = _batch_to_currents(batch)
        loops[Ef] = J
        print(f"    Ef = {Ef:+.2f} eV done")

    if save:
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        ax_full, ax_y, ax_z = axes

        for (Ef, col) in zip(Ef_list, loop_colors):
            J    = loops[Ef]
            jx   = J['jx'];   jy  = J['jy']
            jx_y = J['jx_y']; jy_x = J['jy_x']
            jx_z = J['jx_z']; jy_z = J['jy_z']

            label = rf"$E_F={Ef:+.1f}$ eV"

            # Full current loop
            ax_full.plot(jx, jy, color=col, lw=2, label=label)
            ax_full.scatter(jx[0], jy[0], color=col, s=60, zorder=5,
                            marker='o', edgecolors='k', linewidths=0.8)

            # Dirac-only loop (jx_y, jy_x)
            ax_y.plot(jx_y, jy_x, color=col, lw=2, label=label)
            ax_y.scatter(jx_y[0], jy_x[0], color=col, s=60, zorder=5,
                         marker='o', edgecolors='k', linewidths=0.8)

            # Warping-only loop (jx_z, jy_z)
            ax_z.plot(jx_z, jy_z, color=col, lw=2, label=label)
            ax_z.scatter(jx_z[0], jy_z[0], color=col, s=60, zorder=5,
                         marker='o', edgecolors='k', linewidths=0.8)

        for ax in axes:
            ax.axhline(0, color='gray', lw=0.5)
            ax.axvline(0, color='gray', lw=0.5)
            ax.set_aspect('equal')
            ax.set_xlabel(r"$j_x$ (A/m)")
            ax.set_ylabel(r"$j_y$ (A/m)")
            ax.grid(alpha=0.2)
            ax.legend(fontsize=8, loc='upper right')

        # Annotate the θ=0 dots
        ax_full.annotate(r"$\theta=0$", xy=(0.72, 0.10), xycoords='axes fraction',
                         fontsize=8, color='gray')

        ax_full.set_title(r"Total current  $(j_x, j_y)$")
        ax_y.set_title(r"Dirac contribution  $(j_x^{(y)}, j_y^{(x)})$")
        ax_z.set_title(r"Warping contribution  $(j_x^{(z)}, j_y^{(z)})$")

        fig.suptitle(
            rf"Current loop: T={T} K,  $E={params.E_FIELD:.0e}$ V/m,  "
            rf"$\tau={tau:.0e}$ s  "
            r"(dot = $\theta=0$, E-field along $\hat{x}$)",
            fontsize=10,
        )
        fig.tight_layout()
        fig.savefig(CURR_DIR / "currents_03_loops.png", dpi=200)
        plt.close(fig)
        print("  saved currents_03_loops.png")

    return loops
def _k_drift_physical(e_field, tau):
    """Physical k-drift [Å⁻¹]: k = e·E·τ/ħ × 10⁻¹⁰."""
    return scc.elementary_charge * e_field * tau / scc.hbar * 1e-10
 


def sweep_jx_vs_Efield(
    Ef=None, theta=None, T=None,
    tau=1e-13,               # NOTE: intentionally short — with DEFAULT_TAU=1e-10
                             # k_drift >> k_F even at E=10⁵ V/m, hiding the linear regime.
                             # τ=1e-13 s gives k_drift = k_F at E ~ 2.6×10⁶ V/m.
    E_range=(1e4, 1e8),
    n_pts=50,
    save=True,
):
    """
    |jx_y|, |jx_z| vs E-field on log-log axes.
 
    Uses the physical k_drift formula:
        k_drift [Å⁻¹] = e·E·τ/ħ × 10⁻¹⁰
 
    The transition from linear (Ohm's law) to nonlinear occurs at
        E* = k_F·ħ / (e·τ·10⁻¹⁰)
    shown as a vertical dashed line.
 
    Reference slopes E¹ and E³ are drawn for comparison.
 
    Parameters
    ----------
    tau : float
        Scattering time [s].  Default 1e-13 s so that E* ~ 10⁶ V/m sits
        in the middle of the plot range.  Do NOT use DEFAULT_TAU here (it is
        unphysically large and puts k_drift >> k_F everywhere).
    E_range : (E_min, E_max) in V/m
    n_pts   : number of logspaced E values
 
    Returns
    -------
    E_vals : ndarray (n_used,)   — E-field values actually computed [V/m]
    J      : current_dict        — arrays of length n_used
    """
    if Ef    is None: Ef    = params.DEFAULT_EF
    if T     is None: T     = params.DEFAULT_T
    if theta is None: theta = 0.0   # E along x̂
 
    kBT   = params.kBT_eV(T)
    kF    = Ef / params.V_K              # Fermi wavevector [Å⁻¹]
    k_max = float(H.KK_C.max()) * 0.85  # stay inside the k-grid
 
    E_all     = np.logspace(np.log10(E_range[0]), np.log10(E_range[1]), n_pts)
    k_drifts  = _k_drift_physical(E_all, tau)
 
    # Drop points where k_drift exceeds the grid
    good      = k_drifts < k_max
    E_vals    = E_all[good]
    k_use     = k_drifts[good]
 
    if len(E_vals) == 0:
        raise ValueError(
            f"All k_drifts exceed grid limit {k_max:.2f} Å⁻¹. "
            f"Reduce tau or widen E_range."
        )
 
    # E-field at which k_drift = k_F  (linear → nonlinear transition)
    E_star = kF * scc.hbar / (scc.elementary_charge * tau * 1e-10)
 
    print(f"  kF = {kF:.4f} Å⁻¹,  E* = {E_star:.2e} V/m  (k_drift = k_F)")
    print(f"  k_drift range: {k_use[0]:.3e} … {k_use[-1]:.3e} Å⁻¹")
    print(f"  running {len(E_vals)} points …")
 
    arg_list = [
        (Ef, kBT, -k * np.cos(theta), -k * np.sin(theta))
        for k in k_use
    ]
    batch = run_batch_j(arg_list)
    J = _batch_to_currents(batch)
 
    if save:
        fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
 
        panels = [
            ('jx_y', r"$j_x^{(y)}$",   'royalblue'),
            ('jx_z', r"$j_x^{(z)}$",      'crimson'),
            ('jx',   r"$j_x$ total  ($= j_x^{(y)} + j_x^{(z)}$)",             'darkorange'),
        ]
 
        for ax, (key, title, color) in zip(axes, panels):
            vals = np.abs(J[key])
 
            ax.loglog(E_vals, vals, color=color, lw=2.5, label='computed')
 
            # # ── Reference slope lines ────────────────────────────────────────
            # # Slope-1 anchored to first point (linear / Ohm's law)
            # i0 = 0
            # ax.loglog(E_vals, vals[i0] * (E_vals / E_vals[i0]),
            #           'k--', lw=1.2, alpha=0.55, label=r'$\propto E^1$ (Ohm)')
 
            # # Slope-3 anchored to last point
            # i1 = -1
            # ax.loglog(E_vals, vals[i1] * (E_vals / E_vals[i1])**3,
            #           'k:', lw=1.2, alpha=0.55, label=r'$\propto E^3$ (cubic)')
 
            # # ── Mark E* (linear–nonlinear transition) ────────────────────────
            # if E_vals[0] < E_star < E_vals[-1]:
            #     ax.axvline(E_star, color='seagreen', ls='-.', lw=1.5, alpha=0.8,
            #                label=rf"$E^*$ ($k_{{drift}}=k_F={kF:.3f}$ Å$^{{-1}}$)")
 
            ax.set_xlabel(r"$E$ (V/m)")
            ax.set_ylabel(r"$|j_x|$ (A/m)")
            ax.set_title(title, fontsize=9)
            ax.grid(alpha=0.25, which='both')
            ax.legend(fontsize=7)
 
        fig.suptitle(
            rf"$j_x$ vs E-field:  $E_F={Ef}$ eV,  T={T} K,  "
            rf"$\tau={tau:.0e}$ s,  $\theta={theta:.2f}$ rad"
        )
        fig.tight_layout()
        fig.savefig(CURR_DIR / "currents_04_jx_vs_Efield.png", dpi=200)
        plt.close(fig)
        print("  saved currents_04_jx_vs_Efield.png")
 
    return E_vals, J



# ── Convenience: run all three plots ──────────────────────────────────────

def sweep_all():
    """Run all three current plots and print a summary."""
    print("── jx vs Ef ──")
    Ef_vals, data_ef = sweep_jx_vs_Ef()

    print("── jx vs theta ──")
    theta_vals, J_th = sweep_jx_vs_theta()

    print("── 2D loops ──")
    loops = plot_current_loops()

    print("e field")
    e_field = sweep_jx_vs_Efield()

    # Summary at Ef=DEFAULT_EF, T=DEFAULT_T
    T0   = params.DEFAULT_T
    Ef0  = params.DEFAULT_EF
    idx  = np.argmin(np.abs(Ef_vals - Ef0))
    J_ef = data_ef[T0]

    pref_y, pref_z = _prefs()
    print()
    print(f"── Summary at Ef={Ef0} eV, T={T0} K ──────────────────────")
    print(f"  pref_y = {pref_y:.4e} A/m per Å⁻²")
    print(f"  pref_z = {pref_z:.4e} A/m per Å⁻⁴  (= {pref_z/pref_y:.0f}×pref_y × Å²)")
    print(f"  jx_y   = {J_ef['jx_y'][idx]:.4f} A/m   (Dirac)")
    print(f"  jx_z   = {J_ef['jx_z'][idx]:.4f} A/m   (warping)")
    print(f"  jx     = {J_ef['jx'][idx]:.4f} A/m   (total)")
    print(f"  jx_z / jx_y = {J_ef['jx_z'][idx]/J_ef['jx_y'][idx]*100:.2f}%")
    print("──────────────────────────────────────────────────────────")

    return Ef_vals, data_ef, theta_vals, J_th, loops, e_field