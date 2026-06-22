"""
cisp_sweeps.py — Sweeps of the current-induced spin polarization (CISP)
S_x, S_y, S_z versus (a) the applied electric field E and (b) the charge
conductivity sigma_xx.

Definitions
-----------
The spin accumulations are the *non-equilibrium* parts of <sigma_i>, i.e. the
difference between the drift-shifted Fermi sea and the equilibrium one:

    S_i(E) = sum_s INT d^2k/(2pi)^2  [ f0(E_s(k - dk)) - f0(E_s(k)) ] <sigma_i>_s

with the drift shift  dk = (e tau / hbar) E  supplied by params.drift_k().
The field is applied along x-hat (theta = 0) so dk = (-E*tau, 0) in the code's
convention.

The longitudinal charge conductivity follows from the current density
(see the project notes):

    j_x = -(e v_k / hbar) INT <sigma_y> df  d^2k/(2pi)^2
          -(e lambda / hbar) INT (3 k_x^2 - 3 k_y^2) <sigma_z> df  d^2k/(2pi)^2
    sigma_xx = j_x / E

with df = f0(E_s(k - dk)) - f0(E_s(k)).  Both pieces are evaluated together so
each (E, tau) point needs only a single k-integration pass.

Two sweeps are produced:
  * vs E field     : fixed tau, vary |E|        -> plots/cisp_sweeps/S_*_vs_Efield.png
  * vs conductivity: fixed E,   vary tau (-> sigma_xx) -> plots/cisp_sweeps/S_*_vs_sigma.png

Dependency: params, hamiltonian.
Run:  python cisp_sweeps.py
"""

import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import matplotlib.pyplot as plt
import scipy.constants as scc

import params
import hamiltonian as H

# physical prefactor e/hbar (SI) used for the conductivity scale
_E_OVER_HBAR = scc.elementary_charge / scc.hbar

# ── Multiprocessing worker ─────────────────────────────────────────────────
_W = {}


def _init_worker(KK, TH, v_k, lmbd, vx, vy):
    KX = KK * np.cos(TH)
    KY = KK * np.sin(TH)
    _W.update({
        'KK': KK, 'TH': TH, 'v_k': v_k, 'lmbd': lmbd, 'vx': vx, 'vy': vy,
        'KX': KX, 'KY': KY,
        'k_ax': KK[:, 0], 'th_ax': TH[0, :],
        # band spin textures are field-independent -> cache them
        'sx': {s: H.sigma_x_polar(KK, TH, s) for s in (1, -1)},
        'sy': {s: H.sigma_y_polar(KK, TH, s) for s in (1, -1)},
        'sz': {s: H.sigma_z_polar(KK, TH, s) for s in (1, -1)},
    })


def _integrate(field):
    """INT field * k  dk dtheta  over the cached polar grid."""
    KK = _W['KK']
    return np.trapezoid(
        np.trapezoid(field * KK, x=_W['th_ax'], axis=1),
        x=_W['k_ax'],
    )


def _compute(args):
    """Return (S_x, S_y, S_z [A^-2], j_x [arb. units]) for one (Ef, kBT, kdx, kdy)."""
    Ef, kBT, kdx, kdy = args
    KX, KY = _W['KX'], _W['KY']

    Sx = Sy = Sz = 0.0
    jx = 0.0
    for s in (1, -1):
        f_eq = H.f0(H.E_cartesian(KX, KY, s), Ef, kBT)
        f_sh = H.f0(H.E_cartesian(KX - kdx, KY - kdy, s), Ef, kBT)
        df = f_sh - f_eq

        sxp, syp, szp = _W['sx'][s], _W['sy'][s], _W['sz'][s]

        Sx += _integrate(df * sxp)
        Sy += _integrate(df * syp)
        Sz += _integrate(df * szp)

        # charge current j_x (Drude / Boltzmann form from project notes)
        jx += -_W['v_k'] * _integrate(df * syp)
        jx += -_W['lmbd'] * _integrate((3.0 * KX**2 - 3.0 * KY**2) * df * szp)

    norm = (2.0 * np.pi) ** 2
    jx *= _E_OVER_HBAR
    return float(Sx / norm), float(Sy / norm), float(Sz / norm), float(jx / norm)


def run_batch(arg_list):
    """Parallel map of _compute. Returns array shape (N, 4): Sx, Sy, Sz, jx."""
    with ProcessPoolExecutor(
        max_workers=params.N_WORKERS,
        initializer=_init_worker,
        initargs=(H.KK_C, H.TH_C, params.V_K, params.LAMBDA,
                  params.V_X, params.V_Y),
    ) as ex:
        return np.array(list(ex.map(_compute, arg_list)))


# ── Plot helpers ───────────────────────────────────────────────────────────
_OUT = params.OUT_DIR / "cisp_sweeps"
_LABELS = (r"$S_x$", r"$S_y$", r"$S_z$")
_KEYS = ("S_x", "S_y", "S_z")


def _annotate_params(ax, pinfo):
    """Draw a box highlighting the held-fixed variables (tau, lambda, v_k, ...)."""
    if not pinfo:
        return
    lines = [f"{name} = {val}" for name, val in pinfo.items()]
    ax.text(
        0.025, 0.975, "\n".join(lines),
        transform=ax.transAxes, va="top", ha="left", fontsize=9,
        family="DejaVu Sans",
        bbox=dict(boxstyle="round,pad=0.4", fc="#fffbe6", ec="0.5", alpha=0.92),
    )


def _plot(x, y3, xlabel, fname_stub, title_suffix, pinfo=None):
    _OUT.mkdir(parents=True, exist_ok=True)
    # one combined figure
    fig, ax = plt.subplots(figsize=(7, 5))
    for i, lab in enumerate(_LABELS):
        ax.plot(x, y3[:, i], marker="o", ms=3, label=lab)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(r"Spin accumulation $S_i$ ($\AA^{-2}$)")
    ax.set_title("CISP " + title_suffix)
    ax.axhline(0, color="k", lw=0.6, alpha=0.4)
    ax.grid(alpha=0.25)
    _annotate_params(ax, pinfo)
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(_OUT / f"{fname_stub}_all.png", dpi=200)
    plt.close(fig)

    # individual component figures
    for i, (lab, key) in enumerate(zip(_LABELS, _KEYS)):
        fig, ax = plt.subplots(figsize=(7, 5))
        ax.plot(x, y3[:, i], marker="o", ms=3, color=f"C{i}")
        ax.set_xlabel(xlabel)
        ax.set_ylabel(lab + r" ($\AA^{-2}$)")
        ax.set_title(lab + " " + title_suffix)
        ax.axhline(0, color="k", lw=0.6, alpha=0.4)
        ax.grid(alpha=0.25)
        _annotate_params(ax, pinfo)
        fig.tight_layout()
        fig.savefig(_OUT / f"{key}_{fname_stub}.png", dpi=200)
        plt.close(fig)


# ── Sweeps ─────────────────────────────────────────────────────────────────
def sweep_vs_Efield(E_vals=None, tau=None, Ef=None, T=None, save=True):
    """S_x, S_y, S_z vs electric field magnitude (field along x-hat)."""
    if E_vals is None:
        E_vals = np.linspace(0.0, 2e6, 21)        # V/m
    tau = params.DEFAULT_TAU if tau is None else tau
    Ef = params.DEFAULT_EF if Ef is None else Ef
    T = params.DEFAULT_T if T is None else T
    kBT = params.kBT_eV(T)

    args = []
    for E in E_vals:
        kdx, kdy = params.drift_k(tau, 0.0, E)     # theta = 0 -> field along x
        args.append((Ef, kBT, kdx, kdy))

    res = run_batch(args)
    S = res[:, :3]
    print("sweep_vs_Efield done "
          f"(tau={tau:.2e} s, Ef={Ef} eV, T={T} K, {len(E_vals)} pts)")

    if save:
        pinfo = {
            r"$\tau$": f"{tau:.2e} s",
            r"$\lambda$": f"{params.LAMBDA:g} eV\u00b7\u00c5\u00b3",
            r"$v_k$": f"{params.V_K:g} eV\u00b7\u00c5",
            r"$E_F$": f"{Ef:g} eV",
            r"$T$": f"{T:.0f} K",
            "field": r"$\parallel \hat{x}$",
        }
        _plot(E_vals, S, r"Electric field $E$ (V/m)",
              "vs_Efield", r"vs $E$ field", pinfo=pinfo)
    return E_vals, S


def sweep_vs_conductivity(tau_vals=None, E=None, Ef=None, T=None, save=True):
    """S_x, S_y, S_z vs charge conductivity sigma_xx (varied via tau, fixed E)."""
    if tau_vals is None:
        tau_vals = np.linspace(1e-13, 2e-12, 20)   # s
    E = params.E_FIELD if E is None else E
    Ef = params.DEFAULT_EF if Ef is None else Ef
    T = params.DEFAULT_T if T is None else T
    kBT = params.kBT_eV(T)

    args = []
    for tau in tau_vals:
        kdx, kdy = params.drift_k(tau, 0.0, E)
        args.append((Ef, kBT, kdx, kdy))

    res = run_batch(args)
    S = res[:, :3]
    jx = res[:, 3]
    sigma = jx / E                                  # sigma_xx = j_x / E
    # sort by conductivity so the curve is monotone in x
    order = np.argsort(sigma)
    sigma, S = sigma[order], S[order]
    print("sweep_vs_conductivity done "
          f"(E={E:.2e} V/m, Ef={Ef} eV, T={T} K, {len(tau_vals)} pts)")

    if save:
        pinfo = {
            r"$E$": f"{E:.2e} V/m",
            r"$\lambda$": f"{params.LAMBDA:g} eV\u00b7\u00c5\u00b3",
            r"$v_k$": f"{params.V_K:g} eV\u00b7\u00c5",
            r"$E_F$": f"{Ef:g} eV",
            r"$T$": f"{T:.0f} K",
            r"$\tau$": "swept",
        }
        _plot(sigma, S, r"Conductivity $\sigma_{xx}$ (arb. units)",
              "vs_sigma", r"vs conductivity $\sigma_{xx}$", pinfo=pinfo)
    return sigma, S


def main():
    _OUT.mkdir(parents=True, exist_ok=True)
    print("=== CISP sweep: S_i vs E field ===")
    sweep_vs_Efield()
    print("=== CISP sweep: S_i vs conductivity ===")
    sweep_vs_conductivity()
    print(f"Plots written to {_OUT}")


if __name__ == "__main__":
    main()
