"""
cisp_sweeps.py — CISP (S_x, S_y, S_z) vs conductivity and vs E-field.

Conductivity σ_xx includes the hexagonal-warping correction:
    j_x = -(e²·VK·1e10/ħ)·⟨σ_y⟩  -  (e²·λ·1e10/ħ)·∫(3kx²-3ky²)σ_z df d²k/(2π)²
    σ_xx = j_x / E   [S]   (2D sheet conductance; j_x is A/m for a surface state)

Two sweeps:
  1. vs E-field (fixed τ)  →  σ_xx varies through E
  2. vs τ       (fixed E)  →  σ_xx varies through τ

Run:  python cisp_sweeps.py
"""

import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import matplotlib.pyplot as plt
import scipy.constants as scc

import params
import hamiltonian as H

# ── Physical constants ─────────────────────────────────────────────────────

def _k_drift(E, tau):
    """Physical k-space drift [Å⁻¹] along x̂: dk = e·E·τ/ħ × 10⁻¹⁰."""
    return scc.elementary_charge * E * tau / scc.hbar * 1e-10


def _current_prefactors():
    """Return (pref_VK, pref_lam) in A/m per code-integral unit."""
    e, hbar = scc.elementary_charge, scc.hbar
    pref_VK  = e**2 * params.V_K    * 1e10 / hbar   # A/m per Å⁻²
    pref_lam = e**2 * params.LAMBDA * 1e10 / hbar   # A/m per Å⁻⁴
    return pref_VK, pref_lam


# ── Multiprocessing worker ─────────────────────────────────────────────────
_W = {}


def _init_worker(KK, TH):
    KX = KK * np.cos(TH)
    KY = KK * np.sin(TH)
    _W.update({
        'KK': KK, 'TH': TH, 'KX': KX, 'KY': KY,
        'k_ax': KK[:, 0], 'th_ax': TH[0, :],
        'sx': {s: H.sigma_x_polar(KK, TH, s) for s in (1, -1)},
        'sy': {s: H.sigma_y_polar(KK, TH, s) for s in (1, -1)},
        'sz': {s: H.sigma_z_polar(KK, TH, s) for s in (1, -1)},
    })


def _integrate(field):
    KK = _W['KK']
    return np.trapezoid(
        np.trapezoid(field * KK, x=_W['th_ax'], axis=1),
        x=_W['k_ax'],
    ) / (2.0 * np.pi) ** 2


def _compute(args):
    """Return (S_x, S_y, S_z [Å⁻²], jx_VK [Å⁻²], jx_lam [Å⁻⁴])."""
    Ef, kBT, kdx, kdy = args
    KX, KY = _W['KX'], _W['KY']

    Sx = Sy = Sz = 0.0
    jx_VK = jx_lam = 0.0

    for s in (1, -1):
        f_eq = H.f0(H.E_cartesian(KX, KY, s), Ef, kBT)
        f_sh = H.f0(H.E_cartesian(KX - kdx, KY - kdy, s), Ef, kBT)
        df = f_sh - f_eq

        sxp, syp, szp = _W['sx'][s], _W['sy'][s], _W['sz'][s]

        Sx += _integrate(df * sxp)
        Sy += _integrate(df * syp)
        Sz += _integrate(df * szp)

        jx_VK  += _integrate(df * syp)
        jx_lam += _integrate((3.0 * KX**2 - 3.0 * KY**2) * df * szp)

    return float(Sx), float(Sy), float(Sz), float(jx_VK), float(jx_lam)


def run_batch(arg_list):
    ctx = mp.get_context("fork")
    with ProcessPoolExecutor(
        max_workers=params.N_WORKERS, mp_context=ctx,
        initializer=_init_worker,
        initargs=(H.KK_C, H.TH_C),
    ) as ex:
        return np.array(list(ex.map(_compute, arg_list)))


def _conductivity_S(jx_VK, jx_lam, E_field):
    """
    Compute sheet conductance σ_xx in siemens from code integrals.

    jx [A/m] = -pref_VK * jx_VK - pref_lam * jx_lam   (2D current density)
    σ_xx [S] = jx / E   =  (A/m) / (V/m) = A/V = S
    """
    pVK, plam = _current_prefactors()
    jx = -pVK * jx_VK - plam * jx_lam       # A/m  (2D current density)
    return jx / E_field                       # S  (sheet conductance)


# ── Output directories ────────────────────────────────────────────────────
_OUT        = params.OUT_DIR / "cisp_sweeps"
_OUT_EFIELD = _OUT / "E_field"
_OUT_COND   = _OUT / "conductivity"
_LABELS = (r"$S_x$", r"$S_y$", r"$S_z$")
_KEYS = ("S_x", "S_y", "S_z")


def _title_params(**kw):
    """Build subtitle string showing held-fixed parameters."""
    parts = []
    if 'lam' in kw:
        parts.append(rf"$\lambda$ = {kw['lam']:g} eV·Å³")
    if 'vk' in kw:
        parts.append(rf"$v_k$ = {kw['vk']:g} eV·Å")
    if 'tau' in kw:
        parts.append(rf"$\tau$ = {kw['tau']:.2e} s")
    if 'E' in kw:
        parts.append(rf"$E$ = {kw['E']:.2e} V/m")
    if 'Ef' in kw:
        parts.append(rf"$E_F$ = {kw['Ef']:g} eV")
    if 'T' in kw:
        parts.append(rf"$T$ = {kw['T']:.0f} K")
    return ",  ".join(parts)


def _plot_to_dir(out_dir, x, S3, xlabel, main_title, subtitle, fname_stub):
    """Plot S_x, S_y, S_z (individual + combined)."""
    out_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 5))
    for i, lab in enumerate(_LABELS):
        ax.plot(x, S3[:, i], marker="o", ms=3, lw=1.8, label=lab)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(r"Spin accumulation $S_i$ (Å$^{-2}$)")
    ax.axhline(0, color="k", lw=0.6, alpha=0.4)
    ax.grid(alpha=0.25)
    ax.legend(loc="best")
    fig.suptitle(main_title, fontsize=13)
    ax.set_title(subtitle, fontsize=9, color="dimgray", pad=10)
    fig.tight_layout()
    fig.savefig(out_dir / f"{fname_stub}_all.png", dpi=200)
    plt.close(fig)

    for i, (lab, key) in enumerate(zip(_LABELS, _KEYS)):
        fig, ax = plt.subplots(figsize=(7, 5))
        ax.plot(x, S3[:, i], marker="o", ms=3, color=f"C{i}", lw=1.8)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(lab + r" (Å$^{-2}$)")
        ax.axhline(0, color="k", lw=0.6, alpha=0.4)
        ax.grid(alpha=0.25)
        fig.suptitle(f"{lab}  {main_title}", fontsize=13)
        ax.set_title(subtitle, fontsize=9, color="dimgray", pad=10)
        fig.tight_layout()
        fig.savefig(out_dir / f"{key}_{fname_stub}.png", dpi=200)
        plt.close(fig)


def _plot_vs_loglog(E_vals, S3, E_star, subtitle):
    """Log-log plot of |S_y| and |S_z| vs E showing power-law scaling."""
    _OUT_EFIELD.mkdir(parents=True, exist_ok=True)

    for i, (lab, key, color, slope) in enumerate([
        (r"$|S_y|$", "S_y", "C1", 1),
        (r"$|S_z|$", "S_z", "C2", 3),
    ]):
        fig, ax = plt.subplots(figsize=(7, 5))
        vals = np.abs(S3[:, i + 1])
        mask = vals > 0
        ax.loglog(E_vals[mask], vals[mask], marker="o", ms=4, lw=2,
                  color=color, label="computed")

        i0 = np.argmax(mask)
        ref = vals[i0] * (E_vals / E_vals[i0]) ** slope
        ax.loglog(E_vals, ref, "k--", lw=1, alpha=0.5,
                  label=rf"$\propto E^{slope}$")

        if E_vals[0] < E_star < E_vals[-1]:
            ax.axvline(E_star, color="seagreen", ls="-.", lw=1.5, alpha=0.7,
                       label=rf"$E^*$ ($k_{{drift}}=k_F$)")

        ax.set_xlabel(r"$E$ (V/m)")
        ax.set_ylabel(lab + r" (Å$^{-2}$)")
        ax.grid(alpha=0.25, which="both")
        ax.legend(fontsize=8)
        fig.suptitle(f"{lab} vs E (log-log)", fontsize=13)
        ax.set_title(subtitle, fontsize=9, color="dimgray", pad=10)
        fig.tight_layout()
        fig.savefig(_OUT_EFIELD / f"{key}_vs_Efield_loglog.png", dpi=200)
        plt.close(fig)


# ── Sweep 1: CISP vs E-field (fixed τ) ────────────────────────────────────

def sweep_vs_Efield(E_vals=None, tau=1e-11, Ef=None, T=None, save=True):
    if E_vals is None:
        E_vals = np.logspace(4, 8, 40)
    Ef  = params.DEFAULT_EF  if Ef  is None else Ef
    T   = params.DEFAULT_T   if T   is None else T
    kBT = params.kBT_eV(T)

    k_max = float(H.KK_C.max()) * 0.85
    k_drifts = _k_drift(E_vals, tau)
    good = k_drifts < k_max
    if not good.all():
        print(f"  WARNING: dropping {(~good).sum()} points where k_drift > k_grid")
    E_vals = E_vals[good]
    k_drifts = k_drifts[good]

    args = [(Ef, kBT, -kd, 0.0) for kd in k_drifts]
    res = run_batch(args)
    S     = res[:, :3]
    sigma = _conductivity_S(res[:, 3], res[:, 4], E_vals)

    kF = Ef / params.V_K
    E_star = kF * scc.hbar / (scc.elementary_charge * tau * 1e-10)
    print(f"sweep_vs_Efield done — σ_xx range: {sigma.min():.4e} to {sigma.max():.4e} S")
    print(f"  E* (k_drift=k_F) = {E_star:.2e} V/m")

    if save:
        subtitle = _title_params(tau=tau, lam=params.LAMBDA, vk=params.V_K,
                                 Ef=Ef, T=T)
        _plot_vs_loglog(E_vals, S, E_star, subtitle)

        _plot_to_dir(_OUT_EFIELD, E_vals, S,
                     r"Electric field $E$ (V/m)",
                     r"CISP vs $E$ field",
                     subtitle, "vs_Efield")

        _plot_to_dir(_OUT_COND, sigma, S,
                     r"Conductivity $\sigma_{xx}$ (S)",
                     r"CISP vs $\sigma_{xx}$ (sweep $E$, fixed $\tau$)",
                     subtitle, "vs_sigma_sweepE")

    return E_vals, S, sigma


# ── Sweep 2: CISP vs τ (fixed E) ──────────────────────────────────────────

def sweep_vs_tau(tau_vals=None, E=None, Ef=None, T=None, save=True):
    if tau_vals is None:
        tau_vals = np.linspace(1e-14, 5e-12, 30)
    E  = params.E_FIELD   if E  is None else E
    Ef = params.DEFAULT_EF if Ef is None else Ef
    T  = params.DEFAULT_T  if T  is None else T
    kBT = params.kBT_eV(T)

    k_max = float(H.KK_C.max()) * 0.85
    k_drifts = _k_drift(E, tau_vals)
    good = k_drifts < k_max
    if not good.all():
        print(f"  WARNING: dropping {(~good).sum()} points where k_drift > k_grid")
    tau_vals = tau_vals[good]
    k_drifts = k_drifts[good]

    args = [(Ef, kBT, -kd, 0.0) for kd in k_drifts]
    res = run_batch(args)
    S     = res[:, :3]
    sigma = _conductivity_S(res[:, 3], res[:, 4], E)

    order = np.argsort(sigma)
    sigma_sorted = sigma[order]
    S_sorted = S[order]

    print(f"sweep_vs_tau done — σ_xx range: {sigma.min():.4e} to {sigma.max():.4e} S")

    if save:
        subtitle = _title_params(E=E, lam=params.LAMBDA, vk=params.V_K,
                                 Ef=Ef, T=T)
        _plot_to_dir(_OUT_COND, sigma_sorted, S_sorted,
                     r"Conductivity $\sigma_{xx}$ (S)",
                     r"CISP vs $\sigma_{xx}$ (sweep $\tau$, fixed $E$)",
                     subtitle, "vs_sigma_sweepTau")

    return tau_vals, S, sigma


# ── Main ───────────────────────────────────────────────────────────────────

def main():
    import shutil
    # clean old flat plots
    if _OUT.exists():
        shutil.rmtree(_OUT)
    _OUT_EFIELD.mkdir(parents=True, exist_ok=True)
    _OUT_COND.mkdir(parents=True, exist_ok=True)

    print("=== CISP sweep: vs E field ===")
    sweep_vs_Efield()
    print("=== CISP sweep: vs tau (-> conductivity) ===")
    sweep_vs_tau()
    print(f"Plots written to {_OUT_EFIELD} and {_OUT_COND}")


if __name__ == "__main__":
    main()
