import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d

import params
import hamiltonian as H

# ── Multiprocessing worker ─────────────────────────────────────────────────
# _W holds the per-process shared state initialised by _init_worker.

_W = {}

def _init_worker(KK, TH, v_k, lmbd, vx, vy):
    _W.update({
        'KK': KK, 'TH': TH, 'v_k': v_k, 'lmbd': lmbd, 'vx': vx, 'vy': vy,
        'KX': KK * np.cos(TH), 'KY': KK * np.sin(TH),
        'k_ax': KK[:, 0], 'th_ax': TH[0, :],
    })

def _compute(args):
    """Return (⟨sigma_x⟩, ⟨sigma_y⟩, ⟨sigma_z⟩) [Å⁻²] for a shifted Fermi distribution."""
    Ef, kBT, kdx, kdy = args
    kx_s = _W['KX'] - kdx
    ky_s = _W['KY'] - kdy
    sx = sy = sz = 0.0
    for s in (1, -1):
        occ = H.f0(
            H.E_cartesian(kx_s, ky_s, s),   # uses params.V_X/V_Y/LAMBDA
            Ef, kBT
        ) - H.f0(
            H.E_cartesian(_W['KX'], _W['KY'], s),   # uses params.V_X/V_Y/LAMBDA
            Ef, kBT
        )
        KK, TH = _W['KK'], _W['TH']
        sx += np.trapezoid(
            np.trapezoid(occ * H.sigma_x_polar(KK, TH, s) * KK,
                         x=_W['th_ax'], axis=1), x=_W['k_ax'])
        sy += np.trapezoid(
            np.trapezoid(occ * H.sigma_y_polar(KK, TH, s) * KK,
                         x=_W['th_ax'], axis=1), x=_W['k_ax'])
        sz += np.trapezoid(
            np.trapezoid(occ * H.sigma_z_polar(KK, TH, s) * KK,
                         x=_W['th_ax'], axis=1), x=_W['k_ax'])
    norm = (2 * np.pi) ** 2
    return float(sx / norm), float(sy / norm), float(sz / norm)



def run_batch(arg_list):
    """
    Run _compute over arg_list in parallel.

    Each element: (Ef, kBT, kdx, kdy)
    Returns: np.ndarray shape (len(arg_list), 3) — columns: sigma_x, sigma_y, sigma_z [Å⁻²]
    """
    ctx = mp.get_context("fork")
    with ProcessPoolExecutor(
        max_workers=params.N_WORKERS, mp_context=ctx,
        initializer=_init_worker,
        initargs=(H.KK_C, H.TH_C, params.V_K, params.LAMBDA,
                  params.V_X, params.V_Y),
    ) as ex:
        return np.array(list(ex.map(_compute, arg_list)))


# ── Sweep functions ────────────────────────────────────────────────────────

def sweep_Ef_temperature(Ef_range=(-0.5, 0.5), tau=None, save=True):
    """
    Compute ⟨sigma_y⟩ and ⟨sigma_z⟩ vs E_F for each T in params.T_VALUES.

    Returns
    -------
    Ef_vals : ndarray (N_EF,)
    data    : dict {T: ndarray(N_EF, 3)}   columns: sigma_x, sigma_y, sigma_z [Å⁻²]
    """
    if tau is None:
        tau = params.DEFAULT_TAU
    kdx, kdy = params.drift_k(tau, 0.0)
    Ef_vals  = np.linspace(*Ef_range, params.N_EF)
    data = {}
    for T in params.T_VALUES:
        data[T] = run_batch([(Ef, params.kBT_eV(T), kdx, kdy) for Ef in Ef_vals])
        print(f"  T = {T} K done")

    if save:
        subtitle = rf"($\tau$ = {tau}, $\theta$ = 0.0)"
        dEf = Ef_vals[1] - Ef_vals[0]

        for comp, idx, fname in [
            (r"$\langle\sigma_z\rangle$", 2, "sigma_z_vs_Ef.png"),
            (r"$\langle\sigma_y\rangle$", 1, "sigma_y_vs_Ef.png"),
            (r"$\langle\sigma_x\rangle$", 0, "sigma_x_vs_Ef.png"),
        ]:
            fig, ax = plt.subplots(figsize=(7, 5))
            for T, arr in data.items():
                y = arr[:, idx]
                if idx == 2:
                    kBT = params.kBT_eV(T)
                    w = max(3.0, 5.0 * params.kBT_eV(300) / max(kBT, 1e-6))
                    y = gaussian_filter1d(y, sigma=w, mode='nearest')
                ax.plot(Ef_vals, y, lw=2, label=f"T={T} K")
            ax.axhline(0, color="gray", lw=0.8)
            ax.set_xlabel(r"$E_F$ (eV)")
            ax.set_ylabel(comp)
            
            # Main Title & Subtitle
            fig.suptitle(f"{comp} vs $E_F$", fontsize=14)
            ax.set_title(subtitle, fontsize=10, color='dimgray', pad=10)
            
            ax.grid(alpha=0.25)
            ax.legend()
            fig.tight_layout()
            fig.savefig(params.OUT_DIR / fname, dpi=200)
            plt.close(fig)

    return Ef_vals, data


def sweep_tau(Ef=None, T=None, tau_range=(1e-12, 5e-10), save=True):
    """
    Compute ⟨sigma_z⟩ vs scattering time τ.

    Returns
    -------
    tau_vals : ndarray (N_TAU,)
    out      : ndarray (N_TAU, 3)   sigma_x, sigma_y, sigma_z [Å⁻²]
    """
    if Ef is None: Ef = params.DEFAULT_EF
    if T  is None: T  = params.DEFAULT_T
    kBT      = params.kBT_eV(T)
    tau_vals = np.linspace(*tau_range, params.N_TAU)
    out      = run_batch([(Ef, kBT, *params.drift_k(tau, 0.0)) for tau in tau_vals])

    if save:
        subtitle = rf"($E_F$={Ef} eV, T={T} K, $\theta$ = 0.0)"
        
        fig, ax = plt.subplots(figsize=(7, 5))
        ax.plot(tau_vals * 1e12, out[:, 2], lw=2, color="royalblue")
        ax.axhline(0, color="gray", lw=0.8)
        ax.set_xlabel(r"$\tau$ (ps)")
        ax.set_ylabel(r"$\langle\sigma_z\rangle$")
        
        # Main Title & Subtitle
        fig.suptitle(r"$\langle\sigma_z\rangle$ vs $\tau$", fontsize=14)
        ax.set_title(subtitle, fontsize=10, color='dimgray', pad=10)
        
        ax.grid(alpha=0.25)
        fig.tight_layout()
        fig.savefig(params.OUT_DIR / "sigma_z_vs_tau.png", dpi=200)
        plt.close(fig)

    return tau_vals, out


def sweep_theta(Ef=None, T=None, tau=None, save=True):
    """
    Compute ⟨sigma_y⟩ and ⟨sigma_z⟩ vs drift direction θ.

    Returns
    -------
    theta_vals : ndarray (N_THETA,)
    out        : ndarray (N_THETA, 3)   sigma_x, sigma_y, sigma_z [Å⁻²]
    """
    if Ef  is None: Ef  = params.DEFAULT_EF
    if T   is None: T   = params.DEFAULT_T
    if tau is None: tau = params.DEFAULT_TAU
    theta_vals = np.linspace(0, 2 * np.pi, params.N_THETA)
    kBT        = params.kBT_eV(T)
    out        = run_batch(
        [(Ef, kBT, *params.drift_k(tau, th)) for th in theta_vals]
    )

    if save:
        subtitle = rf"($E_F$={Ef} eV, T={T} K, $\tau$={tau})"
        
        for comp, idx, color, fname in [
            (r"$\langle\sigma_z\rangle$", 2, "crimson",    "sigma_z_vs_theta.png"),
            (r"$\langle\sigma_y\rangle$", 1, "darkorange", "sigma_y_vs_theta.png"),
        ]:
            fig, ax = plt.subplots(figsize=(7, 5))
            ax.plot(theta_vals, out[:, idx], lw=2, color=color)
            ax.set_ylabel(comp)
            
            # Main Title & Subtitle
            fig.suptitle(comp + r" vs $\theta$", fontsize=14)
            ax.set_title(subtitle, fontsize=10, color='dimgray', pad=10)
            
            if hasattr(H, 'style_theta_axis'):
                H.style_theta_axis(ax)
                
            fig.tight_layout()
            fig.savefig(params.OUT_DIR / fname, dpi=200)
            plt.close(fig)

    return theta_vals, out

def sweep_E_field(Ef=None, theta = None, T=None, tau=None, save=True):
    if Ef  is None: Ef  = params.DEFAULT_EF
    if T   is None: T   = params.DEFAULT_T
    if tau is None: tau = params.DEFAULT_TAU
    if theta is None: theta = params.DEFAULT_THETA
    
    E_field = np.linspace(0, 1e7, 25)
    kBT        = params.kBT_eV(T)
    out        = run_batch(
        [(Ef, kBT, *params.drift_k(tau, theta, e)) for e in E_field]
    )

    if save:
        subtitle = rf"($E_F$={Ef} eV, T={T} K, $\tau$ = {tau}, $\theta$ = {theta})"

        for comp, idx, color, fname in [
            (r"$\langle\sigma_z\rangle$", 2, "crimson",    "sigma_z_vs_E_field.png"),
            (r"$\langle\sigma_y\rangle$", 1, "darkorange", "sigma_y_vs_E_field.png"),
        ]:
            fig, ax = plt.subplots(figsize=(7, 5))
            ax.plot(E_field, out[:, idx], lw=2, color=color)
            ax.set_ylabel(comp)
            
            # 1. Main Title (larger font, overarching figure title)
            fig.suptitle(comp + r" vs $E$", fontsize=14)
            
            # 2. Subtitle (smaller font, attached directly to the axes)
            ax.set_title(subtitle, fontsize=10, color='dimgray', pad=10)
            fig.tight_layout()
            fig.savefig(params.OUT_DIR / fname, dpi=200)
            plt.close(fig)
    
    return E_field, out