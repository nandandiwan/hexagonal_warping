"""
fisher.py — Fischer iSGE spin-transfer torque from TI surface CISP.

Physics summary
---------------
Step 1 (Fischer Eq. 3):
    δn_y [nm⁻²] = -j_x / (e·v_F)          — pure Dirac, Ef-independent
    δn_z [nm⁻²] = R_zy(Ef,T) × δn_y       — warping correction from code

Step 2 (direct exchange torque):
    T_vec [J/m²] = (Δ_ex/2) × [m̂×ŷ·δn_y + m̂×ẑ·δn_z] × 10¹⁸
    B_eff  [mT]  = |T_vec| / (M_s·t_FM) × 1000

For in-plane m̂ = (cosφ, sinφ, 0):
    m̂ × ŷ = cosφ ẑ                 → out-of-plane torque (FL) from δn_y
    m̂ × ẑ = sinφ x̂ − cosφ ŷ      → in-plane torque ⊥ m̂ (DL-type) from δn_z

Public API
----------
dn_y(j_2d)                          — physical y-spin density [nm⁻²]
b_eff(dn_y, dn_z, m_phi, ...)       — effective field vector [mT]
sweep_fisher(...)                   — four diagnostic plots → plots/fisher/

Dependency: params, hamiltonian, cisp
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy.constants as scc

import params
import hamiltonian as H
import cisp as C

FISHER_DIR = params.OUT_DIR / "fisher"


# ── Core formulas ──────────────────────────────────────────────────────────

def dn_y(j_2d=None):
    """
    Physical y-spin density [nm⁻²] from Fischer Eq.(3).

    δn_y = -j_x / (e · v_F · 10¹⁸)

    Exact for the pure Dirac surface state; independent of E_F and T.
    """
    if j_2d is None: j_2d = params.FISHER_J_2D
    vF = params.vF_SI()
    return -j_2d / (scc.elementary_charge * vF * 1e18)   # nm⁻²


def b_eff(dn_y_nm2, dn_z_nm2, m_phi,
          dex_meV=None, M_s=None, t_fm=None):
    """
    Exchange torque as effective field vector B_eff [mT].

    T_vec = (Δ_ex/2) × (m̂×ŷ·δn_y + m̂×ẑ·δn_z) × 10¹⁸  [J/m²]
    B_vec = T_vec / (M_s·t_FM) × 1000                    [mT]

    Returns
    -------
    B_vec : ndarray (3,) [mT]
    B_mag : float        |B_vec| [mT]
    """
    if dex_meV is None: dex_meV = params.FISHER_DEL_EX
    if M_s     is None: M_s     = params.FISHER_M_S
    if t_fm    is None: t_fm    = params.FISHER_T_FM

    dex_J   = dex_meV * 1e-3 * scc.elementary_charge
    m       = np.array([np.cos(m_phi), np.sin(m_phi), 0.0])
    cross_y = np.cross(m, [0, 1, 0])   # m̂ × ŷ  = cosφ ẑ
    cross_z = np.cross(m, [0, 0, 1])   # m̂ × ẑ  = sinφ x̂ − cosφ ŷ
    T_vec   = (dex_J / 2) * (cross_y * dn_y_nm2 + cross_z * dn_z_nm2) * 1e18
    B_mT    = T_vec / (M_s * t_fm) * 1000.0
    return B_mT, float(np.linalg.norm(B_mT))


# ── Main sweep ─────────────────────────────────────────────────────────────

def sweep_fisher(j_2d=None, Ef_range=(-0.5, 0.5), save=True):
    """
    Four diagnostic plots saved to plots/fisher/.

    01 — δn_y, δn_z [nm⁻²] and |δn_z/δn_y| vs E_F
    02 — δn_y, δn_z vs current direction θ
    03 — B_eff [mT] vs E_F (pure Dirac only | + warping)
    04 — B_eff [mT] vs j_x (log-log)  +  vs m̂ angle φ_m

    Returns
    -------
    summary : dict with dn_y_ref, dn_z_at_01, r_at_01, B_45deg
    """
    if j_2d is None: j_2d = params.FISHER_J_2D

    Ef_vals   = np.linspace(*Ef_range, params.N_EF)
    th_vals   = np.linspace(0, 2 * np.pi, params.N_THETA)
    colors    = plt.cm.plasma(np.linspace(0.1, 0.9, len(params.T_VALUES)))
    m_phi45   = np.pi / 4     # Fischer geometry: m̂ at 45°
    dn_y_ref  = dn_y(j_2d)   # constant [nm⁻²]

    E_field_vals = np.logspace(5, 10, 51)



    # ── Run CISP sweeps ──────────────────────────────────────────────────
    print("  fisher: CISP vs Ef …")
    kdx0, kdy0 = params.drift_k(params.DEFAULT_TAU, 0.0)
    code_ef = {}
    for T in params.T_VALUES:
        code_ef[T] = C.run_batch(
            [(Ef, params.kBT_eV(T), kdx0, kdy0) for Ef in Ef_vals]
        )
        print(f"    T = {T} K done")

    print("  fisher: CISP vs theta …")
    code_th = C.run_batch([
        (params.DEFAULT_EF, params.kBT_eV(params.DEFAULT_T),
         *params.drift_k(params.DEFAULT_TAU, th ))
        for th in th_vals
    ])

    code_e_field = C.run_batch([
        (params.DEFAULT_EF, params.kBT_eV(params.DEFAULT_T),
         *params.drift_k(params.DEFAULT_TAU, params.DEFAULT_THETA, e_field ))
        for e_field in E_field_vals
    ])

    sy_ref_nm2 = float(np.interp(
        params.DEFAULT_EF, Ef_vals,
        code_ef[params.DEFAULT_T][:, 1] * 100   # Å⁻² → nm⁻²
    ))
    scale = 1   # dimensionless

    # Reference values at Ef=DEFAULT_EF, T=DEFAULT_T
    _sy_ref = code_ef[params.DEFAULT_T][:, 1] * 100 * scale
    _sz_ref = code_ef[params.DEFAULT_T][:, 2] * 100 * scale
    dn_y_at_01 = float(np.interp(params.DEFAULT_EF, Ef_vals, _sy_ref))
    dn_z_at_01 = float(np.interp(params.DEFAULT_EF, Ef_vals, _sz_ref))
    r_at_01    = dn_z_at_01 / dn_y_at_01 if abs(dn_y_at_01) > 1e-30 else 0.0

    # ── Fig 01: spin density vs E_F ──────────────────────────────────────
    fig1, axes1 = plt.subplots(1, 3, figsize=(15, 4.5))
    ax_sy, ax_sz, ax_rat = axes1

    for T, col in zip(params.T_VALUES, colors):
        # Both components scaled the same way — consistent λ=255 treatment
        dn_y_arr = code_ef[T][:, 1] * 100   # nm⁻², varies with Ef
        dn_z_arr = code_ef[T][:, 2] * 100   # nm⁻², varies with Ef
        ratio_zy = np.where(np.abs(dn_y_arr) > 1e-20, dn_z_arr / dn_y_arr, 0.0)

        ax_sy.plot(Ef_vals, dn_y_arr * 1e3, color=col, lw=2, label=f"T={T} K")
        ax_sz.plot(Ef_vals, dn_z_arr * 1e3, color=col, lw=2, label=f"T={T} K")
        ax_rat.plot(Ef_vals, np.abs(ratio_zy), color=col, lw=2)

    # Mark the reference value at Ef=DEFAULT_EF
    # ax_sy.axvline(params.DEFAULT_EF, color='gray', ls=':', lw=1, alpha=0.6)
    # ax_sy.annotate(
    #     rf"@ $E_F={params.DEFAULT_EF}$ eV: $\delta n_y = {dn_y_at_01*1e3:.2f}\times10^{{-3}}$ nm$^{{-2}}$"
    #     "\n" r"(matches Fischer Eq.(3) at reference point)",
    #     xy=(params.DEFAULT_EF, dn_y_at_01 * 1e3),
    #     xytext=(0.25, 0.15), textcoords='axes fraction', fontsize=7,
    #     arrowprops=dict(arrowstyle='->', lw=0.8),
    #     bbox=dict(boxstyle='round', fc='wheat', alpha=0.8),
    # )
    ax_sy.set_ylabel(r"$\delta n_y \ (10^{-3}\ \mathrm{nm}^{-2})$")
    ax_sy.set_title(r"$\langle S_y\rangle_\mathrm{neq}$ — warped Hamiltonian ($\lambda$="
                    + rf"{params.LAMBDA:.0f} eV·Å³)")
    ax_sy.legend(fontsize=7)
    ax_sz.set_ylabel(r"$\delta n_z \ (10^{-3}\ \mathrm{nm}^{-2})$")
    ax_sz.set_title(r"$\langle S_z\rangle_\mathrm{neq}$ — warping correction")
    ax_sz.legend(fontsize=7)
    ax_rat.set_ylabel(r"$|\delta n_z / \delta n_y|$")
    ax_rat.set_title(r"Warping fraction $|S_z / S_y|$")
    for ax in axes1:
        ax.axhline(0, color='gray', lw=0.8)
        ax.set_xlabel(r"$E_F$ (eV)")
        ax.grid(alpha=0.25)
    fig1.suptitle(
        rf"Fischer CISP: $j_x={j_2d:.0f}$ A/m, "
        rf"$v_F={params.vF_SI()/1e5:.2f}\times10^5$ m/s",
        fontsize=10,
    )
    fig1.tight_layout()
    if save:
        fig1.savefig(FISHER_DIR / "fisher_01_spin_density_vs_Ef.png", dpi=200)
        plt.close(fig1)

    # ── Fig 02: spin density vs θ ─────────────────────────────────────────
    dn_y_th = code_th[:, 1] * 100 * scale
    dn_z_th = code_th[:, 2] * 100 * scale

    fig2, (ax2a, ax2b) = plt.subplots(1, 2, figsize=(12, 4.5))
    ax2a.plot(th_vals, dn_y_th * 1e3, color='royalblue', lw=2, label=r"$\delta n_y$ (iSGE)")
    ax2a.plot(th_vals, dn_z_th * 1e3, color='crimson',   lw=2, label=r"$\delta n_z$ (warping)")
    # ax2a.plot(th_vals, np.hypot(dn_y_th, dn_z_th) * 1e3,
    #           color='k', lw=1.5, ls='--', label=r"$|\delta\mathbf{n}|$")
    safe_y = np.where(np.abs(dn_y_th) > 1e-30, dn_y_th, np.nan)
    ax2b.plot(th_vals, np.abs(dn_z_th / safe_y), color='darkorange', lw=2)
    ax2a.set_ylabel(r"$\delta n \ (10^{-3}\ \mathrm{nm}^{-2})$")
    ax2a.set_title(r"Spin density vs current direction $\theta$")
    ax2a.legend(fontsize=8)
    ax2b.set_ylabel(r"$|\delta n_z / \delta n_y|$ ")
    ax2b.set_title(r"Warping fraction vs $\theta$")
    for ax in [ax2a, ax2b]:
        H.style_theta_axis(ax); ax.grid(alpha=0.25)
    fig2.suptitle(
        rf"$E_F={params.DEFAULT_EF}$ eV,  T={params.DEFAULT_T} K,  $j={j_2d:.0f}$ A/m",
        fontsize=11,
    )
    fig2.tight_layout()
    if save:
        fig2.savefig(FISHER_DIR / "fisher_02_spin_density_vs_theta.png", dpi=200)
        plt.close(fig2)


    # ── Fig 03: B_eff vs E_F ─────────────────────────────────────────────
    fig3, (ax3a, ax3b) = plt.subplots(1, 2, figsize=(12, 4.5))
    for T, col in zip(params.T_VALUES, colors):
        dn_y_arr = code_ef[T][:, 1] * 100 * scale   # nm⁻², varies with Ef
        dn_z_arr = code_ef[T][:, 2] * 100 * scale   # nm⁻², varies with Ef

        B_y   = np.array([b_eff(dy,  0.0, m_phi45)[1] for dy in dn_y_arr])
        B_tot = np.array([b_eff(dy,  dz,  m_phi45)[1]
                          for dy, dz in zip(dn_y_arr, dn_z_arr)])
        ax3a.plot(Ef_vals, B_y,   color=col, lw=2, label=f"T={T} K")
        ax3b.plot(Ef_vals, B_tot, color=col, lw=2, label=f"T={T} K")

    for ax in [ax3a, ax3b]:
        ax.axhline(0, color='gray', lw=0.8)
        ax.set_xlabel(r"$E_F$ (eV)")
        ax.set_ylabel(r"$B^\mathrm{FL}_\mathrm{eff}$ (mT)")
        ax.grid(alpha=0.25); ax.legend(fontsize=7)
    ax3a.set_title(r"$B_{eff}$ from $\langle S_y\rangle$ only")
    ax3b.set_title(r"$B_{eff}$ total  ($\langle S_y\rangle + \langle S_z\rangle$)")
    fig3.suptitle(
        rf"Fischer FL torque: $j_x={j_2d:.0f}$ A/m,  "
        rf"$\Delta_{{ex}}={params.FISHER_DEL_EX:.0f}$ meV,  "
        rf"$M_s={params.FISHER_M_S:.0e}$ A/m,  "
        rf"$t_{{FM}}={params.FISHER_T_FM*1e9:.0f}$ nm,  $\hat{{m}}$ at 45°",
        fontsize=10,
    )
    fig3.tight_layout()
    if save:
        fig3.savefig(FISHER_DIR / "fisher_03_torque_vs_Ef.png", dpi=200)
        plt.close(fig3)

    # ── Fig 04: B_eff vs j_x and vs φ_m ──────────────────────────────────
    j_vals = np.logspace(0, 4, 80)
    m_phis = np.linspace(0, 2 * np.pi, 200)

    B_vs_j = np.array([
        b_eff(dn_y(j), dn_y(j) * r_at_01, m_phi45)[1]
        for j in j_vals
    ])
    B_vs_m = np.array([
        b_eff(dn_y_ref, dn_z_at_01, phi)[1]
        for phi in m_phis
    ])

    fig4, (ax4a, ax4b) = plt.subplots(1, 2, figsize=(12, 4.5))

    ax4a.loglog(j_vals, B_vs_j, color='royalblue', lw=2)
    for j_ref, lab in [(10, '10'), (100, '100'), (1000, '1k'), (10000, '10k')]:
        b_ref = b_eff(dn_y(j_ref), dn_y(j_ref) * r_at_01, m_phi45)[1]
        ax4a.annotate(
            f"{b_ref:.3f} mT\n@ {lab} A/m",
            xy=(j_ref, b_ref), xytext=(j_ref * 1.5, b_ref * 0.55),
            fontsize=7, arrowprops=dict(arrowstyle='->', lw=0.8),
        )
    ax4a.annotate("slope 1  (linear)", xy=(0.55, 0.18), xycoords='axes fraction',
                  fontsize=8, color='navy')
    ax4a.set_xlabel(r"$j_x$ (A/m)"); ax4a.set_ylabel(r"$|B_\mathrm{eff}|$ (mT)")
    ax4a.set_title("Torque vs current density  (log–log)")
    ax4a.grid(alpha=0.3, which='both')

    ax4b.plot(m_phis, B_vs_m, color='crimson', lw=2)
    ax4b.axvline(np.pi / 4, color='gray',     ls='--', lw=1, label='45° (Fischer)')
    ax4b.axvline(np.pi / 2, color='steelblue', ls=':', lw=1, label='90°')
    H.style_theta_axis(ax4b, xlabel=r"$\hat{m}$ angle $\phi_m$ (rad)")
    ax4b.set_ylabel(r"$|B_\mathrm{eff}|$ (mT)")
    ax4b.set_title("Torque magnitude vs m̂ direction")
    ax4b.legend(fontsize=8); ax4b.grid(alpha=0.25)

    fig4.suptitle(
        rf"$E_F={params.DEFAULT_EF}$ eV,  T={params.DEFAULT_T} K,  "
        rf"$\Delta_{{ex}}={params.FISHER_DEL_EX:.0f}$ meV",
        fontsize=11,
    )
    fig4.tight_layout()
    if save:
        fig4.savefig(FISHER_DIR / "fisher_04_scaling.png", dpi=200)
        plt.close(fig4)
# figure 

    dn_y_efield = code_e_field[:, 1] * 100 * scale
    dn_z_efield = code_e_field[:, 2] * 100 * scale
    ratio_zy = np.where(np.abs(dn_y_efield) > 1e-20, dn_z_efield / dn_y_efield, 0.0)
    fig5, axes5 = plt.subplots(1, 3, figsize=(15, 4.5))
    ax_sy, ax_sz, ax_rat = axes5


    ax_sy.plot(E_field_vals, np.abs(dn_y_efield) * 1e3, lw=2)
    ax_sz.plot(E_field_vals, dn_z_efield * 1e3, lw=2)
    ax_rat.plot(E_field_vals, np.abs(ratio_zy), color=col, lw=2)

    ax_sy.set_ylabel(r"$|\delta n_y| \ (10^{-3}\ \mathrm{nm}^{-2})$ (log scale)")
    ax_sy.set_title(r"$\langle S_y\rangle_\mathrm{neq}$ — warped Hamiltonian ($\lambda$="
                    + rf"{params.LAMBDA:.0f} eV·Å³)")
    
    ax_sz.set_ylabel(r"$\delta n_z \ (10^{-3}\ \mathrm{nm}^{-2})$ (log scale)")
    ax_sz.set_title(r"$\langle S_z\rangle_\mathrm{neq}$ — warping correction")
    
    ax_rat.set_ylabel(r"$|\delta n_z / \delta n_y|$")
    ax_rat.set_title(r"Warping fraction $|S_z / S_y|$")
    for ax in axes5:
        ax.set_yscale('log')
        ax.set_xscale('log')
        ax.axhline(0, color='gray', lw=0.8)
        ax.set_xlabel(r"$Electric Field$ (V/m)")
        ax.grid(alpha=0.25)
    fig5.suptitle(
        rf"Fischer CISP: $j_x={j_2d:.0f}$ A/m, "
        rf"$v_F={params.vF_SI()/1e5:.2f}\times10^5$ m/s",
        fontsize=10,
    )
    fig5.tight_layout()
    if save:
        fig5.savefig(FISHER_DIR / "fisher_05_spin_density_vs_E_field.png", dpi=200)
        plt.close(fig5)



    # ── Summary ───────────────────────────────────────────────────────────
    b45 = b_eff(dn_y_at_01, dn_z_at_01, m_phi45)[1]
    summary = {
        'vF_SI':       params.vF_SI(),
        'j_ref':       j_2d,
        'dn_y_ref':    dn_y_ref,           # analytical Fischer (pure Dirac)
        'dn_y_at_01':  dn_y_at_01,         # warped code at reference point
        'dn_z_at_01':  dn_z_at_01,
        'r_at_01':     r_at_01,
        'B_45deg':     b45,
    }
    print()
    print("  ─── Fischer torque summary ───────────────────────────────────")
    print(f"  v_F                        = {summary['vF_SI']:.3e} m/s")
    print(f"  j_x reference              = {j_2d:.0f} A/m")
    print(f"  δn_y Fischer (pure Dirac)  = {dn_y_ref*1e3:.4f} × 10⁻³ nm⁻²")
    print(f"  δn_y warped  (Ef=0.1,300K) = {dn_y_at_01*1e3:.4f} × 10⁻³ nm⁻²")
    print(f"  δn_z warped  (Ef=0.1,300K) = {dn_z_at_01*1e3:.4f} × 10⁻³ nm⁻²")
    print(f"  |δn_z / δn_y|              = {abs(r_at_01)*100:.2f} %")
    print(f"  scale factor               = {scale:.4f}  (code → physical at ref point)")
    print(f"  B_eff (m̂ at 45°)           = {b45:.4f} mT")
    print(f"  B_eff @ 10 A/m             = {b45*10/j_2d:.4f} mT")
    print(f"  B_eff @ 1000 A/m           = {b45*1000/j_2d:.3f} mT")
    print("  ─────────────────────────────────────────────────────────────")

    return summary