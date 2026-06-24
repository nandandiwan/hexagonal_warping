"""
negf_kwant.py — NEGF simulation for 3D TI slab with kwant self-energies.

Zhang et al. 4-band model on a slab:
  - finite z  (Nz layers, surface states on top/bottom)
  - Fourier y → ky  (reduces 3D to 2D per ky slice)
  - real-space x  (transport direction, Nx device sites + semi-infinite leads)

Observable: current-induced spin polarization (CISP) on the top surface.
Leads: kwant semi-infinite chains (robust self-energies).
Device Green's function: recursive (block-tridiagonal) algorithm — O(Nx·nb³).
"""

import os
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
           "NUMEXPR_NUM_THREADS"):
    os.environ[_v] = "16"

import sys
import time
from pathlib import Path

import numpy as np
from scipy.sparse import csc_matrix, eye as speye
from scipy.sparse.linalg import splu
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ti_3d_ham import (onsite_4x4, hop_4x4, default_params,
                       kron, I2, sx, sy, sz, TyS0)
import kwant

_S4 = {"x": kron(I2, sx), "y": kron(I2, sy), "z": kron(I2, sz)}

OUT = Path(__file__).resolve().parent / "plots"


# ── Slab Hamiltonian blocks ──────────────────────────────────────────────

def slab_matrices(ky, Nz, params):
    """On-site H(ky) and x-hopping V for the ky-Fourier-transformed slab.

    Returns (H_on, V_hop, V2_hop).
    V2_hop is None when lambda_warp=0, otherwise the 2NN x-hop from warping.
    """
    n, dim = 4, 4 * Nz
    p = params
    H0 = onsite_4x4(p["C"], p["M0"], p["D_par"], p["D_z"], p["B_par"], p["B_z"])
    Hx = hop_4x4("x", p["A_par"], p["A_z"], p["D_par"], p["D_z"], p["B_par"], p["B_z"])
    Hy = hop_4x4("y", p["A_par"], p["A_z"], p["D_par"], p["D_z"], p["B_par"], p["B_z"])
    Hz = hop_4x4("z", p["A_par"], p["A_z"], p["D_par"], p["D_z"], p["B_par"], p["B_z"])

    on_k = H0 + Hy * np.exp(1j * ky) + Hy.conj().T * np.exp(-1j * ky)

    lam = p.get('lambda_warp', 0.0)
    V_warp_NN = 1j * lam * (2 - 3 * np.cos(ky)) * TyS0

    H_on = np.zeros((dim, dim), complex)
    V_hop = np.zeros((dim, dim), complex)
    for z in range(Nz):
        s = slice(n * z, n * (z + 1))
        H_on[s, s] = on_k
        V_hop[s, s] = Hx + V_warp_NN
    for z in range(Nz - 1):
        si = slice(n * z, n * (z + 1))
        sj = slice(n * (z + 1), n * (z + 2))
        H_on[si, sj] = Hz
        H_on[sj, si] = Hz.conj().T

    if lam == 0:
        return H_on, V_hop, None

    V2_hop = np.zeros((dim, dim), complex)
    V_warp_2NN = 0.5j * lam * TyS0
    for z in range(Nz):
        s = slice(n * z, n * (z + 1))
        V2_hop[s, s] = V_warp_2NN
    return H_on, V_hop, V2_hop


# ── kwant lead system ────────────────────────────────────────────────────

def _make_lead_system(H_on, V_hop, V2_hop=None):
    """Minimal kwant system with L/R leads for self-energy extraction.

    When V2_hop is given (hexagonal warping), uses a doubled unit cell.
    """
    if V2_hop is None:
        nb = H_on.shape[0]
        lat = kwant.lattice.chain(norbs=nb)
        syst = kwant.Builder()
        syst[lat(0)] = H_on
        syst[lat(1)] = H_on
        syst[lat(1), lat(0)] = V_hop
        lead = kwant.Builder(kwant.TranslationalSymmetry((-1,)))
        lead[lat(0)] = H_on
        lead[lat(0), lat(-1)] = V_hop
        syst.attach_lead(lead)
        syst.attach_lead(lead.reversed())
        return syst.finalized()

    nb = H_on.shape[0]
    nb2 = 2 * nb
    lat = kwant.lattice.chain(norbs=nb2)
    dag = lambda A: A.conj().T
    H_sc = np.zeros((nb2, nb2), complex)
    H_sc[:nb, :nb] = H_on
    H_sc[nb:, nb:] = H_on
    H_sc[:nb, nb:] = V_hop
    H_sc[nb:, :nb] = dag(V_hop)
    V_sc = np.zeros((nb2, nb2), complex)
    V_sc[:nb, :nb] = V2_hop
    V_sc[nb:, :nb] = V_hop
    V_sc[nb:, nb:] = V2_hop
    syst = kwant.Builder()
    syst[lat(0)] = H_sc
    syst[lat(1)] = H_sc
    syst[lat(1), lat(0)] = V_sc
    lead = kwant.Builder(kwant.TranslationalSymmetry((-1,)))
    lead[lat(0)] = H_sc
    lead[lat(0), lat(-1)] = V_sc
    syst.attach_lead(lead)
    syst.attach_lead(lead.reversed())
    return syst.finalized()


# ── Recursive Green's function (block-tridiagonal) ─────────────────────

def _rgf_glesser_blocks(E_c, H_dev, Sig_L, Sig_R, occ_L, occ_R, nb,
                         Sig_less_L=None, Sig_less_R=None):
    """Block-RGF for a 1D chain with block size nb.

    Parameters
    ----------
    E_c : complex  — energy on the retarded sheet (E + i*eta)
    H_dev : (dim, dim) — full device Hamiltonian (block-tridiagonal)
    Sig_L, Sig_R : (nb, nb) — lead self-energies
    occ_L, occ_R : float or None — Fermi occupation weights (ignored if Sig_less given)
    nb : int — block size (= 4*Nz)
    Sig_less_L, Sig_less_R : optional pre-computed lesser self-energies (for Floquet)

    Returns list of Nx (nb × nb) G^< diagonal blocks.
    """
    dim = H_dev.shape[0]
    Nx = dim // nb
    I_nb = np.eye(nb, dtype=complex)
    dag = lambda A: A.conj().T

    # Extract blocks of (E·I − H)
    D = []
    U = []
    for i in range(Nx):
        s = slice(i * nb, (i + 1) * nb)
        D.append(E_c * I_nb - H_dev[s, s])
        if i < Nx - 1:
            s2 = slice((i + 1) * nb, (i + 2) * nb)
            U.append(-H_dev[s, s2])

    if Sig_less_L is None:
        Gam_L = 1j * (Sig_L - dag(Sig_L))
        Sig_less_L = 1j * occ_L * Gam_L
    if Sig_less_R is None:
        Gam_R = 1j * (Sig_R - dag(Sig_R))
        Sig_less_R = 1j * occ_R * Gam_R

    # ── Forward sweep (left-connected GF) ──
    g_R = [None] * Nx
    g_less = [None] * Nx

    g_R[0] = np.linalg.inv(D[0] - Sig_L)
    g_less[0] = g_R[0] @ Sig_less_L @ dag(g_R[0])

    for i in range(1, Nx):
        L = dag(U[i - 1])
        sig_r = L @ g_R[i - 1] @ U[i - 1]
        g_R[i] = np.linalg.inv(D[i] - sig_r)
        sig_less = L @ g_less[i - 1] @ U[i - 1]
        g_less[i] = g_R[i] @ sig_less @ dag(g_R[i])

    # ── Backward sweep (full GF) ──
    G_R = [None] * Nx
    G_less = [None] * Nx

    L_last = dag(U[-1])
    sig_eff = L_last @ g_R[-2] @ U[-1]
    G_R[-1] = np.linalg.inv(D[-1] - Sig_R - sig_eff)

    sig_less_eff = L_last @ g_less[-2] @ U[-1]
    G_less[-1] = G_R[-1] @ (Sig_less_R + sig_less_eff) @ dag(G_R[-1])

    for i in range(Nx - 2, -1, -1):
        prop = g_R[i] @ U[i] @ G_R[i + 1] @ dag(U[i])
        G_R[i] = g_R[i] + prop @ g_R[i]
        g_R_dag = dag(g_R[i])
        G_less[i] = (g_less[i]
                     + prop @ g_less[i]
                     + g_less[i] @ dag(prop)
                     + g_R[i] @ U[i] @ G_less[i + 1] @ dag(U[i]) @ g_R_dag)

    return G_less


# ── NEGF core ────────────────────────────────────────────────────────────

def _fermi(E, mu, kBT):
    x = np.clip((E - mu) / max(kBT, 1e-10), -200, 200)
    return 1.0 / (1.0 + np.exp(x))


def negf_spin_at_ky(ky, Nz, Nx, V_drop, E_grid, E_F, kBT, params,
                    n_surf=2, eta=5e-4):
    """Spin integrand at one ky using RGF.

    Fermi weights are folded into G^< directly.
    When hexagonal warping is active (lambda_warp != 0), uses super-blocking
    to handle 2NN x-hops: pairs of x-sites become one RGF block.
    Returns array (N_E, 3) — spin summed over top n_surf z-layers, all x-slices.
    """
    H_on, V_hop, V2_hop = slab_matrices(ky, Nz, params)
    fsyst = _make_lead_system(H_on, V_hop, V2_hop)

    nb = 4 * Nz
    dag = lambda A: A.conj().T
    warped = V2_hop is not None

    if warped:
        if Nx % 2 == 1:
            Nx += 1
        N_blocks = Nx // 2
        bs = 2 * nb
        I_nb = np.eye(nb)
        H_dev = np.zeros((N_blocks * bs, N_blocks * bs), complex)
        V_dag = dag(V_hop)
        V2_dag = dag(V2_hop)
        for I in range(N_blocks):
            xa, xb = 2 * I, 2 * I + 1
            phi_a = V_drop * (0.5 - xa / max(Nx - 1, 1))
            phi_b = V_drop * (0.5 - xb / max(Nx - 1, 1))
            sa = slice(I * bs, I * bs + nb)
            sb = slice(I * bs + nb, (I + 1) * bs)
            H_dev[sa, sa] = H_on + phi_a * I_nb
            H_dev[sb, sb] = H_on + phi_b * I_nb
            H_dev[sa, sb] = V_hop
            H_dev[sb, sa] = V_dag
            if I < N_blocks - 1:
                sa2 = slice((I + 1) * bs, (I + 1) * bs + nb)
                sb2 = slice((I + 1) * bs + nb, (I + 2) * bs)
                H_dev[sa, sa2] = V2_hop
                H_dev[sb, sa2] = V_hop
                H_dev[sb, sb2] = V2_hop
                H_dev[sa2, sa] = V2_dag
                H_dev[sa2, sb] = V_dag
                H_dev[sb2, sb] = V2_dag
    else:
        N_blocks = Nx
        bs = nb
        I_nb = np.eye(nb)
        V_dag = dag(V_hop)
        H_dev = np.zeros((Nx * nb, Nx * nb), complex)
        for i in range(Nx):
            s = slice(i * nb, (i + 1) * nb)
            phi = V_drop * (0.5 - i / max(Nx - 1, 1))
            H_dev[s, s] = H_on + phi * I_nb
            if i < Nx - 1:
                s2 = slice((i + 1) * nb, (i + 2) * nb)
                H_dev[s, s2] = V_hop
                H_dev[s2, s] = V_dag

    mu_L = E_F + V_drop / 2
    mu_R = E_F - V_drop / 2
    top_z0 = Nz - n_surf
    N_E = len(E_grid)
    spin_int = np.zeros((N_E, 3))

    for iE, E in enumerate(E_grid):
        E_c = E + 1j * eta
        Sig_L = fsyst.leads[0].selfenergy(E_c)
        Sig_R = fsyst.leads[1].selfenergy(E_c)

        f_L = _fermi(E, mu_L, kBT)
        f_R = _fermi(E, mu_R, kBT)

        G_less = _rgf_glesser_blocks(
            E_c, H_dev, Sig_L, Sig_R, f_L, f_R, bs)

        if warped:
            for I in range(N_blocks):
                block = G_less[I]
                for offset in (0, nb):
                    for iz in range(top_z0, Nz):
                        r = offset + 4 * iz
                        ss = slice(r, r + 4)
                        sub = block[ss, ss]
                        for ia, a in enumerate("xyz"):
                            spin_int[iE, ia] += np.trace(_S4[a] @ sub).imag
        else:
            for ix in range(Nx):
                block = G_less[ix]
                for iz in range(top_z0, Nz):
                    r = 4 * iz
                    ss = slice(r, r + 4)
                    sub = block[ss, ss]
                    for ia, a in enumerate("xyz"):
                        spin_int[iE, ia] += np.trace(_S4[a] @ sub).imag

    return spin_int


# ── Kubo linear response (no scalar potential, PH-symmetric) ────────────
#
# δ⟨S_y⟩ = V/(4π) ∫ dE (-∂f/∂E) ∫ dky/(2π) Tr_surf[S_y G^R (Γ_L - Γ_R) G^A]
#
# Only the occupation changes (δΣ^<), NOT the Hamiltonian.
# Equilibrium G^R has no scalar potential → PH symmetry exact.

def _neg_df_dE(E, E_F, kBT):
    x = np.clip((E - E_F) / max(kBT, 1e-10), -200, 200)
    ex = np.exp(x)
    return ex / (max(kBT, 1e-10) * (1.0 + ex) ** 2)


def kubo_cisp_at_ky(ky, Nz, Nx, E_grid, E_F, kBT, params,
                     n_surf=2, eta=5e-4):
    """Kubo kernel at one ky.

    Uses sparse LU: since Γ_L-Γ_R is nonzero only at the two lead
    blocks, we solve for just 2×nb columns of G^R instead of inverting.

    Returns (spin_kernel, transmission):
      spin_kernel : (N_E, 3) — Tr_surf[S_y G^R (Γ_L-Γ_R) G^A]
      transmission: (N_E,)   — Tr[Γ_L G^R Γ_R G^A]
    Neither is weighted by (-∂f/∂E).
    """
    H_on, V_hop, _ = slab_matrices(ky, Nz, params)
    fsyst = _make_lead_system(H_on, V_hop)

    nb = 4 * Nz
    dim = Nx * nb
    dag = lambda A: A.conj().T
    V_dag = dag(V_hop)

    H_dev = np.zeros((dim, dim), complex)
    for i in range(Nx):
        s = slice(i * nb, (i + 1) * nb)
        H_dev[s, s] = H_on
        if i < Nx - 1:
            s2 = slice((i + 1) * nb, (i + 2) * nb)
            H_dev[s, s2] = V_hop
            H_dev[s2, s] = V_dag

    rhs_L = np.zeros((dim, nb), complex)
    rhs_L[:nb, :] = np.eye(nb)
    rhs_R = np.zeros((dim, nb), complex)
    rhs_R[-nb:, :] = np.eye(nb)

    top_z0 = Nz - n_surf
    N_E = len(E_grid)
    spin_k = np.zeros((N_E, 3))
    trans = np.zeros(N_E)

    for iE, E in enumerate(E_grid):
        E_c = E + 1j * eta
        Sig_L = fsyst.leads[0].selfenergy(E_c)
        Sig_R = fsyst.leads[1].selfenergy(E_c)
        Gam_L = 1j * (Sig_L - dag(Sig_L))
        Gam_R = 1j * (Sig_R - dag(Sig_R))

        A = E_c * np.eye(dim) - H_dev
        A[:nb, :nb] -= Sig_L
        A[-nb:, -nb:] -= Sig_R

        lu = splu(csc_matrix(A))
        X_L = lu.solve(rhs_L)
        X_R = lu.solve(rhs_R)

        # Transmission: T = Tr[Γ_L G^R Γ_R G^A]
        #   = Tr[Γ_L · X_R[0:nb,:] · Γ_R · X_R[0:nb,:]†]
        xR0 = X_R[:nb, :]
        trans[iE] = np.trace(Gam_L @ xR0 @ Gam_R @ dag(xR0)).real

        # CISP kernel on top surface
        for ix in range(Nx):
            for iz in range(top_z0, Nz):
                r = ix * nb + 4 * iz
                xL = X_L[r:r+4, :]
                xR_s = X_R[r:r+4, :]
                K_blk = xL @ Gam_L @ dag(xL) - xR_s @ Gam_R @ dag(xR_s)
                for ia, a in enumerate("xyz"):
                    spin_k[iE, ia] += np.trace(_S4[a] @ K_blk).real

    return spin_k, trans


def run_kubo(E_F, V_drop=0.05, Nz=10, Nx=15, N_ky=41,
             ky_max=1.0, E_range=None, N_E=81, params=None, T=300,
             n_surf=2, eta=5e-4, verbose=True):
    """Kubo linear-response CISP and conductance (no scalar potential).

    Returns dict {'x','y','z','G'} — CISP at the given V_drop, and
    the Landauer conductance G (in units of e²/h).
    """
    if params is None:
        params = default_params
    if E_range is None:
        E_range = (-0.35, 0.35)

    kBT = T * 8.617e-5
    E_grid = np.linspace(*E_range, N_E)
    ky_grid = np.linspace(-ky_max, ky_max, N_ky)
    dky = ky_grid[1] - ky_grid[0]

    nf_weight = np.array([_neg_df_dE(E, E_F, kBT) for E in E_grid])

    spin = np.zeros(3)
    cond = 0.0

    for iky, ky in enumerate(ky_grid):
        kernel, trans = kubo_cisp_at_ky(
            ky, Nz, Nx, E_grid, E_F, kBT, params, n_surf, eta)

        for ia in range(3):
            weighted = kernel[:, ia] * nf_weight
            spin[ia] += np.trapezoid(weighted, E_grid) / (4 * np.pi) * dky / (2 * np.pi)

        cond += np.trapezoid(trans * nf_weight, E_grid) * dky / (2 * np.pi)

        if verbose and (iky % 5 == 0 or iky == N_ky - 1):
            print(f"  ky {iky + 1}/{N_ky}  Sy_cum={spin[1] * V_drop:.4e}"
                  f"  G_cum={cond:.4e}")

    spin *= V_drop
    return {"x": spin[0], "y": spin[1], "z": spin[2], "G": cond}


def sweep_EF_kubo(EF_vals, V_drop=0.05, save=True, **kw):
    """CISP and conductance vs E_F using Kubo formula."""
    verbose = kw.pop("verbose", True)

    results = np.zeros((len(EF_vals), 3))
    G_vals = np.zeros(len(EF_vals))
    for i, EF in enumerate(EF_vals):
        if verbose:
            print(f"E_F={EF:.3f} ({i + 1}/{len(EF_vals)})...")
        s = run_kubo(EF, verbose=False, **kw)
        results[i] = [s[a] for a in "xyz"]
        G_vals[i] = s["G"]
        if verbose:
            print(f"  Sy={results[i, 1]:.3e}  G={G_vals[i]:.3e}  "
                  f"Sy/G={results[i, 1] / G_vals[i]:.3e}")

    if save:
        OUT.mkdir(parents=True, exist_ok=True)
        np.savez(OUT / "cisp_vs_EF_kubo.npz",
                 EF=EF_vals, cisp=results, G=G_vals,
                 V_drop=kw.get("V_drop", 0.05))
    return results, G_vals


def plot_cisp_vs_EF_kubo(EF_vals, cisp, G_vals, V_drop, Nz, Nx):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # Sy/V (odd) | G (even) | Sy/G (should be even)
    ax = axes[0]
    ax.plot(EF_vals, cisp[:, 1], "o-", lw=2)
    ax.set_xlabel(r"$E_F$ (eV)")
    ax.set_ylabel(r"CISP $S_y$ (per $V_\mathrm{drop}$)")
    ax.set_title(r"$S_y$ (odd in $E_F$)")
    ax.axhline(0, color="gray", lw=0.5)
    ax.grid(alpha=0.25)

    ax = axes[1]
    ax.plot(EF_vals, G_vals, "s-", lw=2, color="C1")
    ax.set_xlabel(r"$E_F$ (eV)")
    ax.set_ylabel(r"$G$ ($e^2/h$)")
    ax.set_title(r"Conductance (even in $E_F$)")
    ax.axhline(0, color="gray", lw=0.5)
    ax.grid(alpha=0.25)

    ax = axes[2]
    ratio = cisp[:, 1] / G_vals
    ax.plot(EF_vals, ratio, "D-", lw=2, color="C2")
    ax.set_xlabel(r"$E_F$ (eV)")
    ax.set_ylabel(r"$S_y / G$")
    ax.set_title(r"Edelstein per current (even in $E_F$)")
    ax.axhline(0, color="gray", lw=0.5)
    ax.grid(alpha=0.25)

    fig.suptitle(rf"Kubo CISP (Nz={Nz}, Nx={Nx})")
    fig.tight_layout()
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / "cisp_vs_EF_kubo.png", dpi=150)
    plt.close(fig)
    print(f"Saved: {OUT / 'cisp_vs_EF_kubo.png'}")


def run_negf(E_F, V_drop, Nz=10, Nx=15, N_ky=41,
             ky_max=1.0, E_range=None, N_E=81, params=None, T=300,
             n_surf=2, eta=5e-4, verbose=True):
    """Compute top-surface spin density via NEGF (RGF algorithm).

    ky_max: restrict ky integration to [-ky_max, ky_max] (surface states
            are near ky=0, so full BZ is wasteful).

    Returns dict {'x': float, 'y': float, 'z': float}.
    """
    if params is None:
        params = default_params
    if E_range is None:
        E_range = (-0.35, 0.35)

    kBT = T * 8.617e-5
    E_grid = np.linspace(*E_range, N_E)
    ky_grid = np.linspace(-ky_max, ky_max, N_ky)
    dky = ky_grid[1] - ky_grid[0]

    spin = np.zeros(3)

    for iky, ky in enumerate(ky_grid):
        integrand = negf_spin_at_ky(
            ky, Nz, Nx, V_drop, E_grid, E_F, kBT, params, n_surf, eta)

        for ia in range(3):
            spin[ia] += np.trapezoid(integrand[:, ia], E_grid) / (2 * np.pi) * dky / (2 * np.pi)

        if verbose and (iky % 5 == 0 or iky == N_ky - 1):
            print(f"  ky {iky + 1}/{N_ky}  Sy_cum={spin[1]:.4e}")

    return {"x": spin[0], "y": spin[1], "z": spin[2]}


# ── Sweeps ───────────────────────────────────────────────────────────────

def sweep_Vdrop(V_drop_vals, E_F=0.1, save=True, **kw):
    """CISP vs bias (analogous to E-field sweep).

    Returns array (N_V, 3) — columns Sx, Sy, Sz.
    """
    verbose = kw.pop("verbose", True)

    if verbose:
        print("Computing equilibrium reference (V=0)...")
    s_eq = run_negf(E_F, 0.0, verbose=False, **kw)

    results = np.zeros((len(V_drop_vals), 3))
    for i, V in enumerate(V_drop_vals):
        if verbose:
            print(f"V_drop={V:.4f} ({i + 1}/{len(V_drop_vals)})...")
        s = run_negf(E_F, V, verbose=False, **kw)
        results[i] = [s[a] - s_eq[a] for a in "xyz"]
        if verbose:
            print(f"  Sx={results[i, 0]:.3e}  Sy={results[i, 1]:.3e}  "
                  f"Sz={results[i, 2]:.3e}")

    if save:
        OUT.mkdir(parents=True, exist_ok=True)
        np.savez(OUT / "cisp_vs_Vdrop.npz",
                 V_drop=V_drop_vals, cisp=results, E_F=E_F)
    return results


def sweep_EF(EF_vals, V_drop=0.05, save=True, **kw):
    """CISP vs Fermi energy.

    Returns array (N_EF, 3) — columns Sx, Sy, Sz.
    """
    verbose = kw.pop("verbose", True)

    results = np.zeros((len(EF_vals), 3))
    for i, EF in enumerate(EF_vals):
        if verbose:
            print(f"E_F={EF:.3f} ({i + 1}/{len(EF_vals)})...")
        s_bias = run_negf(EF, V_drop, verbose=False, **kw)
        s_eq = run_negf(EF, 0.0, verbose=False, **kw)
        results[i] = [s_bias[a] - s_eq[a] for a in "xyz"]
        if verbose:
            print(f"  Sx={results[i, 0]:.3e}  Sy={results[i, 1]:.3e}  "
                  f"Sz={results[i, 2]:.3e}")

    if save:
        OUT.mkdir(parents=True, exist_ok=True)
        np.savez(OUT / "cisp_vs_EF.npz",
                 EF=EF_vals, cisp=results, V_drop=V_drop)
    return results


# ── Plotting ─────────────────────────────────────────────────────────────

def plot_slab_bands(Nz=12, params=None, kmax=1.0, Nk=201):
    """Plot slab band structure to verify surface states exist."""
    if params is None:
        params = default_params
    from surface_ham import slab_bloch_H_kxky

    kx = np.linspace(-kmax, kmax, Nk)
    bands = np.zeros((Nk, 4 * Nz))
    for i, k in enumerate(kx):
        bands[i] = np.linalg.eigvalsh(slab_bloch_H_kxky(k, 0.0, Nz, params))

    fig, ax = plt.subplots(figsize=(7, 5))
    for b in range(4 * Nz):
        ax.plot(kx, bands[:, b], "b-", lw=0.5, alpha=0.4)
    ax.set_xlabel(r"$k_x$ (1/a)")
    ax.set_ylabel("Energy (eV)")
    ax.set_title(f"Slab band structure (Nz={Nz})")
    ax.set_ylim(-0.5, 0.5)
    ax.axhline(0, color="gray", lw=0.5)
    ax.grid(alpha=0.2)
    fig.tight_layout()
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / "slab_bands.png", dpi=150)
    plt.close(fig)
    print(f"Saved: {OUT / 'slab_bands.png'}")


def plot_cisp_vs_Vdrop(V_vals, cisp, E_F, Nz, Nx):
    labels = [r"$S_x$", r"$S_y$", r"$S_z$"]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for i, (ax, lab) in enumerate(zip(axes, labels)):
        ax.plot(V_vals, cisp[:, i], "o-", lw=2)
        ax.set_xlabel(r"$V_\mathrm{drop}$ (eV)")
        ax.set_ylabel(f"CISP {lab}")
        ax.set_title(lab)
        ax.axhline(0, color="gray", lw=0.5)
        ax.grid(alpha=0.25)
    fig.suptitle(rf"CISP vs $V_{{\mathrm{{drop}}}}$  (NEGF, $E_F$={E_F}, "
                 f"Nz={Nz}, Nx={Nx})")
    fig.tight_layout()
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / "cisp_vs_Vdrop.png", dpi=150)
    plt.close(fig)
    print(f"Saved: {OUT / 'cisp_vs_Vdrop.png'}")


def plot_cisp_vs_EF(EF_vals, cisp, V_drop, Nz, Nx):
    labels = [r"$S_x$", r"$S_y$", r"$S_z$"]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for i, (ax, lab) in enumerate(zip(axes, labels)):
        ax.plot(EF_vals, cisp[:, i], "o-", lw=2)
        ax.set_xlabel(r"$E_F$ (eV)")
        ax.set_ylabel(f"CISP {lab}")
        ax.set_title(lab)
        ax.axhline(0, color="gray", lw=0.5)
        ax.grid(alpha=0.25)
    fig.suptitle(rf"CISP vs $E_F$  (NEGF, $V_{{\mathrm{{drop}}}}$={V_drop}, "
                 f"Nz={Nz}, Nx={Nx})")
    fig.tight_layout()
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / "cisp_vs_EF.png", dpi=150)
    plt.close(fig)
    print(f"Saved: {OUT / 'cisp_vs_EF.png'}")


# ── Main ─────────────────────────────────────────────────────────────────

def main():
    Nz, Nx = 8, 10
    N_ky, N_E = 31, 81
    T = 300
    params = default_params
    common = dict(Nz=Nz, Nx=Nx, N_ky=N_ky, ky_max=0.6,
                  N_E=N_E, params=params, T=T, n_surf=1, eta=5e-4)

    print("=" * 60)
    print("NEGF 3D TI — CISP simulation (kwant self-energies)")
    print("=" * 60)

    # 1. Slab band structure
    print("\n--- Slab band structure ---")
    plot_slab_bands(Nz=Nz, params=params)

    # 2. Single-point test
    print("\n--- Single-point test (E_F=0.1, V_drop=0.05) ---")
    t0 = time.time()
    s = run_negf(0.1, 0.05, **common)
    print(f"  Sx={s['x']:.4e}  Sy={s['y']:.4e}  Sz={s['z']:.4e}")
    print(f"  Time: {time.time() - t0:.1f}s")

    # 3. CISP vs V_drop
    print("\n--- CISP vs V_drop ---")
    V_vals = np.array([0.005, 0.01, 0.02, 0.04, 0.06, 0.08, 0.10, 0.12])
    t0 = time.time()
    cisp_V = sweep_Vdrop(V_vals, E_F=0.1, **common)
    print(f"  Total time: {time.time() - t0:.1f}s")
    plot_cisp_vs_Vdrop(V_vals, cisp_V, 0.1, Nz, Nx)

    # 4. CISP vs E_F
    print("\n--- CISP vs E_F ---")
    EF_vals = np.linspace(0.02, 0.25, 8)
    t0 = time.time()
    cisp_EF = sweep_EF(EF_vals, V_drop=0.05, **common)
    print(f"  Total time: {time.time() - t0:.1f}s")
    plot_cisp_vs_EF(EF_vals, cisp_EF, 0.05, Nz, Nx)

    print("\nDone!")


if __name__ == "__main__":
    main()
