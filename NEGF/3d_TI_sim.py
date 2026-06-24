from __future__ import annotations

import os
import sys
import time
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from scipy.interpolate import interp1d

# Ensure Eigen headers are discoverable for optional C++ extension build
for _cand in (
    "/usr/include/eigen3",
    "/usr/local/include/eigen3",
    str(Path.home() / ".local" / "eigen-src" / "eigen-3.4.0"),
):
    if os.path.isfile(f"{_cand}/Eigen/Core"):
        os.environ.setdefault("EIGEN_INCLUDE", _cand)
        break

# Ensure local negf package is on the path
_here = Path().resolve()
for base in [_here, *_here.parents]:
    if (base / "negf").is_dir():
        sys.path.insert(0, str(base))
        break
    if (base / "src" / "negf").is_dir():
        sys.path.insert(0, str(base / "src"))
        break
else:
    raise ModuleNotFoundError("Cannot locate the negf package.")

from negf.gf import recursive_greens_functions as rgf
from negf import sancho_rubio_iterative_greens_function
from ti_3d_ham import build_slice_3DTI, default_params


# ─────────────────────────── constants ────────────────────────────

KT = 0.025


def fermi(E, mu, kT=KT):
    return 1.0 / (1.0 + np.exp(np.clip((E - mu) / kT, -200, 200)))


# ─────────────────────────── device build ─────────────────────────

def build_device_H(H_slice, V_slice, Nx, V_drop=0.0):
    """Linear electrostatic ramp inside the device.
    phi_0 = +V_drop/2, phi_{Nx-1} = -V_drop/2."""
    nb = H_slice.shape[0]
    H = np.zeros((Nx * nb, Nx * nb), dtype=complex)
    denom = max(Nx - 1, 1)
    I_nb = np.eye(nb)
    Vd = V_slice.conj().T
    for i in range(Nx):
        s = slice(i * nb, (i + 1) * nb)
        phi_i = V_drop * (0.5 - i / denom)
        H[s, s] = H_slice + phi_i * I_nb
        if i < Nx - 1:
            s2 = slice((i + 1) * nb, (i + 2) * nb)
            H[s, s2] = V_slice
            H[s2, s] = Vd
    return H


# ─────────────────────────── spin operators (sparse) ──────────────
#
# For per-site σ ops in the basis (orbital τ ⊗ spin σ), the nonzeros
# within one 4-d site block are known analytically. We precompute the
# (row, col, value) triples for each surface projection so the trace
# Tr[G_xi @ S] becomes a sparse sum instead of a dense matmul.
#
# I_τ ⊗ σ_x: (0,1, 1)  (1,0, 1)  (2,3, 1)  (3,2, 1)
# I_τ ⊗ σ_y: (0,1,-i)  (1,0, i)  (2,3,-i)  (3,2, i)
# I_τ ⊗ σ_z: (0,0, 1)  (1,1,-1)  (2,2, 1)  (3,3,-1)

_SITE_NNZ = {
    'x': [(0, 1, 1+0j), (1, 0, 1+0j), (2, 3, 1+0j), (3, 2, 1+0j)],
    'y': [(0, 1, -1j),  (1, 0, +1j),  (2, 3, -1j),  (3, 2, +1j)],
    'z': [(0, 0, 1+0j), (1, 1, -1+0j), (2, 2, 1+0j), (3, 3, -1+0j)],
}


def _build_spin_indices(Ny, Nz, n_surf_layers):
    """Pre-compute (rows, cols, vals) of σ_a restricted to top/bot/all sites,
    for a single slice. Used by the sparse trace."""
    n_per_site = 4
    n_sites = Ny * Nz

    masks = {
        'top': set(yi * Nz + zi
                   for yi in range(Ny)
                   for zi in range(Nz - n_surf_layers, Nz)),
        'bot': set(yi * Nz + zi
                   for yi in range(Ny)
                   for zi in range(n_surf_layers)),
        'all': set(range(n_sites)),
    }

    out = {}
    for surf, sites in masks.items():
        out[surf] = {}
        for a, nnz in _SITE_NNZ.items():
            rows, cols, vals = [], [], []
            for site in sites:
                base = site * n_per_site
                for (di, dj, v) in nnz:
                    rows.append(base + di)
                    cols.append(base + dj)
                    vals.append(v)
            out[surf][a] = (
                np.array(rows, dtype=np.int64),
                np.array(cols, dtype=np.int64),
                np.array(vals, dtype=complex),
            )
    return out


def _trace_sparse(G, rows, cols, vals):
    """Tr[G @ S] where S has nonzeros vals[k] at (rows[k], cols[k]).
       Tr[G @ S] = sum_k vals[k] * G[cols[k], rows[k]].
       For Hermitian observables, the imaginary part is what we want."""
    return np.sum(vals * G[cols, rows])


# ─────────────────────────── workers ──────────────────────────────

def _self_energy_at_E_unshifted(args):
    """Self-energy of the un-tilted lead at energy E."""
    E, H_slice, V_slice, eta = args
    sigL = sancho_rubio_iterative_greens_function(
        E, V_slice, H_slice, V_slice.conj().T, damp=eta)
    sigR = sancho_rubio_iterative_greens_function(
        E, V_slice.conj().T, H_slice, V_slice, damp=eta)
    return sigL, sigR


_state = {}


def _init_worker(H_slice, V_slice, eta, nb, Nx, V_drop,
                 Ny, Nz, n_surf_layers):
    _state['eta']      = eta
    _state['Nx']       = Nx
    _state['H_dev']    = build_device_H(H_slice, V_slice, Nx, V_drop=V_drop)
    _state['spin_idx'] = _build_spin_indices(Ny, Nz, n_surf_layers)


def _spectral_spin_at_E(args):
    """Compute spin-weighted spectral functions of L- and R-injected
    electrons at energy E using the sparse trace optimization."""
    E_idx, E, sigL, sigR = args
    H_dev    = _state['H_dev']
    eta      = _state['eta']
    Nx       = _state['Nx']
    spin_idx = _state['spin_idx']

    resL = rgf._recursive_inverse(
        E, H_dev, sigL, sigR,
        compute_lesser=True,
        occ_left=1.0, occ_right=0.0,
        eta=eta, return_trace=True,
    )
    Gl_L = resL[6]

    resR = rgf._recursive_inverse(
        E, H_dev, sigL, sigR,
        compute_lesser=True,
        occ_left=0.0, occ_right=1.0,
        eta=eta, return_trace=True,
    )
    Gl_R = resR[6]

    out_L = {}
    out_R = {}
    for surf, comp in spin_idx.items():
        sL = {'x': 0.0, 'y': 0.0, 'z': 0.0}
        sR = {'x': 0.0, 'y': 0.0, 'z': 0.0}
        for xi in range(Nx):
            GL = Gl_L[xi]
            GR = Gl_R[xi]
            for a in 'xyz':
                rows, cols, vals = comp[a]
                sL[a] += _trace_sparse(GL, rows, cols, vals).imag
                sR[a] += _trace_sparse(GR, rows, cols, vals).imag
        out_L[surf] = (sL['x'], sL['y'], sL['z'])
        out_R[surf] = (sR['x'], sR['y'], sR['z'])
    return E_idx, out_L, out_R


# ─────────────────────────── main driver ──────────────────────────

def run_negf_3dti(
    Nx=10, Ny=6, Nz=10,
    params=None,
    E_grid=None,
    EF_grid=None,
    V_drop_grid=None,
    eta=5e-3,
    n_workers=64,
    n_surf_layers=3,
    verbose=True,
):
    """Option-B NEGF: continuous potential ramp across the whole infinite
    system. Lead Sigma is interpolated from a single un-tilted computation
    on a widened auxiliary grid, and the per-energy spectral spin uses a
    sparse trace for σ_x, σ_y, σ_z."""

    if params      is None: params      = default_params
    if E_grid      is None: E_grid      = EF_grid = np.linspace(-0.15, 0.15, 31)
    if EF_grid     is None: EF_grid     = np.linspace(-0.3, 0.3, 21)
    if V_drop_grid is None: V_drop_grid = np.array([0.0, 0.05, 0.10, 0.20])

    H_slice, V_slice = build_slice_3DTI(Ny, Nz, params, periodic_y=True)
    nb = H_slice.shape[0]

    if verbose:
        print(f"slice dim {nb}, device dim {Nx*nb}, E_grid {len(E_grid)} pts")
        print(f"V_drop grid: {V_drop_grid}")

    # --- Un-tilted Sigma once on a padded grid; interpolate per V_drop. ---
    phi_max = float(np.max(np.abs(V_drop_grid))) / 2.0
    dE_aux  = E_grid[1] - E_grid[0]
    n_pad   = int(np.ceil(phi_max / dE_aux)) + 4
    E_aux   = np.concatenate([
        E_grid[0]  - dE_aux * np.arange(n_pad, 0, -1),
        E_grid,
        E_grid[-1] + dE_aux * np.arange(1, n_pad + 1),
    ])

    t0 = time.time()
    args_se = [(E, H_slice, V_slice, eta) for E in E_aux]
    with ProcessPoolExecutor(max_workers=n_workers) as ex:
        results_se = list(ex.map(_self_energy_at_E_unshifted, args_se, chunksize=8))
    SL_aux = np.array([r[0] for r in results_se])  # (nE_aux, nb, nb)
    SR_aux = np.array([r[1] for r in results_se])
    if verbose:
        print(f"  un-tilted Σ on {len(E_aux)} pts  "
              f"({time.time()-t0:.1f}s)")

    SL_interp = interp1d(E_aux, SL_aux, axis=0, kind='cubic',
                         assume_sorted=True, copy=False)
    SR_interp = interp1d(E_aux, SR_aux, axis=0, kind='cubic',
                         assume_sorted=True, copy=False)

    nE = len(E_grid)
    nV = len(V_drop_grid)
    nF = len(EF_grid)

    surf_keys = ('top', 'bot', 'all')
    spec = {iV: {surf: {lead: {a: np.zeros(nE) for a in 'xyz'}
                        for lead in 'LR'}
                 for surf in surf_keys}
            for iV in range(nV)}

    ctx = mp.get_context('fork')

    for iV, V_drop in enumerate(V_drop_grid):
        phi_L = +V_drop / 2.0
        phi_R = -V_drop / 2.0

        SL_arr = SL_interp(E_grid - phi_L)
        SR_arr = SR_interp(E_grid - phi_R)

        t0 = time.time()
        args_sp = [(k, E_grid[k], SL_arr[k], SR_arr[k]) for k in range(nE)]
        with ProcessPoolExecutor(
            max_workers=n_workers,
            mp_context=ctx,
            initializer=_init_worker,
            initargs=(H_slice, V_slice, eta, nb, Nx, V_drop,
                      Ny, Nz, n_surf_layers),
        ) as ex:
            for k, out_L, out_R in ex.map(_spectral_spin_at_E, args_sp):
                for surf in surf_keys:
                    spec[iV][surf]['L']['x'][k] = out_L[surf][0]
                    spec[iV][surf]['L']['y'][k] = out_L[surf][1]
                    spec[iV][surf]['L']['z'][k] = out_L[surf][2]
                    spec[iV][surf]['R']['x'][k] = out_R[surf][0]
                    spec[iV][surf]['R']['y'][k] = out_R[surf][1]
                    spec[iV][surf]['R']['z'][k] = out_R[surf][2]
        if verbose:
            print(f"  V_drop = {V_drop:.3f}: spec done  "
                  f"({time.time()-t0:.1f}s)")

    # --- Fermi-window integration with tilted-lead bias. ---
    dE = np.gradient(E_grid)
    spin = {surf: {a: np.zeros((nV, nF)) for a in 'xyz'}
            for surf in surf_keys}

    for iV, V_drop in enumerate(V_drop_grid):
        for kF, EF in enumerate(EF_grid):
            fL = fermi(E_grid, EF + V_drop / 2.0, KT)
            fR = fermi(E_grid, EF - V_drop / 2.0, KT)
            w  = dE / (2 * np.pi)
            for surf in surf_keys:
                for a in 'xyz':
                    spin[surf][a][iV, kF] = (
                        w * (fL * spec[iV][surf]['L'][a]
                           + fR * spec[iV][surf]['R'][a])
                    ).sum()

    if np.isclose(V_drop_grid[0], 0.0):
        spin_lin = {surf: {a: spin[surf][a] - spin[surf][a][0:1, :]
                           for a in 'xyz'} for surf in surf_keys}
    else:
        spin_lin = spin

    ratio = {surf: {} for surf in surf_keys}
    for surf in surf_keys:
        sy_safe = np.where(np.abs(spin_lin[surf]['y']) > 1e-14,
                           spin_lin[surf]['y'], np.nan)
        ratio[surf]['z_over_y'] = spin_lin[surf]['z'] / sy_safe

    return dict(
        E_grid=E_grid, EF_grid=EF_grid, V_drop_grid=V_drop_grid,
        spec=spec, spin=spin, spin_lin=spin_lin, ratio=ratio,
    )


# ─────────────────────────── main ─────────────────────────────────

if __name__ == "__main__":

    # Quick sanity test of the sparse trace
    test_idx = _build_spin_indices(Ny=8, Nz=12, n_surf_layers=3)
    rows, cols, vals = test_idx['top']['y']
    G_test = np.random.randn(384, 384) + 1j * np.random.randn(384, 384)

    # Sparse trace
    sparse_result = np.sum(vals * G_test[cols, rows]).imag

    # Dense trace (old way)
    Sy_dense = np.zeros((384, 384), dtype=complex)
    for r, c, v in zip(rows, cols, vals):
        Sy_dense[r, c] = v
    dense_result = np.trace(G_test @ Sy_dense).imag

    print(f"sparse: {sparse_result:.10f}, dense: {dense_result:.10f}, "
        f"diff: {abs(sparse_result - dense_result):.2e}")
    import matplotlib.pyplot as plt

    out = run_negf_3dti(
        V_drop_grid=np.array([0.0, 0.05]),
    )

    EF = out['EF_grid']
    V  = out['V_drop_grid']
    spin = out['spin_lin']

    print("\nPeak magnitudes per surface, per V_drop:")
    for surf in ['top', 'bot', 'all']:
        print(f"  {surf}:")
        for iV, V_drop in enumerate(V):
            print(f"    V_drop={V_drop:.3f}:  "
                  f"|Sx|={np.abs(spin[surf]['x'][iV]).max():.3e}  "
                  f"|Sy|={np.abs(spin[surf]['y'][iV]).max():.3e}  "
                  f"|Sz|={np.abs(spin[surf]['z'][iV]).max():.3e}")

    fig, axes = plt.subplots(3, 4, figsize=(20, 12))
    for row, surf in enumerate(['top', 'bot', 'all']):
        for iV, V_drop in enumerate(V):
            axes[row, 0].plot(EF, spin[surf]['x'][iV], 'o-',
                              label=f'V_drop={V_drop:.2f}')
            axes[row, 1].plot(EF, spin[surf]['y'][iV], 'o-',
                              label=f'V_drop={V_drop:.2f}')
            axes[row, 2].plot(EF, spin[surf]['z'][iV], 'o-',
                              label=f'V_drop={V_drop:.2f}')
            axes[row, 3].plot(EF, out['ratio'][surf]['z_over_y'][iV], 'o-',
                              label=f'V_drop={V_drop:.2f}')
        for col, lbl in enumerate([r'$\langle S_x\rangle$',
                                    r'$\langle S_y\rangle$',
                                    r'$\langle S_z\rangle$',
                                    r'$\langle S_z\rangle/\langle S_y\rangle$']):
            axes[row, col].axhline(0, color='gray', lw=0.5)
            axes[row, col].set_xlabel(r'$E_F$ (eV)')
            axes[row, col].set_ylabel(f'{lbl}  —  {surf}')
            axes[row, col].legend(fontsize=8)
            axes[row, col].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig('negf_3dti_optionB.png', dpi=130)
    plt.show()