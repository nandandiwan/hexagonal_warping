"""Intraband (Boltzmann-like) Edelstein response on the 4-band slab.

Diagonalize the slab Bloch H(kx,ky), weight each eigenstate's spin by its
top-surface localization, and compute the force-driven response
    Sy(EF) = sum_k sum_n <n|S_y^top|n> * v_x,n * (-df/dE)(E_n)
This uses the emergent surface chiral (sigma_z), not the bulk tau_y, so it
should be EVEN in EF if the surface spin is even.
"""
import os
for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[v] = "16"
import sys
sys.path.insert(0, "NEGF")
import numpy as np
from surface_ham import slab_bloch_H_kxky
from ti_3d_ham import default_params, kron, I2, sy as sy2

Nz = 8
n_surf = 3
params = default_params

# Top-surface spin operator S_y^top  (4Nz x 4Nz)
dim = 4 * Nz
Sytop = np.zeros((dim, dim), complex)
blk = kron(I2, sy2)  # tau_0 x sigma_y  (4x4)
for z in range(Nz - n_surf, Nz):
    s = slice(4 * z, 4 * z + 4)
    Sytop[s, s] = blk


def negdf(E, EF, kT):
    x = np.clip((E - EF) / kT, -200, 200)
    ex = np.exp(x)
    return ex / (kT * (1 + ex) ** 2)


def sy_boltzmann(EF, kmax=0.8, Nk=51, kT=0.025, h=1e-4):
    ks = np.linspace(-kmax, kmax, Nk)
    dk = ks[1] - ks[0]
    tot = 0.0
    for kx in ks:
        for ky in ks:
            H = slab_bloch_H_kxky(kx, ky, Nz, params)
            E, V = np.linalg.eigh(H)
            Hp = slab_bloch_H_kxky(kx + h, ky, Nz, params)
            Hm = slab_bloch_H_kxky(kx - h, ky, Nz, params)
            dH = (Hp - Hm) / (2 * h)
            for n in range(dim):
                psi = V[:, n]
                vx = (psi.conj() @ dH @ psi).real
                syn = (psi.conj() @ Sytop @ psi).real
                tot += syn * vx * negdf(E[n], EF, kT)
    return tot * dk * dk


for EF in (0.15, -0.15, 0.10, -0.10, 0.05, -0.05):
    s = sy_boltzmann(EF)
    print(f"EF={EF:+.2f}: Sy_intraband={s:+.5e}", flush=True)
