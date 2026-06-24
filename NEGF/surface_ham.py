import numpy as np
import matplotlib.pyplot as plt
from ti_3d_ham import (build_slice_3DTI, bulk_bloch_H, default_params,
                       onsite_4x4, hop_4x4)


def slab_bloch_H_kxky(kx, ky, Nz, params):
    # the slab hamiltonian 
    n_per = 4
    dim = n_per * Nz

    H_site = onsite_4x4(params['C'], params['M0'], # onsite energy term 
                        params['D_par'], params['D_z'],
                        params['B_par'], params['B_z'])
    Hx = hop_4x4('x', params['A_par'], params['A_z'], # periodoic coupling
                 params['D_par'], params['D_z'],
                 params['B_par'], params['B_z'])
    Hy = hop_4x4('y', params['A_par'], params['A_z'], # periodoic coupling
                 params['D_par'], params['D_z'],
                 params['B_par'], params['B_z'])
    Hz = hop_4x4('z', params['A_par'], params['A_z'], # this is like 1D TB 
                 params['D_par'], params['D_z'],
                 params['B_par'], params['B_z'])

    on_site_full = (H_site
                    + Hx * np.exp(1j*kx) + Hx.conj().T * np.exp(-1j*kx)
                    + Hy * np.exp(1j*ky) + Hy.conj().T * np.exp(-1j*ky))

    H = np.zeros((dim, dim), dtype=complex)

    # periodic cahin + 1D TB 
    for z in range(Nz):
        i = n_per * z
        H[i:i+n_per, i:i+n_per] = on_site_full
    for z in range(Nz - 1):
        i = n_per * z
        j = n_per * (z + 1)
        H[i:i+n_per, j:j+n_per] = Hz
        H[j:j+n_per, i:i+n_per] = Hz.conj().T
    return H


def slab_bands(kx_arr, ky_arr, Nz, params):
    bands = np.zeros((len(kx_arr), 4*Nz))
    states = np.zeros((len(kx_arr), 4*Nz, 4*Nz), dtype=complex)
    for i, (kx, ky) in enumerate(zip(kx_arr, ky_arr)):
        H = slab_bloch_H_kxky(kx, ky, Nz, params)
        E, V = np.linalg.eigh(H)
        bands[i] = E
        states[i] = V
    return bands, states


def surface_weight(states, Nz, n_surf=3):
    # This is a disagonstic function 

    n_per = 4
    n_k, dim, _ = states.shape
    weight = np.zeros((n_k, dim))
    z_indices_bot = np.arange(n_per * n_surf)
    z_indices_top = np.arange(n_per * (Nz - n_surf), n_per * Nz)
    for ik in range(n_k):
        for b in range(dim):
            psi = states[ik, :, b]
            w_bot = np.sum(np.abs(psi[z_indices_bot])**2)
            w_top = np.sum(np.abs(psi[z_indices_top])**2)
            weight[ik, b] = w_bot + w_top
    return weight
