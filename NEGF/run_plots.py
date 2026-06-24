#!/usr/bin/env python
import os
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
           "NUMEXPR_NUM_THREADS"):
    os.environ[_v] = "16"

import numpy as np
from negf_kwant import plot_cisp_vs_Vdrop, plot_cisp_vs_EF

# Plot V_drop sweep
d = np.load("plots/cisp_vs_Vdrop.npz")
plot_cisp_vs_Vdrop(d['V_drop'], d['cisp'], float(d['E_F']), 8, 10)

# Run E_F sweep
import time
from negf_kwant import run_negf

Nz, Nx = 8, 10
N_ky, N_E = 31, 81
common = dict(Nz=Nz, Nx=Nx, N_ky=N_ky, ky_max=0.6, N_E=N_E, T=300,
              n_surf=1, eta=5e-4)
V_drop = 0.05

EF_vals = np.linspace(0.02, 0.25, 8)
results = np.zeros((len(EF_vals), 3))
for i, EF in enumerate(EF_vals):
    print(f"E_F={EF:.3f} ({i+1}/{len(EF_vals)})...", flush=True)
    s_bias = run_negf(EF, V_drop, verbose=False, **common)
    s_eq = run_negf(EF, 0.0, verbose=False, **common)
    results[i] = [s_bias[a] - s_eq[a] for a in 'xyz']
    print(f"  Sx={results[i,0]:.3e}  Sy={results[i,1]:.3e}  Sz={results[i,2]:.3e}", flush=True)

np.savez("plots/cisp_vs_EF.npz", EF=EF_vals, cisp=results, V_drop=V_drop)
plot_cisp_vs_EF(EF_vals, results, V_drop, Nz, Nx)
print("Done!")
