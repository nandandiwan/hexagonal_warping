#!/usr/bin/env python
import os
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
           "NUMEXPR_NUM_THREADS"):
    os.environ[_v] = "16"

import numpy as np, time
from negf_kwant import run_negf

Nz, Nx = 8, 10
N_ky, N_E = 31, 81
common = dict(Nz=Nz, Nx=Nx, N_ky=N_ky, ky_max=0.6, N_E=N_E, T=300,
              n_surf=1, eta=5e-4)

print("Equilibrium...", flush=True)
t0 = time.time()
s_eq = run_negf(0.1, 0.0, verbose=False, **common)
print(f"  done ({time.time()-t0:.0f}s), Sy_eq={s_eq['y']:.3e}", flush=True)

V_vals = np.array([0.005, 0.01, 0.02, 0.04, 0.06, 0.08, 0.10, 0.12])
print(f"Sweeping {len(V_vals)} V_drop values...", flush=True)
hdr = f"{'V_drop':>8s}  {'Sx':>12s}  {'Sy':>12s}  {'Sz':>12s}  {'Sy/V':>12s}  time"
print(hdr, flush=True)

results = []
for V in V_vals:
    t0 = time.time()
    s = run_negf(0.1, V, verbose=False, **common)
    cisp = {a: s[a] - s_eq[a] for a in 'xyz'}
    dt = time.time() - t0
    ratio = cisp['y'] / V if V > 0 else 0
    print(f"{V:8.4f}  {cisp['x']:12.4e}  {cisp['y']:12.4e}  {cisp['z']:12.4e}  {ratio:12.4e}  {dt:.0f}s", flush=True)
    results.append([cisp['x'], cisp['y'], cisp['z']])

results = np.array(results)
np.savez("plots/cisp_vs_Vdrop.npz", V_drop=V_vals, cisp=results, E_F=0.1)
print("Saved plots/cisp_vs_Vdrop.npz")
