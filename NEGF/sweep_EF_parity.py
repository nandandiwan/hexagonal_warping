import os
for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[v] = "16"
import sys, time
sys.path.insert(0, "NEGF")
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from negf_kwant import run_kubo, default_params, OUT

common = dict(Nz=8, Nx=10, N_ky=31, ky_max=0.6, N_E=81, T=300, eta=5e-4)
EF_vals = np.linspace(-0.25, 0.25, 13)

cases = {0.0: "C=D=0 (exact τ_y chiral → odd)",
         0.1: "D=0.1 (chiral broken → even)"}
results = {}

for D, label in cases.items():
    p = {**default_params, "D_par": D, "D_z": D}
    sy = np.zeros(len(EF_vals))
    t0 = time.time()
    for i, EF in enumerate(EF_vals):
        s = run_kubo(EF, V_drop=0.05, n_surf=1, params=p, verbose=False, **common)
        sy[i] = s["y"]
        print(f"  D={D}  EF={EF:+.3f}  Sy={sy[i]:+.3e}", flush=True)
    results[D] = sy
    print(f"D={D} done ({time.time()-t0:.0f}s)", flush=True)

fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
for ax, D in zip(axes, cases):
    ax.plot(EF_vals, results[D], "o-", lw=2)
    ax.axhline(0, color="gray", lw=0.5)
    ax.axvline(0, color="gray", lw=0.5)
    ax.set_xlabel(r"$E_F$ (eV)")
    ax.set_ylabel(r"CISP $S_y$")
    ax.set_title(cases[D])
    ax.grid(alpha=0.25)
fig.suptitle("NEGF (Kubo) top-surface CISP vs $E_F$ — parity vs particle-hole asymmetry D")
fig.tight_layout()
OUT.mkdir(parents=True, exist_ok=True)
fig.savefig(OUT / "cisp_EF_parity.png", dpi=150)
np.savez(OUT / "cisp_EF_parity.npz", EF=EF_vals,
         sy_D0=results[0.0], sy_D01=results[0.1])
print(f"Saved: {OUT / 'cisp_EF_parity.png'}", flush=True)
