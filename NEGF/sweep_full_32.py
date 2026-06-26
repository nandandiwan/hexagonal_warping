"""Full CISP sweeps vs E_F on 32 cores: NEGF (finite V_drop, G^<) AND Kubo
(linear response), each with and without hexagonal warping.

Corrected operators: physical spin S = tau_z (x) sigma, warping = tau_z (x) sigma_z.

Plot is 2 rows (NEGF, Kubo) x 3 cols (Sx, Sy, Sz), warp vs no-warp overlaid.
Expectations:
  - Sy: even, nonzero (Edelstein), both methods.
  - Sz: warping-induced.  Kubo = smooth single-signed dome (linear).  NEGF adds
        an even O(V^2) offset -> dip/zero-crossing near E_F=0.
  - Sx: ZERO in linear response (Kubo ~ 0, M_x-forbidden); NEGF nonzero = O(V^2).
"""
import os
for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[v] = "48"
import sys, time
sys.path.insert(0, "NEGF")
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from negf_kwant import run_negf, run_kubo, default_params, OUT

common = dict(Nz=8, Nx=12, N_ky=48, ky_max=0.6, N_E=201, T=300, n_surf=1, eta=5e-5)
V_drop = 0.02
EF_vals = np.linspace(-0.25, 0.25, 11)
lam = 0.5
pars = {"no_warp": {**default_params}, "warp": {**default_params, "lambda_warp": lam}}

res = {("negf", t): np.zeros((len(EF_vals), 3)) for t in pars}
res.update({("kubo", t): np.zeros((len(EF_vals), 3)) for t in pars})

for tag, p in pars.items():
    # NEGF (finite V_drop): CISP = spin(V) - spin(0)
    t0 = time.time()
    for i, EF in enumerate(EF_vals):
        sv = run_negf(EF, V_drop, params=p, verbose=False, **common)
        s0 = run_negf(EF, 0.0, params=p, verbose=False, **common)
        res[("negf", tag)][i] = [sv[a] - s0[a] for a in "xyz"]
        print(f"  NEGF [{tag}] EF={EF:+.3f}  "
              f"Sx={res[('negf',tag)][i,0]:+.2e} Sy={res[('negf',tag)][i,1]:+.2e} "
              f"Sz={res[('negf',tag)][i,2]:+.2e}", flush=True)
    print(f"NEGF {tag} done ({time.time()-t0:.0f}s)", flush=True)
    # Kubo (linear response)
    t0 = time.time()
    for i, EF in enumerate(EF_vals):
        s = run_kubo(EF, V_drop=V_drop, params=p, verbose=False, **common)
        res[("kubo", tag)][i] = [s[a] for a in "xyz"]
        print(f"  KUBO [{tag}] EF={EF:+.3f}  "
              f"Sx={res[('kubo',tag)][i,0]:+.2e} Sy={res[('kubo',tag)][i,1]:+.2e} "
              f"Sz={res[('kubo',tag)][i,2]:+.2e}", flush=True)
    print(f"KUBO {tag} done ({time.time()-t0:.0f}s)", flush=True)

labels = [r"$S_x$", r"$S_y$", r"$S_z$"]
fig, axes = plt.subplots(2, 3, figsize=(15, 8))
for row, method in enumerate(("negf", "kubo")):
    for c in range(3):
        ax = axes[row, c]
        ax.plot(EF_vals, res[(method, "no_warp")][:, c], "o-", lw=2, label="no warp")
        ax.plot(EF_vals, res[(method, "warp")][:, c], "s-", lw=2, label=rf"warp $\lambda$={lam}")
        ax.axhline(0, color="gray", lw=0.5); ax.axvline(0, color="gray", lw=0.5)
        ax.set_xlabel(r"$E_F$ (eV)")
        ax.set_ylabel(f"{['NEGF','Kubo'][row]} CISP {labels[c]}")
        ax.set_title(f"{['NEGF $G^<$','Kubo (linear)'][row]}  {labels[c]}")
        ax.grid(alpha=0.25); ax.legend(fontsize=8)
fig.suptitle(r"CISP vs $E_F$ — NEGF (finite $V$) vs Kubo (linear), warp vs no-warp "
             r"($S=\tau_z\!\otimes\!\sigma$)")
fig.tight_layout()
OUT.mkdir(parents=True, exist_ok=True)
fig.savefig(OUT / "cisp_full_compare.png", dpi=150)
np.savez(OUT / "cisp_full_compare.npz", EF=EF_vals, V_drop=V_drop, lam=lam,
         **{f"{m}_{t}": res[(m, t)] for m in ("negf", "kubo") for t in pars})
print(f"Saved: {OUT / 'cisp_full_compare.png'}", flush=True)
