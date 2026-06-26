"""Full NEGF (G^<, finite V_drop via RGF) CISP sweep vs E_F, with and without
hexagonal warping. Uses the corrected physical spin operator S = tau_z (x) sigma.

CISP = spin(V_drop) - spin(0), per component.  V_drop kept small (linear regime,
matches the Kubo linear-response result).
"""
import os
for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[v] = "16"
import sys, time
sys.path.insert(0, "NEGF")
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from negf_kwant import run_negf, default_params, OUT

common = dict(Nz=8, Nx=12, N_ky=31, ky_max=0.6, N_E=81, T=300, n_surf=1, eta=5e-4)
V_drop = 0.02
EF_vals = np.linspace(-0.25, 0.25, 13)
lam = 0.5   # hexagonal warping strength

cases = {
    "no_warp": {**default_params},
    "warp":    {**default_params, "lambda_warp": lam},
}
res = {}
for tag, p in cases.items():
    sx = np.zeros(len(EF_vals)); sy = np.zeros(len(EF_vals)); sz = np.zeros(len(EF_vals))
    t0 = time.time()
    for i, EF in enumerate(EF_vals):
        sv = run_negf(EF, V_drop, params=p, verbose=False, **common)
        s0 = run_negf(EF, 0.0, params=p, verbose=False, **common)
        sx[i] = sv["x"] - s0["x"]
        sy[i] = sv["y"] - s0["y"]
        sz[i] = sv["z"] - s0["z"]
        print(f"  [{tag}] EF={EF:+.3f}  Sx={sx[i]:+.3e}  Sy={sy[i]:+.3e}  "
              f"Sz={sz[i]:+.3e}", flush=True)
    res[tag] = dict(sx=sx, sy=sy, sz=sz)
    print(f"{tag} done ({time.time()-t0:.0f}s)", flush=True)

fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
comps = ["sx", "sy", "sz"]
labels = [r"$S_x$", r"$S_y$", r"$S_z$"]
for ax, c, lab in zip(axes, comps, labels):
    ax.plot(EF_vals, res["no_warp"][c], "o-", lw=2, label="no warp")
    ax.plot(EF_vals, res["warp"][c], "s-", lw=2, label=rf"warp $\lambda$={lam}")
    ax.axhline(0, color="gray", lw=0.5); ax.axvline(0, color="gray", lw=0.5)
    ax.set_xlabel(r"$E_F$ (eV)"); ax.set_ylabel(f"CISP {lab}")
    ax.set_title(lab); ax.grid(alpha=0.25); ax.legend()
fig.suptitle(rf"NEGF ($G^<$, $V_\mathrm{{drop}}$={V_drop}) CISP vs $E_F$ — "
             r"with / without hexagonal warping  ($S=\tau_z\!\otimes\!\sigma$)")
fig.tight_layout()
OUT.mkdir(parents=True, exist_ok=True)
fig.savefig(OUT / "cisp_warping_sweep.png", dpi=150)
np.savez(OUT / "cisp_warping_sweep.npz", EF=EF_vals, V_drop=V_drop, lam=lam,
         **{f"{t}_{c}": res[t][c] for t in cases for c in ("sx", "sy", "sz")})
print(f"Saved: {OUT / 'cisp_warping_sweep.png'}", flush=True)
