"""Kubo (linear-response) CISP vs crystal/drift angle theta, with and without
hexagonal warping.  Transport is along x; theta rotates the Liang Fu warping
relative to the current (warp_angle).  Linear response -> no finite-V
nonlinearity, so Sx is mirror-forbidden (~0) and only Sy, Sz are plotted.

The warping has exact C3 symmetry (warp_theta period = 2*pi/3), so we compute
one period of real data and tile it across [0, 2*pi] for the full-circle plot.

Outputs (separate figures, Boltzmann-style):
  sy_vs_theta_kubo.png : Sy(theta), warp vs no-warp
  sz_vs_theta_kubo.png : Sz(theta), warp vs no-warp
"""
import os
for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[v] = "32"
import sys, time
sys.path.insert(0, "NEGF")
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from negf_kwant import run_kubo, default_params, OUT

EF = 0.1
V_drop = 0.05
lam = 0.5
common = dict(Nz=8, Nx=12, N_ky=31, ky_max=0.6, N_E=81, T=300, n_surf=1, eta=5e-4)

# one C3 period of real data
n_per = 13
th_period = np.linspace(0.0, 2 * np.pi / 3, n_per, endpoint=False)

sy_w = np.zeros(n_per); sz_w = np.zeros(n_per)
t0 = time.time()
for i, th in enumerate(th_period):
    p = {**default_params, "lambda_warp": lam, "warp_angle": th}
    s = run_kubo(EF, V_drop=V_drop, params=p, verbose=False, **common)
    sy_w[i], sz_w[i] = s["y"], s["z"]
    print(f"  warp  theta={th/np.pi:.3f}pi  Sy={sy_w[i]:+.4e}  Sz={sz_w[i]:+.4e}", flush=True)
print(f"warp period done ({time.time()-t0:.0f}s)", flush=True)

# no warp (theta-independent): one point
s0 = run_kubo(EF, V_drop=V_drop, params={**default_params, "lambda_warp": 0.0},
              verbose=False, **common)
sy0, sz0 = s0["y"], s0["z"]
print(f"no-warp  Sy={sy0:+.4e}  Sz={sz0:+.4e}", flush=True)

# tile the C3 period across [0, 2*pi] (+ closing point at 2*pi)
th_full = np.concatenate([th_period + k * 2 * np.pi / 3 for k in (0, 1, 2)])
th_full = np.append(th_full, 2 * np.pi)
sy_full = np.append(np.tile(sy_w, 3), sy_w[0])
sz_full = np.append(np.tile(sz_w, 3), sz_w[0])

xt = np.linspace(0, 2 * np.pi, 7)
xtl = ["0", r"$\pi/3$", r"$2\pi/3$", r"$\pi$", r"$4\pi/3$", r"$5\pi/3$", r"$2\pi$"]

def make_plot(comp, warp_curve, flat_val, fname):
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(th_full, warp_curve, "-", lw=2, color="C1", label=rf"warp $\lambda$={lam}")
    ax.axhline(flat_val, ls="--", lw=2, color="C0", label="no warp")
    ax.axhline(0, color="gray", lw=0.5)
    ax.set_xlabel(r"crystal/drift angle $\theta$ (rad)")
    ax.set_ylabel(rf"Kubo CISP $S_{comp}$")
    ax.set_title(rf"$\langle S_{comp}\rangle$ vs $\theta$   "
                 rf"($E_F$={EF} eV, $T$=300 K, Kubo)")
    ax.set_xticks(xt); ax.set_xticklabels(xtl)
    ax.grid(alpha=0.25); ax.legend()
    fig.tight_layout()
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / fname, dpi=150)
    plt.close(fig)
    print(f"Saved: {OUT / fname}", flush=True)

make_plot("y", sy_full, sy0, "sy_vs_theta_kubo.png")
make_plot("z", sz_full, sz0, "sz_vs_theta_kubo.png")
np.savez(OUT / "cisp_vs_theta_kubo.npz", theta_period=th_period,
         sy_warp=sy_w, sz_warp=sz_w, sy_nowarp=sy0, sz_nowarp=sz0,
         EF=EF, lam=lam, V_drop=V_drop)
print("Done!", flush=True)
