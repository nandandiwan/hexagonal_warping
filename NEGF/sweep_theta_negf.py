"""Full NEGF (finite V_drop, G^<) CISP Sx, Sy vs crystal/drift angle theta,
with and without hexagonal warping.

Transport // x; theta rotates the Liang Fu warping vs the current (warp_angle).
NEGF keeps the finite-V nonlinearity, so Sx (mirror-forbidden at linear order)
is nonzero here and worth plotting alongside Sy.

The warping has exact C3 symmetry (warp_theta period = 2*pi/3), so we compute
one period at high resolution (n_per points) and tile it 3x across [0, 2*pi]
for smooth full-circle curves.  Increase n_per for more smoothness.

Outputs (separate Boltzmann-style figures):
  sx_vs_theta_negf.png : Sx(theta), warp vs no-warp
  sy_vs_theta_negf.png : Sy(theta), warp vs no-warp
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
from negf_kwant import run_negf, default_params, OUT

EF = 0.1
V_drop = 0.02
lam = 0.5
n_per = 30                      # points per C3 period (raise for smoother)
common = dict(Nz=8, Nx=12, N_ky=31, ky_max=0.6, N_E=81, T=300, n_surf=1, eta=5e-4)

th_period = np.linspace(0.0, 2 * np.pi / 3, n_per, endpoint=False)


def cisp(params):
    sv = run_negf(EF, V_drop, params=params, verbose=False, **common)
    s0 = run_negf(EF, 0.0, params=params, verbose=False, **common)
    return sv["x"] - s0["x"], sv["y"] - s0["y"]


sx_w = np.zeros(n_per); sy_w = np.zeros(n_per)
t0 = time.time()
for i, th in enumerate(th_period):
    p = {**default_params, "lambda_warp": lam, "warp_angle": th}
    sx_w[i], sy_w[i] = cisp(p)
    print(f"  warp theta={th/np.pi:.3f}pi  Sx={sx_w[i]:+.4e}  Sy={sy_w[i]:+.4e}  "
          f"[{i+1}/{n_per}, {time.time()-t0:.0f}s]", flush=True)
print(f"warp period done ({time.time()-t0:.0f}s)", flush=True)

# no warp: theta-independent -> one point (Sx ~ 0, Sy = const)
sx0, sy0 = cisp({**default_params, "lambda_warp": 0.0})
print(f"no-warp  Sx={sx0:+.4e}  Sy={sy0:+.4e}", flush=True)

# tile the C3 period across [0, 2*pi] (+ closing point)
th_full = np.concatenate([th_period + k * 2 * np.pi / 3 for k in (0, 1, 2)])
th_full = np.append(th_full, 2 * np.pi)
sx_full = np.append(np.tile(sx_w, 3), sx_w[0])
sy_full = np.append(np.tile(sy_w, 3), sy_w[0])

xt = np.linspace(0, 2 * np.pi, 7)
xtl = ["0", r"$\pi/3$", r"$2\pi/3$", r"$\pi$", r"$4\pi/3$", r"$5\pi/3$", r"$2\pi$"]


def make_plot(comp, warp_curve, flat_val, fname):
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(th_full, warp_curve, "-", lw=2, color="C1", label=rf"warp $\lambda$={lam}")
    ax.axhline(flat_val, ls="--", lw=2, color="C0", label="no warp")
    ax.axhline(0, color="gray", lw=0.5)
    ax.set_xlabel(r"crystal/drift angle $\theta$ (rad)")
    ax.set_ylabel(rf"NEGF CISP $S_{comp}$")
    ax.set_title(rf"$\langle S_{comp}\rangle$ vs $\theta$   "
                 rf"($E_F$={EF} eV, $T$=300 K, NEGF $V_\mathrm{{drop}}$={V_drop})")
    ax.set_xticks(xt); ax.set_xticklabels(xtl)
    ax.grid(alpha=0.25); ax.legend()
    fig.tight_layout()
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / fname, dpi=150)
    plt.close(fig)
    print(f"Saved: {OUT / fname}", flush=True)


make_plot("x", sx_full, sx0, "sx_vs_theta_negf.png")
make_plot("y", sy_full, sy0, "sy_vs_theta_negf.png")
np.savez(OUT / "cisp_vs_theta_negf.npz", theta_period=th_period,
         sx_warp=sx_w, sy_warp=sy_w, sx_nowarp=sx0, sy_nowarp=sy0,
         EF=EF, lam=lam, V_drop=V_drop, n_per=n_per)
print("Done!", flush=True)
