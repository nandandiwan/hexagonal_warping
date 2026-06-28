"""NEGF transverse-spin (Sy, perpendicular to the transport // x) vs drift/
crystal angle theta.

Transport is always along x; theta rotates the crystal (the Liang Fu warping
term) relative to the current via warp_angle.  S_y = tau_z (x) sigma_y is the
component perpendicular to the current.

At lambda_warp=0 the warping vanishes, the Hamiltonian is theta-independent, and
the isotropic Edelstein response gives a CONSTANT Sy vs theta (sanity check).
At lambda_warp != 0 the C3 warping makes Sy(theta) 120-degree periodic.
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
lam = 0.0   # no warping -> expect Sy(theta) = const
common = dict(Nz=8, Nx=12, N_ky=31, ky_max=0.6, N_E=81, T=300, n_surf=1, eta=5e-4)

thetas = np.linspace(0, 2 * np.pi, 13)
Sy = np.zeros(len(thetas))
t0 = time.time()
for i, th in enumerate(thetas):
    p = {**default_params, "lambda_warp": lam, "warp_angle": th}
    sv = run_negf(EF, V_drop, params=p, verbose=False, **common)
    s0 = run_negf(EF, 0.0, params=p, verbose=False, **common)
    Sy[i] = sv["y"] - s0["y"]
    print(f"  theta={th:6.3f} ({th/np.pi:.2f} pi)  Sy={Sy[i]:+.5e}", flush=True)
print(f"done ({time.time()-t0:.0f}s)  spread = {Sy.max()-Sy.min():.2e}  "
      f"(should be ~0 at lambda=0)", flush=True)

fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(thetas, Sy, "o-", lw=2, color="C1")
ax.axhline(0, color="gray", lw=0.5)
ax.set_xlabel(r"crystal/drift angle $\theta$ (rad)")
ax.set_ylabel(r"transverse CISP $S_y$")
ax.set_title(rf"NEGF $S_y$ vs $\theta$  ($\lambda$={lam}, $E_F$={EF}, "
             rf"$V_\mathrm{{drop}}$={V_drop})")
ax.set_xticks(np.linspace(0, 2 * np.pi, 7))
ax.set_xticklabels(["0", r"$\pi/3$", r"$2\pi/3$", r"$\pi$",
                    r"$4\pi/3$", r"$5\pi/3$", r"$2\pi$"])
ax.grid(alpha=0.25)
fig.tight_layout()
OUT.mkdir(parents=True, exist_ok=True)
fig.savefig(OUT / "sy_vs_theta_nowarp.png", dpi=150)
np.savez(OUT / "sy_vs_theta_nowarp.npz", theta=thetas, Sy=Sy, lam=lam)
print(f"Saved: {OUT / 'sy_vs_theta_nowarp.png'}", flush=True)
