import os
for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[v] = "16"
import numpy as np, sys
sys.path.insert(0, "NEGF")
import kwant
from negf_kwant import slab_matrices, default_params
from scipy.sparse import csc_matrix
from scipy.sparse.linalg import splu

I2 = np.eye(2, dtype=complex)
sy = np.array([[0, -1j], [1j, 0]], dtype=complex)
S4y = np.kron(I2, sy)


def negdf(E, EF, kT):
    x = np.clip((E - EF) / kT, -200, 200)
    ex = np.exp(x)
    return ex / (kT * (1 + ex) ** 2)


def dag(M):
    return M.conj().T


def make_lead(Htop, Vtop):
    nb = Htop.shape[0]
    lat = kwant.lattice.chain(norbs=nb)
    syst = kwant.Builder()
    syst[lat(0)] = Htop
    syst[lat(1)] = Htop
    syst[lat(1), lat(0)] = Vtop
    lead = kwant.Builder(kwant.TranslationalSymmetry((-1,)))
    lead[lat(0)] = Htop
    lead[lat(0), lat(-1)] = Vtop
    syst.attach_lead(lead)
    syst.attach_lead(lead.reversed())
    return syst.finalized()


def kubo_toplead(EF, Nz=8, Nx=10, n_lead=4, n_surf=1, N_ky=31, ky_max=0.6,
                 N_E=81, kT=0.025, eta=5e-4, params=None):
    if params is None:
        params = default_params
    Eg = np.linspace(-0.35, 0.35, N_E)
    kys = np.linspace(-ky_max, ky_max, N_ky)
    dky = kys[1] - kys[0]
    w = np.array([negdf(E, EF, kT) for E in Eg])
    nb = 4 * Nz
    dim = Nx * nb
    ntop = 4 * n_lead
    topsl = slice(nb - ntop, nb)
    Sy = 0.0
    G = 0.0
    for ky in kys:
        H_on, V_hop, _ = slab_matrices(ky, Nz, params)
        fs = make_lead(H_on[topsl, topsl].copy(), V_hop[topsl, topsl].copy())
        Hdev = np.zeros((dim, dim), complex)
        for i in range(Nx):
            s = slice(i * nb, (i + 1) * nb)
            Hdev[s, s] = H_on
            if i < Nx - 1:
                s2 = slice((i + 1) * nb, (i + 2) * nb)
                Hdev[s, s2] = V_hop
                Hdev[s2, s] = dag(V_hop)
        L0 = nb - ntop
        Lend = nb
        R0 = (Nx - 1) * nb + (nb - ntop)
        Rend = Nx * nb
        rL = np.zeros((dim, ntop), complex)
        rL[L0:Lend, :] = np.eye(ntop)
        rR = np.zeros((dim, ntop), complex)
        rR[R0:Rend, :] = np.eye(ntop)
        kern = np.zeros(N_E)
        tr = np.zeros(N_E)
        top_z0 = Nz - n_surf
        for iE, E in enumerate(Eg):
            Ec = E + 1j * eta
            SL = fs.leads[0].selfenergy(Ec)
            SR = fs.leads[1].selfenergy(Ec)
            GL = 1j * (SL - dag(SL))
            GR = 1j * (SR - dag(SR))
            M = Ec * np.eye(dim) - Hdev
            M[L0:Lend, L0:Lend] -= SL
            M[R0:Rend, R0:Rend] -= SR
            lu = splu(csc_matrix(M))
            XL = lu.solve(rL)
            XR = lu.solve(rR)
            xr0 = XR[L0:Lend, :]
            tr[iE] = np.trace(GL @ xr0 @ GR @ dag(xr0)).real
            ss = 0j
            for i in range(Nx):
                for iz in range(top_z0, Nz):
                    r = i * nb + 4 * iz
                    xl = XL[r:r + 4, :]
                    xrr = XR[r:r + 4, :]
                    ss += np.trace(S4y @ (xl @ GL @ dag(xl) - xrr @ GR @ dag(xrr)))
            kern[iE] = ss.real
        Sy += np.trapezoid(kern * w, Eg) / (4 * np.pi) * dky / (2 * np.pi)
        G += np.trapezoid(tr * w, Eg) * dky / (2 * np.pi)
    return Sy, G


for EF in (0.15, -0.15, 0.10, -0.10):
    Sy, G = kubo_toplead(EF)
    print(f"EF={EF:+.2f}: Sy={Sy:+.4e}  G={G:.4e}  Sy/G={Sy/G:+.4e}", flush=True)
