import numpy as np

I2 = np.eye(2, dtype=complex)
sx = np.array([[0, 1], [1, 0]], dtype=complex)
sy = np.array([[0, -1j], [1j, 0]], dtype=complex)
sz = np.array([[1, 0], [0, -1]], dtype=complex)

def kron(*mats):
    out = mats[0]
    for m in mats[1:]:
        out = np.kron(out, m)
    return out

# 4x4 basis matrices  (orbital tau outer, spin sigma inner)
TyS0 = kron(sy, I2)
T0S0 = kron(I2, I2)
TzS0 = kron(sz, I2)
TxSx = kron(sx, sx)
TxSy = kron(sx, sy)
TxSz = kron(sx, sz)


def onsite_4x4(C, M0, D_par, D_z, B_par, B_z):
    eps0 = C + 2*D_par + D_z
    m    = M0 - 2*B_par - B_z
    return eps0 * T0S0 + m * TzS0




def hop_4x4(direction, A_par, A_z, D_par, D_z, B_par, B_z):
    # Forward hop in +x, +y, or +z (4x4 matrix from site R to site R + delta)
    if direction == 'x':
        return -D_par/2 * T0S0 + B_par/2 * TzS0 - 1j*A_par/2 * TxSx
    if direction == 'y':
        return -D_par/2 * T0S0 + B_par/2 * TzS0 - 1j*A_par/2 * TxSy
    if direction == 'z':
        return -D_z  /2 * T0S0 + B_z  /2 * TzS0 - 1j*A_z  /2 * TxSz
    raise ValueError(direction)


def build_slice_3DTI(Ny, Nz, params, periodic_y=True):
    """Build H_slice, V_slice (nearest slice), and V2_slice (next-nearest slice)."""
    n_per = 4
    n_sites = Ny * Nz
    dim = n_per * n_sites

    H_site = onsite_4x4(**{k: params[k] for k in
                           ['C', 'M0', 'D_par', 'D_z', 'B_par', 'B_z']})
    Hy = hop_4x4('y', params['A_par'], params['A_z'],
                 params['D_par'], params['D_z'],
                 params['B_par'], params['B_z'])
    Hz = hop_4x4('z', params['A_par'], params['A_z'],
                 params['D_par'], params['D_z'],
                 params['B_par'], params['B_z'])
    Hx = hop_4x4('x', params['A_par'], params['A_z'],
                 params['D_par'], params['D_z'],
                 params['B_par'], params['B_z'])

    H_slice = np.zeros((dim, dim), dtype=complex)
    V_slice = np.zeros((dim, dim), dtype=complex)
    V2_slice = np.zeros((dim, dim), dtype=complex) 

    lam = params.get('lambda_warp', 0.0)
    
    # Warping hop matrices derived from sine expansions (multiplying by -i/2 for +kx)
    # -4 * sin(kx)   -> +2i
    V_warp_base  =  2j   * lam * TyS0   
    # -1 * sin(2*kx) -> +0.5i (Distance 2 hopping)
    V2_warp_base =  0.5j * lam * TyS0 
    # +6 * sin(kx)*cos(ky) -> cross-hopping
    V_warp_ky    = -1.5j * lam * TyS0   

    def idx(y, z):
        return y * Nz + z

    for y in range(Ny):
        for z in range(Nz):
            i = n_per * idx(y, z)
            H_slice[i:i+n_per, i:i+n_per] = H_site
            
            # Base transport direction hopping
            V_slice[i:i+n_per, i:i+n_per] = Hx + V_warp_base
            V2_slice[i:i+n_per, i:i+n_per] = V2_warp_base

    # z-hopping (intra-slice)
    for y in range(Ny):
        for z in range(Nz - 1):
            i = n_per * idx(y, z)
            j = n_per * idx(y, z + 1)
            H_slice[i:i+n_per, j:j+n_per] += Hz
            H_slice[j:j+n_per, i:i+n_per] += Hz.conj().T

    # y-hopping (intra-slice) AND Warp cross-terms
    for z in range(Nz):
        for y in range(Ny - 1):
            i = n_per * idx(y,     z)
            j = n_per * idx(y + 1, z)
            
            H_slice[i:i+n_per, j:j+n_per] += Hy
            H_slice[j:j+n_per, i:i+n_per] += Hy.conj().T
            
            # Cross-term: hopping x -> x+1 AND simultaneously y -> y±1
            V_slice[i:i+n_per, j:j+n_per] += V_warp_ky
            V_slice[j:j+n_per, i:i+n_per] += V_warp_ky
            
        if periodic_y and Ny > 2:
            i = n_per * idx(Ny - 1, z)
            j = n_per * idx(0,      z)
            H_slice[i:i+n_per, j:j+n_per] += Hy
            H_slice[j:j+n_per, i:i+n_per] += Hy.conj().T
            
            V_slice[i:i+n_per, j:j+n_per] += V_warp_ky
            V_slice[j:j+n_per, i:i+n_per] += V_warp_ky

    return H_slice, V_slice, V2_slice


def spin_operators_3DTI(Nx_slices, Ny, Nz):
    n_per = 4
    n_sites_total = Nx_slices * Ny * Nz
    eye_sites = np.eye(n_sites_total)
    Sx = np.kron(eye_sites, kron(I2, sx))
    Sy = np.kron(eye_sites, kron(I2, sy))
    Sz = np.kron(eye_sites, kron(I2, sz))
    return Sx, Sy, Sz


def bulk_bloch_H(kx, ky, kz, params):
    p = params
    eps0 = (p['C'] + p['D_par']*(2 - np.cos(kx) - np.cos(ky))
                   + p['D_z']  *(1 - np.cos(kz)))
    m    = (p['M0'] - p['B_par']*(2 - np.cos(kx) - np.cos(ky))
                    - p['B_z']  *(1 - np.cos(kz)))
    
    H = (eps0 * T0S0 + m * TzS0
         + p['A_par']*np.sin(kx) * TxSx
         + p['A_par']*np.sin(ky) * TxSy
         + p['A_z']  *np.sin(kz) * TxSz)
    
    # NEW: Liang Fu Hexagonal Warping
    if 'lambda_warp' in p and p['lambda_warp'] != 0.0:
        # Lattice regularization of k_x^3 - 3*k_x*k_y^2
        warp_k = (2 * np.sin(kx) - np.sin(2*kx) 
                  - 6 * np.sin(kx) * (1 - np.cos(ky)))
        H += p['lambda_warp'] * warp_k * TyS0
        
    return H


default_params = dict(
    C      =  0.0,
    M0     =  0.3,
    A_par  =  0.5,
    A_z    =  0.5,
    B_par  =  1.0,
    B_z    =  1.0,
    D_par  =  0.0,
    D_z    =  0.0,
)