# NEGF for current-induced spin polarization on a 3D-TI surface

Quantum-transport calculation of the **current-induced spin polarization (CISP /
Edelstein effect)** on the surface of a 3D topological insulator, using the
Zhang–Liu four-band model, a real-space/$k$-space slab discretization, `kwant`
lead self-energies, and a recursive-Green's-function (RGF) NEGF solver.
Hexagonal warping (Liang Fu) is included.

---

## 1. The model: Zhang–Liu four-band 3D TI

Low-energy $k\cdot p$ Hamiltonian for the Bi₂Se₃ family
(Zhang *et al.*, *Nat. Phys.* **5**, 438 (2009); Liu *et al.*, *PRB* **82**,
045122 (2010)), in the basis $\{|P1_z^+\uparrow\rangle, |P2_z^-\uparrow\rangle,
|P1_z^+\downarrow\rangle, |P2_z^-\downarrow\rangle\}$ with **orbital** Pauli
matrices $\tau$ (the two opposite-parity $p_z$ orbitals) and **spin** Pauli
matrices $\sigma$:

$$
H(\mathbf{k}) = \varepsilon(\mathbf{k})\,\mathbb{I} + M(\mathbf{k})\,\tau_z
+ A_\parallel\big[\sin k_x\,\tau_x\sigma_x + \sin k_y\,\tau_x\sigma_y\big]
+ A_z\,\sin k_z\,\tau_x\sigma_z
$$

$$
M(\mathbf{k}) = M_0 - B_\parallel(2-\cos k_x-\cos k_y) - B_z(1-\cos k_z),\qquad
\varepsilon(\mathbf{k}) = C + D_\parallel(2-\cos k_x-\cos k_y) + D_z(1-\cos k_z)
$$

($\tau_a\sigma_b \equiv \tau_a\otimes\sigma_b$.) The $\varepsilon(\mathbf{k})\mathbb{I}$
term is particle–hole asymmetry; we work at $C=D=0$, where the spectrum is
symmetric and the model has an exact chiral symmetry $\Gamma=\tau_y$
($\Gamma H\Gamma = -H$), used throughout for parity arguments.

### Why band inversion forces a surface state (Jackiw–Rebbi)

The topology lives in the **mass** $M(\mathbf{k})$. Keep only the $z$-dependence
and linearize about the surface normal $k_z\to -i\partial_z$:

$$
H_z \approx M(z)\,\tau_z + A_z(-i\partial_z)\,\tau_x\sigma_z,\qquad
M = M_0 - B_z(1-\cos k_z) \to M_0 + B_z\,\partial_z^2
$$

A topological insulator is the **inverted** regime $M_0/B_z>0$: deep in the bulk
$M\to M_0>0$, while outside the solid the gap is "normal", $M_{\rm vac}<0$. So the
mass **changes sign at the surface** — a domain wall. The 1D Dirac equation
$\big[M(z)\tau_z - iA_z\partial_z\,\tau_x\sigma_z\big]\psi = E\psi$ with a mass
kink $M(+\infty)>0,\,M(-\infty)<0$ has a **zero-energy bound state**
(Jackiw & Rebbi, *PRD* **13**, 3398 (1976)):

$$
\psi_s(z) \propto \exp\!\Big(\!-\!\int_0^z \tfrac{M(z')}{A_z}\,dz'\Big)\,\chi,
\qquad \tau_y\chi = +\chi
$$

i.e. an exponentially localized mode pinned to the surface and locked to a fixed
eigenvalue of the chiral operator $\tau_y$ (top vs bottom surface $=\tau_y=\pm1$).
Projecting the in-plane Dirac terms $A_\parallel\tau_x(\sigma_x k_x+\sigma_y k_y)$
onto this $\tau_y$ doublet gives the gapless **surface Dirac cone**

$$
H_{\rm surf} = A_\parallel\,(k_x\sigma_y - k_y\sigma_x)
\qquad\text{(spin–momentum locked)}
$$

— a single Dirac cone, protected because the bound state cannot be removed
without unwinding the bulk mass inversion. The bound state, not any boundary
condition we impose, is what makes the surface metallic.

---

## 2. Slab discretization

Geometry (`slab_matrices` in `negf_kwant.py`):

- **$z$**: finite, $N_z$ layers (hosts top/bottom surface states) — real space.
- **$y$**: translationally invariant → Fourier transform, $k_y$ is a good quantum
  number; we integrate over $k_y\in[-k_y^{\max}, k_y^{\max}]$.
- **$x$**: transport direction — real-space chain of $N_x$ sites + two
  semi-infinite leads.

Each site carries the 4-spinor; a "slice" (fixed $x$) is a $4N_z\times 4N_z$
block. The lattice Hamiltonian follows from the standard substitution

$$
\sin k \to \frac{e^{ik}-e^{-ik}}{2i}\ (\text{a hop, }\pm i/2),\qquad
\cos k \to 1 - \frac{e^{ik}+e^{-ik}}{2}\ (\text{on-site} + \text{a real }\pm\tfrac12\text{ hop})
$$

**On-diagonal block** $H_{\rm on}(k_y)$ (intra-slice, $4N_z\times 4N_z$):

$$
H_{\rm on}(k_y) = \sum_z \mathrm{on}_k(k_y) + \sum_z\big[H_z\ (z\!\leftrightarrow\! z{+}1) + \text{h.c.}\big],\qquad
\mathrm{on}_k(k_y) = H_0 + H_y e^{ik_y} + H_y^\dagger e^{-ik_y}
$$

where $H_0=\,$`onsite_4x4` (the $\varepsilon_0\mathbb{I}+m\tau_z$ piece,
$m=M_0-2B_\parallel-B_z$) and $H_y=\,$`hop_4x4('y')` carries the $k_y$-Bloch
phase. The $z$-direction is an explicit 1D tight-binding chain inside the block
(so surface states are resolved).

**Off-diagonal block** $V_{\rm hop}$ (slice $x\to x{+}1$, $4N_z\times 4N_z$,
block-diagonal in $z$): $V_{\rm hop}=\sum_z\,$`hop_4x4('x')`, i.e. the $+\hat{x}$
hop $-\tfrac{D_\parallel}{2}\tau_0 + \tfrac{B_\parallel}{2}\tau_z -
i\tfrac{A_\parallel}{2}\tau_x\sigma_x$ on each layer. It is $k_y$-independent.

The device Hamiltonian is block-tridiagonal in $x$:
$\mathrm{diag}=H_{\rm on}\,(+\,\phi(x)\mathbb{I}$ for a bias$)$,
$\mathrm{upper/lower}=V_{\rm hop}/V_{\rm hop}^\dagger$. This structure is exactly
what the RGF exploits, cost $O(N_x\,(4N_z)^3)$.

---

## 3. Hexagonal warping and why it needs next-nearest neighbors

Liang Fu's warping (Fu, *PRL* **103**, 266801 (2009)) is the leading C₃-symmetric
correction to the surface Dirac cone,

$$
H_w = \lambda\,(k_x^3 - 3k_xk_y^2)\,\sigma_z = \lambda\,\mathrm{Re}(k_+^3)\,\sigma_z,
\qquad k_+ = k_x + ik_y
$$

In the four-band basis the physical $\sigma_z$ is $\tau_z\otimes\sigma_z$ (see
§4), so $H_w=\lambda\,\mathrm{Re}(k_+^3)\,(\tau_z\sigma_z)$. The cubic-in-$k_x$
factor is the issue: a faithful lattice regularization needs **two harmonics in
$k_x$**,

$$
k_x^3 - 3k_xk_y^2 \ \longleftrightarrow\ -2(2-3\cos k_y)\sin k_x - \sin 2k_x
$$

The $\sin 2k_x$ term is a **next-nearest-neighbor (NNN / 2NN) hop in $x$**
(connects slice $i\to i{+}2$). Concretely `slab_matrices` returns a third matrix
$V_2 = \tfrac{i\lambda}{2}(\tau_z\sigma_z)$ for the $i\to i{+}2$ coupling,
alongside the NN piece $V_{\rm warp}^{\rm NN}=i\lambda(2-3\cos k_y)(\tau_z\sigma_z)$
folded into $V_{\rm hop}$.

Consequences:

- The device is no longer block-**tridiagonal** (it has a second off-diagonal),
  so the plain RGF/sparse solver must be adapted: we **super-block** pairs of
  $x$-slices (block size $2\cdot4N_z$) to restore tridiagonal form, and the
  `kwant` leads use a **doubled unit cell**.
- **Crystal-angle rotation.** Rotating the current by $\theta$ w.r.t. the crystal
  rotates only the warping (the Dirac part is isotropic):
  $\mathrm{Re}(k_+^3)\to\mathrm{Re}(e^{-i3\theta}k_+^3) =
  \cos3\theta\,(k_x^3-3k_xk_y^2) + \sin3\theta\,(3k_x^2k_y-k_y^3)$. The new
  $\mathrm{Im}(k_+^3)=3k_x^2k_y-k_y^3$ piece adds $k_y$-dependent on-site terms
  and a real NN $x$-hop (`warp_angle` parameter). Everything is
  $2\pi/3$-periodic in $\theta$.

---

## 4. Measuring spin in the NEGF formalism

(Assuming familiarity with NEGF / Datta, *Electronic Transport in Mesoscopic
Systems*.) Per $k_y$ and energy $E$ we build the retarded GF

$$
G^R(E) = \big[(E+i\eta)\mathbb{I} - H_{\rm dev} - \Sigma_L - \Sigma_R\big]^{-1}
$$

with $\Sigma_{L,R}$ the `kwant` lead self-energies and $\Gamma_\alpha =
i(\Sigma_\alpha-\Sigma_\alpha^\dagger)$. The lesser GF carries the
non-equilibrium occupation,

$$
\Sigma^<(E) = i\big[f_L\Gamma_L + f_R\Gamma_R\big],\qquad
G^<(E) = G^R\,\Sigma^<\,G^A
$$

($f_\alpha=f(E-\mu_\alpha)$). A finite bias enters as the chemical-potential split
$\mu_{L,R}=E_F\pm V/2$ **and**, for the scalar-potential variant `run_negf`, a
ramp $\phi(x)\mathbb{I}$ added to the diagonal. The spin density on a surface
layer is the $G^<$ trace with the spin operator,

$$
\langle S_a\rangle = -i\int\frac{dk_y}{2\pi}\int\frac{dE}{2\pi}\,
\mathrm{Tr}_{\rm surf}\!\big[S_a\,G^<(E,k_y)\big],\qquad
S_a = \tau_z\otimes\sigma_a
$$

In code, $\mathrm{Tr}_{\rm surf}$ sums the $4\times4$ diagonal blocks of $G^<$
over the top `n_surf` $z$-layers; the RGF (`_rgf_glesser_blocks`) returns exactly
those diagonal blocks. CISP $\equiv\langle S_a\rangle(V)-\langle S_a\rangle(0)$.

**Why $S=\tau_z\otimes\sigma$ (not $\tau_0\otimes\sigma$).** The two orbitals have
opposite parity, and the inter-orbital Dirac coupling $\tau_x(\sigma\cdot k)$
dresses the real spin with the orbital-parity factor $\tau_z$. Diagnostically,
the surface eigenstate satisfies $\langle\tau_z\sigma_y\rangle\approx-0.97$,
$\langle\tau_0\sigma_y\rangle\approx 0$ at $k\parallel\hat{x}$ — the
spin-momentum locking is in $\tau_z\otimes\sigma$. It is also the only choice
consistent with the chiral symmetry: $\{\tau_z\sigma_y,\tau_y\}=0$ forces
$S_y(E_F)$ **even**, matching Boltzmann/experiment, whereas
$[\tau_0\sigma_y,\tau_y]=0$ would force it (wrongly) odd.

---

## 5. From NEGF to Kubo to Boltzmann

Linearize the bias. With $\mu_{L,R}=E_F\pm eV/2$,
$f_{L,R}\approx f-(\pm eV/2)(\partial f/\partial E)$, so to first order in $V$
**only the occupation changes** (the no-potential / Kubo route, `run_kubo`):

$$
\delta\Sigma^< = i\,\tfrac{eV}{2}\Big(\!-\frac{\partial f}{\partial E}\Big)(\Gamma_L-\Gamma_R),
\qquad \delta G^< = G^R\,\delta\Sigma^<\,G^A
$$

Insert into $\langle S_a\rangle = -i\int\frac{dE}{2\pi}\mathrm{Tr}[S_aG^<]$
(the $-i\cdot i=1$ is real):

$$
\boxed{\ \delta\langle S_a\rangle = \frac{eV}{2}\int\frac{dk_y}{2\pi}\int\frac{dE}{2\pi}
\Big(\!-\frac{\partial f}{\partial E}\Big)
\mathrm{Tr}_{\rm surf}\!\big[S_a\,G^R(\Gamma_L-\Gamma_R)G^A\big]\ }
\tag{$\star$}
$$

Equation ($\star$) is the **Kubo–Streda Fermi-surface (linear-response) formula**
for the Edelstein response, evaluated by sparse LU in `kubo_cisp_at_ky` (solving
only the two lead-block columns of $G^R$). The accompanying Landauer conductance
is $T(E)=\mathrm{Tr}[\Gamma_L G^R\Gamma_R G^A]$.

**Reduction to Boltzmann.** Write $G^R=\sum_n|n\rangle\langle n|/(E-E_n+i\eta)$.
With diffusive (wide) leads, $\Gamma_{L,R}$ reduce to the boundary projection of
the current operator $j_x=\partial H/\partial k_x$, and the lifetime is
$\tau=\hbar/(2\eta)$. The two spectral functions in ($\star$) collapse onto the
mass shell, $G^R\Gamma G^A\to 2\pi\tau\,A(E)\,j_x\,A(E)$, and ($\star$) becomes
the intraband sum

$$
\delta\langle S_a\rangle \ \propto\ eE_x\tau\sum_{k,\,\rm band}
\langle S_a\rangle_k\,v_{x,k}\Big(\!-\frac{\partial f}{\partial E}\Big)
\qquad\text{(Boltzmann)}
$$

— exactly the relaxation-time Edelstein result (Edelstein, *Solid State Commun.*
**73**, 233 (1990)). So ($\star$) is the quantum (interband-coherent)
generalization of the Boltzmann CISP; they share the same parity in $E_F$ and the
same C₃ angular structure under warping, and agree quantitatively in the
diffusive/linear regime. The finite-$V$ `run_negf` adds the genuine $O(V^2)$
nonlinear pieces (e.g. the longitudinal $S_x$, which is mirror-forbidden at
linear order) that ($\star$) and Boltzmann omit.

---

## References

- H. Zhang *et al.*, *Nat. Phys.* **5**, 438 (2009) — four-band 3D-TI model.
- C.-X. Liu *et al.*, *PRB* **82**, 045122 (2010) — model Hamiltonian, basis,
  C₃ᵥ symmetry.
- L. Fu, *PRL* **103**, 266801 (2009) — hexagonal warping.
- R. Jackiw & C. Rebbi, *PRD* **13**, 3398 (1976) — domain-wall bound states.
- V. M. Edelstein, *Solid State Commun.* **73**, 233 (1990) — CISP.
- S. Datta, *Electronic Transport in Mesoscopic Systems* (1995) — NEGF/RGF.
