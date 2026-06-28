# NEGF for current-induced spin polarization on a 3D-TI surface

Quantum-transport calculation of the **current-induced spin polarization (CISP /
Edelstein effect)** on the surface of a 3D topological insulator, using the
Zhang–Liu four-band model, a real-space/`k`-space slab discretization, `kwant`
lead self-energies, and a recursive-Green's-function (RGF) NEGF solver.
Hexagonal warping (Liang Fu) is included.

---

## 1. The model: Zhang–Liu four-band 3D TI

Low-energy `k·p` Hamiltonian for the Bi₂Se₃ family
(Zhang *et al.*, *Nat. Phys.* **5**, 438 (2009); Liu *et al.*, *PRB* **82**,
045122 (2010)), in the basis `{|P1_z^+,↑⟩, |P2_z^-,↑⟩, |P1_z^+,↓⟩, |P2_z^-,↓⟩}`
with **orbital** Pauli matrices `τ` (the two opposite-parity `p_z` orbitals) and
**spin** Pauli matrices `σ`:

```
H(k) = ε(k) I + M(k) τ_z
       + A∥ [ sin(kx) τ_x σ_x + sin(ky) τ_x σ_y ]
       + A_z  sin(kz) τ_x σ_z
M(k)  = M0 − B∥(2 − cos kx − cos ky) − B_z(1 − cos kz)
ε(k)  = C  + D∥(2 − cos kx − cos ky) + D_z(1 − cos kz)
```

(`τ_a σ_b ≡ kron(τ_a, σ_b)`.) The `ε(k) I` term is particle–hole asymmetry; we
work at `C = D = 0`, where the spectrum is symmetric and the model has an exact
chiral symmetry `Γ = τ_y` (`Γ H Γ = −H`), used throughout for parity arguments.

### Why band inversion forces a surface state (Jackiw–Rebbi)

The topology lives in the **mass** `M(k)`. Keep only the `z`-dependence and
linearize about the surface normal `kz → −i∂_z`:

```
H_z ≈ M(z) τ_z + A_z (−i∂_z) τ_x σ_z ,   M = M0 − B_z(1 − cos kz) → M0 + B_z ∂_z²
```

A topological insulator is the **inverted** regime `M0/B_z > 0`: deep in the bulk
`M → M0 > 0`, while outside the solid the gap is "normal", `M_vac < 0`. So the
mass **changes sign at the surface** — a domain wall. The 1D Dirac equation
`[M(z) τ_z − i A_z ∂_z τ_x σ_z] ψ = E ψ` with a mass kink `M(+∞)>0, M(−∞)<0` has
a **zero-energy bound state** (Jackiw & Rebbi, *PRD* **13**, 3398 (1976)):

```
ψ_s(z) ∝ exp(−∫₀ᶻ M(z')/A_z dz') · χ ,   with  (τ_y) χ = +χ
```

i.e. an exponentially localized mode pinned to the surface and locked to a fixed
eigenvalue of the chiral operator `τ_y` (top vs bottom surface = `τ_y = ±1`).
Projecting the in-plane Dirac terms `A∥ τ_x(σ_x kx + σ_y ky)` onto this `τ_y`
doublet gives the gapless **surface Dirac cone**

```
H_surf = A∥ (kx σ_y − ky σ_x)            (spin–momentum locked)
```

— a single Dirac cone, protected because the bound state cannot be removed
without unwinding the bulk mass inversion. The bound state, not any boundary
condition we impose, is what makes the surface metallic.

---

## 2. Slab discretization

Geometry (`slab_matrices` in `negf_kwant.py`):

- **z**: finite, `Nz` layers (hosts top/bottom surface states) — real space.
- **y**: translationally invariant → Fourier transform, `ky` is a good quantum
  number; we integrate over `ky ∈ [−ky_max, ky_max]`.
- **x**: transport direction — real-space chain of `Nx` sites + two
  semi-infinite leads.

Each site carries the 4-spinor; a "slice" (fixed `x`) is a `4Nz × 4Nz` block.
The lattice Hamiltonian follows from the standard substitution

```
sin k →  (e^{ik} − e^{−ik})/2i   (a hop, ± i/2)
cos k →  1 − (e^{ik} + e^{−ik})/2 (on-site + a real ±1/2 hop)
```

**On-diagonal block** `H_on(ky)` (intra-slice, `4Nz × 4Nz`):

```
H_on(ky) = Σ_z  on_k(ky)            (per-layer 4×4 on-site)
         + Σ_z  [ Hz  between z and z+1 layers + h.c. ]   (z-chain)
on_k(ky) = H0 + Hy e^{iky} + Hy† e^{−iky}
```

where `H0 = onsite_4x4` (the `ε0 I + m τ_z` piece, `m = M0 − 2B∥ − B_z`) and
`Hy = hop_4x4('y')` carries the `ky`-Bloch phase. The z-direction is an explicit
1D tight-binding chain inside the block (so surface states are resolved).

**Off-diagonal block** `V_hop` (slice `x → x+1`, `4Nz × 4Nz`, block-diagonal in
`z`): `V_hop = Σ_z hop_4x4('x')`, i.e. the `+x̂` hop
`−D∥/2 τ_0 + B∥/2 τ_z − i A∥/2 τ_x σ_x` on each layer. It is `ky`-independent.

The device Hamiltonian is block-tridiagonal in `x`:
`diag = H_on (+ φ(x) I for a bias)`, `upper/lower = V_hop / V_hop†`. This
structure is exactly what the RGF exploits, cost `O(Nx · (4Nz)³)`.

---

## 3. Hexagonal warping and why it needs next-nearest neighbors

Liang Fu's warping (Fu, *PRL* **103**, 266801 (2009)) is the leading C₃-symmetric
correction to the surface Dirac cone,

```
H_w = λ (kx³ − 3 kx ky²) σ_z = λ Re(k₊³) σ_z ,   k₊ = kx + i ky
```

In the four-band basis the physical `σ_z` is `τ_z ⊗ σ_z` (see §4), so
`H_w = λ Re(k₊³) (τ_z⊗σ_z)`. The cubic-in-`kx` factor is the issue: a faithful
lattice regularization needs **two harmonics in `kx`**,

```
kx³ − 3 kx ky²  ⟷  −2(2 − 3 cos ky) sin kx − sin 2kx
```

The `sin 2kx` term is a **next-nearest-neighbor (NNN / 2NN) hop in x**
(connects slice `i → i+2`). Concretely `slab_matrices` returns a third matrix
`V2_hop = (i λ/2) (τ_z⊗σ_z)` for the `i→i+2` coupling, alongside the NN piece
`V_warp_NN = i λ (2 − 3 cos ky)(τ_z⊗σ_z)` folded into `V_hop`.

Consequences:
- The device is no longer block-**tridiagonal** (it has a second off-diagonal),
  so the plain RGF/sparse solver must be adapted: we **super-block** pairs of
  `x`-slices (block size `2·4Nz`) to restore tridiagonal form, and the `kwant`
  leads use a **doubled unit cell**.
- **Crystal-angle rotation.** Rotating the current by `θ` w.r.t. the crystal
  rotates only the warping (the Dirac part is isotropic): `Re(k₊³) →
  Re(e^{−i3θ} k₊³) = cos3θ (kx³−3kxky²) + sin3θ (3kx²ky − ky³)`. The new
  `Im(k₊³) = 3kx²ky − ky³` piece adds `ky`-dependent on-site terms and a real
  NN `x`-hop (`warp_angle` parameter). Everything is `2π/3`-periodic in `θ`.

---

## 4. Measuring spin in the NEGF formalism

(Assuming familiarity with NEGF / Datta, *Electronic Transport in Mesoscopic
Systems*.) Per `ky` and energy `E` we build the retarded GF

```
G^R(E) = [ (E + iη) I − H_dev − Σ_L − Σ_R ]^{-1}
```

with `Σ_{L,R}` the `kwant` lead self-energies, `Γ_α = i(Σ_α − Σ_α†)`. The
lesser GF carries the non-equilibrium occupation,

```
Σ^<(E) = i [ f_L Γ_L + f_R Γ_R ] ,   G^<(E) = G^R Σ^< G^A
```

(`f_α = f(E − μ_α)`). A finite bias enters as the chemical-potential split
`μ_{L,R} = E_F ± V/2` **and**, for the scalar-potential variant `run_negf`, a
ramp `φ(x) I` added to the diagonal. The spin density on a surface layer is the
`G^<` trace with the spin operator,

```
⟨S_a⟩ = −i ∫ (dky/2π) ∫ (dE/2π)  Tr_surf[ S_a G^<(E,ky) ]
S_a = τ_z ⊗ σ_a            ← physical spin (see below)
```

In code, `Tr_surf` sums the `4×4` diagonal blocks of `G^<` over the top
`n_surf` z-layers; the RGF (`_rgf_glesser_blocks`) returns exactly those
diagonal blocks. CISP ≡ `⟨S_a⟩(V) − ⟨S_a⟩(0)`.

**Why `S = τ_z ⊗ σ` (not `τ_0 ⊗ σ`).** The two orbitals have opposite parity,
and the inter-orbital Dirac coupling `τ_x(σ·k)` dresses the real spin with the
orbital-parity factor `τ_z`. Diagnostically, the surface eigenstate satisfies
`⟨τ_z⊗σ_y⟩ ≈ −0.97`, `⟨τ_0⊗σ_y⟩ ≈ 0` at `k∥x̂` — the spin-momentum locking is
in `τ_z⊗σ`. It is also the only choice consistent with the chiral symmetry:
`{τ_z⊗σ_y, τ_y} = 0` forces `Sy(E_F)` **even**, matching Boltzmann/experiment,
whereas `[τ_0⊗σ_y, τ_y] = 0` would force it (wrongly) odd.

---

## 5. From NEGF to Kubo to Boltzmann

Linearize the bias. With `μ_{L,R} = E_F ± eV/2`,
`f_{L,R} ≈ f − (±eV/2)(∂f/∂E)`, so to first order in `V` **only the occupation
changes** (the no-potential / Kubo route, `run_kubo`):

```
δΣ^< = i (eV/2)(−∂f/∂E)(Γ_L − Γ_R)
δG^< = G^R δΣ^< G^A
```

Insert into `⟨S_a⟩ = −i ∫ dE/2π Tr[S_a G^<]` (the `−i·i = 1` is real):

```
δ⟨S_a⟩ = (eV/2) ∫ (dky/2π) ∫ (dE/2π) (−∂f/∂E)
                 Tr_surf[ S_a G^R (Γ_L − Γ_R) G^A ]        (★)
```

Equation (★) is the **Kubo–Streda Fermi-surface (linear-response) formula** for
the Edelstein response, evaluated by sparse LU in `kubo_cisp_at_ky` (solving
only the two lead-block columns of `G^R`). The accompanying Landauer
conductance is `T(E) = Tr[Γ_L G^R Γ_R G^A]`.

**Reduction to Boltzmann.** Write `G^R = Σ_n |n⟩⟨n|/(E − E_n + iη)`. With
diffusive (wide) leads, `Γ_{L,R}` reduce to the boundary projection of the
current operator `j_x = ∂H/∂kx`, and the lifetime is `τ = ħ/(2η)`. The two
spectral functions in (★) collapse onto the mass shell, `G^R Γ G^A → 2πτ
A(E) j_x A(E)`, and (★) becomes the intraband sum

```
δ⟨S_a⟩ ∝ eEx τ  Σ_{k,band}  ⟨S_a⟩_k  v_{x,k} (−∂f/∂E)        (Boltzmann)
```

— exactly the relaxation-time Edelstein result (Edelstein, *Solid State Commun.*
**73**, 233 (1990)). So (★) is the quantum (interband-coherent) generalization
of the Boltzmann CISP; they share the same parity in `E_F` and the same C₃
angular structure under warping, and agree quantitatively in the
diffusive/linear regime. The finite-`V` `run_negf` adds the genuine `O(V²)`
nonlinear pieces (e.g. the longitudinal `Sx`, which is mirror-forbidden at
linear order) that (★) and Boltzmann omit.

---

## References

- H. Zhang *et al.*, *Nat. Phys.* **5**, 438 (2009) — four-band 3D-TI model.
- C.-X. Liu *et al.*, *PRB* **82**, 045122 (2010) — model Hamiltonian, basis,
  C₃ᵥ symmetry.
- L. Fu, *PRL* **103**, 266801 (2009) — hexagonal warping.
- R. Jackiw & C. Rebbi, *PRD* **13**, 3398 (1976) — domain-wall bound states.
- V. M. Edelstein, *Solid State Commun.* **73**, 233 (1990) — CISP.
- S. Datta, *Electronic Transport in Mesoscopic Systems* (1995) — NEGF/RGF.
