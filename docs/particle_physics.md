# Particle Physics Foundation

This document describes the particle physics implementation in our universe evolution simulator, based on the 2024 Particle Data Group (PDG) review.

## References

Primary source: Particle Data Group, R.L. Workman et al. (Particle Data Group), Prog. Theor. Exp. Phys. 2024, 083C01 (2024)
DOI: [10.1093/ptep/ptac097](https://doi.org/10.1093/ptep/ptac097)
arXiv: [2311.15995](https://arxiv.org/abs/2311.15995)

## Fundamental Constants

All values from PDG 2024 unless otherwise noted.

| Constant | Symbol | Value | Units |
|----------|--------|-------|-------|
| Speed of light | c | 299 792 458 | m/s (exact) |
| Planck constant | ℏ | 1.054 571 817 × 10⁻³⁴ | J·s |
| Elementary charge | e | 1.602 176 634 × 10⁻¹⁹ | C (exact) |
| Fine structure constant | α | 7.297 352 5693(11) × 10⁻³ | - |
| Gravitational constant | G | 6.674 30(15) × 10⁻¹¹ | m³ kg⁻¹ s⁻² |
| Boltzmann constant | k_B | 1.380 649 × 10⁻²³ | J/K (exact) |

## Standard Model Particles

### Quarks

| Particle | Mass (MeV/c²) | Charge (e) | Spin | Color | Width/Lifetime |
|----------|---------------|------------|------|-------|----------------|
| u (up) | 2.16⁺⁰·⁴⁹₋₀.₂₆ | +2/3 | 1/2 | Yes | Stable* |
| d (down) | 4.67⁺⁰·⁴⁸₋₀.₁₇ | -1/3 | 1/2 | Yes | Stable* |
| c (charm) | 1.27 ± 0.02 GeV | +2/3 | 1/2 | Yes | τ ~ 1.3 × 10⁻¹² s |
| s (strange) | 93⁺¹¹₋₅ | -1/3 | 1/2 | Yes | Stable* |
| t (top) | 172.69 ± 0.30 GeV | +2/3 | 1/2 | Yes | τ ~ 5 × 10⁻²⁵ s |
| b (bottom) | 4.18⁺⁰·⁰³₋₀.₀₂ GeV | -1/3 | 1/2 | Yes | τ ~ 1.6 × 10⁻¹² s |

*Stable within hadrons; free quarks do not exist due to confinement.

### Leptons

| Particle | Mass | Charge (e) | Spin | Lifetime |
|----------|------|------------|------|----------|
| e⁻ (electron) | 0.510 998 950 00(15) MeV | -1 | 1/2 | Stable |
| ν_e (electron neutrino) | < 0.8 eV | 0 | 1/2 | Stable |
| μ⁻ (muon) | 105.658 3745(24) MeV | -1 | 1/2 | 2.196 9811(22) μs |
| ν_μ (muon neutrino) | < 0.19 MeV | 0 | 1/2 | Stable |
| τ⁻ (tau) | 1776.86 ± 0.12 MeV | -1 | 1/2 | 290.3(5) × 10⁻¹⁵ s |
| ν_τ (tau neutrino) | < 18.2 MeV | 0 | 1/2 | Stable |

### Gauge Bosons

| Particle | Mass | Charge (e) | Spin | Width | Mediates |
|----------|------|------------|------|-------|----------|
| γ (photon) | 0 | 0 | 1 | - | Electromagnetic |
| g (gluon) | 0 | 0 | 1 | - | Strong |
| W± | 80.379 ± 0.012 GeV | ±1 | 1 | 2.085 ± 0.042 GeV | Weak |
| Z⁰ | 91.1876 ± 0.0021 GeV | 0 | 1 | 2.4952 ± 0.0023 GeV | Weak |

### Higgs Boson

| Property | Value |
|----------|-------|
| Mass | 125.25 ± 0.17 GeV |
| Charge | 0 |
| Spin | 0 |
| Width | 3.2⁺²·⁴₋₂.₂ MeV |

## CKM Matrix

The Cabibbo-Kobayashi-Maskawa matrix describes quark mixing in weak interactions.

### Wolfenstein Parameters (PDG 2024)
- λ = 0.22453 ± 0.00044
- A = 0.836 ± 0.015
- ρ̄ = 0.122⁺⁰·⁰¹⁸₋₀.₀₁₇
- η̄ = 0.355⁺⁰·⁰¹²₋₀.₀₁₁

### CKM Matrix Elements (absolute values)
```
|V_ud| = 0.97420 ± 0.00021    |V_us| = 0.2243 ± 0.0005     |V_ub| = 0.00394 ± 0.00036
|V_cd| = 0.218 ± 0.004        |V_cs| = 0.997 ± 0.017       |V_cb| = 0.0422 ± 0.0008
|V_td| = 0.0081 ± 0.0005      |V_ts| = 0.0394 ± 0.0023     |V_tb| = 0.999105 ± 0.000032
```

### CP Violation
- Jarlskog invariant: J = (3.04⁺⁰·²¹₋₀.₂₀) × 10⁻⁵
- CP-violating phase: δ = (1.20 ± 0.08) rad

## Electroweak Parameters

### Weak Mixing Angle
- sin²θ_W (M_Z, MS̄) = 0.23122 ± 0.00003

### Running Couplings
- α(M_Z) = 1/127.951 ± 0.009
- α_s(M_Z) = 0.1179 ± 0.0009

## Major Decay Channels

### Muon Decay
- μ⁻ → e⁻ ν̄_e ν_μ: BR = 100%

### Tau Decay (main channels)
- τ⁻ → e⁻ ν̄_e ν_τ: BR = 17.83 ± 0.04%
- τ⁻ → μ⁻ ν̄_μ ν_τ: BR = 17.41 ± 0.04%
- τ⁻ → π⁻ ν_τ: BR = 10.83 ± 0.06%
- τ⁻ → π⁻ π⁰ ν_τ: BR = 25.52 ± 0.09%

### W Boson Decay
- W → l ν_l (each): BR ≈ 10.8%
- W → hadrons: BR ≈ 67.6%

### Z Boson Decay
- Z → l⁺l⁻ (each): BR ≈ 3.37%
- Z → invisible: BR = 20.00 ± 0.06%
- Z → hadrons: BR = 69.91 ± 0.06%

### Top Quark Decay
- t → W b: BR ≈ 100%

### Higgs Decay (m_H = 125 GeV)
- H → bb̄: BR ≈ 58%
- H → WW*: BR ≈ 21%
- H → gg: BR ≈ 8%
- H → ττ: BR ≈ 6%
- H → cc̄: BR ≈ 3%
- H → ZZ*: BR ≈ 3%
- H → γγ: BR ≈ 0.2%

## QED Interactions Implementation

### Compton Scattering (γ + e⁻ → γ + e⁻)

The simulator implements Compton scattering using the Klein-Nishina formula for the differential cross-section:

```
σ_KN(ε) = 2πr_e² [(1+ε)/ε³ {2(1+ε)/(1+2ε) - ln(1+2ε)/ε} + ln(1+2ε)/(2ε) - (1+3ε)/(1+2ε)²]
```

Where:
- ε = E_γ/(m_e c²) is the dimensionless photon energy
- r_e = e²/(4πε₀ m_e c²) is the classical electron radius

The scattering angle is sampled using rejection sampling from the Klein-Nishina angular distribution.

### Pair Production (γ → e⁺ + e⁻)

Pair production in the field of a nucleus is implemented using the Bethe-Heitler cross-section:

```
σ_BH(E_γ, Z) = 4αr_e² Z² [ln(183/Z^(1/3)) - f_c]
```

Where:
- Z is the atomic number of the nucleus
- f_c = α²Z² is the Coulomb correction factor
- The formula is valid for complete screening (high energy limit)

### Implementation Details

1. **Interaction Range**: Set to ~1 fm (10⁻¹⁵ m) for QED processes
2. **Probability Calculation**: P = 1 - exp(-σnvΔt) where:
   - σ = cross-section
   - n = number density
   - v = relative velocity (c for photons)
   - Δt = time step

3. **Energy-Momentum Conservation**: Strictly enforced in all interactions
4. **Relativistic Kinematics**: Full relativistic treatment with E² = (pc)² + (mc²)²

### Validation

The implementation has been validated against:
- Thomson limit (low energy): σ → (8π/3)r_e² 
- Klein-Nishina high-energy limit: σ ∝ ln(ε)/ε
- Pair production threshold: E_γ > 2m_e c² = 1.022 MeV

## References

1. Particle Data Group, R.L. Workman et al., Prog. Theor. Exp. Phys. 2024, 083C01 (2024)
2. Klein, O. and Nishina, Y., Z. Phys. 52, 853 (1929)
3. Bethe, H. and Heitler, W., Proc. R. Soc. Lond. A 146, 83 (1934)
4. Peskin, M.E. and Schroeder, D.V., "An Introduction to Quantum Field Theory" (1995)

## Implementation Notes

### Mass-Energy Conversion
Particle masses are stored in kg using the conversion:
- 1 GeV/c² = 1.78266192 × 10⁻²⁷ kg

### Decay Width to Lifetime
For unstable particles:
- τ = ℏ/Γ
- where Γ is the total decay width

### QCD Running Coupling
The strong coupling α_s runs with energy scale Q according to:
```
α_s(Q) = α_s(M_Z) / [1 + β₀ α_s(M_Z) ln(Q²/M_Z²) / (2π)]
```
where β₀ = (33 - 2n_f)/(12π) and n_f is the number of active quark flavors.

### Numerical Precision
All calculations maintain at least 6 significant figures to ensure conservation laws are satisfied to within numerical precision limits.

## Validation

The implementation has been validated against:
1. PDG 2024 particle properties tables
2. Conservation of energy-momentum in all interactions
3. Unitarity of the CKM matrix
4. CPT symmetry requirements

## Future Enhancements

1. **Neutrino Oscillations**: Implement PMNS matrix and mass differences
2. **QCD Confinement**: Add color confinement dynamics
3. **Electroweak Symmetry Breaking**: Full Higgs mechanism implementation
4. **Beyond Standard Model**: Options for SUSY, extra dimensions, etc.

## References for Specific Values

1. Quark masses: PDG 2024, Chapter 66 (Quark Masses)
2. CKM parameters: PDG 2024, Chapter 12 (CKM Quark-Mixing Matrix)
3. Electroweak parameters: PDG 2024, Chapter 10 (Electroweak Model)
4. QCD parameters: PDG 2024, Chapter 9 (Quantum Chromodynamics)
5. Higgs properties: PDG 2024, Chapter 26 (Higgs Boson)

All uncertainties quoted are 1σ unless otherwise specified.