# Particle-Physics Foundation

This simulator's fundamental layer follows the 2024 Particle-Data-Group review (PDG 2024; *Prog. Theor. Exp. Phys.* **2024**, 083C01, doi:10.48550/arXiv.2311.15995).

## Standard-Model constants

| Symbol | PDG 2024 value |
|--------|----------------|
| c | 299 792 458 m s⁻¹ (exact) |
| ℏ | 1.054 571 817 × 10⁻³⁴ J s |
| e | 1.602 176 634 × 10⁻¹⁹ C |
| α | 7.297 352 5693 × 10⁻³ |

## Fermions & Bosons (subset)

| Particle | m [GeV/c²] | Q [e] | Spin | Γ or τ | Notes |
|----------|-----------|-------|------|--------|-------|
| u | 0.00216 | +2/3 | 1/2 | — | colour |
| d | 0.00467 | −1/3 | 1/2 | — | colour |
| e⁻ | 0.000511 | −1 | 1/2 | stable | — |
| μ⁻ | 0.10566 | −1 | 1/2 | τ = 2.197 µs | BR(μ→e ν̄_e ν_μ)=1 |
| τ⁻ | 1.77686 | −1 | 1/2 | τ = 2.903×10⁻¹³ s | see code |
| W⁺ | 80.379 | +1 | 1 | Γ = 2.085 GeV | BR(lν)=10.8 % |
| Z | 91.1876 | 0 | 1 | Γ = 2.495 GeV | |
| H | 125.25 | 0 | 0 | Γ = 4.1 MeV | |

*Full table lives in `physics_engine::particles`.*

## CKM matrix (|V_ij|)

```
|V_ud| = 0.97420 ± 0.00021   |V_us| = 0.2243 ± 0.0005   |V_ub| = 0.00394 ± 0.00036
|V_cd| = 0.218   ± 0.004     |V_cs| = 0.997  ± 0.017    |V_cb| = 0.0422 ± 0.0008
|V_td| = 0.0081  ± 0.0005    |V_ts| = 0.041  ± 0.0006   |V_tb| = 0.99910 ± 0.00014
```

All values are inserted verbatim in the code for reproducibility.

---
Generated automatically by Cursor AI using PDG-2024 data.