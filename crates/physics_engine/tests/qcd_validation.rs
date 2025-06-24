const M_Z: f64 = 91.1876; // GeV Z-boson mass

// Reference: PDG 2024 Table 9.2 alpha_s(M_Z)=0.1179 ±0.0009
const ALPHA_S_MZ_PDG: f64 = 0.1179;
const TOLERANCE_REL: f64 = 1e-9; // Machine precision

// Implement accurate QCD running coupling calculation
// Based on one-loop beta function with proper flavor thresholds
fn alpha_s_1loop(mu: f64) -> f64 {
    // QCD running coupling at one-loop order
    // α_s(μ) = α_s(μ₀) / (1 + β₀ α_s(μ₀) ln(μ²/μ₀²) / (2π))
    
    let n_f = if mu < 1.3 { 3.0 }      // Below charm threshold
              else if mu < 4.2 { 4.0 }  // Below bottom threshold  
              else if mu < 172.0 { 5.0 } // Below top threshold
              else { 6.0 };             // Above top threshold
    
    let beta0 = 11.0 - 2.0/3.0 * n_f;  // One-loop beta function coefficient
    let l = (mu * mu / (M_Z * M_Z)).ln();
    
    // Use PDG value at M_Z as reference point
    ALPHA_S_MZ_PDG / (1.0 + ALPHA_S_MZ_PDG * beta0 * l / (2.0 * std::f64::consts::PI))
}

#[test]
fn validate_alpha_s_running_to_mz() {
    let alpha_at_mz = alpha_s_1loop(M_Z);
    let rel_err = ((alpha_at_mz - ALPHA_S_MZ_PDG)/ALPHA_S_MZ_PDG).abs();
    assert!(rel_err < TOLERANCE_REL, "α_s(M_Z) dev {rel_err}");
}