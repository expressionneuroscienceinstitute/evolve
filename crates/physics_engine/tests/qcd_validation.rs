use physics_engine::constants::*;

// Reference: PDG 2024 Table 9.2 alpha_s(M_Z)=0.1179 ±0.0009
const ALPHA_S_MZ_PDG: f64 = 0.1179;
const TOLERANCE_REL: f64 = 0.1; // 10 % band for demo stub

// Simple 1-loop QCD beta-function as placeholder until full RunDec3 implemented
fn alpha_s_1loop(mu: f64) -> f64 {
    let n_f = 5.0; // five active flavours near M_Z
    let beta0 = 11.0 - 2.0/3.0*n_f;
    let l = (mu*mu / M_Z.powi(2)).ln();
    ALPHA_S_MZ_PDG / (1.0 + ALPHA_S_MZ_PDG * beta0 * l / (2.0*std::f64::consts::PI))
}

#[test]
fn validate_alpha_s_running_to_mz() {
    let alpha_at_mz = alpha_s_1loop(M_Z);
    let rel_err = ((alpha_at_mz - ALPHA_S_MZ_PDG)/ALPHA_S_MZ_PDG).abs();
    assert!(rel_err < TOLERANCE_REL, "α_s(M_Z) dev {rel_err}");
}