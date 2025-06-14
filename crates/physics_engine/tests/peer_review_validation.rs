use physics_engine::interactions;

// PDG 2024 expected Klein-Nishina Compton cross section for 1 MeV photon on free electron ~0.066 barn = 6.6e-29 m^2
const KN_EXPECTED: f64 = 6.6e-29;
const KN_TOLERANCE: f64 = 2.0; // factor

// Bethe-Heitler pair production on heavy nucleus (Z~82) at 10 MeV ≈ 0.5 barn. We'll only check order of magnitude.
const BH_EXPECTED: f64 = 5.0e-28; // 0.5 barn
const BH_TOLERANCE: f64 = 5.0;

#[test]
fn validate_compton_cross_section() {
    let energy_gev = 0.001; // 1 MeV
    let sigma = interactions::klein_nishina_cross_section(energy_gev);
    assert!(sigma > KN_EXPECTED / KN_TOLERANCE && sigma < KN_EXPECTED * KN_TOLERANCE,
        "Compton σ out of expected range: {}", sigma);
}

#[test]
fn validate_pair_production_cross_section() {
    let energy_gev = 0.01; // 10 MeV
    let sigma = interactions::bethe_heitler_pair_production(energy_gev, 82); // lead nucleus
    assert!(sigma > BH_EXPECTED / BH_TOLERANCE && sigma < BH_EXPECTED * BH_TOLERANCE,
        "Bethe–Heitler σ out of expected range: {}", sigma);
}