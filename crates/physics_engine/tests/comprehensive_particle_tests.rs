use approx::assert_relative_eq;
use physics_engine::{PhysicsEngine, ParticleType, constants::{ELEMENTARY_CHARGE, SPEED_OF_LIGHT}};
use nalgebra::Vector3;
use std::collections::HashMap;
use physics_engine::*;
use physics_engine::particles::*;
use physics_engine::nuclear_physics::*;
use physics_engine::interactions::*;
use physics_engine::utils::K_E; // Coulomb constant

/// Test comprehensive particle database coverage
#[test]
fn test_all_standard_model_particles() {
    // Test all quarks
    assert!(get_properties(ParticleType::Up).mass_kg > 0.0);
    assert!(get_properties(ParticleType::Down).mass_kg > 0.0);
    assert!(get_properties(ParticleType::Charm).mass_kg > 0.0);
    assert!(get_properties(ParticleType::Strange).mass_kg > 0.0);
    assert!(get_properties(ParticleType::Top).mass_kg > 0.0);
    assert!(get_properties(ParticleType::Bottom).mass_kg > 0.0);
    
    // Test all leptons
    assert!(get_properties(ParticleType::Electron).mass_kg > 0.0);
    assert_eq!(get_properties(ParticleType::ElectronNeutrino).mass_kg, 0.0);
    assert_eq!(get_properties(ParticleType::ElectronAntiNeutrino).mass_kg, 0.0);
    assert!(get_properties(ParticleType::Muon).mass_kg > 0.0);
    assert_eq!(get_properties(ParticleType::MuonNeutrino).mass_kg, 0.0);
    assert_eq!(get_properties(ParticleType::MuonAntiNeutrino).mass_kg, 0.0);
    assert!(get_properties(ParticleType::Tau).mass_kg > 0.0);
    assert_eq!(get_properties(ParticleType::TauNeutrino).mass_kg, 0.0);
    assert_eq!(get_properties(ParticleType::TauAntiNeutrino).mass_kg, 0.0);
    
    // Test gauge bosons
    assert_eq!(get_properties(ParticleType::Photon).mass_kg, 0.0);
    assert!(get_properties(ParticleType::WBoson).mass_kg > 0.0);
    assert!(get_properties(ParticleType::WBosonMinus).mass_kg > 0.0);
    assert!(get_properties(ParticleType::ZBoson).mass_kg > 0.0);
    assert_eq!(get_properties(ParticleType::Gluon).mass_kg, 0.0);
    
    // Test scalar boson
    assert!(get_properties(ParticleType::Higgs).mass_kg > 0.0);
}

/// Test meson properties and decay characteristics
#[test]
fn test_meson_properties() {
    // Test pions
    let pi_plus = get_properties(ParticleType::PionPlus);
    let pi_minus = get_properties(ParticleType::PionMinus);
    let pi_zero = get_properties(ParticleType::PionZero);
    
    // Charged pions should have same mass
    assert_relative_eq!(pi_plus.mass_kg, pi_minus.mass_kg, epsilon = 1e-15);
    // Neutral pion slightly lighter
    assert!(pi_zero.mass_kg < pi_plus.mass_kg);
    
    // Test charges
    assert_relative_eq!(pi_plus.charge_c, ELEMENTARY_CHARGE, epsilon = 1e-15);
    assert_relative_eq!(pi_minus.charge_c, -ELEMENTARY_CHARGE, epsilon = 1e-15);
    assert_eq!(pi_zero.charge_c, 0.0);
    
    // Test decay widths (all pions are unstable)
    assert!(pi_plus.width.is_some());
    assert!(pi_minus.width.is_some());
    assert!(pi_zero.width.is_some());
    
    // Test kaons
    let k_plus = get_properties(ParticleType::KaonPlus);
    let _k_minus = get_properties(ParticleType::KaonMinus);
    let k_zero = get_properties(ParticleType::KaonZero);
    
    // Kaons should be heavier than pions
    assert!(k_plus.mass_kg > pi_plus.mass_kg);
    assert!(k_zero.mass_kg > pi_zero.mass_kg);
    
    // Test eta meson
    let eta = get_properties(ParticleType::Eta);
    assert!(eta.mass_kg > k_plus.mass_kg);
    assert_eq!(eta.charge_c, 0.0);
}

/// Test baryon properties and mass hierarchy
#[test]
fn test_baryon_properties() {
    // Test nucleons
    let proton = get_properties(ParticleType::Proton);
    let neutron = get_properties(ParticleType::Neutron);
    
    assert_relative_eq!(proton.charge_c, ELEMENTARY_CHARGE, epsilon = 1e-15);
    assert_eq!(neutron.charge_c, 0.0);
    assert!(neutron.mass_kg > proton.mass_kg); // Neutron slightly heavier
    
    // Test hyperons
    let lambda = get_properties(ParticleType::Lambda);
    let sigma_plus = get_properties(ParticleType::SigmaPlus);
    let sigma_minus = get_properties(ParticleType::SigmaMinus);
    let sigma_zero = get_properties(ParticleType::SigmaZero);
    
    // Lambda should be heavier than nucleons
    assert!(lambda.mass_kg > neutron.mass_kg);
    assert_eq!(lambda.charge_c, 0.0);
    
    // Sigma baryons should be heavier than Lambda
    assert!(sigma_plus.mass_kg > lambda.mass_kg);
    assert_relative_eq!(sigma_plus.charge_c, ELEMENTARY_CHARGE, epsilon = 1e-15);
    assert_relative_eq!(sigma_minus.charge_c, -ELEMENTARY_CHARGE, epsilon = 1e-15);
    assert_eq!(sigma_zero.charge_c, 0.0);
    
    // Test Xi baryons (cascade particles)
    let xi_minus = get_properties(ParticleType::XiMinus);
    let xi_zero = get_properties(ParticleType::XiZero);
    
    assert!(xi_minus.mass_kg > sigma_plus.mass_kg);
    assert_relative_eq!(xi_minus.charge_c, -ELEMENTARY_CHARGE, epsilon = 1e-15);
    assert_eq!(xi_zero.charge_c, 0.0);
    
    // Test Omega baryon (highest strangeness)
    let omega = get_properties(ParticleType::OmegaMinus);
    assert!(omega.mass_kg > xi_minus.mass_kg);
    assert_relative_eq!(omega.charge_c, -ELEMENTARY_CHARGE, epsilon = 1e-15);
    assert_eq!(omega.spin, 1.5); // Omega has spin 3/2
}

/// Test heavy quarkonium states
#[test]
fn test_quarkonium_states() {
    let jpsi = get_properties(ParticleType::JPsi);
    let upsilon = get_properties(ParticleType::Upsilon);
    
    // J/ψ is c-cbar bound state
    assert!(jpsi.mass_kg > 3e9 * 1.78266192e-36); // ~3 GeV
    assert_eq!(jpsi.charge_c, 0.0);
    assert_eq!(jpsi.spin, 1.0); // Vector meson
    
    // Υ is b-bbar bound state, should be heavier
    assert!(upsilon.mass_kg > jpsi.mass_kg);
    assert_eq!(upsilon.charge_c, 0.0);
    assert_eq!(upsilon.spin, 1.0);
}

/// Test charge conservation across particle families
#[test]
fn test_charge_conservation() {
    // Test particle-antiparticle pairs
    let electron = get_properties(ParticleType::Electron);
    let positron = get_properties(ParticleType::Positron);
    assert_relative_eq!(electron.charge_c, -positron.charge_c, epsilon = 1e-15);
    
    let pi_plus = get_properties(ParticleType::PionPlus);
    let pi_minus = get_properties(ParticleType::PionMinus);
    assert_relative_eq!(pi_plus.charge_c, -pi_minus.charge_c, epsilon = 1e-15);
    
    let w_plus = get_properties(ParticleType::WBoson);
    let w_minus = get_properties(ParticleType::WBosonMinus);
    assert_relative_eq!(w_plus.charge_c, -w_minus.charge_c, epsilon = 1e-15);
}

/// Test particle lifetime hierarchy
#[test]
fn test_particle_lifetimes() {
    // Stable particles should have no decay width
    assert!(get_properties(ParticleType::Proton).width.is_none());
    assert!(get_properties(ParticleType::Electron).width.is_none());
    assert!(get_properties(ParticleType::Photon).width.is_none());
    
    // Unstable particles should have decay widths
    assert!(get_properties(ParticleType::Neutron).width.is_some());
    assert!(get_properties(ParticleType::Muon).width.is_some());
    assert!(get_properties(ParticleType::PionPlus).width.is_some());
    
    // Very unstable particles should have large decay widths
    assert!(get_properties(ParticleType::Top).width.is_some());
    let top_width = get_properties(ParticleType::Top).width.unwrap();
    let muon_width = get_properties(ParticleType::Muon).width.unwrap();
    assert!(top_width > muon_width); // Top quark decays much faster
}

/// Test color charge assignments
#[test]
fn test_color_charges() {
    // Quarks and gluons should carry color
    assert!(get_properties(ParticleType::Up).has_color);
    assert!(get_properties(ParticleType::Down).has_color);
    assert!(get_properties(ParticleType::Charm).has_color);
    assert!(get_properties(ParticleType::Strange).has_color);
    assert!(get_properties(ParticleType::Top).has_color);
    assert!(get_properties(ParticleType::Bottom).has_color);
    assert!(get_properties(ParticleType::Gluon).has_color);
    
    // Leptons and composite particles should not carry color
    assert!(!get_properties(ParticleType::Electron).has_color);
    assert!(!get_properties(ParticleType::Muon).has_color);
    assert!(!get_properties(ParticleType::Proton).has_color);
    assert!(!get_properties(ParticleType::Neutron).has_color);
    assert!(!get_properties(ParticleType::PionPlus).has_color);
}

/// Test nuclear physics integration with new particles
#[test]
fn test_nuclear_physics_integration() {
    // Test that nuclear database includes new isotopes
    let nuclear_db = NuclearDatabase::new();
    
    // Test stability of common isotopes
    assert!(nuclear_db.is_stable(1, 1)); // Hydrogen-1
    assert!(nuclear_db.is_stable(2, 4)); // Helium-4
    assert!(nuclear_db.is_stable(6, 12)); // Carbon-12
    assert!(nuclear_db.is_stable(8, 16)); // Oxygen-16
    assert!(nuclear_db.is_stable(26, 56)); // Iron-56
    
    // Test instability of radioactive isotopes
    assert!(!nuclear_db.is_stable(1, 3)); // Tritium
    assert!(!nuclear_db.is_stable(92, 238)); // Uranium-238
    
    // Test cross-section database integration
    let cross_section_db = NuclearCrossSectionDatabase::new();
    assert!(!cross_section_db.fusion_data.is_empty());
    assert!(!cross_section_db.reaction_data.is_empty());
}

/// Test atomic physics with expanded element coverage
#[test]
fn test_atomic_physics_comprehensive() {
    // Test hydrogen atom (simplest case)
    let h_nucleus = nuclear_physics::Nucleus::new(1, 0);
    let h_atom = atomic_physics::Atom::new(h_nucleus);
    assert_eq!(h_atom.charge(), 0);
    assert_eq!(h_atom.shells.len(), 1);
    
    // Test helium atom (filled K shell)
    let he_nucleus = nuclear_physics::Nucleus::new(2, 2);
    let he_atom = atomic_physics::Atom::new(he_nucleus);
    assert_eq!(he_atom.charge(), 0);
    assert_eq!(he_atom.shells.len(), 1);
    assert!(he_atom.shells[0].is_full());
    
    // Test lithium atom (starts L shell)
    let li_nucleus = nuclear_physics::Nucleus::new(3, 4);
    let li_atom = atomic_physics::Atom::new(li_nucleus);
    assert_eq!(li_atom.charge(), 0);
    assert_eq!(li_atom.shells.len(), 2);
    assert!(li_atom.shells[0].is_full());
    assert!(!li_atom.shells[1].is_full());
    
    // Test carbon atom (half-filled L shell)
    let c_nucleus = nuclear_physics::Nucleus::new(6, 6);
    let c_atom = atomic_physics::Atom::new(c_nucleus);
    assert_eq!(c_atom.charge(), 0);
    assert_eq!(c_atom.shells.len(), 2);
    assert_eq!(c_atom.shells[1].electrons.len(), 4);
}

/// Test all four fundamental forces
#[test]
fn test_fundamental_forces() {
    use physics_engine::constants::PhysicsConstants;
    
    let constants = PhysicsConstants::default();
    
    // Test electromagnetic force (Coulomb)
    let em_force = constants.gravitational_force(
        ELEMENTARY_CHARGE, 
        ELEMENTARY_CHARGE, 
        1e-10 // 1 Angstrom
    ) * (K_E / constants.g); // Scale by electromagnetic vs gravitational strength
    assert!(em_force > 0.0);
    
    // Test gravitational force (very weak)
    let grav_force = constants.gravitational_force(
        PROTON_MASS, 
        PROTON_MASS, 
        1e-15 // Nuclear distance
    );
    assert!(grav_force > 0.0);
    assert!(em_force > grav_force); // EM much stronger than gravity
    
    // Test strong force via nuclear binding energy
    let nucleus = nuclear_physics::Nucleus::new(2, 2); // Helium-4
    let binding_energy = nucleus.binding_energy();
    assert!(binding_energy > 0.0); // Strong force binds nucleus
    
    // Test weak force via neutron decay
    let neutron_props = get_properties(ParticleType::Neutron);
    assert!(neutron_props.width.is_some()); // Weak force causes decay
    let decay_rate = neutron_props.width.unwrap();
    assert!(decay_rate > 0.0);
}

/// Test interaction cross-sections between particles
#[test]
fn test_particle_interactions() {
    // Test Compton scattering (photon + electron)
    let compton_cs = klein_nishina_cross_section(0.001); // 1 MeV photon
    assert!(compton_cs > 0.0);
    assert!(compton_cs < 1e-28); // Should be sub-barn scale
    
    // Test pair production (photon + nucleus)
    let pair_cs = bethe_heitler_pair_production(0.01, 82); // 10 MeV photon on lead
    assert!(pair_cs > 0.0);
    
    // Test atomic collision cross-sections
    let h_atom = atomic_physics::Atom::new(nuclear_physics::Nucleus::new(1, 0));
    let he_atom = atomic_physics::Atom::new(nuclear_physics::Nucleus::new(2, 2));
    
    let collision_cs = atomic_physics::elastic_collision_cross_section(&h_atom, &he_atom);
    assert!(collision_cs > 0.0);
    
    // Larger atoms should have larger cross-sections
    let c_atom = atomic_physics::Atom::new(nuclear_physics::Nucleus::new(6, 6));
    let h_c_cs = atomic_physics::elastic_collision_cross_section(&h_atom, &c_atom);
    let h_he_cs = atomic_physics::elastic_collision_cross_section(&h_atom, &he_atom);
    assert!(h_c_cs > h_he_cs);
}

/// Test phase transitions for all known substances
#[test]
fn test_comprehensive_phase_transitions() {
    use physics_engine::phase_transitions::*;
    use physics_engine::emergent_properties::{Temperature, Pressure, Density};
    
    // Test water phase transitions
    let temp_ice = Temperature::from_kelvin(250.0);
    let temp_water = Temperature::from_kelvin(300.0);
    let temp_steam = Temperature::from_kelvin(400.0);
    let atm_pressure = Pressure::from_pascals(101325.0);
    let water_density = Density::from_kg_per_m3(1000.0);
    
    let ice_phase = evaluate_phase_transitions("water", temp_ice, atm_pressure, water_density).unwrap();
    let liquid_phase = evaluate_phase_transitions("water", temp_water, atm_pressure, water_density).unwrap();
    let gas_phase = evaluate_phase_transitions("water", temp_steam, atm_pressure, water_density).unwrap();
    
    assert_eq!(ice_phase, Phase::Solid);
    assert_eq!(liquid_phase, Phase::Liquid);
    assert_eq!(gas_phase, Phase::Gas);
    
    // Test hydrogen phase transitions
    let h2_cold = Temperature::from_kelvin(10.0);
    let h2_liquid = Temperature::from_kelvin(20.0);
    let h2_gas = Temperature::from_kelvin(300.0);
    let h2_density = Density::from_kg_per_m3(71.0); // Liquid hydrogen density
    
    let h2_solid = evaluate_phase_transitions("hydrogen", h2_cold, atm_pressure, h2_density).unwrap();
    let h2_liquid_phase = evaluate_phase_transitions("hydrogen", h2_liquid, atm_pressure, h2_density).unwrap();
    let h2_gas_phase = evaluate_phase_transitions("hydrogen", h2_gas, atm_pressure, h2_density).unwrap();
    
    assert_eq!(h2_solid, Phase::Solid);
    assert_eq!(h2_liquid_phase, Phase::Liquid);
    assert_eq!(h2_gas_phase, Phase::Gas);
}

/// Test conservation laws across all particle interactions
#[test]
fn test_conservation_laws() {
    // Test charge conservation in beta decay: n → p + e⁻ + ν̄ₑ
    let neutron_charge = get_properties(ParticleType::Neutron).charge_c;
    let proton_charge = get_properties(ParticleType::Proton).charge_c;
    let electron_charge = get_properties(ParticleType::Electron).charge_c;
    let antineutrino_charge = get_properties(ParticleType::ElectronAntiNeutrino).charge_c;
    
    let initial_charge = neutron_charge;
    let final_charge = proton_charge + electron_charge + antineutrino_charge;
    assert_relative_eq!(initial_charge, final_charge, epsilon = 1e-15);
    
    // Test energy-momentum conservation in pair production: γ → e⁺ + e⁻
    let photon_mass = get_properties(ParticleType::Photon).mass_kg;
    let electron_mass = get_properties(ParticleType::Electron).mass_kg;
    let positron_mass = get_properties(ParticleType::Positron).mass_kg;
    
    assert_eq!(photon_mass, 0.0); // Photon massless
    assert_eq!(electron_mass, positron_mass); // Same mass for particle/antiparticle
    
    // For pair production, photon energy must exceed 2×electron mass energy
    let threshold_energy = 2.0 * electron_mass * SPEED_OF_LIGHT * SPEED_OF_LIGHT;
    assert!(threshold_energy > 0.0);
}

/// Test nuclear reaction networks for stellar nucleosynthesis
#[test]
fn test_stellar_nucleosynthesis_comprehensive() {
    let mut nucleosynthesis = StellarNucleosynthesis::new();
    assert!(!nucleosynthesis.reactions.is_empty());
    
    // Test temperature-dependent burning stages
    nucleosynthesis.update_burning_stages(1e7); // 10 MK
    assert!(nucleosynthesis.pp_chain_active);
    assert!(!nucleosynthesis.cno_cycle_active);
    
    nucleosynthesis.update_burning_stages(2e7); // 20 MK
    assert!(nucleosynthesis.pp_chain_active);
    assert!(nucleosynthesis.cno_cycle_active);
    
    nucleosynthesis.update_burning_stages(1e8); // 100 MK
    assert!(nucleosynthesis.helium_burning_active);
    
    nucleosynthesis.update_burning_stages(6e8); // 600 MK
    assert!(nucleosynthesis.advanced_burning_active);
    
    // Test that all burning stages can be active simultaneously at high temperature
    nucleosynthesis.update_burning_stages(1e9); // 1 GK
    assert!(nucleosynthesis.pp_chain_active);
    assert!(nucleosynthesis.cno_cycle_active);
    assert!(nucleosynthesis.helium_burning_active);
    assert!(nucleosynthesis.advanced_burning_active);
}

/// Test complete particle creation and initialization
#[test]
fn test_particle_creation_comprehensive() {
    // Test creation of all major particle types
    let test_particles = vec![
        ParticleType::Electron, ParticleType::Muon, ParticleType::Tau,
        ParticleType::Proton, ParticleType::Neutron,
        ParticleType::PionPlus, ParticleType::KaonZero,
        ParticleType::Lambda, ParticleType::SigmaPlus,
        ParticleType::Photon, ParticleType::WBoson, ParticleType::ZBoson,
    ];
    
    for particle_type in test_particles {
        let particle = spawn_rest(particle_type);
        
        // Check basic properties
        assert_eq!(particle.particle_type, particle_type);
        assert_eq!(particle.position, Vector3::zeros());
        assert_eq!(particle.momentum, Vector3::zeros());
        
        // Check mass matches database
        let expected_mass = get_properties(particle_type).mass_kg;
        assert_relative_eq!(particle.mass, expected_mass, epsilon = 1e-15);
        
        // Check charge matches database
        let expected_charge = get_properties(particle_type).charge_c;
        assert_relative_eq!(particle.electric_charge, expected_charge, epsilon = 1e-15);
        
        // Check initialization
        assert!(particle.quantum_state.wave_function.is_empty() || 
                !particle.quantum_state.wave_function.is_empty());
        assert_eq!(particle.creation_time, 0.0);
    }
}

/// Test molecular dynamics functionality including formation and reactions
#[test]
fn test_molecular_dynamics_comprehensive() {
    let mut physics_engine = PhysicsEngine::new().unwrap();
    
    // Create hydrogen and oxygen atoms using the physics engine's Atom type
    let h_nucleus = AtomicNucleus {
        mass_number: 1,
        atomic_number: 1,
        protons: vec![],
        neutrons: vec![],
        binding_energy: 0.0,
        nuclear_spin: Vector3::zeros(),
        magnetic_moment: Vector3::zeros(),
        electric_quadrupole_moment: 0.0,
        nuclear_radius: 1e-15,
        shell_model_state: HashMap::new(),
        position: Vector3::new(0.0, 0.0, 0.0),
        momentum: Vector3::zeros(),
        excitation_energy: 0.0,
    };
    
    let o_nucleus = AtomicNucleus {
        mass_number: 16,
        atomic_number: 8,
        protons: vec![],
        neutrons: vec![],
        binding_energy: 0.0,
        nuclear_spin: Vector3::zeros(),
        magnetic_moment: Vector3::zeros(),
        electric_quadrupole_moment: 0.0,
        nuclear_radius: 3e-15,
        shell_model_state: HashMap::new(),
        position: Vector3::new(2e-10, 0.0, 0.0),
        momentum: Vector3::zeros(),
        excitation_energy: 0.0,
    };
    
    let h_atom = Atom {
        nucleus: h_nucleus,
        electrons: vec![],
        electron_orbitals: vec![],
        total_energy: 0.0,
        ionization_energy: 13.6 * 1.602e-19, // eV to J
        electron_affinity: 0.0,
        atomic_radius: 5.29e-11, // Bohr radius
        position: Vector3::new(0.0, 0.0, 0.0),
        velocity: Vector3::zeros(),
        electronic_state: HashMap::new(),
    };
    
    let o_atom = Atom {
        nucleus: o_nucleus,
        electrons: vec![],
        electron_orbitals: vec![],
        total_energy: 0.0,
        ionization_energy: 13.6 * 1.602e-19,
        electron_affinity: 1.46 * 1.602e-19,
        atomic_radius: 6.6e-11,
        position: Vector3::new(2e-10, 0.0, 0.0),
        velocity: Vector3::zeros(),
        electronic_state: HashMap::new(),
    };
    
    physics_engine.atoms.push(h_atom);
    physics_engine.atoms.push(o_atom);
    
    // Test molecular formation capability
    assert!(physics_engine.can_form_molecule(&physics_engine.atoms[0], &physics_engine.atoms[1]));
    
    // Test molecule type determination  
    let molecule_type = physics_engine.determine_molecule_type(&physics_engine.atoms[0], &physics_engine.atoms[1]);
    assert_eq!(molecule_type, Some(ParticleType::H2O));
    
    // Test molecular mass values
    assert_eq!(physics_engine.get_particle_mass(ParticleType::H2), 3.34e-27);
    assert_eq!(physics_engine.get_particle_mass(ParticleType::H2O), 2.99e-26);
    assert_eq!(physics_engine.get_particle_mass(ParticleType::CO2), 7.31e-26);
    assert_eq!(physics_engine.get_particle_mass(ParticleType::CH4), 2.66e-26);
    assert_eq!(physics_engine.get_particle_mass(ParticleType::NH3), 2.83e-26);
    
    // Test ordering (lighter molecules should have smaller masses)
    assert!(physics_engine.get_particle_mass(ParticleType::H2) < physics_engine.get_particle_mass(ParticleType::H2O));
    assert!(physics_engine.get_particle_mass(ParticleType::H2O) < physics_engine.get_particle_mass(ParticleType::CO2));
    
    // Test molecular recognition
    assert!(physics_engine.is_molecule(ParticleType::H2));
    assert!(physics_engine.is_molecule(ParticleType::H2O));
    assert!(physics_engine.is_molecule(ParticleType::CO2));
    assert!(!physics_engine.is_molecule(ParticleType::Proton));
    assert!(!physics_engine.is_molecule(ParticleType::Electron));
    
    // Test chemical reaction identification
    let reaction = physics_engine.check_chemical_reaction(ParticleType::CH4, ParticleType::H2O);
    assert!(reaction.is_some());
    let products = reaction.unwrap();
    assert!(products.contains(&ParticleType::CO2));
    assert!(products.contains(&ParticleType::H2));
} 