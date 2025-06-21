//! Demo 02: Weak Interactions Showcase
//! 
//! This demo demonstrates the weak nuclear force in action:
//! - Beta decay of neutrons using proper Fermi's golden rule
//! - Neutrino-electron scattering with V-A current structure
//! - Exponential decay curves and cross-section validation

use physics_engine::{PhysicsEngine, ParticleType, FundamentalParticle, QuantumState};
use physics_engine::interactions;
use nalgebra::Vector3;
use anyhow::Result;
use std::collections::HashMap;
use rand::{thread_rng, Rng};
use nalgebra::Complex;

fn main() -> Result<()> {
    println!("=== Universe Evolution Simulator: Weak Interactions Demo ===");
    println!();
    
    // Initialize physics engine with smaller timestep for weak processes
    let dt = 1e-19; // 0.1 attoseconds
    let mut engine = PhysicsEngine::new()?;
    
    // Set up initial conditions for weak interaction studies
    setup_weak_interaction_experiment(&mut engine)?;
    
    println!("Initial conditions:");
    println!("  Temperature: {:.2e} K", engine.temperature);
    println!("  Total particles: {}", engine.particles.len());
    println!("  Time step: {:.2e} s", dt);
    println!("  Neutron lifetime (calculated): {:.1} s", interactions::neutron_lifetime());
    println!();
    
    // CSV header
    println!("tick,time_s,temperature_K,total_particles,neutrons,protons,electrons,positrons,electron_antinus,compton_events,nu_scatter_events,beta_decays");
    
    // Run simulation for longer time to see beta decays
    let total_steps = 10000;
    for tick in 0..total_steps {
        engine.step(dt)?;
        
        // Output every 1000 steps
        if tick % 1000 == 0 {
            let time = tick as f64 * dt;
            let temp = engine.temperature;
            let total = engine.particles.len();
            
            // Count particles by type
            let mut counts: HashMap<ParticleType, usize> = HashMap::new();
            for particle in &engine.particles {
                *counts.entry(particle.particle_type).or_insert(0) += 1;
            }
            
            let neutrons = counts.get(&ParticleType::Neutron).copied().unwrap_or(0);
            let protons = counts.get(&ParticleType::Proton).copied().unwrap_or(0);
            let electrons = counts.get(&ParticleType::Electron).copied().unwrap_or(0);
            let positrons = counts.get(&ParticleType::Positron).copied().unwrap_or(0);
            let electron_antinus = counts.get(&ParticleType::ElectronAntiNeutrino).copied().unwrap_or(0);
            
            println!("{},{:.2e},{:.2e},{},{},{},{},{},{},{},{},{}",
                tick, time, temp, total, neutrons, protons, electrons, positrons, 
                electron_antinus, engine.compton_count, engine.neutrino_scatter_count, 
                engine.neutron_decay_count);
        }
    }
    
    println!();
    println!("=== Weak Interactions Analysis ===");
    
    // Final particle counts
    let mut final_counts: HashMap<ParticleType, usize> = HashMap::new();
    for particle in &engine.particles {
        *final_counts.entry(particle.particle_type).or_insert(0) += 1;
    }
    
    let final_neutrons = final_counts.get(&ParticleType::Neutron).copied().unwrap_or(0);
    let final_protons = final_counts.get(&ParticleType::Proton).copied().unwrap_or(0);
    let final_electrons = final_counts.get(&ParticleType::Electron).copied().unwrap_or(0);
    let final_antinus = final_counts.get(&ParticleType::ElectronAntiNeutrino).copied().unwrap_or(0);
    
    println!("Final state:");
    println!("  Neutrons: {}", final_neutrons);
    println!("  Protons: {}", final_protons);
    println!("  Electrons: {}", final_electrons);
    println!("  Electron antineutrinos: {}", final_antinus);
    println!();
    
    println!("Weak interaction statistics:");
    println!("  Beta decays: {}", engine.neutron_decay_count);
    println!("  Neutrino scatters: {}", engine.neutrino_scatter_count);
    println!("  Compton scatters: {}", engine.compton_count);
    
    // Calculate theoretical vs observed decay rate
    let simulation_time = total_steps as f64 * dt;
    let initial_neutrons = 1000.0; // We'll set this in setup
    let expected_decays = initial_neutrons * (1.0 - (-simulation_time / interactions::neutron_lifetime()).exp());
    let decay_efficiency = engine.neutron_decay_count as f64 / expected_decays;
    
    println!();
    println!("Decay analysis:");
    println!("  Simulation time: {:.2e} s", simulation_time);
    println!("  Expected decays: {:.1}", expected_decays);
    println!("  Observed decays: {}", engine.neutron_decay_count);
    println!("  Efficiency: {:.1}%", decay_efficiency * 100.0);
    
    // Validate weak interaction physics
    validate_weak_physics(&engine)?;
    
    Ok(())
}

fn setup_weak_interaction_experiment(engine: &mut PhysicsEngine) -> Result<()> {
    let mut rng = thread_rng();
    
    // Clear existing particles
    engine.particles.clear();
    
    // Add 1000 neutrons for beta decay studies
    for _ in 0..1000 {
        let neutron = FundamentalParticle {
            particle_type: ParticleType::Neutron,
            position: Vector3::new(
                rng.gen_range(-1e-12..1e-12),
                rng.gen_range(-1e-12..1e-12),
                rng.gen_range(-1e-12..1e-12),
            ),
            momentum: Vector3::zeros(),
            velocity: Vector3::zeros(),
            spin: Vector3::new(Complex::new(0.5, 0.0), Complex::new(0.0, 0.0), Complex::new(0.0, 0.0)),
            color_charge: None,
            electric_charge: 0.0,
            charge: 0.0,
            mass: 939.565,
            energy: 0.0,
            creation_time: engine.current_time,
            decay_time: Some(engine.current_time + sample_exponential_decay(interactions::neutron_lifetime(), &mut rng)),
            quantum_state: QuantumState::default(),
            interaction_history: Vec::new(),
        };
        engine.particles.push(neutron);
    }
    
    // Add some electrons for neutrino scattering
    for _ in 0..500 {
        let electron = FundamentalParticle {
            particle_type: ParticleType::Electron,
            position: Vector3::new(
                rng.gen_range(-1e-12..1e-12),
                rng.gen_range(-1e-12..1e-12),
                rng.gen_range(-1e-12..1e-12),
            ),
            momentum: Vector3::new(
                rng.gen_range(-1e-21..1e-21),
                rng.gen_range(-1e-21..1e-21),
                rng.gen_range(-1e-21..1e-21),
            ),
            velocity: Vector3::zeros(),
            spin: Vector3::zeros(),
            color_charge: None,
            electric_charge: -1.602176634e-19,
            charge: -1.602176634e-19,
            mass: 5.48579909e-4,
            energy: 0.0,
            creation_time: engine.current_time,
            decay_time: None,
            quantum_state: QuantumState::default(),
            interaction_history: Vec::new(),
        };
        engine.particles.push(electron);
    }
    
    // Add some high-energy neutrinos for scattering studies
    for _ in 0..200 {
        let neutrino = FundamentalParticle {
            particle_type: ParticleType::ElectronNeutrino,
            position: Vector3::new(
                rng.gen_range(-1e-12..1e-12),
                rng.gen_range(-1e-12..1e-12),
                rng.gen_range(-1e-12..1e-12),
            ),
            momentum: Vector3::new(
                rng.gen_range(-1e-18..1e-18), // High energy neutrinos
                rng.gen_range(-1e-18..1e-18),
                rng.gen_range(-1e-18..1e-18),
            ),
            velocity: Vector3::zeros(),
            spin: Vector3::zeros(),
            color_charge: None,
            electric_charge: 0.0,
            charge: 0.0,
            mass: 1e-36, // Tiny neutrino mass
            energy: 0.0,
            creation_time: engine.current_time,
            decay_time: None,
            quantum_state: QuantumState::default(),
            interaction_history: Vec::new(),
        };
        engine.particles.push(neutrino);
    }
    
    // Update all particle energies
    engine.update_particle_energies()?;
    
    Ok(())
}

fn sample_exponential_decay(lifetime: f64, rng: &mut impl rand::Rng) -> f64 {
    -lifetime * rng.gen::<f64>().ln()
}

fn validate_weak_physics(engine: &PhysicsEngine) -> Result<()> {
    println!();
    println!("=== Physics Validation ===");
    
    // Check charge conservation
    let mut total_charge = 0.0;
    for particle in &engine.particles {
        total_charge += particle.electric_charge;
    }
    println!("Total charge: {:.2e} C (should be ~0)", total_charge);
    
    // Check lepton number conservation
    let mut electron_lepton_number = 0;
    for particle in &engine.particles {
        match particle.particle_type {
            ParticleType::Electron | ParticleType::ElectronNeutrino => electron_lepton_number += 1,
            ParticleType::Positron | ParticleType::ElectronAntiNeutrino => electron_lepton_number -= 1,
            _ => {}
        }
    }
    println!("Electron lepton number: {} (should be conserved)", electron_lepton_number);
    
    // Validate cross-sections
    let test_energy = 0.01; // 10 MeV neutrino
    let sigma_nu_e = interactions::neutrino_e_scattering_complete(0, test_energy, false);
    println!("ν_e + e⁻ cross-section at 10 MeV: {:.2e} m²", sigma_nu_e);
    
    // Expected: σ ≈ G_F² m_e E_ν / π ≈ 1.7e-44 m² for 10 MeV
    let expected_sigma = 1.7e-44;
    let sigma_ratio = sigma_nu_e / expected_sigma;
    println!("Cross-section ratio (obs/exp): {:.2}", sigma_ratio);
    
    if sigma_ratio > 0.1 && sigma_ratio < 10.0 {
        println!("✓ Cross-section within reasonable range");
    } else {
        println!("⚠ Cross-section may need adjustment");
    }
    
    Ok(())
}