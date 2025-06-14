//! Demo 01: Big Bang Simulation
//! 
//! This demo initializes the physics engine with conditions approximating
//! the early universe (T ~ 10^12 K) and evolves the particle soup through
//! the first microsecond, tracking particle populations and temperature.

use physics_engine::{PhysicsEngine, ParticleType};
use anyhow::Result;
use std::collections::HashMap;

fn main() -> Result<()> {
    println!("=== Universe Evolution Simulator: Big Bang Demo ===");
    println!();
    
    // Initialize physics engine with 10^-23 second timestep
    // This is roughly the QCD scale where quarks begin to confine
    let dt = 1e-23;
    let mut engine = PhysicsEngine::new(dt)?;
    
    // Initialize with Big Bang conditions
    println!("Initializing Big Bang conditions...");
    engine.initialize_big_bang()?;
    
    println!("Initial conditions:");
    println!("  Temperature: {:.2e} K", engine.temperature);
    println!("  Total particles: {}", engine.particles.len());
    println!("  Time step: {:.2e} s", dt);
    println!();
    
    // CSV header for data output
    println!("tick,time_s,temperature_K,total_particles,photons,quarks,leptons,gluons,w_bosons,z_bosons");
    
    // Run simulation for 1000 steps (1e-20 seconds total)
    for tick in 0..1000 {
        // Step the simulation
        engine.step(&mut [])?;
        
        // Every 100 steps, output statistics
        if tick % 100 == 0 {
            let time = tick as f64 * dt;
            let temp = engine.temperature;
            let total = engine.particles.len();
            
            // Count particles by type
            let mut counts: HashMap<ParticleType, usize> = HashMap::new();
            for particle in &engine.particles {
                *counts.entry(particle.particle_type).or_insert(0) += 1;
            }
            
            // Extract specific counts
            let photons = counts.get(&ParticleType::Photon).copied().unwrap_or(0);
            let quarks = counts.get(&ParticleType::Up).copied().unwrap_or(0)
                + counts.get(&ParticleType::Down).copied().unwrap_or(0)
                + counts.get(&ParticleType::Charm).copied().unwrap_or(0)
                + counts.get(&ParticleType::Strange).copied().unwrap_or(0)
                + counts.get(&ParticleType::Top).copied().unwrap_or(0)
                + counts.get(&ParticleType::Bottom).copied().unwrap_or(0);
            let leptons = counts.get(&ParticleType::Electron).copied().unwrap_or(0)
                + counts.get(&ParticleType::Muon).copied().unwrap_or(0)
                + counts.get(&ParticleType::Tau).copied().unwrap_or(0)
                + counts.get(&ParticleType::ElectronNeutrino).copied().unwrap_or(0)
                + counts.get(&ParticleType::MuonNeutrino).copied().unwrap_or(0)
                + counts.get(&ParticleType::TauNeutrino).copied().unwrap_or(0);
            let gluons = counts.get(&ParticleType::Gluon).copied().unwrap_or(0);
            let w_bosons = counts.get(&ParticleType::WBoson).copied().unwrap_or(0);
            let z_bosons = counts.get(&ParticleType::ZBoson).copied().unwrap_or(0);
            
            // Output CSV row
            println!("{},{:.2e},{:.2e},{},{},{},{},{},{},{}",
                tick, time, temp, total, photons, quarks, leptons, gluons, w_bosons, z_bosons);
        }
    }
    
    println!();
    println!("Simulation complete!");
    println!("Final state:");
    println!("  Temperature: {:.2e} K", engine.temperature);
    println!("  Total particles: {}", engine.particles.len());
    
    // Show final particle composition
    let mut final_counts: HashMap<ParticleType, usize> = HashMap::new();
    for particle in &engine.particles {
        *final_counts.entry(particle.particle_type).or_insert(0) += 1;
    }
    
    println!();
    println!("Final particle composition:");
    let mut sorted_counts: Vec<_> = final_counts.into_iter().collect();
    sorted_counts.sort_by_key(|(_, count)| std::cmp::Reverse(*count));
    
    for (particle_type, count) in sorted_counts.iter().take(10) {
        let percentage = (*count as f64 / engine.particles.len() as f64) * 100.0;
        println!("  {:?}: {} ({:.1}%)", particle_type, count, percentage);
    }
    
    Ok(())
}