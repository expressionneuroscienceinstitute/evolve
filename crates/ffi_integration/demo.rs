//! Demonstration of Geant4 FFI integration
//! This shows how to use both stub and real Geant4 implementations

use std::env;
use anyhow::Result;

// Import the local geant4 module
use crate::geant4::{Geant4Engine, is_available, initialize, cleanup};
use physics_types::{FundamentalParticle, ParticleType, QuantumState};
use nalgebra::Vector3;

/// Demonstration of basic Geant4 functionality
pub fn run_geant4_demo() -> Result<()> {
    println!("=== Geant4 FFI Integration Demo ===");
    println!();
    
    // Check availability
    if is_available() {
        println!("✓ Real Geant4 is available!");
        println!("  This will use actual Monte Carlo physics simulation");
    } else {
        println!("⚠ Using Geant4 stubs");
        println!("  This provides API compatibility but no real physics");
        println!("  To use real Geant4, build the wrapper library:");
        println!("  cd crates/ffi_integration && ./build_wrapper.sh");
    }
    println!();
    
    // Initialize Geant4
    println!("Initializing Geant4...");
    initialize()?;
    
    // Create a Geant4 engine
    println!("Creating Geant4 engine with FTFP_BERT physics list...");
    let mut engine = Geant4Engine::new("FTFP_BERT")?;
    
    // Demo 1: Particle transport simulation
    println!("\n--- Demo 1: Particle Transport ---");
    demo_particle_transport(&mut engine)?;
    
    // Demo 2: Cross-section calculations
    println!("\n--- Demo 2: Cross-Section Calculations ---");
    demo_cross_sections(&engine)?;
    
    // Demo 3: Stopping power calculations
    println!("\n--- Demo 3: Stopping Power ---");
    demo_stopping_power(&engine)?;
    
    // Demo 4: Particle decay simulation
    println!("\n--- Demo 4: Particle Decay ---");
    demo_particle_decay(&mut engine)?;
    
    // Cleanup
    println!("\nCleaning up...");
    cleanup()?;
    
    println!("\n=== Demo Complete ===");
    println!("The demo showed how the FFI integration works with both");
    println!("stub and real Geant4 implementations. For scientific");
    println!("applications, use the real Geant4 library for accuracy.");
    
    Ok(())
}

fn demo_particle_transport(engine: &mut Geant4Engine) -> Result<()> {
    println!("Simulating electron transport through water...");
    
    // Create an electron
    let electron = FundamentalParticle {
        particle_type: ParticleType::Electron,
        quantum_state: QuantumState {
            energy: 1.0,  // 1 MeV
            momentum: Vector3::new(1.0, 0.0, 0.0),
            position: Vector3::new(0.0, 0.0, 0.0),
            ..Default::default()
        },
        charge: -1.0,
        mass: 0.511,  // MeV/c^2
        spin: 0.5,
    };
    
    // Simulate transport through 1 cm of water
    let material = "G4_WATER";
    let step_length = 1.0; // cm
    
    println!("  Particle: {} MeV electron", electron.quantum_state.energy);
    println!("  Material: {}", material);
    println!("  Step length: {} cm", step_length);
    
    match engine.transport_particle(&electron, material, step_length) {
        Ok(interactions) => {
            println!("  Result: {} interactions detected", interactions.len());
            
            if is_available() {
                for (i, interaction) in interactions.iter().take(3).enumerate() {
                    println!("    Interaction {}: {} (energy: {:.3} MeV)", 
                            i + 1, 
                            format!("{:?}", interaction.interaction_type),
                            interaction.energy_deposited);
                }
            } else {
                println!("    (Stub mode: no real interactions calculated)");
            }
        }
        Err(e) => println!("  Error: {}", e),
    }
    
    Ok(())
}

fn demo_cross_sections(engine: &Geant4Engine) -> Result<()> {
    println!("Calculating photon interaction cross-sections in lead...");
    
    let particle = ParticleType::Photon;
    let material = "G4_Pb";
    let energies = vec![0.1, 0.5, 1.0, 5.0, 10.0]; // MeV
    
    println!("  Particle: Photon");
    println!("  Material: {} (Lead)", material);
    println!();
    
    for energy in energies {
        println!("  Energy: {} MeV", energy);
        
        // Calculate different process cross-sections
        let processes = vec![
            ("photoelectric", "Photoelectric effect"),
            ("compton", "Compton scattering"),
            ("pairproduction", "Pair production"),
        ];
        
        for (process, description) in processes {
            match engine.calculate_cross_section(&particle, material, process, energy) {
                Ok(xsec) => {
                    if is_available() {
                        println!("    {}: {:.2e} cm²", description, xsec);
                    } else {
                        println!("    {}: {:.2e} cm² (stub value)", description, xsec);
                    }
                }
                Err(e) => println!("    {}: Error - {}", description, e),
            }
        }
        println!();
    }
    
    Ok(())
}

fn demo_stopping_power(engine: &Geant4Engine) -> Result<()> {
    println!("Calculating stopping power for charged particles...");
    
    let particles = vec![
        (ParticleType::Electron, "Electron"),
        (ParticleType::Proton, "Proton"),
        (ParticleType::Alpha, "Alpha particle"),
    ];
    
    let material = "G4_WATER";
    let energy = 1.0; // MeV
    
    println!("  Material: {} (Water)", material);
    println!("  Energy: {} MeV", energy);
    println!();
    
    for (particle_type, name) in particles {
        match engine.get_stopping_power(&particle_type, material, energy) {
            Ok(stopping_power) => {
                if is_available() {
                    println!("  {}: {:.2e} MeV/cm", name, stopping_power);
                } else {
                    println!("  {}: {:.2e} MeV/cm (stub value)", name, stopping_power);
                }
            }
            Err(e) => println!("  {}: Error - {}", name, e),
        }
    }
    
    Ok(())
}

fn demo_particle_decay(engine: &mut Geant4Engine) -> Result<()> {
    println!("Simulating particle decay processes...");
    
    // Create a neutral pion (π⁰)
    let pion = FundamentalParticle {
        particle_type: ParticleType::Pion0,
        quantum_state: QuantumState {
            energy: 139.6,  // MeV (rest mass + kinetic energy)
            momentum: Vector3::new(0.0, 0.0, 0.0),
            position: Vector3::new(0.0, 0.0, 0.0),
            ..Default::default()
        },
        charge: 0.0,
        mass: 134.98,  // MeV/c^2
        spin: 0.0,
    };
    
    println!("  Particle: π⁰ (neutral pion)");
    println!("  Energy: {:.1} MeV", pion.quantum_state.energy);
    println!("  Expected decay: π⁰ → γ + γ");
    println!();
    
    match engine.simulate_decay(&pion) {
        Ok(products) => {
            println!("  Decay products: {}", products.len());
            
            for (i, product) in products.iter().enumerate() {
                println!("    Product {}: {:?} (E = {:.1} MeV)", 
                        i + 1, 
                        product.particle_type,
                        product.quantum_state.energy);
            }
            
            if !is_available() {
                println!("    (Stub mode: simplified decay kinematics)");
            }
        }
        Err(e) => println!("  Error: {}", e),
    }
    
    Ok(())
}

/// Main function for running the demo
fn main() -> Result<()> {
    // Check for help argument
    let args: Vec<String> = env::args().collect();
    if args.len() > 1 && (args[1] == "--help" || args[1] == "-h") {
        println!("Geant4 FFI Integration Demo");
        println!();
        println!("This demo shows how to use the Geant4 FFI integration");
        println!("with both stub and real implementations.");
        println!();
        println!("Usage:");
        println!("  cargo run --bin geant4_demo");
        println!();
        println!("To use real Geant4 instead of stubs:");
        println!("  1. Install Geant4 (see README.md)");
        println!("  2. cd crates/ffi_integration && ./build_wrapper.sh");
        println!("  3. cargo run --bin geant4_demo --features geant4");
        println!();
        return Ok(());
    }
    
    // Run the demo
    match run_geant4_demo() {
        Ok(()) => {
            println!("Demo completed successfully!");
            Ok(())
        }
        Err(e) => {
            eprintln!("Demo failed: {}", e);
            std::process::exit(1);
        }
    }
} 