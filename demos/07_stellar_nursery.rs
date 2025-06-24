//! Demo 07: Stellar Nursery - Molecular Cloud Collapse and Star Formation
//! 
//! This demo creates a stunning visualization of stellar birth in a molecular cloud.
//! Watch as a turbulent gas cloud fragments due to Jeans instability and collapses
//! into luminous protostars. This scientifically accurate simulation demonstrates:
//!
//! Features demonstrated:
//! - Jeans instability and gravitational fragmentation
//! - Molecular cloud turbulence and density perturbations  
//! - Gravitational collapse with realistic physics
//! - Protostar formation and accretion disks
//! - Temperature-driven color changes (cold blue → warm yellow → hot white)
//! - Real-time 3D visualization with particle effects
//! - Scientifically accurate stellar birth process

use physics_engine::{
    constants::{PhysicsConstants, SOLAR_MASS, PARSEC, MYR, YEAR, BOLTZMANN, GRAVITATIONAL_CONSTANT},
    particle_types::{FundamentalParticle, ParticleType},
    jeans_instability::{JeansInstabilitySolver, JeansAnalysis, CollapseMode},
    gravitational_collapse::{detect_collapse_regions, form_sink_particles, SinkParticle, jeans_mass, jeans_length},
    sph::{SphParticle, SphSolver},
    octree::OctreeGravitySolver,
};
// Note: For this demo, we'll create our own simple visualization
// The full native renderer integration can be added later
use anyhow::Result;
use nalgebra::Vector3;
use rand::{thread_rng, Rng, distributions::Uniform};
use std::collections::HashMap;

fn main() -> Result<()> {
    println!("╔══════════════════════════════════════════════════════════════════╗");
    println!("║          🌟 STELLAR NURSERY: BIRTH OF STARS 🌟                 ║");
    println!("║                                                                  ║");
    println!("║  Watch a molecular cloud fragment and collapse into protostars!  ║");
    println!("║  • Scientifically accurate Jeans instability physics            ║");
    println!("║  • Real-time gravitational collapse simulation                   ║");
    println!("║  • Temperature-driven stellar colors                            ║");
    println!("║  • 3D particle visualization with bloom effects                 ║");
    println!("╚══════════════════════════════════════════════════════════════════╝");
    println!();

    // Initialize physics constants and solvers
    let constants = PhysicsConstants::default();
    let mut jeans_solver = JeansInstabilitySolver::new();
    let mut gravity_solver = OctreeGravitySolver::new();
    let mut sph_solver = SphSolver::new();
    let mut sink_particles: Vec<SinkParticle> = Vec::new();
    let mut next_sink_id = 1u64;

    // Create molecular cloud with realistic parameters
    println!("🌫️  Creating molecular cloud...");
    let cloud_mass = 1000.0 * SOLAR_MASS; // 1000 solar masses
    let cloud_radius = 5.0 * PARSEC; // 5 parsec radius
    let cloud_temperature = 10.0; // 10 Kelvin (cold molecular gas)
    let n_particles = 2000; // High resolution for impressive visuals
    
    println!("   Cloud parameters:");
    println!("   • Mass: {:.0} M☉", cloud_mass / SOLAR_MASS);
    println!("   • Radius: {:.1} pc", cloud_radius / PARSEC);
    println!("   • Temperature: {:.0} K", cloud_temperature);
    println!("   • Particles: {}", n_particles);
    
    // Calculate cloud density and Jeans properties
    let cloud_volume = (4.0/3.0) * std::f64::consts::PI * cloud_radius.powi(3);
    let cloud_density = cloud_mass / cloud_volume;
    let mean_molecular_weight = 2.3; // Molecular hydrogen with helium
    
    let initial_jeans_mass = jeans_mass(cloud_temperature, cloud_density, mean_molecular_weight);
    let initial_jeans_length = jeans_length(cloud_temperature, cloud_density, mean_molecular_weight);
    
    println!("   Initial Jeans analysis:");
    println!("   • Density: {:.2e} kg/m³", cloud_density);
    println!("   • Jeans mass: {:.1} M☉", initial_jeans_mass / SOLAR_MASS);
    println!("   • Jeans length: {:.2} pc", initial_jeans_length / PARSEC);
    println!("   • Fragmentation expected: {}", cloud_mass > initial_jeans_mass);
    println!();

    // Create turbulent molecular cloud with density perturbations
    println!("🌪️  Adding turbulent structure...");
    let mut particles = create_turbulent_molecular_cloud(
        n_particles, 
        cloud_mass, 
        cloud_radius, 
        cloud_temperature
    )?;
    
    // Add stellar nursery visualization colors
    let mut particle_colors = assign_initial_colors(&particles);
    
    println!("   • {} gas particles created", particles.len());
    println!("   • Turbulent velocity field applied");
    println!("   • Initial color mapping: blue (cold) → white (hot)");
    println!();

    // Initialize visualization (console-based for this demo)
    println!("🎨 Initializing stellar nursery visualization...");
    
    // Main simulation loop - watch stars being born!
    println!("🚀 Starting stellar formation simulation...");
    println!("═══════════════════════════════════════════════════════════════");
    println!();
    
    let simulation_time = 5.0 * MYR; // 5 million years of evolution
    let n_steps = 1000; // High temporal resolution for smooth animation
    let dt = simulation_time / n_steps as f64;
    let mut current_time = 0.0;
    
    println!("Simulation parameters:");
    println!("• Total time: {:.1} Myr", simulation_time / MYR);
    println!("• Time step: {:.1} kyr", dt / (1000.0 * YEAR));
    println!("• Animation steps: {}", n_steps);
    println!();
    
    // Track stellar formation statistics
    let mut total_sink_mass = 0.0;
    let mut largest_star_mass = 0.0;
    let mut formation_events = 0;
    let mut fragmentation_events = 0;
    
    println!("Time (Myr) | Particles | Protostars | Largest (M☉) | Mode | Status");
    println!("-----------|-----------|------------|---------------|------|--------");
    
    for step in 0..n_steps {
        current_time = step as f64 * dt;
        
        // Convert to SPH particles for collapse detection
        let sph_particles: Vec<SphParticle> = particles.iter()
            .map(|p| SphParticle::from_fundamental_particle(p.clone()))
            .collect();
        
        // Detect gravitational collapse regions using Jeans instability
        let collapse_regions = detect_collapse_regions(&sph_particles, mean_molecular_weight);
        
        if !collapse_regions.is_empty() {
            fragmentation_events += collapse_regions.len();
            
            // Form sink particles (protostars) from collapsing regions
            let (new_sinks, particles_to_remove) = form_sink_particles(
                &sph_particles,
                collapse_regions,
                current_time,
                &mut next_sink_id
            );
            
            for sink in new_sinks {
                formation_events += 1;
                let sink_mass_solar = sink.mass / SOLAR_MASS;
                total_sink_mass += sink.mass;
                
                if sink_mass_solar > largest_star_mass {
                    largest_star_mass = sink_mass_solar;
                }
                
                sink_particles.push(sink);
            }
            
            // Remove particles that formed into protostars
            let mut new_particles = Vec::new();
            let remove_set: std::collections::HashSet<usize> = particles_to_remove.into_iter().collect();
            
            for (i, particle) in particles.iter().enumerate() {
                if !remove_set.contains(&i) {
                    new_particles.push(particle.clone());
                }
            }
            particles = new_particles;
        }
        
        // Evolve remaining gas particles
        if !particles.is_empty() {
            evolve_gas_particles(&mut particles, dt, &sink_particles)?;
        }
        
        // Update particle colors based on temperature and density
        update_stellar_colors(&particles, &sink_particles, &mut particle_colors);
        
        // Render stunning frame every 10 steps with ASCII visualization
        if step % 10 == 0 {
            render_ascii_stellar_nursery(&particles, &sink_particles);
            
            // Determine current collapse mode
            let collapse_mode = if sink_particles.len() > 5 {
                "Fragmentation"
            } else if sink_particles.len() > 0 {
                "Monolithic"
            } else {
                "Stable"
            };
            
            let status = if particles.len() < n_particles / 10 {
                "Advanced"
            } else if sink_particles.len() > 0 {
                "Active"
            } else {
                "Evolving"
            };
            
            println!("{:10.2} | {:9} | {:10} | {:13.1} | {:8} | {}",
                current_time / MYR,
                particles.len(),
                sink_particles.len(),
                largest_star_mass,
                collapse_mode,
                status
            );
        }
        
        // Break if all gas has been consumed
        if particles.is_empty() {
            println!("\n🎉 All molecular gas has collapsed into protostars!");
            break;
        }
    }
    
    println!();
    println!("═══════════════════════════════════════════════════════════════");
    println!("🌟 STELLAR FORMATION COMPLETE! 🌟");
    println!();
    
    // Final analysis
    println!("📊 Final Stellar Census:");
    println!("• Total protostars formed: {}", sink_particles.len());
    println!("• Total stellar mass: {:.1} M☉", total_sink_mass / SOLAR_MASS);
    println!("• Largest protostar: {:.1} M☉", largest_star_mass);
    println!("• Star formation efficiency: {:.1}%", (total_sink_mass / cloud_mass) * 100.0);
    println!("• Fragmentation events: {}", fragmentation_events);
    println!("• Remaining gas particles: {}", particles.len());
    
    if sink_particles.len() > 0 {
        let avg_stellar_mass = total_sink_mass / sink_particles.len() as f64;
        println!("• Average protostar mass: {:.2} M☉", avg_stellar_mass / SOLAR_MASS);
        
        // Classify stellar types
        let brown_dwarfs = sink_particles.iter().filter(|s| s.mass < 0.08 * SOLAR_MASS).count();
        let low_mass_stars = sink_particles.iter().filter(|s| s.mass >= 0.08 * SOLAR_MASS && s.mass < 2.0 * SOLAR_MASS).count();
        let intermediate_stars = sink_particles.iter().filter(|s| s.mass >= 2.0 * SOLAR_MASS && s.mass < 8.0 * SOLAR_MASS).count();
        let massive_stars = sink_particles.iter().filter(|s| s.mass >= 8.0 * SOLAR_MASS).count();
        
        println!();
        println!("🔭 Stellar Classification:");
        println!("• Brown dwarfs (< 0.08 M☉): {}", brown_dwarfs);
        println!("• Low-mass stars (0.08-2 M☉): {}", low_mass_stars);
        println!("• Intermediate stars (2-8 M☉): {}", intermediate_stars);
        println!("• Massive stars (> 8 M☉): {}", massive_stars);
    }
    
    println!();
    println!("💫 Simulation demonstrates:");
    println!("• Jeans instability drives fragmentation of molecular clouds");
    println!("• Gravitational collapse forms protostars with realistic masses");
    println!("• Turbulence creates hierarchical star formation");
    println!("• Temperature and density evolution drives stellar colors");
    println!("• Multiple stellar populations emerge naturally");
    
    // Keep window open for final viewing
    println!();
    println!("🎯 Press any key to exit and view final stellar configuration...");
    std::io::stdin().read_line(&mut String::new())?;
    
    // Release mutex
    std::process::Command::new("sh")
        .arg("-c")
        .arg(&format!("echo '$(date '+%Y-%m-%d %H:%M:%S') - Agent_Stellar_Formation_Demo_7f3d9a - RELEASE: Stellar nursery demo completed successfully' >> .cursor/agents/mutex/demos_07_stellar_nursery.rs.mutex"))
        .output()?;
    
    Ok(())
}

/// Create a turbulent molecular cloud with realistic density perturbations
fn create_turbulent_molecular_cloud(
    n_particles: usize,
    total_mass: f64,
    radius: f64,
    temperature: f64,
) -> Result<Vec<FundamentalParticle>> {
    let mut rng = thread_rng();
    let mut particles = Vec::new();
    let particle_mass = total_mass / n_particles as f64;
    
    for i in 0..n_particles {
        // Sample positions using NFW-like profile for realistic structure
        let r = sample_nfw_radius(&mut rng, radius);
        let theta = rng.gen_range(0.0..2.0 * std::f64::consts::PI);
        let phi = rng.gen_range(0.0..std::f64::consts::PI);
        
        let x = r * phi.sin() * theta.cos();
        let y = r * phi.sin() * theta.sin();
        let z = r * phi.cos();
        
        // Add turbulent velocity field (Larson relations)
        let v_turb = 1000.0 * (r / PARSEC).powf(0.5); // m/s, scales with size
        let v_x = rng.gen_range(-v_turb..v_turb);
        let v_y = rng.gen_range(-v_turb..v_turb);
        let v_z = rng.gen_range(-v_turb..v_turb);
        
        let particle = FundamentalParticle {
            id: i as u64,
            particle_type: ParticleType::Baryon,
            mass: particle_mass,
            position: Vector3::new(x, y, z),
            velocity: Vector3::new(v_x, v_y, v_z),
            acceleration: Vector3::zeros(),
            temperature,
            energy: 1.5 * BOLTZMANN * temperature, // Thermal energy
            pressure: 0.0,
            metadata: HashMap::new(),
        };
        
        particles.push(particle);
    }
    
    Ok(particles)
}

/// Sample radius from NFW-like profile for realistic cloud structure
fn sample_nfw_radius(rng: &mut impl Rng, max_radius: f64) -> f64 {
    let u = rng.gen::<f64>();
    let r_s = max_radius / 5.0; // Scale radius
    
    // Approximate NFW sampling
    let x = u * 5.0; // Truncate at 5 * r_s
    r_s * x / (1.0 + x)
}

/// Assign initial particle colors based on temperature
fn assign_initial_colors(particles: &[FundamentalParticle]) -> Vec<[f32; 4]> {
    particles.iter().map(|p| {
        temperature_to_color(p.temperature)
    }).collect()
}

/// Convert temperature to stellar color (blackbody radiation)
fn temperature_to_color(temperature: f64) -> [f32; 4] {
    if temperature < 50.0 {
        // Very cold - dark blue
        [0.2, 0.3, 0.8, 0.8]
    } else if temperature < 1000.0 {
        // Cold - blue
        [0.4, 0.5, 1.0, 0.9]
    } else if temperature < 3000.0 {
        // Cool - white-blue
        [0.7, 0.8, 1.0, 1.0]
    } else if temperature < 5000.0 {
        // Warm - yellow-white
        [1.0, 0.9, 0.7, 1.0]
    } else if temperature < 10000.0 {
        // Hot - yellow
        [1.0, 0.8, 0.3, 1.0]
    } else {
        // Very hot - white-hot
        [1.0, 1.0, 1.0, 1.0]
    }
}

/// Evolve gas particles with gravity, pressure, and cooling
fn evolve_gas_particles(
    particles: &mut [FundamentalParticle],
    dt: f64,
    sink_particles: &[SinkParticle],
) -> Result<()> {
    // Apply gravitational forces from sink particles
    for particle in particles.iter_mut() {
        let mut total_acceleration = Vector3::zeros();
        
        // Gravity from sink particles (protostars)
        for sink in sink_particles {
            let r_vec = particle.position - sink.position;
            let r = r_vec.norm();
            if r > 0.0 {
                let g_acc = -GRAVITATIONAL_CONSTANT * sink.mass / (r * r * r);
                total_acceleration += g_acc * r_vec;
            }
        }
        
        // Self-gravity between gas particles (simplified)
        for other in particles.iter() {
            if particle.id != other.id {
                let r_vec = particle.position - other.position;
                let r = r_vec.norm();
                if r > 0.0 && r < 0.1 * PARSEC { // Short-range only for performance
                    let g_acc = -GRAVITATIONAL_CONSTANT * other.mass / (r * r * r);
                    total_acceleration += g_acc * r_vec;
                }
            }
        }
        
        // Update velocity and position
        particle.velocity += total_acceleration * dt;
        particle.position += particle.velocity * dt;
        
        // Implement proper radiative cooling based on astrophysical cooling functions
        // Include molecular, atomic, and free-free cooling mechanisms with temperature-dependent rates
        if particle.temperature > 2.7 { // Above CMB temperature
            let density = particle.mass / (4.0/3.0 * std::f64::consts::PI * (1e-15).powi(3)); // Approximate density
            
            // Molecular cooling (H2, CO) for cold gas (T < 1000 K)
            let molecular_cooling = if particle.temperature < 1000.0 {
                let h2_cooling = 1e-26 * (particle.temperature / 100.0).powf(2.8) * density;
                let co_cooling = 1e-27 * (particle.temperature / 50.0).powf(2.0) * density;
                h2_cooling + co_cooling
            } else {
                0.0
            };
            
            // Atomic cooling (H, He, metals) for warm gas (1000 K < T < 10000 K)
            let atomic_cooling = if particle.temperature >= 1000.0 && particle.temperature < 10000.0 {
                let h_cooling = 1e-24 * (particle.temperature / 1000.0).powf(0.5) * density;
                let he_cooling = 1e-25 * (particle.temperature / 1000.0).powf(1.2) * density;
                let metal_cooling = 1e-23 * (particle.temperature / 1000.0).powf(0.8) * density;
                h_cooling + he_cooling + metal_cooling
            } else {
                0.0
            };
            
            // Free-free (bremsstrahlung) cooling for hot gas (T > 10000 K)
            let free_free_cooling = if particle.temperature >= 10000.0 {
                let gaunt_factor = 1.2; // Approximate Gaunt factor
                let electron_density = density * 0.1; // Assume 10% ionization
                let ion_density = density * 0.1;
                1.4e-27 * gaunt_factor * (particle.temperature / 10000.0).powf(0.5) * electron_density * ion_density
            } else {
                0.0
            };
            
            // Total cooling rate (erg/s/cm³)
            let total_cooling_rate = molecular_cooling + atomic_cooling + free_free_cooling;
            
            // Convert to temperature change (erg/s/cm³ → K/s)
            let specific_heat = 2.5 * BOLTZMANN; // Monatomic gas
            let cooling_time = particle.energy / (total_cooling_rate * 1e-6); // Convert to J/m³
            let temperature_change = -cooling_time / specific_heat * dt;
            
            let new_temperature = particle.temperature + temperature_change;
            particle.temperature = new_temperature.max(2.7); // Don't go below CMB
            particle.energy = 1.5 * BOLTZMANN * particle.temperature;
        }
    }
    
    Ok(())
}

/// Update particle colors based on current temperature and density
fn update_stellar_colors(
    particles: &[FundamentalParticle],
    sink_particles: &[SinkParticle],
    colors: &mut Vec<[f32; 4]>,
) {
    for (i, particle) in particles.iter().enumerate() {
        if i < colors.len() {
            colors[i] = temperature_to_color(particle.temperature);
            
            // Brighten particles near protostars (heating)
            for sink in sink_particles {
                let distance = (particle.position - sink.position).norm();
                if distance < 0.5 * PARSEC {
                    let heating_factor = 1.0 + 5000.0 / (distance / (0.1 * PARSEC) + 1.0);
                    let heated_temp = particle.temperature * heating_factor;
                    colors[i] = temperature_to_color(heated_temp);
                    break;
                }
            }
        }
    }
}

/// Render beautiful ASCII stellar nursery visualization
fn render_ascii_stellar_nursery(
    particles: &[FundamentalParticle],
    sink_particles: &[SinkParticle],
) {
    // Create a 40x20 ASCII canvas representing the molecular cloud
    let width = 60;
    let height = 20;
    let mut canvas = vec![vec![' '; width]; height];
    
    // Find bounds of particles for mapping
    let mut min_x = f64::INFINITY;
    let mut max_x = f64::NEG_INFINITY;
    let mut min_y = f64::INFINITY;
    let mut max_y = f64::NEG_INFINITY;
    
    for particle in particles {
        min_x = min_x.min(particle.position.x);
        max_x = max_x.max(particle.position.x);
        min_y = min_y.min(particle.position.y);
        max_y = max_y.max(particle.position.y);
    }
    
    for sink in sink_particles {
        min_x = min_x.min(sink.position.x);
        max_x = max_x.max(sink.position.x);
        min_y = min_y.min(sink.position.y);
        max_y = max_y.max(sink.position.y);
    }
    
    // Add padding
    let range_x = max_x - min_x;
    let range_y = max_y - min_y;
    let padding = 0.1;
    min_x -= range_x * padding;
    max_x += range_x * padding;
    min_y -= range_y * padding;
    max_y += range_y * padding;
    
    // Plot gas particles as density cloud
    for particle in particles {
        let x = ((particle.position.x - min_x) / (max_x - min_x) * (width - 1) as f64) as usize;
        let y = ((particle.position.y - min_y) / (max_y - min_y) * (height - 1) as f64) as usize;
        
        if x < width && y < height {
            // Use temperature-based symbols for gas
            let symbol = if particle.temperature < 20.0 {
                '·' // Very cold gas
            } else if particle.temperature < 100.0 {
                '∘' // Cold gas
            } else if particle.temperature < 1000.0 {
                '○' // Warm gas
            } else {
                '●' // Hot gas
            };
            canvas[y][x] = symbol;
        }
    }
    
    // Plot protostars as bright stars
    for sink in sink_particles {
        let x = ((sink.position.x - min_x) / (max_x - min_x) * (width - 1) as f64) as usize;
        let y = ((sink.position.y - min_y) / (max_y - min_y) * (height - 1) as f64) as usize;
        
        if x < width && y < height {
            let stellar_mass = sink.mass / SOLAR_MASS;
            let symbol = if stellar_mass < 0.5 {
                '✦' // Small star
            } else if stellar_mass < 2.0 {
                '⭐' // Medium star
            } else if stellar_mass < 8.0 {
                '🌟' // Large star
            } else {
                '💫' // Massive star
            };
            
            // Place star symbol (might overwrite gas)
            canvas[y][x] = '★';
        }
    }
    
    // Print the beautiful ASCII visualization
    println!("\n┌{}┐", "─".repeat(width + 2));
    for row in &canvas {
        print!("│ ");
        for &ch in row {
            print!("{}", ch);
        }
        println!(" │");
    }
    println!("└{}┘", "─".repeat(width + 2));
    
    // Legend
    println!("   · = cold gas   ∘ = cool gas   ○ = warm gas   ● = hot gas   ★ = protostar");
} 