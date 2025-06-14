//! Demo 01 – tiny Big-Bang run
//! Usage: `cargo run --bin big_bang_demo`

use physics_engine::PhysicsEngine;
use anyhow::Result;

fn main() -> Result<()> {
    // 10⁻23 s per step (roughly QCD scale)
    let dt = 1e-23;
    let mut engine = PhysicsEngine::new(dt)?;
    engine.initialize_big_bang()?;

    println!("tick,temperature_K,total_particles,photons,quarks,leptons");

    for tick in 0..1000u64 {
        engine.step(&mut [])?; // placeholder empty classical state array
        if tick % 100 == 0 {
            let temp = engine.temperature;
            let total = engine.particles.len();
            let photons = engine.particles.iter().filter(|p| matches!(p.particle_type, physics_engine::ParticleType::Photon)).count();
            let quarks = engine.particles.iter().filter(|p| physics_engine::particles::get_properties(p.particle_type).has_color).count();
            let leptons = total - quarks - photons;
            println!("{tick},{temp},{total},{photons},{quarks},{leptons}");
        }
    }

    Ok(())
}