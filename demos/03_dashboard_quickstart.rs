//! Quickstart demo streaming a handful of particles for dashboard UI testing
use physics_engine::{PhysicsEngine, ParticleType, FundamentalParticle, QuantumState};
use nalgebra::Vector3;
use anyhow::Result;
use std::f64::consts::PI;
use std::f64::Complex;

fn main() -> Result<()> {
    let mut engine = PhysicsEngine::new(1e-19)?;
    engine.particles.clear();
    // Add 100 protons, 100 electrons, 500 photons
    for _ in 0..100 {
        engine.particles.push(simple_particle(ParticleType::Proton, 1.6e-19));
        engine.particles.push(simple_particle(ParticleType::Electron, -1.6e-19));
    }
    for _ in 0..500 {
        engine.particles.push(simple_particle(ParticleType::Photon, 0.0));
    }

    // Run a short simulation and stream frames
    let steps = 600; // ~10 s at 60 FPS
    for _ in 0..steps {
        engine.step(&mut [])?;
        // Export via websocket if flag on physics engine (omitted)
    }
    Ok(())
}

fn simple_particle(ptype: ParticleType, charge: f64) -> FundamentalParticle {
    FundamentalParticle {
        particle_type: ptype,
        position: Vector3::zeros(),
        momentum: Vector3::zeros(),
        spin: Vector3::new(Complex::new(0.0, 0.0), Complex::new(0.0, 0.0), Complex::new(0.0, 0.0)),
        color_charge: None,
        electric_charge: charge,
        mass: 938.272,
        energy: 0.0,
        creation_time: 0.0,
        decay_time: None,
        quantum_state: QuantumState::default(),
        interaction_history: Vec::new(),
    }
}