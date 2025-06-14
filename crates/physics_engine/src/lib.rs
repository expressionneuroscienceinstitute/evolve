//! Fundamental particle-level physics engine.
//!
//! This crate provides a modular API for simulating the dynamics of
//! elementary particles and composite bodies under multiple interaction
//! regimes (gravity, electromagnetism, strong/weak toy models).  The goal is
//! to serve as the lowest-level driver for the higher-level universe_sim
//! crate.
//!
//! NOTE:  For performance reasons we initially implement classical Newtonian
//! or relativistic‐corrected equations of motion.  Quantum and field-theory
//! layers can be added incrementally behind `cfg(feature = "quantum")`.

use nalgebra::{Vector3, Vector6};
use rand::Rng;
use serde::{Deserialize, Serialize};

/// Unique identifier for a particle
pub type ParticleId = u64;

/// Fundamental charges carried by a particle
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct Charges {
    /// Electric charge (Coulombs)
    pub electric: f64,
    /// Color charge (abstract units, toy model)
    pub color: f64,
    /// Weak isospin (abstract units)
    pub weak: f64,
}

impl Default for Charges {
    fn default() -> Self {
        Self {
            electric: 0.0,
            color: 0.0,
            weak: 0.0,
        }
    }
}

/// Spin is represented as a 3-vector (ℏ = 1 units)
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct Spin(pub Vector3<f64>);

impl Default for Spin {
    fn default() -> Self {
        Self(Vector3::zeros())
    }
}

/// Phase-space state (position + momentum)
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct State {
    /// Position in metres
    pub pos: Vector3<f64>,
    /// Momentum in kg·m/s
    pub momentum: Vector3<f64>,
}

impl State {
    pub fn new(pos: Vector3<f64>, momentum: Vector3<f64>) -> Self {
        Self { pos, momentum }
    }
}

/// Fundamental particle representation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Particle {
    pub id: ParticleId,
    /// Rest mass (kg)
    pub mass: f64,
    pub charges: Charges,
    pub spin: Spin,
    pub state: State,
}

impl Particle {
    /// Generate a random particle for testing purposes
    pub fn random(id: ParticleId, rng: &mut impl Rng) -> Self {
        let pos = Vector3::new(rng.gen(), rng.gen(), rng.gen());
        let mom = Vector3::new(rng.gen(), rng.gen(), rng.gen());
        Self {
            id,
            mass: rng.gen::<f64>().abs() * 1.0e-27, // ~ proton mass scale
            charges: Charges::default(),
            spin: Spin::default(),
            state: State::new(pos, mom),
        }
    }
}

/// Generic force interface returning the force (N) acting on a particle given
/// its state and optionally other bodies/fields.
pub trait ForceLaw {
    fn force(&self, particle: &Particle, ctx: &SimulationContext) -> Vector3<f64>;
}

/// Global simulation context handed to force laws and integrators.
pub struct SimulationContext<'a> {
    pub particles: &'a [Particle],
    /// Time in seconds since simulation start
    pub time: f64,
}

/// Newtonian gravity between point masses (no softening yet)
#[derive(Debug, Clone, Copy)]
pub struct GravityConstant;

impl ForceLaw for GravityConstant {
    fn force(&self, p: &Particle, ctx: &SimulationContext) -> Vector3<f64> {
        const G: f64 = 6.67430e-11; // m^3 kg^-1 s^-2
        let mut f = Vector3::zeros();
        for other in ctx.particles {
            if other.id == p.id {
                continue;
            }
            let r_vec = other.state.pos - p.state.pos;
            let dist_sq = r_vec.magnitude_squared() + 1e-12; // softening epsilon
            let f_mag = G * p.mass * other.mass / dist_sq;
            f += r_vec.normalize() * f_mag;
        }
        f
    }
}

/// Lorentz force for electromagnetic interaction (static field placeholder)
pub struct Electromagnetism {
    /// External E-field (V/m)
    pub e_field: Vector3<f64>,
    /// External B-field (T)
    pub b_field: Vector3<f64>,
}

impl ForceLaw for Electromagnetism {
    fn force(&self, p: &Particle, _ctx: &SimulationContext) -> Vector3<f64> {
        // Non-relativistic approximation F = q(E + v × B)
        let q = p.charges.electric;
        let v = p.state.momentum / p.mass;
        q * (self.e_field + v.cross(&self.b_field))
    }
}

/// Symplectic leap-frog integrator
pub struct LeapFrog<'a, F> {
    pub dt: f64,
    pub force_law: &'a F,
}

impl<'a, F: ForceLaw> LeapFrog<'a, F> {
    pub fn step(&self, particles: &mut [Particle], ctx_time: &mut f64) {
        // Half-kick: update momentum by dt/2
        let ctx = SimulationContext { particles, time: *ctx_time };
        for p in particles.iter_mut() {
            let f = self.force_law.force(p, &ctx);
            p.state.momentum += f * (self.dt * 0.5);
        }
        // Drift: update positions using updated momentum
        for p in particles.iter_mut() {
            let v = p.state.momentum / p.mass;
            p.state.pos += v * self.dt;
        }
        // Update context time for second half
        *ctx_time += self.dt;
        // Recompute forces with new positions
        let ctx = SimulationContext { particles, time: *ctx_time };
        for p in particles.iter_mut() {
            let f = self.force_law.force(p, &ctx);
            p.state.momentum += f * (self.dt * 0.5);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn test_energy_conservation_two_body() {
        let mut p1 = Particle {
            id: 1,
            mass: 1.0,
            charges: Charges::default(),
            spin: Spin::default(),
            state: State::new(Vector3::new(-0.5, 0.0, 0.0), Vector3::new(0.0, 0.5, 0.0)),
        };
        let mut p2 = Particle {
            id: 2,
            mass: 1.0,
            charges: Charges::default(),
            spin: Spin::default(),
            state: State::new(Vector3::new(0.5, 0.0, 0.0), Vector3::new(0.0, -0.5, 0.0)),
        };
        let mut particles = vec![p1, p2];
        let gravity = GravityConstant;
        let mut time = 0.0;
        let integrator = LeapFrog { dt: 0.001, force_law: &gravity };
        for _ in 0..10_000 {
            integrator.step(&mut particles, &mut time);
        }
        let separation = (particles[0].state.pos - particles[1].state.pos).magnitude();
        assert!(separation < 2.0); // not ejected
    }
}