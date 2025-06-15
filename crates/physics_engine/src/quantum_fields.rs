//! # Physics Engine: Quantum Field Lattice Module
//!
//! This module provides a representation of a quantum field on a discrete lattice.
//! It's a foundational element for simulations involving quantum field theory (QFT),
//! such as lattice QCD. This implementation focuses on a simple scalar field.

use anyhow::Result;
use ndarray::{Array, Ix3};
use rand::Rng;
use rand::thread_rng;
use rand_distr::{Distribution, Normal};

/// Represents a scalar quantum field on a 3D lattice.
#[derive(Debug, Clone)]
pub struct QuantumField {
    /// The values of the field at each lattice point.
    pub field: Array<f64, Ix3>,
    /// The spacing between lattice points.
    pub lattice_spacing: f64,
}

impl QuantumField {
    /// Creates a new quantum field initialized to a vacuum state.
    pub fn new(size: (usize, usize, usize), lattice_spacing: f64) -> Self {
        QuantumField {
            field: Array::zeros(size),
            lattice_spacing,
        }
    }

    /// Initializes the field to a random state, simulating vacuum fluctuations.
    pub fn initialize_vacuum_fluctuations(&mut self) {
        let mut rng = thread_rng();
        // The standard deviation of the fluctuations is related to the energy scale.
        // This is a simplified model.
        let dist = Normal::new(0.0, 1.0).unwrap();
        for val in self.field.iter_mut() {
            *val = dist.sample(&mut rng);
        }
    }

    /// Evolves the field over a single time step using a simplified wave equation.
    pub fn evolve(&mut self, _time_step: f64) {
        // A proper implementation would solve the Klein-Gordon equation.
        // For now, we'll just add some small random noise to simulate evolution.
        let mut rng = thread_rng();
        let dist = Normal::new(0.0, 0.1).unwrap(); // Small perturbations
        self.field.mapv_inplace(|v| v + dist.sample(&mut rng));
    }

    /// This function simulates the effect of quantum fluctuations on the field.
    pub fn apply_quantum_fluctuations(&mut self, temperature: f64) -> Result<()> {
        let mut rng = thread_rng();
        let std_dev = self.calculate_fluctuation_amplitude(temperature);
        let dist = Normal::new(0.0, std_dev)?;

        for val in self.field.iter_mut().flatten().flatten() {
            *val += dist.sample(&mut rng);
        }

        Ok(())
    }

    /// Add a specified amount of energy to the field, simulating an interaction.
    pub fn add_energy(&mut self, energy: f64) {
        let mut rng = thread_rng();
        // The energy is distributed as a Gaussian pulse.
        let dist = Normal::new(0.0, energy.sqrt()).unwrap();
        self.field.iter_mut().flatten().flatten().for_each(|v| *v += dist.sample(&mut rng));
    }
    
    /// Calculates the amplitude of vacuum fluctuations based on temperature.
    fn calculate_fluctuation_amplitude(&self, temperature: f64) -> f64 {
        // Simplified model: fluctuation amplitude is proportional to temperature.
        temperature * 1e-5
    }
}