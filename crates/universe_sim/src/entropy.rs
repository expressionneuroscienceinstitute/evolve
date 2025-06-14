//! Entropy engine for tracking universal entropy

use crate::types::*;
use crate::{Result, SimError};

/// Entropy engine that enforces the second law of thermodynamics
pub struct EntropyEngine {
    total_entropy: Entropy,
}

impl EntropyEngine {
    pub fn new() -> Self {
        Self {
            total_entropy: Entropy::zero(),
        }
    }
    
    /// Update entropy for one tick
    pub fn tick(&mut self, _tick: Tick, entropy_delta: Entropy) -> Result<()> {
        // Entropy can only increase
        if entropy_delta.as_f64() < 0.0 {
            return Err(SimError::PhysicsViolation(
                "Entropy cannot decrease".to_string()
            ));
        }
        
        self.total_entropy = self.total_entropy + entropy_delta;
        Ok(())
    }
    
    pub fn get_total_entropy(&self) -> Entropy {
        self.total_entropy
    }
}