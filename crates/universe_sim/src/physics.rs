//! Physics engine for the universe simulation
//!
//! This module implements the fundamental physics laws that govern the simulation:
//! - R1: Mass-energy conservation  
//! - R2: Entropy arrow (second law of thermodynamics)
//! - R3: Gravity (Newton + relativistic corrections)  
//! - R4: Fusion threshold (0.08 solar masses)
//! - R5: Nucleosynthesis (heavy elements from supernovae)

use crate::constants::physics::*;
use crate::constants::sim::*;
use crate::types::*;
use crate::{Result, SimError};
use nalgebra::{Vector3, Point3};
use rayon::prelude::*;
use std::collections::HashMap;

/// Main physics engine
pub struct PhysicsEngine {
    /// Current simulation tick
    current_tick: Tick,
    /// Gravitational constant (can be overridden in God-mode)
    gravitational_constant: f64,
    /// Speed of light (can be overridden in God-mode)
    speed_of_light: f64,
    /// Total universe mass-energy (for conservation checks)
    total_mass_energy: MassEnergy,
    /// Total universe entropy  
    total_entropy: Entropy,
    /// Relativistic corrections enabled
    relativistic: bool,
    /// Physics validation level
    validation_level: ValidationLevel,
}

impl PhysicsEngine {
    pub fn new(config: &crate::config::PhysicsConfig) -> Self {
        Self {
            current_tick: Tick::new(0),
            gravitational_constant: G,
            speed_of_light: C,
            total_mass_energy: MassEnergy::zero(),
            total_entropy: Entropy::zero(),
            relativistic: config.relativistic,
            validation_level: ValidationLevel::Standard,
        }
    }
    
    /// Update physics for one simulation tick
    pub fn tick(
        &mut self, 
        tick: Tick,
        celestial_bodies: &mut HashMap<StarId, CelestialBody>,
        dt_years: f64
    ) -> Result<PhysicsResults> {
        self.current_tick = tick;
        let dt_seconds = dt_years * 365.25 * 24.0 * 3600.0;
        
        // Step 1: Validate conservation laws (if enabled)
        if matches!(self.validation_level, ValidationLevel::Standard | ValidationLevel::Strict | ValidationLevel::Paranoid) {
            self.validate_conservation(celestial_bodies)?;
        }
        
        // Step 2: Apply gravitational forces and motion
        let gravity_results = self.apply_gravity(celestial_bodies, dt_seconds)?;
        
        // Step 3: Update stellar evolution and fusion
        let fusion_results = self.update_stellar_evolution(celestial_bodies, dt_seconds)?;
        
        // Step 4: Handle supernovae and nucleosynthesis
        let nucleosynthesis_results = self.process_nucleosynthesis(celestial_bodies, dt_seconds)?;
        
        // Step 5: Update entropy (must increase)
        let entropy_delta = self.calculate_entropy_increase(celestial_bodies, dt_seconds)?;
        self.total_entropy = self.total_entropy + entropy_delta;
        
        // Step 6: Final conservation check
        if matches!(self.validation_level, ValidationLevel::Strict | ValidationLevel::Paranoid) {
            self.validate_conservation(celestial_bodies)?;
        }
        
        Ok(PhysicsResults {
            tick,
            gravity_results,
            fusion_results,
            nucleosynthesis_results,
            entropy_delta,
            total_entropy: self.total_entropy,
            violations: Vec::new(),
        })
    }
    
    /// R1: Validate mass-energy conservation
    fn validate_conservation(&self, bodies: &HashMap<StarId, CelestialBody>) -> Result<()> {
        let current_total = bodies.values()
            .map(|b| b.mass)
            .fold(MassEnergy::zero(), |acc, m| acc + m);
            
        // Allow small numerical errors (0.01%)
        let relative_error = ((current_total.as_kg() - self.total_mass_energy.as_kg()).abs()) 
            / self.total_mass_energy.as_kg().max(1e-100);
            
        if relative_error > 1e-4 {
            return Err(SimError::ConservationViolation(
                format!("Mass-energy conservation violated: {:.2e} relative error", relative_error)
            ));
        }
        
        Ok(())
    }
    
    /// R3: Apply gravitational forces (F ∝ m₁·m₂ / r²)
    fn apply_gravity(
        &self,
        bodies: &mut HashMap<StarId, CelestialBody>,
        dt: f64
    ) -> Result<GravityResults> {
        let mut forces: HashMap<StarId, Vector3<f64>> = HashMap::new();
        let mut total_gravitational_pe = 0.0;
        
        // Calculate all pairwise forces
        let body_positions: Vec<(StarId, Point3<f64>, f64)> = bodies.iter()
            .map(|(id, body)| (*id, body.position.to_point3(), body.mass.as_kg()))
            .collect();
            
        for (i, (id1, pos1, mass1)) in body_positions.iter().enumerate() {
            for (id2, pos2, mass2) in body_positions.iter().skip(i + 1) {
                let r_vec = pos2 - pos1;
                let r = r_vec.magnitude();
                
                if r < 1e-10 {
                    continue; // Avoid singularity
                }
                
                // Newton's law: F = G * m1 * m2 / r²
                let force_magnitude = self.gravitational_constant * mass1 * mass2 / (r * r);
                let force_direction = r_vec.normalize();
                let force_vec = force_direction * force_magnitude;
                
                // Apply to both bodies (Newton's 3rd law)
                *forces.entry(*id1).or_insert(Vector3::zeros()) += force_vec;
                *forces.entry(*id2).or_insert(Vector3::zeros()) -= force_vec;
                
                // Accumulate gravitational potential energy
                total_gravitational_pe -= self.gravitational_constant * mass1 * mass2 / r;
            }
        }
        
        // Update velocities and positions using leap-frog integration
        let mut velocity_updates = Vec::new();
        for (id, force) in forces.iter() {
            if let Some(body) = bodies.get_mut(id) {
                let acceleration = *force / body.mass.as_kg();
                
                // Relativistic correction if enabled and v > 0.1c
                let new_velocity = if self.relativistic {
                    let v_mag = body.velocity.magnitude();
                    if v_mag > 0.1 * self.speed_of_light {
                        self.apply_relativistic_correction(body.velocity.to_vector3(), acceleration, dt)
                    } else {
                        body.velocity.to_vector3() + acceleration * dt
                    }
                } else {
                    body.velocity.to_vector3() + acceleration * dt
                };
                
                // Update velocity
                body.velocity = Velocity::new(new_velocity.x, new_velocity.y, new_velocity.z);
                velocity_updates.push((*id, new_velocity));
                
                // Update position
                let pos_delta = new_velocity * dt;
                body.position.x += pos_delta.x;
                body.position.y += pos_delta.y;
                body.position.z += pos_delta.z;
            }
        }
        
        Ok(GravityResults {
            force_calculations: forces.len(),
            total_gravitational_pe,
            velocity_updates,
        })
    }
    
    /// Apply special relativistic corrections for high velocities
    fn apply_relativistic_correction(
        &self,
        velocity: Vector3<f64>,
        acceleration: Vector3<f64>,
        dt: f64
    ) -> Vector3<f64> {
        let v = velocity.magnitude();
        let gamma = 1.0 / (1.0 - (v / self.speed_of_light).powi(2)).sqrt().max(1e-10);
        
        // Relativistic equation of motion
        let v_dot_a = velocity.dot(&acceleration);
        let new_velocity = velocity + (acceleration * dt - velocity * (v_dot_a * dt / (self.speed_of_light * self.speed_of_light))) / gamma;
        
        // Ensure we don't exceed speed of light
        let new_v_mag = new_velocity.magnitude();
        if new_v_mag >= self.speed_of_light {
            new_velocity * (0.99 * self.speed_of_light / new_v_mag)
        } else {
            new_velocity
        }
    }
    
    /// R4: Update stellar evolution and fusion
    fn update_stellar_evolution(
        &self,
        bodies: &mut HashMap<StarId, CelestialBody>,
        dt: f64
    ) -> Result<FusionResults> {
        let mut fusion_events = Vec::new();
        let mut total_fusion_energy = 0.0;
        
        for (id, body) in bodies.iter_mut() {
            if matches!(body.body_type, CelestialBodyType::Star) {
                let mass_solar = body.mass.as_kg() / M_SOL;
                
                // R4: Fusion threshold check
                if mass_solar >= FUSION_THRESHOLD_M_SOL {
                    // Main sequence fusion rate (simplified)
                    let fusion_rate = self.calculate_fusion_rate(mass_solar, body.temperature.as_kelvin());
                    let mass_converted = fusion_rate * dt;
                    let energy_released = mass_converted * self.speed_of_light * self.speed_of_light;
                    
                    // Convert mass to energy (Einstein's E=mc²)
                    body.mass = body.mass - MassEnergy::new(mass_converted);
                    body.luminosity += energy_released;
                    total_fusion_energy += energy_released;
                    
                    fusion_events.push(FusionEvent {
                        star_id: *id,
                        mass_converted,
                        energy_released,
                        fusion_type: FusionType::HydrogenBurning,
                    });
                    
                    // Update stellar temperature and radius based on mass-luminosity relation
                    self.update_stellar_properties(body, mass_solar);
                    
                    // Check for stellar death conditions
                    if mass_solar < 0.01 {
                        // Star has consumed most of its mass, becomes white dwarf
                        body.body_type = CelestialBodyType::WhiteDwarf;
                        body.luminosity *= 0.01; // White dwarfs are much dimmer
                    } else if mass_solar > 25.0 && body.age_ticks > 1000 {
                        // Massive star ready for supernova
                        fusion_events.push(FusionEvent {
                            star_id: *id,
                            mass_converted: 0.0,
                            energy_released: 1e44, // Typical supernova energy
                            fusion_type: FusionType::Supernova,
                        });
                    }
                } else {
                    // Below fusion threshold - brown dwarf
                    body.body_type = CelestialBodyType::BrownDwarf;
                    body.luminosity *= 0.0001; // Very dim
                }
                
                body.age_ticks += 1;
            }
        }
        
        Ok(FusionResults {
            fusion_events,
            total_energy_released: total_fusion_energy,
        })
    }
    
    /// Calculate stellar fusion rate based on mass and temperature
    fn calculate_fusion_rate(&self, mass_solar: f64, temperature_k: f64) -> f64 {
        // Simplified pp-chain fusion rate (proportional to M^4 for main sequence)
        let base_rate = 1e-20; // kg/s for 1 solar mass at 1.5e7 K
        let mass_factor = mass_solar.powf(4.0);
        let temp_factor = (temperature_k / 1.5e7).powf(4.0);
        
        base_rate * mass_factor * temp_factor
    }
    
    /// Update stellar properties based on mass-luminosity relation
    fn update_stellar_properties(&self, body: &mut CelestialBody, mass_solar: f64) {
        // Main sequence mass-luminosity relation: L ∝ M^3.5
        body.luminosity = L_SOL * mass_solar.powf(3.5);
        
        // Effective temperature from Stefan-Boltzmann law
        // Assuming radius scales as M^0.8
        let radius_factor = mass_solar.powf(0.8);
        let temp_factor = (mass_solar.powf(3.5) / (radius_factor * radius_factor)).powf(0.25);
        body.temperature = Temperature::new(5778.0 * temp_factor); // Sun's temp = 5778 K
    }
    
    /// R5: Process nucleosynthesis during supernovae
    fn process_nucleosynthesis(
        &self,
        bodies: &mut HashMap<StarId, CelestialBody>,
        dt: f64
    ) -> Result<NucleosynthesisResults> {
        let mut nucleosynthesis_events = Vec::new();
        let mut heavy_elements_produced = ElementTable::new();
        
        // Check for supernovae and neutron star mergers
        let supernovae: Vec<StarId> = bodies.iter()
            .filter(|(_, body)| {
                matches!(body.body_type, CelestialBodyType::Star) &&
                body.mass.as_kg() / M_SOL > 8.0 &&
                body.temperature.as_kelvin() > 5e9 // Core collapse threshold
            })
            .map(|(id, _)| *id)
            .collect();
            
        for star_id in supernovae {
            if let Some(body) = bodies.get_mut(&star_id) {
                // Supernova nucleosynthesis - produce heavy elements
                let mass_solar = body.mass.as_kg() / M_SOL;
                
                // R5: Heavy elements (Z > 2) only form in supernovae
                self.synthesize_heavy_elements(&mut heavy_elements_produced, mass_solar);
                
                // Explosion - most mass ejected, core becomes neutron star or black hole
                let ejected_mass = body.mass * 0.9; // 90% ejected
                body.mass = body.mass * 0.1; // Compact remnant
                
                if mass_solar > 25.0 {
                    body.body_type = CelestialBodyType::BlackHole;
                } else {
                    body.body_type = CelestialBodyType::NeutronStar;
                }
                
                nucleosynthesis_events.push(NucleosynthesisEvent {
                    star_id,
                    event_type: NucleosynthesisEventType::CoreCollapse,
                    ejected_mass,
                    elements_produced: heavy_elements_produced.clone(),
                });
                
                log::info!("Supernova at star {} produced {} kg of heavy elements", 
                          star_id, ejected_mass.as_kg());
            }
        }
        
        Ok(NucleosynthesisResults {
            events: nucleosynthesis_events,
            total_heavy_elements: heavy_elements_produced,
        })
    }
    
    /// Synthesize heavy elements during explosive nucleosynthesis
    fn synthesize_heavy_elements(&self, elements: &mut ElementTable, mass_solar: f64) {
        use crate::constants::elements::*;
        
        // Supernova yields based on initial mass (simplified)
        let base_yield = mass_solar * 0.1; // 10% of initial mass becomes heavy elements
        
        // R5: Elements heavier than helium only form in supernovae
        elements.add_abundance(C, (base_yield * 0.2 * 1e6) as u32);   // Carbon
        elements.add_abundance(O, (base_yield * 0.3 * 1e6) as u32);   // Oxygen  
        elements.add_abundance(NE, (base_yield * 0.05 * 1e6) as u32); // Neon
        elements.add_abundance(MG, (base_yield * 0.08 * 1e6) as u32); // Magnesium
        elements.add_abundance(SI, (base_yield * 0.1 * 1e6) as u32);  // Silicon
        elements.add_abundance(S, (base_yield * 0.05 * 1e6) as u32);  // Sulfur
        elements.add_abundance(CA, (base_yield * 0.02 * 1e6) as u32); // Calcium
        elements.add_abundance(FE, (base_yield * 0.15 * 1e6) as u32); // Iron
        
        // Trace heavy elements (including uranium for nuclear tech)
        if mass_solar > 20.0 {
            elements.add_abundance(AU, (base_yield * 1e-6 * 1e6) as u32); // Gold
            elements.add_abundance(U, (base_yield * 1e-7 * 1e6) as u32);  // Uranium
        }
    }
    
    /// R2: Calculate entropy increase (must be positive)
    fn calculate_entropy_increase(
        &self,
        bodies: &HashMap<StarId, CelestialBody>,
        dt: f64
    ) -> Result<Entropy> {
        let mut total_entropy_delta = 0.0;
        
        for body in bodies.values() {
            // Entropy increases from:
            // 1. Stellar fusion (nuclear to thermal)
            // 2. Gravitational radiation
            // 3. Heat conduction
            // 4. Expansion of universe
            
            let fusion_entropy = body.luminosity * dt / body.temperature.as_kelvin();
            let thermal_entropy = self.calculate_thermal_entropy_increase(body, dt);
            let gravitational_entropy = self.calculate_gravitational_entropy_increase(body, dt);
            
            total_entropy_delta += fusion_entropy + thermal_entropy + gravitational_entropy;
        }
        
        // R2: Entropy must increase (or at least not decrease significantly)
        if total_entropy_delta < -1e-12 {
            return Err(SimError::PhysicsViolation(
                format!("Entropy decreased by {:.2e} - violates second law", total_entropy_delta)
            ));
        }
        
        Ok(Entropy::new(total_entropy_delta))
    }
    
    fn calculate_thermal_entropy_increase(&self, body: &CelestialBody, dt: f64) -> f64 {
        // Simplified thermal entropy increase
        let heat_capacity = body.mass.as_kg() * 1000.0; // J/K (approximate)
        let temp_gradient = 100.0; // K (simplified)
        
        heat_capacity * temp_gradient / body.temperature.as_kelvin() * dt / (365.25 * 24.0 * 3600.0)
    }
    
    fn calculate_gravitational_entropy_increase(&self, body: &CelestialBody, dt: f64) -> f64 {
        // Gravitational wave entropy (for accelerating masses)
        let acceleration_magnitude: f64 = 1e-10; // m/s² (rough estimate)
        let gw_power = 32.0 / 5.0 * G.powi(4) * body.mass.as_kg().powi(2) * acceleration_magnitude.powi(2) / self.speed_of_light.powi(5);
        
        gw_power * dt / body.temperature.as_kelvin()
    }
    
    /// Override fundamental constants (God-mode)
    pub fn set_constants(&mut self, g: Option<f64>, c: Option<f64>) -> Result<()> {
        if let Some(new_g) = g {
            if new_g <= 0.0 {
                return Err(SimError::PhysicsViolation("Gravitational constant must be positive".to_string()));
            }
            self.gravitational_constant = new_g;
            log::warn!("God-mode: Gravitational constant changed to {:.2e}", new_g);
        }
        
        if let Some(new_c) = c {
            if new_c <= 0.0 {
                return Err(SimError::PhysicsViolation("Speed of light must be positive".to_string()));
            }
            self.speed_of_light = new_c;
            log::warn!("God-mode: Speed of light changed to {:.2e}", new_c);
        }
        
        Ok(())
    }
    
    /// Get current physics state for diagnostics
    pub fn get_physics_state(&self) -> PhysicsState {
        PhysicsState {
            current_tick: self.current_tick,
            total_mass_energy: self.total_mass_energy,
            total_entropy: self.total_entropy,
            gravitational_constant: self.gravitational_constant,
            speed_of_light: self.speed_of_light,
            relativistic_enabled: self.relativistic,
        }
    }
}

/// Celestial body representation
#[derive(Debug, Clone)]
pub struct CelestialBody {
    pub body_type: CelestialBodyType,
    pub mass: MassEnergy,
    pub position: Coord3D,
    pub velocity: Velocity,
    pub temperature: Temperature,
    pub luminosity: f64, // Watts
    pub age_ticks: u64,
}

#[derive(Debug, Clone)]
pub enum CelestialBodyType {
    Star,
    BrownDwarf,
    WhiteDwarf,
    NeutronStar,
    BlackHole,
    Planet,
}

/// Results from physics calculations
#[derive(Debug)]
pub struct PhysicsResults {
    pub tick: Tick,
    pub gravity_results: GravityResults,
    pub fusion_results: FusionResults,
    pub nucleosynthesis_results: NucleosynthesisResults,
    pub entropy_delta: Entropy,
    pub total_entropy: Entropy,
    pub violations: Vec<String>,
}

#[derive(Debug)]
pub struct GravityResults {
    pub force_calculations: usize,
    pub total_gravitational_pe: f64,
    pub velocity_updates: Vec<(StarId, Vector3<f64>)>,
}

#[derive(Debug)]
pub struct FusionResults {
    pub fusion_events: Vec<FusionEvent>,
    pub total_energy_released: f64,
}

#[derive(Debug)]
pub struct FusionEvent {
    pub star_id: StarId,
    pub mass_converted: f64,
    pub energy_released: f64,
    pub fusion_type: FusionType,
}

#[derive(Debug)]
pub enum FusionType {
    HydrogenBurning,
    HeliumBurning,
    CarbonBurning,
    Supernova,
}

#[derive(Debug)]
pub struct NucleosynthesisResults {
    pub events: Vec<NucleosynthesisEvent>,
    pub total_heavy_elements: ElementTable,
}

#[derive(Debug)]
pub struct NucleosynthesisEvent {
    pub star_id: StarId,
    pub event_type: NucleosynthesisEventType,
    pub ejected_mass: MassEnergy,
    pub elements_produced: ElementTable,
}

#[derive(Debug)]
pub enum NucleosynthesisEventType {
    CoreCollapse,
    NeutronStarMerger,
}

#[derive(Debug)]
pub struct PhysicsState {
    pub current_tick: Tick,
    pub total_mass_energy: MassEnergy,
    pub total_entropy: Entropy,
    pub gravitational_constant: f64,
    pub speed_of_light: f64,
    pub relativistic_enabled: bool,
}

/// Validation level for physics checks
#[derive(Debug, Clone)]
pub enum ValidationLevel {
    None,
    Basic,
    Standard,
    Strict,
    Paranoid,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::PhysicsConfig;
    
    #[test]
    fn test_conservation_laws() {
        let config = PhysicsConfig::default();
        let mut engine = PhysicsEngine::new(&config);
        
        // Test mass-energy conservation
        let initial_mass = MassEnergy::new(M_SOL);
        engine.total_mass_energy = initial_mass;
        
        let mut bodies = HashMap::new();
        bodies.insert(StarId::new(1), CelestialBody {
            body_type: CelestialBodyType::Star,
            mass: initial_mass,
            position: Coord3D::origin(),
            velocity: Velocity::zero(),
            temperature: Temperature::new(5778.0),
            luminosity: L_SOL,
            age_ticks: 0,
        });
        
        assert!(engine.validate_conservation(&bodies).is_ok());
    }
    
    #[test]
    fn test_fusion_threshold() {
        let config = PhysicsConfig::default();
        let mut engine = PhysicsEngine::new(&config);
        
        // Test brown dwarf (below threshold)
        let brown_dwarf_mass = MassEnergy::new(0.05 * M_SOL);
        assert!(brown_dwarf_mass.as_kg() / M_SOL < FUSION_THRESHOLD_M_SOL);
        
        // Test main sequence star (above threshold)  
        let star_mass = MassEnergy::new(1.0 * M_SOL);
        assert!(star_mass.as_kg() / M_SOL >= FUSION_THRESHOLD_M_SOL);
    }
    
    #[test]
    fn test_entropy_increase() {
        let config = PhysicsConfig::default();
        let engine = PhysicsEngine::new(&config);
        
        let bodies = HashMap::new();
        let entropy_delta = engine.calculate_entropy_increase(&bodies, 1.0);
        
        // Entropy should not decrease
        assert!(entropy_delta.is_ok());
        assert!(entropy_delta.unwrap().as_f64() >= 0.0);
    }
}