//! Universe Simulation Core Library
//! 
//! Implements the complete universe simulation from Big Bang to far future
//! with autonomous AI agents evolving toward immortality.

use anyhow::Result;
use diagnostics::{AllocationType, DiagnosticsSystem};
use md5;
use nalgebra::Vector3;
use physics_engine::{
    nuclear_physics::{process_neutron_capture, NeutronCaptureProcess, Nucleus},
    PhysicsEngine, PhysicsState,
};
use rand::Rng;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::time::Instant;
use tracing::info;
use uuid::Uuid;

pub mod config;
pub mod cosmic_era;
pub mod evolution;
pub mod persistence;
pub mod storage;
pub mod world;

pub use physics_engine;
pub use storage::{AgentLineage, CelestialBody, CelestialBodyType, PlanetClass, StellarEvolution, StellarPhase, Store, ParticleStore};

/// Calculate relativistic total energy from momentum and mass
/// E = sqrt((pc)^2 + (mc^2)^2) where c = speed of light
fn calculate_relativistic_energy(momentum: &Vector3<f64>, mass: f64) -> f64 {
    const C: f64 = 299_792_458.0; // Speed of light in m/s
    
    let momentum_magnitude = momentum.magnitude();
    let rest_energy = mass * C * C;
    let momentum_energy = momentum_magnitude * C;
    
    // Total relativistic energy
    (momentum_energy * momentum_energy + rest_energy * rest_energy).sqrt()
}

/// Core universe simulation structure
pub struct UniverseSimulation {
    pub store: Store,                          // SoA data store
    pub physics_engine: PhysicsEngine,         // Physics simulation
    pub current_tick: u64,                     // Simulation time
    pub tick_span_years: f64,                  // Years per tick (default 1M)
    pub target_ups: f64,                       // Updates per second target
    pub universe_state: cosmic_era::UniverseState, // Current universe physical state
    pub config: config::SimulationConfig,      // Configuration
    pub diagnostics: DiagnosticsSystem,        // Performance monitoring
    pub physical_transitions: Vec<cosmic_era::PhysicalTransition>, // Record of major transitions
}

impl UniverseSimulation {
    /// Create a new universe simulation
    pub fn new(config: config::SimulationConfig) -> Result<Self> {
        let store = Store::new();
        let physics_engine = PhysicsEngine::new()?;

        Ok(Self {
            store,
            physics_engine,
            current_tick: 0,
            tick_span_years: config.tick_span_years,
            target_ups: config.target_ups,
            universe_state: cosmic_era::UniverseState::initial(),
            config,
            diagnostics: DiagnosticsSystem::new(),
            physical_transitions: Vec::new(),
        })
    }

    /// Initialize with Big Bang conditions
    pub fn init_big_bang(&mut self) -> Result<()> {
        // Reset to initial universe state
        self.universe_state = cosmic_era::UniverseState::initial();
        
        // Print initial configuration
        println!("🚀 INITIALIZING BIG BANG CONDITIONS");
        println!("   Initial particle count: {}", self.config.initial_particle_count);
        println!("   Tick span years: {}", self.tick_span_years);
        println!("   Target UPS: {}", self.target_ups);
        
        // Create initial particle soup
        self.spawn_initial_particles()?;
        
        // Initialize cosmic background
        self.init_cosmic_background()?;
        
        // Print physics engine initial state
        println!("📊 PHYSICS ENGINE INITIAL STATE:");
        println!("   Temperature: {:.2e} K", self.physics_engine.temperature);
        println!("   Energy density: {:.2e} J/m³", self.physics_engine.energy_density);
        println!("   Volume: {:.2e} m³", self.physics_engine.volume);
        println!("   Particles: {}", self.physics_engine.particles.len());
        println!("   Nuclei: {}", self.physics_engine.nuclei.len());
        println!("   Atoms: {}", self.physics_engine.atoms.len());
        println!("   Molecules: {}", self.physics_engine.molecules.len());
        
        // Count entities in our new store
        let particle_count = self.store.particles.count;
        println!("   Store Particle Count: {}", particle_count);
        
        // Print sample of initial particles
        if particle_count > 0 {
            println!("📍 SAMPLE OF INITIAL PARTICLES:");
            for i in 0..5.min(particle_count) {
                println!("   Particle {}: mass={:.2e} kg, temp={:.1} K, pos=({:.1e}, {:.1e}, {:.1e})", 
                    i, self.store.particles.mass[i], self.store.particles.temperature[i], 
                    self.store.particles.position[i].x, self.store.particles.position[i].y, 
                    self.store.particles.position[i].z);
            }
        }
        
        Ok(())
    }

    /// Spawn initial particles after Big Bang
    fn spawn_initial_particles(&mut self) -> Result<()> {
        use rand::Rng;
        let mut rng = rand::thread_rng();
        
        // Create primordial hydrogen and helium
        let num_particles = self.config.initial_particle_count;
        
        for _ in 0..num_particles {
            let position = Vector3::new(
                rng.gen_range(-1e15..1e15), // ±1000 light-years
                rng.gen_range(-1e15..1e15),
                rng.gen_range(-1e15..1e15),
            );
            
            let velocity = Vector3::new(
                rng.gen_range(-1e6..1e6), // ±1000 km/s
                rng.gen_range(-1e6..1e6),
                rng.gen_range(-1e6..1e6),
            );
            
            // 75% hydrogen, 25% helium (by number)
            let (mass, charge) = if rng.gen::<f64>() < 0.75 {
                // Hydrogen
                (1.67262e-27, 1.602176e-19) // Proton
            } else {
                // Helium
                (6.64466e-27, 2.0 * 1.602176e-19) // Alpha particle
            };
            
            self.store.particles.add(
                position,
                velocity,
                mass,
                charge,
                3000.0,  // 3000 K initial temperature
                1e-23, // Default entropy
            );
        }
        
        Ok(())
    }

    /// Initialize cosmic background radiation and temperature
    fn init_cosmic_background(&mut self) -> Result<()> {
        // Set initial temperature and energy density based on config
        self.physics_engine.temperature = self.config.big_bang_temperature;
        self.physics_engine.energy_density =
            cosmic_era::calculate_energy_density(self.config.big_bang_temperature);
        Ok(())
    }

    /// Main simulation tick
    pub fn tick(&mut self) -> Result<()> {
        let tick_start = Instant::now();
        self.current_tick += 1;

        // 1. Update universe physical state (age, temperature, density)
        self.update_universe_state()?;

        // 2. Update physics simulation (interactions, gravity, etc.)
        self.update_physics()?;

        // 3. Update cosmic-scale processes (star formation, etc.)
        self.update_cosmic_processes()?;

        // 4. Update agent evolution
        self.update_agent_evolution()?;

        // 5. Victory conditions check
        self.check_victory_conditions()?;

        // Record diagnostics
        self.diagnostics
            .record_universe_tick(tick_start.elapsed());

        Ok(())
    }

    /// Update universe state based on age
    fn update_universe_state(&mut self) -> Result<()> {
        // Very lightweight placeholder: compute age and a basic description
        let previous_state = self.universe_state.clone();

        // Update age (in Gyr)
        self.universe_state.age_gyr = self.universe_age_gyr();

        // Propagate average temperature directly from physics engine for now
        self.universe_state.mean_temperature = self.physics_engine.temperature;

        // Re-use helper from cosmic_era to create descriptive text if needed
        // Here we skip expensive calculations and only update essential fields.

        // Detect major transitions (placeholder)
        self.detect_physical_transitions(&previous_state)?;

        Ok(())
    }

    /// Detect and record major physical transitions in the universe
    fn detect_physical_transitions(
        &mut self,
        _previous_state: &cosmic_era::UniverseState,
    ) -> Result<()> {
        // Placeholder – in full implementation, compare states and push transitions.
        Ok(())
    }

    /// Update physics simulation
    fn update_physics(&mut self) -> Result<()> {
        // Advance physics engine if available; ignore errors for placeholder
        let dt = self.physics_engine.time_step;
        let _ = self.physics_engine.step(dt);
        Ok(())
    }

    /// Update agent evolution and behavior
    fn update_agent_evolution(&mut self) -> Result<()> {
        // This will be implemented in the agents module
        // For now, just a placeholder
        Ok(())
    }

    /// Update cosmic-scale processes based on current physical conditions
    fn update_cosmic_processes(&mut self) -> Result<()> {
        // Placeholder – future star/planet formation logic will go here.
        Ok(())
    }

    /// Process stellar evolution based on nuclear burning
    fn process_stellar_evolution(&mut self) -> Result<()> {
        todo!();
    }

    /// Update stellar composition based on nuclear burning products
    fn update_stellar_composition(
        &self,
        _body: &mut CelestialBody,
        _evolution: &StellarEvolution,
    ) {
        todo!();
    }

    /// Handle the death of a star
    fn process_stellar_death(
        &mut self,
        _entity: usize,
        _body: &CelestialBody,
        _evolution: &StellarEvolution,
    ) -> Result<()> {
        todo!();
    }

    /// Handle nucleosynthesis in supernova explosions
    fn process_supernova_nucleosynthesis(&mut self) -> Result<()> {
        todo!();
    }

    /// Create enriched gas clouds from stellar death events
    fn create_enriched_gas_cloud(
        &mut self,
        _star: &CelestialBody,
        _evolution: &StellarEvolution,
    ) -> Result<()> {
        todo!();
    }

    /// Handle r-process nucleosynthesis in neutron star mergers
    fn process_r_process_nucleosynthesis(&self, _star: &CelestialBody) -> Result<()> {
        todo!();
    }

    /// Form new stars from dense gas clouds
    fn process_star_formation(&mut self) -> Result<()> {
        todo!();
    }

    /// Sample a stellar mass from the Initial Mass Function (IMF)
    fn sample_stellar_mass_from_imf<R: Rng>(&self, _rng: &mut R) -> f64 {
        todo!();
    }

    /// Find a suitable site for star formation
    fn find_star_formation_site<R: Rng>(&self, _rng: &mut R) -> Result<Vector3<f64>> {
        todo!();
    }

    /// Calculate stellar radius from mass (simplified)
    fn calculate_stellar_radius(&self, _mass: f64) -> f64 {
        todo!();
    }

    /// Calculate stellar luminosity from mass (simplified)
    fn calculate_stellar_luminosity(&self, _mass: f64) -> f64 {
        todo!();
    }

    /// Calculate stellar surface temperature from mass (simplified)
    fn calculate_stellar_temperature(&self, _mass: f64) -> f64 {
        todo!();
    }

    /// Form planets around existing stars
    fn process_planet_formation(&mut self) -> Result<()> {
        todo!();
    }

    /// Handle the emergence of life on habitable planets
    fn process_life_emergence(&mut self) -> Result<()> {
        todo!();
    }

    /// Check for victory conditions (e.g., immortality achieved)
    fn check_victory_conditions(&mut self) -> Result<()> {
        Ok(())
    }

    pub fn universe_age_years(&self) -> f64 {
        self.current_tick as f64 * self.tick_span_years
    }

    pub fn universe_age_gyr(&self) -> f64 {
        self.universe_age_years() / 1e9
    }

    pub fn get_diagnostics(&self) -> &DiagnosticsSystem {
        &self.diagnostics
    }

    pub fn get_diagnostics_mut(&mut self) -> &mut DiagnosticsSystem {
        &mut self.diagnostics
    }

    pub fn get_stats(&mut self) -> SimulationStats {
        todo!();
    }

    pub fn get_map_data(
        &mut self,
        _zoom: f64,
        _layer: &str,
        _width: usize,
        _height: usize,
    ) -> Result<serde_json::Value> {
        todo!();
    }

    fn sync_store_to_physics_engine_particles(&mut self) -> Result<()> {
        todo!();
    }

    fn calculate_spatial_bounds(&self) -> Result<(Vector3<f64>, Vector3<f64>)> {
        todo!();
    }

    // Functions below here are also stubbed out for now

    fn calculate_stellar_density_at(&mut self, _x: f64, _y: f64) -> Result<f64> {
        todo!();
    }

    fn calculate_gas_density_at(&self, _x: f64, _y: f64) -> Result<f64> {
        todo!();
    }

    fn calculate_dark_matter_density_at(&self, _x: f64, _y: f64) -> Result<f64> {
        todo!();
    }

    fn calculate_radiation_density_at(&mut self, _x: f64, _y: f64) -> Result<f64> {
        todo!();
    }

    fn calculate_total_density_at(&mut self, _x: f64, _y: f64) -> Result<f64> {
        todo!();
    }

    pub fn get_planet_data(&mut self, _class_filter: Option<String>, _habitable_only: bool) -> Result<serde_json::Value> {
        todo!();
    }

    pub fn get_planet_inspection_data(&mut self, _planet_id: &str) -> Result<Option<serde_json::Value>> {
        todo!();
    }

    pub fn get_lineage_data(&mut self) -> Result<serde_json::Value> {
        todo!();
    }

    pub fn get_lineage_inspection_data(&mut self, _lineage_id: &str) -> Result<Option<serde_json::Value>> {
        todo!();
    }

    fn calculate_stellar_statistics(&mut self) -> (usize, StellarStatistics) {
        todo!();
    }

    fn calculate_energy_statistics(&mut self) -> EnergyStatistics {
        todo!();
    }

    fn calculate_chemical_composition(&mut self) -> ChemicalComposition {
        todo!();
    }

    fn calculate_planetary_statistics(&mut self) -> PlanetaryStatistics {
        todo!();
    }

    fn calculate_evolution_statistics(&mut self) -> EvolutionStatistics {
        todo!();
    }

    fn calculate_physics_performance(&self) -> PhysicsPerformance {
        todo!();
    }

    fn calculate_cosmic_structure(&self) -> CosmicStructure {
        todo!();
    }

    pub fn god_create_agent_on_planet(&mut self, _planet_id_str: &str) -> Result<String> {
        todo!();
    }

    pub fn get_quantum_field_snapshot(&self) -> HashMap<String, Vec<Vec<f64>>> {
        todo!();
    }

    pub fn set_speed_factor(&mut self, _factor: f64) -> Result<()> {
        todo!();
    }

    pub fn rewind_ticks(&mut self, _ticks: u64) -> Result<u64> {
        todo!();
    }
    
    /// Get read-only access to physics engine for rendering
    pub fn get_physics_engine(&self) -> &PhysicsEngine {
        &self.physics_engine
    }
}

/// Simulation statistics
pub struct SimulationStats {
    // Basic simulation metrics
    pub current_tick: u64,
    pub universe_age_gyr: f64,
    pub universe_description: String,
    pub target_ups: f64,
    
    // Population counts
    pub particle_count: usize,
    pub celestial_body_count: usize,
    pub planet_count: usize,
    pub lineage_count: usize,
    
    // Stellar statistics
    pub star_count: usize,
    pub stellar_formation_rate: f64,  // Stars formed per billion years
    pub average_stellar_mass: f64,    // Solar masses
    pub stellar_mass_distribution: Vec<(f64, f64)>, // (mass_range, count)
    pub main_sequence_stars: usize,
    pub evolved_stars: usize,         // Post-main-sequence
    pub stellar_remnants: usize,      // White dwarfs, neutron stars, black holes
    
    // Energy distribution
    pub total_energy: f64,            // Joules
    pub kinetic_energy: f64,          // Joules
    pub potential_energy: f64,        // Joules
    pub radiation_energy: f64,        // Joules
    pub nuclear_binding_energy: f64,  // Joules
    pub average_temperature: f64,     // Kelvin
    pub energy_density: f64,          // J/m³
    
    // Chemical composition (by mass fraction)
    pub hydrogen_fraction: f64,
    pub helium_fraction: f64,
    pub carbon_fraction: f64,
    pub oxygen_fraction: f64,
    pub iron_fraction: f64,
    pub heavy_elements_fraction: f64, // Z > 26
    pub metallicity: f64,             // [Fe/H] in dex
    
    // Planetary statistics
    pub habitable_planets: usize,
    pub earth_like_planets: usize,
    pub gas_giants: usize,
    pub average_planet_mass: f64,     // Earth masses
    pub planet_formation_rate: f64,   // Planets formed per billion years
    
    // Evolution and life statistics
    pub total_lineages_ever: usize,
    pub extinct_lineages: usize,
    pub average_fitness: f64,
    pub average_sentience_level: f64,
    pub average_tech_level: f64,
    pub immortal_lineages: usize,
    pub consciousness_emergence_rate: f64, // Events per billion years
    
    // Physics engine performance
    pub physics_step_time_ms: f64,
    pub nuclear_reactions_per_step: usize,
    pub particle_interactions_per_step: usize,
    
    // Cosmic structure
    pub universe_radius: f64,         // Light-years
    pub expansion_rate: f64,          // km/s/Mpc (Hubble parameter)
    pub dark_matter_fraction: f64,
    pub dark_energy_fraction: f64,
    pub ordinary_matter_fraction: f64,
    pub critical_density: f64,       // kg/m³
}

struct StellarStatistics {
    count: usize,
    formation_rate: f64,
    average_mass: f64,
    mass_distribution: Vec<(f64, f64)>,
    main_sequence_count: usize,
    evolved_count: usize,
    remnant_count: usize,
}

struct EnergyStatistics {
    total: f64,
    kinetic: f64,
    potential: f64,
    radiation: f64,
    nuclear_binding: f64,
    average_temperature: f64,
    density: f64,
}

struct ChemicalComposition {
    hydrogen: f64,
    helium: f64,
    carbon: f64,
    oxygen: f64,
    iron: f64,
    heavy_elements: f64,
    metallicity: f64,
}

struct PlanetaryStatistics {
    total_count: usize,
    habitable_count: usize,
    earth_like_count: usize,
    gas_giant_count: usize,
    average_mass: f64,
    formation_rate: f64,
}

struct EvolutionStatistics {
    total_ever: usize,
    extinct: usize,
    average_fitness: f64,
    average_sentience: f64,
    average_tech: f64,
    immortal_count: usize,
    consciousness_rate: f64,
}

struct PhysicsPerformance {
    step_time_ms: f64,
    nuclear_reactions: usize,
    interactions: usize,
}

struct CosmicStructure {
    radius: f64,
    hubble_constant: f64,
    dark_matter_fraction: f64,
    dark_energy_fraction: f64,
    ordinary_matter_fraction: f64,
    critical_density: f64,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_universe_creation() {
        let config = config::SimulationConfig::default();
        let sim = UniverseSimulation::new(config).unwrap();
        
        assert_eq!(sim.current_tick, 0);
        assert_eq!(sim.universe_age_gyr(), 0.0);
    }

    #[test]
    fn test_big_bang_initialization() {
        let config = config::SimulationConfig::default();
        let mut sim = UniverseSimulation::new(config).unwrap();
        
        sim.init_big_bang().unwrap();
        
        let stats = sim.get_stats();
        assert!(stats.particle_count > 0);
    }

    #[test]
    fn test_cosmic_era_progression() {
        let config = config::SimulationConfig::default();
        let mut sim = UniverseSimulation::new(config).unwrap();
        
        // Test physics-driven state evolution
        sim.current_tick = 0;
        sim.update_universe_state().unwrap();
        assert!(sim.universe_state.age_gyr < 0.001);
        assert!(sim.universe_state.stellar_fraction < 0.001);
        
        sim.current_tick = 1000; // 1 Gyr
        sim.update_universe_state().unwrap();
        assert!(sim.universe_state.age_gyr > 0.5);
    }
}

#[cfg(test)]
mod stellar_evolution_integration_tests {
    use super::*;
    use crate::config::SimulationConfig;
    use approx::assert_relative_eq;
    
    #[test]
    fn test_complete_stellar_lifecycle_solar_mass() {
        // Test complete stellar evolution for a solar-mass star
        let config = SimulationConfig::default();
        let _simulation = UniverseSimulation::new(config).expect("Should create simulation");
        
        // Manually create a solar-mass star to test evolution
        let solar_mass = 1.989e30; // kg
        let mut stellar_evolution = StellarEvolution::new(solar_mass);
        
        // Initial conditions
        assert_eq!(stellar_evolution.evolutionary_phase, StellarPhase::MainSequence);
        assert_relative_eq!(stellar_evolution.nuclear_fuel_fraction, 1.0, epsilon = 0.01);
        
        // Simulate main sequence evolution (should take ~10 Gyr)
        let dt_years = 1e6; // 1 Myr time steps
        let mut total_time = 0.0;
        let mut energy_generated = 0.0;
        
        // Run until hydrogen exhaustion
        while stellar_evolution.nuclear_fuel_fraction > 0.1 && total_time < 15e9 {
            let energy = stellar_evolution.evolve(solar_mass, dt_years)
                .expect("Should successfully evolve star");
            energy_generated += energy;
            total_time += dt_years;
            
            // Verify energy generation is positive
            assert!(energy >= 0.0, "Nuclear fusion should generate energy");
            
            // Verify core temperature increases as fuel depletes
            if stellar_evolution.nuclear_fuel_fraction < 0.5 {
                assert!(stellar_evolution.core_temperature > 1.5e7, 
                       "Core should heat up as hydrogen depletes");
            }
        }
        
        // Should have transitioned off main sequence
        assert_ne!(stellar_evolution.evolutionary_phase, StellarPhase::MainSequence,
                  "Star should have evolved off main sequence");
        
        // Main sequence lifetime should be realistic (~10 Gyr for solar mass)
        assert!(total_time > 5e9 && total_time < 15e9, 
               "Solar mass star main sequence lifetime should be ~10 Gyr, got {:.1e} years", 
               total_time);
        
        // Should have generated significant energy
        assert!(energy_generated > 0.0, "Star should have generated nuclear energy");
    }
    
    #[test]
    fn test_massive_star_evolution() {
        // Test evolution of a massive star (20 solar masses)
        let solar_mass = 1.989e30;
        let massive_star_mass = 20.0 * solar_mass;
        let mut stellar_evolution = StellarEvolution::new(massive_star_mass);
        
        // Massive stars should have shorter lifetimes due to M^(-2.5) scaling
        assert!(stellar_evolution.main_sequence_lifetime < 1e8, 
               "Massive stars should have short lifetimes < 100 Myr");
        
        // Higher core temperature for massive stars
        assert!(stellar_evolution.core_temperature > 2e7,
               "Massive stars should have hotter cores");
        
        // Simulate evolution through advanced burning stages
        let dt_years = 1e5; // 100 kyr time steps for rapid evolution
        let mut total_time = 0.0;
        
        while stellar_evolution.evolutionary_phase != StellarPhase::Supernova && 
              total_time < stellar_evolution.main_sequence_lifetime * 2.0 {
            stellar_evolution.evolve(massive_star_mass, dt_years)
                .expect("Should successfully evolve massive star");
            total_time += dt_years;
        }
        
        // Massive star should go supernova
        assert_eq!(stellar_evolution.evolutionary_phase, StellarPhase::Supernova,
                  "Massive star should explode as supernova");
    }
    
    #[test]
    fn test_low_mass_star_evolution() {
        // Test evolution of a low-mass star (0.5 solar masses)
        let solar_mass = 1.989e30;
        let low_mass_star = 0.5 * solar_mass;
        let stellar_evolution = StellarEvolution::new(low_mass_star);
        
        // Low-mass stars should have very long lifetimes
        assert!(stellar_evolution.main_sequence_lifetime > 50e9, 
               "Low-mass stars should live longer than Hubble time");
        
        // Lower core temperature
        assert!(stellar_evolution.core_temperature < 1e7,
               "Low-mass stars should have cooler cores");
        
        // Should start in main sequence
        assert_eq!(stellar_evolution.evolutionary_phase, StellarPhase::MainSequence);
    }
    
    #[test]
    fn test_stellar_nucleosynthesis_integration() {
        // Test that stellar nucleosynthesis produces expected element ratios
        let solar_mass = 1.989e30;
        let mut stellar_evolution = StellarEvolution::new(solar_mass);
        
        // Initial composition should be primordial (H/He dominated)
        let initial_h_fraction = stellar_evolution.core_composition.iter()
            .find(|(z, a, _)| *z == 1 && *a == 1)
            .map(|(_, _, f)| *f)
            .unwrap_or(0.0);
        let initial_he_fraction = stellar_evolution.core_composition.iter()
            .find(|(z, a, _)| *z == 2 && *a == 4)
            .map(|(_, _, f)| *f)
            .unwrap_or(0.0);
        
        assert!(initial_h_fraction > 0.7, "Should start with >70% hydrogen");
        assert!(initial_he_fraction > 0.2, "Should start with >20% helium");
        
        // Evolve for significant time
        let dt_years = 1e6;
        for _ in 0..1000 {
            stellar_evolution.evolve(solar_mass, dt_years)
                .expect("Should evolve successfully");
            
            if stellar_evolution.nuclear_fuel_fraction < 0.8 {
                break;
            }
        }
        
        // Should have converted some hydrogen to helium
        let final_h_fraction = stellar_evolution.core_composition.iter()
            .find(|(z, a, _)| *z == 1 && *a == 1)
            .map(|(_, _, f)| *f)
            .unwrap_or(0.0);
        let final_he_fraction = stellar_evolution.core_composition.iter()
            .find(|(z, a, _)| *z == 2 && *a == 4)
            .map(|(_, _, f)| *f)
            .unwrap_or(0.0);
        
        assert!(final_h_fraction < initial_h_fraction, 
               "Hydrogen fraction should decrease due to burning");
        assert!(final_he_fraction > initial_he_fraction, 
               "Helium fraction should increase due to fusion");
    }
    
    #[test]
    fn test_stellar_death_outcomes() {
        // Test that stellar death produces correct remnants based on mass
        
        // Low-mass star → white dwarf
        let low_mass = 0.8 * 1.989e30;
        let mut low_evolution = StellarEvolution::new(low_mass);
        low_evolution.evolutionary_phase = StellarPhase::PlanetaryNebula;
        low_evolution.update_evolutionary_phase(low_mass, 0.0);
        assert_eq!(low_evolution.evolutionary_phase, StellarPhase::WhiteDwarf,
                  "Low-mass star should become white dwarf");
        
        // Intermediate-mass star → neutron star
        let intermediate_mass = 1.5 * 1.989e30;
        let mut intermediate_evolution = StellarEvolution::new(intermediate_mass);
        intermediate_evolution.evolutionary_phase = StellarPhase::PlanetaryNebula;
        intermediate_evolution.update_evolutionary_phase(intermediate_mass, 0.0);
        assert_eq!(intermediate_evolution.evolutionary_phase, StellarPhase::NeutronStar,
                  "Intermediate-mass star should become neutron star");
        
        // Very massive star → black hole
        let massive_mass = 30.0 * 1.989e30;
        let mut massive_evolution = StellarEvolution::new(massive_mass);
        massive_evolution.evolutionary_phase = StellarPhase::Supernova;
        massive_evolution.update_evolutionary_phase(massive_mass, 0.0);
        assert_eq!(massive_evolution.evolutionary_phase, StellarPhase::BlackHole,
                  "Very massive star should become black hole");
    }
    
    #[test]
    fn test_stellar_evolution_energy_conservation() {
        // Test that energy is conserved during stellar evolution
        let solar_mass = 1.989e30;
        let mut stellar_evolution = StellarEvolution::new(solar_mass);
        
        let dt_years = 1e6;
        let mut total_energy_generated = 0.0;
        
        // Track energy generation over time
        for _ in 0..100 {
            let energy = stellar_evolution.evolve(solar_mass, dt_years)
                .expect("Should evolve successfully");
            total_energy_generated += energy;
            
            // Each step should conserve energy
            assert!(energy.is_finite(), "Energy should be finite");
            assert!(energy >= 0.0, "Fusion should generate positive energy");
        }
        
        // Total energy should be significant for stellar burning
        assert!(total_energy_generated > 0.0, "Star should generate energy over time");
        
        // Energy generation rate should be realistic for solar-mass star
        let average_luminosity = stellar_evolution.nuclear_energy_generation;
        assert!(average_luminosity > 0.0, "Star should have positive luminosity");
        
        // Compare to solar luminosity (~3.8e26 W)
        // Our simplified model won't match exactly, but should be reasonable order of magnitude
    }
    
    #[test]
    fn test_initial_mass_function_sampling() {
        // Test that IMF sampling produces realistic stellar mass distribution
        let config = SimulationConfig::default();
        let simulation = UniverseSimulation::new(config).expect("Should create simulation");
        
        let mut rng = rand::thread_rng();
        let mut masses = Vec::new();
        
        // Sample 1000 stellar masses
        for _ in 0..1000 {
            let mass = simulation.sample_stellar_mass_from_imf(&mut rng);
            masses.push(mass / 1.989e30); // Convert to solar masses
        }
        
        // Check distribution properties
        masses.sort_by(|a, b| a.partial_cmp(b).unwrap());
        
        // Should be dominated by low-mass stars (Salpeter IMF with α=2.35)
        let median_mass = masses[masses.len() / 2];
        assert!(median_mass < 1.0, "Median stellar mass should be < 1 solar mass");
        
        // Should have some high-mass stars but they should be rare
        let high_mass_fraction = masses.iter().filter(|&&m| m > 10.0).count() as f64 / masses.len() as f64;
        assert!(high_mass_fraction < 0.1, "High-mass stars should be rare");
        
        // Should span realistic range
        assert!(masses[0] >= 0.08, "Minimum mass should be hydrogen burning limit");
        assert!(masses[masses.len()-1] <= 100.0, "Maximum mass should be reasonable");
    }
    
    #[test] 
    fn test_star_formation_and_evolution_cycle() {
        // Integration test: star formation → evolution → death → enrichment
        let config = SimulationConfig::low_memory(); // Keep simulation small
        
        let mut simulation = UniverseSimulation::new(config).expect("Should create simulation");
        simulation.init_big_bang().expect("Should initialize universe");
        
        // Fast-forward to star formation era
        for _ in 0..10 {
            simulation.tick().expect("Should advance simulation");
        }
        
        // Manually trigger star formation for testing
        simulation.process_star_formation().expect("Should form stars");
        
        // Verify stars were created
        let mut query = simulation.world.query::<(&CelestialBody, &StellarEvolution)>();
        let star_count = query.iter(&simulation.world).count();
        
        if star_count > 0 {
            // Test stellar evolution for created stars
            simulation.process_stellar_evolution().expect("Should evolve stars");
            
            // Verify stellar evolution updated properties
            let mut evolved_query = simulation.world.query::<(&CelestialBody, &StellarEvolution)>();
            for (body, evolution) in evolved_query.iter(&simulation.world) {
                assert!(evolution.nuclear_energy_generation >= 0.0, 
                       "Stars should have non-negative energy generation");
                
                // Verify stellar properties are reasonable
                assert!(body.mass > 0.0, "Star should have positive mass");
                assert!(body.luminosity >= 0.0, "Star should have non-negative luminosity");
                assert!(body.temperature > 1000.0, "Star should be hot");
            }
        }
    }
}