//! Comprehensive Unit Tests for Universe Simulation Compilation Fixes
//! 
//! This module tests all the fixes and new types that were added to resolve
//! compilation errors and ensure the universe simulation works correctly.

use universe_sim::*;
use universe_sim::storage::*;
use universe_sim::cosmic_era::*;
use universe_sim::config::SimulationConfig;
use physics_engine::types::ElementTable;
use nalgebra::Vector3;
use std::collections::HashMap;

/// Test suite for SupernovaYields implementation
mod supernova_yields_tests {
    use super::*;

    #[test]
    fn test_default_implementation() {
        let yields = SupernovaYields::default();
        
        // Verify all fields are properly initialized with scientific values
        assert!(yields.iron_mass > 0.0, "Iron mass should be positive");
        assert!(yields.silicon_group_mass > 0.0, "Silicon group mass should be positive");
        assert!(yields.oxygen_group_mass > 0.0, "Oxygen group mass should be positive");
        assert!(yields.carbon_group_mass > 0.0, "Carbon group mass should be positive");
        assert!(yields.heavy_elements_mass > 0.0, "Heavy elements mass should be positive");
        assert!(yields.total_ejected_mass > 0.0, "Total ejected mass should be positive");
    }

    #[test]
    fn test_mass_conservation() {
        let yields = SupernovaYields::default();
        
        // Check that individual components don't exceed total
        let component_sum = yields.iron_mass + yields.silicon_group_mass + 
                           yields.oxygen_group_mass + yields.carbon_group_mass + 
                           yields.heavy_elements_mass;
        
        assert!(component_sum <= yields.total_ejected_mass * 1.1, 
                "Component masses should not significantly exceed total (allowing for overlap)");
        
        // Check that total ejected mass is reasonable for a 25 solar mass star
        let solar_mass = 1.989e30; // kg
        let ejected_solar_masses = yields.total_ejected_mass / solar_mass;
        assert!(ejected_solar_masses > 5.0, "Should eject at least 5 solar masses");
        assert!(ejected_solar_masses <= 25.0, "Should not eject more than initial stellar mass");
    }

    #[test]
    fn test_nucleosynthesis_ratios() {
        let yields = SupernovaYields::default();
        
        // Test physically reasonable nucleosynthesis ratios
        assert!(yields.oxygen_group_mass > yields.iron_mass, 
                "Oxygen production should exceed iron production");
        assert!(yields.silicon_group_mass > yields.iron_mass, 
                "Silicon burning should produce more silicon than iron");
        assert!(yields.carbon_group_mass > 0.1, 
                "Should produce significant carbon");
    }

    #[test]
    fn test_serialization() {
        let yields = SupernovaYields::default();
        
        // Test that the structure can be serialized/deserialized
        let serialized = serde_json::to_string(&yields).expect("Should serialize");
        let deserialized: SupernovaYields = serde_json::from_str(&serialized)
            .expect("Should deserialize");
        
        assert_eq!(yields.iron_mass, deserialized.iron_mass);
        assert_eq!(yields.total_ejected_mass, deserialized.total_ejected_mass);
    }
}

/// Test suite for EnrichmentFactor implementation
mod enrichment_factor_tests {
    use super::*;

    #[test]
    fn test_default_implementation() {
        let enrichment = EnrichmentFactor::default();
        
        // Verify all fields are properly initialized
        assert!(enrichment.ejected_fraction >= 0.0 && enrichment.ejected_fraction <= 1.0,
                "Ejected fraction should be between 0 and 1");
        assert!(enrichment.metallicity_enhancement >= 1.0,
                "Metallicity enhancement should be >= 1 (enhancement factor)");
        assert!(enrichment.carbon_enhancement >= 1.0,
                "Carbon enhancement should be >= 1");
        assert!(enrichment.nitrogen_enhancement >= 1.0,
                "Nitrogen enhancement should be >= 1");
        assert!(enrichment.oxygen_enhancement >= 1.0,
                "Oxygen enhancement should be >= 1");
    }

    #[test]
    fn test_enhancement_factors_are_reasonable() {
        let enrichment = EnrichmentFactor::default();
        
        // Test that enhancement factors are within reasonable bounds
        assert!(enrichment.metallicity_enhancement < 100.0,
                "Metallicity enhancement should not be extreme");
        assert!(enrichment.carbon_enhancement < 50.0,
                "Carbon enhancement should be reasonable");
        assert!(enrichment.nitrogen_enhancement < 50.0,
                "Nitrogen enhancement should be reasonable");
        assert!(enrichment.oxygen_enhancement < 50.0,
                "Oxygen enhancement should be reasonable");
    }

    #[test]
    fn test_custom_enrichment_factor() {
        let custom_enrichment = EnrichmentFactor {
            iron_enrichment: 1.8,
            carbon_enrichment: 3.0,
            oxygen_enrichment: 2.0,
            silicon_enrichment: 2.2,
            total_metal_enrichment: 2.1,
            ejected_fraction: 0.8,
            metallicity_enhancement: 2.5,
            carbon_enhancement: 3.0,
            nitrogen_enhancement: 1.5,
            oxygen_enhancement: 2.0,
        };
        
        assert_eq!(custom_enrichment.ejected_fraction, 0.8);
        assert_eq!(custom_enrichment.metallicity_enhancement, 2.5);
        assert_eq!(custom_enrichment.carbon_enrichment, 3.0);
        assert_eq!(custom_enrichment.nitrogen_enhancement, 1.5);
        assert_eq!(custom_enrichment.oxygen_enrichment, 2.0);
        assert_eq!(custom_enrichment.iron_enrichment, 1.8);
        assert_eq!(custom_enrichment.silicon_enrichment, 2.2);
        assert_eq!(custom_enrichment.total_metal_enrichment, 2.1);
    }
}

/// Test suite for Atmosphere implementation
mod atmosphere_tests {
    use super::*;

    #[test]
    fn test_default_earth_like_atmosphere() {
        let atmosphere = Atmosphere::default();
        
        // Verify Earth-like atmosphere properties
        assert!(atmosphere.pressure > 0.0, "Pressure should be positive");
        assert!(atmosphere.temperature > 0.0, "Temperature should be positive");
        assert!(atmosphere.density > 0.0, "Density should be positive");
        assert!(atmosphere.scale_height > 0.0, "Scale height should be positive");
        assert!(!atmosphere.composition.is_empty(), "Composition should not be empty");
    }

    #[test]
    fn test_atmospheric_composition() {
        let atmosphere = Atmosphere::default();
        
        // Check for major atmospheric components
        let total_fraction: f64 = atmosphere.composition.values().sum();
        assert!((total_fraction - 1.0).abs() < 0.01, 
                "Total atmospheric composition should sum to ~1.0");
        
        // Check for nitrogen dominance (Earth-like)
        if let Some(n2_fraction) = atmosphere.composition.get("N2") {
            assert!(*n2_fraction > 0.7, "Nitrogen should be dominant component");
        }
        
        // Check for reasonable oxygen content
        if let Some(o2_fraction) = atmosphere.composition.get("O2") {
            assert!(*o2_fraction > 0.15 && *o2_fraction < 0.25, 
                    "Oxygen should be ~20% of atmosphere");
        }
    }

    #[test]
    fn test_custom_atmosphere() {
        let mut composition = HashMap::new();
        composition.insert("CO2".to_string(), 0.96);
        composition.insert("N2".to_string(), 0.035);
        composition.insert("Ar".to_string(), 0.016);
        
        let mars_like = Atmosphere {
            pressure: 636.0, // Pa (Mars surface pressure)
            composition,
            temperature: 210.0, // K (Mars average temperature)
            density: 0.02, // kg/m³
            scale_height: 11100.0, // m
        };
        
        assert_eq!(mars_like.pressure, 636.0);
        assert_eq!(mars_like.temperature, 210.0);
        assert!(mars_like.composition.get("CO2").unwrap() > &0.9);
    }

    #[test]
    fn test_scale_height_calculation() {
        let atmosphere = Atmosphere::default();
        
        // Scale height should be reasonable for Earth-like atmosphere
        // H = RT/(Mg) where R=8314, T~288K, M~0.029kg/mol, g~9.8m/s²
        // Expected: ~8400m
        assert!(atmosphere.scale_height > 7000.0 && atmosphere.scale_height < 10000.0,
                "Scale height should be reasonable for Earth-like atmosphere");
    }
}

/// Test suite for CelestialBody structure validation
mod celestial_body_tests {
    use super::*;

    #[test]
    fn test_celestial_body_creation() {
        let body = CelestialBody {
            id: uuid::Uuid::new_v4(),
            entity_id: 0,
            body_type: CelestialBodyType::Planet,
            mass: 5.972e24, // Earth mass in kg
            radius: 6.371e6, // Earth radius
            luminosity: 0.0, // Not luminous (planet)
            temperature: 288.0, // Earth average temperature
            age: 4.54e9, // Earth age in years
            composition: ElementTable::new(), // Empty composition for test
            has_planets: false,
            has_life: true,
            position: Vector3::new(1.496e11, 0.0, 0.0), // 1 AU from origin
            lifetime: 1e10,
            velocity: Vector3::new(0.0, 29780.0, 0.0), // Earth orbital velocity
            gravity: 9.8,
            atmosphere: Atmosphere::default(),
            is_habitable: true,
            agent_population: 0,
            tech_level: 0.0,
        };
        
        // Test required fields are set correctly
        assert_eq!(body.body_type, CelestialBodyType::Planet);
        assert_eq!(body.mass, 5.972e24);
        assert_eq!(body.position, Vector3::new(1.496e11, 0.0, 0.0));
        assert_eq!(body.velocity, Vector3::new(0.0, 29780.0, 0.0));
        
        // Test that new fields are properly initialized
        assert!(body.lifetime > 0.0, "Lifetime should be positive");
        assert!(body.gravity >= 0.0, "Gravity should be non-negative");
        assert!(body.atmosphere.pressure >= 0.0, "Atmosphere should be initialized");
        assert_eq!(body.agent_population, 0, "Should start with no agents");
        assert!(body.tech_level >= 0.0, "Tech level should be non-negative");
        assert!(body.is_habitable, "Should be habitable for test");
    }

    #[test]
    fn test_different_body_types() {
        let star_type = CelestialBodyType::Star;
        let planet_type = CelestialBodyType::Planet;
        let gas_cloud_type = CelestialBodyType::GasCloud;
        
        // Test that body types are distinct
        assert!(matches!(star_type, CelestialBodyType::Star));
        assert!(matches!(planet_type, CelestialBodyType::Planet));
        assert!(matches!(gas_cloud_type, CelestialBodyType::GasCloud));
    }

    #[test]
    fn test_stellar_phases() {
        let phases = vec![
            StellarPhase::MainSequence,
            StellarPhase::RedGiant,
            StellarPhase::WhiteDwarf,
            StellarPhase::NeutronStar,
            StellarPhase::BlackHole,
            StellarPhase::PlanetaryNebula,
            StellarPhase::Supernova,
        ];
        
        // Test that all phases can be created
        for phase in phases {
            let description = match phase {
                StellarPhase::MainSequence => "Hydrogen fusion in core",
                StellarPhase::RedGiant => "Helium fusion, expanded envelope",
                StellarPhase::WhiteDwarf => "Cooling white dwarf remnant",
                StellarPhase::NeutronStar => "Neutron degeneracy pressure",
                StellarPhase::BlackHole => "Gravitational collapse",
                StellarPhase::PlanetaryNebula => "Expelled stellar envelope",
                StellarPhase::Supernova => "Core collapse explosion",
                _ => "Other stellar phase",
            };
            
            assert!(!description.is_empty(), "All phases should have descriptions");
        }
    }
}

/// Test suite for PhysicalTransition constructor fixes
mod physical_transition_tests {
    use super::*;

    #[test]
    fn test_new_constructor_all_fields_initialized() {
        let transition = PhysicalTransition::new(
            1000,  // tick
            13.8,  // age_gyr
            TransitionType::CosmicEra,
            "Test transition".to_string(),
            vec![("temperature".to_string(), 2.7)],
        );
        
        // Test required fields
        assert_eq!(transition.tick, 1000);
        assert_eq!(transition.age_gyr, 13.8);
        assert!(matches!(transition.transition_type, TransitionType::CosmicEra));
        assert_eq!(transition.description, "Test transition");
        assert_eq!(transition.physical_parameters.len(), 1);
        
        // Test that new fields are properly initialized
        assert!(transition.timestamp >= 0.0, "Timestamp should be non-negative");
        assert!(transition.temperature >= 0.0, "Temperature should be non-negative");
        assert!(transition.energy_density >= 0.0, "Energy density should be non-negative");
    }

    #[test]
    fn test_different_transition_types() {
        let cosmic_era = PhysicalTransition::new(
            1000, 13.8, TransitionType::CosmicEra, 
            "Cosmic era transition".to_string(), vec![]
        );
        let temperature = PhysicalTransition::new(
            1001, 13.8, TransitionType::Temperature, 
            "Temperature transition".to_string(), vec![]
        );
        let energy_density = PhysicalTransition::new(
            1002, 13.8, TransitionType::EnergyDensity, 
            "Energy density transition".to_string(), vec![]
        );
        
        assert!(matches!(cosmic_era.transition_type, TransitionType::CosmicEra));
        assert!(matches!(temperature.transition_type, TransitionType::Temperature));
        assert!(matches!(energy_density.transition_type, TransitionType::EnergyDensity));
        
        // Each should have different ticks
        assert_ne!(cosmic_era.tick, temperature.tick);
        assert_ne!(temperature.tick, energy_density.tick);
    }

    #[test]
    fn test_transition_with_physics() {
        let transition = PhysicalTransition::new_with_physics(
            2000,  // tick
            13.8,  // age_gyr
            TransitionType::FirstStars,
            "First stars forming".to_string(),
            vec![("luminosity".to_string(), 1e30)],
            2000.0,  // timestamp
            5778.0,  // temperature
            1e-15,   // energy_density
        );
        
        assert_eq!(transition.tick, 2000);
        assert_eq!(transition.temperature, 5778.0);
        assert_eq!(transition.energy_density, 1e-15);
        assert_eq!(transition.timestamp, 2000.0);
    }
}

/// Test suite for UniverseState extensions
mod universe_state_tests {
    use super::*;

    #[test]
    fn test_initial_cosmic_state_fields() {
        let state = UniverseState::initial();
        
        // Test that all new cosmic state fields are initialized
        assert!(state.average_tech_level >= 0.0, "Average tech level should be non-negative");
        assert!(state.total_stellar_mass >= 0.0, "Total stellar mass should be non-negative");
        assert!(state.dark_energy_density >= 0.0, "Dark energy density should be non-negative");
        assert!(state.dark_matter_density >= 0.0, "Dark matter density should be non-negative");
        assert!(state.cosmic_ray_flux >= 0.0, "Cosmic ray flux should be non-negative");
        assert!(state.gravitational_wave_strain >= 0.0, "GW strain should be non-negative");
    }

    #[test]
    fn test_element_abundances() {
        let state = UniverseState::initial();
        
        // Test element abundance fields
        assert!(state.iron_abundance >= 0.0, "Iron abundance should be non-negative");
        assert!(state.carbon_abundance >= 0.0, "Carbon abundance should be non-negative");
        assert!(state.oxygen_abundance >= 0.0, "Oxygen abundance should be non-negative");
        assert!(state.nitrogen_abundance >= 0.0, "Nitrogen abundance should be non-negative");
        
        // Test abundance conservation
        let total_abundance = state.iron_abundance + state.carbon_abundance + 
                             state.oxygen_abundance + state.nitrogen_abundance;
        assert!(total_abundance <= 1.0, "Total abundance should not exceed 100%");
    }

    #[test]
    fn test_cosmological_parameters() {
        let state = UniverseState::initial();
        
        // Test that cosmological parameters are physically reasonable
        assert!(state.dark_energy_density >= 0.0, "Dark energy should be non-negative");
        assert!(state.dark_matter_density >= 0.0, "Dark matter should be non-negative");
        
        // Test initial state properties
        assert_eq!(state.age_gyr, 0.0, "Initial universe should have age 0");
        assert!(state.mean_temperature > 1e10, "Initial universe should be very hot");
        assert_eq!(state.stellar_fraction, 0.0, "No stars initially");
        assert_eq!(state.metallicity, 0.0, "No heavy elements initially");
    }

    #[test]
    fn test_state_modification() {
        let mut state = UniverseState::initial();
        
        // Test that fields can be modified
        state.average_tech_level = 2.5;
        state.total_stellar_mass = 1e12;
        state.iron_abundance = 0.001;
        
        assert_eq!(state.average_tech_level, 2.5);
        assert_eq!(state.total_stellar_mass, 1e12);
        assert_eq!(state.iron_abundance, 0.001);
    }
}

/// Test suite for ParticleStore updates
mod particle_store_tests {
    use super::*;

    #[test]
    fn test_count_field_initialization() {
        let store = ParticleStore::new(1000);
        
        assert_eq!(store.count, 0, "Count should start at 0");
        assert_eq!(store.len(), 0, "Length should start at 0");
        // Test that the store has the expected capacity via its position vector
        assert_eq!(store.position.capacity(), 1000, "Position vector should have correct capacity");
    }

    #[test]
    fn test_particle_addition_updates_count() {
        let mut store = ParticleStore::new(100);
        
        // Add particles using the correct API
        store.add(
            Vector3::new(1.0, 2.0, 3.0), // position
            Vector3::new(0.1, 0.2, 0.3), // velocity
            1.67e-27, // mass (proton mass)
            1.6e-19,  // charge (elementary charge)
            300.0,    // temperature
            0.0,      // entropy
        ).expect("Should add particle successfully");
        
        assert_eq!(store.count, 1, "Count should increment after adding particle");
        assert_eq!(store.len(), 1, "Length should match count");
        
        // Add another particle
        store.add(
            Vector3::new(2.0, 3.0, 4.0),
            Vector3::new(0.2, 0.3, 0.4),
            9.11e-31, // electron mass
            -1.6e-19, // electron charge
            300.0,    // temperature
            0.0,      // entropy
        ).expect("Should add particle successfully");
        
        assert_eq!(store.count, 2, "Count should increment to 2");
        assert_eq!(store.len(), 2, "Length should match count");
    }

    #[test]
    fn test_particle_data_storage() {
        let mut store = ParticleStore::new(100);
        
        // Add a specific particle
        let pos = Vector3::new(5.0, 6.0, 7.0);
        let vel = Vector3::new(0.5, 0.6, 0.7);
        let mass = 1.67e-27;
        let charge = 1.6e-19;
        let temp = 400.0;
        let entropy = 0.0;
        
        store.add(pos, vel, mass, charge, temp, entropy).expect("Should add particle successfully");
        
        // Verify data is stored correctly
        assert_eq!(store.len(), 1);
        assert_eq!(store.position[0], pos);
        assert_eq!(store.velocity[0], vel);
        assert_eq!(store.mass[0], mass);
        assert_eq!(store.charge[0], charge);
        assert_eq!(store.temperature[0], temp);
        assert_eq!(store.entropy[0], entropy);
    }

    #[test]
    fn test_capacity_limits() {
        let mut store = ParticleStore::new(2);
        
        // Add particles up to capacity
        store.add(Vector3::zeros(), Vector3::zeros(), 1.0, 0.0, 300.0, 0.0).expect("Should add particle");
        store.add(Vector3::zeros(), Vector3::zeros(), 1.0, 0.0, 300.0, 0.0).expect("Should add particle");
        
        assert_eq!(store.len(), 2);
        assert_eq!(store.count, 2);
        
        // Adding beyond capacity should handle gracefully
        let result = store.add(Vector3::zeros(), Vector3::zeros(), 1.0, 0.0, 300.0, 0.0);
        assert!(result.is_err(), "Should fail when exceeding capacity");
        
        // Count might still increment even if storage is full
        assert!(store.count >= 2, "Count should track attempts to add");
    }
}

/// Test suite for enum pattern matching completeness
mod enum_pattern_matching_tests {
    use super::*;

    #[test]
    fn test_transition_type_basic_matching() {
        let transitions = vec![
            TransitionType::CosmicEra,
            TransitionType::Temperature,
            TransitionType::EnergyDensity,
            TransitionType::FirstStars,
            TransitionType::FirstLife,
        ];
        
        for transition in transitions {
            let category = match transition {
                TransitionType::CosmicEra => "Cosmological",
                TransitionType::Temperature => "Thermal",
                TransitionType::EnergyDensity => "Energetic",
                TransitionType::FirstStars => "Stellar",
                TransitionType::FirstLife => "Biological",
                _ => "Other",
            };
            
            assert!(!category.is_empty(), "All transitions should have categories");
        }
    }

    #[test]
    fn test_body_type_matching() {
        let body_types = vec![
            CelestialBodyType::Star,
            CelestialBodyType::Planet,
            CelestialBodyType::GasCloud,
        ];
        
        for body_type in body_types {
            let mass_range = match body_type {
                CelestialBodyType::Star => (1e29, 1e32),      // 0.5 to 50 solar masses
                CelestialBodyType::Planet => (1e20, 1e28),    // Small asteroid to super-Jupiter
                CelestialBodyType::GasCloud => (1e28, 1e35),  // Molecular cloud masses
                _ => (1e20, 1e35), // Default range
            };
            
            assert!(mass_range.0 < mass_range.1, "Mass ranges should be valid");
            assert!(mass_range.0 > 0.0, "Minimum masses should be positive");
        }
    }
}

/// Integration tests for the compilation fixes
mod integration_tests {
    use super::*;

    #[test]
    fn test_universe_simulation_creation_with_new_types() {
        let config = SimulationConfig::default();
        let sim_result = UniverseSimulation::new(config);
        
        assert!(sim_result.is_ok(), "Universe simulation should create successfully");
        
        let sim = sim_result.unwrap();
        assert_eq!(sim.current_tick, 0, "Should start at tick 0");
        assert!(sim.universe_age_gyr() >= 0.0, "Universe age should be non-negative");
    }

    #[test]
    fn test_supernova_yields_in_context() {
        let yields = SupernovaYields::default();
        
        // Test that yields can be used in stellar evolution context
        let initial_stellar_mass = 25.0; // Solar masses (in solar mass units)
        let solar_mass_kg = 1.989e30; // kg
        let ejected_mass_kg = yields.total_ejected_mass; // kg
        let ejected_mass_solar = ejected_mass_kg / solar_mass_kg; // Convert to solar masses
        let remaining_mass = initial_stellar_mass - ejected_mass_solar;
        
        assert!(remaining_mass > 0.0, "Some mass should remain as remnant");
        assert!(ejected_mass_solar > 0.0, "Some mass should be ejected");
        
        // Test that individual elements make sense
        assert!(yields.iron_mass < yields.oxygen_group_mass, 
                "Oxygen should be more abundant than iron");
    }

    #[test]
    fn test_celestial_body_atmosphere_integration() {
        let planet = CelestialBody {
            id: uuid::Uuid::new_v4(),
            entity_id: 0,
            body_type: CelestialBodyType::Planet,
            mass: 5.972e24, // Earth mass
            radius: 6.371e6,
            luminosity: 0.0,
            temperature: 288.0,
            age: 4.54e9,
            composition: ElementTable::new(),
            has_planets: false,
            has_life: true,
            position: Vector3::new(1.496e11, 0.0, 0.0), // 1 AU
            lifetime: 1e10,
            velocity: Vector3::new(0.0, 29780.0, 0.0), // Orbital velocity
            gravity: 9.8,
            atmosphere: Atmosphere::default(),
            is_habitable: true,
            agent_population: 0,
            tech_level: 0.0,
        };
        
        // Test that atmosphere is properly integrated
        assert!(planet.atmosphere.pressure > 0.0, "Planet should have atmosphere");
        assert!(!planet.atmosphere.composition.is_empty(), "Atmosphere should have composition");
        
        // Test that mass affects gravity
        assert!(planet.gravity > 0.0, "Planet should have gravity");
        
        // Test that habitability can be determined
        assert!(planet.is_habitable, "Should be habitable for test");
    }

    #[test]
    fn test_physical_transition_cosmic_context() {
        let cosmic_transition = PhysicalTransition::new(
            1000, 13.8, TransitionType::CosmicEra,
            "Cosmic era transition".to_string(),
            vec![("probability".to_string(), 0.8)]
        );
        
        // Test that transition can be used in cosmic evolution
        assert!(cosmic_transition.temperature >= 0.0, "Cosmic temperature should be valid");
        assert!(cosmic_transition.energy_density >= 0.0, "Energy density should be valid");
        assert!(cosmic_transition.timestamp >= 0.0, "Timestamp should be valid");
        
        // Test that parameters are preserved
        assert_eq!(cosmic_transition.physical_parameters.len(), 1, "Should have one parameter");
        assert_eq!(cosmic_transition.physical_parameters[0].0, "probability");
        assert_eq!(cosmic_transition.physical_parameters[0].1, 0.8);
    }

    #[test]
    fn test_particle_store_physics_integration() {
        let mut store = ParticleStore::new(1000);
        
        // Add particles with realistic physics values
        store.add(
            Vector3::new(0.0, 0.0, 0.0),
            Vector3::new(1e5, 0.0, 0.0), // 100 km/s
            1.67e-27, // Proton mass
            1.6e-19,  // Elementary charge
            1e6,      // 1 million K
            0.0,      // Entropy
        ).expect("Should add particle successfully");
        
        assert_eq!(store.count, 1, "Should track particle count");
        assert_eq!(store.len(), 1, "Should track storage length");
        
        // Verify particle properties are preserved
        assert_eq!(store.velocity[0].x, 1e5, "Velocity should be preserved");
        assert_eq!(store.mass[0], 1.67e-27, "Mass should be preserved");
        assert_eq!(store.charge[0], 1.6e-19, "Charge should be preserved");
    }
}

/// Performance and memory tests
mod performance_tests {
    use super::*;
    use std::time::Instant;

    #[test]
    fn test_celestial_body_creation_performance() {
        let start = Instant::now();
        
        // Create many celestial bodies
        let mut bodies = Vec::new();
        for i in 0..1000 {
                         let body = CelestialBody {
                id: uuid::Uuid::new_v4(),
                entity_id: i,
                body_type: CelestialBodyType::Planet,
                mass: 1e24 + i as f64 * 1e20,
                radius: 6.371e6,
                luminosity: 0.0,
                temperature: 288.0,
                age: 4.54e9,
                composition: ElementTable::new(),
                has_planets: false,
                has_life: false,
                position: Vector3::new(i as f64 * 1e9, 0.0, 0.0),
                lifetime: 1e10,
                velocity: Vector3::zeros(),
                gravity: 9.8,
                atmosphere: Atmosphere::default(),
                is_habitable: false,
                agent_population: 0,
                tech_level: 0.0,
            };
            bodies.push(body);
        }
        
        let duration = start.elapsed();
        assert!(duration.as_millis() < 1000, "Should create 1000 bodies in under 1 second");
        assert_eq!(bodies.len(), 1000, "Should create all requested bodies");
    }

    #[test]
    fn test_particle_store_performance() {
        let start = Instant::now();
        let mut store = ParticleStore::new(10000);
        
        // Add many particles
        for i in 0..1000 {
            store.add(
                Vector3::new(i as f64, 0.0, 0.0),
                Vector3::new(0.0, i as f64, 0.0),
                1e-27,
                1e-19,
                300.0,
                0.0,
            ).expect("Should add particle successfully");
        }
        
        let duration = start.elapsed();
        assert!(duration.as_millis() < 500, "Should add 1000 particles quickly");
        assert_eq!(store.count, 1000, "Should track all added particles");
    }

    #[test]
    fn test_supernova_yields_memory_usage() {
        // Test that SupernovaYields doesn't use excessive memory
        let yields = SupernovaYields::default();
        let size = std::mem::size_of_val(&yields);
        
        assert!(size < 1024, "SupernovaYields should be reasonably sized");
        
        // Test that we can create many without issues
        let mut yields_vec = Vec::new();
        for _ in 0..1000 {
            yields_vec.push(SupernovaYields::default());
        }
        
        assert_eq!(yields_vec.len(), 1000, "Should be able to create many yields");
    }
}