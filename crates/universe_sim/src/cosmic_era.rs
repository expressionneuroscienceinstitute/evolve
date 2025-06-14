//! Cosmic Era Management
//! 
//! Defines the different epochs of universe evolution and manages transitions

use serde::{Serialize, Deserialize};

/// Cosmic eras with different gameplay mechanics
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum CosmicEra {
    /// 0-0.3 Gyr: Particle Soup (observe only)
    ParticleSoup,
    
    /// 0.3-1 Gyr: Starbirth (set star-forming nebulae parameters)
    Starbirth,
    
    /// 1-5 Gyr: Planetary Age (influence protoplanetary disks)
    PlanetaryAge,
    
    /// 5-10 Gyr: Biogenesis (seed organic molecules, tweak atmospheres)
    Biogenesis,
    
    /// 10-13+ Gyr: Digital Evolution (guide agent mutations, infrastructure)
    DigitalEvolution,
    
    /// Post-Intelligence (direct R&D funding, megastructures)
    PostIntelligence,
}

impl CosmicEra {
    /// Get the age range for this era in billion years
    pub fn age_range_gyr(&self) -> (f64, f64) {
        match self {
            CosmicEra::ParticleSoup => (0.0, 0.0003),
            CosmicEra::Starbirth => (0.0003, 1.0),
            CosmicEra::PlanetaryAge => (1.0, 5.0),
            CosmicEra::Biogenesis => (5.0, 10.0),
            CosmicEra::DigitalEvolution => (10.0, 13.0),
            CosmicEra::PostIntelligence => (13.0, f64::INFINITY),
        }
    }

    /// Get the key resources for this era
    pub fn key_resources(&self) -> Vec<&'static str> {
        match self {
            CosmicEra::ParticleSoup => vec!["Photon density"],
            CosmicEra::Starbirth => vec!["Gas clouds"],
            CosmicEra::PlanetaryAge => vec!["Heavy elements"],
            CosmicEra::Biogenesis => vec!["Carbon", "Water"],
            CosmicEra::DigitalEvolution => vec!["Compute capacity", "Energy"],
            CosmicEra::PostIntelligence => vec!["Stellar energy", "Metals"],
        }
    }

    /// Get the primary failure risks for this era
    pub fn fail_risks(&self) -> Vec<&'static str> {
        match self {
            CosmicEra::ParticleSoup => vec!["None"],
            CosmicEra::Starbirth => vec!["Under-seeding → sterile universe"],
            CosmicEra::PlanetaryAge => vec!["Too few habitable planets"],
            CosmicEra::Biogenesis => vec!["Runaway greenhouse", "Frozen planets"],
            CosmicEra::DigitalEvolution => vec!["Data corruption", "Hardware loss"],
            CosmicEra::PostIntelligence => vec!["Societal collapse", "AI rebellion"],
        }
    }

    /// Get unlockable actions for this era
    pub fn unlockable_actions(&self) -> Vec<&'static str> {
        match self {
            CosmicEra::ParticleSoup => vec!["*None* (observe only)"],
            CosmicEra::Starbirth => vec![
                "Set parameters for star-forming nebulae (density, metallicity)"
            ],
            CosmicEra::PlanetaryAge => vec![
                "Influence protoplanetary disks (planet count, distance, water fraction)"
            ],
            CosmicEra::Biogenesis => vec![
                "Seed basic organic molecules",
                "Tweak atmospheric composition"
            ],
            CosmicEra::DigitalEvolution => vec![
                "Guide agent mutation rates", 
                "Infrastructure shielding",
                "Network topology"
            ],
            CosmicEra::PostIntelligence => vec![
                "Direct R&D funding",
                "Megastructure builds (Dyson swarms, jump drives)"
            ],
        }
    }

    /// Check if this era allows player intervention
    pub fn allows_intervention(&self) -> bool {
        !matches!(self, CosmicEra::ParticleSoup)
    }

    /// Get the next era
    pub fn next(&self) -> Option<CosmicEra> {
        match self {
            CosmicEra::ParticleSoup => Some(CosmicEra::Starbirth),
            CosmicEra::Starbirth => Some(CosmicEra::PlanetaryAge),
            CosmicEra::PlanetaryAge => Some(CosmicEra::Biogenesis),
            CosmicEra::Biogenesis => Some(CosmicEra::DigitalEvolution),
            CosmicEra::DigitalEvolution => Some(CosmicEra::PostIntelligence),
            CosmicEra::PostIntelligence => None, // Infinite era
        }
    }

    /// Get era description
    pub fn description(&self) -> &'static str {
        match self {
            CosmicEra::ParticleSoup => 
                "The universe is a hot, dense soup of fundamental particles. Matter and antimatter annihilate, leaving behind the first stable atoms.",
            CosmicEra::Starbirth => 
                "Gas clouds begin to collapse under gravity. The first stars ignite, flooding the universe with light and heavy elements.",
            CosmicEra::PlanetaryAge => 
                "Around young stars, protoplanetary disks form rocky worlds. The first solid surfaces appear in the cosmos.",
            CosmicEra::Biogenesis => 
                "On worlds with suitable conditions, the first organic molecules begin to replicate and evolve toward complexity.",
            CosmicEra::DigitalEvolution => 
                "Life develops intelligence and begins to modify its own nature through technology and digital substrates.",
            CosmicEra::PostIntelligence => 
                "Transcendent intelligences reshape matter and energy on cosmic scales, building megastructures that span star systems.",
        }
    }

    /// Get era duration in billion years
    pub fn duration_gyr(&self) -> f64 {
        let (start, end) = self.age_range_gyr();
        if end == f64::INFINITY {
            f64::INFINITY
        } else {
            end - start
        }
    }

    /// Check if age is within this era
    pub fn contains_age(&self, age_gyr: f64) -> bool {
        let (start, end) = self.age_range_gyr();
        age_gyr >= start && (age_gyr < end || end == f64::INFINITY)
    }

    /// Get era from universe age
    pub fn from_age_gyr(age_gyr: f64) -> Self {
        if age_gyr < 0.0003 {
            CosmicEra::ParticleSoup
        } else if age_gyr < 1.0 {
            CosmicEra::Starbirth
        } else if age_gyr < 5.0 {
            CosmicEra::PlanetaryAge
        } else if age_gyr < 10.0 {
            CosmicEra::Biogenesis
        } else if age_gyr < 13.0 {
            CosmicEra::DigitalEvolution
        } else {
            CosmicEra::PostIntelligence
        }
    }
}

/// Era transition event
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EraTransition {
    pub from_era: CosmicEra,
    pub to_era: CosmicEra,
    pub universe_age_gyr: f64,
    pub tick: u64,
    pub major_events: Vec<String>,
}

impl EraTransition {
    pub fn new(from: CosmicEra, to: CosmicEra, age_gyr: f64, tick: u64) -> Self {
        let major_events = match to {
            CosmicEra::Starbirth => vec![
                "First hydrogen clouds begin collapsing".to_string(),
                "Dark matter halos provide gravitational scaffolding".to_string(),
            ],
            CosmicEra::PlanetaryAge => vec![
                "First generation stars have forged heavy elements".to_string(),
                "Stellar winds disperse metals into space".to_string(),
                "Second generation stars ignite with rocky cores".to_string(),
            ],
            CosmicEra::Biogenesis => vec![
                "Terrestrial planets have cooled and formed atmospheres".to_string(),
                "Liquid water appears on suitable worlds".to_string(),
                "Complex organic chemistry begins".to_string(),
            ],
            CosmicEra::DigitalEvolution => vec![
                "Self-replicating organisms achieve intelligence".to_string(),
                "Technology enables modification of biological processes".to_string(),
                "Digital substrates supplement biological computation".to_string(),
            ],
            CosmicEra::PostIntelligence => vec![
                "Artificial general intelligence achieves superintelligence".to_string(),
                "Matter and energy can be manipulated at the molecular level".to_string(),
                "Megascale engineering projects become feasible".to_string(),
            ],
            _ => vec![],
        };

        Self {
            from_era: from,
            to_era: to,
            universe_age_gyr: age_gyr,
            tick,
            major_events,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_era_age_ranges() {
        assert!(CosmicEra::ParticleSoup.contains_age(0.0001));
        assert!(!CosmicEra::ParticleSoup.contains_age(0.5));
        
        assert!(CosmicEra::Starbirth.contains_age(0.5));
        assert!(!CosmicEra::Starbirth.contains_age(2.0));
        
        assert!(CosmicEra::PostIntelligence.contains_age(100.0)); // Infinite era
    }

    #[test]
    fn test_era_from_age() {
        assert_eq!(CosmicEra::from_age_gyr(0.0001), CosmicEra::ParticleSoup);
        assert_eq!(CosmicEra::from_age_gyr(0.5), CosmicEra::Starbirth);
        assert_eq!(CosmicEra::from_age_gyr(3.0), CosmicEra::PlanetaryAge);
        assert_eq!(CosmicEra::from_age_gyr(7.0), CosmicEra::Biogenesis);
        assert_eq!(CosmicEra::from_age_gyr(12.0), CosmicEra::DigitalEvolution);
        assert_eq!(CosmicEra::from_age_gyr(20.0), CosmicEra::PostIntelligence);
    }

    #[test]
    fn test_era_transitions() {
        let transition = EraTransition::new(
            CosmicEra::ParticleSoup, 
            CosmicEra::Starbirth, 
            0.0003, 
            300
        );
        
        assert_eq!(transition.from_era, CosmicEra::ParticleSoup);
        assert_eq!(transition.to_era, CosmicEra::Starbirth);
        assert!(!transition.major_events.is_empty());
    }

    #[test]
    fn test_era_properties() {
        assert!(!CosmicEra::ParticleSoup.allows_intervention());
        assert!(CosmicEra::Starbirth.allows_intervention());
        
        assert_eq!(CosmicEra::PlanetaryAge.next(), Some(CosmicEra::Biogenesis));
        assert_eq!(CosmicEra::PostIntelligence.next(), None);
    }
}