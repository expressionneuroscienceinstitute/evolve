//! Technology tree for agent research and progression

use crate::types::*;
use crate::{Result, SimError};
use std::collections::{HashMap, HashSet};
use serde::{Deserialize, Serialize};

/// Technology node in the research tree
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Technology {
    pub id: TechId,
    pub name: String,
    pub description: String,
    pub prerequisites: Vec<TechId>,
    pub cost: ElementTable,
    pub tool_level: ToolLevel,
    pub unlocks_actions: Vec<String>,
    pub energy_cost: f64,
}

/// Technology tree management
pub struct TechTree {
    technologies: HashMap<TechId, Technology>,
    lineage_progress: HashMap<LineageId, HashSet<TechId>>,
}

impl TechTree {
    pub fn new() -> Self {
        let mut tree = Self {
            technologies: HashMap::new(),
            lineage_progress: HashMap::new(),
        };
        
        tree.initialize_default_techs();
        tree
    }
    
    /// Initialize default technology tree
    fn initialize_default_techs(&mut self) {
        // Stone Age technologies
        self.add_tech(Technology {
            id: TechId::new(1),
            name: "Stone Tools".to_string(),
            description: "Basic stone implements for survival".to_string(),
            prerequisites: vec![],
            cost: ElementTable::new(),
            tool_level: ToolLevel::Stone,
            unlocks_actions: vec!["dig".to_string(), "craft_stone_tools".to_string()],
            energy_cost: 10.0,
        });
        
        // Bronze Age
        let mut bronze_cost = ElementTable::new();
        bronze_cost.set_abundance(crate::constants::elements::CU, 1000);
        bronze_cost.set_abundance(crate::constants::elements::S, 100);
        
        self.add_tech(Technology {
            id: TechId::new(2),
            name: "Bronze Working".to_string(),
            description: "Alloy copper and tin to create bronze tools".to_string(),
            prerequisites: vec![TechId::new(1)],
            cost: bronze_cost,
            tool_level: ToolLevel::Bronze,
            unlocks_actions: vec!["smelt_bronze".to_string(), "craft_bronze_tools".to_string()],
            energy_cost: 100.0,
        });
        
        // Iron Age
        let mut iron_cost = ElementTable::new();
        iron_cost.set_abundance(crate::constants::elements::FE, 5000);
        iron_cost.set_abundance(crate::constants::elements::C, 500);
        
        self.add_tech(Technology {
            id: TechId::new(3),
            name: "Iron Working".to_string(),
            description: "Smelt iron ore and create steel alloys".to_string(),
            prerequisites: vec![TechId::new(2)],
            cost: iron_cost,
            tool_level: ToolLevel::Iron,
            unlocks_actions: vec!["smelt_iron".to_string(), "craft_steel".to_string()],
            energy_cost: 500.0,
        });
        
        // Industrial technologies
        let mut steam_cost = ElementTable::new();
        steam_cost.set_abundance(crate::constants::elements::FE, 10000);
        steam_cost.set_abundance(crate::constants::elements::C, 2000);
        
        self.add_tech(Technology {
            id: TechId::new(4),
            name: "Steam Power".to_string(),
            description: "Harness steam for mechanical power".to_string(),
            prerequisites: vec![TechId::new(3)],
            cost: steam_cost,
            tool_level: ToolLevel::Steel,
            unlocks_actions: vec!["build_steam_engine".to_string(), "industrialize".to_string()],
            energy_cost: 10000.0,
        });
        
        // Nuclear technologies
        let mut nuclear_cost = ElementTable::new();
        nuclear_cost.set_abundance(crate::constants::elements::U, 100);
        nuclear_cost.set_abundance(crate::constants::elements::FE, 50000);
        
        self.add_tech(Technology {
            id: TechId::new(5),
            name: "Nuclear Fission".to_string(),
            description: "Split atomic nuclei for massive energy release".to_string(),
            prerequisites: vec![TechId::new(4)],
            cost: nuclear_cost,
            tool_level: ToolLevel::Composite,
            unlocks_actions: vec!["build_reactor".to_string(), "nuclear_power".to_string()],
            energy_cost: 1000000.0,
        });
        
        // Fusion technologies
        let mut fusion_cost = ElementTable::new();
        fusion_cost.set_abundance(crate::constants::elements::LI, 1000);
        fusion_cost.set_abundance(crate::constants::elements::H, 10000);
        
        self.add_tech(Technology {
            id: TechId::new(6),
            name: "Nuclear Fusion".to_string(),
            description: "Fuse light nuclei for clean energy".to_string(),
            prerequisites: vec![TechId::new(5)],
            cost: fusion_cost,
            tool_level: ToolLevel::Exotic,
            unlocks_actions: vec!["build_fusion_reactor".to_string(), "stellar_engineering".to_string()],
            energy_cost: 100000000.0,
        });
        
        // Dyson Swarm (Type II civilization)
        let mut dyson_cost = ElementTable::new();
        dyson_cost.set_abundance(crate::constants::elements::FE, 1000000);
        dyson_cost.set_abundance(crate::constants::elements::SI, 500000);
        
        self.add_tech(Technology {
            id: TechId::new(7),
            name: "Dyson Swarm".to_string(),
            description: "Surround star with energy collectors".to_string(),
            prerequisites: vec![TechId::new(6)],
            cost: dyson_cost,
            tool_level: ToolLevel::Exotic,
            unlocks_actions: vec!["build_dyson_swarm".to_string(), "stellar_harvest".to_string()],
            energy_cost: 1e12,
        });
        
        // Warp Drive (interstellar travel)
        let mut warp_cost = ElementTable::new();
        warp_cost.set_abundance(crate::constants::elements::H, 100000); // Exotic matter
        
        self.add_tech(Technology {
            id: TechId::new(8),
            name: "Warp Drive".to_string(),
            description: "Faster-than-light travel via spacetime manipulation".to_string(),
            prerequisites: vec![TechId::new(7)],
            cost: warp_cost,
            tool_level: ToolLevel::Exotic,
            unlocks_actions: vec!["interstellar_travel".to_string(), "galactic_expansion".to_string()],
            energy_cost: 1e15,
        });
    }
    
    /// Add a technology to the tree
    pub fn add_tech(&mut self, tech: Technology) {
        self.technologies.insert(tech.id, tech);
    }
    
    /// Check if lineage can research a technology
    pub fn can_research(&self, lineage_id: LineageId, tech_id: TechId) -> bool {
        let Some(tech) = self.technologies.get(&tech_id) else {
            return false;
        };
        
        let researched = self.lineage_progress.get(&lineage_id)
            .map(|s| s.clone())
            .unwrap_or_default();
        
        // Check if already researched
        if researched.contains(&tech_id) {
            return false;
        }
        
        // Check prerequisites
        tech.prerequisites.iter().all(|&prereq| researched.contains(&prereq))
    }
    
    /// Research a technology for a lineage
    pub fn research_tech(&mut self, lineage_id: LineageId, tech_id: TechId) -> Result<()> {
        if !self.can_research(lineage_id, tech_id) {
            return Err(SimError::AgentError(
                format!("Cannot research tech {} for lineage {}", tech_id, lineage_id)
            ));
        }
        
        self.lineage_progress.entry(lineage_id)
            .or_insert_with(HashSet::new)
            .insert(tech_id);
        
        Ok(())
    }
    
    /// Get available technologies for a lineage
    pub fn get_available_techs(&self, lineage_id: LineageId) -> Vec<TechId> {
        let mut available = Vec::new();
        
        for (&tech_id, _) in &self.technologies {
            if self.can_research(lineage_id, tech_id) {
                available.push(tech_id);
            }
        }
        
        available
    }
    
    /// Get researched technologies for a lineage
    pub fn get_researched_techs(&self, lineage_id: LineageId) -> HashSet<TechId> {
        self.lineage_progress.get(&lineage_id)
            .cloned()
            .unwrap_or_default()
    }
    
    /// Get technology details
    pub fn get_tech(&self, tech_id: TechId) -> Option<&Technology> {
        self.technologies.get(&tech_id)
    }
    
    /// Check if lineage has reached a milestone
    pub fn check_milestone(&self, lineage_id: LineageId) -> TechMilestone {
        let researched = self.get_researched_techs(lineage_id);
        
        if researched.contains(&TechId::new(8)) {
            TechMilestone::Immortality // Warp drive = galactic civilization
        } else if researched.contains(&TechId::new(7)) {
            TechMilestone::TransTechFrontier // Dyson swarm
        } else if researched.contains(&TechId::new(6)) {
            TechMilestone::Digitalization // Fusion power
        } else if researched.contains(&TechId::new(4)) {
            TechMilestone::Industrialization // Steam power
        } else if researched.contains(&TechId::new(1)) {
            TechMilestone::Sentience // Basic tool use
        } else {
            TechMilestone::None
        }
    }
}

/// Technology milestones for win conditions
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub enum TechMilestone {
    None,
    Sentience,
    Industrialization,
    Digitalization,
    TransTechFrontier,
    Immortality,
}