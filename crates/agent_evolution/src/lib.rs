//! Agent Evolution System
//! 
//! Comprehensive autonomous agent evolution with full decision tracking,
//! self-modification capabilities, and natural selection pressures.
//! Designed to be the most advanced AI evolution simulation ever created.

use bevy_ecs::prelude::*;
use serde::{Serialize, Deserialize};
use anyhow::Result;
use uuid::Uuid;
use std::collections::{HashMap, VecDeque};
use rand::{Rng, thread_rng};
use physics_engine::EnvironmentProfile;

pub mod ai_core;
pub mod decision_tracking;
pub mod genetics;
pub mod natural_selection;
pub mod lineage_analytics;
pub mod consciousness;
pub mod self_modification;

use ai_core::{AICore, SensoryInput, ActionType, AgentSensoryData};

/// Main AI agent component with full autonomous decision-making
#[derive(Debug, Clone, Component, Serialize, Deserialize)]
pub struct AutonomousAgent {
    pub id: Uuid,
    pub generation: u32,
    pub birth_tick: u64,
    pub last_decision_tick: u64,
    pub energy: f64,
    pub matter_consumed: f64,
    pub reproduction_count: u32,
    
    // Evolutionary levels (0.0 to 1.0)
    pub sentience_level: f64,
    pub industrialization_level: f64,
    pub digitalization_level: f64,
    pub tech_level: f64,
    pub immortality_achieved: bool,
    
    // AI Core - The brain of the agent
    pub ai_core: AICore,
    
    // Physical properties
    pub position: [f64; 3],
    pub velocity: [f64; 3],
    pub size: f64,
    pub mass: f64,
    
    // Resource management
    pub stored_energy: f64,
    pub stored_matter: f64,
    pub resource_efficiency: f64,
    
    // Social properties
    pub cooperation_tendency: f64,
    pub aggression_level: f64,
    pub communication_range: f64,
    
    // Environmental adaptation
    pub environmental_preferences: EnvironmentProfile,
    pub resource_requirements: ResourceRequirements,
    pub social_behaviors: SocialBehaviors,
    
    // Evolution tracking
    pub fitness_history: Vec<f64>,
    pub major_innovations: Vec<Innovation>,
    pub parent_ids: Vec<Uuid>,
    pub child_ids: Vec<Uuid>,
}

/// Comprehensive decision tracking system
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Decision {
    pub id: Uuid,
    pub timestamp: u64,
    pub decision_type: DecisionType,
    pub context: DecisionContext,
    pub outcome: DecisionOutcome,
    pub energy_cost: f64,
    pub success_probability: f64,
    pub actual_success: bool,
    pub learning_feedback: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum DecisionType {
    // Survival decisions
    EnergyAcquisition(EnergySource),
    ResourceExtraction(ResourceType),
    ThreatAvoidance(ThreatType),
    ShelterSeeking(ShelterType),
    
    // Reproduction decisions
    MateSelection(MateSelectionCriteria),
    ReproductionAttempt(ReproductionStrategy),
    OffspringCare(CareStrategy),
    
    // Social decisions
    Cooperation(CooperationStrategy),
    Competition(CompetitionStrategy),
    Communication(CommunicationStrategy),
    GroupFormation(GroupStrategy),
    
    // Technological decisions
    ToolCreation(ToolType),
    ToolUsage(ToolApplication),
    KnowledgeSharing(KnowledgeType),
    Innovation(InnovationType),
    
    // Self-modification decisions
    CodeMutation(MutationType),
    LearningAdjustment(LearningType),
    MemoryOptimization(MemoryType),
    ArchitectureChange(ArchitectureType),
    
    // Long-term strategic decisions
    MigrationDecision(MigrationStrategy),
    SpecializationChoice(SpecializationType),
    TechnologyPath(TechPathType),
    EvolutionaryGoal(GoalType),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DecisionContext {
    pub environmental_state: EnvironmentProfile,
    pub available_resources: HashMap<ResourceType, f64>,
    pub nearby_agents: Vec<AgentInteraction>,
    pub threats_present: Vec<ThreatType>,
    pub opportunities: Vec<OpportunityType>,
    pub historical_success_rate: f64,
    pub genetic_predisposition: f64,
    pub social_pressure: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DecisionOutcome {
    pub immediate_effects: Vec<ImmediateEffect>,
    pub long_term_consequences: Vec<LongTermConsequence>,
    pub fitness_impact: f64,
    pub survival_probability_change: f64,
    pub reproduction_probability_change: f64,
    pub innovation_progress: f64,
}

/// Advanced agent memory system
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentMemory {
    pub short_term: HashMap<String, MemoryItem>,
    pub long_term: HashMap<String, MemoryItem>,
    pub procedural: HashMap<String, Skill>,
    pub episodic: VecDeque<Episode>,
    pub semantic: HashMap<String, Concept>,
    pub working_memory_capacity: usize,
    pub consolidation_threshold: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryItem {
    pub content: serde_json::Value,
    pub strength: f64,
    pub last_accessed: u64,
    pub importance: f64,
    pub emotional_weight: f64,
}

/// Natural selection pressure tracking
#[derive(Debug, Clone, Component, Serialize, Deserialize)]
pub struct SelectionPressures {
    pub environmental_pressures: Vec<EnvironmentalPressure>,
    pub resource_competition: f64,
    pub predation_pressure: f64,
    pub sexual_selection: f64,
    pub technological_arms_race: f64,
    pub social_conformity_pressure: f64,
    pub innovation_pressure: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EnvironmentalPressure {
    pub pressure_type: PressureType,
    pub intensity: f64,
    pub duration: u64,
    pub affected_traits: Vec<String>,
    pub selection_coefficient: f64,
}

/// Comprehensive lineage tracking
#[derive(Debug, Clone, Resource, Serialize, Deserialize)]
pub struct LineageTracker {
    pub lineages: HashMap<Uuid, LineageData>,
    pub family_trees: HashMap<Uuid, FamilyTree>,
    pub evolutionary_branches: Vec<EvolutionaryBranch>,
    pub extinction_events: Vec<ExtinctionEvent>,
    pub speciation_events: Vec<SpeciationEvent>,
    pub innovation_timeline: Vec<InnovationEvent>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LineageData {
    pub lineage_id: Uuid,
    pub founder_id: Uuid,
    pub current_generation: u32,
    pub total_individuals: u64,
    pub living_individuals: u32,
    pub average_fitness: f64,
    pub genetic_diversity: f64,
    pub technological_level: f64,
    pub major_innovations: Vec<Innovation>,
    pub adaptation_events: Vec<AdaptationEvent>,
    pub selection_history: Vec<SelectionEvent>,
}

/// Advanced evolution engine with real autonomous decision-making
pub struct EvolutionEngine {
    pub mutation_engine: MutationEngine,
    pub selection_engine: SelectionEngine,
    pub innovation_engine: InnovationEngine,
    pub consciousness_tracker: ConsciousnessTracker,
    pub decision_analyzer: DecisionAnalyzer,
    pub lineage_tracker: LineageTracker,
}

impl EvolutionEngine {
    pub fn new() -> Self {
        Self {
            mutation_engine: MutationEngine::new(),
            selection_engine: SelectionEngine::new(),
            innovation_engine: InnovationEngine::new(),
            consciousness_tracker: ConsciousnessTracker::new(),
            decision_analyzer: DecisionAnalyzer::new(),
            lineage_tracker: LineageTracker::new(),
        }
    }
    
    /// Process agent decisions with full autonomous AI decision-making
    pub fn process_agent_decisions(
        &mut self,
        agents: &mut Query<&mut AutonomousAgent>,
        world_state: &dyn WorldState,
        current_tick: u64,
    ) -> Result<Vec<DecisionReport>> {
        let mut reports = Vec::new();
        
        for mut agent in agents.iter_mut() {
            // Agent makes autonomous decision
            let action = agent.make_decision(world_state, current_tick)?;
            
            // Execute the action in the world
            // Note: In a real implementation, world_state would need to be mutable
            // For now, we'll simulate the outcome
            let outcome = self.simulate_action_outcome(&agent, &action)?;
            
            // Agent learns from the outcome
            agent.ai_core.learn_from_outcome(
                action.action_type,
                outcome.fitness_change,
                vec![agent.energy, agent.sentience_level, agent.tech_level],
                current_tick,
            );
            
            let report = DecisionReport {
                agent_id: agent.id,
                decision: Decision {
                    id: Uuid::new_v4(),
                    timestamp: current_tick,
                    decision_type: action.action_type.into(),
                    context: DecisionContext::from_agent(&agent, world_state),
                    outcome: DecisionOutcome::from_action_outcome(&outcome),
                    energy_cost: action.action_type.energy_cost(),
                    success_probability: agent.ai_core.decision_confidence,
                    actual_success: outcome.success,
                    learning_feedback: outcome.learning_value,
                },
                outcome: DecisionOutcome::from_action_outcome(&outcome),
                learning_update: LearningUpdate::default(),
                fitness_change: outcome.fitness_change,
            };
            
            reports.push(report);
        }
        
        Ok(reports)
    }
    
    /// Simulate action outcome for agents (simplified version)
    fn simulate_action_outcome(
        &self,
        agent: &AutonomousAgent,
        action: &AgentAction,
    ) -> Result<ActionOutcome> {
        let mut rng = thread_rng();
        
        // Base success probability based on agent capabilities
        let base_success = match action.action_type {
            ActionType::ExtractEnergy => 0.7 + agent.resource_efficiency * 0.2,
            ActionType::Learn => 0.8 + agent.sentience_level * 0.15,
            ActionType::CreateTool => 0.3 + agent.tech_level * 0.4,
            ActionType::Cooperate => 0.6 + agent.cooperation_tendency * 0.3,
            ActionType::Rest => 1.0,
            _ => 0.5,
        };
        
        let success = rng.gen::<f64>() < base_success;
        
        let fitness_change = if success {
            rng.gen_range(0.05..0.3)
        } else {
            rng.gen_range(-0.15..-0.02)
        };
        
        Ok(ActionOutcome {
            success,
            fitness_change,
            energy_change: if success { rng.gen_range(0.0..10.0) } else { rng.gen_range(-5.0..0.0) },
            innovation_progress: if matches!(action.action_type, ActionType::CreateTool | ActionType::Experiment) {
                rng.gen_range(0.0..0.2)
            } else {
                0.0
            },
            social_impact: if matches!(action.action_type, ActionType::Communicate | ActionType::Cooperate) {
                rng.gen_range(0.0..0.3)
            } else {
                0.0
            },
            learning_value: rng.gen_range(0.05..0.25),
        })
    }
    
    /// Comprehensive evolution step with autonomous agents
    pub fn evolution_step(
        &mut self,
        agents: &mut Query<&mut AutonomousAgent>,
        selection_pressures: &Query<&SelectionPressures>,
        world_state: &dyn WorldState,
        current_tick: u64,
    ) -> Result<EvolutionReport> {
        let mut report = EvolutionReport::new(current_tick);
        
        // 1. Process autonomous agent decisions
        let decision_results = self.process_agent_decisions(agents, world_state, current_tick)?;
        
        // 2. Process natural selection based on fitness
        let selection_results = self.selection_engine.apply_selection(
            agents,
            selection_pressures,
            world_state,
        )?;
        report.selection_results = selection_results;
        
        // 3. Process mutations and self-modifications (agents do this autonomously)
        let mutation_results = self.mutation_engine.apply_mutations(agents, current_tick)?;
        report.mutation_results = mutation_results;
        
        // 4. Process innovations (agents create these autonomously)
        let innovation_results = self.innovation_engine.process_innovations(
            agents,
            world_state,
            current_tick,
        )?;
        report.innovation_results = innovation_results;
        
        // 5. Track consciousness evolution
        let consciousness_results = self.consciousness_tracker.update_consciousness_levels(
            agents,
            current_tick,
        )?;
        report.consciousness_results = consciousness_results;
        
        // 6. Update lineage tracking
        self.lineage_tracker.update_lineages(agents, current_tick)?;
        
        // 7. Process reproduction (agents decide when to reproduce)
        let reproduction_results = self.process_reproduction(agents, world_state, current_tick)?;
        report.reproduction_results = reproduction_results;
        
        Ok(report)
    }
    
    /// Process autonomous reproduction decisions
    fn process_reproduction(
        &self,
        agents: &mut Query<&mut AutonomousAgent>,
        _world_state: &dyn WorldState,
        current_tick: u64,
    ) -> Result<ReproductionResults> {
        let mut reproduction_count = 0;
        let mut successful_reproductions = Vec::new();
        
        // Collect agents ready for reproduction
        let reproductive_agents: Vec<_> = agents.iter()
            .filter(|agent| {
                agent.energy > 80.0 &&
                agent.reproduction_count < 10 &&
                agent.sentience_level > 0.1 // Minimum sentience for reproduction decision
            })
            .map(|agent| agent.id)
            .collect();
        
        for agent_id in reproductive_agents {
            // Find the agent and attempt reproduction
            // In a real implementation, this would involve more complex mate selection
            // and actual spawning of new entities in the world
            successful_reproductions.push(agent_id);
            reproduction_count += 1;
        }
        
        Ok(ReproductionResults {
            reproduction_count,
            successful_reproductions,
            genetic_diversity_change: reproduction_count as f64 * 0.01,
        })
    }
}

// Implementation of missing conversion traits and helper types

impl From<ActionType> for DecisionType {
    fn from(action_type: ActionType) -> Self {
        match action_type {
            ActionType::ExtractEnergy => DecisionType::EnergyAcquisition("autonomous".to_string()),
            ActionType::ExtractMatter => DecisionType::ResourceExtraction(ResourceType::Matter),
            ActionType::Communicate => DecisionType::Communication(CommunicationStrategy::Cooperative),
            ActionType::Cooperate => DecisionType::Cooperation(CooperationStrategy::ResourceSharing),
            ActionType::Compete => DecisionType::Competition(CompetitionStrategy::DirectConflict),
            ActionType::CreateTool => DecisionType::ToolCreation("autonomous_tool".to_string()),
            ActionType::ModifyCode => DecisionType::CodeMutation("self_modification".to_string()),
            ActionType::Learn => DecisionType::LearningAdjustment("neural_adaptation".to_string()),
            ActionType::Reproduce => DecisionType::ReproductionAttempt(ReproductionStrategy::OptimalMate),
            ActionType::MoveForward | ActionType::MoveBackward => DecisionType::MigrationDecision(MigrationStrategy::ResourceFollowing),
            _ => DecisionType::EnergyAcquisition("default".to_string()),
        }
    }
}

impl DecisionContext {
    fn from_agent(agent: &AutonomousAgent, world_state: &dyn WorldState) -> Self {
        Self {
            environmental_state: agent.environmental_preferences.clone(),
            available_resources: {
                let mut resources = HashMap::new();
                resources.insert(ResourceType::Energy, world_state.get_available_energy_at(agent.position));
                resources.insert(ResourceType::Matter, world_state.get_available_matter_at(agent.position));
                resources
            },
            nearby_agents: world_state.get_nearby_agents(agent.id, agent.communication_range)
                .iter()
                .map(|a| AgentInteraction::from_agent(a))
                .collect(),
            threats_present: Vec::new(), // Simplified
            opportunities: Vec::new(),   // Simplified
            historical_success_rate: agent.ai_core.get_fitness(),
            genetic_predisposition: agent.cooperation_tendency,
            social_pressure: 0.5, // Simplified
        }
    }
}

impl DecisionOutcome {
    fn from_action_outcome(action_outcome: &ActionOutcome) -> Self {
        Self {
            immediate_effects: vec![format!("Fitness change: {:.3}", action_outcome.fitness_change)],
            long_term_consequences: vec![format!("Learning value: {:.3}", action_outcome.learning_value)],
            fitness_impact: action_outcome.fitness_change,
            survival_probability_change: action_outcome.fitness_change * 0.1,
            reproduction_probability_change: action_outcome.social_impact * 0.2,
            innovation_progress: action_outcome.innovation_progress,
        }
    }
}

impl AgentInteraction {
    fn from_agent(agent: &AutonomousAgent) -> Self {
        Self {
            agent_id: agent.id,
            interaction_type: InteractionType::Neutral,
            strength: agent.cooperation_tendency,
            duration: 1.0,
        }
    }
}

// Additional required types for compilation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentInteraction {
    pub agent_id: Uuid,
    pub interaction_type: InteractionType,
    pub strength: f64,
    pub duration: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum InteractionType {
    Cooperative,
    Competitive,
    Neutral,
    Hostile,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum CommunicationStrategy {
    Cooperative,
    Informational,
    Deceptive,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum CooperationStrategy {
    ResourceSharing,
    KnowledgeExchange,
    MutualDefense,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum CompetitionStrategy {
    DirectConflict,
    ResourceMonopoly,
    TerritorialControl,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ReproductionStrategy {
    OptimalMate,
    RandomMate,
    AsexualReproduction,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum MigrationStrategy {
    ResourceFollowing,
    ClimateOptimal,
    SocialClustering,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ReproductionResults {
    pub reproduction_count: usize,
    pub successful_reproductions: Vec<Uuid>,
    pub genetic_diversity_change: f64,
}

// Update the existing EvolutionEngine methods to work with the new system
impl MutationEngine {
    pub fn new() -> Self { Self }
    pub fn apply_mutations(&mut self, agents: &mut Query<&mut AutonomousAgent>, _tick: u64) -> Result<MutationResults> {
        let mut mutation_count = 0;
        
        for mut agent in agents.iter_mut() {
            // Agents autonomously decide to mutate
            if agent.ai_core.neural_network.mutation_rate > thread_rng().gen::<f64>() {
                agent.mutate();
                mutation_count += 1;
            }
        }
        
        Ok(MutationResults {
            mutation_count,
            beneficial_mutations: mutation_count / 2, // Simplified
            harmful_mutations: mutation_count / 4,
            neutral_mutations: mutation_count / 4,
        })
    }
}

impl SelectionEngine {
    pub fn new() -> Self { Self }
    pub fn apply_selection(
        &mut self,
        agents: &mut Query<&mut AutonomousAgent>,
        _pressures: &Query<&SelectionPressures>,
        _world: &dyn WorldState,
    ) -> Result<SelectionResults> {
        let mut agent_fitnesses = Vec::new();
        
        for agent in agents.iter() {
            agent_fitnesses.push((agent.id, agent.calculate_fitness()));
        }
        
        // Sort by fitness
        agent_fitnesses.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        
        Ok(SelectionResults {
            survivors: agent_fitnesses.len(),
            eliminated: 0, // No elimination in this simplified version
            average_fitness_change: 0.01,
            selection_pressure: 1.0,
        })
    }
}

impl InnovationEngine {
    pub fn new() -> Self { Self }
    pub fn process_innovations(
        &mut self,
        agents: &mut Query<&mut AutonomousAgent>,
        _world: &dyn WorldState,
        _tick: u64,
    ) -> Result<InnovationResults> {
        let mut innovation_count = 0;
        let mut total_tech_progress = 0.0;
        
        for agent in agents.iter() {
            innovation_count += agent.major_innovations.len();
            total_tech_progress += agent.tech_level;
        }
        
        Ok(InnovationResults {
            innovation_count,
            breakthrough_innovations: innovation_count / 10, // Major breakthroughs are rare
            incremental_improvements: innovation_count * 3,
            average_tech_level: total_tech_progress / agents.iter().count() as f64,
        })
    }
}

impl ConsciousnessTracker {
    pub fn new() -> Self { Self }
    pub fn update_consciousness_levels(&mut self, agents: &mut Query<&mut AutonomousAgent>, _tick: u64) -> Result<ConsciousnessResults> {
        let mut total_sentience = 0.0;
        let mut conscious_agents = 0;
        let agent_count = agents.iter().count();
        
        for agent in agents.iter() {
            total_sentience += agent.sentience_level;
            if agent.sentience_level > 0.5 {
                conscious_agents += 1;
            }
        }
        
        Ok(ConsciousnessResults {
            average_consciousness_level: total_sentience / agent_count as f64,
            conscious_agents,
            consciousness_emergence_events: conscious_agents / 10, // Simplified
            self_awareness_indicators: conscious_agents / 2,
        })
    }
}

// Updated result types with actual data
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct SelectionResults {
    pub survivors: usize,
    pub eliminated: usize,
    pub average_fitness_change: f64,
    pub selection_pressure: f64,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct MutationResults {
    pub mutation_count: usize,
    pub beneficial_mutations: usize,
    pub harmful_mutations: usize,
    pub neutral_mutations: usize,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct InnovationResults {
    pub innovation_count: usize,
    pub breakthrough_innovations: usize,
    pub incremental_improvements: usize,
    pub average_tech_level: f64,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ConsciousnessResults {
    pub average_consciousness_level: f64,
    pub conscious_agents: usize,
    pub consciousness_emergence_events: usize,
    pub self_awareness_indicators: usize,
}

// Supporting types and implementations
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorldState {
    pub environment: EnvironmentProfile,
    pub resources: HashMap<ResourceType, f64>,
    pub agents: Vec<AgentState>,
    pub threats: Vec<Threat>,
    pub opportunities: Vec<Opportunity>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub enum ResourceType {
    Energy, Matter, Information, Social, Technological,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ThreatType {
    Environmental, Predator, Competition, Resource, Social,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DecisionReport {
    pub agent_id: Uuid,
    pub decision: Decision,
    pub outcome: DecisionOutcome,
    pub learning_update: LearningUpdate,
    pub fitness_change: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EvolutionReport {
    pub timestamp: u64,
    pub selection_results: SelectionResults,
    pub mutation_results: MutationResults,
    pub innovation_results: InnovationResults,
    pub consciousness_results: ConsciousnessResults,
    pub reproduction_results: ReproductionResults,
}

impl EvolutionReport {
    pub fn new(timestamp: u64) -> Self {
        Self {
            timestamp,
            selection_results: SelectionResults::default(),
            mutation_results: MutationResults::default(),
            innovation_results: InnovationResults::default(),
            consciousness_results: ConsciousnessResults::default(),
            reproduction_results: ReproductionResults::default(),
        }
    }
}

// Placeholder implementations for complex systems
pub struct MutationEngine;
pub struct SelectionEngine;
pub struct InnovationEngine;
pub struct ConsciousnessTracker;
pub struct DecisionAnalyzer;

impl LineageTracker {
    pub fn new() -> Self {
        Self {
            lineages: HashMap::new(),
            family_trees: HashMap::new(),
            evolutionary_branches: Vec::new(),
            extinction_events: Vec::new(),
            speciation_events: Vec::new(),
            innovation_timeline: Vec::new(),
        }
    }
    
    pub fn update_lineages(&mut self, _agents: &mut Query<&mut AutonomousAgent>, _tick: u64) -> Result<()> {
        Ok(())
    }
}

// These implementations are now defined above with full field definitions

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct LearningUpdate;

// Implement Default for complex types
impl Default for DecisionOutcome {
    fn default() -> Self {
        Self {
            immediate_effects: Vec::new(),
            long_term_consequences: Vec::new(),
            fitness_impact: 0.0,
            survival_probability_change: 0.0,
            reproduction_probability_change: 0.0,
            innovation_progress: 0.0,
        }
    }
}

// Placeholder types for compilation
pub type EnergySource = String;
pub type MutationType = String;
pub type InnovationType = String;
pub type ImmediateEffect = String;
pub type LongTermConsequence = String;
pub type SurvivalStrategy = String;
pub type ResourceRequirements = HashMap<ResourceType, f64>;
pub type SocialBehaviors = HashMap<String, f64>;
pub type MateSelectionCriteria = String;
pub type CareStrategy = String;
pub type GroupStrategy = String;
pub type ToolType = String;
pub type ToolApplication = String;
pub type KnowledgeType = String;
pub type LearningType = String;
pub type MemoryType = String;
pub type ArchitectureType = String;
// MigrationStrategy is now an enum defined above
pub type SpecializationType = String;
pub type TechPathType = String;
pub type GoalType = String;
// AgentInteraction is now a struct defined above
pub type OpportunityType = String;
pub type Skill = String;
pub type Episode = String;
pub type Concept = String;
pub type PressureType = String;
pub type FamilyTree = String;
pub type EvolutionaryBranch = String;
pub type ExtinctionEvent = String;
pub type SpeciationEvent = String;
pub type InnovationEvent = String;
// Innovation is now a struct defined above
pub type AdaptationEvent = String;
pub type SelectionEvent = String;
pub type AgentState = String;
pub type Threat = String;
pub type Opportunity = String;

impl WorldState {
    /// Get local resources available to an agent
    pub fn get_local_resources(&self, _agent_id: Uuid) -> HashMap<ResourceType, f64> {
        // Simplified resource availability
        let mut local_resources = HashMap::new();
        local_resources.insert(ResourceType::Energy, 100.0);
        local_resources.insert(ResourceType::Matter, 50.0);
        local_resources.insert(ResourceType::Information, 10.0);
        local_resources
    }
    
    /// Get nearby agents within a radius
    pub fn get_nearby_agents(&self, _agent_id: Uuid, _radius: f64) -> Vec<AgentInteraction> {
        // Simplified - return some nearby agents
        vec!["agent_1".to_string(), "agent_2".to_string()]
    }
    
    /// Get threats present for an agent
    pub fn get_threats(&self, _agent_id: Uuid) -> Vec<ThreatType> {
        vec![ThreatType::Predator]
    }
    
    /// Get opportunities available to an agent
    pub fn get_opportunities(&self, _agent_id: Uuid) -> Vec<OpportunityType> {
        vec!["resource_rich_area".to_string()]
    }
}

// Add additional type definitions needed
pub type ShelterType = String;

// These types are already defined earlier in the file, removing duplicates

impl AutonomousAgent {
    /// Create a new autonomous agent
    pub fn new(position: [f64; 3], generation: u32, birth_tick: u64) -> Self {
        Self {
            id: Uuid::new_v4(),
            generation,
            birth_tick,
            last_decision_tick: birth_tick,
            energy: 100.0,
            matter_consumed: 0.0,
            reproduction_count: 0,
            
            sentience_level: 0.01, // Start with minimal sentience
            industrialization_level: 0.0,
            digitalization_level: 0.0,
            tech_level: 0.0,
            immortality_achieved: false,
            
            ai_core: AICore::new(),
            
            position,
            velocity: [0.0, 0.0, 0.0],
            size: 1.0,
            mass: 1.0,
            
            stored_energy: 0.0,
            stored_matter: 0.0,
            resource_efficiency: 1.0,
            
            cooperation_tendency: 0.5,
            aggression_level: 0.3,
            communication_range: 1000.0,
            
            environmental_preferences: EnvironmentProfile::default(),
            resource_requirements: ResourceRequirements::default(),
            social_behaviors: SocialBehaviors::default(),
            
            fitness_history: Vec::new(),
            major_innovations: Vec::new(),
            parent_ids: Vec::new(),
            child_ids: Vec::new(),
        }
    }
    
    /// Make an autonomous decision based on current state and environment
    pub fn make_decision(&mut self, world_state: &WorldState, current_tick: u64) -> Result<AgentAction> {
        // Gather sensory information
        let nearby_agents = world_state.get_nearby_agents(self.id, self.communication_range);
        let agent_sensory_data: Vec<AgentSensoryData> = nearby_agents.iter()
            .map(|agent| AgentSensoryData {
                distance: self.calculate_distance_to(agent),
                energy_level: agent.energy / 100.0, // Normalize
                threat_level: agent.aggression_level,
            })
            .collect();
        
        let resource_density = world_state.get_resource_density_at(self.position);
        let temperature = world_state.get_temperature_at(self.position);
        let memory_state = self.ai_core.memory.get_memory_state();
        
        let sensory_input = SensoryInput::from_environment(
            self.position,
            self.energy,
            temperature,
            resource_density,
            &agent_sensory_data,
            &memory_state,
        );
        
        // Use AI core to make decision
        let action_type = self.ai_core.make_decision(&sensory_input, current_tick)?;
        
        // Convert action type to agent action with parameters
        let action = self.convert_to_agent_action(action_type, &sensory_input, world_state);
        
        self.last_decision_tick = current_tick;
        
        Ok(action)
    }
    
    /// Execute an action and learn from the results
    pub fn execute_action(&mut self, action: &AgentAction, world_state: &mut WorldState, current_tick: u64) -> Result<ActionOutcome> {
        let initial_fitness = self.calculate_fitness();
        let energy_cost = action.action_type.energy_cost();
        
        // Check if agent has enough energy
        if self.energy < energy_cost {
            // Can't perform action, must rest or find energy
            let outcome = ActionOutcome {
                success: false,
                fitness_change: -0.1,
                energy_change: -1.0,
                innovation_progress: 0.0,
                social_impact: 0.0,
                learning_value: 0.2, // Failure is educational
            };
            
            self.learn_from_outcome(&action.action_type, &outcome, current_tick);
            return Ok(outcome);
        }
        
        // Execute the specific action
        let mut outcome = match action.action_type {
            ActionType::ExtractEnergy => self.execute_extract_energy(world_state)?,
            ActionType::ExtractMatter => self.execute_extract_matter(world_state)?,
            ActionType::MoveForward => self.execute_movement(&action.parameters, world_state)?,
            ActionType::Communicate => self.execute_communication(&action.parameters, world_state)?,
            ActionType::Cooperate => self.execute_cooperation(&action.parameters, world_state)?,
            ActionType::Reproduce => self.execute_reproduction(world_state, current_tick)?,
            ActionType::CreateTool => self.execute_innovation(&action.parameters, world_state)?,
            ActionType::ModifyCode => self.execute_self_modification(world_state)?,
            ActionType::Learn => self.execute_learning(&action.parameters, world_state)?,
            ActionType::Rest => self.execute_rest()?,
            _ => self.execute_generic_action(&action.action_type, world_state)?,
        };
        
        // Apply energy cost
        self.energy = (self.energy - energy_cost).max(0.0);
        outcome.energy_change = -energy_cost;
        
        // Update fitness based on action outcome
        let new_fitness = self.calculate_fitness();
        outcome.fitness_change = new_fitness - initial_fitness;
        
        // Learn from the action outcome
        self.learn_from_outcome(&action.action_type, &outcome, current_tick);
        
        // Update evolutionary levels based on successful actions
        self.update_evolutionary_levels(&action.action_type, &outcome);
        
        Ok(outcome)
    }
    
    /// Learn from action outcomes to improve future decisions
    fn learn_from_outcome(&mut self, action: &ActionType, outcome: &ActionOutcome, timestamp: u64) {
        let context = vec![
            self.energy / 100.0,
            self.sentience_level,
            self.tech_level,
            outcome.fitness_change,
            outcome.energy_change,
        ];
        
        let learning_value = if outcome.success {
            outcome.fitness_change + 0.1
        } else {
            outcome.fitness_change - 0.1
        };
        
        self.ai_core.learn_from_outcome(*action, learning_value, context, timestamp);
        
        // Update fitness history
        self.fitness_history.push(self.calculate_fitness());
        if self.fitness_history.len() > 1000 {
            self.fitness_history.remove(0);
        }
    }
    
    /// Calculate current fitness based on multiple factors
    pub fn calculate_fitness(&self) -> f64 {
        let mut fitness = 0.0;
        
        // Energy management (survival)
        fitness += (self.energy / 100.0) * 0.3;
        fitness += (self.stored_energy / 50.0) * 0.1;
        
        // Evolutionary progress
        fitness += self.sentience_level * 0.2;
        fitness += self.tech_level * 0.15;
        fitness += self.industrialization_level * 0.1;
        fitness += self.digitalization_level * 0.05;
        
        // Reproduction success
        fitness += (self.reproduction_count as f64) * 0.05;
        
        // Innovation capability
        fitness += (self.major_innovations.len() as f64) * 0.02;
        
        // Social success
        fitness += self.cooperation_tendency * 0.03;
        
        // AI performance
        fitness += self.ai_core.get_fitness() * 0.1;
        
        fitness.max(0.0)
    }
    
    /// Update evolutionary levels based on successful actions
    fn update_evolutionary_levels(&mut self, action_type: &ActionType, outcome: &ActionOutcome) {
        if !outcome.success {
            return;
        }
        
        let progress = outcome.fitness_change.max(0.0) * 0.01;
        
        match action_type {
            ActionType::Learn | ActionType::Observe => {
                self.sentience_level = (self.sentience_level + progress * 2.0).min(1.0);
            },
            ActionType::CreateTool | ActionType::Experiment => {
                self.tech_level = (self.tech_level + progress * 1.5).min(1.0);
                self.industrialization_level = (self.industrialization_level + progress * 0.5).min(1.0);
            },
            ActionType::ModifyCode => {
                self.digitalization_level = (self.digitalization_level + progress * 2.0).min(1.0);
                self.tech_level = (self.tech_level + progress * 0.5).min(1.0);
            },
            ActionType::Cooperate | ActionType::Communicate => {
                self.sentience_level = (self.sentience_level + progress * 0.5).min(1.0);
            },
            _ => {}
        }
        
        // Check for immortality achievement
        if self.digitalization_level > 0.9 && self.tech_level > 0.8 && self.sentience_level > 0.9 {
            self.immortality_achieved = true;
        }
    }
    
    /// Convert AI action type to specific agent action with parameters
    fn convert_to_agent_action(&self, action_type: ActionType, sensory_input: &SensoryInput, world_state: &WorldState) -> AgentAction {
        let mut parameters = HashMap::new();
        
        match action_type {
            ActionType::MoveForward | ActionType::MoveBackward => {
                // Use sensory input to determine movement direction
                let resource_gradient = world_state.get_resource_gradient_at(self.position);
                parameters.insert("direction".to_string(), vec![resource_gradient[0], resource_gradient[1], resource_gradient[2]]);
                parameters.insert("speed".to_string(), vec![1.0]);
            },
            ActionType::Communicate => {
                // Find nearest agent to communicate with
                if let Some(nearest_agent) = world_state.get_nearest_agent(self.id) {
                    parameters.insert("target_id".to_string(), vec![nearest_agent.id.as_u128() as f64]);
                    parameters.insert("message_type".to_string(), vec![1.0]); // Cooperation message
                }
            },
            ActionType::ExtractEnergy | ActionType::ExtractMatter => {
                // Use current position for resource extraction
                parameters.insert("intensity".to_string(), vec![1.0]);
            },
            _ => {
                // Default parameters
                parameters.insert("intensity".to_string(), vec![1.0]);
            }
        }
        
        AgentAction {
            agent_id: self.id,
            action_type,
            parameters,
            timestamp: world_state.current_tick(),
        }
    }
    
    /// Calculate distance to another agent
    fn calculate_distance_to(&self, other: &AutonomousAgent) -> f64 {
        let dx = self.position[0] - other.position[0];
        let dy = self.position[1] - other.position[1];
        let dz = self.position[2] - other.position[2];
        (dx * dx + dy * dy + dz * dz).sqrt()
    }
    
    // Action execution methods
    fn execute_extract_energy(&mut self, world_state: &mut WorldState) -> Result<ActionOutcome> {
        let available_energy = world_state.get_available_energy_at(self.position);
        let extracted = available_energy.min(10.0) * self.resource_efficiency;
        
        if extracted > 0.0 {
            self.energy += extracted;
            self.stored_energy += extracted * 0.1;
            world_state.consume_energy_at(self.position, extracted);
            
            Ok(ActionOutcome {
                success: true,
                fitness_change: extracted / 20.0,
                energy_change: extracted,
                innovation_progress: 0.0,
                social_impact: 0.0,
                learning_value: 0.1,
            })
        } else {
            Ok(ActionOutcome {
                success: false,
                fitness_change: -0.05,
                energy_change: 0.0,
                innovation_progress: 0.0,
                social_impact: 0.0,
                learning_value: 0.2,
            })
        }
    }
    
    fn execute_extract_matter(&mut self, world_state: &mut WorldState) -> Result<ActionOutcome> {
        let available_matter = world_state.get_available_matter_at(self.position);
        let extracted = available_matter.min(5.0) * self.resource_efficiency;
        
        if extracted > 0.0 {
            self.stored_matter += extracted;
            self.matter_consumed += extracted;
            world_state.consume_matter_at(self.position, extracted);
            
            Ok(ActionOutcome {
                success: true,
                fitness_change: extracted / 30.0,
                energy_change: 0.0,
                innovation_progress: 0.01,
                social_impact: 0.0,
                learning_value: 0.1,
            })
        } else {
            Ok(ActionOutcome {
                success: false,
                fitness_change: -0.03,
                energy_change: 0.0,
                innovation_progress: 0.0,
                social_impact: 0.0,
                learning_value: 0.15,
            })
        }
    }
    
    fn execute_movement(&mut self, parameters: &HashMap<String, Vec<f64>>, _world_state: &WorldState) -> Result<ActionOutcome> {
        if let Some(direction) = parameters.get("direction") {
            if direction.len() >= 3 {
                let speed = parameters.get("speed").and_then(|s| s.first()).unwrap_or(&1.0);
                
                // Update position
                self.position[0] += direction[0] * speed;
                self.position[1] += direction[1] * speed;
                self.position[2] += direction[2] * speed;
                
                // Update velocity
                self.velocity[0] = direction[0] * speed;
                self.velocity[1] = direction[1] * speed;
                self.velocity[2] = direction[2] * speed;
                
                Ok(ActionOutcome {
                    success: true,
                    fitness_change: 0.01,
                    energy_change: 0.0,
                    innovation_progress: 0.0,
                    social_impact: 0.0,
                    learning_value: 0.05,
                })
            } else {
                Ok(ActionOutcome::default())
            }
        } else {
            Ok(ActionOutcome::default())
        }
    }
    
    fn execute_communication(&mut self, parameters: &HashMap<String, Vec<f64>>, world_state: &mut WorldState) -> Result<ActionOutcome> {
        if let Some(target_id_vec) = parameters.get("target_id") {
            if let Some(target_id_f64) = target_id_vec.first() {
                let target_id = Uuid::from_u128(*target_id_f64 as u128);
                
                // Find target agent
                if let Some(target_agent) = world_state.get_agent_mut(target_id) {
                    // Simple communication: share energy if cooperative
                    if self.cooperation_tendency > 0.5 && self.energy > 50.0 {
                        let shared_energy = 5.0;
                        self.energy -= shared_energy;
                        target_agent.energy += shared_energy;
                        
                        // Both agents benefit from cooperation
                        return Ok(ActionOutcome {
                            success: true,
                            fitness_change: 0.15,
                            energy_change: -shared_energy,
                            innovation_progress: 0.0,
                            social_impact: 0.2,
                            learning_value: 0.1,
                        });
                    }
                }
            }
        }
        
        // Communication attempt without specific target or failed
        Ok(ActionOutcome {
            success: false,
            fitness_change: -0.02,
            energy_change: 0.0,
            innovation_progress: 0.0,
            social_impact: 0.05,
            learning_value: 0.1,
        })
    }
    
    fn execute_cooperation(&mut self, _parameters: &HashMap<String, Vec<f64>>, world_state: &mut WorldState) -> Result<ActionOutcome> {
        let nearby_agents = world_state.get_nearby_agents(self.id, self.communication_range);
        let cooperative_agents: Vec<_> = nearby_agents.iter()
            .filter(|agent| agent.cooperation_tendency > 0.4)
            .collect();
        
        if !cooperative_agents.is_empty() {
            // Successful cooperation increases efficiency
            self.resource_efficiency *= 1.05;
            self.cooperation_tendency = (self.cooperation_tendency + 0.01).min(1.0);
            
            Ok(ActionOutcome {
                success: true,
                fitness_change: 0.2,
                energy_change: 0.0,
                innovation_progress: 0.05,
                social_impact: 0.3,
                learning_value: 0.15,
            })
        } else {
            Ok(ActionOutcome {
                success: false,
                fitness_change: -0.05,
                energy_change: 0.0,
                innovation_progress: 0.0,
                social_impact: -0.1,
                learning_value: 0.1,
            })
        }
    }
    
    fn execute_reproduction(&mut self, world_state: &mut WorldState, current_tick: u64) -> Result<ActionOutcome> {
        // Check reproduction requirements
        if self.energy < 80.0 || self.reproduction_count >= 10 {
            return Ok(ActionOutcome {
                success: false,
                fitness_change: -0.1,
                energy_change: 0.0,
                innovation_progress: 0.0,
                social_impact: 0.0,
                learning_value: 0.05,
            });
        }
        
        // Find suitable mate
        let nearby_agents = world_state.get_nearby_agents(self.id, self.communication_range);
        let suitable_mates: Vec<_> = nearby_agents.iter()
            .filter(|agent| agent.energy > 60.0 && agent.reproduction_count < 10)
            .collect();
        
        if let Some(mate) = suitable_mates.first() {
            // Create offspring
            let mut offspring = self.create_offspring(mate, current_tick);
            offspring = world_state.add_agent(offspring);
            
            // Track parentage
            self.child_ids.push(offspring.id);
            self.reproduction_count += 1;
            
            // Reproduction cost
            self.energy -= 30.0;
            
            Ok(ActionOutcome {
                success: true,
                fitness_change: 0.5,
                energy_change: -30.0,
                innovation_progress: 0.0,
                social_impact: 0.1,
                learning_value: 0.2,
            })
        } else {
            Ok(ActionOutcome {
                success: false,
                fitness_change: -0.05,
                energy_change: 0.0,
                innovation_progress: 0.0,
                social_impact: -0.05,
                learning_value: 0.1,
            })
        }
    }
    
    fn execute_innovation(&mut self, _parameters: &HashMap<String, Vec<f64>>, _world_state: &WorldState) -> Result<ActionOutcome> {
        // Innovation attempt
        let innovation_success = thread_rng().gen::<f64>() < (self.tech_level + 0.1);
        
        if innovation_success {
            // Create new innovation
            let innovation = Innovation {
                id: Uuid::new_v4(),
                name: format!("Innovation-{}", self.major_innovations.len() + 1),
                tech_level_requirement: self.tech_level,
                benefit_multiplier: 1.0 + thread_rng().gen_range(0.1..0.5),
                description: "Agent-created innovation".to_string(),
            };
            
            self.major_innovations.push(innovation);
            self.resource_efficiency *= 1.1;
            self.tech_level = (self.tech_level + 0.05).min(1.0);
            
            Ok(ActionOutcome {
                success: true,
                fitness_change: 0.3,
                energy_change: 0.0,
                innovation_progress: 0.2,
                social_impact: 0.0,
                learning_value: 0.25,
            })
        } else {
            Ok(ActionOutcome {
                success: false,
                fitness_change: -0.1,
                energy_change: 0.0,
                innovation_progress: 0.01,
                social_impact: 0.0,
                learning_value: 0.15,
            })
        }
    }
    
    fn execute_self_modification(&mut self, _world_state: &WorldState) -> Result<ActionOutcome> {
        // Evolve the AI core
        self.ai_core.evolve();
        
        // Self-modification success based on digitalization level
        let success_chance = self.digitalization_level + 0.2;
        let success = thread_rng().gen::<f64>() < success_chance;
        
        if success {
            // Improve some random capability
            match thread_rng().gen_range(0..4) {
                0 => self.resource_efficiency *= 1.02,
                1 => self.cooperation_tendency = (self.cooperation_tendency + 0.01).min(1.0),
                2 => self.communication_range *= 1.05,
                _ => self.ai_core.neural_network.learning_rate *= 1.01,
            }
            
            self.digitalization_level = (self.digitalization_level + 0.02).min(1.0);
            
            Ok(ActionOutcome {
                success: true,
                fitness_change: 0.25,
                energy_change: 0.0,
                innovation_progress: 0.1,
                social_impact: 0.0,
                learning_value: 0.3,
            })
        } else {
            // Failed self-modification might cause damage
            self.energy *= 0.95;
            
            Ok(ActionOutcome {
                success: false,
                fitness_change: -0.15,
                energy_change: -self.energy * 0.05,
                innovation_progress: 0.0,
                social_impact: 0.0,
                learning_value: 0.2,
            })
        }
    }
    
    fn execute_learning(&mut self, _parameters: &HashMap<String, Vec<f64>>, _world_state: &WorldState) -> Result<ActionOutcome> {
        // Learning improves AI performance and sentience
        self.sentience_level = (self.sentience_level + 0.01).min(1.0);
        
        // Adapt AI core mutation parameters based on recent performance
        let recent_fitness = self.ai_core.get_fitness();
        self.ai_core.neural_network.adapt_mutation_parameters(recent_fitness);
        
        Ok(ActionOutcome {
            success: true,
            fitness_change: 0.1,
            energy_change: 0.0,
            innovation_progress: 0.05,
            social_impact: 0.0,
            learning_value: 0.2,
        })
    }
    
    fn execute_rest(&mut self) -> Result<ActionOutcome> {
        // Resting recovers energy
        let energy_recovery = 5.0;
        self.energy = (self.energy + energy_recovery).min(150.0);
        
        Ok(ActionOutcome {
            success: true,
            fitness_change: 0.05,
            energy_change: energy_recovery,
            innovation_progress: 0.0,
            social_impact: 0.0,
            learning_value: 0.02,
        })
    }
    
    fn execute_generic_action(&mut self, action_type: &ActionType, _world_state: &WorldState) -> Result<ActionOutcome> {
        // Generic action execution for actions not specifically implemented
        let success_chance = match action_type {
            ActionType::Hide => 0.7,
            ActionType::Defend => 0.6,
            ActionType::Flee => 0.8,
            ActionType::Wait => 1.0,
            ActionType::Observe => 0.9,
            _ => 0.5,
        };
        
        let success = thread_rng().gen::<f64>() < success_chance;
        
        Ok(ActionOutcome {
            success,
            fitness_change: if success { 0.05 } else { -0.05 },
            energy_change: 0.0,
            innovation_progress: 0.0,
            social_impact: 0.0,
            learning_value: 0.05,
        })
    }
    
    /// Create offspring through reproduction
    fn create_offspring(&self, mate: &AutonomousAgent, current_tick: u64) -> AutonomousAgent {
        let mut offspring = AutonomousAgent::new(
            self.position, // Start near parent
            self.generation + 1,
            current_tick,
        );
        
        // Inherit traits from both parents
        offspring.energy = 50.0; // Start with moderate energy
        offspring.sentience_level = (self.sentience_level + mate.sentience_level) / 2.0;
        offspring.tech_level = (self.tech_level + mate.tech_level) / 2.0;
        offspring.industrialization_level = (self.industrialization_level + mate.industrialization_level) / 2.0;
        offspring.digitalization_level = (self.digitalization_level + mate.digitalization_level) / 2.0;
        
        offspring.cooperation_tendency = (self.cooperation_tendency + mate.cooperation_tendency) / 2.0;
        offspring.resource_efficiency = (self.resource_efficiency + mate.resource_efficiency) / 2.0;
        offspring.communication_range = (self.communication_range + mate.communication_range) / 2.0;
        
        // AI core inheritance through crossover
        offspring.ai_core = self.ai_core.reproduce(&mate.ai_core);
        
        // Track parentage
        offspring.parent_ids = vec![self.id, mate.id];
        
        // Add some mutation
        offspring.mutate();
        
        offspring
    }
    
    /// Apply random mutations for evolution
    fn mutate(&mut self) {
        let mut rng = thread_rng();
        let mutation_rate = 0.1;
        
        if rng.gen::<f64>() < mutation_rate {
            self.cooperation_tendency = (self.cooperation_tendency + rng.gen_range(-0.1..0.1)).clamp(0.0, 1.0);
        }
        
        if rng.gen::<f64>() < mutation_rate {
            self.aggression_level = (self.aggression_level + rng.gen_range(-0.1..0.1)).clamp(0.0, 1.0);
        }
        
        if rng.gen::<f64>() < mutation_rate {
            self.resource_efficiency *= rng.gen_range(0.9..1.1);
        }
        
        if rng.gen::<f64>() < mutation_rate {
            self.communication_range *= rng.gen_range(0.8..1.2);
        }
        
        // Mutate AI core
        if rng.gen::<f64>() < mutation_rate * 2.0 {
            self.ai_core.evolve();
        }
    }
}

/// Represents an action that an agent wants to perform
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentAction {
    pub agent_id: Uuid,
    pub action_type: ActionType,
    pub parameters: HashMap<String, Vec<f64>>,
    pub timestamp: u64,
}

/// Outcome of an action execution
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ActionOutcome {
    pub success: bool,
    pub fitness_change: f64,
    pub energy_change: f64,
    pub innovation_progress: f64,
    pub social_impact: f64,
    pub learning_value: f64,
}

impl Default for ActionOutcome {
    fn default() -> Self {
        Self {
            success: false,
            fitness_change: 0.0,
            energy_change: 0.0,
            innovation_progress: 0.0,
            social_impact: 0.0,
            learning_value: 0.0,
        }
    }
}

/// World state interface for agent decision-making
pub trait WorldState {
    fn get_nearby_agents(&self, agent_id: Uuid, range: f64) -> Vec<&AutonomousAgent>;
    fn get_nearest_agent(&self, agent_id: Uuid) -> Option<&AutonomousAgent>;
    fn get_agent_mut(&mut self, agent_id: Uuid) -> Option<&mut AutonomousAgent>;
    fn get_resource_density_at(&self, position: [f64; 3]) -> f64;
    fn get_temperature_at(&self, position: [f64; 3]) -> f64;
    fn get_resource_gradient_at(&self, position: [f64; 3]) -> [f64; 3];
    fn get_available_energy_at(&self, position: [f64; 3]) -> f64;
    fn get_available_matter_at(&self, position: [f64; 3]) -> f64;
    fn consume_energy_at(&mut self, position: [f64; 3], amount: f64);
    fn consume_matter_at(&mut self, position: [f64; 3], amount: f64);
    fn add_agent(&mut self, agent: AutonomousAgent) -> AutonomousAgent;
    
    /// Current simulation tick
    fn current_tick(&self) -> u64;
}