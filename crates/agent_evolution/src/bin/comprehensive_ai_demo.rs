//! # Comprehensive AI Demo
//!
//! This demo showcases all the advanced AI capabilities implemented in the agent_evolution crate:
//! - Meta-learning with adaptive parameter tuning
//! - Hypernetwork for dynamic architecture generation
//! - Curiosity-driven learning and exploration
//! - Advanced self-modification capabilities
//! - Open-ended evolution with novelty detection
//!
//! The demo runs a multi-agent simulation where agents evolve, learn, and discover novel behaviors
//! through open-ended exploration.

use agent_evolution::{
    ai_core::AICore,
    meta_learning::{MetaLearner, MetaParameter},
    hypernetwork::{Hypernetwork, TaskEmbedding, TaskType},
    curiosity::{CuriositySystem, ActionType, Experience},
    self_modification::AdvancedSelfModification,
    open_ended_evolution::OpenEndedEvolution,
    genetics::Genome,
};
use anyhow::Result;
use nalgebra::DVector;
use rand::Rng;
use std::collections::HashMap;
use std::thread;
use std::time::{Duration, Instant};
use uuid::Uuid;

/// Agent with all AI capabilities
#[derive(Debug)]
struct ComprehensiveAgent {
    id: Uuid,
    name: String,
    neural_core: AICore,
    meta_learner: MetaLearner,
    hypernetwork: Hypernetwork,
    curiosity_system: CuriositySystem,
    self_modification: AdvancedSelfModification,
    genome: Genome,
    open_ended_evolution: OpenEndedEvolution,
    performance_history: Vec<f64>,
    discovery_count: usize,
    generation: u64,
}

impl ComprehensiveAgent {
    /// Create a new agent with all AI capabilities
    fn new(name: String) -> Self {
        let id = Uuid::new_v4();
        
        Self {
            id,
            name,
            neural_core: AICore::new(),
            meta_learner: MetaLearner::new(),
            hypernetwork: Hypernetwork::new(),
            curiosity_system: CuriositySystem::new(10),
            self_modification: AdvancedSelfModification::new(),
            genome: Genome::new(),
            open_ended_evolution: OpenEndedEvolution::new(),
            performance_history: Vec::new(),
            discovery_count: 0,
            generation: 0,
        }
    }

    /// Process a learning cycle
    fn process_cycle(&mut self, environment_state: &DVector<f64>, cycle: u64) -> Result<AgentCycleOutput> {
        let start_time = Instant::now();
        
        // 1. Generate task embedding for hypernetwork
        let task_embedding = self.generate_task_embedding(environment_state)?;
        
        // 2. Generate optimal architecture for current task
        let architecture = self.hypernetwork.generate_architecture(&task_embedding)?;
        
        // 3. Adapt neural core to new architecture
        self.adapt_neural_architecture(&architecture)?;
        
        // 4. Select action using curiosity-driven exploration
        let available_actions = vec![
            ActionType::Explore,
            ActionType::Investigate,
            ActionType::Experiment,
            ActionType::Observe,
            ActionType::Interact,
            ActionType::Learn,
            ActionType::Create,
            ActionType::Discover,
        ];
        
        let action_selection = self.curiosity_system.select_action(available_actions, environment_state)?;
        
        // 5. Execute action and get outcome
        let outcome = self.simulate_action_outcome(&action_selection.selected_action, environment_state);
        
        // 6. Create experience for learning
        let experience = Experience {
            id: Uuid::new_v4(),
            timestamp: cycle as f64,
            sensory_input: environment_state.clone(),
            context: DVector::zeros(5),
            action_taken: Some(action_selection.selected_action.clone()),
            outcome: Some(outcome),
            novelty_score: action_selection.curiosity_contribution,
            curiosity_value: action_selection.exploration_bonus,
        };
        
        // 7. Process experience through curiosity system
        let curiosity_output = self.curiosity_system.process_experience(experience.clone())?;
        
        // 8. Check for novelty using open-ended evolution
        let novelty_output = self.open_ended_evolution.process_experience(
            self.id,
            &experience,
            &self.neural_core,
            &self.curiosity_system,
            cycle as f64,
        )?;
        
        // 9. Update meta-learning based on performance
        self.update_meta_learning(outcome, cycle)?;
        
        // 10. Perform self-modification if beneficial
        let modification_output = self.perform_self_modification(cycle)?;
        
        // 11. Update performance tracking
        self.update_performance(outcome, novelty_output.novelty_found);
        
        // 12. Provide feedback to hypernetwork
        self.hypernetwork.update_from_feedback(architecture.id, outcome)?;
        
        let cycle_time = start_time.elapsed();
        
        Ok(AgentCycleOutput {
            agent_id: self.id,
            agent_name: self.name.clone(),
            action_taken: action_selection.selected_action,
            outcome,
            curiosity_level: curiosity_output.curiosity_level,
            novelty_detected: novelty_output.novelty_found,
            novelty_score: novelty_output.novelty_score,
            self_modifications: modification_output.modifications_performed,
            performance: outcome,
            cycle_time: cycle_time.as_millis() as u64,
            generation: self.generation,
            discovery_count: self.discovery_count,
        })
    }

    /// Generate task embedding for hypernetwork
    fn generate_task_embedding(&self, environment_state: &DVector<f64>) -> Result<TaskEmbedding> {
        // Create a task embedding based on environment state
        let complexity = environment_state.iter().sum::<f64>() / environment_state.len() as f64;
        let performance_target = self.performance_history.last().unwrap_or(&0.5);
        
        Ok(TaskEmbedding {
            task_type: TaskType::ReinforcementLearning,
            complexity: complexity.clamp(0.0, 1.0),
            input_dim: environment_state.len(),
            output_dim: 8, // Number of action types
            constraints: Vec::new(),
            performance_target: *performance_target,
        })
    }

    /// Adapt neural architecture based on hypernetwork output
    fn adapt_neural_architecture(&mut self, architecture: &agent_evolution::hypernetwork::GeneratedArchitecture) -> Result<()> {
        // Update the neural network's learning rate from the architecture
        self.neural_core.neural_network.learning_rate = architecture.learning_parameters.learning_rate;
        
        Ok(())
    }

    /// Simulate action outcome
    fn simulate_action_outcome(&self, action: &ActionType, environment_state: &DVector<f64>) -> f64 {
        let base_outcome = match action {
            ActionType::Explore => 0.3,
            ActionType::Investigate => 0.5,
            ActionType::Experiment => 0.7,
            ActionType::Observe => 0.2,
            ActionType::Interact => 0.6,
            ActionType::Learn => 0.8,
            ActionType::Create => 0.9,
            ActionType::Discover => 1.0,
        };
        
        // Add some randomness and environment interaction
        let mut rng = rand::thread_rng();
        let noise = rng.gen_range(-0.2..0.2);
        let environment_factor = environment_state.iter().sum::<f64>() / environment_state.len() as f64;
        
        (base_outcome + noise + environment_factor * 0.1).max(-1.0).min(1.0)
    }

    /// Update meta-learning system
    fn update_meta_learning(&mut self, outcome: f64, cycle: u64) -> Result<()> {
        // Create parameter map for meta-learning
        let mut params = HashMap::new();
        params.insert(MetaParameter::LearningRate, self.neural_core.neural_network.learning_rate);
        params.insert(MetaParameter::ExplorationRate, self.neural_core.exploration_rate);
        params.insert(MetaParameter::MutationRate, self.neural_core.neural_network.mutation_rate);
        
        // Update meta-learner
        let suggestions = self.meta_learner.update_core(self.id, outcome, &params)?;
        
        // Apply suggested parameter changes
        if let Some(&learning_rate) = suggestions.get(&MetaParameter::LearningRate) {
            self.neural_core.neural_network.learning_rate = learning_rate;
        }
        if let Some(&exploration_rate) = suggestions.get(&MetaParameter::ExplorationRate) {
            self.neural_core.exploration_rate = exploration_rate;
        }
        if let Some(&mutation_rate) = suggestions.get(&MetaParameter::MutationRate) {
            self.neural_core.neural_network.mutation_rate = mutation_rate;
        }
        
        Ok(())
    }

    /// Perform self-modification
    fn perform_self_modification(&mut self, cycle: u64) -> Result<agent_evolution::self_modification::SelfModificationOutput> {
        // Only perform self-modification every 10 cycles to avoid excessive changes
        if cycle % 10 == 0 {
            self.self_modification.perform_self_modification(
                &mut self.neural_core,
                &mut self.genome,
                &mut self.meta_learner,
                &mut self.curiosity_system,
                &mut self.hypernetwork,
                cycle as f64,
            )
        } else {
            Ok(agent_evolution::self_modification::SelfModificationOutput {
                modifications_performed: 0,
                success_rate: 0.0,
                performance_impact: 0.0,
                modifications: Vec::new(),
            })
        }
    }

    /// Update performance tracking
    fn update_performance(&mut self, outcome: f64, novelty_detected: bool) {
        self.performance_history.push(outcome);
        
        // Keep only recent history
        if self.performance_history.len() > 100 {
            self.performance_history.remove(0);
        }
        
        if novelty_detected {
            self.discovery_count += 1;
        }
    }

    /// Get agent statistics
    fn get_statistics(&self) -> AgentStatistics {
        let recent_performance = self.performance_history.iter().rev().take(10).sum::<f64>() / 10.0;
        let average_performance = self.performance_history.iter().sum::<f64>() / self.performance_history.len().max(1) as f64;
        
        AgentStatistics {
            agent_id: self.id,
            agent_name: self.name.clone(),
            generation: self.generation,
            discovery_count: self.discovery_count,
            average_performance,
            recent_performance,
            curiosity_stats: self.curiosity_system.get_statistics(),
            meta_learning_stats: self.meta_learner.cores.get(&self.id).cloned().unwrap_or_default(),
            self_modification_stats: self.self_modification.get_statistics(),
            open_ended_stats: self.open_ended_evolution.get_statistics(),
        }
    }
}

/// Output from a single agent cycle
#[derive(Debug, Clone)]
struct AgentCycleOutput {
    agent_id: Uuid,
    agent_name: String,
    action_taken: ActionType,
    outcome: f64,
    curiosity_level: f64,
    novelty_detected: bool,
    novelty_score: f64,
    self_modifications: usize,
    performance: f64,
    cycle_time: u64,
    generation: u64,
    discovery_count: usize,
}

/// Comprehensive agent statistics
#[derive(Debug)]
struct AgentStatistics {
    agent_id: Uuid,
    agent_name: String,
    generation: u64,
    discovery_count: usize,
    average_performance: f64,
    recent_performance: f64,
    curiosity_stats: agent_evolution::curiosity::CuriosityStatistics,
    meta_learning_stats: agent_evolution::meta_learning::CoreMetaStats,
    self_modification_stats: agent_evolution::self_modification::SelfModificationStatistics,
    open_ended_stats: agent_evolution::open_ended_evolution::OpenEndedEvolutionStatistics,
}

/// Environment generator for dynamic simulation
#[derive(Debug)]
struct EnvironmentGenerator {
    complexity: f64,
    novelty_factor: f64,
    change_rate: f64,
}

impl EnvironmentGenerator {
    fn new() -> Self {
        Self {
            complexity: 0.5,
            novelty_factor: 0.3,
            change_rate: 0.1,
        }
    }

    fn generate_state(&mut self, cycle: u64) -> DVector<f64> {
        let mut rng = rand::thread_rng();
        
        // Base state with some complexity
        let mut state = Vec::new();
        for i in 0..10 {
            let base_value = (cycle as f64 * 0.01 + i as f64 * 0.1).sin();
            let complexity_modifier = self.complexity * (i as f64 * 0.2).sin();
            let novelty_modifier = self.novelty_factor * rng.gen_range(-0.5..0.5);
            let change_modifier = self.change_rate * (cycle as f64 * 0.05).sin();
            
            let value = (base_value + complexity_modifier + novelty_modifier + change_modifier).clamp(-1.0, 1.0);
            state.push(value);
        }
        
        DVector::from_vec(state)
    }
}

fn main() -> Result<()> {
    println!("🚀 Starting Comprehensive AI Demo");
    println!("This demo showcases all advanced AI capabilities in the agent_evolution crate");
    println!();

    // Create environment generator
    let mut environment = EnvironmentGenerator::new();

    // Create multiple agents with different characteristics
    let mut agents = vec![
        ComprehensiveAgent::new("Explorer".to_string()),
        ComprehensiveAgent::new("Innovator".to_string()),
        ComprehensiveAgent::new("Learner".to_string()),
        ComprehensiveAgent::new("Creator".to_string()),
    ];

    // Register agents with meta-learner
    for agent in &mut agents {
        agent.meta_learner.register_core(agent.id);
    }

    println!("Created {} agents with full AI capabilities", agents.len());
    println!("Starting simulation for 100 cycles...");
    println!();

    // Run simulation
    let num_cycles = 100;
    let mut all_outputs = Vec::new();

    for cycle in 0..num_cycles {
        let environment_state = environment.generate_state(cycle);
        let mut cycle_outputs = Vec::new();

        // Process each agent
        for agent in &mut agents {
            match agent.process_cycle(&environment_state, cycle) {
                Ok(output) => cycle_outputs.push(output),
                Err(e) => eprintln!("Error processing agent {}: {}", agent.name, e),
            }
        }

        all_outputs.extend(cycle_outputs.clone());

        // Report progress every 10 cycles
        if cycle % 10 == 0 {
            report_progress(cycle, &agents, &cycle_outputs);
        }

        // Small delay to make output readable
        thread::sleep(Duration::from_millis(50));
    }

    println!("\n🎉 Simulation completed!");
    println!("Final statistics:");
    println!();

    // Report final statistics for each agent
    for agent in &agents {
        let stats = agent.get_statistics();
        report_agent_statistics(&stats);
        println!();
    }

    // Report overall statistics
    report_overall_statistics(&agents);

    println!("\n✨ Demo completed successfully!");
    println!("All AI capabilities demonstrated:");
    println!("  • Meta-learning with adaptive parameter tuning");
    println!("  • Hypernetwork for dynamic architecture generation");
    println!("  • Curiosity-driven learning and exploration");
    println!("  • Advanced self-modification capabilities");
    println!("  • Open-ended evolution with novelty detection");

    Ok(())
}

fn report_progress(cycle: u64, agents: &[ComprehensiveAgent], outputs: &[AgentCycleOutput]) {
    println!("Cycle {}: ", cycle);
    
    for output in outputs {
        println!("  {}: {} → {:.3} (curiosity: {:.3}, novelty: {})", 
            output.agent_name, 
            format!("{:?}", output.action_taken), 
            output.outcome,
            output.curiosity_level,
            output.novelty_detected
        );
    }
    println!();
}

fn report_agent_statistics(stats: &AgentStatistics) {
    println!("Agent: {}", stats.agent_name);
    println!("  Generation: {}", stats.generation);
    println!("  Discoveries: {}", stats.discovery_count);
    println!("  Average Performance: {:.3}", stats.average_performance);
    println!("  Recent Performance: {:.3}", stats.recent_performance);
    println!("  Curiosity Level: {:.3}", stats.curiosity_stats.current_curiosity_level);
    println!("  Self-Modifications: {}", stats.self_modification_stats.total_modifications);
    println!("  Novelty Detections: {}", stats.open_ended_stats.total_behaviors);
}

fn report_overall_statistics(agents: &[ComprehensiveAgent]) {
    let total_discoveries: usize = agents.iter().map(|a| a.discovery_count).sum();
    let avg_performance: f64 = agents.iter()
        .map(|a| a.performance_history.iter().sum::<f64>() / a.performance_history.len().max(1) as f64)
        .sum::<f64>() / agents.len() as f64;
    
    println!("Overall Statistics:");
    println!("  Total Discoveries: {}", total_discoveries);
    println!("  Average Performance: {:.3}", avg_performance);
    println!("  Agents: {}", agents.len());
} 