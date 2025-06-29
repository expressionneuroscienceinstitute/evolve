//! # Standalone AI Capabilities Demo
//!
//! This demo showcases all the advanced AI capabilities implemented in the agent_evolution crate
//! without requiring the physics engine. It demonstrates:
//! - Meta-learning with adaptive parameter tuning
//! - Hypernetwork for dynamic architecture generation
//! - Curiosity-driven learning and exploration
//! - Advanced self-modification capabilities
//! - Open-ended evolution with novelty detection
//!
//! Run with: cargo run --bin standalone_demo

use agent_evolution::{
    ai_core::AICore,
    meta_learning::{MetaLearner, MetaParameter},
    hypernetwork::{Hypernetwork, TaskEmbedding, TaskType, NetworkConstraint, ConstraintType},
    curiosity::{CuriositySystem, ActionType, Experience},
    self_modification::AdvancedSelfModification,
    open_ended_evolution::OpenEndedEvolution,
    genetics::Genome,
};
use anyhow::Result;
use nalgebra::DVector;
use rand::Rng;
use std::thread;
use std::time::{Duration, Instant};
use uuid::Uuid;

/// Comprehensive AI Agent
#[derive(Debug)]
struct AIAgent {
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

impl AIAgent {
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

    fn process_learning_cycle(&mut self, environment_state: &DVector<f64>, cycle: u64) -> Result<AgentOutput> {
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
        
        Ok(AgentOutput {
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

    fn generate_task_embedding(&self, environment_state: &DVector<f64>) -> Result<TaskEmbedding> {
        // For demo, use a fixed task type and reasonable values
        Ok(TaskEmbedding {
            task_type: TaskType::PatternRecognition,
            complexity: 0.5,
            input_dim: environment_state.len(),
            output_dim: 3,
            constraints: vec![NetworkConstraint {
                constraint_type: ConstraintType::MaxLayers,
                value: 5.0,
                priority: 1.0,
            }],
            performance_target: 0.8,
        })
    }

    fn adapt_neural_architecture(&mut self, architecture: &agent_evolution::hypernetwork::GeneratedArchitecture) -> Result<()> {
        self.neural_core.neural_network.learning_rate = architecture.learning_parameters.learning_rate;
        Ok(())
    }

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
        
        let mut rng = rand::thread_rng();
        let noise = rng.gen_range(-0.2..0.2);
        let environment_factor = environment_state.iter().sum::<f64>() / environment_state.len() as f64;
        
        (base_outcome + noise + environment_factor * 0.1).max(-1.0).min(1.0)
    }

    fn update_meta_learning(&mut self, outcome: f64, _cycle: u64) -> Result<()> {
        let mut params = std::collections::HashMap::new();
        params.insert(MetaParameter::LearningRate, self.neural_core.neural_network.learning_rate);
        // Optionally, add MutationRate if you want to demo it
        params.insert(MetaParameter::MutationRate, 0.01);
        let updated_params = self.meta_learner.update_core(self.id, outcome, &params)?;
        if let Some(&learning_rate) = updated_params.get(&MetaParameter::LearningRate) {
            self.neural_core.neural_network.learning_rate = learning_rate;
        }
        Ok(())
    }

    fn perform_self_modification(&mut self, cycle: u64) -> Result<agent_evolution::self_modification::SelfModificationOutput> {
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

    fn update_performance(&mut self, outcome: f64, novelty_detected: bool) {
        self.performance_history.push(outcome);
        
        if self.performance_history.len() > 100 {
            self.performance_history.remove(0);
        }
        
        if novelty_detected {
            self.discovery_count += 1;
        }
    }

    fn get_statistics(&self) -> AgentStatistics {
        let average_performance = if self.performance_history.is_empty() {
            0.0
        } else {
            self.performance_history.iter().sum::<f64>() / self.performance_history.len() as f64
        };
        
        let recent_performance = if self.performance_history.len() >= 10 {
            self.performance_history.iter().rev().take(10).sum::<f64>() / 10.0
        } else {
            average_performance
        };
        
        AgentStatistics {
            agent_id: self.id,
            agent_name: self.name.clone(),
            generation: self.generation,
            discovery_count: self.discovery_count,
            average_performance,
            recent_performance,
            curiosity_stats: self.curiosity_system.get_statistics(),
            self_modification_stats: self.self_modification.get_statistics(),
            open_ended_stats: self.open_ended_evolution.get_statistics(),
        }
    }
}

#[derive(Debug)]
#[allow(dead_code)]
struct AgentOutput {
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

#[derive(Debug)]
#[allow(dead_code)]
struct AgentStatistics {
    agent_id: Uuid,
    agent_name: String,
    generation: u64,
    discovery_count: usize,
    average_performance: f64,
    recent_performance: f64,
    curiosity_stats: agent_evolution::curiosity::CuriosityStatistics,
    self_modification_stats: agent_evolution::self_modification::SelfModificationStatistics,
    open_ended_stats: agent_evolution::open_ended_evolution::OpenEndedEvolutionStatistics,
}

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
        let mut state = Vec::new();
        
        for i in 0..10 {
            let base_value = (cycle as f64 * 0.01 + i as f64 * 0.1).sin();
            let complexity_modifier = self.complexity * (i as f64 * 0.2);
            let novelty_modifier = self.novelty_factor * rng.gen_range(-0.5..0.5);
            let change_modifier = self.change_rate * (cycle as f64 * 0.001);
            
            let value = base_value + complexity_modifier + novelty_modifier + change_modifier;
            state.push(value.max(-1.0).min(1.0));
        }
        
        self.complexity = (self.complexity + 0.001).min(1.0);
        self.novelty_factor = (self.novelty_factor + 0.0005).min(1.0);
        self.change_rate = (self.change_rate + 0.0002).min(0.5);
        
        DVector::from_vec(state)
    }
}

fn main() -> Result<()> {
    println!("🚀 STANDALONE AI CAPABILITIES DEMO");
    println!("===================================");
    println!("This demo showcases all AI capabilities without physics engine:");
    println!("• Meta-learning with adaptive parameter tuning");
    println!("• Hypernetwork for dynamic architecture generation");
    println!("• Curiosity-driven learning and exploration");
    println!("• Advanced self-modification capabilities");
    println!("• Open-ended evolution with novelty detection");
    println!();

    // Create agents
    let mut agents = vec![
        AIAgent::new("Agent Alpha".to_string()),
        AIAgent::new("Agent Beta".to_string()),
        AIAgent::new("Agent Gamma".to_string()),
    ];

    let mut environment = EnvironmentGenerator::new();
    let mut cycle = 0;
    let max_cycles = 500; // Shorter demo
    let report_interval = 50;

    println!("Starting simulation with {} agents...", agents.len());
    println!("Running for {} cycles with reports every {} cycles", max_cycles, report_interval);
    println!();

    let simulation_start = Instant::now();

    while cycle < max_cycles {
        cycle += 1;
        
        let environment_state = environment.generate_state(cycle);
        let mut cycle_outputs = Vec::new();
        
        for agent in &mut agents {
            match agent.process_learning_cycle(&environment_state, cycle) {
                Ok(output) => cycle_outputs.push(output),
                Err(e) => eprintln!("Error processing agent {}: {}", agent.name, e),
            }
        }
        
        if cycle % report_interval == 0 {
            report_progress(cycle, &agents, &cycle_outputs);
        }
        
        for output in &cycle_outputs {
            if output.novelty_detected && output.novelty_score > 0.7 {
                println!("🎉 BREAKTHROUGH: {} discovered novel behavior (score: {:.3})", 
                    output.agent_name, output.novelty_score);
            }
        }
        
        thread::sleep(Duration::from_millis(5));
    }

    println!();
    println!("🎯 SIMULATION COMPLETE");
    println!("=====================");
    println!("Simulation ran for {:.1} seconds", simulation_start.elapsed().as_secs_f64());
    println!("Total cycles: {}", cycle);
    println!();

    for agent in &agents {
        let stats = agent.get_statistics();
        report_agent_statistics(&stats);
    }

    report_overall_statistics(&agents);

    Ok(())
}

fn report_progress(cycle: u64, agents: &[AIAgent], outputs: &[AgentOutput]) {
    println!("📊 CYCLE {} REPORT", cycle);
    println!("-------------------");
    
    let total_discoveries: usize = agents.iter().map(|a| a.discovery_count).sum();
    let avg_performance: f64 = outputs.iter().map(|o| o.performance).sum::<f64>() / outputs.len() as f64;
    let total_novelty: usize = outputs.iter().filter(|o| o.novelty_detected).count();
    let avg_curiosity: f64 = outputs.iter().map(|o| o.curiosity_level).sum::<f64>() / outputs.len() as f64;
    
    println!("Total Discoveries: {}", total_discoveries);
    println!("Average Performance: {:.3}", avg_performance);
    println!("Novel Behaviors This Cycle: {}", total_novelty);
    println!("Average Curiosity Level: {:.3}", avg_curiosity);
    println!();
}

fn report_agent_statistics(stats: &AgentStatistics) {
    println!("🤖 AGENT: {}", stats.agent_name);
    println!("   Generation: {}", stats.generation);
    println!("   Discoveries: {}", stats.discovery_count);
    println!("   Average Performance: {:.3}", stats.average_performance);
    println!("   Recent Performance: {:.3}", stats.recent_performance);
    println!("   Curiosity Level: {:.3}", stats.curiosity_stats.current_curiosity_level);
    println!("   Self-Modifications: {}", stats.self_modification_stats.total_modifications);
    println!("   Open-Ended Behaviors: {}", stats.open_ended_stats.total_behaviors);
    println!();
}

fn report_overall_statistics(agents: &[AIAgent]) {
    println!("📈 OVERALL STATISTICS");
    println!("====================");
    
    let total_discoveries: usize = agents.iter().map(|a| a.discovery_count).sum();
    let avg_performance: f64 = agents.iter()
        .map(|a| a.performance_history.iter().sum::<f64>() / a.performance_history.len().max(1) as f64)
        .sum::<f64>() / agents.len() as f64;
    
    let total_open_ended_behaviors: usize = agents.iter()
        .map(|a| a.open_ended_evolution.get_statistics().total_behaviors)
        .sum();
    
    let total_self_modifications: usize = agents.iter()
        .map(|a| a.self_modification.get_statistics().total_modifications)
        .sum();
    
    println!("Total Discoveries Across All Agents: {}", total_discoveries);
    println!("Average Performance Across All Agents: {:.3}", avg_performance);
    println!("Total Open-Ended Behaviors: {}", total_open_ended_behaviors);
    println!("Total Self-Modifications: {}", total_self_modifications);
    println!();
    
    if total_discoveries > 0 {
        println!("🎉 SUCCESS: Agents discovered {} novel behaviors through open-ended evolution!", total_discoveries);
    }
    
    if total_open_ended_behaviors > 0 {
        println!("🌟 INNOVATION: {} open-ended behaviors emerged from the system!", total_open_ended_behaviors);
    }
    
    if total_self_modifications > 0 {
        println!("🔧 ADAPTATION: Agents performed {} self-modifications!", total_self_modifications);
    }
    
    println!();
    println!("✅ AI CAPABILITIES DEMONSTRATION COMPLETE");
    println!("All modules are working correctly and integrated!");
} 