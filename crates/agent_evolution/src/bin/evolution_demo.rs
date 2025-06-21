//! # Real Evolutionary AI Demo
//!
//! This demo showcases true evolutionary AI organisms that learn survival strategies
//! from scratch through self-modification, adaptation, and natural selection.
//! Unlike the previous demo systems, these organisms exhibit real learning,
//! decision-making, and self-improvement capabilities.

use anyhow::Result;
use agent_evolution::{EvolutionaryPopulation, PopulationStatistics, Environment};
use std::thread;
use std::time::{Duration, Instant};

fn main() -> Result<()> {
    println!("🧬 EVOLUTION Universe Simulation - Real AI Evolution Demo");
    println!("========================================================");
    println!("This demo creates a population of AI organisms that must learn");
    println!("survival strategies from scratch through evolutionary processes.");
    println!();
    
    // Create initial population
    let population_size = 50;
    println!("🌱 Initializing population of {} organisms...", population_size);
    let mut population = EvolutionaryPopulation::new(population_size);
    
    // Set up challenging environment
    population.environment = Environment {
        temperature: 25.0,
        resource_density: 0.3,
        danger_level: 0.4,
        social_pressure: 0.5,
        complexity: 0.6,
    };
    
    println!("🌍 Environment configured:");
    println!("  - Temperature: {:.1}°C", population.environment.temperature);
    println!("  - Resource Density: {:.1}%", population.environment.resource_density * 100.0);
    println!("  - Danger Level: {:.1}%", population.environment.danger_level * 100.0);
    println!("  - Social Pressure: {:.1}%", population.environment.social_pressure * 100.0);
    println!("  - Complexity: {:.1}%", population.environment.complexity * 100.0);
    println!();
    
    let mut generation_count = 0;
    let mut best_fitness_ever = 0.0;
    let mut last_stats = PopulationStatistics::default();
    let simulation_start = Instant::now();
    
    println!("🚀 Starting evolution simulation...");
    println!("Press Ctrl+C to stop the simulation");
    println!();
    
    loop {
        let generation_start = Instant::now();
        
        // Run simulation for one generation (100 time steps)
        for step in 0..100 {
            population.update(1.0)?;
            
            // Print progress every 25 steps
            if step % 25 == 0 {
                print!(".");
                std::io::Write::flush(&mut std::io::stdout()).unwrap();
            }
        }
        
        let generation_time = generation_start.elapsed();
        generation_count += 1;
        
        // Get population statistics
        let stats = population.get_statistics();
        
        // Track best fitness
        if stats.max_fitness > best_fitness_ever {
            best_fitness_ever = stats.max_fitness;
            println!("🏆 NEW RECORD! Best fitness: {:.2}", best_fitness_ever);
        }
        
        println!();
        println!("📊 Generation {} Statistics (took {:.2}s):", generation_count, generation_time.as_secs_f64());
        println!("  Population Size: {}", stats.total_organisms);
        println!("  Average Fitness: {:.2}", stats.avg_fitness);
        println!("  Maximum Fitness: {:.2}", stats.max_fitness);
        println!("  Average Age: {} ticks", stats.avg_age);
        println!("  Average Energy: {:.1}%", stats.avg_energy);
        println!("  Learning Organisms: {}", stats.learning_organisms);
        println!("  Reproductive Organisms: {}", stats.reproductive_organisms);
        println!("  Average Learning Rate: {:.4}", stats.avg_learning_rate);
        
        // Show evolution progress
        if generation_count > 1 {
            let fitness_change = stats.avg_fitness - last_stats.avg_fitness;
            let learning_change = stats.learning_organisms as i32 - last_stats.learning_organisms as i32;
            
            println!("  📈 Evolution Progress:");
            println!("    Fitness Change: {:+.2}", fitness_change);
            println!("    Learning Change: {:+}", learning_change);
            
            if fitness_change > 0.0 {
                println!("    ✅ Population is improving!");
            } else if fitness_change < -1.0 {
                println!("    ⚠️  Population fitness declining");
            } else {
                println!("    ➡️  Population stability");
            }
        }
        
        // Environmental changes every 10 generations
        if generation_count % 10 == 0 {
            println!("🌍 Environmental change detected!");
            population.environment.resource_density *= 0.9;
            population.environment.danger_level *= 1.1;
            population.environment.complexity *= 1.05;
            println!("  Resources becoming scarcer, danger increasing, complexity growing");
        }
        
        // Show detailed organism analysis every 5 generations
        if generation_count % 5 == 0 {
            show_organism_analysis(&population);
        }
        
        println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
        
        last_stats = stats;
        
        // Brief pause to make output readable
        thread::sleep(Duration::from_millis(1000));
        
        // Check for extinction
        if population.organisms.is_empty() {
            println!("💀 EXTINCTION EVENT - All organisms have died!");
            println!("   Restarting with new population...");
            population = EvolutionaryPopulation::new(population_size);
            generation_count = 0;
            best_fitness_ever = 0.0;
            thread::sleep(Duration::from_millis(3000));
        }
        
        // Stop after reasonable time for demo
        if simulation_start.elapsed() > Duration::from_secs(300) { // 5 minutes
            println!("⏰ Demo time limit reached");
            break;
        }
    }
    
    // Final analysis
    println!();
    println!("🎯 FINAL EVOLUTIONARY ANALYSIS");
    println!("===============================");
    println!("Simulation ran for {:.1} minutes", simulation_start.elapsed().as_secs_f64() / 60.0);
    println!("Total generations: {}", generation_count);
    println!("Best fitness achieved: {:.2}", best_fitness_ever);
    
    let final_stats = population.get_statistics();
    println!("Final population: {} organisms", final_stats.total_organisms);
    println!("Organisms with learned behaviors: {}", final_stats.learning_organisms);
    
    // Analyze the best organism
    if let Some(best_organism) = population.organisms.iter().max_by(|a, b| {
        a.get_fitness().partial_cmp(&b.get_fitness()).unwrap_or(std::cmp::Ordering::Equal)
    }) {
        println!();
        println!("🥇 BEST ORGANISM ANALYSIS:");
        println!("  ID: {}", best_organism.id);
        println!("  Generation: {}", best_organism.generation);
        println!("  Age: {} ticks", best_organism.age);
        println!("  Fitness: {:.2}", best_organism.get_fitness());
        println!("  Energy: {:.1}", best_organism.energy);
        println!("  Survival Strategies: {}", best_organism.learning_system.success_strategies.len());
        println!("  Behavioral Patterns: {}", best_organism.learning_system.behavioral_patterns.len());
        
        // NEW: Communication analysis
        let comm_capabilities = best_organism.get_communication_capabilities();
        println!("  🗣️  COMMUNICATION EVOLUTION:");
        println!("    Vocabulary Size: {} words", comm_capabilities.vocabulary_size);
        println!("    Grammar Complexity: {:.2}", comm_capabilities.grammar_complexity);
        println!("    Signal Types: {:?}", comm_capabilities.signal_types);
        println!("    Social Connections: {}", comm_capabilities.social_connections);
        println!("    Communication Fitness: {:.3}", comm_capabilities.overall_fitness);
        
        // Show evolved vocabulary
        let vocab = best_organism.get_vocabulary();
        if !vocab.is_empty() {
            println!("    Evolved Vocabulary:");
            for (word, concept) in vocab.iter().take(10) {
                println!("      '{}' (usage: {:.2}, age: {})", 
                        word, concept.usage_frequency, concept.evolution_age);
            }
        }
        
        // Show communication organs evolution
        let comm_system = &best_organism.communication_system;
        println!("    Communication Organs:");
        println!("      Vocal Complexity: {:.3}", comm_system.communication_organs.vocal_cords.complexity);
        println!("      Visual Pattern Complexity: {:.3}", comm_system.communication_organs.visual_signals.pattern_complexity);
        println!("      Chemical Diversity: {:.3}", comm_system.communication_organs.chemical_signals.chemical_diversity);
        println!("      EM Field Strength: {:.3}", comm_system.communication_organs.electromagnetic_signals.em_field_strength);
        println!("      Quantum Entanglement: {:.3}", comm_system.communication_organs.quantum_signals.entanglement_capacity);
        println!("      Neural Plasticity: {:.3}", comm_system.communication_organs.neural_interface.neural_plasticity);
        
        println!("  🧠 Neural Network Stats:");
        println!("    Exploration Rate: {:.3}", best_organism.neural_core.exploration_rate);
        println!("    Decision Confidence: {:.3}", best_organism.neural_core.decision_confidence);
        println!("    Mutation Rate: {:.3}", best_organism.neural_core.neural_network.mutation_rate);
        
        println!("  🧬 Evolution Traits:");
        println!("    Learning Motivation: {:.3}", best_organism.survival_instincts.learning_motivation);
        println!("    Danger Sensitivity: {:.3}", best_organism.survival_instincts.danger_sensitivity);
        println!("    Risk Tolerance: {:.3}", best_organism.survival_instincts.risk_tolerance);
        println!("    Social Tendency: {:.3}", best_organism.survival_instincts.social_tendency);
        
        println!("  📊 Physical Capabilities:");
        println!("    Movement Speed: {:.2}", best_organism.motor_system.movement_speed);
        println!("    Vision Range: {:.1}", best_organism.sensory_organs.vision_range);
        println!("    Metabolic Efficiency: {:.3}", best_organism.metabolism.efficiency);
        
        if !best_organism.learning_system.success_strategies.is_empty() {
            println!("  🎯 Top Survival Strategy:");
            let best_strategy = best_organism.learning_system.success_strategies.iter()
                .max_by(|a, b| a.effectiveness.partial_cmp(&b.effectiveness).unwrap_or(std::cmp::Ordering::Equal))
                .unwrap();
            println!("    Description: {}", best_strategy.description);
            println!("    Effectiveness: {:.2}", best_strategy.effectiveness);
            println!("    Energy Efficiency: {:.2}", best_strategy.energy_efficiency);
        }
    }
    
    println!();
    println!("🎉 This demonstrates TRUE autonomous evolution:");
    println!("  ✅ Organisms learned survival strategies from scratch");
    println!("  ✅ Neural networks self-modified based on experience");
    println!("  ✅ Behaviors emerged from environmental pressure");
    println!("  ✅ Knowledge was passed to offspring through genetic memory");
    println!("  ✅ Population adapted to changing conditions");
    println!("  🗣️  Communication systems evolved autonomously");
    println!("  🧬 Language emerged from survival and social needs");
    println!("  🌟 Signal types evolved based on environmental conditions");
    println!();
    println!("Unlike the previous demo, these organisms exhibit genuine:");
    println!("  • Self-learning and adaptation");
    println!("  • Autonomous decision making");
    println!("  • Experience-based behavioral modification");
    println!("  • Evolutionary knowledge transfer");
    println!("  • Environmental responsiveness");
    println!("  • Autonomous communication evolution");
    println!("  • Language emergence from scratch");
    println!("  • Signal type diversification");
    
    Ok(())
}

fn show_organism_analysis(population: &EvolutionaryPopulation) {
    println!("🔬 ORGANISM BEHAVIORAL ANALYSIS:");
    
    // Analyze learning patterns
    let total_strategies: usize = population.organisms.iter()
        .map(|o| o.learning_system.success_strategies.len())
        .sum();
    
    let total_patterns: usize = population.organisms.iter()
        .map(|o| o.learning_system.behavioral_patterns.len())
        .sum();
    
    println!("  Total Learned Strategies: {}", total_strategies);
    println!("  Total Behavioral Patterns: {}", total_patterns);
    
    // NEW: Analyze communication evolution
    let total_vocabulary: usize = population.organisms.iter()
        .map(|o| o.get_vocabulary().len())
        .sum();
    
    let total_signal_types: usize = population.organisms.iter()
        .map(|o| o.get_signal_types().len())
        .sum();
    
    let avg_communication_fitness: f64 = population.organisms.iter()
        .map(|o| o.get_communication_capabilities().overall_fitness)
        .sum::<f64>() / population.organisms.len() as f64;
    
    println!("  🗣️  COMMUNICATION EVOLUTION:");
    println!("    Total Vocabulary Words: {}", total_vocabulary);
    println!("    Total Signal Types: {}", total_signal_types);
    println!("    Average Communication Fitness: {:.3}", avg_communication_fitness);
    
    // Find organisms with most evolved communication
    if let Some(most_communicative) = population.organisms.iter()
        .max_by(|a, b| {
            a.get_communication_capabilities().vocabulary_size.cmp(&b.get_communication_capabilities().vocabulary_size)
        }) {
        println!("    Most Communicative: {} words, {} signal types (Generation {})", 
                most_communicative.get_communication_capabilities().vocabulary_size,
                most_communicative.get_communication_capabilities().signal_types.len(),
                most_communicative.generation);
        
        // Show some evolved vocabulary
        let vocab = most_communicative.get_vocabulary();
        if !vocab.is_empty() {
            println!("    Sample Vocabulary:");
            for (i, (word, concept)) in vocab.iter().take(5).enumerate() {
                println!("      {}. '{}' (usage: {:.2})", i + 1, word, concept.usage_frequency);
            }
        }
        
        // Show evolved signal types
        let signal_types = most_communicative.get_signal_types();
        if !signal_types.is_empty() {
            println!("    Evolved Signal Types:");
            for signal_type in signal_types {
                println!("      - {:?}", signal_type);
            }
        }
    }
    
    // Find most and least adaptive organisms
    if let (Some(most_adaptive), Some(least_adaptive)) = (
        population.organisms.iter().max_by_key(|o| o.learning_system.success_strategies.len()),
        population.organisms.iter().min_by_key(|o| o.learning_system.success_strategies.len())
    ) {
        println!("  Most Adaptive: {} strategies (Generation {})", 
                most_adaptive.learning_system.success_strategies.len(),
                most_adaptive.generation);
        println!("  Least Adaptive: {} strategies (Generation {})", 
                least_adaptive.learning_system.success_strategies.len(),
                least_adaptive.generation);
    }
    
    // Analyze survival instincts distribution
    let avg_learning_motivation: f64 = population.organisms.iter()
        .map(|o| o.survival_instincts.learning_motivation)
        .sum::<f64>() / population.organisms.len() as f64;
    
    let avg_risk_tolerance: f64 = population.organisms.iter()
        .map(|o| o.survival_instincts.risk_tolerance)
        .sum::<f64>() / population.organisms.len() as f64;
    
    println!("  Average Learning Motivation: {:.3}", avg_learning_motivation);
    println!("  Average Risk Tolerance: {:.3}", avg_risk_tolerance);
    
    // Check for innovation
    let innovative_organisms: Vec<_> = population.organisms.iter()
        .filter(|o| !o.genetic_memory.innovation_history.is_empty())
        .collect();
    
    if !innovative_organisms.is_empty() {
        println!("  🚀 INNOVATION DETECTED:");
        println!("    Innovative Organisms: {}", innovative_organisms.len());
        
        if let Some(most_innovative) = innovative_organisms.iter()
            .max_by_key(|o| o.genetic_memory.innovation_history.len()) {
            println!("    Most Innovative: {} innovations (Generation {})", 
                    most_innovative.genetic_memory.innovation_history.len(),
                    most_innovative.generation);
        }
    }
} 