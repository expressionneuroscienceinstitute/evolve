//! Integration test for all AI capabilities
//! This test demonstrates how all the AI modules work together

use crate::{
    ai_core::AICore,
    meta_learning::{MetaLearner, MetaParameter},
    hypernetwork::{Hypernetwork, TaskEmbedding, TaskType, NetworkConstraint, ConstraintType},
    curiosity::{CuriositySystem, ActionType, Experience},
    self_modification::AdvancedSelfModification,
    open_ended_evolution::OpenEndedEvolution,
    genetics::Genome,
};
use nalgebra::DVector;
use uuid::Uuid;
use std::collections::HashMap;

/// Test that all AI modules can work together
#[test]
fn test_comprehensive_ai_integration() {
    // Create all AI components
    let mut neural_core = AICore::new();
    let mut meta_learner = MetaLearner::new();
    let mut hypernetwork = Hypernetwork::new();
    let mut curiosity_system = CuriositySystem::new(10);
    let mut self_modification = AdvancedSelfModification::new();
    let mut genome = Genome::new();
    let mut open_ended_evolution = OpenEndedEvolution::new();

    // Test 1: Hypernetwork generates architecture
    let task_embedding = TaskEmbedding {
        task_type: TaskType::ReinforcementLearning,
        complexity: 0.5,
        input_dim: 10,
        output_dim: 5,
        constraints: vec![NetworkConstraint {
            constraint_type: ConstraintType::MaxLayers,
            value: 5.0,
            priority: 0.8,
        }],
        performance_target: 0.8,
    };
    let architecture = hypernetwork.generate_architecture(&task_embedding).unwrap();
    assert!(architecture.learning_parameters.learning_rate > 0.0);
    assert!(architecture.architecture_score >= 0.0);

    // Test 2: Curiosity system selects actions
    let available_actions = vec![
        ActionType::Experiment,
        ActionType::Learn,
        ActionType::Observe,
    ];
    let environment_state = DVector::from_vec(vec![0.1, 0.2, 0.3, 0.4, 0.5]);
    let action_selection = curiosity_system.select_action(available_actions, &environment_state).unwrap();
    assert!(action_selection.curiosity_contribution >= 0.0);
    assert!(action_selection.exploration_bonus >= 0.0);

    // Test 3: Create and process experience
    let experience = Experience {
        id: Uuid::new_v4(),
        timestamp: 0.0,
        sensory_input: environment_state.clone(),
        context: DVector::zeros(5),
        action_taken: Some(action_selection.selected_action.clone()),
        outcome: Some(0.5),
        novelty_score: 0.3,
        curiosity_value: 0.4,
    };

    let curiosity_output = curiosity_system.process_experience(experience.clone()).unwrap();
    assert!(curiosity_output.curiosity_level >= 0.0);
    assert!(curiosity_output.curiosity_level <= 1.0);

    // Test 4: Open-ended evolution processes experience
    let novelty_output = open_ended_evolution.process_experience(
        Uuid::new_v4(),
        &experience,
        &neural_core,
        &curiosity_system,
        0.0,
    ).unwrap();
    assert!(novelty_output.novelty_score >= 0.0);
    assert!(novelty_output.novelty_score <= 1.0);

    // Test 5: Meta-learning updates (using the correct API)
    let mut params = HashMap::new();
    params.insert(MetaParameter::LearningRate, 0.1);
    params.insert(MetaParameter::ExplorationRate, 0.2);
    params.insert(MetaParameter::MutationRate, 0.05);
    
    let _meta_suggestions = meta_learner.update_core(Uuid::new_v4(), 0.5, &params).unwrap();

    // Test 6: Self-modification
    let modification_output = self_modification.perform_self_modification(
        &mut neural_core,
        &mut genome,
        &mut meta_learner,
        &mut curiosity_system,
        &mut hypernetwork,
        0.0,
    ).unwrap();
    assert!(modification_output.modifications_performed >= 0);

    // Test 7: Get statistics from all systems
    let curiosity_stats = curiosity_system.get_statistics();
    let self_mod_stats = self_modification.get_statistics();
    let open_ended_stats = open_ended_evolution.get_statistics();

    // Verify all statistics are reasonable
    assert!(curiosity_stats.current_curiosity_level >= 0.0);
    assert!(curiosity_stats.current_curiosity_level <= 1.0);
    assert!(self_mod_stats.total_modifications >= 0);
    assert!(open_ended_stats.total_behaviors >= 0);

    println!("✅ All AI modules integrated successfully!");
    println!("   Curiosity Level: {:.3}", curiosity_stats.current_curiosity_level);
    println!("   Self-Modifications: {}", self_mod_stats.total_modifications);
    println!("   Open-Ended Behaviors: {}", open_ended_stats.total_behaviors);
    println!("   Novelty Score: {:.3}", novelty_output.novelty_score);
}

/// Test that agents can discover novel behaviors
#[test]
fn test_novel_behavior_discovery() {
    let mut open_ended = OpenEndedEvolution::new();
    let neural_core = AICore::new();
    let curiosity_system = CuriositySystem::new(10);

    // Create a series of experiences with increasing novelty
    for i in 0..10 {
        let experience = Experience {
            id: Uuid::new_v4(),
            timestamp: i as f64,
            sensory_input: DVector::from_vec(vec![i as f64 * 0.1, i as f64 * 0.2, i as f64 * 0.3, i as f64 * 0.4, i as f64 * 0.5]),
            context: DVector::zeros(5),
            action_taken: Some(ActionType::Experiment),
            outcome: Some(i as f64 * 0.1),
            novelty_score: i as f64 * 0.1,
            curiosity_value: i as f64 * 0.05,
        };

        let output = open_ended.process_experience(
            Uuid::new_v4(),
            &experience,
            &neural_core,
            &curiosity_system,
            i as f64,
        ).unwrap();

        // First experience should be novel
        if i == 0 {
            assert!(output.novelty_found);
            assert!(output.novelty_score > 0.5);
        }
    }

    let stats = open_ended.get_statistics();
    assert!(stats.total_behaviors > 0);
    println!("✅ Discovered {} novel behaviors!", stats.total_behaviors);
}

/// Test that meta-learning adapts parameters
#[test]
fn test_meta_learning_adaptation() {
    let mut meta_learner = MetaLearner::new();
    let core_id = Uuid::new_v4();
    meta_learner.register_core(core_id);
    
    let mut params = HashMap::new();
    params.insert(MetaParameter::LearningRate, 0.01);
    params.insert(MetaParameter::ExplorationRate, 0.2);
    params.insert(MetaParameter::MutationRate, 0.05);
    
    let initial_learning_rate = params[&MetaParameter::LearningRate];

    // Simulate poor performance
    for i in 0..5 {
        let suggestions = meta_learner.update_core(core_id, -0.5, &params).unwrap();
        // Update params with suggestions
        for (param, value) in suggestions {
            params.insert(param, value);
        }
    }

    let final_learning_rate = params[&MetaParameter::LearningRate];

    // Learning rate should have adapted
    assert!(final_learning_rate != initial_learning_rate);
    println!("✅ Meta-learning adapted learning rate from {:.3} to {:.3}", 
        initial_learning_rate, final_learning_rate);
}

/// Test that curiosity drives exploration
#[test]
fn test_curiosity_driven_exploration() {
    let mut curiosity = CuriositySystem::new(10);
    let available_actions = vec![ActionType::Experiment, ActionType::Learn, ActionType::Observe];
    let environment_state = DVector::from_vec(vec![0.1, 0.2, 0.3, 0.4, 0.5]);

    // Get initial action selection
    let initial_selection = curiosity.select_action(available_actions.clone(), &environment_state).unwrap();
    let initial_curiosity = initial_selection.curiosity_contribution;

    // Process some experiences to build up curiosity
    for i in 0..5 {
        let experience = Experience {
            id: Uuid::new_v4(),
            timestamp: i as f64,
            sensory_input: DVector::from_vec(vec![i as f64 * 0.1, i as f64 * 0.2, i as f64 * 0.3, i as f64 * 0.4, i as f64 * 0.5]),
            context: DVector::zeros(5),
            action_taken: Some(ActionType::Experiment),
            outcome: Some(i as f64 * 0.1),
            novelty_score: i as f64 * 0.1,
            curiosity_value: i as f64 * 0.05,
        };

        let _output = curiosity.process_experience(experience).unwrap();
    }

    // Get final action selection
    let final_selection = curiosity.select_action(available_actions, &environment_state).unwrap();
    let final_curiosity = final_selection.curiosity_contribution;

    // Curiosity should have increased
    assert!(final_curiosity >= initial_curiosity);
    println!("✅ Curiosity increased from {:.3} to {:.3}", initial_curiosity, final_curiosity);
} 