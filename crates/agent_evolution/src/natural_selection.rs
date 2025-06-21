//! # Agent Evolution: Natural Selection Module
//!
//! This module implements the core mechanism of natural selection, which drives
//! the evolutionary process. It evaluates the fitness of agents based on their
//! performance and determines their reproductive success.

use anyhow::Result;
use rand::prelude::SliceRandom;
use crate::genetics::Genome;
use std::collections::HashMap;
use uuid::Uuid;

/// Represents the fitness of an agent, a measure of its success.
#[derive(Debug, Clone, Copy, PartialEq, PartialOrd)]
pub struct Fitness(f64);

impl Fitness {
    /// Creates a new fitness score. The value should be non-negative.
    pub fn new(score: f64) -> Self {
        assert!(score >= 0.0, "Fitness score cannot be negative.");
        Fitness(score)
    }

    /// Returns the raw fitness score.
    pub fn score(&self) -> f64 {
        self.0
    }
}

/// A collection of fitness scores for a population of agents.
#[derive(Debug, Default)]
pub struct FitnessLandscape {
    scores: HashMap<Uuid, Fitness>,
}

impl FitnessLandscape {
    /// Creates a new, empty fitness landscape.
    pub fn new() -> Self {
        FitnessLandscape { scores: HashMap::new() }
    }

    /// Updates the fitness score for a given agent.
    pub fn update_fitness(&mut self, agent_id: Uuid, score: f64) {
        self.scores.insert(agent_id, Fitness::new(score));
    }

    /// Gets the fitness of a specific agent.
    pub fn get_fitness(&self, agent_id: &Uuid) -> Option<Fitness> {
        self.scores.get(agent_id).copied()
    }
}

/// Implements the selection process based on agent fitness.
pub struct Selection;

impl Selection {
    /// Selects agents for reproduction based on their fitness.
    ///
    /// This example uses tournament selection, a common method in genetic algorithms.
    /// Other methods, like roulette wheel selection, could also be implemented.
    ///
    /// # Arguments
    /// * `population` - A map of agent IDs to their genomes.
    /// * `fitness_landscape` - The fitness scores for the population.
    /// * `tournament_size` - The number of agents competing in each tournament.
    ///
    /// # Returns
    /// A vector of genomes from the "winning" agents, ready for reproduction.
    pub fn tournament_selection<'a>(
        population: &'a HashMap<Uuid, Genome>,
        fitness_landscape: &FitnessLandscape,
        tournament_size: usize,
    ) -> Vec<&'a Genome> {
        let mut winners = Vec::new();
        let mut rng = rand::thread_rng();
        let population_keys: Vec<_> = population.keys().collect();

        if population_keys.is_empty() {
            return winners;
        }

        // Run a series of tournaments to select the parents.
        for _ in 0..population.len() {
            let mut tournament_contestants = Vec::new();
            for _ in 0..tournament_size {
                if let Some(key) = population_keys.choose(&mut rng) {
                    tournament_contestants.push(*key);
                }
            }

            let winner = tournament_contestants.iter()
                .max_by(|a, b| {
                    let fitness_a = fitness_landscape.get_fitness(a).unwrap_or(Fitness(0.0));
                    let fitness_b = fitness_landscape.get_fitness(b).unwrap_or(Fitness(0.0));
                    fitness_a.partial_cmp(&fitness_b).unwrap_or(std::cmp::Ordering::Equal)
                });

            if let Some(winner_id) = winner {
                if let Some(genome) = population.get(winner_id) {
                    winners.push(genome);
                }
            }
        }
        winners
    }
}

/// Applies the process of natural selection to a population.
pub fn apply_selection<'a>(
    population: &'a HashMap<Uuid, Genome>,
    fitness_landscape: &FitnessLandscape,
) -> Result<Vec<&'a Genome>> {
    let tournament_size = 5; // Example tournament size
    let new_parent_pool = Selection::tournament_selection(population, fitness_landscape, tournament_size);
    Ok(new_parent_pool)
}