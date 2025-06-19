//! # Agent Evolution: Genetics Module
//!
//! This module provides the foundational structures and functions for representing
//! and manipulating the genetic makeup of agents in the simulation. It includes
//! representations for DNA, genes, and genomes, along with mechanisms for mutation.
//!
//! The design is influenced by principles of molecular biology, providing a robust
//! and extensible framework for evolving complex agent behaviors.

use anyhow::Result;
use rand::prelude::*;
use serde::{Deserialize, Serialize};

/// Represents a single nucleotide in a DNA sequence.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Nucleotide {
    A, // Adenine
    C, // Cytosine
    G, // Guanine
    T, // Thymine
}

impl Nucleotide {
    /// Returns a random nucleotide.
    pub fn random<R: Rng + ?Sized>(rng: &mut R) -> Self {
        match rng.gen_range(0..4) {
            0 => Nucleotide::A,
            1 => Nucleotide::C,
            2 => Nucleotide::G,
            _ => Nucleotide::T,
        }
    }
}

/// Represents a DNA sequence, composed of nucleotides.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Dna {
    sequence: Vec<Nucleotide>,
}

impl Dna {
    /// Creates a new random DNA sequence of a given length.
    pub fn new_random<R: Rng + ?Sized>(rng: &mut R, length: usize) -> Self {
        let sequence = (0..length).map(|_| Nucleotide::random(rng)).collect();
        Dna { sequence }
    }

    /// Mutates the DNA sequence based on a given mutation rate.
    ///
    /// This function introduces point mutations (substitutions) in the DNA sequence.
    /// Other types of mutations (insertions, deletions, etc.) could be added here.
    pub fn mutate<R: Rng + ?Sized>(&mut self, rng: &mut R, mutation_rate: f64) {
        for nucleotide in &mut self.sequence {
            if rng.gen_bool(mutation_rate) {
                *nucleotide = Nucleotide::random(rng);
            }
        }
    }
}

/// Represents a gene, which is a segment of DNA that codes for a specific trait.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Gene {
    pub id: String,
    pub dna: Dna,
    pub expression_level: f64,  // 0.0 to 1.0 indicating how actively this gene is expressed
}

impl Gene {
    /// Creates a new gene with a random DNA sequence.
    pub fn new_random<R: Rng + ?Sized>(rng: &mut R, id: &str, dna_length: usize) -> Self {
        Gene {
            id: id.to_string(),
            dna: Dna::new_random(rng, dna_length),
            expression_level: rng.gen_range(0.0..1.0),  // Random initial expression level
        }
    }
}

/// Represents the complete set of genes for an agent.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Genome {
    pub genes: Vec<Gene>,
}

impl Default for Genome {
    fn default() -> Self {
        Self::new()
    }
}

impl Genome {
    /// Creates a new empty genome.
    pub fn new() -> Self {
        Genome { genes: vec![] }
    }

    /// Adds a gene to the genome.
    pub fn add_gene(&mut self, gene: Gene) {
        self.genes.push(gene);
    }

    /// Mutates the entire genome.
    pub fn mutate<R: Rng + ?Sized>(&mut self, rng: &mut R, mutation_rate: f64) {
        for gene in &mut self.genes {
            gene.dna.mutate(rng, mutation_rate);
        }
    }
}

/// A placeholder function to demonstrate a genetic operation.
/// In a real scenario, this would involve more complex processes like crossover and selection.
pub fn mutate_genome(genome: &mut Genome, mutation_rate: f64) -> Result<()> {
    let mut rng = thread_rng();
    genome.mutate(&mut rng, mutation_rate);
    Ok(())
}