//! # Embodied Agent Demo
//!
//! This demo shows how AI agents are physically embodied in a simplified
//! physics environment, demonstrating their ability to interact with
//! the physical world through particle-based bodies.

use agent_evolution::{
    embodied_agent::{EmbodiedAgent, PhysicsEngineInterface, ParticleData},
    curiosity::ActionType,
};
use nalgebra::Vector3;
use anyhow::Result;
use uuid::Uuid;

/// Simplified physics engine for demo
#[derive(Debug)]
struct SimplePhysicsEngine {
    particles: Vec<SimpleParticle>,
    time: f64,
}

#[derive(Debug, Clone)]
struct SimpleParticle {
    position: Vector3<f64>,
    velocity: Vector3<f64>,
    energy: f64,
    mass: f64,
    electric_charge: f64,
}

impl SimplePhysicsEngine {
    fn new() -> Self {
        Self {
            particles: Vec::new(),
            time: 0.0,
        }
    }
    
    fn add_particle(&mut self, position: Vector3<f64>, velocity: Vector3<f64>, mass: f64, charge: f64) -> usize {
        let particle = SimpleParticle {
            position,
            velocity,
            energy: 0.5 * mass * velocity.magnitude_squared(),
            mass,
            electric_charge: charge,
        };
        self.particles.push(particle);
        self.particles.len() - 1
    }
    
    fn step(&mut self, dt: f64) {
        self.time += dt;
        
        // Simple physics update
        for particle in &mut self.particles {
            // Update position
            particle.position += particle.velocity * dt;
            
            // Update energy
            particle.energy = 0.5 * particle.mass * particle.velocity.magnitude_squared();
        }
    }
    
    fn get_particle_data(&self, index: usize) -> Option<ParticleData> {
        if index < self.particles.len() {
            let particle = &self.particles[index];
            Some(ParticleData {
                position: particle.position,
                velocity: particle.velocity,
                energy: particle.energy,
                mass: particle.mass,
                electric_charge: particle.electric_charge,
            })
        } else {
            None
        }
    }
}

impl PhysicsEngineInterface for SimplePhysicsEngine {
    fn get_particle_count(&self) -> usize {
        self.particles.len()
    }
    
    fn get_particle_data(&self, index: usize) -> Option<ParticleData> {
        self.get_particle_data(index)
    }
    
    fn apply_force_to_particle(&mut self, index: usize, force: Vector3<f64>) -> Result<()> {
        if index < self.particles.len() {
            let particle = &mut self.particles[index];
            let acceleration = force / particle.mass;
            particle.velocity += acceleration * 1e-12; // Small timestep
        }
        Ok(())
    }
    
    fn transfer_energy(&mut self, from: usize, to: usize, amount: f64) -> Result<()> {
        if from < self.particles.len() && to < self.particles.len() {
            if self.particles[from].energy >= amount {
                self.particles[from].energy -= amount;
                self.particles[to].energy += amount;
            }
        }
        Ok(())
    }
    
    fn create_particle(&mut self, position: Vector3<f64>, velocity: Vector3<f64>, mass: f64, charge: f64) -> Result<usize> {
        Ok(self.add_particle(position, velocity, mass, charge))
    }
}

/// Demo simulation with embodied agents
#[derive(Debug)]
struct EmbodiedSimulation {
    physics_engine: SimplePhysicsEngine,
    agents: Vec<EmbodiedAgent>,
    time: f64,
}

impl EmbodiedSimulation {
    fn new() -> Self {
        Self {
            physics_engine: SimplePhysicsEngine::new(),
            agents: Vec::new(),
            time: 0.0,
        }
    }
    
    fn create_agent(&mut self, name: String, position: Vector3<f64>) -> Result<Uuid> {
        let mut agent = EmbodiedAgent::new(name, position);
        
        // Create agent's physical body from particles
        let body_particles = self.create_agent_body(position)?;
        agent.body_particles = body_particles;
        
        // Set up distributed AI components
        self.setup_agent_components(&mut agent)?;
        
        // Calculate agent's physical properties
        self.update_agent_physics(&mut agent)?;
        
        let agent_id = agent.id;
        self.agents.push(agent);
        Ok(agent_id)
    }
    
    fn create_agent_body(&mut self, position: Vector3<f64>) -> Result<Vec<usize>> {
        let mut body_particles = Vec::new();
        
        // Create a simple body from 5 particles
        for i in 0..5 {
            let offset = Vector3::new(
                (i as f64 - 2.0) * 1e-8, // Spread particles out
                0.0,
                0.0,
            );
            
            let particle_idx = self.physics_engine.add_particle(
                position + offset,
                Vector3::zeros(),
                if i < 3 { 1.673e-27 } else { 9.109e-31 }, // Protons and electrons
                if i < 3 { 1.602e-19 } else { -1.602e-19 }, // Positive and negative charges
            );
            
            body_particles.push(particle_idx);
        }
        
        Ok(body_particles)
    }
    
    fn setup_agent_components(&mut self, agent: &mut EmbodiedAgent) -> Result<()> {
        // Set up neural core particles
        agent.neural_core.processing_particles = vec![agent.body_particles[0], agent.body_particles[1]];
        agent.neural_core.memory_particles = vec![agent.body_particles[2]];
        agent.neural_core.communication_particles = vec![agent.body_particles[3]];
        
        // Set up curiosity system particles
        agent.curiosity_system.sensory_particles = vec![agent.body_particles[4]];
        agent.curiosity_system.exploration_particles = vec![agent.body_particles[0]];
        agent.curiosity_system.novelty_detector_particles = vec![agent.body_particles[1]];
        agent.curiosity_system.prediction_particles = vec![agent.body_particles[2]];
        
        // Set up other components
        agent.meta_learner.meta_particles = vec![agent.body_particles[3]];
        agent.self_modification.modification_particles = vec![agent.body_particles[4]];
        agent.open_ended_evolution.evolution_particles = vec![agent.body_particles[0]];
        
        Ok(())
    }
    
    fn update_agent_physics(&mut self, agent: &mut EmbodiedAgent) -> Result<()> {
        let mut total_mass = 0.0;
        let mut total_energy = 0.0;
        let mut weighted_position = Vector3::zeros();
        
        for &particle_idx in &agent.body_particles {
            if let Some(particle_data) = self.physics_engine.get_particle_data(particle_idx) {
                total_mass += particle_data.mass;
                total_energy += particle_data.energy;
                weighted_position += particle_data.position * particle_data.mass;
            }
        }
        
        if total_mass > 0.0 {
            agent.position = weighted_position / total_mass;
            agent.mass = total_mass;
            agent.energy = total_energy;
        }
        
        Ok(())
    }
    
    fn step(&mut self, dt: f64) -> Result<()> {
        self.physics_engine.step(dt);
        self.time += dt;
        let agent_count = self.agents.len();
        let mut grow_indices = Vec::new();
        // 1. Update age, get sensory input, execute action, collect grow indices
        for i in 0..agent_count {
            // Only borrow agent for this block
            {
                let agent = &mut self.agents[i];
                agent.update_age(dt);
                let _sensory_input = agent.get_sensory_input(&self.physics_engine)?;
                let actions = vec![
                    ActionType::Explore,
                    ActionType::Investigate,
                    ActionType::Experiment,
                    ActionType::Observe,
                    ActionType::Interact,
                    ActionType::Learn,
                    ActionType::Create,
                    ActionType::Discover,
                ];
                let action = actions[self.time as usize % actions.len()].clone();
                let outcome = agent.execute_action(action, &mut self.physics_engine)?;
                if outcome > 0.2 {
                    grow_indices.push(i);
                }
            }
        }
        // 2. Update agent physics in a separate loop
        for i in 0..agent_count {
            // Only borrow agent for this block
            let agent_ptr: *mut EmbodiedAgent = &mut self.agents[i];
            // SAFETY: No other references to self.agents[i] exist in this loop
            unsafe {
                self.update_agent_physics(&mut *agent_ptr)?;
            }
        }
        // 3. Grow agents in a final loop
        for &i in &grow_indices {
            let agent_ptr: *mut EmbodiedAgent = &mut self.agents[i];
            unsafe {
                self.grow_agent(&mut *agent_ptr)?;
            }
        }
        Ok(())
    }
    
    fn grow_agent(&mut self, agent: &mut EmbodiedAgent) -> Result<()> {
        // Add a new particle to the agent's body
        let new_particle_idx = self.physics_engine.add_particle(
            agent.position + Vector3::new(1e-8, 0.0, 0.0),
            Vector3::zeros(),
            1.673e-27, // Proton mass
            1.602e-19, // Proton charge
        );
        
        agent.body_particles.push(new_particle_idx);
        
        // Update component assignments
        if agent.neural_core.processing_particles.len() < 3 {
            agent.neural_core.processing_particles.push(new_particle_idx);
        } else if agent.curiosity_system.sensory_particles.len() < 2 {
            agent.curiosity_system.sensory_particles.push(new_particle_idx);
        }
        
        Ok(())
    }
    
    fn print_statistics(&self) {
        println!("\n=== Embodied Agent Simulation Statistics ===");
        println!("Time: {:.2e} seconds", self.time);
        println!("Total particles: {}", self.physics_engine.particles.len());
        println!("Total agents: {}", self.agents.len());
        for agent in &self.agents {
            let stats = agent.get_statistics();
            println!("\nAgent: {}", stats.name);
            println!("  ID: {}", stats.id);
            println!("  Generation: {}", stats.generation);
            println!("  Age: {:.2e} seconds", stats.age);
            println!("  Particle count: {}", stats.particle_count);
            println!("  Position: ({:.2e}, {:.2e}, {:.2e})", 
                stats.position.x, stats.position.y, stats.position.z);
            println!("  Velocity: ({:.2e}, {:.2e}, {:.2e})", 
                stats.velocity.x, stats.velocity.y, stats.velocity.z);
            println!("  Energy: {:.2e} J", stats.energy);
            println!("  Mass: {:.2e} kg", stats.mass);
            println!("  Discoveries: {}", stats.discovery_count);
            println!("  Average performance: {:.3}", stats.average_performance);
            println!("  Recent performance: {:.3}", stats.recent_performance);
        }
    }
}

fn main() -> Result<()> {
    println!("🤖 Starting Embodied Agent Demo");
    println!("This demonstrates how AI agents are physically embodied in a physics simulation");
    
    // Create simulation
    let mut simulation = EmbodiedSimulation::new();
    
    // Create embodied agents
    let _agent1_id = simulation.create_agent(
        "Alpha".to_string(), 
        Vector3::new(-1e-7, 0.0, 0.0)
    )?;
    
    let _agent2_id = simulation.create_agent(
        "Beta".to_string(), 
        Vector3::new(1e-7, 0.0, 0.0)
    )?;
    
    let _agent3_id = simulation.create_agent(
        "Gamma".to_string(), 
        Vector3::new(0.0, 1e-7, 0.0)
    )?;
    
    println!("Created 3 embodied agents with physical bodies");
    
    // Run simulation
    let dt = 1e-12; // 1 picosecond timestep
    let total_steps = 100;
    
    for step in 0..total_steps {
        simulation.step(dt)?;
        
        if step % 20 == 0 {
            println!("\n--- Step {} ---", step);
            simulation.print_statistics();
        }
    }
    
    println!("\n🎯 Simulation Complete!");
    println!("Final statistics:");
    simulation.print_statistics();
    
    println!("\n📊 Key Insights:");
    println!("• Agents are physically embodied as collections of particles");
    println!("• Each agent has distributed AI components across its particle network");
    println!("• Agents can interact with the physics environment through their bodies");
    println!("• Successful agents grow by adding more particles");
    println!("• All behaviors emerge from physical interactions");
    
    Ok(())
} 