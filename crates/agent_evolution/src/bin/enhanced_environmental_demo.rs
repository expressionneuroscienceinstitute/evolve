//! # Enhanced Environmental Demo
//!
//! This demo showcases environmental challenges, agent-agent interaction, and trajectory logging.
//! Features include:
//! - Environmental fields (energy, temperature, electromagnetic)
//! - Obstacles (walls, energy barriers, force fields)
//! - Energy gradients that affect movement costs
//! - Resources that agents can collect and compete for
//! - Agent-agent interaction (competition, cooperation, avoidance)
//! - Trajectory logging and analysis

use agent_evolution::{
    embodied_agent::{
        EmbodiedAgent, PhysicsEngineInterface, ParticleData, 
        EnvironmentalFeature, FieldType, Obstacle, ObstacleType,
        EnergyGradient, GradientType, Resource, InteractionEvent
    },
    curiosity::ActionType,
};
use nalgebra::Vector3;
use anyhow::Result;
use uuid::Uuid;
use std::collections::HashMap;
use serde_json;

/// Enhanced physics engine with environmental features
#[derive(Debug)]
struct EnhancedPhysicsEngine {
    particles: Vec<SimpleParticle>,
    fields: Vec<EnvironmentalFeature>,
    obstacles: Vec<Obstacle>,
    gradients: Vec<EnergyGradient>,
    resources: Vec<Resource>,
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

impl EnhancedPhysicsEngine {
    fn new() -> Self {
        let mut engine = Self {
            particles: Vec::new(),
            fields: Vec::new(),
            obstacles: Vec::new(),
            gradients: Vec::new(),
            resources: Vec::new(),
            time: 0.0,
        };
        
        // Create environmental features
        engine.create_environmental_features();
        
        engine
    }
    
    fn create_environmental_features(&mut self) {
        // Energy field in the center
        self.fields.push(EnvironmentalFeature {
            id: Uuid::new_v4(),
            position: Vector3::new(0.0, 0.0, 0.0),
            radius: 5e-8, // 50 nanometers
            field_type: FieldType::EnergyField { 
                strength: 1e-12, 
                decay_rate: 0.1 
            },
            is_active: true,
            lifetime: None,
            age: 0.0,
        });
        
        // Temperature field
        self.fields.push(EnvironmentalFeature {
            id: Uuid::new_v4(),
            position: Vector3::new(1e-7, 0.0, 0.0),
            radius: 3e-8,
            field_type: FieldType::TemperatureField { 
                temperature: 300.0, 
                conductivity: 0.1 
            },
            is_active: true,
            lifetime: None,
            age: 0.0,
        });
        
        // Electromagnetic field
        self.fields.push(EnvironmentalFeature {
            id: Uuid::new_v4(),
            position: Vector3::new(-1e-7, 0.0, 0.0),
            radius: 4e-8,
            field_type: FieldType::ElectromagneticField {
                electric_strength: Vector3::new(1e-10, 0.0, 0.0),
                magnetic_strength: Vector3::new(0.0, 1e-10, 0.0),
            },
            is_active: true,
            lifetime: None,
            age: 0.0,
        });
        
        // Obstacles
        self.obstacles.push(Obstacle {
            id: Uuid::new_v4(),
            position: Vector3::new(5e-8, 0.0, 0.0),
            size: Vector3::new(1e-8, 1e-8, 1e-8),
            obstacle_type: ObstacleType::Wall { friction: 0.1 },
            is_dynamic: false,
            velocity: Vector3::zeros(),
            damage_potential: 0.0,
        });
        
        self.obstacles.push(Obstacle {
            id: Uuid::new_v4(),
            position: Vector3::new(-5e-8, 0.0, 0.0),
            size: Vector3::new(1e-8, 1e-8, 1e-8),
            obstacle_type: ObstacleType::EnergyBarrier { damage_per_second: 1e-12 },
            is_dynamic: false,
            velocity: Vector3::zeros(),
            damage_potential: 1e-12,
        });
        
        // Energy gradient
        self.gradients.push(EnergyGradient {
            id: Uuid::new_v4(),
            start_position: Vector3::new(0.0, -1e-7, 0.0),
            end_position: Vector3::new(0.0, 1e-7, 0.0),
            start_energy_cost: 1e-15,
            end_energy_cost: 1e-13,
            gradient_type: GradientType::Linear,
        });
        
        // Resources
        for i in 0..5 {
            self.resources.push(Resource {
                id: Uuid::new_v4(),
                position: Vector3::new(
                    (i as f64 - 2.0) * 2e-8,
                    (i as f64 - 2.0) * 2e-8,
                    0.0
                ),
                resource_type: "energy_crystal".to_string(),
                value: 1.0 + i as f64,
                energy_content: (1.0 + i as f64) * 1e-12,
                is_collected: false,
                collection_time: None,
                collected_by: None,
            });
        }
    }
    
    fn step(&mut self, dt: f64) {
        // Update particle physics
        for particle in &mut self.particles {
            particle.position += particle.velocity * dt;
            particle.energy *= (1.0 - 0.01 * dt).max(0.0); // Energy decay
        }
        
        // Update field ages
        for field in &mut self.fields {
            field.age += dt;
            if let Some(lifetime) = field.lifetime {
                if field.age > lifetime {
                    field.is_active = false;
                }
            }
        }
        
        // Update dynamic obstacles
        for obstacle in &mut self.obstacles {
            if obstacle.is_dynamic {
                obstacle.position += obstacle.velocity * dt;
            }
        }
        
        self.time += dt;
    }
}

impl PhysicsEngineInterface for EnhancedPhysicsEngine {
    fn get_particle_count(&self) -> usize {
        self.particles.len()
    }
    
    fn get_particle_data(&self, index: usize) -> Option<ParticleData> {
        self.particles.get(index).map(|p| ParticleData {
            position: p.position,
            velocity: p.velocity,
            energy: p.energy,
            mass: p.mass,
            electric_charge: p.electric_charge,
        })
    }
    
    fn apply_force_to_particle(&mut self, index: usize, force: Vector3<f64>) -> Result<()> {
        if let Some(particle) = self.particles.get_mut(index) {
            particle.velocity += force / particle.mass;
        }
        Ok(())
    }
    
    fn transfer_energy(&mut self, from: usize, to: usize, amount: f64) -> Result<()> {
        if from < self.particles.len() && to < self.particles.len() && from != to {
            let from_energy = self.particles[from].energy;
            if from_energy >= amount {
                self.particles[from].energy -= amount;
                self.particles[to].energy += amount;
            }
        }
        Ok(())
    }
    
    fn create_particle(&mut self, position: Vector3<f64>, velocity: Vector3<f64>, mass: f64, charge: f64) -> Result<usize> {
        let particle = SimpleParticle {
            position,
            velocity,
            energy: 1e-12,
            mass,
            electric_charge: charge,
        };
        self.particles.push(particle);
        Ok(self.particles.len() - 1)
    }
}

/// Enhanced simulation with environmental challenges and agent interaction
#[derive(Debug)]
struct EnhancedSimulation {
    physics_engine: EnhancedPhysicsEngine,
    agents: Vec<EmbodiedAgent>,
    time: f64,
    trajectory_log: Vec<TrajectoryEntry>,
    interaction_log: Vec<InteractionEvent>,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
struct TrajectoryEntry {
    timestamp: f64,
    agent_id: Uuid,
    position: Vector3<f64>,
    velocity: Vector3<f64>,
    energy: f64,
    action: Option<String>,
    environmental_events: Vec<String>,
}

impl EnhancedSimulation {
    fn new() -> Self {
        Self {
            physics_engine: EnhancedPhysicsEngine::new(),
            agents: Vec::new(),
            time: 0.0,
            trajectory_log: Vec::new(),
            interaction_log: Vec::new(),
        }
    }
    
    fn create_agent(&mut self, name: String, position: Vector3<f64>) -> Uuid {
        let mut agent = EmbodiedAgent::new(name, position);
        
        // Create particles for the agent
        for _ in 0..5 {
            let particle_idx = self.physics_engine.create_particle(
                position + Vector3::new(
                    (rand::random::<f64>() - 0.5) * 1e-9,
                    (rand::random::<f64>() - 0.5) * 1e-9,
                    0.0
                ),
                Vector3::zeros(),
                1e-27, // 1 atomic mass unit
                1e-19, // Elementary charge
            ).unwrap();
            agent.body_particles.push(particle_idx);
        }
        
        let agent_id = agent.id;
        self.agents.push(agent);
        agent_id
    }
    
    fn step(&mut self, dt: f64) -> Result<()> {
        self.physics_engine.step(dt);
        self.time += dt;
        
        // Precompute environmental events for all agents
        let all_environmental_events: Vec<Vec<String>> = self.agents.iter()
            .map(|agent| self.get_environmental_events(agent))
            .collect();
        
        // Update each agent
        for (agent, environmental_events) in self.agents.iter_mut().zip(all_environmental_events) {
            agent.update_age(dt);
            
            // Update environmental awareness
            let events = agent.update_environmental_awareness(
                &self.physics_engine.fields,
                &self.physics_engine.obstacles,
                &self.physics_engine.gradients,
                &self.physics_engine.resources,
                &[], // Empty slice to avoid borrow checker issues
                dt
            )?;
            
            // Log interaction events
            self.interaction_log.extend(events);
            
            // Execute AI-driven actions
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
            let action = actions[(self.time / dt) as usize % actions.len()].clone();
            let _outcome = agent.execute_action(action.clone(), &mut self.physics_engine)?;
            
            // Log trajectory
            self.trajectory_log.push(TrajectoryEntry {
                timestamp: self.time,
                agent_id: agent.id,
                position: agent.position,
                velocity: agent.velocity,
                energy: agent.energy,
                action: Some(format!("{:?}", action)),
                environmental_events,
            });
        }
        
        // Agent-agent interactions
        self.handle_agent_interactions(dt)?;
        
        Ok(())
    }
    
    fn handle_agent_interactions(&mut self, dt: f64) -> Result<()> {
        for i in 0..self.agents.len() {
            for j in (i + 1)..self.agents.len() {
                let distance = (self.agents[i].position - self.agents[j].position).norm();
                
                if distance < 1e-7 { // 100 nanometers
                    // Use split_at_mut to get two mutable references
                    let (left, right) = self.agents.split_at_mut(j);
                    let agent_i = &mut left[i];
                    let agent_j = &mut right[0];
                    
                    // Decide interaction type based on agent states
                    let interaction_type = if agent_i.energy > agent_j.energy {
                        "compete"
                    } else if agent_i.energy < 1e-12 && agent_j.energy < 1e-12 {
                        "cooperate"
                    } else {
                        "avoid"
                    };
                    
                    let outcome = agent_i.interact_with_agent(agent_j, interaction_type)?;
                    
                    if outcome != 0.0 {
                        self.interaction_log.push(InteractionEvent {
                            timestamp: self.time,
                            event_type: agent_evolution::embodied_agent::InteractionEventType::CompeteWithAgent {
                                opponent_id: agent_j.id,
                                success: outcome > 0.0,
                            },
                            target_id: Some(agent_j.id),
                            outcome,
                            energy_cost: outcome.abs(),
                        });
                    }
                }
            }
        }
        Ok(())
    }
    
    fn get_environmental_events(&self, agent: &EmbodiedAgent) -> Vec<String> {
        let mut events = Vec::new();
        
        // Check field interactions
        for field in &self.physics_engine.fields {
            let distance = (agent.position - field.position).norm();
            if distance < field.radius {
                events.push(format!("In {:?} field", field.field_type));
            }
        }
        
        // Check obstacle proximity
        for obstacle in &self.physics_engine.obstacles {
            let distance = (agent.position - obstacle.position).norm();
            if distance < obstacle.size.norm() {
                events.push(format!("Near {:?} obstacle", obstacle.obstacle_type));
            }
        }
        
        // Check resource proximity
        for resource in &self.physics_engine.resources {
            let distance = (agent.position - resource.position).norm();
            if distance < 1e-8 && !resource.is_collected {
                events.push(format!("Near {} resource", resource.resource_type));
            }
        }
        
        events
    }
    
    fn export_trajectory_data(&self, filename: &str) -> Result<()> {
        let data = serde_json::to_string_pretty(&self.trajectory_log)?;
        std::fs::write(filename, data)?;
        println!("Trajectory data exported to {}", filename);
        Ok(())
    }
    
    fn export_interaction_data(&self, filename: &str) -> Result<()> {
        let data = serde_json::to_string_pretty(&self.interaction_log)?;
        std::fs::write(filename, data)?;
        println!("Interaction data exported to {}", filename);
        Ok(())
    }
    
    fn print_statistics(&self) {
        println!("\n=== Enhanced Environmental Simulation Statistics ===");
        println!("Time: {:.2e} seconds", self.time);
        println!("Total particles: {}", self.physics_engine.particles.len());
        println!("Total agents: {}", self.agents.len());
        println!("Environmental features:");
        println!("  Fields: {}", self.physics_engine.fields.len());
        println!("  Obstacles: {}", self.physics_engine.obstacles.len());
        println!("  Gradients: {}", self.physics_engine.gradients.len());
        println!("  Resources: {}", self.physics_engine.resources.len());
        println!("Trajectory entries: {}", self.trajectory_log.len());
        println!("Interaction events: {}", self.interaction_log.len());
        
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
        
        // Resource collection statistics
        let collected_resources: Vec<_> = self.physics_engine.resources.iter()
            .filter(|r| r.is_collected)
            .collect();
        println!("\nResource Collection:");
        println!("  Collected: {}/{}", collected_resources.len(), self.physics_engine.resources.len());
        
        // Environmental interaction statistics
        let field_interactions = self.interaction_log.iter()
            .filter(|e| matches!(e.event_type, agent_evolution::embodied_agent::InteractionEventType::InteractWithField { .. }))
            .count();
        let obstacle_interactions = self.interaction_log.iter()
            .filter(|e| matches!(e.event_type, agent_evolution::embodied_agent::InteractionEventType::AvoidObstacle { .. }))
            .count();
        let gradient_interactions = self.interaction_log.iter()
            .filter(|e| matches!(e.event_type, agent_evolution::embodied_agent::InteractionEventType::NavigateGradient { .. }))
            .count();
        let resource_interactions = self.interaction_log.iter()
            .filter(|e| matches!(e.event_type, agent_evolution::embodied_agent::InteractionEventType::CollectResource { .. }))
            .count();
        let agent_interactions = self.interaction_log.iter()
            .filter(|e| matches!(e.event_type, agent_evolution::embodied_agent::InteractionEventType::CompeteWithAgent { .. }))
            .count();
        
        println!("\nEnvironmental Interactions:");
        println!("  Field interactions: {}", field_interactions);
        println!("  Obstacle interactions: {}", obstacle_interactions);
        println!("  Gradient interactions: {}", gradient_interactions);
        println!("  Resource interactions: {}", resource_interactions);
        println!("  Agent-agent interactions: {}", agent_interactions);
    }
}

fn main() -> Result<()> {
    println!("🤖 Starting Enhanced Environmental Demo");
    println!("This demonstrates environmental challenges, agent-agent interaction, and trajectory logging");
    
    let mut simulation = EnhancedSimulation::new();
    
    // Create multiple agents
    let _agent1_id = simulation.create_agent(
        "Alpha".to_string(),
        Vector3::new(-1e-7, 0.0, 0.0)
    );
    let _agent2_id = simulation.create_agent(
        "Beta".to_string(),
        Vector3::new(1e-7, 0.0, 0.0)
    );
    let _agent3_id = simulation.create_agent(
        "Gamma".to_string(),
        Vector3::new(0.0, 1e-7, 0.0)
    );
    let _agent4_id = simulation.create_agent(
        "Delta".to_string(),
        Vector3::new(0.0, -1e-7, 0.0)
    );
    
    println!("Created {} embodied agents with environmental challenges", simulation.agents.len());
    
    // Run simulation
    let dt = 1e-12; // 1 picosecond
    let total_steps = 100;
    
    for step in 0..=total_steps {
        simulation.step(dt)?;
        
        if step % 20 == 0 {
            println!("\n--- Step {} ---", step);
            simulation.print_statistics();
        }
    }
    
    println!("\n🎯 Simulation Complete!");
    println!("Final statistics:");
    simulation.print_statistics();
    
    // Export trajectory and interaction data
    simulation.export_trajectory_data("trajectory_data.json")?;
    simulation.export_interaction_data("interaction_data.json")?;
    
    println!("\n📊 Key Insights:");
    println!("• Agents navigate complex environmental challenges");
    println!("• Energy gradients influence movement strategies");
    println!("• Resource competition drives agent behavior");
    println!("• Environmental fields provide energy and constraints");
    println!("• Agent-agent interactions create emergent behaviors");
    println!("• Trajectory data enables detailed behavioral analysis");
    
    Ok(())
} 