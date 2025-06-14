//! AI Agent System Performance Benchmarks

use criterion::{black_box, criterion_group, criterion_main, Criterion};
use universe_sim::agent::{Agent, Lineage};
use universe_sim::types::*;
use universe_sim::tech::TechTree;

fn create_test_agent(id: AgentId, lineage_id: LineageId) -> Agent {
    Agent::new(id, lineage_id, Coord2D::new(0, 0))
}

fn create_test_observation() -> Observation {
    Observation {
        tick: Tick::new(1000),
        location: Coord2D::new(10, 10),
        local_resources: ElementTable::earth_baseline(),
        nearby_agents: vec![
            (AgentId::new(2), 5.0),
            (AgentId::new(3), 12.3),
            (AgentId::new(4), 8.7),
        ],
        environment: EnvironmentSnapshot {
            temperature: Temperature::new(298.0),
            pressure: Pressure::from_atmospheres(1.0),
            radiation: RadiationDose::new(0.002),
            energy_flux: EnergyFlux::new(1.361),
            liquid_water_fraction: 0.7,
            atmospheric_oxygen: 0.21,
            hazard_rate: 0.001,
        },
        hazards: vec![
            CosmicHazard::SolarFlare { intensity: 2.5, eta_ticks: 100 },
        ],
        energy_budget: 1000.0,
        oracle_message: None,
        available_techs: vec![TechId::new(1), TechId::new(2)],
    }
}

fn bench_agent_decision_making(c: &mut Criterion) {
    let lineage_id = LineageId::new(1);
    let mut agent = create_test_agent(AgentId::new(1), lineage_id);
    let observation = create_test_observation();
    
    c.bench_function("agent_observe_and_act", |b| {
        b.iter(|| {
            let _ = agent.observe_and_act(black_box(&observation));
        })
    });
}

fn bench_agent_survival_check(c: &mut Criterion) {
    let lineage_id = LineageId::new(1);
    let agent = create_test_agent(AgentId::new(1), lineage_id);
    let environment = EnvironmentSnapshot {
        temperature: Temperature::new(298.0),
        pressure: Pressure::from_atmospheres(1.0),
        radiation: RadiationDose::new(0.002),
        energy_flux: EnergyFlux::new(1.361),
        liquid_water_fraction: 0.7,
        atmospheric_oxygen: 0.21,
        hazard_rate: 0.001,
    };
    
    c.bench_function("agent_survival_check", |b| {
        b.iter(|| {
            let _ = agent.check_survival(black_box(&environment));
        })
    });
}

fn bench_multiple_agents(c: &mut Criterion) {
    let lineage_id = LineageId::new(1);
    let observation = create_test_observation();
    
    // Benchmark different numbers of agents
    for agent_count in [10, 100, 1000].iter() {
        let mut agents: Vec<Agent> = (0..*agent_count)
            .map(|i| create_test_agent(AgentId::new(i as u64), lineage_id))
            .collect();
        
        c.bench_function(&format!("agents_{}_decision_making", agent_count), |b| {
            b.iter(|| {
                for agent in &mut agents {
                    let _ = agent.observe_and_act(black_box(&observation));
                }
            })
        });
    }
}

fn bench_lineage_management(c: &mut Criterion) {
    let _lineage = Lineage::new(LineageId::new(1), Tick::new(0));
    
    c.bench_function("lineage_tracking", |b| {
        b.iter(|| {
            // Simulate lineage operations - simplified since methods don't exist yet
            let _creation_tick = Tick::new(0);
            let _fitness_placeholder = 1.0;
        })
    });
}

fn bench_tech_tree_operations(c: &mut Criterion) {
    let mut tech_tree = TechTree::new();
    let lineage_id = LineageId::new(1);
    
    c.bench_function("tech_research", |b| {
        b.iter(|| {
            let _ = tech_tree.research_tech(
                black_box(lineage_id),
                black_box(TechId::new(1))
            );
        })
    });
    
    c.bench_function("tech_availability_check", |b| {
        b.iter(|| {
            let _ = tech_tree.get_available_techs(black_box(lineage_id));
        })
    });
}

criterion_group!(
    benches,
    bench_agent_decision_making,
    bench_agent_survival_check,
    bench_multiple_agents,
    bench_lineage_management,
    bench_tech_tree_operations
);
criterion_main!(benches);