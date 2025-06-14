//! Physics Engine Performance Benchmarks

use criterion::{black_box, criterion_group, criterion_main, Criterion};
use universe_sim::config::PhysicsConfig;
use universe_sim::physics::{PhysicsEngine, CelestialBody, CelestialBodyType};
use universe_sim::types::*;
use std::collections::HashMap;

fn create_test_engine() -> PhysicsEngine {
    let config = PhysicsConfig::default();
    PhysicsEngine::new(&config)
}

fn create_test_bodies(count: usize) -> HashMap<StarId, CelestialBody> {
    let mut bodies = HashMap::new();
    
    for i in 0..count {
        let body = CelestialBody {
            body_type: CelestialBodyType::Star,
            mass: MassEnergy::new(1.989e30), // Solar mass
            position: Coord3D::new(
                (i as f64) * 1e12, // 1 parsec spacing
                0.0,
                0.0,
            ),
            velocity: Velocity::zero(),
            temperature: Temperature::new(5778.0),
            luminosity: 3.828e26, // Solar luminosity
            age_ticks: 0,
        };
        
        bodies.insert(StarId::new(i as u64 + 1), body);
    }
    
    bodies
}

fn bench_gravity_simulation(c: &mut Criterion) {
    let mut engine = create_test_engine();
    
    // Benchmark different numbers of bodies
    for body_count in [10, 50, 100].iter() {
        let mut bodies = create_test_bodies(*body_count);
        
        c.bench_function(&format!("gravity_{}_bodies", body_count), |b| {
            b.iter(|| {
                let _ = engine.tick(
                    black_box(Tick::new(1)),
                    black_box(&mut bodies),
                    black_box(1_000_000.0), // 1 million years per tick
                );
            })
        });
    }
}

fn bench_conservation_checks(c: &mut Criterion) {
    let engine = create_test_engine();
    let _bodies = create_test_bodies(50);
    
    c.bench_function("conservation_validation", |b| {
        b.iter(|| {
            // Test conservation law validation
            let _ = black_box(&engine).get_physics_state();
        })
    });
}

fn bench_relativity_corrections(c: &mut Criterion) {
    let mut config = PhysicsConfig::default();
    config.relativistic = true;
    let mut engine = PhysicsEngine::new(&config);
    
    let mut bodies = create_test_bodies(20);
    
    // Set high velocities to trigger relativistic corrections
    for body in bodies.values_mut() {
        body.velocity = Velocity::new(1e8, 0.0, 0.0); // 0.33c
    }
    
    c.bench_function("relativistic_gravity", |b| {
        b.iter(|| {
            let _ = engine.tick(
                black_box(Tick::new(1)),
                black_box(&mut bodies),
                black_box(1_000_000.0),
            );
        })
    });
}

criterion_group!(
    benches,
    bench_gravity_simulation,
    bench_conservation_checks,
    bench_relativity_corrections
);
criterion_main!(benches);