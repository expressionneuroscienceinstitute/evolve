use anyhow::Result;
use std::time::{Instant, Duration};
use universe_sim::{config::SimulationConfig, UniverseSimulation};

/// Simple 60-second wall-clock benchmark. Runs a tiny universe and prints achieved UPS.
fn main() -> Result<()> {
    let mut cfg = SimulationConfig::default();
    cfg.initial_particle_count = 1000000;
    cfg.target_ups = 250.0;
    cfg.tick_span_years = 1e6; // keep default

    let mut sim = UniverseSimulation::new(cfg)?;
    sim.init_big_bang()?;

    let wall = Duration::from_secs(60);
    let start = Instant::now();
    let mut ticks: u64 = 0;
    while start.elapsed() < wall {
        sim.tick()?;
        ticks += 1;
    }

    let elapsed = start.elapsed().as_secs_f64();
    let ups = ticks as f64 / elapsed;

    println!(
        "Benchmark complete: {} ticks in {:.1}s => {:.1} UPS (target {})",
        ticks, elapsed, ups, sim.target_ups
    );
    Ok(())
} 