//! Universe Simulation CLI (universectl)
//! 
//! Command-line interface for the universe simulation with full God-Mode and diagnostics

use anyhow::Result;
use clap::{Parser, Subcommand};
use serde::Deserialize;
use serde_json::json;
use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};
use tokio::time::sleep;
use tracing::{error, info, warn};
use universe_sim::{config::SimulationConfig, persistence, UniverseSimulation, CelestialBodyType};

mod rpc;

// Add import after other use lines
use warp::Filter;

/// A struct to hold the shared state of the simulation for RPC.
#[derive(Clone)]
struct SharedState {
    sim: Arc<Mutex<UniverseSimulation>>,
    last_save_time: Option<Instant>,
}

#[derive(Parser)]
#[command(
    name = "universectl",
    author = "Universe Simulation Team",
    version = "0.1.0",
    about = "Command-line interface for the Universe Simulation",
    long_about = "Control and monitor the universe simulation from Big Bang to far future"
)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
    
    #[arg(short, long, global = true)]
    config: Option<PathBuf>,
    
    #[arg(long, global = true)]
    godmode: bool,
    
    #[arg(long, global = true)]
    verbose: bool,
    
    #[arg(long, global = true)]
    trace: bool,
}

#[derive(Subcommand)]
enum Commands {
    /// Show simulation status
    Status,
    
    /// Start the simulation
    Start {
        #[arg(long)]
        load: Option<PathBuf>,
        
        #[arg(long)]
        preset: Option<String>,
        
        #[arg(long)]
        tick_span: Option<f64>,
        
        #[arg(long)]
        low_mem: bool,
        
        #[arg(long)]
        native_render: bool,
        
        /// On macOS, enables the experimental Metal renderer on Apple Silicon.
        /// This overrides the default safety disable.
        #[arg(long)]
        silicon: bool,
        
        #[arg(long, default_value = "9001")]
        rpc_port: u16,
        
        #[arg(long)]
        allow_net: bool,
    },
    
    /// Stop the simulation
    Stop,
    
    /// Render ASCII heat-map of universe
    Map {
        #[arg(default_value = "1")]
        zoom: f64,
        
        #[arg(long, default_value = "stars")]
        layer: String,
    },
    
    /// List planets with filtering
    ListPlanets {
        #[arg(long)]
        class: Option<String>,
        
        #[arg(long)]
        habitable: bool,
    },
    
    /// Inspect specific entities
    Inspect {
        #[command(subcommand)]
        target: InspectTarget,
    },
    
    /// Create snapshot for analysis
    Snapshot {
        file: PathBuf,
        
        #[arg(long)]
        format: Option<String>,
    },
    
    /// Control simulation speed
    Speed {
        factor: f64,
    },
    
    /// Rewind to previous tick
    Rewind {
        ticks: u64,
    },
    
    /// God-Mode commands (requires --godmode flag)
    #[command(name = "godmode")]
    GodMode {
        #[command(subcommand)]
        action: GodModeAction,
    },
    
    /// Resource management
    Resources {
        #[command(subcommand)]
        action: ResourceAction,
    },
    
    /// Oracle-Link communication
    Oracle {
        #[command(subcommand)]
        action: OracleAction,
    },
    
    /// Interactive mode for real-time simulation control
    Interactive,
}

#[derive(Subcommand, Debug)]
enum InspectTarget {
    Planet { id: String },
    Lineage { id: String },
    Universe,
    Physics,
    /// Historical trend analysis of universe statistics
    UniverseHistory,
}

#[derive(Subcommand)]
enum GodModeAction {
    CreateBody {
        #[arg(long)]
        mass: f64,
        
        #[arg(long)]
        body_type: String,
        
        #[arg(long)]
        x: f64,
        
        #[arg(long)]
        y: f64,
        
        #[arg(long)]
        z: f64,
    },
    
    DeleteBody {
        id: String,
    },
    
    SetConstant {
        name: String,
        value: f64,
    },
    
    SpawnLineage {
        #[arg(long)]
        code_hash: String,
        
        #[arg(long)]
        planet_id: String,
    },
    
    Miracle {
        planet_id: String,
        
        #[arg(long)]
        miracle_type: String,
        
        #[arg(long)]
        duration: Option<u64>,
        
        #[arg(long)]
        intensity: Option<f64>,
    },
    
    TimeWarp {
        factor: f64,
    },
    
    InspectEval {
        expression: String,
    },
    
    /// Create a new life lineage (agent) on a given planet (God-mode)
    #[command(name = "create-agent")]
    CreateAgent {
        #[arg(long)]
        planet_id: String,
    },
}

#[derive(Subcommand)]
enum ResourceAction {
    Queue,
    Grant {
        id: String,
        
        #[arg(long)]
        expires: Option<String>,
    },
    Status,
    Reload,
}

#[derive(Subcommand)]
enum OracleAction {
    Inbox,
    Reply {
        petition_id: String,
        
        #[arg(long)]
        action: String,
        
        #[arg(long)]
        message: Option<String>,
    },
    Stats,
}

#[derive(Clone, Copy)]
struct ParticleSnapshot {
    position: [f64; 3],
    momentum: [f64; 3],
    energy: f64,
}

#[tokio::main]
async fn main() -> Result<()> {
    let cli = Cli::parse();
    
    init_logging(cli.verbose, cli.trace);
    
    let config = load_config(cli.config.as_ref()).await?;
    
    match cli.command {
        Commands::Status => cmd_status().await,
        Commands::Start { load, preset, tick_span, low_mem, native_render, silicon, rpc_port, allow_net } => {
            cmd_start(config, load, preset, tick_span, low_mem, native_render, silicon, rpc_port, allow_net).await
        },
        Commands::Stop => cmd_stop().await,
        Commands::Map { zoom, layer } => cmd_map(zoom, &layer).await,
        Commands::ListPlanets { class, habitable } => cmd_list_planets(class, habitable).await,
        Commands::Inspect { target } => cmd_inspect(target).await,
        Commands::Snapshot { file, format } => cmd_snapshot(file, format).await,
        Commands::Speed { factor } => cmd_speed(factor).await,
        Commands::Rewind { ticks } => cmd_rewind(ticks).await,
        Commands::GodMode { action } => cmd_godmode(action).await,
        Commands::Resources { action } => cmd_resources(action).await,
        Commands::Oracle { action } => cmd_oracle(action).await,
        Commands::Interactive => cmd_interactive().await,
    }
}

fn init_logging(verbose: bool, trace_json: bool) {
    use tracing_subscriber::{fmt, EnvFilter};
    use tracing_appender::non_blocking;

    let level = if verbose { "debug" } else { "info" };
    let env_filter = EnvFilter::new(format!(
        "universectl={},universe_sim={},physics_engine={}",
        level, level, level
    ));

    // Non-blocking writer reduces stalls when the terminal is slow (e.g. verbose mode)
    let (writer, _guard) = non_blocking(std::io::stderr());

    // Handle the different subscriber types by setting them directly
    if trace_json {
        let subscriber = fmt::Subscriber::builder()
            .with_env_filter(env_filter)
            .with_writer(writer)
            .json()
            .finish();
        let _ = tracing::subscriber::set_global_default(subscriber);
    } else {
        let subscriber = fmt::Subscriber::builder()
            .with_env_filter(env_filter)
            .with_writer(writer)
            .finish();
        let _ = tracing::subscriber::set_global_default(subscriber);
    }

    let _ = tracing_log::LogTracer::init();
}

async fn load_config(config_path: Option<&PathBuf>) -> Result<SimulationConfig> {
    match config_path {
        Some(path) => {
            tracing::info!("Loading configuration from {:?}", path);
            SimulationConfig::from_file(path)
        },
        None => {
            let default_paths = [
                "config/default.toml",
                "/etc/universe/config.toml",
                "~/.config/universe/config.toml",
            ];
            
            for path in &default_paths {
                if std::path::Path::new(path).exists() {
                    tracing::info!("Loading configuration from {}", path);
                    return SimulationConfig::from_file(path);
                }
            }
            
            tracing::info!("Using default configuration");
            Ok(SimulationConfig::default())
        }
    }
}

async fn cmd_status() -> Result<()> {
    println!("Universe Simulation Status");
    println!("==========================");

    let client = reqwest::Client::new();
    let req_body = json!({
        "jsonrpc": "2.0",
        "method": "status",
        "params": {},
        "id": 1
    });

    let res = client
        .post("http://127.0.0.1:9001/rpc")
        .json(&req_body)
        .send()
        .await?;

    if !res.status().is_success() {
        println!("Error: Failed to connect to simulation RPC server.");
        println!("Is the simulation running with 'universectl start'?");
        return Ok(());
    }

    let rpc_res: rpc::RpcResponse<rpc::StatusResponse> = res.json().await?;

    if let Some(error) = rpc_res.error {
        println!("RPC Error: {} (code: {})", error.message, error.code);
        return Ok(());
    }

    if let Some(status) = rpc_res.result {
        println!("--- Simulation Status ---");
        println!("Status: {}", status.status);
        println!("Tick: {}", status.tick);
        println!("UPS: {:.2}", status.ups);
        println!("Age: {:.3} Gyr", status.universe_age_gyr);
        println!("State: {}", status.universe_description);
        println!("Lineages: {}", status.lineage_count);
        if let Some(age) = status.save_file_age_sec {
            println!("Last save: {} seconds ago", age);
        } else {
            println!("Last save: Never");
        }
    }

    Ok(())
}

async fn cmd_start(
    mut config: SimulationConfig,
    load: Option<PathBuf>,
    _preset: Option<String>,
    tick_span: Option<f64>,
    low_mem: bool,
    native_render: bool,
    silicon: bool,
    rpc_port: u16,
    _allow_net: bool,
) -> Result<()> {
    if let Some(ts) = tick_span {
        config.tick_span_years = ts;
    }
    if low_mem {
        config.initial_particle_count = 1000; // Lower particle count for low-mem
    }

    let sim = if let Some(path) = load {
        println!("Loading simulation from checkpoint: {:?}", path);
        persistence::load_checkpoint(&path)?
    } else {
        println!("Starting new simulation from Big Bang...");
        let mut sim = UniverseSimulation::new(config.clone())?;
        sim.init_big_bang()?;
        
        // Also initialize Big Bang conditions in the physics engine
        println!("🔬 Initializing physics engine Big Bang conditions...");
        sim.physics_engine.initialize_big_bang()?;
        
        println!("✅ Simulation initialized with {} particles in store and {} particles in physics engine",
            sim.store.particles.count,
            sim.physics_engine.particles.len());
        
        sim
    };

    let sim = Arc::new(Mutex::new(sim));

    // Start RPC server
    let shared_state = Arc::new(Mutex::new(SharedState { sim: sim.clone(), last_save_time: None }));
    tokio::spawn(start_rpc_server(rpc_port, shared_state.clone()));

    // Start native renderer if requested
    #[cfg(not(target_os = "macos"))]
    if native_render {
        info!("Starting high-performance native renderer");
        let render_sim = sim.clone();
        tokio::task::spawn_blocking(move || {
            tokio::runtime::Handle::current().block_on(async {
                if let Err(e) = native_renderer::run_renderer(render_sim).await {
                    error!("Native renderer error: {}", e);
                }
            });
        });
    }

    #[cfg(target_os = "macos")]
    if native_render {
        if silicon {
            info!("Starting experimental native renderer on Apple Silicon (main-thread mode)");

            // Spawn simulation loop on a background task so the main thread can own the EventLoop.
            let sim_for_loop = sim.clone();
            tokio::spawn(async move {
                loop {
                    {
                        let mut guard = sim_for_loop.lock().unwrap();
                        if let Err(e) = guard.tick() {
                            error!("Simulation tick error: {}", e);
                            break;
                        }
                    }
                    sleep(Duration::from_millis(1)).await;
                }
            });

            // Run renderer on the current (main) thread – blocks until window closed.
            native_renderer::run_renderer(sim.clone()).await?;

            return Ok(());
        } else {
            warn!("Native renderer is disabled on macOS. Pass --silicon in addition to --native-render to enable the experimental Metal renderer (requires window on main thread).");
        }
    }

    println!("Simulation started. Press Ctrl+C to stop.");

    let mut _ups = 0.0;
    let mut sent_initial_state = false;
    let mut last_particle_state: HashMap<usize, ParticleSnapshot> = HashMap::new();

    loop {
        let start_time = tokio::time::Instant::now();
        let mut sim_guard = sim.lock().unwrap();

        if let Err(e) = sim_guard.tick() {
            eprintln!("Error during simulation tick: {}", e);
            break;
        }

        // Auto-saving logic
        if config.auto_save_interval > 0 && sim_guard.current_tick % config.auto_save_interval == 0 {
            let save_path = PathBuf::from(&config.auto_save_path);
            if !save_path.exists() {
                std::fs::create_dir_all(&save_path)?;
            }
            let checkpoint_file = save_path.join(format!("snapshot_{}.bin", sim_guard.current_tick));
            info!("Auto-saving checkpoint to {:?}", checkpoint_file);
            if let Err(e) = persistence::save_checkpoint(&mut sim_guard, &checkpoint_file) {
                error!("Failed to save checkpoint: {}", e);
            } else {
                // Update last save time on successful save
                drop(sim_guard);
                if let Ok(mut state) = shared_state.lock() {
                    state.last_save_time = Some(Instant::now());
                }
                sim_guard = sim.lock().unwrap();
            }
        }
        
        let _stats = sim_guard.get_stats().unwrap();

        drop(sim_guard); // Release lock before sleeping

        // Frame limiting
        let elapsed = start_time.elapsed();
        
        // Send WebSocket updates every 10 ticks or so for better responsiveness
        if _stats.current_tick % 10 == 0 {
            // Get a fresh lock on the simulation to get more detailed data
            let sim_guard = sim.lock().unwrap();
            
            // Get physics engine reference (extract data we need before borrowing mutably)
            let physics_data = {
                let physics = &sim_guard.physics_engine;
                
                // Extract particle data from simulation (no hard limit – rely on compression)
                let particle_data: Vec<serde_json::Value> = physics.particles
                    .iter()
                    .enumerate()
                    .map(|(i, p)| {
                        // Calculate safe numeric values outside the json! macro
                        let safe_pos = [
                            if p.position.x.is_finite() { p.position.x } else { 0.0 },
                            if p.position.y.is_finite() { p.position.y } else { 0.0 },
                            if p.position.z.is_finite() { p.position.z } else { 0.0 }
                        ];
                        let safe_momentum = [
                            if p.momentum.x.is_finite() { p.momentum.x } else { 0.0 },
                            if p.momentum.y.is_finite() { p.momentum.y } else { 0.0 },
                            if p.momentum.z.is_finite() { p.momentum.z } else { 0.0 }
                        ];
                        let safe_energy = if p.energy.is_finite() { p.energy } else { 0.0 };
                        let safe_mass = if p.mass.is_finite() { p.mass } else { 0.0 };
                        let safe_charge = if p.electric_charge.is_finite() { p.electric_charge } else { 0.0 };
                        let safe_spin = [
                            if p.spin.x.re.is_finite() { p.spin.x.re } else { 0.0 },
                            if p.spin.y.re.is_finite() { p.spin.y.re } else { 0.0 },
                            if p.spin.z.re.is_finite() { p.spin.z.re } else { 0.0 }
                        ];
                        let safe_age = {
                            let age = physics.current_time - p.creation_time;
                            if age.is_finite() && age >= 0.0 { age } else { 0.0 }
                        };
                        let safe_decay_prob = if let Some(decay_time) = p.decay_time { 
                            let time_diff = physics.current_time - p.creation_time;
                            if decay_time > 0.0 && time_diff.is_finite() && time_diff >= 0.0 {
                                let prob = 1.0 - ((-time_diff / decay_time).exp());
                                if prob.is_finite() { prob.clamp(0.0, 1.0) } else { 0.0 }
                            } else { 0.0 }
                        } else { 0.0 };

                        json!({
                            "id": i,
                            "particle_type": format!("{:?}", p.particle_type),
                            "position": safe_pos,
                            "momentum": safe_momentum,
                            "energy": safe_energy,
                            "mass": safe_mass,
                            "charge": safe_charge,
                            "spin": safe_spin,
                            "color_charge": p.color_charge.as_ref().map(|c| format!("{:?}", c)),
                            "interaction_count": p.interaction_history.len() as u32,
                            "age": safe_age,
                            "decay_probability": safe_decay_prob
                        })
                    })
                    .collect();
                
                // Extract raw particles for delta processing
                let raw_particles: Vec<_> = physics.particles.clone();
                
                // Extract nuclear data as strings (frontend expects String type)
                let nuclei_data: Vec<String> = physics.nuclei
                    .iter()
                    .take(50)
                    .map(|n| format!("{}{}(A={}, Z={}, BE={:.2e}J)", 
                        match n.atomic_number {
                            1 => "H", 2 => "He", 3 => "Li", 6 => "C", 8 => "O", 26 => "Fe", _ => "X"
                        },
                        n.mass_number, n.mass_number, n.atomic_number, n.binding_energy))
                    .collect();
                
                // Extract atomic data as strings (frontend expects String type)
                let atoms_data: Vec<String> = physics.atoms
                    .iter()
                    .take(50)
                    .map(|a| format!("{}{}(e-={}, E={:.2e}J)", 
                        match a.nucleus.atomic_number {
                            1 => "H", 2 => "He", 3 => "Li", 6 => "C", 8 => "O", 26 => "Fe", _ => "X"
                        },
                        a.nucleus.mass_number, a.electrons.len(), a.total_energy))
                    .collect();
                
                (particle_data, raw_particles, nuclei_data, atoms_data, physics.temperature, physics.energy_density)
            };
            
            let (particle_data, raw_particles, nuclei_data, atoms_data, temperature, energy_density) = physics_data;
            
            // Extract celestial body data (stars, planets, etc.)
            let celestial_bodies: Vec<serde_json::Value> = sim_guard
                .store
                .celestials
                .iter()
                .take(100)
                .map(|body| {
                    let body_type_str = match body.body_type {
                        CelestialBodyType::Star => "Star",
                        CelestialBodyType::Planet => "Planet",
                        CelestialBodyType::Moon => "Moon",
                        CelestialBodyType::Asteroid => "Asteroid",
                        CelestialBodyType::BlackHole => "BlackHole",
                        CelestialBodyType::NeutronStar => "NeutronStar",
                        CelestialBodyType::WhiteDwarf => "WhiteDwarf",
                        CelestialBodyType::BrownDwarf => "BrownDwarf",
                    };

                    json!({
                        "id": body.id.to_string(),
                        "body_type": body_type_str,
                        "position": [0.0, 0.0, 0.0], // Placeholder until position integrated
                        "velocity": [0.0, 0.0, 0.0], // Placeholder until velocity integrated
                        "mass": if body.mass.is_finite() { body.mass } else { 0.0 },
                        "radius": if body.radius.is_finite() { body.radius } else { 0.0 },
                        "temperature": if body.temperature.is_finite() { body.temperature } else { 0.0 },
                        "luminosity": if body.luminosity.is_finite() { body.luminosity } else { 0.0 },
                        "age": if body.age.is_finite() { body.age } else { 0.0 }
                    })
                })
                .collect();
            
            // Capture quantum field snapshot (downsampled)
            let qfield_snapshot = sim_guard.get_quantum_field_snapshot();

            // ── Decide full snapshot vs delta ───────────────────────────
            let payload_json = if !sent_initial_state {
                // Full snapshot
                sent_initial_state = true;

                // Build and cache particle snapshots
                last_particle_state.clear();
                for (idx, part) in raw_particles.iter().enumerate() {
                    last_particle_state.insert(idx, ParticleSnapshot {
                        position: [part.position.x, part.position.y, part.position.z],
                        momentum: [part.momentum.x, part.momentum.y, part.momentum.z],
                        energy: part.energy,
                    });
                }

                json!({
                    "kind": "full",
                    "current_tick": _stats.current_tick,
                    "universe_age_gyr": _stats.universe_age_gyr,
                    "universe_description": _stats.universe_description,
                    "temperature": temperature,
                    "energy_density": energy_density,
                    "particles": particle_data,
                    "nuclei": nuclei_data,
                    "atoms": atoms_data,
                    "celestial_bodies": celestial_bodies,
                    "quantum_fields": qfield_snapshot,
                })
            } else {
                // Delta snapshot
                let mut new_particles = Vec::new();
                let mut updated_particles = Vec::new();
                let mut current_map: HashMap<usize, ParticleSnapshot> = HashMap::new();

                for (idx, p) in raw_particles.iter().enumerate() {
                    let snapshot = ParticleSnapshot {
                        position: [p.position.x, p.position.y, p.position.z],
                        momentum: [p.momentum.x, p.momentum.y, p.momentum.z],
                        energy: p.energy,
                    };
                    current_map.insert(idx, snapshot);

                    if !last_particle_state.contains_key(&idx) {
                        // New particle
                        new_particles.push(particle_data[idx].clone());
                    } else {
                        // Possibly updated
                        let prev = last_particle_state.get(&idx).unwrap();
                        let changed = prev.position != snapshot.position || prev.momentum != snapshot.momentum || (prev.energy - snapshot.energy).abs() > 1e-9;
                        if changed {
                            updated_particles.push(particle_data[idx].clone());
                        }
                    }
                }

                // Removed particles
                let mut removed_ids = Vec::new();
                for id in last_particle_state.keys() {
                    if !current_map.contains_key(id) {
                        removed_ids.push(*id);
                    }
                }

                // Update cache
                last_particle_state = current_map;

                json!({
                    "kind": "delta",
                    "current_tick": _stats.current_tick,
                    "new_particles": new_particles,
                    "updated_particles": updated_particles,
                    "removed_particle_ids": removed_ids,
                    "celestial_bodies": celestial_bodies,
                    "quantum_fields": qfield_snapshot,
                })
            };

            drop(sim_guard); // Release the lock quickly

            // ── Serialize and broadcast plain JSON ─────────────────────
            let _json_str = payload_json.to_string();
        }

        let tick_duration = Duration::from_secs_f64(1.0 / config.target_ups);
        if let Some(sleep_duration) = tick_duration.checked_sub(elapsed) {
            sleep(sleep_duration).await;
        }
    }

    Ok(())
}

async fn cmd_stop() -> Result<()> {
    println!("Stopping simulation...");

    // Build a JSON-RPC stop request
    let client = reqwest::Client::new();
    let req_body = json!({
        "jsonrpc": "2.0",
        "method": "stop",
        "params": {},
        "id": 3
    });

    // Attempt to contact the local RPC server (default port 9001)
    let res = client
        .post("http://127.0.0.1:9001/rpc")
        .json(&req_body)
        .send()
        .await?;

    // Handle connection failures gracefully so the CLI doesn't crash
    if !res.status().is_success() {
        println!("Error: Failed to connect to simulation RPC server.");
        println!("Is the simulation running with 'universectl start'?\n");
        return Ok(());
    }

    // Parse the generic RPC response (payload is not important here)
    let rpc_res: rpc::RpcResponse<serde_json::Value> = res.json().await?;

    if let Some(error) = rpc_res.error {
        println!("RPC Error: {} (code: {})", error.message, error.code);
    } else {
        println!("Stop command sent successfully. Waiting for simulation to shut down...");
    }

    Ok(())
}

async fn cmd_map(zoom: f64, layer: &str) -> Result<()> {
    println!("Universe Map (zoom: {:.1}, layer: {})", zoom, layer);
    println!("================================");
    
    // Try to get real simulation data via RPC
    let client = reqwest::Client::new();
    let params = json!({ "zoom": zoom, "layer": layer });
    let req_body = json!({
        "jsonrpc": "2.0",
        "method": "map",
        "params": params,
        "id": 5
    });

    let width = 60;
    let height = 20;
    
    match client
        .post("http://127.0.0.1:9001/rpc")
        .json(&req_body)
        .send()
        .await
    {
        Ok(res) if res.status().is_success() => {
            // Try to parse simulation data
            if let Ok(rpc_res) = res.json::<rpc::RpcResponse<serde_json::Value>>().await {
                if let Some(map_data) = rpc_res.result {
                    println!("🌌 Connected to simulation - showing real {} data", layer);
                    render_simulation_map(&map_data, width, height, layer)?;
                    return Ok(());
                }
            }
        }
        _ => {
            println!("⚠️  Could not connect to simulation RPC server");
            println!("💡 Start simulation with 'universectl start' for real data");
        }
    }
    
    // Fallback to more realistic sample data when simulation isn't running
    render_sample_map(width, height, layer, zoom);
    
    Ok(())
}

fn render_simulation_map(map_data: &serde_json::Value, fallback_width: usize, fallback_height: usize, layer: &str) -> Result<()> {
    // Determine grid size from simulation response, with sane fallbacks
    let width = map_data
        .get("width")
        .and_then(|v| v.as_u64())
        .map(|v| v as usize)
        .unwrap_or(fallback_width);
    let height = map_data
        .get("height")
        .and_then(|v| v.as_u64())
        .map(|v| v as usize)
        .unwrap_or(fallback_height);
    
    if let Some(grid) = map_data.get("density_grid").and_then(|g| g.as_array()) {
        // Guard against malformed payloads
        if grid.len() != width * height {
            render_sample_map(fallback_width, fallback_height, layer, 1.0);
            print_map_legend(layer);
            return Ok(());
        }
        
        for y in 0..height {
            for x in 0..width {
                let idx = y * width + x;
                let density = grid
                    .get(idx)
                    .and_then(|v| v.as_f64())
                    .unwrap_or(0.0);
                let ch = density_to_char(density, layer);
                print!("{}", ch);
            }
            println!();
        }
    } else {
        // Unexpected format – fallback to procedural sample
        render_sample_map(fallback_width, fallback_height, layer, 1.0);
    }
    
    print_map_legend(layer);
    Ok(())
}

fn render_sample_map(width: usize, height: usize, layer: &str, zoom: f64) {
    println!("📍 Showing sample {} data (simulation not connected)", layer);
    
    for y in 0..height {
        for x in 0..width {
            // More realistic cosmic structure simulation
            let nx = (x as f64 / width as f64 - 0.5) * zoom;
            let ny = (y as f64 / height as f64 - 0.5) * zoom;
            
            let density = match layer {
                "dark_matter" => {
                    // Simulate dark matter filaments
                    let filament = (nx * 10.0).sin() * (ny * 10.0).cos();
                    let halo = (-((nx * nx + ny * ny) * 20.0)).exp();
                    (filament * 0.3 + halo * 0.7 + 0.1).max(0.0).min(1.0)
                },
                "gas" => {
                    // Simulate gas distribution following dark matter
                    let shock_fronts = ((nx * 15.0).sin() + (ny * 12.0).cos()) * 0.5 + 0.5;
                    let void_regions = (-((nx * nx + ny * ny) * 5.0)).exp();
                    (shock_fronts * 0.6 + void_regions * 0.4).max(0.0).min(1.0)
                },
                "stars" => {
                    // Stellar density peaks in galaxy clusters
                    let cluster_1 = (-((nx + 0.3).powi(2) + (ny - 0.2).powi(2)) * 30.0).exp();
                    let cluster_2 = (-((nx - 0.4).powi(2) + (ny + 0.1).powi(2)) * 40.0).exp();
                    let background = ((nx * 25.0 + ny * 17.0).sin() * 0.1).max(0.0);
                    (cluster_1 + cluster_2 + background).max(0.0).min(1.0)
                },
                "radiation" => {
                    // Cosmic microwave background + point sources
                    let cmb = 0.1 + ((nx * 50.0).sin() + (ny * 47.0).cos()) * 0.05;
                    let agn = (-((nx - 0.1).powi(2) + (ny + 0.3).powi(2)) * 100.0).exp() * 0.8;
                    (cmb + agn).max(0.0).min(1.0)
                },
                _ => {
                    // Default density pattern
                    let base = ((nx * 8.0).sin() * (ny * 6.0).cos()).abs();
                    let structure = (-((nx * nx + ny * ny) * 3.0)).exp();
                    (base * 0.7 + structure * 0.3).max(0.0).min(1.0)
                }
            };
            
            let char = density_to_char(density, layer);
            print!("{}", char);
        }
        println!();
    }
    
    print_map_legend(layer);
}

fn density_to_char(density: f64, layer: &str) -> char {
    match layer {
        "dark_matter" => {
            match density {
                d if d < 0.1 => ' ',
                d if d < 0.3 => '░',
                d if d < 0.6 => '▒',
                d if d < 0.8 => '▓',
                _ => '█',
            }
        },
        "gas" => {
            match density {
                d if d < 0.2 => ' ',
                d if d < 0.4 => '.',
                d if d < 0.6 => ':',
                d if d < 0.8 => '~',
                _ => '≈',
            }
        },
        "stars" => {
            match density {
                d if d < 0.1 => ' ',
                d if d < 0.3 => '·',
                d if d < 0.5 => '+',
                d if d < 0.7 => '*',
                d if d < 0.9 => '✦',
                _ => '★',
            }
        },
        "radiation" => {
            match density {
                d if d < 0.2 => ' ',
                d if d < 0.4 => '.',
                d if d < 0.6 => 'o',
                d if d < 0.8 => 'O',
                _ => '◉',
            }
        },
        _ => {
            match density {
                d if d < 0.2 => ' ',
                d if d < 0.4 => '.',
                d if d < 0.6 => ':',
                d if d < 0.8 => '*',
                _ => '#',
            }
        }
    }
}

fn print_map_legend(layer: &str) {
    println!();
    match layer {
        "dark_matter" => println!("Legend: [space]=void, ░=thin, ▒=moderate, ▓=dense, █=very dense"),
        "gas" => println!("Legend: [space]=void, .=tenuous, :=moderate, ~=dense, ≈=shock fronts"),
        "stars" => println!("Legend: [space]=void, ·=sparse, +=moderate, *=dense, ✦=cluster, ★=core"),
        "radiation" => println!("Legend: [space]=cold, .=CMB, o=warm, O=hot, ◉=AGN/quasars"),
        _ => println!("Legend: [space]=void, .=gas, :=stars, *=dense, #=very dense"),
    }
    println!("Available layers: stars, gas, dark_matter, radiation");
}

async fn cmd_list_planets(class: Option<String>, habitable: bool) -> Result<()> {
    println!("Planetary Bodies");
    println!("================");
    
    // Try to get real simulation data via RPC
    let client = reqwest::Client::new();
    let params = json!({ 
        "class_filter": class, 
        "habitable_only": habitable 
    });
    let req_body = json!({
        "jsonrpc": "2.0",
        "method": "list_planets",
        "params": params,
        "id": 6
    });

    match client
        .post("http://127.0.0.1:9001/rpc")
        .json(&req_body)
        .send()
        .await
    {
        Ok(res) if res.status().is_success() => {
            if let Ok(rpc_res) = res.json::<rpc::RpcResponse<serde_json::Value>>().await {
                if let Some(planets_data) = rpc_res.result {
                    println!("🌍 Connected to simulation - showing real data");
                    render_planet_list(&planets_data, &class, habitable)?;
                    return Ok(());
                }
            }
        }
        _ => {
            println!("⚠️  Could not connect to simulation RPC server");
            println!("📍 Showing sample planetary data");
            println!("💡 Start simulation with 'universectl start' for real data\n");
        }
    }
    
    // Fallback to sample data when simulation isn't running
    render_sample_planets(&class, habitable);
    
    Ok(())
}

fn render_planet_list(planets_data: &serde_json::Value, class_filter: &Option<String>, habitable_only: bool) -> Result<()> {
    println!("ID         | Class | Temp (°C) | Water | O2   | Radiation | Habitable | Age (Gyr)");
    println!("-----------|-------|-----------|-------|------|-----------|-----------|----------");
    
    if let Some(planets) = planets_data.as_array() {
        for planet in planets {
            let id = planet.get("id").and_then(|v| v.as_str()).unwrap_or("UNKNOWN");
            let pclass = planet.get("class").and_then(|v| v.as_str()).unwrap_or("?");
            let temp = planet.get("temperature").and_then(|v| v.as_f64()).unwrap_or(0.0);
            let water = planet.get("water_fraction").and_then(|v| v.as_f64()).unwrap_or(0.0);
            let o2 = planet.get("oxygen_fraction").and_then(|v| v.as_f64()).unwrap_or(0.0);
            let radiation = planet.get("radiation_level").and_then(|v| v.as_f64()).unwrap_or(0.0);
            let habitable = planet.get("habitable").and_then(|v| v.as_bool()).unwrap_or(false);
            let age = planet.get("age_gyr").and_then(|v| v.as_f64()).unwrap_or(0.0);
            
            // Apply filters
            if let Some(ref filter_class) = class_filter {
                if pclass != filter_class {
                    continue;
                }
            }
            
            if habitable_only && !habitable {
                continue;
            }
            
            println!("{:<10} | {:<5} | {:<9.1} | {:<5.2} | {:<4.3} | {:<9.3} | {:<9} | {:<.2}", 
                     id, pclass, temp, water, o2, radiation, 
                     if habitable { "Yes" } else { "No" }, age);
        }
    } else {
        println!("Error: Unexpected data format from simulation");
    }
    
    Ok(())
}

fn render_sample_planets(class_filter: &Option<String>, habitable_only: bool) {
    println!("ID         | Class | Temp (°C) | Water | O2   | Radiation | Habitable | Age (Gyr)");
    println!("-----------|-------|-----------|-------|------|-----------|-----------|----------");
    
    let planets = [
        ("SOL-3", "E", 15.0, 0.71, 0.21, 0.002, true, 4.54),
        ("PROX-B", "D", -60.0, 0.01, 0.001, 0.1, false, 4.85),
        ("KEPLER-442B", "E", 5.0, 0.45, 0.18, 0.003, true, 2.9),
        ("TRAPPIST-1E", "E", -22.0, 0.3, 0.15, 0.005, true, 7.6),
        ("HD-40307G", "E", 22.0, 0.6, 0.19, 0.002, true, 4.2),
        ("GLIESE-667CC", "D", 45.0, 0.1, 0.05, 0.02, false, 6.0),
        ("K2-18B", "I", -40.0, 0.8, 0.12, 0.01, false, 3.4),
        ("TOI-715B", "E", 10.0, 0.5, 0.22, 0.001, true, 6.6),
        ("WASP-96B", "G", 1200.0, 0.0, 0.0, 0.5, false, 1.3),
        ("55-CANCRI-E", "T", 2000.0, 0.0, 0.0, 1.2, false, 8.2),
    ];
    
    for (id, pclass, temp, water, o2, rad, hab, age) in planets {
        if let Some(ref filter_class) = class_filter {
            if pclass != filter_class {
                continue;
            }
        }
        
        if habitable_only && !hab {
            continue;
        }
        
        println!("{:<10} | {:<5} | {:<9.1} | {:<5.2} | {:<4.3} | {:<9.3} | {:<9} | {:<.2}", 
                 id, pclass, temp, water, o2, rad, 
                 if hab { "Yes" } else { "No" }, age);
    }
    
    println!("\nPlanet Classes:");
    println!("E = Earth-like (temperate, water, oxygen)");
    println!("D = Desert (dry, thin atmosphere)");
    println!("I = Ice (frozen, thick atmosphere)");  
    println!("T = Toxic (hostile chemistry)");
    println!("G = Gas dwarf (no solid surface)");
}

fn render_planet_inspection(planet_data: &serde_json::Value) -> Result<()> {
    println!("=== PLANET DETAILED INSPECTION ===");
    
    if let Some(id) = planet_data.get("id").and_then(|v| v.as_str()) {
        println!("Planet ID: {}", id);
    }
    
    if let Some(class) = planet_data.get("class").and_then(|v| v.as_str()) {
        println!("Class: {} ({})", class, match class {
            "E" => "Earth-like",
            "D" => "Desert world", 
            "I" => "Ice world",
            "T" => "Toxic world",
            "G" => "Gas dwarf",
            _ => "Unknown"
        });
    }
    
    if let Some(mass) = planet_data.get("mass_earth").and_then(|v| v.as_f64()) {
        println!("Mass: {:.2} Earth masses", mass);
    }
    
    if let Some(radius) = planet_data.get("radius_earth").and_then(|v| v.as_f64()) {
        println!("Radius: {:.2} Earth radii", radius);
    }
    
    if let Some(temp) = planet_data.get("temperature").and_then(|v| v.as_f64()) {
        println!("Temperature: {:.1}°C", temp);
    }
    
    if let Some(water) = planet_data.get("water_fraction").and_then(|v| v.as_f64()) {
        println!("Water coverage: {:.1}%", water * 100.0);
    }
    
    if let Some(oxygen) = planet_data.get("oxygen_fraction").and_then(|v| v.as_f64()) {
        println!("Atmospheric O₂: {:.1}%", oxygen * 100.0);
    }
    
    if let Some(radiation) = planet_data.get("radiation_level").and_then(|v| v.as_f64()) {
        println!("Radiation level: {:.3} (relative)", radiation);
    }
    
    if let Some(age) = planet_data.get("age_gyr").and_then(|v| v.as_f64()) {
        println!("Age: {:.1} billion years", age);
    }
    
    if let Some(habitable) = planet_data.get("habitable").and_then(|v| v.as_bool()) {
        println!("Habitable: {}", if habitable { "Yes" } else { "No" });
    }
    
    Ok(())
}

fn render_sample_planet_inspection(id: &str) {
    println!("=== PLANET DETAILED INSPECTION (SAMPLE DATA) ===");
    println!("Planet ID: {}", id);
    println!("Class: E (Earth-like)");
    println!("Mass: 1.1 Earth masses");
    println!("Radius: 1.05 Earth radii");
    println!("Temperature: 18.5°C");
    println!("Water coverage: 67.0%");
    println!("Atmospheric O₂: 19.0%");
    println!("Radiation level: 0.003 (relative)");
    println!("Age: 3.2 billion years");
    println!("Habitable: Yes");
    println!("\nNote: Connect to running simulation for real data");
}

fn render_lineage_inspection(lineage_data: &serde_json::Value) -> Result<()> {
    println!("=== LINEAGE DETAILED INSPECTION ===");
    
    if let Some(id) = lineage_data.get("id").and_then(|v| v.as_str()) {
        println!("Lineage ID: {}", id);
    }
    
    if let Some(generation) = lineage_data.get("generation").and_then(|v| v.as_u64()) {
        println!("Generation: {}", generation);
    }
    
    if let Some(population) = lineage_data.get("population").and_then(|v| v.as_u64()) {
        println!("Current population: {}", population);
    }
    
    if let Some(fitness) = lineage_data.get("average_fitness").and_then(|v| v.as_f64()) {
        println!("Average fitness: {:.3}", fitness);
    }
    
    if let Some(sentience) = lineage_data.get("sentience_level").and_then(|v| v.as_f64()) {
        println!("Sentience level: {:.2}%", sentience * 100.0);
    }
    
    if let Some(tech) = lineage_data.get("tech_level").and_then(|v| v.as_f64()) {
        println!("Technology level: {:.2}%", tech * 100.0);
    }
    
    if let Some(immortal) = lineage_data.get("immortality_achieved").and_then(|v| v.as_bool()) {
        println!("Immortality achieved: {}", if immortal { "Yes" } else { "No" });
    }
    
    Ok(())
}

fn render_sample_lineage_inspection(id: &str) {
    println!("=== LINEAGE DETAILED INSPECTION (SAMPLE DATA) ===");
    println!("Lineage ID: {}", id);
    println!("Generation: 42");
    println!("Current population: 1,247");
    println!("Average fitness: 0.742");
    println!("Sentience level: 23.5%");
    println!("Technology level: 12.8%");
    println!("Immortality achieved: No");
    println!("\nNote: Connect to running simulation for real data");
}

fn render_universe_stats(stats_data: &serde_json::Value) -> Result<()> {
    println!("=== UNIVERSE STATISTICS ===");
    
    if let Some(age) = stats_data.get("age_gyr").and_then(|v| v.as_f64()) {
        println!("Universe age: {:.2} billion years", age);
    }
    
    if let Some(description) = stats_data.get("universe_description").and_then(|v| v.as_str()) {
        println!("Universe state: {}", description);
    }
    
    if let Some(particles) = stats_data.get("total_particles").and_then(|v| v.as_u64()) {
        println!("Total particles: {}", particles);
    }
    
    if let Some(stars) = stats_data.get("star_count").and_then(|v| v.as_u64()) {
        println!("Star count: {}", stars);
    }
    
    if let Some(planets) = stats_data.get("planet_count").and_then(|v| v.as_u64()) {
        println!("Planet count: {}", planets);
    }
    
    if let Some(lineages) = stats_data.get("lineage_count").and_then(|v| v.as_u64()) {
        println!("Active lineages: {}", lineages);
    }
    
    if let Some(temp) = stats_data.get("average_temperature").and_then(|v| v.as_f64()) {
        println!("Average temperature: {:.1} K", temp);
    }
    
    if let Some(energy) = stats_data.get("total_energy").and_then(|v| v.as_f64()) {
        println!("Total energy: {:.2e} J", energy);
    }
    
    Ok(())
}

fn render_sample_universe_stats() {
    println!("=== UNIVERSE STATISTICS (SAMPLE DATA) ===");
    println!("Universe age: 13.8 billion years");
    println!("Universe state: Universe age 13.8 Gyr: Life complexity 8.5. Biological evolution developing.");
    println!("Total particles: 10,847,293,521");
    println!("Star count: 2,847");
    println!("Planet count: 8,291");
    println!("Active lineages: 127");
    println!("Average temperature: 2.7 K");
    println!("Total energy: 4.23e+42 J");
    println!("\nNote: Connect to running simulation for real data");
}

fn render_physics_diagnostics(diagnostics_data: &serde_json::Value) -> Result<()> {
    println!("=== PHYSICS ENGINE DIAGNOSTICS ===\n");

    // Performance metrics
    println!("🚀 Performance Metrics:");
    if let Some(step_time) = diagnostics_data.get("average_step_time_ms").and_then(|v| v.as_f64()) {
        println!("  Average physics step time: {:.3} ms", step_time);
    }
    
    if let Some(step_95th) = diagnostics_data.get("physics_step_95th_percentile_ms").and_then(|v| v.as_f64()) {
        println!("  95th percentile step time: {:.3} ms", step_95th);
    }
    
    if let Some(tick_time) = diagnostics_data.get("universe_tick_time_ms").and_then(|v| v.as_f64()) {
        println!("  Average universe tick time: {:.3} ms", tick_time);
    }

    // Interaction statistics
    println!("\n⚛️  Nuclear Physics Events:");
    if let Some(interactions) = diagnostics_data.get("interactions_per_step").and_then(|v| v.as_u64()) {
        println!("  Interactions per step: {}", interactions);
    }
    if let Some(fusion_events) = diagnostics_data.get("fusion_events").and_then(|v| v.as_u64()) {
        println!("  Fusion events: {}", fusion_events);
    }
    if let Some(fission_events) = diagnostics_data.get("fission_events").and_then(|v| v.as_u64()) {
        println!("  Fission events: {}", fission_events);
    }
    if let Some(decay_events) = diagnostics_data.get("particle_decay_events").and_then(|v| v.as_u64()) {
        println!("  Particle decay events: {}", decay_events);
    }

    println!("\n⚡️ Electromagnetic & Weak Events:");
    if let Some(compton_events) = diagnostics_data.get("compton_scattering_events").and_then(|v| v.as_u64()) {
        println!("  Compton scattering events: {}", compton_events);
    }
    if let Some(pair_events) = diagnostics_data.get("pair_production_events").and_then(|v| v.as_u64()) {
        println!("  Pair production events: {}", pair_events);
    }
    if let Some(neutrino_events) = diagnostics_data.get("neutrino_scattering_events").and_then(|v| v.as_u64()) {
        println!("  Neutrino scattering events: {}", neutrino_events);
    }

    // System state
    println!("\n🌡️  System State:");
    if let Some(temp) = diagnostics_data.get("system_temperature").and_then(|v| v.as_f64()) {
        println!("  System temperature: {:.1} K", temp);
    }
    
    if let Some(pressure) = diagnostics_data.get("system_pressure").and_then(|v| v.as_f64()) {
        println!("  System pressure: {:.2e} Pa", pressure);
    }
    
    if let Some(energy_conservation) = diagnostics_data.get("energy_conservation_error").and_then(|v| v.as_f64()) {
        println!("  Energy conservation error: {:.2e}", energy_conservation);
    }

    // Resource usage
    println!("\n💾 Resource Usage:");
    if let Some(memory_mb) = diagnostics_data.get("memory_usage_mb").and_then(|v| v.as_f64()) {
        println!("  Memory usage: {:.1} MB", memory_mb);
    }
    
    if let Some(cpu_percent) = diagnostics_data.get("cpu_usage_percent").and_then(|v| v.as_f64()) {
        println!("  CPU usage: {:.1}%", cpu_percent);
    }
    
    if let Some(allocation_rate) = diagnostics_data.get("allocation_rate_mb_per_sec").and_then(|v| v.as_f64()) {
        println!("  Allocation rate: {:.2} MB/s", allocation_rate);
    }

    // Performance alerts
    println!("\n🔍 Performance Status:");
    if let Some(bottlenecks) = diagnostics_data.get("bottlenecks_detected").and_then(|v| v.as_u64()) {
        if bottlenecks > 0 {
            println!("  ⚠️ Performance bottlenecks detected: {}", bottlenecks);
        } else {
            println!("  ✅ No performance bottlenecks detected");
        }
    }

    // Particle counts
    println!("\n🔬 Particle Inventory:");
    if let Some(particles) = diagnostics_data.get("particle_count").and_then(|v| v.as_u64()) {
        println!("  Total particles: {}", particles);
    }
    
    if let Some(nuclei) = diagnostics_data.get("nuclei_count").and_then(|v| v.as_u64()) {
        println!("  Atomic nuclei: {}", nuclei);
    }
    
    if let Some(atoms) = diagnostics_data.get("atoms_count").and_then(|v| v.as_u64()) {
        println!("  Complete atoms: {}", atoms);
    }
    
    if let Some(molecules) = diagnostics_data.get("molecules_count").and_then(|v| v.as_u64()) {
        println!("  Molecules: {}", molecules);
    }
    
    Ok(())
}

fn render_sample_physics_diagnostics() {
    println!("=== PHYSICS ENGINE DIAGNOSTICS (SAMPLE DATA) ===");
    println!("\n🚀 Performance Metrics:");
    println!("  Average physics step time: 12.3 ms");
    println!("  95th percentile step time: 25.1 ms");
    println!("  Average universe tick time: 15.8 ms");

    println!("\n⚛️  Nuclear Physics Events:");
    println!("  Interactions per step: 284,571");
    println!("  Fusion events: 1,247");
    println!("  Fission events: 23");
    println!("  Particle decay events: 8,291");

    println!("\n⚡️ Electromagnetic & Weak Events:");
    println!("  Compton scattering events: 150,432");
    println!("  Pair production events: 12,345");
    println!("  Neutrino scattering events: 112,233");

    println!("\n🌡️  System State:");
    println!("  System temperature: 2,847.3 K");
    println!("  System pressure: 1.24e+8 Pa");
    println!("  Energy conservation error: 2.3e-12");

    println!("\n💾 Resource Usage:");
    println!("  Memory usage: 512.4 MB");
    println!("  CPU usage: 75.2%");
    println!("  Allocation rate: 10.5 MB/s");

    println!("\n🔍 Performance Status:");
    println!("  ✅ No performance bottlenecks detected");

    println!("\n🔬 Particle Inventory:");
    println!("  Total particles: 1,234,567");
    println!("  Atomic nuclei: 12,345");
    println!("  Complete atoms: 6,789");
    println!("  Molecules: 42");

    println!("\nNote: Connect to a running simulation for real data.");
}

async fn cmd_inspect(target: InspectTarget) -> Result<()> {
    let client = reqwest::Client::new();
    
    match target {
        InspectTarget::Planet { id } => {
            println!("Inspecting planet {}...\n", id);
            
            let params = json!({ "planet_id": id });
            let req_body = json!({
                "jsonrpc": "2.0",
                "method": "inspect_planet",
                "params": params,
                "id": 7
            });

            match client.post("http://127.0.0.1:9001/rpc").json(&req_body).send().await {
                Ok(res) if res.status().is_success() => {
                    if let Ok(rpc_res) = res.json::<rpc::RpcResponse<serde_json::Value>>().await {
                        if let Some(planet_data) = rpc_res.result {
                            render_planet_inspection(&planet_data)?;
                        } else if let Some(error) = rpc_res.error {
                            println!("RPC Error: {} (code: {})", error.message, error.code);
                        }
                    }
                }
                _ => {
                    println!("Warning: Could not connect to simulation. Showing sample data.\n");
                    render_sample_planet_inspection(&id);
                }
            }
        }
        
        InspectTarget::Lineage { id } => {
            println!("Inspecting lineage {}...\n", id);
            
            let params = json!({ "lineage_id": id });
            let req_body = json!({
                "jsonrpc": "2.0",
                "method": "inspect_lineage", 
                "params": params,
                "id": 8
            });

            match client.post("http://127.0.0.1:9001/rpc").json(&req_body).send().await {
                Ok(res) if res.status().is_success() => {
                    if let Ok(rpc_res) = res.json::<rpc::RpcResponse<serde_json::Value>>().await {
                        if let Some(lineage_data) = rpc_res.result {
                            render_lineage_inspection(&lineage_data)?;
                        } else if let Some(error) = rpc_res.error {
                            println!("RPC Error: {} (code: {})", error.message, error.code);
                        }
                    }
                }
                _ => {
                    println!("Warning: Could not connect to simulation. Showing sample data.\n");
                    render_sample_lineage_inspection(&id);
                }
            }
        }
        
        InspectTarget::Universe => {
            println!("Inspecting universe statistics...\n");
            
            let req_body = json!({
                "jsonrpc": "2.0",
                "method": "universe_stats",
                "params": {},
                "id": 9
            });

            match client.post("http://127.0.0.1:9001/rpc").json(&req_body).send().await {
                Ok(res) if res.status().is_success() => {
                    if let Ok(rpc_res) = res.json::<rpc::RpcResponse<serde_json::Value>>().await {
                        if let Some(stats_data) = rpc_res.result {
                            render_universe_stats(&stats_data)?;
                        } else if let Some(error) = rpc_res.error {
                            println!("RPC Error: {} (code: {})", error.message, error.code);
                        }
                    }
                }
                _ => {
                    println!("Warning: Could not connect to simulation. Showing sample data.\n");
                    render_sample_universe_stats();
                }
            }
        }
        
        InspectTarget::Physics => {
            println!("Inspecting physics engine diagnostics...\n");
            
            let req_body = json!({
                "jsonrpc": "2.0",
                "method": "physics_diagnostics",
                "params": {},
                "id": 10
            });

            match client.post("http://127.0.0.1:9001/rpc").json(&req_body).send().await {
                Ok(res) if res.status().is_success() => {
                    if let Ok(rpc_res) = res.json::<rpc::RpcResponse<serde_json::Value>>().await {
                        if let Some(diagnostics_data) = rpc_res.result {
                            render_physics_diagnostics(&diagnostics_data)?;
                        } else if let Some(error) = rpc_res.error {
                            println!("RPC Error: {} (code: {})", error.message, error.code);
                        }
                    }
                }
                _ => {
                    println!("Warning: Could not connect to simulation. Showing sample data.\n");
                    render_sample_physics_diagnostics();
                }
            }
        }
        InspectTarget::UniverseHistory => {
            println!("Inspecting historical trends of universe statistics...\n");
            
            let req_body = json!({
                "jsonrpc": "2.0",
                "method": "universe_stats_history",
                "params": {},
                "id": 11
            });

            match client.post("http://127.0.0.1:9001/rpc").json(&req_body).send().await {
                Ok(res) if res.status().is_success() => {
                    if let Ok(rpc_res) = res.json::<rpc::RpcResponse<serde_json::Value>>().await {
                        if let Some(history_data) = rpc_res.result {
                            render_universe_history(&history_data)?;
                        } else if let Some(error) = rpc_res.error {
                            println!("RPC Error: {} (code: {})", error.message, error.code);
                        }
                    }
                }
                _ => {
                    println!("Warning: Could not connect to simulation. Showing sample data.\n");
                    render_sample_universe_history();
                }
            }
        }
    }
    
    Ok(())
}

async fn cmd_snapshot(file: PathBuf, _format: Option<String>) -> Result<()> {
    println!("Requesting snapshot to be saved to {:?}...", file);

    let client = reqwest::Client::new();
    let params = json!({ "path": file });
    let req_body = json!({
        "jsonrpc": "2.0",
        "method": "snapshot",
        "params": params,
        "id": 2
    });

    let res = client
        .post("http://127.0.0.1:9001/rpc")
        .json(&req_body)
        .send()
        .await?;

    if !res.status().is_success() {
        println!("Error: Failed to connect to simulation RPC server.");
        println!("Is the simulation running with 'universectl start'?");
        return Ok(());
    }

    let rpc_res: rpc::RpcResponse<String> = res.json().await?;

    if let Some(error) = rpc_res.error {
        println!("RPC Error: {} (code: {})", error.message, error.code);
        return Ok(());
    }

    if let Some(result) = rpc_res.result {
        println!("Success: {}", result);
    }

    Ok(())
}

async fn cmd_speed(factor: f64) -> Result<()> {
    println!("Setting simulation speed factor to {}x", factor);

    // Build JSON-RPC request
    let client = reqwest::Client::new();
    let params = json!({ "factor": factor });
    let req_body = json!({
        "jsonrpc": "2.0",
        "method": "speed",
        "params": params,
        "id": 4
    });

    let res = client
        .post("http://127.0.0.1:9001/rpc")
        .json(&req_body)
        .send()
        .await?;

    if !res.status().is_success() {
        println!("Error: Failed to connect to simulation RPC server.");
        println!("Is the simulation running with 'universectl start'?\n");
        return Ok(());
    }

    let rpc_res: rpc::RpcResponse<serde_json::Value> = res.json().await?;

    if let Some(error) = rpc_res.error {
        println!("RPC Error: {} (code: {})", error.message, error.code);
    } else {
        println!("Speed updated successfully.");
    }

    Ok(())
}

async fn cmd_rewind(ticks: u64) -> Result<()> {
    println!("Rewinding simulation by {} ticks...", ticks);

    // Build JSON-RPC request
    let client = reqwest::Client::new();
    let params = json!({ "ticks": ticks });
    let req_body = json!({
        "jsonrpc": "2.0",
        "method": "rewind",
        "params": params,
        "id": 5
    });

    let res = client
        .post("http://127.0.0.1:9001/rpc")
        .json(&req_body)
        .send()
        .await?;

    if !res.status().is_success() {
        println!("Error: Failed to connect to simulation RPC server.\nIs the simulation running with 'universectl start'?\n");
        return Ok(());
    }

    let rpc_res: rpc::RpcResponse<serde_json::Value> = res.json().await?;

    if let Some(error) = rpc_res.error {
        println!("RPC Error: {} (code: {})", error.message, error.code);
    } else {
        println!("Rewind successful.");
    }

    Ok(())
}

async fn cmd_godmode(action: GodModeAction) -> Result<()> {
    println!("Executing God-Mode action...");
    match action {
        GodModeAction::CreateBody { mass, body_type, x, y, z } => {
            println!(
                "Creating body of type '{}' (mass = {} kg) at ({}, {}, {})",
                body_type, mass, x, y, z
            );

            let client = reqwest::Client::new();
            let params = json!({
                "mass": mass,
                "body_type": body_type,
                "position": { "x": x, "y": y, "z": z }
            });
            let req_body = json!({
                "jsonrpc": "2.0",
                "method": "create_body",
                "params": params,
                "id": 10
            });

            let res = client
                .post("http://127.0.0.1:9001/rpc")
                .json(&req_body)
                .send()
                .await?;

            if !res.status().is_success() {
                println!("Error: Failed to reach simulation RPC server.");
            } else {
                let rpc_res: rpc::RpcResponse<serde_json::Value> = res.json().await?;
                if let Some(error) = rpc_res.error {
                    println!("RPC Error: {} (code: {})", error.message, error.code);
                } else {
                    println!("Body creation successful.");
                }
            }
        }
        GodModeAction::DeleteBody { id } => {
            println!("Deleting body with ID '{}'", id);

            let client = reqwest::Client::new();
            let params = json!({ "id": id });
            let req_body = json!({
                "jsonrpc": "2.0",
                "method": "delete_body",
                "params": params,
                "id": 11
            });

            let res = client
                .post("http://127.0.0.1:9001/rpc")
                .json(&req_body)
                .send()
                .await?;

            if !res.status().is_success() {
                println!("Error: Failed to reach simulation RPC server.");
            } else {
                let rpc_res: rpc::RpcResponse<serde_json::Value> = res.json().await?;
                if let Some(error) = rpc_res.error {
                    println!("RPC Error: {} (code: {})", error.message, error.code);
                } else {
                    println!("Body deletion successful.");
                }
            }
        }
        GodModeAction::SetConstant { name, value } => {
            println!("Updating constant '{}' to {}", name, value);

            let client = reqwest::Client::new();
            let params = json!({ "name": name, "value": value });
            let req_body = json!({
                "jsonrpc": "2.0",
                "method": "set_constant",
                "params": params,
                "id": 12
            });

            let res = client
                .post("http://127.0.0.1:9001/rpc")
                .json(&req_body)
                .send()
                .await?;

            if !res.status().is_success() {
                println!("Error: Failed to reach simulation RPC server.");
            } else {
                let rpc_res: rpc::RpcResponse<serde_json::Value> = res.json().await?;
                if let Some(error) = rpc_res.error {
                    println!("RPC Error: {} (code: {})", error.message, error.code);
                } else {
                    println!("Constant updated successfully.");
                }
            }
        }
        GodModeAction::SpawnLineage { code_hash, planet_id } => {
            println!("Spawning lineage with code hash {} on planet {}", code_hash, planet_id);

            let client = reqwest::Client::new();
            let params = json!({ "code_hash": code_hash, "planet_id": planet_id });
            let req_body = json!({
                "jsonrpc": "2.0",
                "method": "spawn_lineage",
                "params": params,
                "id": 13
            });

            let res = client
                .post("http://127.0.0.1:9001/rpc")
                .json(&req_body)
                .send()
                .await?;

            if !res.status().is_success() {
                println!("Error: Failed to reach simulation RPC server.");
            } else {
                let rpc_res: rpc::RpcResponse<serde_json::Value> = res.json().await?;
                if let Some(error) = rpc_res.error {
                    println!("RPC Error: {} (code: {})", error.message, error.code);
                } else {
                    println!("Lineage spawned successfully.");
                }
            }
        }
        GodModeAction::Miracle { planet_id, miracle_type, duration, intensity } => {
            println!("Performing miracle '{}' on planet {}", miracle_type, planet_id);
            
            let client = reqwest::Client::new();
            let req_body = json!({
                "jsonrpc": "2.0",
                "method": "perform_miracle",
                "params": {
                    "planet_id": planet_id,
                    "miracle_type": miracle_type,
                    "duration": duration,
                    "intensity": intensity
                },
                "id": 16
            });

            let res = client
                .post("http://127.0.0.1:9001/rpc")
                .json(&req_body)
                .send()
                .await?;

            if !res.status().is_success() {
                println!("Error: Failed to reach simulation RPC server.");
            } else {
                let rpc_res: rpc::RpcResponse<String> = res.json().await?;
                if let Some(error) = rpc_res.error {
                    println!("RPC Error: {} (code: {})", error.message, error.code);
                } else {
                    println!("Divine miracle '{}' successfully enacted upon planet {}", miracle_type, planet_id);
                    if let Some(dur) = duration {
                        println!("Miracle will last for {} ticks", dur);
                    }
                    if let Some(int) = intensity {
                        println!("Miracle intensity: {}", int);
                    }
                }
            }
        }
        GodModeAction::TimeWarp { factor } => {
            println!("Applying time warp with factor: {}", factor);
            
            let client = reqwest::Client::new();
            let req_body = json!({
                "jsonrpc": "2.0",
                "method": "time_warp",
                "params": {
                    "factor": factor
                },
                "id": 17
            });

            let res = client
                .post("http://127.0.0.1:9001/rpc")
                .json(&req_body)
                .send()
                .await?;

            if !res.status().is_success() {
                println!("Error: Failed to reach simulation RPC server.");
            } else {
                let rpc_res: rpc::RpcResponse<String> = res.json().await?;
                if let Some(error) = rpc_res.error {
                    println!("RPC Error: {} (code: {})", error.message, error.code);
                } else {
                    println!("Time warp applied successfully. Reality is now accelerated by {}x", factor);
                }
            }
        }
        GodModeAction::InspectEval { expression } => {
            println!("Evaluating expression: {}", expression);
            
            let client = reqwest::Client::new();
            let req_body = json!({
                "jsonrpc": "2.0",
                "method": "inspect_eval",
                "params": {
                    "expression": expression
                },
                "id": 18
            });

            let res = client
                .post("http://127.0.0.1:9001/rpc")
                .json(&req_body)
                .send()
                .await?;

            if !res.status().is_success() {
                println!("Error: Failed to reach simulation RPC server.");
            } else {
                let rpc_res: rpc::RpcResponse<String> = res.json().await?;
                if let Some(error) = rpc_res.error {
                    println!("RPC Error: {} (code: {})", error.message, error.code);
                } else if let Some(result) = rpc_res.result {
                    println!("Expression result: {}", result);
                } else {
                    println!("Expression evaluated successfully (no return value)");
                }
            }
        }
        GodModeAction::CreateAgent { planet_id } => {
            println!("Creating new life lineage on planet {}", planet_id);

            let client = reqwest::Client::new();
            let params = json!({ "planet_id": planet_id });
            let req_body = json!({
                "jsonrpc": "2.0",
                "method": "godmode_create_agent",
                "params": params,
                "id": 19
            });

            let res = client
                .post("http://127.0.0.1:9001/rpc")
                .json(&req_body)
                .send()
                .await?;

            if !res.status().is_success() {
                println!("Error: Failed to reach simulation RPC server.");
            } else {
                let rpc_res: rpc::RpcResponse<serde_json::Value> = res.json().await?;
                if let Some(error) = rpc_res.error {
                    println!("RPC Error: {} (code: {})", error.message, error.code);
                } else {
                    println!("New life lineage created successfully.");
                }
            }
        }
    }
    println!("God-Mode action completed.");
    Ok(())
}

async fn cmd_resources(action: ResourceAction) -> Result<()> {
    match action {
        ResourceAction::Queue => {
            println!("Fetching pending resource requests...");

            let client = reqwest::Client::new();
            let req_body = json!({
                "jsonrpc": "2.0",
                "method": "resources_queue",
                "params": {},
                "id": 20
            });

            let res = client
                .post("http://127.0.0.1:9001/rpc")
                .json(&req_body)
                .send()
                .await?;

            if !res.status().is_success() {
                println!("Error: Failed to reach simulation RPC server.");
            } else {
                let rpc_res: rpc::RpcResponse<Vec<rpc::ResourceRequest>> = res.json().await?;
                if let Some(error) = rpc_res.error {
                    println!("RPC Error: {} (code: {})", error.message, error.code);
                } else if let Some(queue) = rpc_res.result {
                    if queue.is_empty() {
                        println!("No pending requests.");
                    } else {
                        println!("ID       | Resource | Amount | Requested By | Expires");
                        println!("---------|----------|--------|--------------|--------");
                        for r in queue {
                            println!("{:<8} | {:<8} | {:<6} | {:<12} | {}", r.id, r.resource, r.amount, r.requester, r.expires.unwrap_or_else(|| "N/A".into()));
                        }
                    }
                }
            }
        }
        ResourceAction::Grant { id, expires } => {
            println!("Granting resource request {} (expires: {:?})", id, expires);
            
            let client = reqwest::Client::new();
            let req_body = json!({
                "jsonrpc": "2.0",
                "method": "resources_grant",
                "params": {
                    "id": id,
                    "expires": expires
                },
                "id": 22
            });

            let res = client
                .post("http://127.0.0.1:9001/rpc")
                .json(&req_body)
                .send()
                .await?;

            if !res.status().is_success() {
                println!("Error: Failed to reach simulation RPC server.");
            } else {
                let rpc_res: rpc::RpcResponse<String> = res.json().await?;
                if let Some(error) = rpc_res.error {
                    println!("RPC Error: {} (code: {})", error.message, error.code);
                } else {
                    println!("Resource request {} granted successfully", id);
                }
            }
        }
        ResourceAction::Status => {
            println!("Fetching current resource usage...");

            let client = reqwest::Client::new();
            let req_body = json!({
                "jsonrpc": "2.0",
                "method": "resources_status",
                "params": {},
                "id": 21
            });

            let res = client
                .post("http://127.0.0.1:9001/rpc")
                .json(&req_body)
                .send()
                .await?;

            if !res.status().is_success() {
                println!("Error: Failed to reach simulation RPC server.");
            } else {
                let rpc_res: rpc::RpcResponse<rpc::ResourceStatus> = res.json().await?;
                if let Some(error) = rpc_res.error {
                    println!("RPC Error: {} (code: {})", error.message, error.code);
                } else if let Some(status) = rpc_res.result {
                    println!("Resource | In-Use | Limit");
                    println!("---------|--------|------");
                    for (res_name, usage) in status.usage.iter() {
                        let limit = status.limits.get(res_name).cloned().unwrap_or(0);
                        println!("{:<8} | {:<6} | {}", res_name, usage, limit);
                    }
                }
            }
        }
        ResourceAction::Reload => {
            println!("Reloading resource limits...");
            
            let client = reqwest::Client::new();
            let req_body = json!({
                "jsonrpc": "2.0",
                "method": "resources_reload",
                "params": {},
                "id": 23
            });

            let res = client
                .post("http://127.0.0.1:9001/rpc")
                .json(&req_body)
                .send()
                .await?;

            if !res.status().is_success() {
                println!("Error: Failed to reach simulation RPC server.");
            } else {
                let rpc_res: rpc::RpcResponse<String> = res.json().await?;
                if let Some(error) = rpc_res.error {
                    println!("RPC Error: {} (code: {})", error.message, error.code);
                } else {
                    println!("Resource limits reloaded successfully");
                }
            }
        }
    }
    Ok(())
}

async fn cmd_oracle(action: OracleAction) -> Result<()> {
    match action {
        OracleAction::Inbox => {
            // Get pending messages from agents
            let response = rpc::call_rpc("oracle_messages", json!({})).await?;
            
            println!("🔮 Oracle-Link Messages");
            println!("======================");
            
            if let Some(messages) = response.get("messages").and_then(|v| v.as_array()) {
                if messages.is_empty() {
                    println!("No pending messages from agents.");
                } else {
                    for (i, message) in messages.iter().enumerate() {
                        println!("\n📨 Message {}", i + 1);
                        if let Some(content) = message.get("content").and_then(|v| v.as_str()) {
                            println!("Content: {}", content);
                        }
                        if let Some(agent_id) = message.get("agent_id").and_then(|v| v.as_str()) {
                            println!("From Agent: {}", agent_id);
                        }
                        if let Some(timestamp) = message.get("timestamp").and_then(|v| v.as_u64()) {
                            println!("Timestamp: {}", timestamp);
                        }
                    }
                }
            } else {
                println!("Error: Could not retrieve messages");
            }
        },
        
        OracleAction::Reply { petition_id, action, message } => {
            // Send reply to agent
            let response = rpc::call_rpc("oracle_reply", json!({
                "petition_id": petition_id,
                "action": action,
                "message": message
            })).await?;
            
            if response.get("success").and_then(|v| v.as_bool()).unwrap_or(false) {
                println!("✅ Reply sent successfully");
            } else {
                println!("❌ Failed to send reply");
            }
        },
        
        OracleAction::Stats => {
            // Get communication statistics
            let response = rpc::call_rpc("oracle_stats", json!({})).await?;
            
            println!("🔮 Oracle-Link Statistics");
            println!("========================");
            
            if let Some(total_messages) = response.get("total_messages").and_then(|v| v.as_u64()) {
                println!("Total Messages: {}", total_messages);
            }
            if let Some(active_agents) = response.get("active_agents").and_then(|v| v.as_u64()) {
                println!("Active Agents: {}", active_agents);
            }
            if let Some(avg_response_time) = response.get("avg_response_time_ms").and_then(|v| v.as_f64()) {
                println!("Average Response Time: {:.2}ms", avg_response_time);
            }
        }
    }
    Ok(())
}

/// Interactive mode for real-time simulation control and monitoring
async fn cmd_interactive() -> Result<()> {
    use std::io::{self, Write};
    
    println!("🎮 Interactive Simulation Control");
    println!("=================================");
    println!("Commands: status, stats, physics, speed <factor>, map [layer], planets,");
    println!("          stop, rewind <ticks>, snapshot <file>, inspect <type> <id>,");
    println!("          godmode <action>, help, quit");
    println!("Press Ctrl+C to exit\n");
    
    // Real-time monitoring setup - much less frequent to avoid interference
    let mut last_stats_update = Instant::now();
    let stats_update_interval = Duration::from_secs(60); // Only update every minute
    
    loop {
        // Check if we should show status update (but only if no recent command)
        if last_stats_update.elapsed() >= stats_update_interval {
            match rpc::call_rpc("status", json!({})).await {
                Ok(response) => {
                    print!("📊 Status: ");
                    if let Some(age_gyr) = response.get("universe_age_gyr").and_then(|v| v.as_f64()) {
                        print!("Age: {:.2} GYr", age_gyr);
                    }
                    if let Some(description) = response.get("universe_description").and_then(|v| v.as_str()) {
                        // Show just the first part for brevity in interactive mode
                        let brief = description.split(':').next().unwrap_or(description);
                        print!(", State: {}", brief);
                    }
                    println!();
                },
                Err(_) => println!("📊 Status: Simulation offline"),
            }
            last_stats_update = Instant::now();
        }
        
        // Show prompt
        print!("universe> ");
        io::stdout().flush()?;
        
        // Read user input (blocking)
        let mut input_line = String::new();
        match io::stdin().read_line(&mut input_line) {
            Ok(_) => {
                let command = input_line.trim();
                if command.is_empty() {
                    continue;
                }
                
                // Execute command
                match execute_interactive_command(command).await {
                    Ok(should_exit) => {
                        if should_exit {
                            break;
                        }
                    },
                    Err(e) => {
                        println!("❌ Error: {}", e);
                    }
                }
                
                // Reset auto-update timer after command execution
                last_stats_update = Instant::now();
                
                // Add some spacing after command output
                println!();
            },
            Err(e) => {
                println!("❌ Input error: {}", e);
                break;
            }
        }
    }
    
    println!("👋 Exiting interactive mode");
    Ok(())
}

/// Execute interactive command and return whether to exit
async fn execute_interactive_command(command: &str) -> Result<bool> {
    let parts: Vec<&str> = command.split_whitespace().collect();
    if parts.is_empty() {
        return Ok(false);
    }
    
    println!("🔧 Processing command: {}", parts[0]); // Debug output
    
    match parts[0] {
        "quit" | "exit" | "q" => {
            return Ok(true);
        },
        
        "status" => {
            println!("📊 Fetching simulation status...");
            cmd_status().await?;
            println!("✅ Status command completed.");
        },
        
        "stats" => {
            println!("📈 Fetching universe statistics...");  
            match rpc::call_rpc("universe_stats", json!({})).await {
                Ok(response) => {
                    render_universe_stats(&response)?;
                },
                Err(e) => {
                    println!("⚠️  Could not connect to simulation: {}", e);
                    render_sample_universe_stats();
                }
            }
            println!("✅ Stats command completed.");
        },
        
        "physics" => {
            println!("⚗️ Fetching physics diagnostics...");
            match rpc::call_rpc("physics_diagnostics", json!({})).await {
                Ok(response) => {
                    render_physics_diagnostics(&response)?;
                },
                Err(e) => {
                    println!("⚠️  Could not connect to simulation: {}", e);
                    render_sample_physics_diagnostics();
                }
            }
            println!("✅ Physics command completed.");
        },
        
        "speed" => {
            if parts.len() > 1 {
                if let Ok(factor) = parts[1].parse::<f64>() {
                    println!("⏱️ Setting simulation speed to {}x...", factor);
                    match rpc::call_rpc("speed", json!({ "factor": factor })).await {
                        Ok(response) => {
                            if let Some(message) = response.get("message").and_then(|v| v.as_str()) {
                                println!("✅ {}", message);
                            } else {
                                println!("✅ Speed set to {}x", factor);
                            }
                        },
                        Err(e) => {
                            println!("❌ Failed to set speed: {}", e);
                        }
                    }
                    println!("✅ Speed command completed.");
                } else {
                    println!("❌ Invalid speed factor. Use: speed <number>");
                }
            } else {
                println!("❌ Usage: speed <factor>");
            }
        },
        
        "map" => {
            let layer = parts.get(1).unwrap_or(&"stars");
            println!("🗺️ Generating {} map...", layer);
            match rpc::call_rpc("map", json!({ "zoom": 1.0, "layer": layer })).await {
                Ok(response) => {
                    render_simulation_map(&response, 60, 20, layer)?;
                },
                Err(e) => {
                    println!("⚠️  Could not connect to simulation: {}", e);
                    render_sample_map(60, 20, layer, 1.0);
                }
            }
            println!("✅ Map command completed.");
        },
        
        "planets" => {
            println!("🪐 Fetching planetary data...");
            match rpc::call_rpc("list_planets", json!({ "class_filter": null, "habitable_only": false })).await {
                Ok(response) => {
                    render_planet_list(&response, &None, false)?;
                },
                Err(e) => {
                    println!("⚠️  Could not connect to simulation: {}", e);
                    render_sample_planets(&None, false);
                }
            }
            println!("✅ Planets command completed.");
        },

        "stop" => {
            println!("🛑 Stopping simulation...");
            match rpc::call_rpc("stop", json!({})).await {
                Ok(_) => {
                    println!("✅ Stop command sent successfully.");
                },
                Err(e) => {
                    println!("❌ Failed to stop simulation: {}", e);
                }
            }
            println!("✅ Stop command completed.");
            return Ok(true); // Exit after stop
        },

        "rewind" => {
            if parts.len() > 1 {
                if let Ok(ticks) = parts[1].parse::<u64>() {
                    println!("⏪ Rewinding {} ticks...", ticks);
                    match rpc::call_rpc("rewind", json!({ "ticks": ticks })).await {
                        Ok(response) => {
                            if let Some(message) = response.get("message").and_then(|v| v.as_str()) {
                                println!("✅ {}", message);
                            } else {
                                println!("✅ Rewound {} ticks", ticks);
                            }
                        },
                        Err(e) => {
                            println!("❌ Failed to rewind: {}", e);
                        }
                    }
                    println!("✅ Rewind command completed.");
                } else {
                    println!("❌ Invalid tick count. Use: rewind <number>");
                }
            } else {
                println!("❌ Usage: rewind <ticks>");
            }
        },

        "snapshot" => {
            if parts.len() > 1 {
                let filename = parts[1];
                println!("📸 Creating snapshot: {}...", filename);
                match rpc::call_rpc("snapshot", json!({ "path": filename })).await {
                    Ok(response) => {
                        if let Some(status) = response.get("status").and_then(|v| v.as_str()) {
                            println!("✅ {}", status);
                        } else {
                            println!("✅ Snapshot saved to {}", filename);
                        }
                    },
                    Err(e) => {
                        println!("❌ Failed to create snapshot: {}", e);
                    }
                }
                println!("✅ Snapshot command completed.");
            } else {
                println!("❌ Usage: snapshot <filename>");
            }
        },

        "inspect" => {
            if parts.len() > 2 {
                let inspect_type = parts[1];
                let id = parts[2];
                
                match inspect_type {
                    "planet" => {
                        println!("🔍 Inspecting planet {}...", id);
                        match rpc::call_rpc("inspect_planet", json!({ "planet_id": id })).await {
                            Ok(response) => {
                                render_planet_inspection(&response)?;
                            },
                            Err(e) => {
                                println!("⚠️  Could not connect to simulation: {}", e);
                                render_sample_planet_inspection(id);
                            }
                        }
                        println!("✅ Planet inspection completed.");
                    },
                    "lineage" => {
                        println!("🧬 Inspecting lineage {}...", id);
                        match rpc::call_rpc("inspect_lineage", json!({ "lineage_id": id })).await {
                            Ok(response) => {
                                render_lineage_inspection(&response)?;
                            },
                            Err(e) => {
                                println!("⚠️  Could not connect to simulation: {}", e);
                                render_sample_lineage_inspection(id);
                            }
                        }
                        println!("✅ Lineage inspection completed.");
                    },
                    "universe" => {
                        println!("🌌 Inspecting universe...");
                        match rpc::call_rpc("universe_stats", json!({})).await {
                            Ok(response) => {
                                render_universe_stats(&response)?;
                            },
                            Err(e) => {
                                println!("⚠️  Could not connect to simulation: {}", e);
                                render_sample_universe_stats();
                            }
                        }
                        println!("✅ Universe inspection completed.");
                    },
                    "physics" => {
                        println!("⚗️ Inspecting physics...");
                        match rpc::call_rpc("physics_diagnostics", json!({})).await {
                            Ok(response) => {
                                render_physics_diagnostics(&response)?;
                            },
                            Err(e) => {
                                println!("⚠️  Could not connect to simulation: {}", e);
                                render_sample_physics_diagnostics();
                            }
                        }
                        println!("✅ Physics inspection completed.");
                    },
                    "universe_history" => {
                        println!("🔄 Inspecting historical trends of universe statistics...");
                        match rpc::call_rpc("universe_stats_history", json!({})).await {
                            Ok(response) => {
                                render_universe_history(&response)?;
                            },
                            Err(e) => {
                                println!("⚠️  Could not connect to simulation: {}", e);
                                render_sample_universe_history();
                            }
                        }
                        println!("✅ Universe history inspection completed.");
                    },
                    _ => {
                        println!("❌ Invalid inspect type. Use: inspect <planet|lineage|universe|physics|universe_history> [id]");
                    }
                }
            } else if parts.len() == 2 {
                match parts[1] {
                    "universe" => {
                        println!("🌌 Inspecting universe...");
                        match rpc::call_rpc("universe_stats", json!({})).await {
                            Ok(response) => {
                                render_universe_stats(&response)?;
                            },
                            Err(e) => {
                                println!("⚠️  Could not connect to simulation: {}", e);
                                render_sample_universe_stats();
                            }
                        }
                        println!("✅ Universe inspection completed.");
                    },
                    "physics" => {
                        println!("⚗️ Inspecting physics...");
                        match rpc::call_rpc("physics_diagnostics", json!({})).await {
                            Ok(response) => {
                                render_physics_diagnostics(&response)?;
                            },
                            Err(e) => {
                                println!("⚠️  Could not connect to simulation: {}", e);
                                render_sample_physics_diagnostics();
                            }
                        }
                        println!("✅ Physics inspection completed.");
                    },
                    "universe_history" => {
                        println!("🔄 Inspecting historical trends of universe statistics...");
                        match rpc::call_rpc("universe_stats_history", json!({})).await {
                            Ok(response) => {
                                render_universe_history(&response)?;
                            },
                            Err(e) => {
                                println!("⚠️  Could not connect to simulation: {}", e);
                                render_sample_universe_history();
                            }
                        }
                        println!("✅ Universe history inspection completed.");
                    },
                    _ => {
                        println!("❌ Usage: inspect <planet|lineage> <id> OR inspect <universe|physics|universe_history>");
                    }
                }
            } else {
                println!("❌ Usage: inspect <type> [id]");
            }
        },

        "godmode" => {
            if parts.len() > 1 {
                match parts[1] {
                    "create-agent" => {
                        if parts.len() > 2 {
                            let planet_id = parts[2];
                            println!("🧙 Creating agent on planet {}...", planet_id);
                            match rpc::call_rpc("godmode_create_agent", json!({ "planet_id": planet_id })).await {
                                Ok(response) => {
                                    if let Some(lineage_id) = response.get("new_lineage_id").and_then(|v| v.as_str()) {
                                        println!("✅ Created new agent lineage: {}", lineage_id);
                                    } else {
                                        println!("✅ Agent created successfully");
                                    }
                                },
                                Err(e) => {
                                    println!("❌ Failed to create agent: {}", e);
                                }
                            }
                            println!("✅ God-mode create-agent completed.");
                        } else {
                            println!("❌ Usage: godmode create-agent <planet_id>");
                        }
                    },
                    _ => {
                        println!("❌ Available god-mode actions: create-agent");
                    }
                }
            } else {
                println!("❌ Usage: godmode <action> [args]");
            }
        },
        
        "help" => {
            println!("Available commands:");
            println!("  status                    - Show simulation status");
            println!("  stats                     - Show universe statistics");
            println!("  physics                   - Show physics diagnostics");
            println!("  speed <factor>            - Change simulation speed");
            println!("  map [layer]               - Show ASCII map (layers: stars, gas, dark_matter, radiation)");
            println!("  planets                   - List planets");
            println!("  stop                      - Stop simulation and exit");
            println!("  rewind <ticks>            - Rewind simulation by specified ticks");
            println!("  snapshot <filename>       - Create simulation snapshot");
            println!("  inspect planet <id>       - Inspect specific planet");
            println!("  inspect lineage <id>      - Inspect specific lineage");
            println!("  inspect universe          - Inspect universe statistics");
            println!("  inspect physics           - Inspect physics diagnostics");
            println!("  godmode create-agent <id> - Create agent on planet (requires god-mode)");
            println!("  help                      - Show this help message");
            println!("  quit                      - Exit interactive mode");
        },
        
        _ => {
            println!("❌ Unknown command: {}. Type 'help' for available commands.", parts[0]);
        }
    }
    
    Ok(false)
}

// WebSocket server functions removed - replaced with high-performance native renderer

async fn handle_rpc_request(
    request: rpc::RpcRequest,
    state: Arc<Mutex<SharedState>>,
) -> Result<impl warp::Reply, warp::Rejection> {
    let response_id = request.id;
    let mut shared_state = state.lock().unwrap();

    let response = match request.method.as_str() {
        "status" => {
            let mut sim_guard = shared_state.sim.lock().unwrap();
            let stats = sim_guard.get_stats().unwrap();

            let save_file_age_sec = shared_state.last_save_time
                .map(|save_time| save_time.elapsed().as_secs());

            let response_data = rpc::StatusResponse {
                status: "running".to_string(),
                tick: stats.current_tick,
                ups: stats.target_ups,
                universe_age_gyr: stats.universe_age_gyr,
                universe_description: stats.universe_description.clone(),
                lineage_count: stats.lineage_count as u64,
                save_file_age_sec,
            };

            let rpc_response: rpc::RpcResponse<rpc::StatusResponse> = rpc::RpcResponse {
                jsonrpc: "2.0".to_string(),
                result: Some(response_data),
                error: None,
                id: response_id,
            };
            Ok(warp::reply::json(&rpc_response))
        }

        "snapshot" => {
            #[derive(Deserialize)]
            struct SnapshotParams {
                path: PathBuf,
            }

            match serde_json::from_value::<SnapshotParams>(request.params) {
                Ok(params) => {
                    let stats = {
                        let mut sim_guard = shared_state.sim.lock().unwrap();
                        match persistence::save_checkpoint(&mut sim_guard, &params.path) {
                            Ok(_) => sim_guard.get_stats().unwrap(),
                            Err(e) => {
                                let error = rpc::RpcError {
                                    code: rpc::INTERNAL_ERROR,
                                    message: format!("Failed to save snapshot: {}", e),
                                };
                                let rpc_response: rpc::RpcResponse<rpc::StatusResponse> = rpc::RpcResponse {
                                    jsonrpc: "2.0".to_string(),
                                    result: None,
                                    error: Some(error),
                                    id: response_id,
                                };
                                return Ok(warp::reply::json(&rpc_response));
                            }
                        }
                    };
                    
                    // Update save time after releasing the sim lock
                    shared_state.last_save_time = Some(Instant::now());
                            let response_data = rpc::StatusResponse {
                                status: format!("Snapshot saved to {:?}", params.path),
                                tick: stats.current_tick,
                                ups: stats.target_ups,
                                universe_age_gyr: stats.universe_age_gyr,
                                universe_description: stats.universe_description.clone(),
                                lineage_count: stats.lineage_count as u64,
                                save_file_age_sec: Some(0),
                            };
                            let rpc_response: rpc::RpcResponse<rpc::StatusResponse> = rpc::RpcResponse {
                                jsonrpc: "2.0".to_string(),
                                result: Some(response_data),
                                error: None,
                                id: response_id,
                            };
                            Ok(warp::reply::json(&rpc_response))
                }
                Err(e) => {
                    let error = rpc::RpcError {
                        code: rpc::INVALID_PARAMS,
                        message: format!("Invalid parameters for snapshot: {}", e),
                    };
                    let rpc_response: rpc::RpcResponse<rpc::StatusResponse> = rpc::RpcResponse {
                        jsonrpc: "2.0".to_string(),
                        result: None,
                        error: Some(error),
                        id: response_id,
                    };
                    Ok(warp::reply::json(&rpc_response))
                }
            }
        }

        "map" => {
            #[derive(Deserialize)]
            struct MapParams {
                zoom: Option<f64>,
                layer: Option<String>,
            }

            match serde_json::from_value::<MapParams>(request.params) {
                Ok(params) => {
                    let zoom = params.zoom.unwrap_or(1.0);
                    let layer = params.layer.as_deref().unwrap_or("stars");
                    
                    // Get real map data from simulation
                    let mut sim_guard = shared_state.sim.lock().unwrap();
                    let width = 60;
                    let height = 20;
                    
                    let response_data = match sim_guard.get_map_data(zoom, layer, width, height) {
                        Ok(map_data) => map_data,
                        Err(e) => {
                            // Fallback to synthetic data if real data fails
                            warn!("Failed to get real map data: {}, using synthetic data", e);
                            let mut density_grid = Vec::new();
                            
                            for y in 0..height {
                                for x in 0..width {
                                    let nx = (x as f64 / width as f64 - 0.5) * zoom;
                                    let ny = (y as f64 / height as f64 - 0.5) * zoom;
                                    
                                    let density = match layer {
                                        "stars" => {
                                            let cluster_1 = (-((nx + 0.3).powi(2) + (ny - 0.2).powi(2)) * 30.0).exp();
                                            let cluster_2 = (-((nx - 0.4).powi(2) + (ny + 0.1).powi(2)) * 40.0).exp();
                                            (cluster_1 + cluster_2).max(0.0).min(1.0)
                                        },
                                        "gas" => {
                                            let shock_fronts = ((nx * 15.0).sin() + (ny * 12.0).cos()) * 0.5 + 0.5;
                                            let void_regions = (-((nx * nx + ny * ny) * 5.0)).exp();
                                            (shock_fronts * 0.6 + void_regions * 0.4).max(0.0).min(1.0)
                                        },
                                        "dark_matter" => {
                                            let filament = (nx * 10.0).sin() * (ny * 10.0).cos();
                                            let halo = (-((nx * nx + ny * ny) * 20.0)).exp();
                                            (filament * 0.3 + halo * 0.7 + 0.1).max(0.0).min(1.0)
                                        },
                                        _ => 0.1
                                    };
                                    density_grid.push(density);
                                }
                            }
                            
                            json!({
                                "density_grid": density_grid,
                                "width": width,
                                "height": height,
                                "layer": layer,
                                "zoom": zoom,
                                "fallback": true,
                                "generated_at": std::time::SystemTime::now()
                                    .duration_since(std::time::UNIX_EPOCH)
                                    .unwrap()
                                    .as_secs()
                            })
                        }
                    };
                    
                    let rpc_response: rpc::RpcResponse<serde_json::Value> = rpc::RpcResponse {
                        jsonrpc: "2.0".to_string(),
                        result: Some(response_data),
                        error: None,
                        id: response_id,
                    };
                    
                    Ok(warp::reply::json(&rpc_response))
                }
                Err(_) => {
                    let error = rpc::RpcError {
                        code: rpc::INVALID_PARAMS,
                        message: "Invalid map parameters".to_string(),
                    };
                    let rpc_response: rpc::RpcResponse<serde_json::Value> = rpc::RpcResponse {
                        jsonrpc: "2.0".to_string(),
                        result: None,
                        error: Some(error),
                        id: response_id,
                    };
                    Ok(warp::reply::json(&rpc_response))
                }
            }
        }

        "list_planets" => {
            #[derive(Deserialize)]
            struct PlanetParams {
                class_filter: Option<String>,
                habitable_only: Option<bool>,
            }

            match serde_json::from_value::<PlanetParams>(request.params) {
                Ok(params) => {
                    let habitable_only = params.habitable_only.unwrap_or(false);
                    
                    // Get real planet data from simulation
                    let mut sim_guard = shared_state.sim.lock().unwrap();
                    let planets = match sim_guard.get_planet_data(params.class_filter.clone(), habitable_only) {
                        Ok(real_planets) => real_planets,
                        Err(e) => {
                            // Fallback to synthetic data if real data fails
                            warn!("Failed to get real planet data: {}, using synthetic data", e);
                            let mut fallback_planets = vec![
                                json!({
                                    "id": "SIM-001",
                                    "class": "E",
                                    "temperature": 18.5,
                                    "water_fraction": 0.67,
                                    "oxygen_fraction": 0.19,
                                    "radiation_level": 0.003,
                                    "habitable": true,
                                    "age_gyr": 3.2,
                                    "mass_earth": 1.1,
                                    "radius_earth": 1.05,
                                    "fallback": true
                                }),
                                json!({
                                    "id": "SIM-002", 
                                    "class": "D",
                                    "temperature": -45.0,
                                    "water_fraction": 0.02,
                                    "oxygen_fraction": 0.001,
                                    "radiation_level": 0.08,
                                    "habitable": false,
                                    "age_gyr": 5.1,
                                    "mass_earth": 0.8,
                                    "radius_earth": 0.9,
                                    "fallback": true
                                }),
                                json!({
                                    "id": "SIM-003",
                                    "class": "E", 
                                    "temperature": 12.0,
                                    "water_fraction": 0.43,
                                    "oxygen_fraction": 0.16,
                                    "radiation_level": 0.004,
                                    "habitable": true,
                                    "age_gyr": 2.8,
                                    "mass_earth": 0.95,
                                    "radius_earth": 0.98,
                                    "fallback": true
                                }),
                            ];
                            
                            // Apply filters to fallback data
                            if let Some(ref filter_class) = params.class_filter {
                                fallback_planets.retain(|planet| {
                                    planet.get("class").and_then(|v| v.as_str()).unwrap_or("") == filter_class
                                });
                            }
                            
                            if habitable_only {
                                fallback_planets.retain(|planet| {
                                    planet.get("habitable").and_then(|v| v.as_bool()).unwrap_or(false)
                                });
                            }
                            
                            json!(fallback_planets)
                        }
                    };
                    
                    let rpc_response: rpc::RpcResponse<serde_json::Value> = rpc::RpcResponse {
                        jsonrpc: "2.0".to_string(),
                        result: Some(planets),
                        error: None,
                        id: response_id,
                    };
                    
                    Ok(warp::reply::json(&rpc_response))
                }
                Err(_) => {
                    let error = rpc::RpcError {
                        code: rpc::INVALID_PARAMS,
                        message: "Invalid planet parameters".to_string(),
                    };
                    let rpc_response: rpc::RpcResponse<serde_json::Value> = rpc::RpcResponse {
                        jsonrpc: "2.0".to_string(),
                        result: None,
                        error: Some(error),
                        id: response_id,
                    };
                    Ok(warp::reply::json(&rpc_response))
                }
            }
        }

        "stop" => {
            info!("Received stop command via RPC. Performing graceful shutdown...");
            
            // Save final checkpoint before shutdown
            let (final_checkpoint, save_success) = {
                let mut sim_guard = shared_state.sim.lock().unwrap();
                let final_checkpoint = format!("final_checkpoint_{}.bin", sim_guard.current_tick);
                
                let save_success = match persistence::save_checkpoint(&mut sim_guard, &PathBuf::from(&final_checkpoint)) {
                    Ok(_) => {
                        info!("Final checkpoint saved to {}", final_checkpoint);
                        true
                    },
                    Err(e) => {
                        error!("Failed to save final checkpoint: {}", e);
                        false
                    }
                };
                (final_checkpoint, save_success)
            };
            
            if save_success {
                shared_state.last_save_time = Some(Instant::now());
            }
            
            let response_data = json!({
                "status": "stopping",
                "message": "Graceful shutdown initiated",
                "final_checkpoint": final_checkpoint
            });
            
            let rpc_response: rpc::RpcResponse<serde_json::Value> = rpc::RpcResponse {
                jsonrpc: "2.0".to_string(),
                result: Some(response_data),
                error: None,
                id: response_id,
            };
            
            // Give the response a moment to be sent before exiting
            tokio::spawn(async {
                tokio::time::sleep(Duration::from_millis(100)).await;
                info!("Graceful shutdown complete. Exiting...");
                std::process::exit(0);
            });
            
            Ok(warp::reply::json(&rpc_response))
        }

        "universe_stats" => {
            let mut sim_guard = shared_state.sim.lock().unwrap();
            let stats = sim_guard.get_stats().unwrap();
            let physics = &sim_guard.physics_engine;
            let average_temperature = physics.temperature;
            let total_energy = physics.energy_density * physics.volume;
            let universe_stats = json!({
                "age_gyr": stats.universe_age_gyr,
                "universe_description": stats.universe_description.clone(),
                "total_particles": stats.particle_count,
                "star_count": stats.celestial_body_count,
                "planet_count": stats.planet_count,
                "lineage_count": stats.lineage_count,
                "average_temperature": average_temperature,
                "total_energy": total_energy,
                "hubble_constant": 67.4, // km/s/Mpc
                "dark_matter_fraction": 0.264,
                "dark_energy_fraction": 0.686,
                "ordinary_matter_fraction": 0.05
            });
            
            let rpc_response: rpc::RpcResponse<serde_json::Value> = rpc::RpcResponse {
                jsonrpc: "2.0".to_string(),
                result: Some(universe_stats),
                error: None,
                id: response_id,
            };
            
            Ok(warp::reply::json(&rpc_response))
        }

        "physics_diagnostics" => {
            let sim_guard = shared_state.sim.lock().unwrap();
            
            // Get actual physics engine and diagnostics data
            let physics = &sim_guard.physics_engine;
            let diagnostics_system = sim_guard.get_diagnostics();
            
            // Get performance report from integrated diagnostics
            let performance_report = diagnostics_system.get_performance_report();
            
            // Calculate physics-specific metrics
            let average_step_time_ms = if performance_report.metrics.physics_step_times.stats.mean > 0.0 {
                performance_report.metrics.physics_step_times.stats.mean
            } else {
                physics.time_step * 1000.0
            };
            
            let interactions_per_step = if performance_report.metrics.interaction_rates.stats.mean > 0.0 {
                performance_report.metrics.interaction_rates.stats.mean as u64
            } else {
                (physics.compton_count + physics.pair_production_count + physics.neutrino_scatter_count
               + physics.particle_decay_count + physics.fusion_count + physics.fission_count).min(10000)
            };
            
            let system_pressure = physics.calculate_system_pressure();
            
            let diagnostics = json!({
                "average_step_time_ms": average_step_time_ms,
                "interactions_per_step": interactions_per_step,
                "compton_scattering_events": physics.compton_count,
                "pair_production_events": physics.pair_production_count,
                "neutrino_scattering_events": physics.neutrino_scatter_count,
                "particle_decay_events": physics.particle_decay_count,
                "fusion_events": physics.fusion_count,
                "fission_events": physics.fission_count,
                "system_temperature": physics.temperature,
                "system_pressure": system_pressure,
                "energy_conservation_error": 0.0,
                "particle_count": physics.particles.len(),
                "nuclei_count": physics.nuclei.len(),
                "atoms_count": physics.atoms.len(),
                "molecules_count": physics.molecules.len(),
                "current_time": physics.current_time,
                "time_step": physics.time_step,
                // Additional diagnostics metrics
                "physics_step_95th_percentile_ms": performance_report.metrics.physics_step_times.stats.percentile_95,
                "universe_tick_time_ms": performance_report.metrics.universe_tick_times.stats.mean,
                "memory_usage_mb": performance_report.memory_usage.mean / (1024.0 * 1024.0),
                "cpu_usage_percent": performance_report.system_load.cpu_usage,
                "bottlenecks_detected": performance_report.bottlenecks.len(),
                "allocation_rate_mb_per_sec": performance_report.metrics.allocation_rates.stats.mean / (1024.0 * 1024.0)
            });
            
            let rpc_response: rpc::RpcResponse<serde_json::Value> = rpc::RpcResponse {
                jsonrpc: "2.0".to_string(),
                result: Some(diagnostics),
                error: None,
                id: response_id,
            };
            
            Ok(warp::reply::json(&rpc_response))
        }

        "inspect_planet" => {
            #[derive(Deserialize)]
            struct InspectPlanetParams {
                planet_id: String,
            }

            match serde_json::from_value::<InspectPlanetParams>(request.params) {
                Ok(params) => {
                    // Get real planet data from simulation ECS
                    let mut sim_guard = shared_state.sim.lock().unwrap();
                    let planet_data = match sim_guard.get_planet_inspection_data(&params.planet_id) {
                        Ok(Some(real_planet_data)) => real_planet_data,
                        Ok(None) => {
                            // Planet not found in simulation, return error
                            let error = rpc::RpcError {
                                code: rpc::INVALID_PARAMS,
                                message: format!("Planet '{}' not found in simulation", params.planet_id),
                            };
                            let rpc_response: rpc::RpcResponse<serde_json::Value> = rpc::RpcResponse {
                                jsonrpc: "2.0".to_string(),
                                result: None,
                                error: Some(error),
                                id: response_id,
                            };
                            return Ok(warp::reply::json(&rpc_response));
                        }
                        Err(e) => {
                            // Fallback to synthetic data if real data fails
                            warn!("Failed to get real planet data for {}: {}, using fallback", params.planet_id, e);
                            json!({
                                "id": params.planet_id,
                                "class": "E",
                                "mass_earth": 1.1,
                                "radius_earth": 1.05,
                                "temperature": 18.5,
                                "water_fraction": 0.67,
                                "oxygen_fraction": 0.19,
                                "radiation_level": 0.003,
                                "age_gyr": 3.2,
                                "habitable": true,
                                "orbital_distance_au": 1.0,
                                "orbital_period_days": 365.25,
                                "magnetic_field_strength": 0.5, // Relative to Earth
                                "atmospheric_pressure_atm": 1.0,
                                "surface_gravity_g": 1.1,
                                "tidal_locked": false,
                                "fallback": true
                            })
                        }
                    };
                    
                    let rpc_response: rpc::RpcResponse<serde_json::Value> = rpc::RpcResponse {
                        jsonrpc: "2.0".to_string(),
                        result: Some(planet_data),
                        error: None,
                        id: response_id,
                    };
                    
                    Ok(warp::reply::json(&rpc_response))
                }
                Err(_) => {
                    let error = rpc::RpcError {
                        code: rpc::INVALID_PARAMS,
                        message: "Invalid planet inspect parameters".to_string(),
                    };
                    let rpc_response: rpc::RpcResponse<serde_json::Value> = rpc::RpcResponse {
                        jsonrpc: "2.0".to_string(),
                        result: None,
                        error: Some(error),
                        id: response_id,
                    };
                    Ok(warp::reply::json(&rpc_response))
                }
            }
        }

        "inspect_lineage" => {
            #[derive(Deserialize)]
            struct InspectLineageParams {
                lineage_id: String,
            }

            match serde_json::from_value::<InspectLineageParams>(request.params) {
                Ok(params) => {
                    // Get real lineage data from simulation ECS
                    let mut sim_guard = shared_state.sim.lock().unwrap();
                    let lineage_data = match sim_guard.get_lineage_inspection_data(&params.lineage_id) {
                        Ok(Some(real_lineage_data)) => real_lineage_data,
                        Ok(None) => {
                            // Lineage not found in simulation, return error
                            let error = rpc::RpcError {
                                code: rpc::INVALID_PARAMS,
                                message: format!("Lineage '{}' not found in simulation", params.lineage_id),
                            };
                            let rpc_response: rpc::RpcResponse<serde_json::Value> = rpc::RpcResponse {
                                jsonrpc: "2.0".to_string(),
                                result: None,
                                error: Some(error),
                                id: response_id,
                            };
                            return Ok(warp::reply::json(&rpc_response));
                        }
                        Err(e) => {
                            // Fallback to synthetic data if real data fails
                            warn!("Failed to get real lineage data for {}: {}, using fallback", params.lineage_id, e);
                            json!({
                                "id": params.lineage_id,
                                "generation": 42,
                                "population": 1247,
                                "average_fitness": 0.742,
                                "sentience_level": 0.235,
                                "tech_level": 0.128,
                                "industrialization_level": 0.067,
                                "digitalization_level": 0.023,
                                "immortality_achieved": false,
                                "birth_tick": 1250,
                                "last_mutation_tick": 15847,
                                "code_hash": "a7b4c8d9e2f1",
                                "parent_lineage_id": "LIN-003",
                                "dominant_strategies": ["cooperation", "resource_hoarding", "exploration"],
                                "average_energy": 847.2,
                                "reproduction_rate": 0.24,
                                "fallback": true
                            })
                        }
                    };
                    
                    let rpc_response: rpc::RpcResponse<serde_json::Value> = rpc::RpcResponse {
                        jsonrpc: "2.0".to_string(),
                        result: Some(lineage_data),
                        error: None,
                        id: response_id,
                    };
                    
                    Ok(warp::reply::json(&rpc_response))
                }
                Err(_) => {
                    let error = rpc::RpcError {
                        code: rpc::INVALID_PARAMS,
                        message: "Invalid lineage inspect parameters".to_string(),
                    };
                    let rpc_response: rpc::RpcResponse<serde_json::Value> = rpc::RpcResponse {
                        jsonrpc: "2.0".to_string(),
                        result: None,
                        error: Some(error),
                        id: response_id,
                    };
                    Ok(warp::reply::json(&rpc_response))
                }
            }
        }

        "godmode_create_agent" => {
            #[derive(Deserialize)]
            struct AgentParams {
                planet_id: String,
            }

            match serde_json::from_value::<AgentParams>(request.params) {
                Ok(params) => {
                    let mut sim_guard = shared_state.sim.lock().unwrap();
                    match sim_guard.god_create_agent_on_planet(&params.planet_id) {
                        Ok(lineage_id) => {
                            let response_data = json!({
                                "new_lineage_id": lineage_id,
                            });
                            let rpc_response: rpc::RpcResponse<serde_json::Value> = rpc::RpcResponse {
                                jsonrpc: "2.0".to_string(),
                                result: Some(response_data),
                                error: None,
                                id: response_id,
                            };
                            Ok(warp::reply::json(&rpc_response))
                        }
                        Err(e) => {
                            let error = rpc::RpcError {
                                code: rpc::INVALID_PARAMS,
                                message: e.to_string(),
                            };
                            let rpc_response: rpc::RpcResponse<serde_json::Value> = rpc::RpcResponse {
                                jsonrpc: "2.0".to_string(),
                                result: None,
                                error: Some(error),
                                id: response_id,
                            };
                            Ok(warp::reply::json(&rpc_response))
                        }
                    }
                }
                Err(e) => {
                    let error = rpc::RpcError {
                        code: rpc::INVALID_PARAMS,
                        message: format!("Invalid parameters for godmode_create_agent: {}", e),
                    };
                    let rpc_response: rpc::RpcResponse<serde_json::Value> = rpc::RpcResponse {
                        jsonrpc: "2.0".to_string(),
                        result: None,
                        error: Some(error),
                        id: response_id,
                    };
                    Ok(warp::reply::json(&rpc_response))
                }
            }
        }

        "speed" => {
            #[derive(Deserialize)]
            struct SpeedParams {
                factor: f64,
            }

            match serde_json::from_value::<SpeedParams>(request.params) {
                Ok(params) => {
                    let mut sim_guard = shared_state.sim.lock().unwrap();
                    match sim_guard.set_speed_factor(params.factor) {
                        Ok(_) => {
                            let response_data = json!({
                                "speed_factor": params.factor,
                                "message": format!("Simulation speed set to {}x", params.factor)
                            });
                            let rpc_response: rpc::RpcResponse<serde_json::Value> = rpc::RpcResponse {
                                jsonrpc: "2.0".to_string(),
                                result: Some(response_data),
                                error: None,
                                id: response_id,
                            };
                            Ok(warp::reply::json(&rpc_response))
                        }
                        Err(e) => {
                            let error = rpc::RpcError {
                                code: rpc::INVALID_PARAMS,
                                message: format!("Failed to set speed: {}", e),
                            };
                            let rpc_response: rpc::RpcResponse<serde_json::Value> = rpc::RpcResponse {
                                jsonrpc: "2.0".to_string(),
                                result: None,
                                error: Some(error),
                                id: response_id,
                            };
                            Ok(warp::reply::json(&rpc_response))
                        }
                    }
                }
                Err(e) => {
                    let error = rpc::RpcError {
                        code: rpc::INVALID_PARAMS,
                        message: format!("Invalid parameters for speed: {}", e),
                    };
                    let rpc_response: rpc::RpcResponse<serde_json::Value> = rpc::RpcResponse {
                        jsonrpc: "2.0".to_string(),
                        result: None,
                        error: Some(error),
                        id: response_id,
                    };
                    Ok(warp::reply::json(&rpc_response))
                }
            }
        }

        "rewind" => {
            #[derive(Deserialize)]
            struct RewindParams {
                ticks: u64,
            }

            match serde_json::from_value::<RewindParams>(request.params) {
                Ok(params) => {
                    let mut sim_guard = shared_state.sim.lock().unwrap();
                    match sim_guard.rewind_ticks(params.ticks) {
                        Ok(actual_ticks) => {
                            let response_data = json!({
                                "requested_ticks": params.ticks,
                                "actual_ticks": actual_ticks,
                                "message": format!("Rewound {} ticks", actual_ticks)
                            });
                            let rpc_response: rpc::RpcResponse<serde_json::Value> = rpc::RpcResponse {
                                jsonrpc: "2.0".to_string(),
                                result: Some(response_data),
                                error: None,
                                id: response_id,
                            };
                            Ok(warp::reply::json(&rpc_response))
                        }
                        Err(e) => {
                            let error = rpc::RpcError {
                                code: rpc::INVALID_PARAMS,
                                message: format!("Failed to rewind: {}", e),
                            };
                            let rpc_response: rpc::RpcResponse<serde_json::Value> = rpc::RpcResponse {
                                jsonrpc: "2.0".to_string(),
                                result: None,
                                error: Some(error),
                                id: response_id,
                            };
                            Ok(warp::reply::json(&rpc_response))
                        }
                    }
                }
                Err(e) => {
                    let error = rpc::RpcError {
                        code: rpc::INVALID_PARAMS,
                        message: format!("Invalid parameters for rewind: {}", e),
                    };
                    let rpc_response: rpc::RpcResponse<serde_json::Value> = rpc::RpcResponse {
                        jsonrpc: "2.0".to_string(),
                        result: None,
                        error: Some(error),
                        id: response_id,
                    };
                    Ok(warp::reply::json(&rpc_response))
                }
            }
        }

        "universe_stats_history" => {
            let sim_guard = shared_state.sim.lock().unwrap();
            let history_json = sim_guard.get_stats_history_json().unwrap_or_else(|_| serde_json::json!([]));

            let rpc_response: rpc::RpcResponse<serde_json::Value> = rpc::RpcResponse {
                jsonrpc: "2.0".to_string(),
                result: Some(history_json),
                error: None,
                id: response_id,
            };

            Ok(warp::reply::json(&rpc_response))
        }

        _ => {
            let error = rpc::RpcError {
                code: rpc::METHOD_NOT_FOUND,
                message: format!("Method '{}' not found", request.method),
            };
            let rpc_response: rpc::RpcResponse<()> = rpc::RpcResponse {
                jsonrpc: "2.0".to_string(),
                result: None,
                error: Some(error),
                id: response_id,
            };
            Ok(warp::reply::json(&rpc_response))
        }
    };

    response
}

async fn start_rpc_server(port: u16, state: Arc<Mutex<SharedState>>) {
    let rpc_route = warp::path("rpc")
        .and(warp::post())
        .and(warp::body::json())
        .and(with_state(state))
        .and_then(handle_rpc_request);

    println!("RPC server listening on 127.0.0.1:{}", port);
    warp::serve(rpc_route).run(([127, 0, 0, 1], port)).await;
}

fn with_state(
    state: Arc<Mutex<SharedState>>,
) -> impl Filter<Extract = (Arc<Mutex<SharedState>>,), Error = std::convert::Infallible> + Clone {
    warp::any().map(move || state.clone())
}

/// Render historical universe statistics in a simple ASCII table.
fn render_universe_history(history_data: &serde_json::Value) -> Result<()> {
    println!("=== UNIVERSE STATISTICS HISTORY ===");

    if let Some(array) = history_data.as_array() {
        println!(
            "{:>8} {:>8} {:>8} {:>8} {:>14} {:>10}",
            "Tick", "Age(Gyr)", "Stars", "Planets", "Particles", "Temp(K)"
        );
        for entry in array.iter().take(100) {
            let tick = entry.get("tick").and_then(|v| v.as_u64()).unwrap_or(0);
            let age = entry.get("age_gyr").and_then(|v| v.as_f64()).unwrap_or(0.0);
            let stars = entry.get("star_count").and_then(|v| v.as_u64()).unwrap_or(0);
            let planets = entry.get("planet_count").and_then(|v| v.as_u64()).unwrap_or(0);
            let particles = entry.get("total_particles").and_then(|v| v.as_u64()).unwrap_or(0);
            let temp = entry.get("average_temperature").and_then(|v| v.as_f64()).unwrap_or(0.0);

            println!(
                "{:>8} {:>8.2} {:>8} {:>8} {:>14} {:>10.1}",
                tick, age, stars, planets, particles, temp
            );
        }
    } else {
        println!("No historical data available.");
    }
    Ok(())
}

fn render_sample_universe_history() {
    println!("=== UNIVERSE STATISTICS HISTORY (SAMPLE) ===");
    println!("Tick  Age(Gyr) Stars Planets   Particles     Temp");
    for i in 0..10 {
        println!("{:>4}   {:>6.2}  {:>5}   {:>6}  {:>11}   {:>6.1}", i * 100, 0.5 + i as f64 * 0.1, 2000 + i * 50, 5000 + i * 70, 1_000_000 + i * 10_000, 3.0 + i as f64);
    }
    println!("\nNote: Connect to a running simulation for real data.");
}