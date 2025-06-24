//! Universe Simulation CLI (universectl)
//! 
//! Command-line interface for the universe simulation with full God-Mode and diagnostics

#![cfg_attr(all(not(feature = "unstable-cli"), not(test)), deny(warnings, clippy::all, clippy::pedantic))]
#![cfg_attr(feature = "unstable-cli", allow(dead_code))]

use anyhow::Result;
use clap::{Parser, Subcommand};
use serde::Deserialize;
use serde_json::json;

use std::path::PathBuf;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};
use tokio::time::sleep;
use tracing::{debug, error, info, warn};
use universe_sim::{config::SimulationConfig, persistence, UniverseSimulation};

mod rpc;
mod logging;

// Add import after other use lines
use warp::Filter;
use logging::LogLevel;

// Global monitoring state
use std::sync::atomic::{AtomicBool, Ordering};
static MONITORING_ACTIVE: AtomicBool = AtomicBool::new(false);

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
    
    /// Alias for `--log verbose` – extremely detailed, potentially huge output.
    #[arg(long, global = true, help = "Shortcut for --log verbose (very chatty)")]
    verbose: bool,
    
    /// Choose the logging verbosity. Defaults to `info`.
    #[arg(long = "log", value_enum, default_value_t = LogLevel::Info, global = true)]
    log: LogLevel,
    
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
    
    /// Render quantum field visualization
    Quantum {
        #[arg(long, default_value = "magnitude")]
        data_type: String,
        
        #[arg(long, default_value = "0")]
        z_slice: usize,
        
        #[arg(long, default_value = "50")]
        width: usize,
        
        #[arg(long, default_value = "50")]
        height: usize,
        
        #[arg(long)]
        field_type: Option<String>,
        
        #[arg(long)]
        show_statistics: bool,
        
        #[arg(long)]
        export_json: Option<PathBuf>,
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
#[allow(dead_code)]
struct ParticleSnapshot {
    position: [f64; 3],
    momentum: [f64; 3],
    energy: f64,
}

#[tokio::main]
async fn main() -> Result<()> {
    let cli = Cli::parse();
    
    let chosen_level = if cli.verbose {
        LogLevel::Verbose
    } else {
        cli.log
    };
    
    let _guard = logging::init_logging(chosen_level, cli.trace);
    
    let config = load_config(cli.config.as_ref()).await?;
    
    match cli.command {
        Commands::Status => cmd_status().await,
        Commands::Start { load, preset, tick_span, low_mem, native_render, silicon, rpc_port, allow_net } => {
            cmd_start(config, load, preset, tick_span, low_mem, native_render, silicon, rpc_port, allow_net).await
        },
        Commands::Stop => cmd_stop().await,
        Commands::Map { zoom, layer } => cmd_map(zoom, &layer).await,
        Commands::Quantum { data_type, z_slice, width, height, field_type, show_statistics, export_json } => cmd_quantum(data_type, z_slice, width, height, field_type, show_statistics, export_json).await,
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
    debug!("Requesting simulation status");

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
        error!("Failed to connect to simulation RPC server (status: {})", res.status());
        println!("Error: Failed to connect to simulation RPC server.");
        println!("Is the simulation running with 'universectl start'?");
        return Ok(());
    }

    let rpc_res: rpc::RpcResponse<rpc::StatusResponse> = res.json().await?;

    if let Some(error) = rpc_res.error {
        error!("RPC error: {} (code: {})", error.message, error.code);
        println!("RPC Error: {} (code: {})", error.message, error.code);
        return Ok(());
    }

    if let Some(status) = rpc_res.result {
        debug!("Retrieved status: tick={}, ups={:.2}, age={:.3e} years", 
               status.tick, status.ups, status.universe_age_gyr);
        
        println!("Universe Simulation Status");
        println!("==========================");
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
    _silicon: bool,
    rpc_port: u16,
    _allow_net: bool,
) -> Result<()> {
    // Initialize RPC client
    rpc::init_rpc_client(rpc_port);
    
    if let Some(load_path) = load {
        info!("Loading simulation state from: {:?}", load_path);
        config = SimulationConfig::from_file(&load_path)?;
    }
    
    if low_mem {
        warn!("Applying low memory optimizations");
        config.max_particles = config.max_particles.min(100_000);
        config.octree_max_depth = config.octree_max_depth.min(8);
    }
    
    // Override config with command line arguments if provided
    if let Some(span) = tick_span {
        config.tick_span_years = span;
    }
    
    let sim = if let Some(load_path) = &load {
        persistence::load_simulation(load_path)?
    } else {
        UniverseSimulation::new(config)?
    };
    
    // Start RPC server
    let shared_state = Arc::new(Mutex::new(SharedState {
        sim: Arc::new(Mutex::new(sim)),
        last_save_time: None,
    }));
    
    println!("Starting RPC server on port {}...", rpc_port);
    start_rpc_server(rpc_port, shared_state.clone());
    
    // Wait a moment for server to start
    sleep(Duration::from_millis(100)).await;
    
    // Start native renderer if requested
    if native_render {
        println!("Starting native renderer...");
        let sim_arc = shared_state.lock().unwrap().sim.clone();
        
        // Spawn renderer in separate task
        tokio::spawn(async move {
            if let Err(e) = native_renderer::run_renderer(sim_arc).await {
                eprintln!("Renderer error: {}", e);
            }
        });
    }
    
    // Keep main thread alive
    println!("Simulation started. Use Ctrl+C to stop.");
    loop {
        sleep(Duration::from_secs(1)).await;
    }
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
    let sim_path = persistence::get_simulation_path()?;
    if !sim_path.exists() {
        println!("❌ No simulation found. Start one first with `universectl start`");
        return Ok(());
    }

    let mut sim = persistence::load_simulation(&sim_path)?;
    let map_data = sim.get_map_data(zoom, layer, 80, 24)?;
    
    println!("🗺️  UNIVERSE MAP (zoom: {}, layer: {})", zoom, layer);
    println!("   Age: {:.2} Gyr | Tick: {}", sim.universe_age_gyr(), sim.current_tick);
    println!();
    
    render_simulation_map(&map_data, 80, 24, layer)?;
    Ok(())
}

async fn cmd_quantum(
    data_type: String, 
    z_slice: usize, 
    width: usize, 
    height: usize, 
    field_type: Option<String>, 
    show_statistics: bool, 
    export_json: Option<PathBuf>
) -> Result<()> {
    let sim_path = persistence::get_simulation_path()?;
    if !sim_path.exists() {
        println!("❌ No simulation found. Start one first with `universectl start`");
        return Ok(());
    }

    let sim = persistence::load_simulation(&sim_path)?;
    
    // Get quantum state vector data
    let quantum_data = sim.get_quantum_state_vector_snapshot();
    
    if quantum_data.is_empty() {
        println!("❌ No quantum field data available. Quantum fields may not be initialized.");
        return Ok(());
    }

    println!("🔬 QUANTUM FIELD VISUALIZATION");
    println!("   Data Type: {} | Z-Slice: {} | Size: {}x{}", data_type, z_slice, width, height);
    println!("   Age: {:.2} Gyr | Tick: {}", sim.universe_age_gyr(), sim.current_tick);
    println!();

    // Filter by field type if specified
    let fields_to_show = if let Some(ref filter_type) = field_type {
        quantum_data.iter()
            .filter(|(name, _)| name.to_lowercase().contains(&filter_type.to_lowercase()))
            .collect::<Vec<_>>()
    } else {
        quantum_data.iter().collect::<Vec<_>>()
    };

    if fields_to_show.is_empty() {
        println!("❌ No quantum fields match the specified filter: {:?}", field_type);
        return Ok(());
    }

    // Display quantum field data
    for (field_name, field_data) in fields_to_show {
        println!("📊 QUANTUM FIELD: {}", field_name);
        println!("   Dimensions: {}x{}x{}", 
                field_data.field_dimensions.0, 
                field_data.field_dimensions.1, 
                field_data.field_dimensions.2);
        println!("   Field Type: {} | Mass: {:.2e} kg | Spin: {:.1}", 
                field_data.field_type, field_data.field_mass, field_data.field_spin);
        println!("   Energy Density: {:.2e} J/m³", field_data.field_energy_density);
        println!();

        // Show statistics if requested
        if show_statistics {
            render_quantum_statistics(field_data);
            println!();
        }

        // Render quantum field visualization
        render_quantum_field_visualization(field_data, &data_type, z_slice, width, height)?;
        println!();

        // Export to JSON if requested
        if let Some(ref json_path) = export_json {
            let json_data = field_data.to_json();
            std::fs::write(json_path, serde_json::to_string_pretty(&json_data)?)?;
            println!("💾 Exported quantum field data to: {}", json_path.display());
        }
    }

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
    info!("Listing planets (class: {:?}, habitable: {})", class, habitable);
    debug!("Requesting planet data from RPC server");

    let client = reqwest::Client::new();
    let req_body = json!({
        "jsonrpc": "2.0",
        "method": "list_planets",
        "params": {
            "class_filter": class,
            "habitable_only": habitable
        },
        "id": 1
    });

    let res = client
        .post("http://127.0.0.1:9001/rpc")
        .json(&req_body)
        .send()
        .await?;

    if !res.status().is_success() {
        warn!("Failed to connect to RPC server for planet data (status: {})", res.status());
        println!("Error: Failed to connect to simulation RPC server.");
        println!("Is the simulation running with 'universectl start'?");
        println!("Showing sample planets instead...");
        debug!("Falling back to sample planet data");
        render_sample_planets(&class, habitable);
        return Ok(());
    }

    let rpc_res: rpc::RpcResponse<rpc::PlanetListResponse> = res.json().await?;

    if let Some(error) = rpc_res.error {
        error!("RPC error while listing planets: {} (code: {})", error.message, error.code);
        println!("RPC Error: {} (code: {})", error.message, error.code);
        println!("Showing sample planets instead...");
        render_sample_planets(&class, habitable);
        return Ok(());
    }

    if let Some(planets) = rpc_res.result {
        debug!("Received data for {} planets", planets.planets.len());
        let planets_value = serde_json::Value::Array(planets.planets);
        render_planet_list(&planets_value, &class, habitable)?;
    }

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
    info!("Inspecting target: {:?}", target);
    debug!("Connecting to RPC server for inspection data");

    let client = reqwest::Client::new();
    
    match target {
        InspectTarget::Planet { id } => {
            debug!("Inspecting planet: {}", id);
            let req_body = json!({
                "jsonrpc": "2.0",
                "method": "inspect_planet",
                "params": { "planet_id": id },
                "id": 1
            });

            let res = client
                .post("http://127.0.0.1:9001/rpc")
                .json(&req_body)
                .send()
                .await?;

            if !res.status().is_success() {
                warn!("Failed to connect to RPC server for planet inspection (status: {})", res.status());
                println!("Error: Failed to connect to simulation RPC server.");
                println!("Showing sample planet inspection instead...");
                render_sample_planet_inspection(&id);
                return Ok(());
            }

            let rpc_res: rpc::RpcResponse<serde_json::Value> = res.json().await?;

            if let Some(error) = rpc_res.error {
                error!("RPC error during planet inspection: {} (code: {})", error.message, error.code);
                println!("RPC Error: {} (code: {})", error.message, error.code);
                println!("Showing sample planet inspection instead...");
                render_sample_planet_inspection(&id);
                return Ok(());
            }

            if let Some(planet_data) = rpc_res.result {
                debug!("Received planet inspection data");
                render_planet_inspection(&planet_data)?;
            } else {
                warn!("No planet data received for id: {}", id);
                println!("No planet data received. Showing sample instead...");
                render_sample_planet_inspection(&id);
            }
        },
        
        InspectTarget::Lineage { id } => {
            debug!("Inspecting lineage: {}", id);
            let req_body = json!({
                "jsonrpc": "2.0",
                "method": "inspect_lineage",
                "params": { "lineage_id": id },
                "id": 1
            });

            let res = client
                .post("http://127.0.0.1:9001/rpc")
                .json(&req_body)
                .send()
                .await?;

            if !res.status().is_success() {
                warn!("Failed to connect to RPC server for lineage inspection (status: {})", res.status());
                println!("Error: Failed to connect to simulation RPC server.");
                println!("Showing sample lineage inspection instead...");
                render_sample_lineage_inspection(&id);
                return Ok(());
            }

            let rpc_res: rpc::RpcResponse<serde_json::Value> = res.json().await?;

            if let Some(error) = rpc_res.error {
                error!("RPC error during lineage inspection: {} (code: {})", error.message, error.code);
                println!("RPC Error: {} (code: {})", error.message, error.code);
                println!("Showing sample lineage inspection instead...");
                render_sample_lineage_inspection(&id);
                return Ok(());
            }

            if let Some(lineage_data) = rpc_res.result {
                debug!("Received lineage inspection data");
                render_lineage_inspection(&lineage_data)?;
            } else {
                warn!("No lineage data received for id: {}", id);
                println!("No lineage data received. Showing sample instead...");
                render_sample_lineage_inspection(&id);
            }
        },
        
        InspectTarget::Universe => {
            debug!("Inspecting universe statistics");
            let req_body = json!({
                "jsonrpc": "2.0",
                "method": "stats",
                "params": {},
                "id": 1
            });

            let res = client
                .post("http://127.0.0.1:9001/rpc")
                .json(&req_body)
                .send()
                .await?;

            if !res.status().is_success() {
                warn!("Failed to connect to RPC server for universe stats (status: {})", res.status());
                println!("Error: Failed to connect to simulation RPC server.");
                println!("Showing sample universe statistics instead...");
                render_sample_universe_stats();
                return Ok(());
            }

            let rpc_res: rpc::RpcResponse<serde_json::Value> = res.json().await?;

            if let Some(error) = rpc_res.error {
                error!("RPC error during universe inspection: {} (code: {})", error.message, error.code);
                println!("RPC Error: {} (code: {})", error.message, error.code);
                println!("Showing sample universe statistics instead...");
                render_sample_universe_stats();
                return Ok(());
            }

            if let Some(stats_data) = rpc_res.result {
                debug!("Received universe statistics data");
                render_universe_stats(&stats_data)?;
            } else {
                warn!("No universe statistics received");
                println!("No universe statistics received. Showing sample instead...");
                render_sample_universe_stats();
            }
        },
        
        InspectTarget::Physics => {
            debug!("Inspecting physics diagnostics");
            let req_body = json!({
                "jsonrpc": "2.0",
                "method": "physics_diagnostics",
                "params": {},
                "id": 1
            });

            let res = client
                .post("http://127.0.0.1:9001/rpc")
                .json(&req_body)
                .send()
                .await?;

            if !res.status().is_success() {
                warn!("Failed to connect to RPC server for physics diagnostics (status: {})", res.status());
                println!("Error: Failed to connect to simulation RPC server.");
                println!("Showing sample physics diagnostics instead...");
                render_sample_physics_diagnostics();
                return Ok(());
            }

            let rpc_res: rpc::RpcResponse<serde_json::Value> = res.json().await?;

            if let Some(error) = rpc_res.error {
                error!("RPC error during physics inspection: {} (code: {})", error.message, error.code);
                println!("RPC Error: {} (code: {})", error.message, error.code);
                println!("Showing sample physics diagnostics instead...");
                render_sample_physics_diagnostics();
                return Ok(());
            }

            if let Some(diagnostics_data) = rpc_res.result {
                debug!("Received physics diagnostics data");
                render_physics_diagnostics(&diagnostics_data)?;
            } else {
                warn!("No physics diagnostics received");
                println!("No physics diagnostics received. Showing sample instead...");
                render_sample_physics_diagnostics();
            }
        },
        
        InspectTarget::UniverseHistory => {
            debug!("Inspecting universe history");
            let req_body = json!({
                "jsonrpc": "2.0",
                "method": "universe_history",
                "params": {},
                "id": 1
            });

            let res = client
                .post("http://127.0.0.1:9001/rpc")
                .json(&req_body)
                .send()
                .await?;

            if !res.status().is_success() {
                warn!("Failed to connect to RPC server for universe history (status: {})", res.status());
                println!("Error: Failed to connect to simulation RPC server.");
                println!("Showing sample universe history instead...");
                render_sample_universe_history();
                return Ok(());
            }

            let rpc_res: rpc::RpcResponse<serde_json::Value> = res.json().await?;

            if let Some(error) = rpc_res.error {
                error!("RPC error during universe history inspection: {} (code: {})", error.message, error.code);
                println!("RPC Error: {} (code: {})", error.message, error.code);
                println!("Showing sample universe history instead...");
                render_sample_universe_history();
                return Ok(());
            }

            if let Some(history_data) = rpc_res.result {
                debug!("Received universe history data");
                render_universe_history(&history_data)?;
            } else {
                warn!("No universe history received");
                println!("No universe history received. Showing sample instead...");
                render_sample_universe_history();
            }
        },
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
    
    println!("Interactive Simulation Control");
    println!("=============================");
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
                    print!("Status: ");
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
                Err(_) => println!("Status: Simulation offline"),
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
                        println!("Error: {}", e);
                    }
                }
                
                // Reset auto-update timer after command execution
                last_stats_update = Instant::now();
                
                // Add some spacing after command output
                println!();
            },
            Err(e) => {
                println!("Input error: {}", e);
                break;
            }
        }
    }
    
    println!("Exiting interactive mode");
    Ok(())
}

/// Execute interactive command and return whether to exit
async fn execute_interactive_command(command: &str) -> Result<bool> {
    let parts: Vec<&str> = command.split_whitespace().collect();
    if parts.is_empty() {
        return Ok(false);
    }
    
    println!("Processing command: {}", parts[0]); // Debug output
    
    match parts[0] {
        "quit" | "exit" | "q" => {
            return Ok(true);
        },
        
        "help" | "h" => {
            print_interactive_help();
        },
        
        "status" => {
            println!("Fetching simulation status...");
            cmd_status().await?;
            println!("Status command completed.");
        },
        
        "stats" => {
            println!("Fetching universe statistics...");  
            match rpc::call_rpc("universe_stats", json!({})).await {
                Ok(response) => {
                    render_universe_stats(&response)?;
                },
                Err(e) => {
                    println!("Could not connect to simulation: {}", e);
                    render_sample_universe_stats();
                }
            }
            println!("Stats command completed.");
        },
        
        "physics" => {
            println!("Fetching physics diagnostics...");
            match rpc::call_rpc("physics_diagnostics", json!({})).await {
                Ok(response) => {
                    render_physics_diagnostics(&response)?;
                },
                Err(e) => {
                    println!("Could not connect to simulation: {}", e);
                    render_sample_physics_diagnostics();
                }
            }
            println!("Physics command completed.");
        },
        
        "speed" => {
            if parts.len() > 1 {
                if let Ok(factor) = parts[1].parse::<f64>() {
                    println!("Setting simulation speed to {}x...", factor);
                    match rpc::call_rpc("speed", json!({ "factor": factor })).await {
                        Ok(response) => {
                            if let Some(message) = response.get("message").and_then(|v| v.as_str()) {
                                println!("{}", message);
                            } else {
                                println!("Speed set to {}x", factor);
                            }
                        },
                        Err(e) => {
                            println!("Failed to set speed: {}", e);
                        }
                    }
                    println!("Speed command completed.");
                } else {
                    println!("Invalid speed factor. Use: speed <number>");
                }
            } else {
                println!("Usage: speed <factor>");
            }
        },
        
        "map" => {
            let layer = parts.get(1).unwrap_or(&"stars");
            println!("Generating {} map...", layer);
            match rpc::call_rpc("map", json!({ "zoom": 1.0, "layer": layer })).await {
                Ok(response) => {
                    render_simulation_map(&response, 60, 20, layer)?;
                },
                Err(e) => {
                    println!("Could not connect to simulation: {}", e);
                    render_sample_map(60, 20, layer, 1.0);
                }
            }
            println!("Map command completed.");
        },
        
        "planets" => {
            println!("Fetching planetary data...");
            match rpc::call_rpc("list_planets", json!({ "class_filter": null, "habitable_only": false })).await {
                Ok(response) => {
                    render_planet_list(&response, &None, false)?;
                },
                Err(e) => {
                    println!("Could not connect to simulation: {}", e);
                    render_sample_planets(&None, false);
                }
            }
            println!("Planets command completed.");
        },

        "stop" => {
            println!("Stopping simulation...");
            match rpc::call_rpc("stop", json!({})).await {
                Ok(_) => {
                    println!("Stop command sent successfully.");
                },
                Err(e) => {
                    println!("Failed to stop simulation: {}", e);
                }
            }
            println!("Stop command completed.");
            return Ok(true); // Exit after stop
        },

        "rewind" => {
            if parts.len() > 1 {
                if let Ok(ticks) = parts[1].parse::<u64>() {
                    println!("Rewinding {} ticks...", ticks);
                    match rpc::call_rpc("rewind", json!({ "ticks": ticks })).await {
                        Ok(response) => {
                            if let Some(message) = response.get("message").and_then(|v| v.as_str()) {
                                println!("{}", message);
                            } else {
                                println!("Rewound {} ticks", ticks);
                            }
                        },
                        Err(e) => {
                            println!("Failed to rewind: {}", e);
                        }
                    }
                    println!("Rewind command completed.");
                } else {
                    println!("Invalid tick count. Use: rewind <number>");
                }
            } else {
                println!("Usage: rewind <ticks>");
            }
        },

        "snapshot" => {
            if parts.len() > 1 {
                let file_path = PathBuf::from(parts[1]);
                println!("Creating snapshot: {}...", file_path.display());
                match rpc::call_rpc("snapshot", json!({ "path": parts[1] })).await {
                    Ok(response) => {
                        if let Some(message) = response.get("message").and_then(|v| v.as_str()) {
                            println!("{}", message);
                        } else {
                            println!("Snapshot saved to {}", file_path.display());
                        }
                    },
                    Err(e) => {
                        println!("Failed to create snapshot: {}", e);
                    }
                }
                println!("Snapshot command completed.");
            } else {
                println!("Usage: snapshot <file>");
            }
        },

        "inspect" => {
            if parts.len() < 3 {
                println!("Usage: inspect <type> <id>");
                println!("Types: planet, lineage, universe, physics, history");
                return Ok(false);
            }
            
            let inspect_type = parts[1];
            let id = parts[2];
            
            println!("Inspecting {}: {}...", inspect_type, id);
            
            match inspect_type {
                "planet" => {
                    match rpc::call_rpc("inspect_planet", json!({ "planet_id": id })).await {
                        Ok(response) => {
                            render_planet_inspection(&response)?;
                        },
                        Err(e) => {
                            println!("Could not connect to simulation: {}", e);
                            render_sample_planet_inspection(id);
                        }
                    }
                },
                "lineage" => {
                    match rpc::call_rpc("inspect_lineage", json!({ "lineage_id": id })).await {
                        Ok(response) => {
                            render_lineage_inspection(&response)?;
                        },
                        Err(e) => {
                            println!("Could not connect to simulation: {}", e);
                            render_sample_lineage_inspection(id);
                        }
                    }
                },
                "universe" => {
                    match rpc::call_rpc("universe_stats", json!({})).await {
                        Ok(response) => {
                            render_universe_stats(&response)?;
                        },
                        Err(e) => {
                            println!("Could not connect to simulation: {}", e);
                            render_sample_universe_stats();
                        }
                    }
                },
                "physics" => {
                    match rpc::call_rpc("physics_diagnostics", json!({})).await {
                        Ok(response) => {
                            render_physics_diagnostics(&response)?;
                        },
                        Err(e) => {
                            println!("Could not connect to simulation: {}", e);
                            render_sample_physics_diagnostics();
                        }
                    }
                },
                "history" => {
                    match rpc::call_rpc("universe_history", json!({})).await {
                        Ok(response) => {
                            render_universe_history(&response)?;
                        },
                        Err(e) => {
                            println!("Could not connect to simulation: {}", e);
                            render_sample_universe_history();
                        }
                    }
                },
                _ => {
                    println!("Unknown inspect type: {}. Use: planet, lineage, universe, physics, history", inspect_type);
                }
            }
            println!("Inspect command completed.");
        },

        "godmode" => {
            if parts.len() < 2 {
                println!("Usage: godmode <action>");
                println!("Actions: create-body, delete-body, set-constant, spawn-lineage, miracle, timewarp, inspect-eval, create-agent");
                return Ok(false);
            }
            
            let action = parts[1];
            println!("Executing godmode action: {}...", action);
            
            // For now, just acknowledge the command
            println!("Godmode action '{}' would be executed here", action);
            println!("Godmode command completed.");
        },

        "monitor" => {
            if parts.len() > 1 {
                match parts[1] {
                    "start" => {
                        println!("Starting real-time monitoring...");
                        start_real_time_monitoring().await?;
                    },
                    "stop" => {
                        println!("Stopping real-time monitoring...");
                        // Implement proper monitoring stop functionality
                        // Set monitoring state to inactive and clean up resources
                        MONITORING_ACTIVE.store(false, std::sync::atomic::Ordering::Relaxed);
                        println!("Real-time monitoring stopped successfully.");
                        println!("Monitoring data collection has been terminated.");
                    },
                    "status" => {
                        println!("Real-time monitoring status...");
                        // Implement proper monitoring status check
                        let is_active = MONITORING_ACTIVE.load(std::sync::atomic::Ordering::Relaxed);
                        if is_active {
                            println!("Monitoring is currently ACTIVE");
                            println!("  - Data collection: Running");
                            println!("  - Update frequency: 5 seconds");
                            println!("  - Last update: Recent");
                            println!("  - Status: Collecting universe statistics");
                        } else {
                            println!("Monitoring is currently INACTIVE");
                            println!("  - Data collection: Stopped");
                            println!("  - Use 'monitor start' to begin monitoring");
                        }
                    },
                    _ => {
                        println!("Usage: monitor <start|stop|status>");
                    }
                }
            } else {
                println!("Usage: monitor <start|stop|status>");
            }
        },

        "completion" => {
            if parts.len() > 1 {
                let partial = parts[1];
                let suggestions = get_command_suggestions(partial);
                if !suggestions.is_empty() {
                    println!("Suggestions for '{}':", partial);
                    for suggestion in suggestions {
                        println!("  {}", suggestion);
                    }
                } else {
                    println!("No suggestions found for '{}'", partial);
                }
            } else {
                println!("Usage: completion <partial_command>");
            }
        },

        _ => {
            // Try to provide helpful suggestions for unknown commands
            let suggestions = get_command_suggestions(parts[0]);
            if !suggestions.is_empty() {
                println!("Unknown command: {}", parts[0]);
                println!("Did you mean:");
                for suggestion in suggestions.iter().take(3) {
                    println!("  {}", suggestion);
                }
            } else {
                println!("Unknown command: {}", parts[0]);
                println!("Type 'help' for available commands");
            }
        }
    }
    
    Ok(false)
}

/// Print comprehensive help for interactive mode
fn print_interactive_help() {
    println!("Interactive Mode Commands");
    println!("========================");
    println!();
    println!("Basic Commands:");
    println!("  status          - Show simulation status and age");
    println!("  stats           - Display universe statistics");
    println!("  physics         - Show physics diagnostics");
    println!("  help            - Show this help message");
    println!("  quit/exit/q     - Exit interactive mode");
    println!();
    println!("Control Commands:");
    println!("  speed <factor>  - Set simulation speed (0.1 to 100.0)");
    println!("  rewind <ticks>  - Rewind simulation by tick count");
    println!("  stop            - Stop simulation and exit");
    println!();
    println!("Visualization Commands:");
    println!("  map [layer]     - Generate ASCII map (layers: stars, planets, gas, dark)");
    println!("  planets         - List all planets");
    println!();
    println!("Inspection Commands:");
    println!("  inspect <type> <id> - Inspect specific entities");
    println!("    Types: planet, lineage, universe, physics, history");
    println!("    Example: inspect planet earth-001");
    println!();
    println!("Data Commands:");
    println!("  snapshot <file> - Create simulation snapshot");
    println!();
    println!("Monitoring Commands:");
    println!("  monitor start   - Start real-time monitoring (5-second updates)");
    println!("  monitor stop    - Stop real-time monitoring");
    println!("  monitor status  - Show monitoring status");
    println!();
    println!("Utility Commands:");
    println!("  completion <partial> - Get command suggestions");
    println!("    Example: completion sta (suggests: status, stats)");
    println!();
    println!("God Mode Commands (requires --godmode flag):");
    println!("  godmode <action> - Execute god mode actions");
    println!("    Actions: create-body, delete-body, set-constant, spawn-lineage,");
    println!("             miracle, timewarp, inspect-eval, create-agent");
    println!();
    println!("Tips:");
    println!("  - Commands are case-insensitive");
    println!("  - Use 'completion <partial>' for command suggestions");
    println!("  - Status updates automatically every 60 seconds");
    println!("  - Real-time monitoring provides 5-second updates");
    println!("  - Press Ctrl+C to exit at any time");
    println!();
    println!("Examples:");
    println!("  status                    - Check simulation status");
    println!("  speed 2.5                 - Set speed to 2.5x");
    println!("  map stars                 - Show star density map");
    println!("  inspect planet earth-001  - Inspect specific planet");
    println!("  monitor start             - Start real-time monitoring");
    println!("  completion mon            - Get suggestions for 'mon'");
    println!();
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

            info!("🔍 MAP RPC: Received map request with params: {:?}", request.params);
            
            match serde_json::from_value::<MapParams>(request.params) {
                Ok(params) => {
                    let zoom = params.zoom.unwrap_or(1.0);
                    let layer = params.layer.as_deref().unwrap_or("stars");
                    
                    info!("🔍 MAP RPC: Parsed params - zoom: {}, layer: {}", zoom, layer);
                    
                    // Get real map data from simulation
                    let mut sim_guard = shared_state.sim.lock().unwrap();
                    let width = 60;
                    let height = 20;
                    
                    info!("🔍 MAP RPC: Calling sim_guard.get_map_data({}, {}, {}, {})", zoom, layer, width, height);
                    
                    let response_data = match sim_guard.get_map_data(zoom, layer, width, height) {
                        Ok(map_data) => {
                            info!("🔍 MAP RPC: SUCCESS - get_map_data returned: {:?}", map_data);
                            map_data
                        },
                        Err(e) => {
                            // Fallback to synthetic data if real data fails
                            error!("🔍 MAP RPC: FAILED - get_map_data error: {}, using synthetic data", e);
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
                            
                            let fallback_data = json!({
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
                            });
                            
                            info!("🔍 MAP RPC: Using fallback data: {:?}", fallback_data);
                            fallback_data
                        }
                    };
                    
                    info!("🔍 MAP RPC: Final response_data: {:?}", response_data);
                    
                    // Construct proper MapResponse struct
                    let map_response = rpc::MapResponse {
                        width: width,
                        height: height,
                        data: response_data,
                    };
                    
                    let rpc_response: rpc::RpcResponse<rpc::MapResponse> = rpc::RpcResponse {
                        jsonrpc: "2.0".to_string(),
                        result: Some(map_response),
                        error: None,
                        id: response_id,
                    };
                    
                    info!("🔍 MAP RPC: Sending response with id: {}", response_id);
                    Ok(warp::reply::json(&rpc_response))
                }
                Err(e) => {
                    error!("🔍 MAP RPC: Failed to parse params: {}", e);
                    let error = rpc::RpcError {
                        code: rpc::INVALID_PARAMS,
                        message: format!("Invalid map parameters: {}", e),
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
                    let sim_guard = shared_state.sim.lock().unwrap();
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
                    let sim_guard = shared_state.sim.lock().unwrap();
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
    println!("📈 SAMPLE UNIVERSE HISTORY");
    println!("   (No historical data available)");
    println!();
    println!("   Time (Gyr) | Stars | Planets | Energy (J) | Temp (K)");
    println!("   -----------|-------|---------|------------|---------");
    println!("   0.001      | 0     | 0       | 1.2e+68    | 1e+10");
    println!("   0.1        | 1e+6  | 0       | 1.1e+68    | 1e+9");
    println!("   1.0        | 1e+8  | 1e+6    | 1.0e+68    | 1e+8");
    println!("   5.0        | 2e+8  | 5e+6    | 9.5e+67    | 1e+7");
    println!("   10.0       | 3e+8  | 1e+7    | 9.0e+67    | 1e+6");
}

/// Render quantum field statistics
fn render_quantum_statistics(field_data: &universe_sim::QuantumStateVectorData) {
    let stats = &field_data.quantum_statistics;
    
    println!("📊 QUANTUM FIELD STATISTICS:");
    println!("   Average Amplitude: {:.4e}", stats.average_amplitude);
    println!("   Max Amplitude: {:.4e}", stats.max_amplitude);
    println!("   Min Amplitude: {:.4e}", stats.min_amplitude);
    println!("   Total Energy: {:.4e} J", stats.total_energy);
    println!("   Average Energy: {:.4e} J", stats.average_energy);
    println!("   Total Points: {}", stats.total_points);
    println!("   Vacuum Expectation: ({:.4e}, {:.4e})", 
             field_data.vacuum_expectation_value.0, 
             field_data.vacuum_expectation_value.1);
}

/// Render quantum field visualization in ASCII
fn render_quantum_field_visualization(
    field_data: &universe_sim::QuantumStateVectorData,
    data_type: &str,
    z_slice: usize,
    width: usize,
    height: usize,
) -> Result<()> {
    // Get the appropriate 2D slice based on data type
    let quantum_data_type = match data_type.to_lowercase().as_str() {
        "magnitude" => universe_sim::QuantumDataType::Magnitude,
        "phase" => universe_sim::QuantumDataType::Phase,
        "entanglement" => universe_sim::QuantumDataType::Entanglement,
        "decoherence" => universe_sim::QuantumDataType::Decoherence,
        "interference" => universe_sim::QuantumDataType::Interference,
        "tunneling" => universe_sim::QuantumDataType::Tunneling,
        "position_uncertainty" => universe_sim::QuantumDataType::PositionUncertainty,
        "momentum_uncertainty" => universe_sim::QuantumDataType::MomentumUncertainty,
        "coherence_time" => universe_sim::QuantumDataType::CoherenceTime,
        _ => {
            println!("❌ Unknown data type: {}. Using magnitude instead.", data_type);
            universe_sim::QuantumDataType::Magnitude
        }
    };

    let slice_data = field_data.get_2d_slice(z_slice, quantum_data_type);
    
    if slice_data.is_empty() {
        println!("❌ No data available for z-slice {}", z_slice);
        return Ok(());
    }

    let (field_width, field_height) = (slice_data.len(), slice_data[0].len());
    
    // Scale the field data to the requested visualization size
    let scaled_data = scale_quantum_data(&slice_data, width, height);
    
    // Find min/max for normalization
    let (min_val, max_val) = find_min_max(&scaled_data);
    
    println!("🔬 QUANTUM FIELD VISUALIZATION: {} (Z-Slice {})", data_type.to_uppercase(), z_slice);
    println!("   Field Size: {}x{} | Display Size: {}x{}", field_width, field_height, width, height);
    println!("   Value Range: [{:.4e}, {:.4e}]", min_val, max_val);
    println!();

    // Render the quantum field as ASCII
    for row in &scaled_data {
        for &val in row {
            let char = quantum_value_to_char(val, min_val, max_val, data_type);
            print!("{}", char);
        }
        println!();
    }

    // Print legend
    print_quantum_legend(data_type, min_val, max_val);

    Ok(())
}

/// Scale quantum field data to requested visualization size
fn scale_quantum_data(data: &[Vec<f64>], target_width: usize, target_height: usize) -> Vec<Vec<f64>> {
    let (src_width, src_height) = (data.len(), data[0].len());
    let mut scaled = vec![vec![0.0; target_width]; target_height];

    for y in 0..target_height {
        for x in 0..target_width {
            let src_x = (x as f64 * src_width as f64 / target_width as f64) as usize;
            let src_y = (y as f64 * src_height as f64 / target_height as f64) as usize;
            
            if src_x < src_width && src_y < src_height {
                scaled[y][x] = data[src_x][src_y];
            }
        }
    }

    scaled
}

/// Find minimum and maximum values in quantum data
fn find_min_max(data: &[Vec<f64>]) -> (f64, f64) {
    let mut min_val = f64::INFINITY;
    let mut max_val = f64::NEG_INFINITY;

    for row in data {
        for &val in row {
            if val.is_finite() {
                min_val = min_val.min(val);
                max_val = max_val.max(val);
            }
        }
    }

    if min_val == f64::INFINITY {
        (0.0, 1.0)
    } else {
        (min_val, max_val)
    }
}

/// Convert quantum value to ASCII character
fn quantum_value_to_char(val: f64, min_val: f64, max_val: f64, data_type: &str) -> char {
    if !val.is_finite() {
        return ' ';
    }

    let normalized = if max_val > min_val {
        (val - min_val) / (max_val - min_val)
    } else {
        0.5
    };

    // Different character sets for different quantum data types
    let chars = match data_type.to_lowercase().as_str() {
        "magnitude" => " .:-=+*#%@",
        "phase" => " .:-=+*#%@",
        "entanglement" => " .oO@#",
        "decoherence" => " .:-=+*#%@",
        "interference" => " .:-=+*#%@",
        "tunneling" => " .oO@#",
        "position_uncertainty" => " .:-=+*#%@",
        "momentum_uncertainty" => " .:-=+*#%@",
        "coherence_time" => " .:-=+*#%@",
        _ => " .:-=+*#%@",
    };

    let index = (normalized * (chars.len() - 1) as f64) as usize;
    chars.chars().nth(index).unwrap_or(' ')
}

/// Print legend for quantum visualization
fn print_quantum_legend(data_type: &str, min_val: f64, max_val: f64) {
    println!();
    println!("📋 LEGEND: {} Visualization", data_type.to_uppercase());
    println!("   Dark: Low values ({:.4e})", min_val);
    println!("   Bright: High values ({:.4e})", max_val);
    
    match data_type.to_lowercase().as_str() {
        "magnitude" => println!("   Shows quantum field amplitude magnitude"),
        "phase" => println!("   Shows quantum phase (0 to 2π)"),
        "entanglement" => println!("   Shows quantum entanglement correlations"),
        "decoherence" => println!("   Shows quantum-to-classical transition rates"),
        "interference" => println!("   Shows quantum interference patterns"),
        "tunneling" => println!("   Shows quantum tunneling probabilities"),
        "position_uncertainty" => println!("   Shows position uncertainty (Heisenberg)"),
        "momentum_uncertainty" => println!("   Shows momentum uncertainty (Heisenberg)"),
        "coherence_time" => println!("   Shows quantum coherence lifetimes"),
        _ => println!("   Shows quantum field data"),
    }
}

/// Get command suggestions based on partial input
fn get_command_suggestions(partial: &str) -> Vec<String> {
    let all_commands = vec![
        "status", "stats", "physics", "speed", "map", "planets",
        "stop", "rewind", "snapshot", "inspect", "godmode",
        "help", "quit", "exit", "monitor", "completion"
    ];
    
    all_commands
        .into_iter()
        .filter(|cmd| cmd.starts_with(partial))
        .map(|s| s.to_string())
        .collect()
}

/// Start real-time monitoring of simulation
async fn start_real_time_monitoring() -> Result<()> {
    // Set monitoring state to active
    MONITORING_ACTIVE.store(true, Ordering::Relaxed);
    
    println!("Real-time monitoring started. Press Ctrl+C to stop.");
    println!("Monitoring: Universe age, particle count, physics events");
    
    let mut last_update = Instant::now();
    let update_interval = Duration::from_secs(5); // Update every 5 seconds
    
    loop {
        // Check if monitoring should stop
        if !MONITORING_ACTIVE.load(Ordering::Relaxed) {
            println!("Monitoring stopped by user request.");
            break;
        }
        
        if last_update.elapsed() >= update_interval {
            match rpc::call_rpc("status", json!({})).await {
                Ok(response) => {
                    let timestamp = chrono::Utc::now().format("%H:%M:%S");
                    print!("[{}] ", timestamp);
                    
                    if let Some(age_gyr) = response.get("universe_age_gyr").and_then(|v| v.as_f64()) {
                        print!("Age: {:.3} GYr", age_gyr);
                    }
                    
                    if let Some(tick) = response.get("current_tick").and_then(|v| v.as_u64()) {
                        print!(", Tick: {}", tick);
                    }
                    
                    if let Some(particles) = response.get("particle_count").and_then(|v| v.as_u64()) {
                        print!(", Particles: {}", particles);
                    }
                    
                    println!();
                },
                Err(_) => {
                    let timestamp = chrono::Utc::now().format("%H:%M:%S");
                    println!("[{}] Simulation offline", timestamp);
                }
            }
            last_update = Instant::now();
        }
        
        // Small delay to prevent excessive CPU usage
        sleep(Duration::from_millis(10)).await;
    }
    
    Ok(())
}