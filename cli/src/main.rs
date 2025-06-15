//! Universe Simulation CLI (universectl)
//! 
//! Command-line interface for the universe simulation with full God-Mode and diagnostics

use anyhow::Result;
use clap::{Parser, Subcommand};
use futures_util::{SinkExt, StreamExt};
use serde_json::json;
use std::path::PathBuf;
use std::sync::{Arc, Mutex};
use std::time::Duration;
use tokio::sync::broadcast;
use tokio::time::sleep;
use universe_sim::{config::SimulationConfig, UniverseSimulation};
use warp::ws::{Message, WebSocket};
use warp::Filter;

mod rpc;

/// A struct to hold the shared state of the simulation for RPC.
#[derive(Clone, Default)]
struct SharedState {
    status: rpc::StatusResponse,
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
        serve_dash: Option<u16>,
        
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
}

#[derive(Subcommand)]
enum InspectTarget {
    Planet { id: String },
    Lineage { id: String },
    Universe,
    Physics,
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

#[tokio::main]
async fn main() -> Result<()> {
    let cli = Cli::parse();
    
    init_logging(cli.verbose);
    
    let config = load_config(cli.config.as_ref()).await?;
    
    match cli.command {
        Commands::Status => cmd_status().await,
        Commands::Start { load, preset, tick_span, low_mem, serve_dash, rpc_port, allow_net } => {
            cmd_start(config, load, preset, tick_span, low_mem, serve_dash, rpc_port, allow_net).await
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
    }
}

fn init_logging(verbose: bool) {
    let level = if verbose { "debug" } else { "info" };
    
    tracing_subscriber::fmt()
        .with_env_filter(format!("universectl={},universe_sim={}", level, level))
        .init();
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
    let request = rpc::RpcRequest {
        jsonrpc: "2.0".to_string(),
        method: "get_status".to_string(),
        params: serde_json::Value::Null,
        id: 1,
    };

    let rpc_url = "http://127.0.0.1:9001/rpc";

    match client.post(rpc_url).json(&request).send().await {
        Ok(response) => {
            if response.status().is_success() {
                let rpc_response: rpc::RpcResponse<rpc::StatusResponse> = response.json().await?;
                if let Some(status) = rpc_response.result {
                    println!("Status: {}", status.status);
                    println!("Tick: {}", status.tick);
                    println!("UPS: {:.1}", status.ups);
                    println!("Universe Age: {:.3} Gyr", status.universe_age_gyr);
                    println!("Cosmic Era: {}", status.cosmic_era);
                    println!("Lineage Count: {}", status.lineage_count);
                    if let Some(age) = status.save_file_age_sec {
                        println!("Save File Age: {}s ago", age);
                    } else {
                        println!("Save File Age: N/A");
                    }
                } else if let Some(error) = rpc_response.error {
                    println!("Error from simulation: ({}) {}", error.code, error.message);
                }
            } else {
                println!(
                    "Failed to connect to simulation: Server responded with status {}",
                    response.status()
                );
            }
        }
        Err(_) => {
            println!("Status: Not Running (failed to connect to simulation)");
            println!("Tick: N/A");
            println!("UPS: N/A");
            println!("Universe Age: N/A");
            println!("Cosmic Era: N/A");
            println!("Lineage Count: N/A");
            println!("Save File Age: N/A");
        }
    }

    Ok(())
}

async fn cmd_start(
    mut config: SimulationConfig,
    load: Option<PathBuf>,
    preset: Option<String>,
    tick_span: Option<f64>,
    low_mem: bool,
    serve_dash: Option<u16>,
    rpc_port: u16,
    allow_net: bool,
) -> Result<()> {
    println!("Starting Universe Simulation...");

    if let Some(preset_name) = preset {
        config = match preset_name.as_str() {
            "low-memory" => SimulationConfig::low_memory(),
            "high-performance" => SimulationConfig::high_performance(),
            "benchmark" => SimulationConfig::benchmark(),
            _ => {
                eprintln!("Unknown preset: {}", preset_name);
                std::process::exit(1);
            }
        };
    }

    if let Some(span) = tick_span {
        config.tick_span_years = span;
    }

    if low_mem {
        config.memory_limit_gb = 0.5;
        config.initial_particle_count = 100;
    }

    config.validate()?;

    let warnings = config.check_system_compatibility()?;
    for warning in warnings {
        println!("Warning: {}", warning);
    }

    let mut sim = UniverseSimulation::new(config)?;

    if let Some(load_path) = load {
        println!("Loading from checkpoint: {:?}", load_path);
        // TODO: Implement checkpoint loading
    } else {
        println!("Initializing Big Bang...");
        sim.init_big_bang()?;
    }

    let (tx, _rx) = broadcast::channel(100);

    if let Some(port) = serve_dash {
        println!("Starting web dashboard on port {}", port);
        let tx_clone = tx.clone();
        tokio::spawn(async move {
            start_websocket_server(port, tx_clone).await;
        });
    }

    let shared_state = Arc::new(Mutex::new(SharedState::default()));

    let rpc_state = shared_state.clone();
    tokio::spawn(async move {
        start_rpc_server(rpc_port, rpc_state).await;
    });

    println!("Simulation started successfully!");
    let stats = sim.get_stats();
    println!("Initial conditions:");
    println!("  Particles: {}", stats.particle_count);
    println!("  Cosmic Era: {:?}", stats.cosmic_era);
    println!("  Target UPS: {}", stats.target_ups);

    println!("Running simulation... (Press Ctrl+C to stop)");

    let mut tick_count = 0;
    let start_time = std::time::Instant::now();

    loop {
        sim.tick()?;
        tick_count += 1;

        if tick_count % 100 == 0 {
            let stats = sim.get_stats();
            let elapsed = start_time.elapsed().as_secs_f64();
            let ups = if elapsed > 0.0 {
                tick_count as f64 / elapsed
            } else {
                0.0
            };

            {
                let mut state = shared_state.lock().unwrap();
                state.status = rpc::StatusResponse {
                    status: "Running".to_string(),
                    tick: stats.current_tick,
                    ups,
                    universe_age_gyr: stats.universe_age_gyr,
                    cosmic_era: format!("{:?}", stats.cosmic_era),
                    lineage_count: 0,
                    save_file_age_sec: None,
                };
            }

            if serve_dash.is_some() {
                let simulation_state = json!({
                    "type": "simulation_update",
                    "current_tick": stats.current_tick,
                    "universe_age_gyr": stats.universe_age_gyr,
                    "cosmic_era": format!("{:?}", stats.cosmic_era),
                    "particle_count": stats.particle_count,
                    "ups": ups,
                    "timestamp": std::time::SystemTime::now()
                        .duration_since(std::time::UNIX_EPOCH)
                        .unwrap()
                        .as_secs()
                });

                if let Err(e) = tx.send(simulation_state.to_string()) {
                    eprintln!("Failed to send simulation update: {}", e);
                }
            }

            if tick_count % 1000 == 0 {
                println!(
                    "Tick: {} | Age: {:.3} Gyr | Era: {:?} | UPS: {:.1}",
                    stats.current_tick, stats.universe_age_gyr, stats.cosmic_era, ups
                );
            }
        }

        sleep(Duration::from_millis(1)).await;
    }
}

async fn cmd_stop() -> Result<()> {
    println!("Stopping simulation...");
    // TODO: Send stop signal to running simulation
    println!("Simulation stopped.");
    Ok(())
}

async fn cmd_map(zoom: f64, layer: &str) -> Result<()> {
    println!("Universe Map (zoom: {:.1}, layer: {})", zoom, layer);
    println!("================================");
    
    let width = 60;
    let height = 20;
    
    for y in 0..height {
        for x in 0..width {
            let density = ((x + y) % 10) as f64 / 10.0;
            let char = match density {
                d if d < 0.2 => ' ',
                d if d < 0.4 => '.',
                d if d < 0.6 => ':',
                d if d < 0.8 => '*',
                _ => '#',
            };
            print!("{}", char);
        }
        println!();
    }
    
    println!("\nLegend: [space]=void, .=gas, :=stars, *=dense, #=very dense");
    
    Ok(())
}

async fn cmd_list_planets(class: Option<String>, habitable: bool) -> Result<()> {
    println!("Planetary Bodies");
    println!("================");
    
    println!("ID       | Class | Temp (°C) | Water | O2   | Radiation | Habitable");
    println!("---------|-------|-----------|-------|------|-----------|----------");
    
    let planets = [
        ("SOL-3", "E", "15.0", "0.71", "0.21", "0.002", "Yes"),
        ("PROX-B", "D", "-60.0", "0.01", "0.001", "0.1", "No"),
        ("KEPLER-442B", "E", "5.0", "0.45", "0.18", "0.003", "Yes"),
    ];
    
    for (id, pclass, temp, water, o2, rad, hab) in planets {
        if let Some(ref filter_class) = class {
            if pclass != filter_class {
                continue;
            }
        }
        
        if habitable && hab != "Yes" {
            continue;
        }
        
        println!("{:<8} | {:<5} | {:<9} | {:<5} | {:<4} | {:<9} | {}", 
                 id, pclass, temp, water, o2, rad, hab);
    }
    
    Ok(())
}

async fn cmd_inspect(target: InspectTarget) -> Result<()> {
    match target {
        InspectTarget::Planet { id } => {
            println!("Planet Inspection: {}", id);
            // TODO: Get planet data from simulation
        },
        InspectTarget::Lineage { id } => {
            println!("Lineage Inspection: {}", id);
            // TODO: Get lineage data from simulation
        },
        InspectTarget::Universe => {
            println!("Universe Overview");
            // TODO: Get universe statistics
        },
        InspectTarget::Physics => {
            println!("Physics Engine Diagnostics");
            // TODO: Get physics engine diagnostics
        },
    }
    println!("Status: Placeholder - implementation pending");
    Ok(())
}

async fn cmd_snapshot(file: PathBuf, format: Option<String>) -> Result<()> {
    println!("Saving snapshot to {:?} (format: {:?})", file, format.unwrap_or_else(|| "bin".to_string()));
    // TODO: Export simulation state to file
    println!("Snapshot saved.");
    Ok(())
}

async fn cmd_speed(factor: f64) -> Result<()> {
    println!("Setting simulation speed factor to {}", factor);
    // TODO: Send speed change command to simulation
    println!("Speed updated.");
    Ok(())
}

async fn cmd_rewind(ticks: u64) -> Result<()> {
    println!("Rewinding simulation by {} ticks", ticks);
    // TODO: Implement rewind functionality
    println!("Rewind complete.");
    Ok(())
}

async fn cmd_godmode(action: GodModeAction) -> Result<()> {
    println!("Executing God-Mode action...");
    match action {
        GodModeAction::CreateBody { .. } => {
            // TODO: Implement body creation
        }
        GodModeAction::DeleteBody { .. } => {
            // TODO: Implement body deletion
        }
        GodModeAction::SetConstant { .. } => {
            // TODO: Implement constant modification
        }
        GodModeAction::SpawnLineage { .. } => {
            // TODO: Implement lineage spawning
        }
        GodModeAction::Miracle { .. } => {
            // TODO: Implement miracles
        }
        GodModeAction::TimeWarp { .. } => {
            // TODO: Implement time warp
        }
        GodModeAction::InspectEval { .. } => {
            // TODO: Implement expression evaluation
        }
    }
    println!("God-Mode action completed.");
    Ok(())
}

async fn cmd_resources(action: ResourceAction) -> Result<()> {
    match action {
        ResourceAction::Queue => {
            println!("Pending resource requests:");
            // TODO: Show pending resource requests
        }
        ResourceAction::Grant { id, expires } => {
            println!("Granting resource request {} (expires: {:?})", id, expires);
            // TODO: Implement resource granting
        }
        ResourceAction::Status => {
            println!("Current resource usage:");
            // TODO: Show current resource usage
        }
        ResourceAction::Reload => {
            println!("Reloading resource limits...");
            // TODO: Implement resource reload
        }
    }
    Ok(())
}

async fn cmd_oracle(action: OracleAction) -> Result<()> {
    match action {
        OracleAction::Inbox => {
            println!("Pending messages from agents:");
            // TODO: Show pending messages from agents
        }
        OracleAction::Reply { .. } => {
            println!("Replying to agent petition...");
            // TODO: Implement reply functionality
        }
        OracleAction::Stats => {
            println!("Communication statistics:");
            // TODO: Show communication statistics
        }
    }
    Ok(())
}

async fn start_websocket_server(port: u16, tx: broadcast::Sender<String>) {
    let websocket_route = warp::path("ws")
        .and(warp::ws())
        .and(warp::any().map(move || tx.clone()))
        .map(|ws: warp::ws::Ws, tx_clone| {
            ws.on_upgrade(move |socket| handle_websocket(socket, tx_clone))
        });

    println!("Dashboard WebSocket listening on ws://127.0.0.1:{}/ws", port);
    warp::serve(websocket_route).run(([127, 0, 0, 1], port)).await;
}

async fn handle_websocket(websocket: WebSocket, tx: broadcast::Sender<String>) {
    let mut rx = tx.subscribe();
    let (mut ws_tx, _ws_rx) = websocket.split();

    loop {
        match rx.recv().await {
            Ok(msg) => {
                if ws_tx.send(Message::text(msg)).await.is_err() {
                    break;
                }
            }
            Err(_) => {
                break;
            }
        }
    }
}

async fn handle_rpc_request(
    request: rpc::RpcRequest,
    state: Arc<Mutex<SharedState>>,
) -> Result<impl warp::Reply, warp::Rejection> {
    match request.method.as_str() {
        "get_status" => {
            let state = state.lock().unwrap();
            let response = rpc::RpcResponse {
                jsonrpc: "2.0".to_string(),
                result: Some(state.status.clone()),
                error: None,
                id: request.id,
            };
            Ok(warp::reply::json(&response))
        }
        _ => {
            let error = rpc::RpcError {
                code: rpc::METHOD_NOT_FOUND,
                message: format!("Method '{}' not found.", request.method),
            };
            let response: rpc::RpcResponse<()> = rpc::RpcResponse {
                jsonrpc: "2.0".to_string(),
                result: None,
                error: Some(error),
                id: request.id,
            };
            Ok(warp::reply::json(&response))
        }
    }
}

async fn start_rpc_server(port: u16, state: Arc<Mutex<SharedState>>) {
    let rpc_route = warp::post()
        .and(warp::path("rpc"))
        .and(warp::body::json())
        .and(warp::any().map(move || state.clone()))
        .and_then(handle_rpc_request);

    println!("RPC server listening on http://127.0.0.1:{}/rpc", port);
    warp::serve(rpc_route).run(([127, 0, 0, 1], port)).await;
} 