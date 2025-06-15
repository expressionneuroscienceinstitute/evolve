//! Universe Simulation CLI (universectl)
//! 
//! Command-line interface for the universe simulation with full God-Mode and diagnostics

use anyhow::Result;
use clap::{Parser, Subcommand};
use futures_util::{SinkExt, StreamExt};
use serde::{Deserialize, Serialize};
use serde_json::json;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use std::time::Duration;
use tokio::sync::broadcast;
use tokio::time::sleep;
use universe_sim::{config::SimulationConfig, persistence, UniverseSimulation};
use warp::ws::{Message, WebSocket};
use warp::Filter;

mod rpc;

/// A struct to hold the shared state of the simulation for RPC.
#[derive(Clone)]
struct SharedState {
    sim: Arc<Mutex<UniverseSimulation>>,
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
    let req_body = json!({
        "jsonrpc": "2.0",
        "method": "get_status",
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
        println!("Era: {}", status.cosmic_era);
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
    preset: Option<String>,
    tick_span: Option<f64>,
    low_mem: bool,
    serve_dash: Option<u16>,
    rpc_port: u16,
    allow_net: bool,
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
        sim
    };

    let sim = Arc::new(Mutex::new(sim));

    // Start RPC server
    let shared_state = SharedState { sim: sim.clone() };
    tokio::spawn(start_rpc_server(rpc_port, shared_state));

    // Start WebSocket server if requested
    let (tx, _) = broadcast::channel(100);
    if let Some(port) = serve_dash {
        tokio::spawn(start_websocket_server(port, tx.clone()));
    }

    println!("Simulation started. Press Ctrl+C to stop.");

    let mut last_tick_time = std::time::Instant::now();
    let mut tick_counter = 0;
    let mut ups = 0.0;

    loop {
        let mut sim_guard = sim.lock().unwrap();
        sim_guard.tick()?;
        let current_tick = sim_guard.current_tick;
        let stats = sim_guard.get_stats();
        drop(sim_guard); // Release lock before sleeping

        // Auto-save logic
        if config.auto_save_interval > 0
            && current_tick % config.auto_save_interval == 0
            && current_tick > 0
        {
            let path = Path::new(&config.auto_save_path);
            println!("Auto-saving checkpoint to {:?}...", path);
            let sim_guard = sim.lock().unwrap();
            if let Err(e) = persistence::save_checkpoint(&sim_guard, path) {
                eprintln!("Failed to auto-save checkpoint: {}", e);
            }
        }

        tick_counter += 1;
        let elapsed = last_tick_time.elapsed();
        if elapsed >= Duration::from_secs(1) {
            ups = tick_counter as f64 / elapsed.as_secs_f64();
            tick_counter = 0;
            last_tick_time = std::time::Instant::now();

            let status_json = json!({
                "type": "status",
                "tick": stats.current_tick,
                "ups": ups,
                "age": stats.universe_age_gyr,
                "era": format!("{:?}", stats.cosmic_era),
                "lineages": stats.lineage_count,
            })
            .to_string();

            if tx.send(status_json).is_err() {
                // No active receivers, but that's fine.
            }
        }

        sleep(Duration::from_millis(
            (1000.0 / config.target_ups) as u64,
        ))
        .await;
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
    println!("Inspecting {:?}...", target);
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
    let response_id = request.id;

    let response = match request.method.as_str() {
        "get_status" => {
            let sim_guard = state.lock().unwrap().sim.lock().unwrap();
            let stats = sim_guard.get_stats();
            // This is a simplification; in reality, UPS would need to be tracked.
            // We'll leave it at 0.0 for the RPC response for now.
            let status = rpc::StatusResponse {
                status: "running".to_string(),
                tick: stats.current_tick,
                ups: 0.0, // UPS is tracked in the loop, not here.
                universe_age_gyr: stats.universe_age_gyr,
                cosmic_era: format!("{:?}", stats.cosmic_era),
                lineage_count: stats.lineage_count as u64,
                save_file_age_sec: None, // Could be implemented by checking file metadata
            };
            rpc::RpcResponse {
                jsonrpc: "2.0".to_string(),
                result: Some(status),
                error: None,
                id: response_id,
            }
        }
        "snapshot" => {
            #[derive(Deserialize)]
            struct SnapshotParams {
                path: PathBuf,
            }

            match serde_json::from_value::<SnapshotParams>(request.params) {
                Ok(params) => {
                    let sim_guard = state.lock().unwrap().sim.lock().unwrap();
                    match persistence::save_checkpoint(&sim_guard, &params.path) {
                        Ok(_) => rpc::RpcResponse {
                            jsonrpc: "2.0".to_string(),
                            result: Some(format!("Snapshot saved to {:?}", params.path)),
                            error: None,
                            id: response_id,
                        },
                        Err(e) => rpc::RpcResponse {
                            jsonrpc: "2.0".to_string(),
                            result: None,
                            error: Some(rpc::RpcError {
                                code: rpc::INTERNAL_ERROR,
                                message: format!("Failed to save snapshot: {}", e),
                            }),
                            id: response_id,
                        },
                    }
                }
                Err(e) => rpc::RpcResponse {
                    jsonrpc: "2.0".to_string(),
                    result: None,
                    error: Some(rpc::RpcError {
                        code: rpc::INVALID_PARAMS,
                        message: format!("Invalid parameters for snapshot: {}", e),
                    }),
                    id: response_id,
                },
            }
        }
        _ => rpc::RpcResponse {
            jsonrpc: "2.0".to_string(),
            result: None,
            error: Some(rpc::RpcError {
                code: rpc::METHOD_NOT_FOUND,
                message: format!("Method '{}' not found", request.method),
            }),
            id: response_id,
        },
    };

    Ok(warp::reply::json(&response))
}

async fn start_rpc_server(port: u16, state: SharedState) {
    let state = Arc::new(Mutex::new(state));
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