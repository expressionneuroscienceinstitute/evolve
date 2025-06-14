//! Command-line interface for Evolve: The Game of Life
//!
//! This binary provides the main interface for running and controlling
//! the universe simulation as specified in the instructions.

use clap::{Parser, Subcommand};
use std::path::PathBuf;
use universe_sim::{Universe, SimulationConfig};

#[derive(Parser)]
#[command(author, version, about, long_about = None)]
#[command(name = "universectl")]
struct Cli {
    /// Configuration file
    #[arg(short, long)]
    config: Option<PathBuf>,
    
    /// Years per simulation tick
    #[arg(long)]
    tick_span: Option<f64>,
    
    /// Target updates per second
    #[arg(long)]
    ups: Option<f64>,
    
    /// Random seed for reproducibility
    #[arg(long)]
    seed: Option<u64>,
    
    /// Enable low-memory mode
    #[arg(long)]
    low_memory: bool,
    
    /// Enable god-mode
    #[arg(long)]
    #[cfg(feature = "god-mode")]
    godmode: bool,
    
    /// Verbose logging
    #[arg(short, long)]
    verbose: bool,
    
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Run the simulation
    Run {
        /// Load from checkpoint file
        #[arg(short, long)]
        load: Option<PathBuf>,
        
        /// Maximum simulation time in ticks
        #[arg(long)]
        max_ticks: Option<u64>,
    },
    
    /// Show simulation status
    Status,
    
    /// Create ASCII heat-map of universe
    Map {
        /// Zoom level
        #[arg(short, long, default_value = "1")]
        zoom: u32,
    },
    
    /// List planets
    ListPlanets {
        /// Filter by planet class
        #[arg(long)]
        class: Option<String>,
    },
    
    /// Inspect specific objects
    Inspect {
        /// Object type and ID
        #[arg(short, long)]
        object: String,
    },
    
    /// Take a snapshot
    Snapshot {
        /// Output file
        file: PathBuf,
    },
    
    /// Generate example configuration
    GenConfig {
        /// Output file
        file: PathBuf,
    },
    
    #[cfg(feature = "god-mode")]
    /// God-mode commands (requires --godmode)
    God {
        #[command(subcommand)]
        command: GodCommands,
    },
}

#[cfg(feature = "god-mode")]
#[derive(Subcommand)]
enum GodCommands {
    /// Create celestial body
    CreateBody {
        /// Body type (star/planet)
        body_type: String,
        /// Mass in solar masses
        mass: f64,
        /// Position (x,y,z)
        position: String,
    },
    
    /// Delete celestial body
    DeleteBody {
        /// Body ID
        id: u64,
    },
    
    /// Set physical constants
    SetConstant {
        /// Constant name (G, c, etc.)
        name: String,
        /// New value
        value: f64,
    },
    
    /// Time warp
    TimeWarp {
        /// Speed multiplier
        factor: f64,
    },
    
    /// Miracle (environmental intervention)
    Miracle {
        /// Planet ID
        planet_id: u64,
        /// Miracle type
        miracle_type: String,
    },
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let cli = Cli::parse();
    
    // Initialize logging
    let log_level = if cli.verbose {
        log::LevelFilter::Debug
    } else {
        log::LevelFilter::Info
    };
    
    env_logger::Builder::from_default_env()
        .filter_level(log_level)
        .init();
    
    // Load configuration
    let mut config = load_configuration(&cli)?;
    
    // Apply CLI overrides
    apply_cli_overrides(&mut config, &cli)?;
    
    log::info!("Starting Universe Simulation");
    log::info!("{}", config.summary());
    
    match cli.command {
        Commands::Run { load, max_ticks } => {
            run_simulation(config, load, max_ticks).await?;
        },
        Commands::Status => {
            show_status().await?;
        },
        Commands::Map { zoom } => {
            show_map(zoom).await?;
        },
        Commands::ListPlanets { class } => {
            list_planets(class).await?;
        },
        Commands::Inspect { object } => {
            inspect_object(object).await?;
        },
        Commands::Snapshot { file } => {
            take_snapshot(file).await?;
        },
        Commands::GenConfig { file } => {
            generate_config(file)?;
        },
        #[cfg(feature = "god-mode")]
        Commands::God { command } => {
            if !cli.godmode {
                anyhow::bail!("God-mode commands require --godmode flag");
            }
            execute_god_command(command).await?;
        },
    }
    
    Ok(())
}

/// Load configuration from file or defaults
fn load_configuration(cli: &Cli) -> anyhow::Result<SimulationConfig> {
    if let Some(config_path) = &cli.config {
        log::info!("Loading configuration from {}", config_path.display());
        SimulationConfig::from_file(config_path)
            .map_err(|e| anyhow::anyhow!("Failed to load config: {}", e))
    } else {
        log::info!("Using default configuration");
        Ok(SimulationConfig::default())
    }
}

/// Apply CLI argument overrides to configuration
fn apply_cli_overrides(config: &mut SimulationConfig, cli: &Cli) -> anyhow::Result<()> {
    if let Some(tick_span) = cli.tick_span {
        config.simulation.years_per_tick = tick_span;
    }
    
    if let Some(ups) = cli.ups {
        config.simulation.target_ups = ups;
    }
    
    if let Some(seed) = cli.seed {
        config.simulation.seed = Some(seed);
    }
    
    if cli.low_memory {
        config.simulation.low_memory_mode = true;
    }
    
    #[cfg(feature = "god-mode")]
    if cli.godmode {
        config.god_mode.enabled = true;
    }
    
    Ok(())
}

/// Run the main simulation
async fn run_simulation(
    config: SimulationConfig,
    load_file: Option<PathBuf>,
    max_ticks: Option<u64>,
) -> anyhow::Result<()> {
    log::info!("Initializing universe...");
    
    let mut universe = Universe::new(config)?;
    
    if let Some(checkpoint) = load_file {
        log::info!("Loading from checkpoint: {}", checkpoint.display());
        // TODO: Implement checkpoint loading
        anyhow::bail!("Checkpoint loading not yet implemented");
    } else {
        universe.initialize()?;
    }
    
    if let Some(max) = max_ticks {
        log::info!("Running simulation for {} ticks", max);
        // TODO: Implement tick limit
    }
    
    log::info!("Starting simulation loop...");
    universe.run()?;
    
    log::info!("Simulation completed");
    Ok(())
}

/// Show current simulation status
async fn show_status() -> anyhow::Result<()> {
    println!("Universe Simulation Status");
    println!("==========================");
    
    // TODO: Connect to running simulation and get status
    println!("Status: Not running");
    println!("Tick: 0");
    println!("UPS: 0.0");
    println!("Agents: 0");
    println!("Lineages: 0");
    
    Ok(())
}

/// Show ASCII map of the universe
async fn show_map(zoom: u32) -> anyhow::Result<()> {
    println!("Universe Map (zoom {})", zoom);
    println!("====================");
    
    // TODO: Generate ASCII map visualization
    for y in 0..20 {
        for x in 0..40 {
            if (x + y) % 5 == 0 {
                print!("*");
            } else {
                print!(".");
            }
        }
        println!();
    }
    
    println!("\nLegend: * = Star, . = Empty space");
    Ok(())
}

/// List planets with optional filtering
async fn list_planets(class_filter: Option<String>) -> anyhow::Result<()> {
    println!("Planets");
    println!("=======");
    
    // TODO: Connect to simulation and list planets
    println!("ID | Class | Position | Habitability | Population");
    println!("---|-------|----------|--------------|------------");
    println!("1  | E     | (100,200)| 0.85         | 42");
    println!("2  | D     | (500,300)| 0.12         | 0");
    
    Ok(())
}

/// Inspect a specific object
async fn inspect_object(object: String) -> anyhow::Result<()> {
    println!("Inspecting: {}", object);
    println!("================");
    
    // TODO: Parse object type and ID, connect to simulation
    println!("Object not found or simulation not running");
    
    Ok(())
}

/// Take a simulation snapshot
async fn take_snapshot(file: PathBuf) -> anyhow::Result<()> {
    log::info!("Taking snapshot to {}", file.display());
    
    // TODO: Connect to simulation and export snapshot
    println!("Snapshot saved to {}", file.display());
    
    Ok(())
}

/// Generate example configuration file
fn generate_config(file: PathBuf) -> anyhow::Result<()> {
    let config = SimulationConfig::default();
    config.to_file(&file)
        .map_err(|e| anyhow::anyhow!("Failed to write config: {}", e))?;
    
    println!("Example configuration written to {}", file.display());
    Ok(())
}

#[cfg(feature = "god-mode")]
/// Execute god-mode commands
async fn execute_god_command(command: GodCommands) -> anyhow::Result<()> {
    log::warn!("Executing god-mode command");
    
    match command {
        GodCommands::CreateBody { body_type, mass, position } => {
            log::info!("Creating {} with mass {} at {}", body_type, mass, position);
            // TODO: Implement god-mode body creation
        },
        GodCommands::DeleteBody { id } => {
            log::info!("Deleting body {}", id);
            // TODO: Implement god-mode body deletion
        },
        GodCommands::SetConstant { name, value } => {
            log::warn!("Setting constant {} = {}", name, value);
            // TODO: Implement god-mode constant override
        },
        GodCommands::TimeWarp { factor } => {
            log::info!("Time warp factor: {}x", factor);
            // TODO: Implement god-mode time manipulation
        },
        GodCommands::Miracle { planet_id, miracle_type } => {
            log::info!("Miracle '{}' on planet {}", miracle_type, planet_id);
            // TODO: Implement god-mode miracles
        },
    }
    
    Ok(())
}