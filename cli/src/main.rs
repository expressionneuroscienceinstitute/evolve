use anyhow::Result;
use clap::{Parser, Subcommand};
use tracing::{info, warn};
use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt};

mod commands;
mod data_models;
mod formatters;
mod logging;
mod rpc;
mod ascii_viz;
mod gpu_stress_test;

use commands::{
    inspect::InspectCommand,
    list_planets::ListPlanetsCommand,
    snapshot::SnapshotCommand,
    status::StatusCommand,
};

#[derive(Parser)]
#[command(name = "universectl")]
#[command(about = "Command-line interface for the Evolve universe simulation engine")]
#[command(version = "0.1.0")]
#[command(propagate_version = true)]
pub struct Cli {
    #[command(subcommand)]
    pub command: Commands,
    
    /// Enable verbose logging
    #[arg(short, long, global = true)]
    pub verbose: bool,
    
    /// Simulation socket path
    #[arg(long, global = true, default_value = "/tmp/universe.sock")]
    pub socket: String,
    
    /// Configuration file path
    #[arg(long, global = true)]
    pub config: Option<String>,
    
    /// Enable God Mode (requires authentication)
    #[arg(long, global = true)]
    pub godmode: bool,
}

#[derive(Subcommand)]
pub enum Commands {
    /// Show simulation status (tick, UPS, lineage count, mean entropy, save-file age)
    Status {
        /// Refresh interval in seconds (0 for single shot)
        #[arg(short, long, default_value = "0")]
        refresh: u64,
        
        /// Output format
        #[arg(short, long, default_value = "table")]
        format: String,
    },
    
    /// Render an ASCII heat-map of star or entropy densities
    Map {
        /// Zoom level (1-10)
        zoom: Option<u8>,
        
        /// Map type (stars, entropy, temperature, density)
        #[arg(short, long, default_value = "stars")]
        map_type: String,
        
        /// Width of the map in characters
        #[arg(short, long, default_value = "80")]
        width: u16,
        
        /// Height of the map in characters
        #[arg(long, default_value = "40")]
        height: u16,
    },
    
    /// List planets with optional filtering
    ListPlanets {
        /// Planet class filter (E=Earth-like, D=Desert, I=Ice, T=Toxic, G=Gas)
        #[arg(long)]
        class: Option<String>,
        
        /// Minimum habitability score
        #[arg(long)]
        min_habitability: Option<f64>,
        
        /// Output format (table, json, csv)
        #[arg(short, long, default_value = "table")]
        format: String,
        
        /// Sort by field
        #[arg(short, long, default_value = "habitability")]
        sort: String,
        
        /// Limit number of results
        #[arg(short, long)]
        limit: Option<usize>,
    },
    
    /// Inspect detailed information about simulation entities
    Inspect {
        #[command(subcommand)]
        target: InspectTarget,
    },
    
    /// Export a human-readable snapshot for offline analysis
    Snapshot {
        /// Output file path
        file: String,
        
        /// Format (toml, json, yaml)
        #[arg(short, long, default_value = "toml")]
        format: String,
        
        /// Include full state (otherwise summary only)
        #[arg(long)]
        full: bool,
        
        /// Compress output
        #[arg(long)]
        compress: bool,
    },
    
    /// Performance and stress testing utilities
    StressTest {
        /// Test type (gpu, cpu, memory, io)
        test_type: String,
        
        /// Duration in seconds
        #[arg(short, long, default_value = "30")]
        duration: u64,
        
        /// Intensity level (1-10)
        #[arg(short, long, default_value = "5")]
        intensity: u8,
    },
    
    /// Resource management commands
    Resources {
        #[command(subcommand)]
        action: ResourceAction,
    },
    
    /// God Mode commands (requires --godmode flag)
    Divine {
        #[command(subcommand)]
        action: DivineAction,
    },
    
    /// Oracle communication interface
    Oracle {
        #[command(subcommand)]
        action: OracleAction,
    },
}

#[derive(Subcommand)]
pub enum InspectTarget {
    /// Inspect a planet's environment and inhabitants
    Planet {
        /// Planet ID
        id: String,
        
        /// Show detailed environmental data
        #[arg(long)]
        environment: bool,
        
        /// Show active lineages
        #[arg(long)]
        lineages: bool,
        
        /// Show energy budget
        #[arg(long)]
        energy: bool,
        
        /// Show resource composition
        #[arg(long)]
        resources: bool,
    },
    
    /// Inspect a lineage's history and characteristics
    Lineage {
        /// Lineage ID
        id: String,
        
        /// Show fitness history
        #[arg(long)]
        fitness: bool,
        
        /// Show code evolution
        #[arg(long)]
        code: bool,
        
        /// Show genealogy
        #[arg(long)]
        genealogy: bool,
        
        /// Time range for history (e.g., "1h", "1d", "1y")
        #[arg(long)]
        time_range: Option<String>,
    },
    
    /// Inspect a star system
    System {
        /// System ID
        id: String,
        
        /// Show orbital mechanics
        #[arg(long)]
        orbits: bool,
        
        /// Show stellar properties
        #[arg(long)]
        stellar: bool,
    },
    
    /// Inspect simulation performance metrics
    Performance {
        /// Metric type (cpu, memory, network, simulation)
        #[arg(long)]
        metric: Option<String>,
        
        /// Time window for analysis
        #[arg(long, default_value = "1h")]
        window: String,
    },
}

#[derive(Subcommand)]
pub enum ResourceAction {
    /// List resource requests
    Queue,
    
    /// Grant resources to a lineage
    Grant {
        /// Request ID
        request_id: String,
        
        /// Expiration duration
        #[arg(long, default_value = "90d")]
        expires: String,
    },
    
    /// Show resource usage statistics
    Usage {
        /// Entity type (lineage, planet, system)
        #[arg(long)]
        entity_type: Option<String>,
        
        /// Entity ID
        #[arg(long)]
        entity_id: Option<String>,
    },
}

#[derive(Subcommand)]
pub enum DivineAction {
    /// Create a celestial body
    CreateBody {
        /// Body type (planet, star, asteroid)
        body_type: String,
        
        /// Mass (in Earth masses)
        mass: f64,
        
        /// Element composition file
        #[arg(long)]
        composition: Option<String>,
        
        /// Coordinates (x,y,z)
        #[arg(long)]
        position: Option<String>,
    },
    
    /// Remove a celestial body
    DeleteBody {
        /// Body ID
        id: String,
        
        /// Force deletion without confirmation
        #[arg(long)]
        force: bool,
    },
    
    /// Perform a miracle on a planet
    Miracle {
        /// Planet ID
        planet_id: String,
        
        /// Miracle type
        miracle_type: String,
        
        /// Parameters for the miracle
        #[arg(long)]
        params: Option<String>,
    },
    
    /// Adjust time flow
    TimeWarp {
        /// Time factor (0.1 to 1000.0)
        factor: f64,
        
        /// Duration in simulation ticks
        #[arg(long)]
        duration: Option<u64>,
    },
    
    /// Spawn a new lineage
    SpawnLineage {
        /// Code hash or template
        code: String,
        
        /// Planet ID for spawning
        planet_id: String,
        
        /// Initial parameters
        #[arg(long)]
        params: Option<String>,
    },
}

#[derive(Subcommand)]
pub enum OracleAction {
    /// View incoming petitions from sentient agents
    Inbox {
        /// Show only unread messages
        #[arg(long)]
        unread: bool,
        
        /// Filter by lineage ID
        #[arg(long)]
        lineage: Option<String>,
        
        /// Limit number of messages
        #[arg(short, long, default_value = "20")]
        limit: usize,
    },
    
    /// Respond to a petition
    Reply {
        /// Petition ID
        petition_id: String,
        
        /// Response type (ack, nack, grant, message)
        response_type: String,
        
        /// Response message
        #[arg(long)]
        message: Option<String>,
    },
    
    /// Show communication statistics
    Stats {
        /// Time window
        #[arg(long, default_value = "24h")]
        window: String,
    },
}

#[tokio::main]
async fn main() -> Result<()> {
    let cli = Cli::parse();
    
    // Initialize logging
    let log_level = if cli.verbose {
        tracing::Level::DEBUG
    } else {
        tracing::Level::INFO
    };
    
    tracing_subscriber::registry()
        .with(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| {
                    format!("universectl={}", log_level).into()
                })
        )
        .with(tracing_subscriber::fmt::layer())
        .init();
    
    info!("Starting universectl CLI interface");
    
    // Validate God Mode access if required
    if matches!(cli.command, Commands::Divine { .. }) && !cli.godmode {
        anyhow::bail!("Divine commands require --godmode flag. Use with caution!");
    }
    
    // Execute the command
    match cli.command {
        Commands::Status { refresh, format } => {
            let cmd = StatusCommand::new(&cli.socket)?;
            cmd.execute(refresh, &format).await
        }
        
        Commands::Map { zoom, map_type, width, height } => {
            let zoom_level = zoom.unwrap_or(5);
            ascii_viz::render_map(&cli.socket, &map_type, zoom_level, width, height).await
        }
        
        Commands::ListPlanets { class, min_habitability, format, sort, limit } => {
            let cmd = ListPlanetsCommand::new(&cli.socket)?;
            cmd.execute(class.as_deref(), min_habitability, &format, &sort, limit).await
        }
        
        Commands::Inspect { target } => {
            let cmd = InspectCommand::new(&cli.socket)?;
            cmd.execute(target).await
        }
        
        Commands::Snapshot { file, format, full, compress } => {
            let cmd = SnapshotCommand::new(&cli.socket)?;
            cmd.execute(&file, &format, full, compress).await
        }
        
        Commands::StressTest { test_type, duration, intensity } => {
            gpu_stress_test::run_stress_test(&test_type, duration, intensity).await
        }
        
        Commands::Resources { action } => {
            handle_resource_command(&cli.socket, action).await
        }
        
        Commands::Divine { action } => {
            if !cli.godmode {
                anyhow::bail!("Divine commands require --godmode flag!");
            }
            handle_divine_command(&cli.socket, action).await
        }
        
        Commands::Oracle { action } => {
            handle_oracle_command(&cli.socket, action).await
        }
    }
}

async fn handle_resource_command(socket: &str, action: ResourceAction) -> Result<()> {
    match action {
        ResourceAction::Queue => {
            println!("Resource request queue:");
            // Implementation for showing resource queue
            Ok(())
        }
        ResourceAction::Grant { request_id, expires } => {
            println!("Granting resources for request {} (expires: {})", request_id, expires);
            // Implementation for granting resources
            Ok(())
        }
        ResourceAction::Usage { entity_type, entity_id } => {
            println!("Resource usage statistics");
            // Implementation for showing resource usage
            Ok(())
        }
    }
}

async fn handle_divine_command(socket: &str, action: DivineAction) -> Result<()> {
    warn!("Executing divine command - this will be logged!");
    
    match action {
        DivineAction::CreateBody { body_type, mass, composition, position } => {
            println!("Creating {} with mass {} Earth masses", body_type, mass);
            // Implementation for creating celestial body
            Ok(())
        }
        DivineAction::DeleteBody { id, force } => {
            if !force {
                use dialoguer::Confirm;
                if !Confirm::new()
                    .with_prompt(&format!("Are you sure you want to delete body {}?", id))
                    .interact()? 
                {
                    return Ok(());
                }
            }
            println!("Deleting body {}", id);
            // Implementation for deleting body
            Ok(())
        }
        DivineAction::Miracle { planet_id, miracle_type, params } => {
            println!("Performing {} miracle on planet {}", miracle_type, planet_id);
            // Implementation for miracles
            Ok(())
        }
        DivineAction::TimeWarp { factor, duration } => {
            println!("Setting time warp to {}x", factor);
            // Implementation for time warp
            Ok(())
        }
        DivineAction::SpawnLineage { code, planet_id, params } => {
            println!("Spawning lineage {} on planet {}", code, planet_id);
            // Implementation for spawning lineage
            Ok(())
        }
    }
}

async fn handle_oracle_command(socket: &str, action: OracleAction) -> Result<()> {
    match action {
        OracleAction::Inbox { unread, lineage, limit } => {
            println!("Oracle inbox (limit: {})", limit);
            // Implementation for showing oracle inbox
            Ok(())
        }
        OracleAction::Reply { petition_id, response_type, message } => {
            println!("Replying to petition {} with {}", petition_id, response_type);
            // Implementation for replying to petitions
            Ok(())
        }
        OracleAction::Stats { window } => {
            println!("Oracle communication stats for window: {}", window);
            // Implementation for oracle stats
            Ok(())
        }
    }
}