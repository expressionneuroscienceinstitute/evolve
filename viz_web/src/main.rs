//! Universe Portal - Interactive Web Interface for Evolve OS
//! 
//! Provides real-time visualization, monitoring, and manipulation tools
//! for the universe simulation, with special focus on tracking civilizations'
//! quest for existential understanding.

use anyhow::Result;
use axum::{
    extract::{ws::WebSocketUpgrade, Extension, Path, Query, State},
    http::StatusCode,
    response::{Html, IntoResponse, Response},
    routing::{get, post},
    Json, Router,
};
use serde::{Deserialize, Serialize};
use std::{
    net::SocketAddr,
    sync::Arc,
    time::Duration,
};
use tokio::sync::{broadcast, RwLock};
use tower_http::{
    cors::CorsLayer,
    services::{ServeDir, ServeFile},
    trace::TraceLayer,
};
use tracing::{info, warn};
use universe_sim::{Universe, UniverseState};
use uuid::Uuid;

mod auth;
mod handlers;
mod realtime;
mod templates;

use crate::{
    auth::{AuthLayer, Claims},
    handlers::*,
    realtime::handle_websocket,
};

/// Global application state
#[derive(Clone)]
pub struct AppState {
    /// Connection to the universe simulation
    universe: Arc<RwLock<UniverseConnection>>,
    /// Broadcast channel for real-time updates
    update_tx: broadcast::Sender<UniverseUpdate>,
    /// Active sessions
    sessions: Arc<RwLock<SessionStore>>,
}

/// Connection to the running universe simulation
pub struct UniverseConnection {
    /// RPC client to universe daemon
    client: UniverseClient,
    /// Cached state for fast queries
    cached_state: Option<UniverseState>,
    /// Last update timestamp
    last_update: std::time::Instant,
}

/// Real-time update message
#[derive(Clone, Debug, Serialize)]
#[serde(tag = "type")]
pub enum UniverseUpdate {
    TickAdvanced {
        tick: u64,
        year: f64,
    },
    CivilizationProgress {
        id: String,
        milestone: String,
        progress: f32,
    },
    ExistentialDiscovery {
        civilization_id: String,
        discovery: PhilosophicalInsight,
    },
    CosmicEvent {
        description: String,
        location: Vec3,
        magnitude: f32,
    },
    AgentPetition {
        id: String,
        from: String,
        content: String,
    },
}

/// Philosophical or scientific discovery
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct PhilosophicalInsight {
    pub title: String,
    pub description: String,
    pub category: InsightCategory,
    pub confidence: f32,
    pub implications: Vec<String>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum InsightCategory {
    Metaphysics,
    Epistemology,
    Ethics,
    Cosmology,
    Consciousness,
    Mathematics,
    Physics,
    ExistentialTruth,
}

/// Session storage
pub type SessionStore = std::collections::HashMap<Uuid, Session>;

#[derive(Clone)]
pub struct Session {
    pub id: Uuid,
    pub user: Option<String>,
    pub god_mode: bool,
    pub subscriptions: Vec<String>,
}

/// 3D vector for positions
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Vec3 {
    pub x: f64,
    pub y: f64,
    pub z: f64,
}

/// Stub for universe RPC client
pub struct UniverseClient;

impl UniverseClient {
    async fn get_status(&self) -> Result<UniverseStatus> {
        // TODO: Implement actual RPC call
        Ok(UniverseStatus {
            tick: 1337,
            year: 1337000000.0,
            civilizations: 42,
            total_agents: 1_000_000,
            entropy: 0.23,
            ups: 1234.5,
        })
    }
}

#[derive(Debug, Serialize)]
pub struct UniverseStatus {
    pub tick: u64,
    pub year: f64,
    pub civilizations: u32,
    pub total_agents: u64,
    pub entropy: f64,
    pub ups: f64,
}

#[tokio::main]
async fn main() -> Result<()> {
    // Initialize tracing
    tracing_subscriber::fmt()
        .with_target(false)
        .compact()
        .init();

    // Create broadcast channel for updates
    let (update_tx, _) = broadcast::channel(1024);

    // Initialize universe connection
    let universe = Arc::new(RwLock::new(UniverseConnection {
        client: UniverseClient,
        cached_state: None,
        last_update: std::time::Instant::now(),
    }));

    // Create app state
    let app_state = AppState {
        universe: universe.clone(),
        update_tx: update_tx.clone(),
        sessions: Arc::new(RwLock::new(SessionStore::new())),
    };

    // Start background update task
    tokio::spawn(universe_update_loop(universe, update_tx));

    // Build router
    let app = Router::new()
        // Dashboard & visualization routes
        .route("/", get(dashboard_handler))
        .route("/universe/view", get(universe_view_handler))
        .route("/civilizations", get(civilizations_list_handler))
        .route("/civilization/:id", get(civilization_detail_handler))
        
        // Progress tracking
        .route("/progress/existential", get(existential_progress_handler))
        .route("/discoveries", get(discoveries_timeline_handler))
        .route("/philosophy/:civ_id", get(philosophy_tree_handler))
        
        // Interactive tools
        .route("/tools/inspector", get(inspector_tool_handler))
        .route("/tools/timeline", get(timeline_scrubber_handler))
        .route("/tools/influence", get(influence_map_handler))
        
        // API endpoints
        .route("/api/status", get(api_status))
        .route("/api/planets", get(api_planets_list))
        .route("/api/agents/:planet", get(api_agents_on_planet))
        .route("/api/metrics", get(api_metrics))
        
        // Manipulation endpoints (require auth)
        .route("/api/godmode/spawn", post(godmode_spawn_agent))
        .route("/api/godmode/miracle", post(godmode_miracle))
        .route("/api/godmode/timewarp", post(godmode_timewarp))
        .route("/api/oracle/respond", post(oracle_respond))
        
        // WebSocket for real-time updates
        .route("/ws", get(websocket_handler))
        
        // Authentication
        .route("/auth/login", post(auth_login))
        .route("/auth/logout", post(auth_logout))
        
        // Static assets
        .nest_service("/static", ServeDir::new("static"))
        .nest_service("/assets", ServeDir::new("assets"))
        
        // State and middleware
        .layer(CorsLayer::permissive())
        .layer(TraceLayer::new_for_http())
        .with_state(app_state);

    let addr = SocketAddr::from(([0, 0, 0, 0], 8080));
    info!("Universe Portal listening on http://{}", addr);

    axum::serve(
        tokio::net::TcpListener::bind(addr).await?,
        app.into_make_service(),
    )
    .await?;

    Ok(())
}

/// Background task to poll universe state and broadcast updates
async fn universe_update_loop(
    universe: Arc<RwLock<UniverseConnection>>,
    tx: broadcast::Sender<UniverseUpdate>,
) {
    let mut interval = tokio::time::interval(Duration::from_millis(100));
    
    loop {
        interval.tick().await;
        
        // Get latest state
        let status = {
            let conn = universe.read().await;
            match conn.client.get_status().await {
                Ok(status) => status,
                Err(e) => {
                    warn!("Failed to get universe status: {}", e);
                    continue;
                }
            }
        };
        
        // Broadcast tick update
        let _ = tx.send(UniverseUpdate::TickAdvanced {
            tick: status.tick,
            year: status.year,
        });
        
        // TODO: Check for other events and broadcast them
    }
}

// Handler implementations
async fn dashboard_handler(State(state): State<AppState>) -> impl IntoResponse {
    let status = {
        let conn = state.universe.read().await;
        conn.client.get_status().await.unwrap_or_default()
    };
    
    Html(format!(r#"
<!DOCTYPE html>
<html>
<head>
    <title>Universe Portal - Evolve OS</title>
    <link rel="stylesheet" href="/static/portal.css">
    <script src="/static/portal.js" defer></script>
</head>
<body>
    <div class="portal-container">
        <header>
            <h1>🌌 Universe Portal</h1>
            <div class="subtitle">Witnessing the Quest for Existential Understanding</div>
        </header>
        
        <nav>
            <a href="/universe/view">3D Universe</a>
            <a href="/civilizations">Civilizations</a>
            <a href="/discoveries">Discoveries</a>
            <a href="/tools/inspector">Inspector</a>
            <a href="/auth/login" class="auth-link">God Mode</a>
        </nav>
        
        <main>
            <div class="status-grid">
                <div class="status-card">
                    <h3>⏱️ Current Tick</h3>
                    <div class="value">{}</div>
                    <div class="label">Year: {:.2e}</div>
                </div>
                
                <div class="status-card">
                    <h3>🧠 Civilizations</h3>
                    <div class="value">{}</div>
                    <div class="label">Seeking truth</div>
                </div>
                
                <div class="status-card">
                    <h3>🔥 Entropy</h3>
                    <div class="value">{:.1}%</div>
                    <div class="label">Universal decay</div>
                </div>
                
                <div class="status-card">
                    <h3>⚡ Performance</h3>
                    <div class="value">{:.1}</div>
                    <div class="label">Updates/sec</div>
                </div>
            </div>
            
            <section class="existential-tracker">
                <h2>Path to Understanding</h2>
                <div class="progress-timeline">
                    <div class="milestone achieved">
                        <span class="icon">🔬</span>
                        <span class="label">Sentience</span>
                    </div>
                    <div class="milestone achieved">
                        <span class="icon">🏭</span>
                        <span class="label">Industry</span>
                    </div>
                    <div class="milestone partial">
                        <span class="icon">💾</span>
                        <span class="label">Digital</span>
                    </div>
                    <div class="milestone">
                        <span class="icon">🚀</span>
                        <span class="label">Trans-Tech</span>
                    </div>
                    <div class="milestone">
                        <span class="icon">🤔</span>
                        <span class="label">Philosophy</span>
                    </div>
                    <div class="milestone">
                        <span class="icon">🌍</span>
                        <span class="label">Cosmic Aware</span>
                    </div>
                    <div class="milestone ultimate">
                        <span class="icon">✨</span>
                        <span class="label">Truth</span>
                    </div>
                </div>
            </section>
            
            <section class="live-feed">
                <h2>Live Universe Feed</h2>
                <div id="event-stream" class="event-stream">
                    <div class="event">Connecting to universe stream...</div>
                </div>
            </section>
        </main>
        
        <footer>
            <p>Evolve OS - The quest to understand why we are here</p>
        </footer>
    </div>
    
    <script>
        // Connect to WebSocket for live updates
        const ws = new WebSocket('ws://localhost:8080/ws');
        
        ws.onmessage = (event) => {{
            const update = JSON.parse(event.data);
            handleUniverseUpdate(update);
        }};
    </script>
</body>
</html>
    "#, 
        status.tick,
        status.year,
        status.civilizations,
        status.entropy * 100.0,
        status.ups
    ))
}

async fn websocket_handler(
    ws: WebSocketUpgrade,
    State(state): State<AppState>,
) -> impl IntoResponse {
    ws.on_upgrade(|socket| handle_websocket(socket, state))
}

// Stub handlers - to be implemented
async fn universe_view_handler() -> impl IntoResponse {
    Html("<h1>3D Universe View - Coming Soon</h1>")
}

async fn civilizations_list_handler() -> impl IntoResponse {
    Html("<h1>Civilizations List - Coming Soon</h1>")
}

async fn civilization_detail_handler(Path(id): Path<String>) -> impl IntoResponse {
    Html(format!("<h1>Civilization {} - Coming Soon</h1>", id))
}

async fn existential_progress_handler() -> impl IntoResponse {
    Html("<h1>Existential Progress Tracker - Coming Soon</h1>")
}

async fn discoveries_timeline_handler() -> impl IntoResponse {
    Html("<h1>Discoveries Timeline - Coming Soon</h1>")
}

async fn philosophy_tree_handler(Path(civ_id): Path<String>) -> impl IntoResponse {
    Html(format!("<h1>Philosophy Tree for {} - Coming Soon</h1>", civ_id))
}

async fn inspector_tool_handler() -> impl IntoResponse {
    Html("<h1>Universe Inspector - Coming Soon</h1>")
}

async fn timeline_scrubber_handler() -> impl IntoResponse {
    Html("<h1>Timeline Scrubber - Coming Soon</h1>")
}

async fn influence_map_handler() -> impl IntoResponse {
    Html("<h1>Influence Map - Coming Soon</h1>")
}

async fn api_status(State(state): State<AppState>) -> impl IntoResponse {
    let conn = state.universe.read().await;
    match conn.client.get_status().await {
        Ok(status) => Json(status).into_response(),
        Err(e) => (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()).into_response(),
    }
}

async fn api_planets_list() -> impl IntoResponse {
    Json(vec!["Earth", "Mars", "Kepler-442b"])
}

async fn api_agents_on_planet(Path(planet): Path<String>) -> impl IntoResponse {
    Json(format!("Agents on {}", planet))
}

async fn api_metrics() -> impl IntoResponse {
    "# Prometheus metrics endpoint"
}

async fn godmode_spawn_agent() -> impl IntoResponse {
    (StatusCode::UNAUTHORIZED, "Requires God Mode")
}

async fn godmode_miracle() -> impl IntoResponse {
    (StatusCode::UNAUTHORIZED, "Requires God Mode")
}

async fn godmode_timewarp() -> impl IntoResponse {
    (StatusCode::UNAUTHORIZED, "Requires God Mode")
}

async fn oracle_respond() -> impl IntoResponse {
    (StatusCode::UNAUTHORIZED, "Requires Oracle privileges")
}

async fn auth_login() -> impl IntoResponse {
    "Login endpoint"
}

async fn auth_logout() -> impl IntoResponse {
    "Logout endpoint"
}

impl Default for UniverseStatus {
    fn default() -> Self {
        Self {
            tick: 0,
            year: 0.0,
            civilizations: 0,
            total_agents: 0,
            entropy: 0.0,
            ups: 0.0,
        }
    }
}