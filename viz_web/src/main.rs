// Evolve: Advanced AI Evolution Monitoring Portal
//
// The most sophisticated real-time monitoring system ever created for
// fundamental particle simulation and AI agent evolution tracking.
//
// Features:
// - Real-time particle physics visualization
// - Comprehensive AI decision tracking
// - Advanced lineage analysis and family trees
// - Consciousness emergence monitoring
// - Technology development timelines
// - Natural selection analytics
// - Innovation tracking
// - Population dynamics
// - Environmental monitoring
// - Quantum field visualization

use wasm_bindgen::prelude::*;
use web_sys::*;
use js_sys::*;
use serde::{Serialize, Deserialize};
use std::collections::HashMap;
use uuid::Uuid;
use serde_json;
use std::cell::RefCell;
use std::rc::Rc;

// Import our simulation types
use universe_sim::*;
use universe_sim::cosmic_era::CosmicEra;
use agent_evolution::*;
use physics_engine::*;

#[wasm_bindgen]
extern "C" {
    #[wasm_bindgen(js_namespace = console)]
    fn log(s: &str);
    
    #[wasm_bindgen(js_namespace = Math)]
    fn random() -> f64;
}

macro_rules! console_log {
    ($($t:tt)*) => (log(&format_args!($($t)*).to_string()))
}

/// Main application state for the monitoring portal
#[wasm_bindgen]
pub struct EvolutionMonitor {
    canvas: HtmlCanvasElement,
    context: CanvasRenderingContext2d,
    websocket: Option<web_sys::WebSocket>,
    simulation_state: SimulationState,
    view_mode: ViewMode,
    selected_agent: Option<Uuid>,
    selected_lineage: Option<Uuid>,
    time_scale: f64,
    // Visualization config
    particle_size_scale: f64,
    energy_filter_min: f64,
    particle_filter: ParticleFilter,
    decision_analyzer: DecisionAnalyzer,
    lineage_tracker: LineageTracker,
    consciousness_monitor: ConsciousnessMonitor,
    innovation_tracker: InnovationTracker,
    analytics_engine: AnalyticsEngine,
    connected: bool,
}

/// Complete simulation state received from backend
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SimulationState {
    pub current_tick: u64,
    pub universe_age_gyr: f64,
    pub cosmic_era: CosmicEra,
    pub temperature: f64,
    pub energy_density: f64,
    
    // Fundamental physics
    pub particles: Vec<ParticleVisualization>,
    pub quantum_fields: HashMap<String, FieldVisualization>,
    pub nuclei: Vec<NucleusVisualization>,
    pub atoms: Vec<AtomVisualization>,
    pub molecules: Vec<MoleculeVisualization>,
    
    // AI agents and evolution
    pub agents: Vec<AgentVisualization>,
    pub lineages: HashMap<Uuid, LineageVisualization>,
    pub decisions: Vec<DecisionVisualization>,
    pub innovations: Vec<InnovationVisualization>,
    pub consciousness_events: Vec<ConsciousnessEvent>,
    
    // Environmental conditions
    pub environments: Vec<EnvironmentVisualization>,
    pub selection_pressures: Vec<SelectionPressureVisualization>,
    
    // Statistics and analytics
    pub population_stats: PopulationStatistics,
    pub evolution_metrics: EvolutionMetrics,
    pub physics_metrics: PhysicsMetrics,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum ViewMode {
    ParticlePhysics,
    AtomicStructure,
    MolecularDynamics,
    AgentOverview,
    LineageTree,
    DecisionTracking,
    ConsciousnessMap,
    InnovationTimeline,
    SelectionPressures,
    EnvironmentalMap,
    PopulationDynamics,
    QuantumFields,
    EnergyFlows,
    EmergentComplexity,
}

/// Particle visualization data
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ParticleVisualization {
    pub id: usize,
    pub particle_type: String,
    pub position: [f64; 3],
    pub momentum: [f64; 3],
    pub energy: f64,
    pub mass: f64,
    pub charge: f64,
    pub spin: [f64; 3],
    pub color_charge: Option<String>,
    pub interaction_count: u32,
    pub age: f64,
    pub decay_probability: f64,
}

/// AI agent visualization with comprehensive tracking
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentVisualization {
    pub id: Uuid,
    pub generation: u32,
    pub birth_tick: u64,
    pub energy: f64,
    pub fitness: f64,
    
    // Evolution levels
    pub sentience_level: f64,
    pub industrialization_level: f64,
    pub digitalization_level: f64,
    pub tech_level: f64,
    pub consciousness_level: f64,
    pub immortality_achieved: bool,
    
    // Decision making
    pub recent_decisions: Vec<DecisionSummary>,
    pub success_rate: f64,
    pub learning_rate: f64,
    pub mutation_rate: f64,
    
    // Social and environmental
    pub social_connections: Vec<Uuid>,
    pub environmental_adaptations: Vec<String>,
    pub innovations_created: Vec<String>,
    
    // Lineage information
    pub parent_id: Option<Uuid>,
    pub children_ids: Vec<Uuid>,
    pub lineage_id: Uuid,
    
    // Physical representation
    pub position: [f64; 3],
    pub size: f64,
    pub color: [f32; 4],
}

/// Comprehensive decision tracking
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DecisionVisualization {
    pub id: Uuid,
    pub agent_id: Uuid,
    pub timestamp: u64,
    pub decision_type: String,
    pub success: bool,
    pub fitness_impact: f64,
    pub energy_cost: f64,
    pub learning_feedback: f64,
    
    // Context information
    pub environmental_factors: Vec<String>,
    pub social_factors: Vec<String>,
    pub genetic_influences: Vec<String>,
    
    // Outcome tracking
    pub immediate_effects: Vec<String>,
    pub long_term_consequences: Vec<String>,
    pub innovation_progress: f64,
}

/// Lineage visualization with complete family tree
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LineageVisualization {
    pub lineage_id: Uuid,
    pub founder_id: Uuid,
    pub current_generation: u32,
    pub total_individuals: u64,
    pub living_individuals: u32,
    pub extinct_branches: u32,
    
    // Evolutionary metrics
    pub average_fitness: f64,
    pub fitness_variance: f64,
    pub genetic_diversity: f64,
    pub mutation_rate: f64,
    
    // Achievement tracking
    pub major_innovations: Vec<InnovationRecord>,
    pub consciousness_milestones: Vec<ConsciousnessMilestone>,
    pub technology_achievements: Vec<TechnologyAchievement>,
    
    // Natural selection
    pub selection_events: Vec<SelectionEventRecord>,
    pub adaptation_events: Vec<AdaptationRecord>,
    pub speciation_events: Vec<SpeciationRecord>,
    
    // Visualization data
    pub family_tree: FamilyTreeVisualization,
    pub evolution_timeline: Vec<EvolutionTimelineEvent>,
}

/// Innovation tracking and visualization
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InnovationVisualization {
    pub id: Uuid,
    pub creator_id: Uuid,
    pub innovation_type: String,
    pub timestamp: u64,
    pub impact_score: f64,
    pub adoption_rate: f64,
    pub complexity_level: f64,
    
    // Innovation characteristics
    pub prerequisites: Vec<String>,
    pub enables: Vec<String>,
    pub resource_requirements: HashMap<String, f64>,
    
    // Spread and evolution
    pub adopters: Vec<Uuid>,
    pub variations: Vec<InnovationVariation>,
    pub obsolescence_risk: f64,
}

/// Consciousness emergence monitoring
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConsciousnessEvent {
    pub agent_id: Uuid,
    pub timestamp: u64,
    pub consciousness_type: ConsciousnessType,
    pub level: f64,
    pub triggers: Vec<String>,
    pub neural_complexity: f64,
    pub self_awareness_indicators: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ConsciousnessType {
    SelfAwareness,
    TheoryOfMind,
    AbstractThinking,
    MetaCognition,
    CreativeInsight,
    ExistentialReflection,
    CollectiveConsciousness,
}

/// Environmental visualization
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EnvironmentVisualization {
    pub id: Uuid,
    pub environment_type: String,
    pub position: [f64; 3],
    pub size: f64,
    pub habitability: f64,
    pub resource_abundance: HashMap<String, f64>,
    pub hazard_level: f64,
    pub agent_count: u32,
    pub carrying_capacity: u32,
}

/// Population statistics and dynamics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PopulationStatistics {
    pub total_agents: u32,
    pub active_lineages: u32,
    pub extinct_lineages: u32,
    pub birth_rate: f64,
    pub death_rate: f64,
    pub mutation_rate: f64,
    pub average_fitness: f64,
    pub fitness_variance: f64,
    pub genetic_diversity: f64,
    pub consciousness_distribution: HashMap<String, u32>,
    pub technology_distribution: HashMap<String, u32>,
}

#[wasm_bindgen]
impl EvolutionMonitor {
    #[wasm_bindgen(constructor)]
    pub fn new(canvas_id: &str) -> Result<EvolutionMonitor, JsValue> {
        console_log!("Initializing Evolve: AI Evolution Monitor");
        
        let document = window().unwrap().document().unwrap();
        let canvas = document
            .get_element_by_id(canvas_id)
            .unwrap()
            .dyn_into::<HtmlCanvasElement>()?;
        
        let context = canvas
            .get_context("2d")?
            .unwrap()
            .dyn_into::<CanvasRenderingContext2d>()?;
        
        // Set up high-DPI canvas
        let dpr = window().unwrap().device_pixel_ratio();
        canvas.set_width((canvas.offset_width() as f64 * dpr) as u32);
        canvas.set_height((canvas.offset_height() as f64 * dpr) as u32);
        context.scale(dpr, dpr)?;
        
        Ok(EvolutionMonitor {
            canvas,
            context,
            websocket: None,
            simulation_state: SimulationState::default(),
            view_mode: ViewMode::AgentOverview,
            selected_agent: None,
            selected_lineage: None,
            time_scale: 1.0,
            particle_size_scale: 1.0,
            energy_filter_min: 0.0,
            particle_filter: ParticleFilter::default(),
            decision_analyzer: DecisionAnalyzer::new(),
            lineage_tracker: LineageTracker::new(),  
            consciousness_monitor: ConsciousnessMonitor::new(),
            innovation_tracker: InnovationTracker::new(),
            analytics_engine: AnalyticsEngine::new(),
            connected: false,
        })
    }
    
    /// Establish websocket and start reading JSON messages representing SimulationState
    pub fn connect_ws(&mut self, websocket_url: String) {
        console_log!("Connecting to WebSocket at {}", websocket_url);
        let ws = match web_sys::WebSocket::new(&websocket_url) {
            Ok(ws) => ws,
            Err(err) => {
                console_log!("Failed to create WebSocket: {:?}", err);
                return;
            }
        };
        
        // Set up connection success handler
        let onopen_callback = Closure::wrap(Box::new(move |_: web_sys::Event| {
            console_log!("WebSocket connected successfully");
            if let Some(doc) = web_sys::window().and_then(|w| w.document()) {
                if let Some(elem) = doc.get_element_by_id("connection-status") {
                    elem.set_inner_html("\u{1F7E2} Connected");
                    elem.set_class_name("connection-status connected");
                }
            }
        }) as Box<dyn FnMut(web_sys::Event)>);
        
        ws.set_onopen(Some(onopen_callback.as_ref().unchecked_ref()));
        onopen_callback.forget();

        // Set up message handler
        let onmessage_callback = Closure::wrap(Box::new(move |e: MessageEvent| {
            if let Ok(text) = e.data().dyn_into::<JsString>() {
                let data_str = text.as_string().unwrap_or_default();
                console_log!("Received WebSocket message: {}", data_str);
                // TODO: Parse and update simulation state
            } else {
                let message_type = match e.data().dyn_into::<ArrayBuffer>() {
                    Ok(buffer) => format!("binary (size: {} bytes)", buffer.byte_length()),
                    Err(_) => "unknown type".to_string(),
                };
                console_log!("Received non-text WebSocket message: {}", message_type);
            }
        }) as Box<dyn FnMut(MessageEvent)>);
        
        ws.set_onmessage(Some(onmessage_callback.as_ref().unchecked_ref()));
        onmessage_callback.forget();

        // Set up error handler
        let onerror_callback = Closure::wrap(Box::new(move |_: web_sys::Event| {
            console_log!("WebSocket connection error");
            if let Some(doc) = web_sys::window().and_then(|w| w.document()) {
                if let Some(elem) = doc.get_element_by_id("connection-status") {
                    elem.set_inner_html("\u{1F534} Disconnected");
                    elem.set_class_name("connection-status disconnected");
                }
            }
        }) as Box<dyn FnMut(web_sys::Event)>);
        
        ws.set_onerror(Some(onerror_callback.as_ref().unchecked_ref()));
        onerror_callback.forget();

        // Set up close handler
        let onclose_callback = Closure::wrap(Box::new(move |_: web_sys::CloseEvent| {
            console_log!("WebSocket connection closed");
            if let Some(doc) = web_sys::window().and_then(|w| w.document()) {
                if let Some(elem) = doc.get_element_by_id("connection-status") {
                    elem.set_inner_html("\u{1F534} Disconnected");
                    elem.set_class_name("connection-status disconnected");
                }
            }
        }) as Box<dyn FnMut(web_sys::CloseEvent)>);
        
        ws.set_onclose(Some(onclose_callback.as_ref().unchecked_ref()));
        onclose_callback.forget();

        self.websocket = Some(ws);
        self.connected = true;
    }
    
    /// Start RAF loop for continuous rendering
    pub fn start_render_loop(&mut self) -> Result<(), JsValue> {
        // Placeholder RAF loop – to be replaced with full implementation.
        Ok(())
    }
    
    /// Main rendering loop
    #[wasm_bindgen]
    pub fn render(&mut self) -> Result<(), JsValue> {
        // Clear canvas
        self.context.clear_rect(0.0, 0.0, 
            self.canvas.width() as f64, 
            self.canvas.height() as f64);
        
        match self.view_mode {
            ViewMode::ParticlePhysics => self.render_particle_physics()?,
            ViewMode::AgentOverview => self.render_agent_overview()?,
            ViewMode::LineageTree => self.render_lineage_tree()?,
            ViewMode::DecisionTracking => self.render_decision_tracking()?,
            ViewMode::ConsciousnessMap => self.render_consciousness_map()?,
            ViewMode::InnovationTimeline => self.render_innovation_timeline()?,
            ViewMode::SelectionPressures => self.render_selection_pressures(&Vec::new())?,
            ViewMode::QuantumFields => self.render_quantum_fields()?,
            _ => self.render_agent_overview()?,
        }
        
        // Render UI overlay
        self.render_ui_overlay()?;
        
        Ok(())
    }
    
    /// Render fundamental particle physics visualization
    fn render_particle_physics(&mut self) -> Result<(), JsValue> {
        console_log!("Rendering particle physics - {} particles", 
                    self.simulation_state.particles.len());
        
        // Render particles with different colors based on type
        for particle in self.simulation_state.particles.clone() {
            if particle.energy < self.energy_filter_min { continue; }
            let (r, g, b, a) = self.get_particle_color(&particle.particle_type);
            self.context.set_fill_style(&JsValue::from_str(&format!(
                "rgba({}, {}, {}, {})", r, g, b, a
            )));
            
            // Map 3D position to 2D screen coordinates
            let (x, y) = self.map_3d_to_2d(particle.position);
            let size = self.get_particle_size(particle.mass) * self.particle_size_scale;
            
            // Draw particle
            self.context.begin_path();
            self.context.arc(x, y, size, 0.0, 2.0 * std::f64::consts::PI)?;
            self.context.fill();
            
            // Draw particle trails for high-energy particles
            if particle.energy > 1e-13 {
                self.draw_particle_trail(&particle)?;
            }
            
            // Draw interaction lines
            if particle.interaction_count > 0 {
                self.draw_interaction_indicators(&particle)?;
            }
        }
        
        // Render quantum field fluctuations
        self.render_quantum_field_overlay()?;
        
        Ok(())
    }
    
    /// Render AI agent overview with comprehensive tracking
    fn render_agent_overview(&mut self) -> Result<(), JsValue> {
        console_log!("Rendering {} AI agents", self.simulation_state.agents.len());
        
        for agent in self.simulation_state.agents.clone() {
            // Agent position and basic rendering
            let (x, y) = self.map_3d_to_2d(agent.position);
            let size = 5.0 + agent.consciousness_level * 15.0;
            
            // Agent color based on evolution level
            let color = self.get_agent_color(&agent);
            self.context.set_fill_style(&JsValue::from_str(&format!(
                "rgba({}, {}, {}, {})", 
                (color[0] * 255.0) as u8,
                (color[1] * 255.0) as u8, 
                (color[2] * 255.0) as u8,
                color[3]
            )));
            
            // Draw agent
            self.context.begin_path();
            self.context.arc(x, y, size, 0.0, 2.0 * std::f64::consts::PI)?;
            self.context.fill();
            
            // Draw consciousness indicator
            if agent.consciousness_level > 0.1 {
                self.draw_consciousness_indicator(&agent, x, y)?;
            }
            
            // Draw innovation aura
            if !agent.innovations_created.is_empty() {
                self.draw_innovation_aura(&agent, x, y)?;
            }
            
            // Draw social connections
            self.draw_social_connections(&agent)?;
            
            // Show decision trails
            if !agent.recent_decisions.is_empty() {
                self.draw_decision_trail(&agent)?;
            }
            
            // Highlight selected agent
            if Some(agent.id) == self.selected_agent {
                self.highlight_selected_agent(&agent, x, y)?;
            }
        }
        
        Ok(())
    }
    
    /// Render comprehensive lineage tree visualization
    fn render_lineage_tree(&mut self) -> Result<(), JsValue> {
        console_log!("Rendering lineage trees for {} lineages", 
                    self.simulation_state.lineages.len());
        
        let mut y_offset = 50.0;
        
        for (lineage_id, lineage) in self.simulation_state.lineages.clone() {
            // Draw lineage header
            self.context.set_font("16px Arial");
            self.context.set_fill_style(&JsValue::from_str("white"));
            self.context.fill_text(&format!(
                "Lineage {} - Gen {} - {} living", 
                lineage_id.to_string()[..8].to_string(),
                lineage.current_generation,
                lineage.living_individuals
            ), 10.0, y_offset)?;
            
            y_offset += 30.0;
            
            // Draw family tree structure
            self.render_family_tree(&lineage.family_tree, 10.0, y_offset)?;
            
            // Draw evolution timeline
            self.render_evolution_timeline(&lineage.evolution_timeline, y_offset + 100.0)?;
            
            // Draw innovation milestones
            self.render_innovation_milestones(&lineage.major_innovations, y_offset + 150.0)?;
            
            y_offset += 200.0;
        }
        
        Ok(())
    }
    
    /// Render decision tracking visualization
    fn render_decision_tracking(&mut self) -> Result<(), JsValue> {
        // Create decision flow diagram
        let mut x_pos = 50.0;
        let y_base = 200.0;
        
        for decision in self.simulation_state.decisions.clone() {
            let (x, y) = (x_pos, y_base + (decision.fitness_impact * 100.0));
            
            // Color based on success/failure
            let color = if decision.success { "green" } else { "red" };
            self.context.set_fill_style(&JsValue::from_str(color));
            
            // Draw decision point
            self.context.begin_path();
            self.context.arc(x, y, 5.0, 0.0, 2.0 * std::f64::consts::PI)?;
            self.context.fill();
            
            // Draw decision type label
            self.context.set_font("10px Arial");
            self.context.set_fill_style(&JsValue::from_str("white"));
            self.context.fill_text(&decision.decision_type, x - 20.0, y - 10.0)?;
            
            // Draw impact line
            self.context.set_stroke_style(&JsValue::from_str(color));
            self.context.set_line_width(2.0);
            self.context.begin_path();
            self.context.move_to(x, y_base);
            self.context.line_to(x, y);
            self.context.stroke();
            
            x_pos += 20.0;
        }
        
        Ok(())
    }
    
    /// Render consciousness emergence map
    fn render_consciousness_map(&mut self) -> Result<(), JsValue> {
        // Create consciousness landscape
        for agent in self.simulation_state.agents.clone() {
            let (x, y) = self.map_3d_to_2d(agent.position);
            let consciousness = agent.consciousness_level;
            
            if consciousness > 0.01 {
                // Draw consciousness "height" as vertical bars
                let height = consciousness * 100.0;
                let alpha = consciousness;
                
                self.context.set_stroke_style(&JsValue::from_str(&format!(
                    "rgba(255, 255, 0, {})", alpha
                )));
                self.context.set_line_width(2.0);
                
                self.context.begin_path();
                self.context.move_to(x, y);
                self.context.line_to(x, y - height);
                self.context.stroke();
                
                // Draw consciousness level indicator
                self.context.set_fill_style(&JsValue::from_str(&format!(
                    "rgba(255, 255, 0, {})", alpha
                )));
                self.context.begin_path();
                self.context.arc(x, y - height, 3.0, 0.0, 2.0 * std::f64::consts::PI)?;
                self.context.fill();
            }
        }
        
        Ok(())
    }
    
    /// Render innovation timeline
    fn render_innovation_timeline(&mut self) -> Result<(), JsValue> {
        let timeline_y = 400.0;
        let timeline_start = 50.0;
        let timeline_width = self.canvas.width() as f64 - 100.0;
        
        // Draw timeline base
        self.context.set_stroke_style(&JsValue::from_str("white"));
        self.context.set_line_width(2.0);
        self.context.begin_path();
        self.context.move_to(timeline_start, timeline_y);
        self.context.line_to(timeline_start + timeline_width, timeline_y);
        self.context.stroke();
        
        // Plot innovations
        for innovation in self.simulation_state.innovations.clone() {
            let time_ratio = innovation.timestamp as f64 / self.simulation_state.current_tick as f64;
            let x = timeline_start + (time_ratio * timeline_width);
            let impact_height = innovation.impact_score * 50.0;
            
            // Innovation marker
            self.context.set_fill_style(&JsValue::from_str("cyan"));
            self.context.begin_path();
            self.context.arc(x, timeline_y - impact_height, 4.0, 0.0, 2.0 * std::f64::consts::PI)?;
            self.context.fill();
            
            // Innovation label
            self.context.set_font("8px Arial");
            self.context.set_fill_style(&JsValue::from_str("white"));
            self.context.fill_text(&innovation.innovation_type, x - 15.0, timeline_y - impact_height - 8.0)?;
        }
        
        Ok(())
    }
    
    /// Get color for particle type
    fn get_particle_color(&self, particle_type: &str) -> (u8, u8, u8, f32) {
        match particle_type {
            "Electron" => (255, 255, 0, 1.0),     // Yellow
            "Proton" => (255, 0, 0, 1.0),         // Red  
            "Neutron" => (128, 128, 128, 1.0),    // Gray
            "Photon" => (255, 255, 255, 0.7),     // White, translucent
            "Up" | "Down" => (0, 255, 0, 1.0),    // Green (quarks)
            "Gluon" => (255, 0, 255, 0.8),        // Magenta
            "Higgs" => (255, 215, 0, 1.0),        // Gold
            _ => (128, 128, 255, 0.8),             // Light blue (other)
        }
    }
    
    /// Get agent color based on evolution level
    fn get_agent_color(&self, agent: &AgentVisualization) -> [f32; 4] {
        let r = agent.sentience_level as f32;
        let g = agent.tech_level as f32;
        let b = agent.consciousness_level as f32;
        let a = 0.8 + agent.industrialization_level as f32 * 0.2;
        [r, g, b, a]
    }
    
    /// Map 3D position to 2D screen coordinates
    fn map_3d_to_2d(&self, pos: [f64; 3]) -> (f64, f64) {
        let scale = 1000.0;
        let center_x = self.canvas.width() as f64 / 2.0;
        let center_y = self.canvas.height() as f64 / 2.0;
        
        let x = center_x + pos[0] * scale;
        let y = center_y + pos[1] * scale;
        
        (x, y)
    }
    
    /// Set view mode
    #[wasm_bindgen]
    pub fn set_view_mode(&mut self, mode: u8) {
        self.view_mode = match mode {
            0 => ViewMode::ParticlePhysics,
            1 => ViewMode::AgentOverview,
            2 => ViewMode::LineageTree,
            3 => ViewMode::DecisionTracking,
            4 => ViewMode::ConsciousnessMap,
            5 => ViewMode::InnovationTimeline,
            6 => ViewMode::SelectionPressures,
            7 => ViewMode::QuantumFields,
            _ => ViewMode::AgentOverview,
        };
        console_log!("View mode changed to: {:?}", self.view_mode);
    }
    
    /// Select an agent for detailed tracking
    #[wasm_bindgen]
    pub fn select_agent(&mut self, agent_id: &str) {
        if let Ok(uuid) = Uuid::parse_str(agent_id) {
            self.selected_agent = Some(uuid);
            console_log!("Selected agent: {}", agent_id);
        }
    }
    
    // Placeholder implementations for complex rendering methods
    fn get_particle_size(&self, mass: f64) -> f64 { 2.0 + (mass / 1e-27).log10().max(0.0) }
    fn draw_particle_trail(&mut self, _particle: &ParticleVisualization) -> Result<(), JsValue> { Ok(()) }
    fn draw_interaction_indicators(&mut self, _particle: &ParticleVisualization) -> Result<(), JsValue> { Ok(()) }
    fn render_quantum_field_overlay(&mut self) -> Result<(), JsValue> { Ok(()) }
    fn render_quantum_fields(&mut self) -> Result<(), JsValue> { Ok(()) }
    fn render_selection_pressures(&mut self, _: &Vec<SelectionPressureVisualization>) -> Result<(), JsValue> { Ok(()) }
    fn draw_consciousness_indicator(&mut self, _agent: &AgentVisualization, _x: f64, _y: f64) -> Result<(), JsValue> { Ok(()) }
    fn draw_innovation_aura(&mut self, _agent: &AgentVisualization, _x: f64, _y: f64) -> Result<(), JsValue> { Ok(()) }
    fn draw_social_connections(&mut self, _agent: &AgentVisualization) -> Result<(), JsValue> { Ok(()) }
    fn draw_decision_trail(&mut self, _agent: &AgentVisualization) -> Result<(), JsValue> { Ok(()) }
    fn highlight_selected_agent(&mut self, _agent: &AgentVisualization, _x: f64, _y: f64) -> Result<(), JsValue> { Ok(()) }
    fn render_family_tree(&mut self, _tree: &FamilyTreeVisualization, _x: f64, _y: f64) -> Result<(), JsValue> { Ok(()) }
    fn render_evolution_timeline(&mut self, _timeline: &Vec<EvolutionTimelineEvent>, _y: f64) -> Result<(), JsValue> { Ok(()) }
    fn render_innovation_milestones(&mut self, _innovations: &Vec<InnovationRecord>, _y: f64) -> Result<(), JsValue> { Ok(()) }
    fn render_ui_overlay(&mut self) -> Result<(), JsValue> { Ok(()) }
    
    #[wasm_bindgen]
    pub fn set_particle_size_scale(&mut self, scale: f64) { self.particle_size_scale = scale; }
    #[wasm_bindgen]
    pub fn set_energy_filter_min(&mut self, min_kev: f64) { self.energy_filter_min = min_kev * 1.60218e-16; }

    #[wasm_bindgen]
    pub fn connect(&mut self, websocket_url: &str) -> Result<(), JsValue> {
        self.connect_ws(websocket_url.to_string());
        Ok(())
    }

    fn set_fill_style(&self, color: &str) -> Result<(), JsValue> {
        self.context.set_fill_style(&JsValue::from_str(color));
        Ok(())
    }

    fn set_stroke_style(&self, color: &str) -> Result<(), JsValue> {
        self.context.set_stroke_style(&JsValue::from_str(color));
        Ok(())
    }

    fn draw_particle(&mut self, particle: &ParticleVisualization) -> Result<(), JsValue> {
        let (x, y) = self.map_3d_to_2d(particle.position);
        let size = self.get_particle_size(particle.mass) * self.particle_size_scale;
        let (r, g, b, a) = self.get_particle_color(&particle.particle_type);
        
        self.context.begin_path();
        self.context.arc(x, y, size, 0.0, std::f64::consts::PI * 2.0)?;
        self.set_fill_style(&format!("rgba({},{},{},{})", r, g, b, a))?;
        self.context.fill();
        
        Ok(())
    }

    fn draw_agent(&mut self, agent: &AgentVisualization) -> Result<(), JsValue> {
        let (x, y) = self.map_3d_to_2d(agent.position);
        let size = 10.0 + agent.tech_level as f64 * 2.0;
        let color = self.get_agent_color(agent);
        
        self.context.begin_path();
        self.context.arc(x, y, size, 0.0, std::f64::consts::PI * 2.0)?;
        self.set_fill_style(&format!("rgba({},{},{},{})", 
            (color[0] * 255.0) as u8,
            (color[1] * 255.0) as u8,
            (color[2] * 255.0) as u8,
            color[3]))?;
        self.context.fill();
        
        Ok(())
    }

    fn draw_ui_text(&mut self, text: &str, x: f64, y: f64) -> Result<(), JsValue> {
        self.set_fill_style("white")?;
        self.context.fill_text(text, x, y)?;
        Ok(())
    }

    fn draw_ui_box(&mut self, x: f64, y: f64, width: f64, height: f64, color: &str) -> Result<(), JsValue> {
        self.context.begin_path();
        self.context.rect(x, y, width, height);
        self.set_fill_style(color)?;
        self.context.fill();
        self.set_stroke_style("white")?;
        self.context.stroke();
        Ok(())
    }

    fn draw_ui_line(&mut self, x1: f64, y1: f64, x2: f64, y2: f64, color: &str) -> Result<(), JsValue> {
        self.context.begin_path();
        self.context.move_to(x1, y1);
        self.context.line_to(x2, y2);
        self.set_stroke_style(color)?;
        self.context.stroke();
        Ok(())
    }

    fn draw_ui_graph(&mut self, data: &[f64], x: f64, y: f64, width: f64, height: f64) -> Result<(), JsValue> {
        if data.is_empty() {
            return Ok(());
        }
        
        let max = data.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
        let min = data.iter().fold(f64::INFINITY, |a, &b| a.min(b));
        let range = max - min;
        
        self.context.begin_path();
        self.context.move_to(x, y + height);
        
        for (i, &value) in data.iter().enumerate() {
            let px = x + (i as f64 / (data.len() - 1) as f64) * width;
            let py = y + height - ((value - min) / range) * height;
            self.context.line_to(px, py);
        }
        
        self.set_stroke_style(&format!("rgba(0,255,255,0.8)"))?;
        self.context.stroke();
        
        // Draw axes
        self.set_stroke_style("white")?;
        self.context.begin_path();
        self.context.move_to(x, y);
        self.context.line_to(x, y + height);
        self.context.line_to(x + width, y + height);
        self.context.stroke();
        
        Ok(())
    }

    fn draw_ui_panel(&mut self) -> Result<(), JsValue> {
        let width = self.canvas.width() as f64;
        let height = self.canvas.height() as f64;
        
        // Draw semi-transparent background
        self.context.begin_path();
        self.context.rect(0.0, 0.0, width, 60.0);
        self.set_fill_style("rgba(0,0,0,0.7)")?;
        self.context.fill();
        
        // Draw stats
        self.set_fill_style("white")?;
        self.context.fill_text(&format!("Age: {:.2} Gyr", self.simulation_state.universe_age_gyr), 10.0, 20.0)?;
        self.context.fill_text(&format!("Particles: {}", self.simulation_state.particles.len()), 200.0, 20.0)?;
        self.context.fill_text(&format!("Agents: {}", self.simulation_state.agents.len()), 400.0, 20.0)?;
        
        Ok(())
    }
}

// Supporting types and structures
#[derive(Debug, Default)]
pub struct ParticleFilter {
    pub show_quarks: bool,
    pub show_leptons: bool,
    pub show_bosons: bool,
    pub min_energy: f64,
    pub max_energy: f64,
}

pub struct DecisionAnalyzer;
impl DecisionAnalyzer {
    pub fn new() -> Self { Self }
}

pub struct ConsciousnessMonitor;
impl ConsciousnessMonitor {
    pub fn new() -> Self { Self }
}

pub struct InnovationTracker;
impl InnovationTracker {
    pub fn new() -> Self { Self }
}

pub struct AnalyticsEngine;
impl AnalyticsEngine {
    pub fn new() -> Self { Self }
}

pub struct LineageTracker;

impl LineageTracker {
    pub fn new() -> Self { Self }
}

// Placeholder types for visualization data
pub type DecisionSummary = String;
pub type InnovationRecord = String;
pub type ConsciousnessMilestone = String;
pub type TechnologyAchievement = String;
pub type SelectionEventRecord = String;
pub type AdaptationRecord = String;
pub type SpeciationRecord = String;
pub type FamilyTreeVisualization = String;
pub type EvolutionTimelineEvent = String;
pub type InnovationVariation = String;
pub type FieldVisualization = String;
pub type NucleusVisualization = String;
pub type AtomVisualization = String;
pub type MoleculeVisualization = String;
pub type SelectionPressureVisualization = String;
pub type EvolutionMetrics = String;
pub type PhysicsMetrics = String;

impl Default for SimulationState {
    fn default() -> Self {
        Self {
            current_tick: 0,
            universe_age_gyr: 0.0,
            cosmic_era: CosmicEra::ParticleSoup,
            temperature: 0.0,
            energy_density: 0.0,
            particles: Vec::new(),
            quantum_fields: HashMap::new(),
            nuclei: Vec::new(),
            atoms: Vec::new(),
            molecules: Vec::new(),
            agents: Vec::new(),
            lineages: HashMap::new(),
            decisions: Vec::new(),
            innovations: Vec::new(),
            consciousness_events: Vec::new(),
            environments: Vec::new(),
            selection_pressures: Vec::new(),
            population_stats: PopulationStatistics::default(),
            evolution_metrics: "".to_string(),
            physics_metrics: "".to_string(),
        }
    }
}

impl Default for PopulationStatistics {
    fn default() -> Self {
        Self {
            total_agents: 0,
            active_lineages: 0,
            extinct_lineages: 0,
            birth_rate: 0.0,
            death_rate: 0.0,
            mutation_rate: 0.0,
            average_fitness: 0.0,
            fitness_variance: 0.0,
            genetic_diversity: 0.0,
            consciousness_distribution: HashMap::new(),
            technology_distribution: HashMap::new(),
        }
    }
}

// Utility functions
fn window() -> Option<Window> {
    web_sys::window()
}

// Add helper to get window
fn web_window() -> web_sys::Window { web_sys::window().expect("no global window") }

/// Convenience exported function to start the dashboard from JS
#[wasm_bindgen]
pub fn start_dashboard(canvas_id: &str, websocket_url: &str) -> Result<EvolutionMonitor, JsValue> {
    let mut monitor = EvolutionMonitor::new(canvas_id)?;
    monitor.connect_ws(websocket_url.to_string());
    monitor.start_render_loop()?;
    Ok(monitor)
}