//! Cosmic events system for supernovae, asteroid impacts, etc.

use crate::types::*;
use crate::{Result, SimError};
use serde::{Deserialize, Serialize};

/// Cosmic event that can affect the simulation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum CosmicEvent {
    Supernova {
        star_id: StarId,
        energy_released: f64,
        heavy_elements: ElementTable,
    },
    AsteroidImpact {
        planet_id: PlanetId,
        impact_energy: f64,
        size_km: f64,
    },
    SolarFlare {
        star_id: StarId,
        intensity: f64,
        duration_ticks: u64,
    },
    GammaRayBurst {
        energy: f64,
        direction: Coord3D,
        duration_ticks: u64,
    },
}

/// Event system for managing cosmic events
pub struct EventSystem {
    pending_events: Vec<(Tick, CosmicEvent)>,
    event_history: Vec<(Tick, CosmicEvent)>,
}

impl EventSystem {
    pub fn new() -> Self {
        Self {
            pending_events: Vec::new(),
            event_history: Vec::new(),
        }
    }
    
    /// Schedule a future event
    pub fn schedule_event(&mut self, trigger_tick: Tick, event: CosmicEvent) {
        self.pending_events.push((trigger_tick, event));
        self.pending_events.sort_by_key(|&(tick, _)| tick);
    }
    
    /// Process events for current tick
    pub fn tick(&mut self, current_tick: Tick) -> Vec<CosmicEvent> {
        let mut triggered_events = Vec::new();
        
        while let Some(&(tick, _)) = self.pending_events.first() {
            if tick <= current_tick {
                let (_, event) = self.pending_events.remove(0);
                triggered_events.push(event.clone());
                self.event_history.push((current_tick, event));
            } else {
                break;
            }
        }
        
        triggered_events
    }
    
    /// Get event history
    pub fn get_history(&self) -> &[(Tick, CosmicEvent)] {
        &self.event_history
    }
}