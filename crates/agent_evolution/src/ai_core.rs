//! AI Core placeholder (stub)

use anyhow::Result;

pub struct AICore;

impl AICore {
    pub fn new() -> Self { Self }
    pub fn tick(&mut self, _dt: f64) -> Result<()> { Ok(()) }
}