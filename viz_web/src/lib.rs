//! Web Visualization
//! 
//! WASM-based web visualization for the universe simulation

#![cfg(target_arch = "wasm32")]

// Re-export everything from the original dashboard code when targeting WebAssembly.
// The original dashboard lived in `src/main.rs`; we simply include that file here
// so the code is built as a cdylib for wasm targets while native targets skip it.
include!("main.rs");

pub fn render() {
    // Placeholder for web visualization
}