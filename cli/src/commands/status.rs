use crate::data_models::SimulationState;
use anyhow::Result;
use crossterm::{
    cursor::{Hide, MoveTo, Show},
    event::{self, Event, KeyCode},
    execute,
    style::{Color, ResetColor, SetForegroundColor},
    terminal::{self, Clear, ClearType},
};
use std::{
    io::{stdout},
    time::{Duration, Instant},
};

pub async fn show_status(source: &str, detailed: bool) -> Result<()> {
    let state = crate::load_simulation_state(source).await?;
    
    println!("╔══════════════════════════════════════════════╗");
    println!("║         Universe Simulation Status           ║");
    println!("╠══════════════════════════════════════════════╣");
    println!("║ Tick:         {:>30} ║", state.tick);
    println!("║ UPS:          {:>28.1} Hz ║", state.ups);
    println!("║ Sim Time:     {:>24.2} Gyr ║", state.current_time / 1e9);
    println!("║ Entropy:      {:>28.2e} ║", state.entropy);
    println!("║ Temperature:  {:>26.3} K ║", state.temperature);
    println!("╠══════════════════════════════════════════════╣");
    println!("║ Galaxies:     {:>30} ║", state.galaxies.len());
    println!("║ Stars:        {:>30} ║", state.stars.len());
    println!("║ Planets:      {:>30} ║", state.planets.len());
    println!("║ Life-bearing: {:>30} ║", state.planets.iter().filter(|p| p.has_life).count());
    
    if detailed {
        println!("╠══════════════════════════════════════════════╣");
        println!("║           Cosmological Parameters            ║");
        println!("╠══════════════════════════════════════════════╣");
        println!("║ H₀:           {:>26.1} km/s/Mpc ║", state.cosmological_params.h0);
        println!("║ Ωₘ:           {:>30.3} ║", state.cosmological_params.omega_m);
        println!("║ ΩΛ:           {:>30.3} ║", state.cosmological_params.omega_lambda);
        println!("║ Ωb:           {:>30.3} ║", state.cosmological_params.omega_b);
        println!("║ σ₈:           {:>30.3} ║", state.cosmological_params.sigma8);
        
        // Galaxy statistics
        let total_stellar_mass: f64 = state.galaxies.iter().map(|g| g.stellar_mass).sum();
        let avg_sfr: f64 = state.galaxies.iter().map(|g| g.star_formation_rate).sum::<f64>() 
                          / state.galaxies.len() as f64;
        
        println!("╠══════════════════════════════════════════════╣");
        println!("║            Galaxy Statistics                 ║");
        println!("╠══════════════════════════════════════════════╣");
        println!("║ Total Stellar Mass: {:>19.2e} M☉ ║", total_stellar_mass);
        println!("║ Avg Star Form Rate: {:>20.1} M☉/yr ║", avg_sfr);
        
        // Planet statistics
        let habitable_planets = state.planets.iter().filter(|p| p.habitability_score > 0.5).count();
        let total_population: u64 = state.planets.iter().map(|p| p.population).sum();
        
        println!("╠══════════════════════════════════════════════╣");
        println!("║            Planet Statistics                 ║");
        println!("╠══════════════════════════════════════════════╣");
        println!("║ Habitable Planets:  {:>24} ║", habitable_planets);
        println!("║ Total Population:   {:>24} ║", total_population);
    }
    
    println!("╚══════════════════════════════════════════════╝");
    
    Ok(())
}

pub async fn run_monitor(source: &str, refresh_secs: u64, detailed: bool) -> Result<()> {
    // Enable raw mode for terminal control
    terminal::enable_raw_mode()?;
    execute!(stdout(), Hide)?;
    
    let mut last_update = Instant::now();
    let refresh_duration = Duration::from_secs(refresh_secs);
    
    loop {
        // Clear screen and move cursor to top
        execute!(
            stdout(),
            Clear(ClearType::All),
            MoveTo(0, 0)
        )?;
        
        // Show current status
        let state = crate::load_simulation_state(source).await?;
        render_live_status(&state, detailed)?;
        
        // Check for quit key
        if event::poll(Duration::from_millis(100))? {
            if let Event::Key(key_event) = event::read()? {
                if key_event.code == KeyCode::Char('q') || key_event.code == KeyCode::Esc {
                    break;
                }
            }
        }
        
        // Wait for refresh interval
        let elapsed = last_update.elapsed();
        if elapsed < refresh_duration {
            std::thread::sleep(refresh_duration - elapsed);
        }
        last_update = Instant::now();
    }
    
    // Restore terminal
    execute!(stdout(), Show)?;
    terminal::disable_raw_mode()?;
    
    Ok(())
}

fn render_live_status(state: &SimulationState, detailed: bool) -> Result<()> {
    use std::io::Write;
    let mut stdout = stdout();
    
    // Header
    execute!(stdout, SetForegroundColor(Color::Cyan))?;
    writeln!(stdout, "╔══════════════════════════════════════════════════════════════╗")?;
    writeln!(stdout, "║              UNIVERSE SIMULATION MONITOR                     ║")?;
    writeln!(stdout, "╠══════════════════════════════════════════════════════════════╣")?;
    execute!(stdout, ResetColor)?;
    
    // Basic stats with color coding
    writeln!(stdout, "║ Tick:         {:>30} │ {:>12} ║", 
        state.tick, 
        format!("{:.1} Hz", state.ups)
    )?;
    
    // Time with gradient color based on age
    let age_color = if state.current_time < 1e9 {
        Color::Red
    } else if state.current_time < 5e9 {
        Color::Yellow
    } else {
        Color::Green
    };
    
    execute!(stdout, SetForegroundColor(age_color))?;
    writeln!(stdout, "║ Cosmic Time:  {:>28.2} Gyr │ {:>12} ║", 
        state.current_time / 1e9,
        "▓▓▓▓▓▓▓▓▓▓"
    )?;
    execute!(stdout, ResetColor)?;
    
    // Entropy and temperature
    writeln!(stdout, "║ Entropy:      {:>30.2e} │ Temperature: {:.3} K ║", 
        state.entropy, 
        state.temperature
    )?;
    
    // Object counts with visual bars
    let galaxy_bar = create_bar(state.galaxies.len(), 1000, 10);
    let star_bar = create_bar(state.stars.len(), 5000, 10);
    let planet_bar = create_bar(state.planets.len(), 100, 10);
    
    writeln!(stdout, "╠══════════════════════════════════════════════════════════════╣")?;
    writeln!(stdout, "║ Galaxies:  {:>8} {} │ Stars: {:>8} {} ║",
        state.galaxies.len(), galaxy_bar,
        state.stars.len(), star_bar
    )?;
    writeln!(stdout, "║ Planets:   {:>8} {} │ Life:  {:>8} {} ║",
        state.planets.len(), planet_bar,
        state.planets.iter().filter(|p| p.has_life).count(),
        create_bar(state.planets.iter().filter(|p| p.has_life).count(), 10, 10)
    )?;
    
    if detailed {
        writeln!(stdout, "╠══════════════════════════════════════════════════════════════╣")?;
        
        // Show top 5 most massive galaxies
        let mut galaxies = state.galaxies.clone();
        galaxies.sort_by(|a, b| b.mass.partial_cmp(&a.mass).unwrap());
        
        writeln!(stdout, "║ Top Galaxy Clusters:                                         ║")?;
        for (i, galaxy) in galaxies.iter().take(5).enumerate() {
            writeln!(stdout, "║  {}. {} - {:.2e} M☉ @ ({:>6.1}, {:>6.1}) Mpc            ║",
                i + 1,
                &galaxy.id[0..8],
                galaxy.mass,
                galaxy.position.x,
                galaxy.position.y
            )?;
        }
    }
    
    writeln!(stdout, "╠══════════════════════════════════════════════════════════════╣")?;
    writeln!(stdout, "║ Press 'q' or ESC to quit                                     ║")?;
    writeln!(stdout, "╚══════════════════════════════════════════════════════════════╝")?;
    
    stdout.flush()?;
    Ok(())
}

fn create_bar(value: usize, max_value: usize, bar_length: usize) -> String {
    let filled = (value as f64 / max_value as f64 * bar_length as f64).min(bar_length as f64) as usize;
    let empty = bar_length - filled;
    format!("{}{}", "█".repeat(filled), "░".repeat(empty))
}