mod gpu_acceleration_stress_test;

use std::env;
use std::process;

fn main() {
    let args: Vec<String> = env::args().collect();
    
    if args.len() != 2 {
        println!("Usage: cargo run --bin demo <demo_number>");
        println!("Available demos:");
        println!("  1 - Big Bang Simulation");
        println!("  2 - Weak Interactions");
        println!("  3 - Cosmological N-Body");
        println!("  4 - Cosmological SPH");
        println!("  5 - Tree-PM Gravity");
        println!("  6 - Statistical Analysis");
        println!("  7 - Stellar Nursery");
        println!("  8 - GPU Acceleration Stress Test");
        process::exit(1);
    }
    
    let demo_number = args[1].parse::<u32>().unwrap_or(0);
    
    match demo_number {
        1 => {
            println!("Running Big Bang Simulation...");
            // TODO: Implement big bang demo
        }
        2 => {
            println!("Running Weak Interactions...");
            // TODO: Implement weak interactions demo
        }
        3 => {
            println!("Running Cosmological N-Body...");
            // TODO: Implement cosmological n-body demo
        }
        4 => {
            println!("Running Cosmological SPH...");
            // TODO: Implement cosmological SPH demo
        }
        5 => {
            println!("Running Tree-PM Gravity...");
            // TODO: Implement tree-PM gravity demo
        }
        6 => {
            println!("Running Statistical Analysis...");
            // TODO: Implement statistical analysis demo
        }
        7 => {
            println!("Running Stellar Nursery...");
            // TODO: Implement stellar nursery demo
        }
        8 => {
            println!("Running GPU Acceleration Stress Test...");
            match gpu_acceleration_stress_test::run_gpu_acceleration_demo() {
                Ok(_) => println!("GPU acceleration stress test completed successfully!"),
                Err(e) => {
                    eprintln!("Error running GPU acceleration stress test: {}", e);
                    process::exit(1);
                }
            }
        }
        _ => {
            eprintln!("Invalid demo number: {}", demo_number);
            process::exit(1);
        }
    }
} 