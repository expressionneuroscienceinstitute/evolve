//! # Build Script for Physics Engine
//!
//! This script is responsible for preparing data and other resources needed
//! by the physics engine at compile time. It handles the ingestion of
//! nuclear data from external files, compiling it directly into the binary
//! when the `data-ingestion` feature is enabled.

// The `build.rs` script needs access to the same modules as the main crate
// to deserialize the data. We include the `data_ingestion` module here.
#[path = "src/data_ingestion.rs"]
mod data_ingestion;
#[path = "src/nuclear_physics.rs"]
mod nuclear_physics;

use std::env;
use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::Path;

fn main() {
    // The primary task of this build script is to ingest NuDat data
    // when the `data-ingestion` feature is enabled.
    if env::var("CARGO_FEATURE_DATA_INGESTION").is_ok() {
        ingest_nuclear_data();
    }
}

fn ingest_nuclear_data() {
    // Tell Cargo to rerun this script if the data file changes.
    let data_path = Path::new("../data/nudat3.json");
    println!("cargo:rerun-if-changed={}", data_path.display());

    let out_dir = env::var("OUT_DIR").unwrap();
    let dest_path = Path::new(&out_dir).join("nuclear_data.rs");
    let mut file = BufWriter::new(File::create(&dest_path).unwrap());

    // Load and parse the NuDat data.
    let nuclides = match data_ingestion::load_nudat_data(data_path.to_str().unwrap()) {
        Ok(data) => data,
        Err(e) => {
            // If the file is just the placeholder, don't panic. Generate an empty map.
            if e.to_string().contains("placeholder") {
                println!("cargo:warning=NuDat data file is a placeholder. Generating empty nuclear database.");
                write_empty_map(&mut file);
                return;
            } else {
                panic!("Failed to load or parse NuDat data: {}", e);
            }
        }
    };

    // Begin writing the generated Rust code.
    writeln!(
        &mut file,
        "use phf::phf_map;\n\
         use std::sync::LazyLock;\n\
         use crate::nuclear_physics::{{NuclearDecayData, DecayMode}};\n\n\
         pub static NUCLEAR_DECAY_DATA: LazyLock<phf::Map<(u32, u32), NuclearDecayData>> = LazyLock::new(|| {{ phf_map! {{",
    )
    .unwrap();

    // Process each nuclide and write it to the map.
    for nuclide in nuclides {
        if let Some(((z, a), decay_data)) = data_ingestion::convert_to_decay_data(&nuclide) {
            writeln!(
                &mut file,
                "    ({}u32, {}u32) => NuclearDecayData {{ half_life_seconds: {:?}, primary_mode: {:?}, decay_energy: {:?}, branching_ratio: {:?} }},",
                z, a, decay_data.half_life_seconds, decay_data.primary_mode, decay_data.decay_energy, decay_data.branching_ratio
            ).unwrap();
        }
    }

    // Close the map and the LazyLock.
    writeln!(&mut file, "}} }});").unwrap();
}

fn write_empty_map(file: &mut BufWriter<File>) {
    writeln!(
        file,
        "use phf::phf_map;\n\
         use std::sync::LazyLock;\n\
         use crate::nuclear_physics::{{NuclearDecayData, DecayMode}};\n\n\
         pub static NUCLEAR_DECAY_DATA: LazyLock<phf::Map<(u32, u32), NuclearDecayData>> = LazyLock::new(|| {{ phf_map! {{}} }});",
    )
    .unwrap();
}