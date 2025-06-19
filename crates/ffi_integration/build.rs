use std::env;
use std::path::PathBuf;

fn main() {
    // Re-run when any header wrapper changes
    for header in [
        "src/geant4_wrapper.h",
        "src/lammps_wrapper.h",
        "src/gadget_wrapper.h",
        "src/endf_wrapper.h",
    ] {
        println!("cargo:rerun-if-changed={}", header);
    }
    
    // Build configuration for different scientific libraries only if the
    // corresponding Cargo feature is enabled. This prevents unconditional
    // linking against libraries that may be absent on the system.
    let mut missing_required = Vec::new();
    if env::var("CARGO_FEATURE_GEANT4").is_ok() {
        let wrapper_lib_so = "build/lib/libgeant4_wrapper.so";
        let wrapper_lib_dylib = "build/lib/libgeant4_wrapper.dylib";
        let wrapper_lib_versioned_dylib = "build/lib/libgeant4_wrapper.1.0.0.dylib";
        let found = std::path::Path::new(wrapper_lib_so).exists() ||
                    std::path::Path::new(wrapper_lib_dylib).exists() ||
                    std::path::Path::new(wrapper_lib_versioned_dylib).exists();
        if !found {
            missing_required.push("Geant4");
        }
        build_geant4_bindings();
    }
    if env::var("CARGO_FEATURE_LAMMPS").is_ok() {
        if env::var("LAMMPS_DIR").is_err() {
            missing_required.push("LAMMPS");
        }
        build_lammps_bindings();
    }
    if env::var("CARGO_FEATURE_GADGET").is_ok() {
        if env::var("GADGET_SRC").is_err() {
            missing_required.push("GADGET");
        }
        build_gadget_bindings();
    }
    if env::var("CARGO_FEATURE_ENDF").is_ok() {
        if env::var("ENDF_LIB_DIR").is_err() {
            missing_required.push("ENDF");
        }
        build_endf_parser();
    }
    
    // Set library search paths
    configure_library_paths();
    if !missing_required.is_empty() {
        println!("cargo:warning=Missing native libraries for optional features – falling back to stub implementations: {}", missing_required.join(", "));
    }
}

fn build_geant4_bindings() {
    // Check if the real Geant4 wrapper library is available
    // Try both Linux (.so) and macOS (.dylib) extensions
    let wrapper_lib_so = "build/lib/libgeant4_wrapper.so";
    let wrapper_lib_dylib = "build/lib/libgeant4_wrapper.dylib";
    let wrapper_lib_versioned_dylib = "build/lib/libgeant4_wrapper.1.0.0.dylib";
    
    let use_real_library = std::path::Path::new(wrapper_lib_so).exists() || 
                          std::path::Path::new(wrapper_lib_dylib).exists() ||
                          std::path::Path::new(wrapper_lib_versioned_dylib).exists();
    
    if use_real_library {
        // Use the real Geant4 wrapper library
        let current_dir = std::env::current_dir().unwrap();
        let lib_path = current_dir.join("build/lib");
        println!("cargo:rustc-link-search=native={}", lib_path.display());
        println!("cargo:rustc-link-lib=geant4_wrapper");
        
        // Link appropriate C++ standard library for the platform
        if cfg!(target_os = "macos") {
            println!("cargo:rustc-link-lib=c++");  // macOS uses libc++
            println!("cargo:rustc-link-arg=-Wl,-rpath,{}", lib_path.display());
        } else {
            println!("cargo:rustc-link-lib=stdc++");  // Linux uses libstdc++
        }
    
    // If we have geant4-config available, use it to get library paths
    if let Ok(output) = std::process::Command::new("geant4-config")
        .arg("--libs")
        .output() {
        if output.status.success() {
            let lib_line = String::from_utf8_lossy(&output.stdout);
            for token in lib_line.split_whitespace() {
                if let Some(stripped) = token.strip_prefix("-l") {
                    println!("cargo:rustc-link-lib={}", stripped);
                } else if let Some(stripped) = token.strip_prefix("-L") {
                    println!("cargo:rustc-link-search=native={}", stripped);
                }
            }
        }
    }
    
    println!("cargo:warning=Using real Geant4 wrapper library");
} else {
    // Fall back to C stubs if the real library isn't available
    cc::Build::new()
        .file("src/geant4_stubs.c")
        .flag("-std=c11")
        .compile("geant4_stub");
    println!("cargo:warning=Using Geant4 stubs - build the real library with 'make' in the ffi_integration directory");
}

// Always generate basic bindings from the header
// This works whether using real library or stubs since the header defines the same interface
let bindings = bindgen::Builder::default()
    .header("src/geant4_wrapper.h")
    .parse_callbacks(Box::new(bindgen::CargoCallbacks::new()))
    .generate()
    .expect("Unable to generate Geant4 bindings");
    
let out_path = PathBuf::from(env::var("OUT_DIR").unwrap());
bindings
    .write_to_file(out_path.join("geant4_bindings.rs"))
    .expect("Couldn't write Geant4 bindings!");
}

fn build_lammps_bindings() {
    // Check if LAMMPS is installed
    if let Ok(lammps_dir) = env::var("LAMMPS_DIR") {
        println!("cargo:rustc-link-search=native={}/lib", lammps_dir);
        println!("cargo:rustc-link-lib=lammps");
        
        // Generate bindings for LAMMPS C library interface
        let bindings = bindgen::Builder::default()
            .header("src/lammps_wrapper.h")
            .clang_arg(format!("-I{}/src", lammps_dir))
            .parse_callbacks(Box::new(bindgen::CargoCallbacks::new()))
            .generate()
            .expect("Unable to generate LAMMPS bindings");
            
        let out_path = PathBuf::from(env::var("OUT_DIR").unwrap());
        bindings
            .write_to_file(out_path.join("lammps_bindings.rs"))
            .expect("Couldn't write LAMMPS bindings!");
    } else {
        // No local LAMMPS installation; emit an empty stub binding file so that the
        // `include!` in `lammps.rs` still compiles when the feature is enabled.
        let out_path = PathBuf::from(env::var("OUT_DIR").unwrap());
        let stub_path = out_path.join("lammps_bindings.rs");
        std::fs::write(&stub_path, "// Stub LAMMPS bindings generated because LAMMPS_DIR env var was not set.\n")
            .expect("Couldn't write stub LAMMPS bindings!");
        println!("cargo:warning=Using LAMMPS stubs - set the LAMMPS_DIR environment variable to build against the real library");
    }
}

fn build_gadget_bindings() {
    // GADGET is typically built as standalone executable, but we can create
    // a library version by compiling key components
    if let Ok(gadget_src) = env::var("GADGET_SRC") {
        cc::Build::new()
            .files(&[
                format!("{}/allvars.c", gadget_src),
                format!("{}/gravtree.c", gadget_src),
                format!("{}/forcetree.c", gadget_src),
                format!("{}/timestep.c", gadget_src),
                format!("{}/begrun.c", gadget_src),
                format!("{}/run.c", gadget_src),
                format!("{}/predict.c", gadget_src),
                format!("{}/drift.c", gadget_src),
                format!("{}/domain.c", gadget_src),
                format!("{}/io.c", gadget_src),
            ])
            .include(&gadget_src)
            .flag("-DGADGET_FFI") // Custom flag for FFI build
            .flag("-fPIC")
            .compile("gadget");
            
        // Generate bindings
        let bindings = bindgen::Builder::default()
            .header("src/gadget_wrapper.h")
            .clang_arg(format!("-I{}", gadget_src))
            .parse_callbacks(Box::new(bindgen::CargoCallbacks::new()))
            .generate()
            .expect("Unable to generate GADGET bindings");
            
        let out_path = PathBuf::from(env::var("OUT_DIR").unwrap());
        bindings
            .write_to_file(out_path.join("gadget_bindings.rs"))
            .expect("Couldn't write GADGET bindings!");
    } else {
        // No GADGET source provided; emit stub bindings so compile succeeds.
        let out_path = PathBuf::from(env::var("OUT_DIR").unwrap());
        let stub_path = out_path.join("gadget_bindings.rs");
        std::fs::write(&stub_path, "// Stub GADGET bindings generated because GADGET_SRC env var was not set.\n")
            .expect("Couldn't write stub GADGET bindings!");
        println!("cargo:warning=Using GADGET stubs - set the GADGET_SRC environment variable to build against the real library");
    }
}

fn build_endf_parser() {
    // Build ENDF parsing library (there are C libraries for this)
    if let Ok(endf_lib) = env::var("ENDF_LIB_DIR") {
        println!("cargo:rustc-link-search=native={}", endf_lib);
        println!("cargo:rustc-link-lib=endf");
        
        let bindings = bindgen::Builder::default()
            .header("src/endf_wrapper.h")
            .parse_callbacks(Box::new(bindgen::CargoCallbacks::new()))
            .generate()
            .expect("Unable to generate ENDF bindings");
            
        let out_path = PathBuf::from(env::var("OUT_DIR").unwrap());
        bindings
            .write_to_file(out_path.join("endf_bindings.rs"))
            .expect("Couldn't write ENDF bindings!");
    } else {
        // Emit stub bindings to allow compilation without the native ENDF library.
        let out_path = PathBuf::from(env::var("OUT_DIR").unwrap());
        let stub_path = out_path.join("endf_bindings.rs");
        std::fs::write(&stub_path, "// Stub ENDF bindings generated because ENDF_LIB_DIR env var was not set.\n\nextern \"C\" {\n    pub fn endf_version() -> ::std::os::raw::c_int;\n    pub fn endf_init();\n    pub fn endf_cleanup();\n    pub fn endf_load_library(path: *const ::std::os::raw::c_char) -> ::std::os::raw::c_int;\n    pub fn endf_get_n_isotopes() -> ::std::os::raw::c_int;\n    pub fn endf_get_isotope_list(list: *mut u32);\n    pub fn endf_set_temperature(temp: ::std::os::raw::c_double);\n    pub fn endf_get_cross_section(target_za: ::std::os::raw::c_int, reaction_mt: ::std::os::raw::c_int, energy_ev: ::std::os::raw::c_double) -> ::std::os::raw::c_double;\n    pub fn endf_get_q_value(target_za: ::std::os::raw::c_int, reaction_mt: ::std::os::raw::c_int) -> ::std::os::raw::c_double;\n    pub fn endf_get_half_life(isotope_za: ::std::os::raw::c_int) -> ::std::os::raw::c_double;\n    pub fn endf_get_n_fission_products(fissile_za: ::std::os::raw::c_int) -> ::std::os::raw::c_int;\n    pub fn endf_get_fission_yields(fissile_za: ::std::os::raw::c_int, neutron_energy_ev: ::std::os::raw::c_double, product_za: *mut u32, yields: *mut ::std::os::raw::c_double);\n    pub fn endf_get_n_resonances(isotope_za: ::std::os::raw::c_int, energy_min_ev: ::std::os::raw::c_double, energy_max_ev: ::std::os::raw::c_double) -> ::std::os::raw::c_int;\n    pub fn endf_get_resonance_data(isotope_za: ::std::os::raw::c_int, index: ::std::os::raw::c_int, energy: *mut ::std::os::raw::c_double, gamma_n: *mut ::std::os::raw::c_double, gamma_gamma: *mut ::std::os::raw::c_double, gamma_f: *mut ::std::os::raw::c_double);\n}\n")
            .expect("Couldn't write stub ENDF bindings!");
        println!("cargo:warning=Using ENDF stubs - set the ENDF_LIB_DIR environment variable to build against the real library");
    }
}

fn configure_library_paths() {
    // Add common library paths
    if let Ok(lib_path) = env::var("LD_LIBRARY_PATH") {
        for path in lib_path.split(':') {
            println!("cargo:rustc-link-search=native={}", path);
        }
    }
    
    // Platform-specific library paths
    match env::consts::OS {
        "linux" => {
            println!("cargo:rustc-link-search=native=/usr/lib64");
            println!("cargo:rustc-link-search=native=/usr/local/lib64");
            println!("cargo:rustc-link-search=native=/opt/local/lib");
        },
        "macos" => {
            println!("cargo:rustc-link-search=native=/usr/local/lib");
            println!("cargo:rustc-link-search=native=/opt/homebrew/lib");
            println!("cargo:rustc-link-search=native=/opt/local/lib");
        },
        _ => {}
    }
} 