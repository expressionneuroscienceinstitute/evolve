use std::env;
use std::path::PathBuf;

fn main() {
    println!("cargo:rerun-if-changed=wrapper.h");
    
    // Build configuration for different scientific libraries
    build_geant4_bindings();
    build_lammps_bindings();
    build_gadget_bindings();
    build_endf_parser();
    
    // Set library search paths
    configure_library_paths();
}

fn build_geant4_bindings() {
    // Check if Geant4 is installed
    if let Ok(geant4_dir) = env::var("GEANT4_DIR") {
        println!("cargo:rustc-link-search=native={}/lib64", geant4_dir);
        println!("cargo:rustc-link-search=native={}/lib", geant4_dir);
        
        // Link Geant4 libraries
        println!("cargo:rustc-link-lib=G4run");
        println!("cargo:rustc-link-lib=G4event");
        println!("cargo:rustc-link-lib=G4tracking");
        println!("cargo:rustc-link-lib=G4parmodels");
        println!("cargo:rustc-link-lib=G4processes");
        println!("cargo:rustc-link-lib=G4digits");
        println!("cargo:rustc-link-lib=G4track");
        println!("cargo:rustc-link-lib=G4particles");
        println!("cargo:rustc-link-lib=G4geometry");
        println!("cargo:rustc-link-lib=G4materials");
        println!("cargo:rustc-link-lib=G4graphics_reps");
        println!("cargo:rustc-link-lib=G4intercoms");
        println!("cargo:rustc-link-lib=G4global");
        
        // Generate bindings
        let bindings = bindgen::Builder::default()
            .header("src/ffi/geant4_wrapper.h")
            .clang_arg(format!("-I{}/include/Geant4", geant4_dir))
            .parse_callbacks(Box::new(bindgen::CargoCallbacks::new()))
            .generate()
            .expect("Unable to generate Geant4 bindings");
            
        let out_path = PathBuf::from(env::var("OUT_DIR").unwrap());
        bindings
            .write_to_file(out_path.join("geant4_bindings.rs"))
            .expect("Couldn't write Geant4 bindings!");
    }
}

fn build_lammps_bindings() {
    // Check if LAMMPS is installed
    if let Ok(lammps_dir) = env::var("LAMMPS_DIR") {
        println!("cargo:rustc-link-search=native={}/lib", lammps_dir);
        println!("cargo:rustc-link-lib=lammps");
        
        // Generate bindings for LAMMPS C library interface
        let bindings = bindgen::Builder::default()
            .header("src/ffi/lammps_wrapper.h")
            .clang_arg(format!("-I{}/src", lammps_dir))
            .parse_callbacks(Box::new(bindgen::CargoCallbacks::new()))
            .generate()
            .expect("Unable to generate LAMMPS bindings");
            
        let out_path = PathBuf::from(env::var("OUT_DIR").unwrap());
        bindings
            .write_to_file(out_path.join("lammps_bindings.rs"))
            .expect("Couldn't write LAMMPS bindings!");
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
            .header("src/ffi/gadget_wrapper.h")
            .clang_arg(format!("-I{}", gadget_src))
            .parse_callbacks(Box::new(bindgen::CargoCallbacks::new()))
            .generate()
            .expect("Unable to generate GADGET bindings");
            
        let out_path = PathBuf::from(env::var("OUT_DIR").unwrap());
        bindings
            .write_to_file(out_path.join("gadget_bindings.rs"))
            .expect("Couldn't write GADGET bindings!");
    }
}

fn build_endf_parser() {
    // Build ENDF parsing library (there are C libraries for this)
    if let Ok(endf_lib) = env::var("ENDF_LIB_DIR") {
        println!("cargo:rustc-link-search=native={}", endf_lib);
        println!("cargo:rustc-link-lib=endf");
        
        let bindings = bindgen::Builder::default()
            .header("src/ffi/endf_wrapper.h")
            .parse_callbacks(Box::new(bindgen::CargoCallbacks::new()))
            .generate()
            .expect("Unable to generate ENDF bindings");
            
        let out_path = PathBuf::from(env::var("OUT_DIR").unwrap());
        bindings
            .write_to_file(out_path.join("endf_bindings.rs"))
            .expect("Couldn't write ENDF bindings!");
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