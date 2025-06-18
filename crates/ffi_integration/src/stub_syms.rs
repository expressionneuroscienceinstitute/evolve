// Stub implementations of external C symbols expected by FFI wrappers when
// the corresponding native libraries are not linked. These stubs are *not*
// scientifically accurate – they only guarantee that unit tests and
// documentation builds pass on machines without the heavy dependencies.

#![allow(non_camel_case_types)]

use std::os::raw::{c_char, c_int, c_void};

// ---------------- ENDF ----------------
#[no_mangle]
pub extern "C" fn endf_version() -> c_int { 0 }

// ---------------- GADGET ----------------
#[no_mangle]
pub extern "C" fn gadget_version() -> c_int { 0 }

#[no_mangle]
pub extern "C" fn gadget_cleanup() {}

// ---------------- LAMMPS ----------------
#[no_mangle]
pub extern "C" fn lammps_version() -> c_int { 0 }

#[no_mangle]
pub extern "C" fn lammps_close(_handle: *mut c_void) {} 