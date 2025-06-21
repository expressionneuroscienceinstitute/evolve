#ifndef LAMMPS_WRAPPER_H
#define LAMMPS_WRAPPER_H

#ifdef __cplusplus
extern "C" {
#endif

// LAMMPS C library interface
// Based on the official LAMMPS C library interface

typedef void* lammps_t;

// Core LAMMPS functions
lammps_t lammps_open_no_mpi(int argc, char **argv);
void lammps_close(lammps_t handle);
void lammps_command(lammps_t handle, const char *cmd);

// Data extraction functions
void* lammps_extract_atom(lammps_t handle, const char *name);
void* lammps_extract_global(lammps_t handle, const char *name);

// Version information
int lammps_version();

// Utility functions for EVOLVE integration
void lammps_set_variable(lammps_t handle, const char *name, const char *value);
int lammps_get_natoms(lammps_t handle);
double lammps_get_thermo(lammps_t handle, const char *keyword);

// Force field management
void lammps_set_pair_style(lammps_t handle, const char *style);
void lammps_set_pair_coeff(lammps_t handle, const char *coeff);

// Simulation control
void lammps_run(lammps_t handle, int nsteps);
void lammps_reset_timestep(lammps_t handle, int step);

// Box manipulation
void lammps_change_box(lammps_t handle, double xlo, double xhi, 
                       double ylo, double yhi, double zlo, double zhi);

// Atom manipulation
void lammps_create_atoms(lammps_t handle, int n, int *id, int *type,
                         double *x, double *v, double *image, int bexpand);

#ifdef __cplusplus
}
#endif

#endif // LAMMPS_WRAPPER_H 