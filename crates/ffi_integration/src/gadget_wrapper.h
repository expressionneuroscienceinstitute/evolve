#ifndef GADGET_WRAPPER_H
#define GADGET_WRAPPER_H

#ifdef __cplusplus
extern "C" {
#endif

// GADGET FFI interface for EVOLVE
// Custom wrapper around GADGET core functionality

// Initialization and cleanup
int gadget_version();
void gadget_init();
void gadget_cleanup();

// Cosmological parameters
void gadget_set_cosmology(double omega_matter, double omega_lambda, double hubble_param);
void gadget_set_box_size(double box_size);

// Particle management
void gadget_init_particles(int n_particles);
void gadget_set_particle_data(int index, double x, double y, double z,
                              double vx, double vy, double vz, 
                              double mass, int type);
void gadget_get_particle_data(int index, double *pos, double *vel, 
                              double *acc, double *potential);

// Tree structure
void gadget_init_tree();
void gadget_rebuild_tree();

// Time integration
void gadget_set_time_parameters(double time_current, double time_target, double dt_max);
void gadget_calculate_forces();
double gadget_get_timestep();
void gadget_update_particles(double dt);

// Energy calculations
double gadget_calculate_potential_energy();
double gadget_calculate_kinetic_energy();

// Friends-of-friends halo finder
void gadget_run_fof(double linking_length);
int gadget_get_n_halos();

// Halo data structure
typedef struct {
    int n_particles;
    double total_mass;
    double center_of_mass[3];
    double virial_radius;
    double velocity_dispersion;
} gadget_halo_data_t;

void gadget_get_halo_data(int halo_index, gadget_halo_data_t *halo_data);

// Diagnostic functions
void gadget_print_statistics();
double gadget_get_simulation_time();
int gadget_get_n_particles();

// I/O functions
int gadget_read_snapshot(const char *filename);
int gadget_write_snapshot(const char *filename);

// Performance monitoring
void gadget_start_timing(const char *timer_name);
void gadget_stop_timing(const char *timer_name);
double gadget_get_timing(const char *timer_name);

#ifdef __cplusplus
}
#endif

#endif // GADGET_WRAPPER_H 