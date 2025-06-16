#ifndef ENDF_WRAPPER_H
#define ENDF_WRAPPER_H

#ifdef __cplusplus
extern "C" {
#endif

// ENDF nuclear data library interface for EVOLVE
// Interface to ENDF/B-VIII.0 evaluated nuclear data

// Library management
int endf_version();
void endf_init();
void endf_cleanup();
int endf_load_library(const char *library_path);

// Isotope management
int endf_get_n_isotopes();
void endf_get_isotope_list(unsigned int *isotope_list);
int endf_isotope_exists(unsigned int isotope);

// Temperature control
void endf_set_temperature(double temperature_k);
double endf_get_temperature();

// Cross-section data
double endf_get_cross_section(int isotope, int mt, double energy_ev);
double endf_get_cross_section_temperature(int isotope, int mt, double energy_ev, double temp_k);

// Q-values and thresholds
double endf_get_q_value(int isotope, int mt);
double endf_get_threshold(int isotope, int mt);

// Decay data
double endf_get_half_life(int isotope);
double endf_get_decay_constant(int isotope);
int endf_get_decay_modes(int isotope, int *modes, double *branching_ratios);

// Fission data
int endf_get_n_fission_products(int parent_isotope);
void endf_get_fission_yields(int parent_isotope, double neutron_energy_ev,
                             unsigned int *products, double *yields);
double endf_get_nu_fission(int isotope, double energy_ev);
double endf_get_delayed_neutron_fraction(int isotope, double energy_ev);

// Resonance parameters
int endf_get_n_resonances(int isotope, double energy_min, double energy_max);
void endf_get_resonance_data(int isotope, int resonance_index,
                             double *energy, double *gamma_n, 
                             double *gamma_gamma, double *gamma_f);

// Thermal data (at 2200 m/s)
double endf_get_thermal_absorption(int isotope);
double endf_get_thermal_fission(int isotope);
double endf_get_thermal_scattering(int isotope);

// Doppler broadening
double endf_doppler_broaden_cross_section(int isotope, int mt, 
                                          double energy_ev, double temp_k);

// Angular distributions
int endf_get_n_angles(int isotope, int mt, double energy_ev);
void endf_get_angular_distribution(int isotope, int mt, double energy_ev,
                                   double *angles, double *probabilities);

// Energy distributions
int endf_get_n_outgoing_energies(int isotope, int mt, double energy_ev);
void endf_get_energy_distribution(int isotope, int mt, double energy_ev,
                                  double *energies, double *probabilities);

// Multi-group data
void endf_generate_multigroup_data(int isotope, int n_groups,
                                   double *group_bounds, double *group_xs);

// Uncertainty data
double endf_get_cross_section_uncertainty(int isotope, int mt, double energy_ev);
void endf_get_covariance_matrix(int isotope, int mt, int n_points,
                                double *energies, double *covariance);

// File I/O utilities
int endf_read_tape(const char *filename);
int endf_write_processed_data(const char *filename, int isotope);

// Interpolation and utilities
double endf_interpolate_cross_section(int isotope, int mt, double energy_ev);
void endf_set_interpolation_scheme(int scheme); // 1=histogram, 2=linear, etc.

// Error handling
const char* endf_get_last_error();
void endf_clear_error();

// Constants for common MT numbers
#define ENDF_MT_ELASTIC         2
#define ENDF_MT_INELASTIC       4
#define ENDF_MT_FISSION         18
#define ENDF_MT_ABSORPTION      27
#define ENDF_MT_TOTAL           1
#define ENDF_MT_PROTON_CAPTURE  103
#define ENDF_MT_ALPHA_CAPTURE   107

#ifdef __cplusplus
}
#endif

#endif // ENDF_WRAPPER_H 