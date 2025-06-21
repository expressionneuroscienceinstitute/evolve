#ifndef GEANT4_WRAPPER_H
#define GEANT4_WRAPPER_H

#ifdef __cplusplus
extern "C" {
#endif

// Forward declarations as opaque types for C compatibility
// These will be replaced by the actual C++ types when using the real library
typedef void G4RunManager;
typedef void G4VUserDetectorConstruction;
typedef void G4VUserPhysicsList;
typedef void G4SteppingManager;
typedef void G4ParticleGun;

// Data structures for FFI communication
typedef struct {
    int process_type;
    double energy_deposited;
    int n_secondaries;
    int* secondary_particles;
    double* secondary_energies;
    double step_length;
    double position[3];
} G4InteractionData;

typedef struct {
    int pdg_code;
    double energy;
    double momentum[3];
    double position[3];
} G4ParticleData;

// Core Geant4 functionality
int g4_is_available(void);
int g4_global_initialize(void);
void g4_global_cleanup(void);

// Run manager
G4RunManager* g4_create_run_manager(void);
void g4_delete_run_manager(G4RunManager* manager);
void g4_initialize_geant4(G4RunManager* manager);
void g4_run_beam_on(G4RunManager* manager, int n_events);

// Detector construction
G4VUserDetectorConstruction* g4_create_simple_detector(void);
void g4_delete_detector(G4VUserDetectorConstruction* detector);
void g4_set_detector_construction(G4RunManager* manager, G4VUserDetectorConstruction* detector);

// Physics list
G4VUserPhysicsList* g4_create_physics_list(const char* name);
void g4_delete_physics_list(G4VUserPhysicsList* physics_list);
void g4_set_physics_list(G4RunManager* manager, G4VUserPhysicsList* physics_list);

// Stepping manager
G4SteppingManager* g4_create_stepping_manager(void);
void g4_delete_stepping_manager(G4SteppingManager* manager);

// Particle gun
G4ParticleGun* g4_create_particle_gun(void);
void g4_delete_particle_gun(G4ParticleGun* gun);
void g4_set_particle_gun_particle(G4ParticleGun* gun, int pdg_code);
void g4_set_particle_gun_energy(G4ParticleGun* gun, double energy_mev);
void g4_set_particle_gun_position(G4ParticleGun* gun, double x, double y, double z);
void g4_set_particle_gun_direction(G4ParticleGun* gun, double dx, double dy, double dz);

// Material and geometry
void g4_set_material(const char* material);
void g4_set_step_limit(double step_cm);

// Data extraction
int g4_get_n_interactions(void);
void g4_get_interaction_data(int index, G4InteractionData* data);

// Cross-sections and physics
double g4_get_cross_section(const char* particle, const char* material, 
                           const char* process, double energy_mev);
double g4_get_stopping_power(const char* particle, const char* material, 
                            double energy_mev);

// Decay simulation
int g4_simulate_decay(int pdg_code, double energy_mev);
void g4_get_decay_product(int index, G4ParticleData* data);

#ifdef __cplusplus
}
#endif

#endif // GEANT4_WRAPPER_H 