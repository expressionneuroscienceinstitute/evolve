// Minimal stub implementations of the C wrapper API expected by the
// Rust <-> Geant-4 FFI layer.  These do nothing but satisfy the linker so
// that developers can build the project without a full Geant-4 toolchain.
// When you have a proper C++ wrapper library compiled against Geant-4 you
// should remove this file from the build.

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

// Opaque pointer typedefs (match the bindgen layout)
typedef void G4RunManager;
typedef void G4VUserDetectorConstruction;
typedef void G4VUserPhysicsList;
typedef void G4SteppingManager;
typedef void G4ParticleGun;

typedef struct {
    int32_t process_type;
    double  energy_deposited;
    int32_t n_secondaries;
    int32_t *secondary_particles;
    double  *secondary_energies;
    double  step_length;
    double  position[3];
} G4InteractionData;

typedef struct {
    int32_t pdg_code;
    double  energy;
    double  momentum[3];
    double  position[3];
} G4ParticleData;

// Helper macro – generate no-op functions returning the appropriate default
#define STUB_FN(ret, name, ...) ret name(__VA_ARGS__) { (void)0; return (ret)0; }
#define STUB_PROC(name, ...)    void name(__VA_ARGS__) { (void)0; }

// Availability & lifecycle
STUB_FN(int32_t, g4_is_available, void)
STUB_FN(int32_t, g4_global_initialize, void)
STUB_PROC(g4_global_cleanup, void)

// Construction / deletion
STUB_FN(G4RunManager*, g4_create_run_manager, void)
STUB_PROC(g4_delete_run_manager, G4RunManager* m)
STUB_FN(G4VUserDetectorConstruction*, g4_create_simple_detector, void)
STUB_PROC(g4_delete_detector, G4VUserDetectorConstruction* d)
STUB_FN(G4VUserPhysicsList*, g4_create_physics_list, const char* name)
STUB_PROC(g4_delete_physics_list, G4VUserPhysicsList* p)
STUB_FN(G4SteppingManager*, g4_create_stepping_manager, void)
STUB_PROC(g4_delete_stepping_manager, G4SteppingManager* s)
STUB_FN(G4ParticleGun*, g4_create_particle_gun, void)
STUB_PROC(g4_delete_particle_gun, G4ParticleGun* g)

// Hooking things together
STUB_PROC(g4_set_detector_construction, G4RunManager* rm, G4VUserDetectorConstruction* dc)
STUB_PROC(g4_set_physics_list,        G4RunManager* rm, G4VUserPhysicsList* pl)
STUB_PROC(g4_initialize_geant4,       G4RunManager* rm)

// Particle-gun setters
STUB_PROC(g4_set_particle_gun_particle,  G4ParticleGun* g, int32_t pdg)
STUB_PROC(g4_set_particle_gun_energy,    G4ParticleGun* g, double mev)
STUB_PROC(g4_set_particle_gun_position,  G4ParticleGun* g, double x, double y, double z)
STUB_PROC(g4_set_particle_gun_direction, G4ParticleGun* g, double dx, double dy, double dz)

// Material & stepping
STUB_PROC(g4_set_material, const char* name)
STUB_PROC(g4_set_step_limit, double cm)

// Run beam
STUB_PROC(g4_run_beam_on, G4RunManager* rm, int32_t n_events)

// Result extraction
STUB_FN(int32_t, g4_get_n_interactions, void)
STUB_PROC(g4_get_interaction_data, int32_t idx, G4InteractionData* out)

STUB_FN(double, g4_get_cross_section, const char* particle, const char* material, const char* process, double energy_mev)
STUB_FN(double, g4_get_stopping_power, const char* particle, const char* material, double energy_mev)

// Decay simulation
STUB_FN(int32_t, g4_simulate_decay, int32_t pdg, double energy_mev)
STUB_PROC(g4_get_decay_product, int32_t idx, G4ParticleData* out)

#ifdef __cplusplus
}
#endif 