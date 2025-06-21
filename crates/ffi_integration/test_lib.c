#include <dlfcn.h>
#include <stdio.h>
int main() {
  void* lib = dlopen("build/lib/libgeant4_wrapper.1.0.0.dylib", RTLD_LAZY);
  if (!lib) { printf("Failed to load library: %s\n", dlerror()); return 1; }
  int (*g4_is_available)() = dlsym(lib, "g4_is_available");
  if (!g4_is_available) { printf("Failed to find g4_is_available\n"); return 1; }
  printf("Geant4 available: %d\n", g4_is_available());
  dlclose(lib);
  return 0;
}
