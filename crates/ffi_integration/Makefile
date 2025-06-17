# Makefile for building Geant4 FFI wrapper dynamic library
# This creates a shared library that provides real Geant4 functionality
# instead of the stub implementations

# Library name and version
LIB_NAME = geant4_wrapper
LIB_VERSION = 1.0.0

# Platform-specific library names
UNAME_S := $(shell uname -s)
ifeq ($(UNAME_S),Darwin)
    # macOS
    SHARED_LIB = lib$(LIB_NAME).$(LIB_VERSION).dylib
    SHARED_LIB_LINK = lib$(LIB_NAME).dylib
    INSTALL_NAME_OPTION = -install_name
else
    # Linux
    SHARED_LIB = lib$(LIB_NAME).so.$(LIB_VERSION)
    SHARED_LIB_LINK = lib$(LIB_NAME).so
    INSTALL_NAME_OPTION = -soname
endif

# Source files
CPP_SOURCES = src/geant4_wrapper.cpp
HEADERS = src/geant4_wrapper.h

# Build directories
BUILD_DIR = build
OBJ_DIR = $(BUILD_DIR)/obj
LIB_DIR = $(BUILD_DIR)/lib

# Compiler settings
CXX = g++
CXXFLAGS = -std=c++17 -fPIC -O2 -Wall -Wextra

# Geant4 configuration
GEANT4_CONFIG = geant4-config
GEANT4_AVAILABLE := $(shell command -v $(GEANT4_CONFIG) 2> /dev/null)

ifdef GEANT4_AVAILABLE
    GEANT4_CFLAGS := $(shell $(GEANT4_CONFIG) --cflags)
    GEANT4_LIBS := $(shell $(GEANT4_CONFIG) --libs)
    GEANT4_LIBDIR := $(shell $(GEANT4_CONFIG) --libs | grep -o '\-L[^[:space:]]*' | sed 's/-L//')
else
    # Fallback configuration if geant4-config is not available
    ifdef GEANT4_DIR
        GEANT4_CFLAGS = -I$(GEANT4_DIR)/include/Geant4
        GEANT4_LIBDIR = $(GEANT4_DIR)/lib64:$(GEANT4_DIR)/lib
        GEANT4_LIBS = -L$(GEANT4_DIR)/lib64 -L$(GEANT4_DIR)/lib \
            -lG4run -lG4event -lG4tracking -lG4parmodels -lG4processes \
            -lG4track -lG4particles -lG4geometry -lG4materials \
            -lG4graphics_reps -lG4intercoms -lG4global
    else
        $(error "Geant4 not found. Please install Geant4 and set GEANT4_DIR or ensure geant4-config is in PATH")
    endif
endif

# Additional system libraries that Geant4 might need
SYSTEM_LIBS = -lm -lpthread -ldl

# Final compiler and linker flags
CPPFLAGS = $(GEANT4_CFLAGS)
ifeq ($(UNAME_S),Darwin)
    LDFLAGS = -shared -Wl,-install_name,@rpath/$(SHARED_LIB)
else
    LDFLAGS = -shared -Wl,-soname,$(SHARED_LIB)
endif
LDLIBS = $(GEANT4_LIBS) $(SYSTEM_LIBS)

# Object files
OBJECTS = $(CPP_SOURCES:src/%.cpp=$(OBJ_DIR)/%.o)

# Default target
all: $(LIB_DIR)/$(SHARED_LIB)

# Create build directories
$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

$(OBJ_DIR): | $(BUILD_DIR)
	mkdir -p $(OBJ_DIR)

$(LIB_DIR): | $(BUILD_DIR)
	mkdir -p $(LIB_DIR)

# Compile object files
$(OBJ_DIR)/%.o: src/%.cpp $(HEADERS) | $(OBJ_DIR)
	$(CXX) $(CXXFLAGS) $(CPPFLAGS) -c $< -o $@

# Link shared library
$(LIB_DIR)/$(SHARED_LIB): $(OBJECTS) | $(LIB_DIR)
	$(CXX) $(LDFLAGS) -o $@ $(OBJECTS) $(LDLIBS)
	cd $(LIB_DIR) && ln -sf $(SHARED_LIB) $(SHARED_LIB_LINK)

# Install the library (optional)
install: $(LIB_DIR)/$(SHARED_LIB)
	@echo "Installing $(SHARED_LIB) to /usr/local/lib"
	sudo cp $(LIB_DIR)/$(SHARED_LIB) /usr/local/lib/
	sudo ln -sf /usr/local/lib/$(SHARED_LIB) /usr/local/lib/$(SHARED_LIB_LINK)
	sudo ldconfig

# Clean build artifacts
clean:
	rm -rf $(BUILD_DIR)

# Test the library (basic smoke test)
test: $(LIB_DIR)/$(SHARED_LIB)
	@echo "Testing Geant4 wrapper library..."
	@echo '#include <dlfcn.h>' > test_lib.c
	@echo '#include <stdio.h>' >> test_lib.c
	@echo 'int main() {' >> test_lib.c
	@echo '  void* lib = dlopen("$(LIB_DIR)/$(SHARED_LIB)", RTLD_LAZY);' >> test_lib.c
	@echo '  if (!lib) { printf("Failed to load library: %s\\n", dlerror()); return 1; }' >> test_lib.c
	@echo '  int (*g4_is_available)() = dlsym(lib, "g4_is_available");' >> test_lib.c
	@echo '  if (!g4_is_available) { printf("Failed to find g4_is_available\\n"); return 1; }' >> test_lib.c
	@echo '  printf("Geant4 available: %d\\n", g4_is_available());' >> test_lib.c
	@echo '  dlclose(lib);' >> test_lib.c
	@echo '  return 0;' >> test_lib.c
	@echo '}' >> test_lib.c
	$(CC) -o test_lib test_lib.c -ldl
ifeq ($(UNAME_S),Darwin)
	DYLD_LIBRARY_PATH=$(LIB_DIR):$$DYLD_LIBRARY_PATH ./test_lib
else
	LD_LIBRARY_PATH=$(LIB_DIR):$$LD_LIBRARY_PATH ./test_lib
endif
	rm -f test_lib test_lib.c

# Show configuration
config:
	@echo "Configuration:"
	@echo "  CXX: $(CXX)"
	@echo "  CXXFLAGS: $(CXXFLAGS)"
	@echo "  CPPFLAGS: $(CPPFLAGS)"
	@echo "  LDFLAGS: $(LDFLAGS)"
	@echo "  LDLIBS: $(LDLIBS)"
	@echo "  Geant4 Config: $(GEANT4_CONFIG)"
	@echo "  Geant4 Available: $(GEANT4_AVAILABLE)"

# Debug target with additional flags
debug: CXXFLAGS += -g -DDEBUG -O0
debug: $(LIB_DIR)/$(SHARED_LIB)

# Help target
help:
	@echo "Available targets:"
	@echo "  all      - Build the shared library (default)"
	@echo "  install  - Install the library to /usr/local/lib"
	@echo "  clean    - Remove build artifacts"
	@echo "  test     - Test the built library"
	@echo "  config   - Show build configuration"
	@echo "  debug    - Build with debug symbols"
	@echo "  help     - Show this help"
	@echo ""
	@echo "Environment variables:"
	@echo "  GEANT4_DIR - Path to Geant4 installation (if geant4-config not available)"
	@echo "  CXX        - C++ compiler to use (default: g++)"

.PHONY: all install clean test config debug help 