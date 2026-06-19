# ==============================================================
# CMakeLists.txt — CosmicMuonSim
#
# Build:
#   mkdir build && cd build
#   cmake -DGeant4_DIR=/path/to/geant4-install/lib/cmake/Geant4 ..
#   make -j$(nproc)
#
# Run:
#   ./CosmicMuonSim              (interactive with vis)
#   ./CosmicMuonSim run.mac      (batch mode)
# ==============================================================

cmake_minimum_required(VERSION 3.16)
project(CosmicMuonSim)

# --- Find Geant4 (with or without visualisation) ---
find_package(Geant4 REQUIRED)
# If you want OpenGL vis, use:
#   find_package(Geant4 REQUIRED ui_all vis_all)

include(${Geant4_USE_FILE})

# --- Sources ---
file(GLOB sources ${PROJECT_SOURCE_DIR}/src/*.cc)
file(GLOB headers ${PROJECT_SOURCE_DIR}/include/*.hh)

# --- Executable ---
add_executable(CosmicMuonSim main.cc ${sources} ${headers})
target_include_directories(CosmicMuonSim PRIVATE ${PROJECT_SOURCE_DIR}/include)
target_link_libraries(CosmicMuonSim ${Geant4_LIBRARIES})

# --- Copy macro files into build directory ---
file(GLOB MACROS ${PROJECT_SOURCE_DIR}/*.mac)
foreach(_macro ${MACROS})
    get_filename_component(_name ${_macro} NAME)
    configure_file(${_macro} ${PROJECT_BINARY_DIR}/${_name} COPYONLY)
endforeach()
