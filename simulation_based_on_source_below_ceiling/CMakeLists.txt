cmake_minimum_required(VERSION 3.16)
project(MuonLifetimeSim)

find_package(Geant4 REQUIRED)
include(${Geant4_USE_FILE})

# ── Headers ────────────────────────────────────────────────────────────────
include_directories(${PROJECT_SOURCE_DIR}/include)

# ── Sources ────────────────────────────────────────────────────────────────
file(GLOB SOURCES ${PROJECT_SOURCE_DIR}/src/*.cc)

# ── Executable ─────────────────────────────────────────────────────────────
# Replace "main.cc" with the name of your main file.
add_executable(muon_sim main.cc ${SOURCES})
target_link_libraries(muon_sim ${Geant4_LIBRARIES})
