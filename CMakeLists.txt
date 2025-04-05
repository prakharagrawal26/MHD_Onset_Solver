cmake_minimum_required(VERSION 3.12)
project(OnsetSolverCollocation LANGUAGES CXX) # Changed project name slightly

# --- Compiler Settings ---
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# --- Find Dependencies ---
find_package(Eigen3 3.3 REQUIRED CONFIG)
find_package(OpenMP REQUIRED)

# --- Project Structure ---
set(SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/src)
set(EXTERNAL_DIR ${CMAKE_CURRENT_SOURCE_DIR}/external)

# --- Include Directories ---
include_directories(
    ${SOURCE_DIR}
    ${EXTERNAL_DIR}/spectra/include
    ${EIGEN3_INCLUDE_DIR}
)

# --- Executable ---
# Updated source file names
add_executable(onset_solver
    ${SOURCE_DIR}/main.cpp
    ${SOURCE_DIR}/params.cpp
    ${SOURCE_DIR}/utils.cpp
    ${SOURCE_DIR}/cheb.cpp
    ${SOURCE_DIR}/bc.cpp
    ${SOURCE_DIR}/coeffs.cpp
    ${SOURCE_DIR}/matrix_builder.cpp
    ${SOURCE_DIR}/eigen_solver.cpp
)

# --- Linking ---
target_link_libraries(onset_solver PRIVATE
    Eigen3::Eigen
    OpenMP::OpenMP_CXX
)

# --- Output ---
message(STATUS "Project Source Directory: ${CMAKE_CURRENT_SOURCE_DIR}")
message(STATUS "Eigen Include Directory: ${EIGEN3_INCLUDE_DIR}")
message(STATUS "Spectra Include Directory: ${EXTERNAL_DIR}/spectra/include")