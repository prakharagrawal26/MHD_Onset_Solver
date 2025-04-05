# C++ MHD Convection Onset Solver using Chebyshev Collocation method

This program implements a C++ solver to determine the critical parameters (Rayleigh number `Ra`, wavenumber `kx`) for the onset of convection in a magnetohydrodynamic (MHD) system. It uses a Chebyshev collocation method in the Y and Z dimensions and solves the resulting large, sparse generalized eigenvalue problem.

## Core Features

*   Discretization using Chebyshev collocation method.
*   Solves the linearized MHD equations (Can also include the effect of mean flows).
*   Determines critical Rayleigh number via binary search over `Ra` for various wavenumbers `kx`.
*   Solves the generalized eigenvalue problem `Ax = lambda Bx` using the Spectra library with a shift-and-invert strategy.
*   Utilizes the Eigen library for sparse matrix operations.
*   Supports parallel execution over wavenumbers using OpenMP.
*   Parameters can be configured using a parameter file in the directory containing the executable file (`params.ini`).

## Requirements

*   C++17 compliant compiler (e.g., GCC >= 7, Clang >= 5)
*   CMake (>= 3.12 recommended)
*   Eigen3 library (>= 3.3 recommended) - Header-only or installed via package manager (sudo apt-get install libeigen3-dev on Debian/Ubuntu).
*   Spectra library (v1.0.0 or later recommended) - Header-only.
*   OpenMP library (usually included with the compiler or installed separately, e.g., `libomp-dev` on Debian/Ubuntu).

## Directory Structure

<pre lang="markdown"> ```text . ├── CMakeLists.txt # Build script ├── params.ini # Parameter input file ├── external/ # Dependencies (Requires manual setup) │ └── spectra/ # <--- Place Spectra headers here │ └── include/ │ └── Spectra/ # (Core Spectra headers) ├── src/ # Source code │ ├── main.cpp # Main driver │ ├── params.h # Parameter definitions │ ├── params.cpp # Parameter loading from file │ ├── utils.h # Utility function declarations │ ├── utils.cpp # Utility function implementations │ ├── cheb.h # Chebyshev function declaration │ ├── cheb.cpp # Chebyshev function implementation │ ├── bc.h # Boundary condition declarations │ ├── bc.cpp # Boundary condition implementation │ ├── coeffs.h # Variable coefficient declarations │ ├── coeffs.cpp # Variable coefficient implementation │ ├── matrix_builder.h # GEP matrix assembly declaration │ ├── matrix_builder.cpp # GEP matrix assembly implementation │ ├── eigen_solver.h # Eigensolver wrapper declaration │ └── eigen_solver.cpp # Eigensolver wrapper implementation ├── build/ # Build directory (created by user) ├── output/ # Output directory (created by program) └── README.md # This file ``` </pre>

## Setup & Dependencies

1.  **Compiler & CMake:** Install a suitable C++ compiler and CMake (see previous instructions for your OS).
2.  **Eigen:** Install via your system's package manager (e.g., `sudo apt install libeigen3-dev`) or download from the Eigen website and ensure CMake can find it.
3.  **Spectra:**
    *   Download Spectra (e.g., v1.1.0) from its GitHub repository releases page.
    *   Extract the archive.
    *   Copy the `include/` directory from the extracted Spectra files *into* the `external/spectra/` directory within this project. The final path should look like `external/spectra/include/Spectra/`.

## Build Instructions

1.  **Create Build Directory:**
    ```bash
    mkdir build
    cd build
    ```
2.  **Configure with CMake:**
    ```bash
    cmake ..
    ```
    *(Check the output to ensure Eigen and OpenMP were found correctly.)*
3.  **Compile:**
    ```bash
    make -jN
    ```
    *(Replace `N` with the number of cores you want to use for compilation, e.g., `make -j8`)*

## Running the Solver

1.  **Prepare Input File:** Copy the `params.ini` file from the src directory into the `build` directory. Edit `params.ini` to set your desired parameters. Parallelism can be controlled using `outer_threads` and `inner_threads` variables.
2.  **Execute:** From within the `build` directory, run the compiled executable:
    ```bash
    ./onset_solver
    ```
    Or, specify a different parameter file:
    ```bash
    ./onset_solver ../my_custom_params.ini
    ```
3.  **Parallel Execution:** The number of threads used for the outer loop (over `kx`) is controlled by the `outer_threads` parameter read from `params.ini` (which sets `omp_set_num_threads`). Nested parallelism (for Eigen operations, if `inner_threads > 1`) is enabled/disabled based on the `inner_threads` parameter. You can also override the outer thread count using the environment variable *before* running:
    ```bash
    export OMP_NUM_THREADS=16 # Example: Use 16 threads for the outer loop
    ./onset_solver
    ```
    *(Note: `OMP_NESTED` and `OMP_MAX_ACTIVE_LEVELS` are set internally based on `inner_threads`)*

## Input Parameters (`params.ini`)

The `params.ini` file uses a simple `key = value` format. Comments start with `#`. Key parameters include:

*   `Ek`, `Pr`, `Pm`, `elsm`, `delta`, `m`, `chim`: Physical parameters. `elsm` and `chim` can be lists (only first `chim` used, loops over `elsm`).

     (Ekman, Prandtl, magnetic Prandtl, Elsasser, field length scale, polytropic index and density ratio respectively)
*   `ny`, `nz`: Grid points (N+1).
*   `p`, `sigma1`: Eigenvalue solver parameters (number sought near `sigma1`).
*   `k1` or (`k_length`, `kstrt`, `kdiff`): Wavenumber specification.
*   `BC*`: Boundary condition flags.
*   `mean_flow`, `B_profile`: Mean flow and background field options.
*   `Ra_*`: Binary search control parameters.
*   `outer_threads`, `inner_threads`: Parallelism control.

## Output Files

Results are saved in the `output/` directory (created in the run location):

*   `Racvskc_chi*_els*.txt`: Summary file containing critical Ra vs. kx for each Elsasser number run. Includes search status flags and leading eigenvalue information.
*   `critical_vector_chi*_els*.txt`: Contains the complex eigenvector corresponding to the overall minimum critical Ra found for that Elsasser number. Format is `RealPart ImaginaryPart` per line.

## Notes

*   The shift-and-invert eigenvalue solver targets eigenvalues closest to `sigma1`.
*   The binary search finds the lowest `Ra` where the real part of a filtered eigenvalue becomes positive.
*   Memory usage can be high for large grids (`ny`, `nz`), primarily due to the LU factorization in the shift-and-invert step. The outer loop parallelism distributes this memory requirement across iterations but doesn't reduce the peak per iteration.
*   Nested parallelism (`inner_threads > 1`) may not provide significant speedup if the serial LU factorization dominates the runtime. Monitor CPU usage (`htop`) to assess effectiveness. It may not work properly in certain environments. So, benchmark it in your system first to ensure it is working as expected. This feature is not rigorously tested in the current version.