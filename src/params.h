#ifndef PARAMS_H
#define PARAMS_H

#include <vector>
#include <string>
#include <cmath>

// Structure to hold simulation parameters
struct Params {
    // Provide sensible defaults, can be overridden by input file
    // Physical Params
    double Ek = 1e-5;
    double Pr = 1.0;
    double Pm = 1.0;
    double q = 1.0; // Derived: Pm / Pr
    std::vector<double> elsm = {0.0};
    double delta = 1.0;
    double m = 1.495;
    std::vector<double> chim = {1.0};
    double theta = 0.0; // Derived from chim, m
    double Ra = 91.0; // Dummy start value for Ra search

    // Numerical Params
    int ny = 71; // Grid points in y (N+1)
    int nz = 71; // Grid points in z (N+1)
    int p = 10;  // Number of eigenvalues requested from solver
    double sigma1 = 1e-6; // Target shift for eigenvalue solver

    // Grid Params
    double Asp = 1.0;   // Aspect ratio (used for scaling Dy)
    double Y_range = 2.0; // Domain scaling: 1 for [-1,1]*Asp/2, 2 for [0,Asp]
    double Z_range = 1.0; // Domain scaling: 1 for [-1,1], 2 for [0,1]

    // Wavenumber Params (Defaults overridden by file if specified)
    int k_length = 1;
    double kstrt = 10.0;
    double kdiff = 1.0;
    std::vector<double> k1; // Wavenumbers (calculated or read)

    // BC Params
    int BCzmag = 1; // 1=insulating
    int BCzvel = 2; // 2=stress-free
    int BCymag = 1; // 1=insulating
    int BCyvel = 2; // 2=stress-free

    // Mean Flow Params
    int mean_flow = 1; // 1 = include mean flow terms
    int B_profile = 1; // B field: 1=anti, 2=symm, 3=poly, 4=const

    // Binary Search Params
    double Ra_start = 50.0;      // Initial lower bound for Ra search
    double Ra_end_init = 140.0;  // Initial upper bound
    double Ra_extend_step = 10.0;// Amount to add if both bounds stable
    double Ra_reduce_step = 10.0;// Amount to subtract if both unstable
    double Ra_search_limit = 250.0;// Upper limit for extending search
    double Ra_accuracy = 0.05;   // Tolerance |Ra_end - Ra_start|

    // Parallelism Params (Informational - control via OMP env vars/calls)
    int outer_threads = 4; // Example default
    int inner_threads = 1; // Example default

    // --- Derived Parameter Calculation ---
    // Call this *after* loading parameters from file
    void calculate_derived() {
        q = Pm / Pr;

        if (!chim.empty()) {
             double current_chim = chim[0]; // Use first chim value
             if (current_chim > 0 && std::abs(m) > 1e-15) {
                 if (std::abs(current_chim - 1.0) < 1e-12) {
                     theta = 0.0;
                 } else {
                     double chim_pow_inv_m = std::pow(current_chim, 1.0 / m);
                     theta = -(1.0 / chim_pow_inv_m) * (-1.0 + chim_pow_inv_m);
                 }
             } else { theta = 0.0; }
        } else { theta = 0.0; }

        // Calculate k1 wavenumbers if not read explicitly
        // (File reading logic in params.cpp will handle 'k1 = ...' line)
        if (k1.empty()) { // Generate sequence only if k1 wasn't specified in file
            k1.resize(k_length);
            if (k_length > 0) {
                k1[0] = kstrt;
                for (int ttt = 1; ttt < k_length; ++ttt) {
                    k1[ttt] = k1[ttt - 1] + kdiff;
                }
            }
        }
    }
};

// Function declarations
bool load_params_from_file(const std::string& filename, Params& params);

#endif // PARAMS_H