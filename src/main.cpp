#include "params.h" 
#include "utils.h" 
#include "cheb.h"  
#include "bc.h"    
#include "coeffs.h"    
#include "matrix_builder.h"  
#include "eigen_solver.h"    
#include <iostream>
#include <vector>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <limits>
#include <string>
#include <chrono>
#include <omp.h>
#include <filesystem>

int main(int argc, char* argv[]) { 
    auto t_start_main = std::chrono::high_resolution_clock::now();
    double omp_t_start_main = omp_get_wtime();

    //  Load Parameters 
        std::string param_filename = "params.ini"; // Default filename
    if (argc > 1) { // Check if filename provided on command line
        param_filename = argv[1];
    }
    std::cout << "Loading parameters from: " << param_filename << std::endl;
    Params params; // Use default constructor first
    if (!load_params_from_file(param_filename, params)) {
        std::cerr << "Failed to load parameters. Exiting." << std::endl;
        return 1;
    }

    const int ny = params.ny; const int nz = params.nz; const int N_tot = ny * nz;
    const double Y_range_scale = (params.Y_range == 2.0) ? 2.0 : 1.0;
    const double Z_range_scale = (params.Z_range == 2.0) ? 2.0 : 1.0;

    std::cout << "Grid size: ny=" << ny << ", nz=" << nz << std::endl;
    std::cout << "Running for chi = " << (params.chim.empty() ? 0.0 : params.chim[0]) << std::endl;

    // --- Setup grid and derivative matrices ---
    std::cout << "Setting up Chebyshev grid and matrices..." << std::endl;
    Eigen::MatrixXd Dy1d_dense, D2y1d_dense, Dz1d_dense, D2z1d_dense;
    Eigen::VectorXd yy_cheb, zz_cheb, yy, zz;

    cheb(ny - 1, Dy1d_dense, yy_cheb);
    Dy1d_dense *= (Y_range_scale / params.Asp);
    D2y1d_dense = Dy1d_dense * Dy1d_dense;

    if (params.Y_range == 2.0) yy = params.Asp * (yy_cheb.array() + 1.0) / 2.0;
    else yy = params.Asp * yy_cheb.array() / 2.0;

    cheb(nz - 1, Dz1d_dense, zz_cheb);
    Dz1d_dense *= Z_range_scale;
    D2z1d_dense = Dz1d_dense * Dz1d_dense;
    if (params.Z_range == 2.0) zz = (zz_cheb.array() + 1.0) / 2.0;
    else zz = zz_cheb;

    Eigen::SparseMatrix<double> Dy1d = Dy1d_dense.sparseView(); 
    Eigen::SparseMatrix<double> Dz1d = Dz1d_dense.sparseView();
    Eigen::SparseMatrix<double> D2y1d = D2y1d_dense.sparseView(); 
    Eigen::SparseMatrix<double> D2z1d = D2z1d_dense.sparseView();
    Eigen::SparseMatrix<double> Iy = sparse_identity(ny); 
    Eigen::SparseMatrix<double> Iz = sparse_identity(nz);
    Eigen::SparseMatrix<double> I_full = kron_sparse(Iy, Iz);
    Eigen::SparseMatrix<double> I2_full = create_I2(ny, nz); 
    Eigen::SparseMatrix<double> Dy_full = kron_sparse(Dy1d, Iz); 
    Eigen::SparseMatrix<double> Dz_full = kron_sparse(Iy, Dz1d);
    Eigen::SparseMatrix<double> D2y_full = kron_sparse(D2y1d, Iz);
    Eigen::SparseMatrix<double> D2z_full = kron_sparse(Iy, D2z1d);

    Eigen::MatrixXd Y_grid, Z_grid;
    meshgrid(zz, yy, Z_grid, Y_grid);
    std::cout << "Grid setup complete." << std::endl;

    // Loop over chim and elsm
    if (params.chim.empty() || params.elsm.empty()) { std::cerr << "Error: chim or elsm empty!\n"; return 1; }
    double chi = params.chim[0];

    for (size_t j = 0; j < params.elsm.size(); ++j) {
        double els_base = params.elsm[j];
        double els = (std::abs(chi) > 1e-15) ? els_base / chi : els_base;
        std::cout << "\n===== Running for chi = " << chi << ", els_base = " << els_base << " (els = " << els << ") =====" << std::endl;
        auto t_start_loop = std::chrono::high_resolution_clock::now();

        BoundaryConditions bc = setup_BC(ny, nz, Dy1d_dense, Dz1d_dense, params.BCzvel, params.BCzmag, params.BCyvel, params.BCymag);
        VariableCoeffs coeffs = calculate_coeffs(params.delta, params.theta, params.m, Y_grid, Z_grid, params.B_profile, els);

        int num_k = params.k1.size();
        if (num_k == 0) { std::cerr << "Error: Wavenumber vector k1 empty!\n"; continue; }
        std::vector<double> Rac_results(num_k); std::vector<int> binary_search_flags(num_k);
        std::vector<double> leading_eig_real(num_k, std::nan("")); std::vector<double> leading_eig_imag(num_k, std::nan(""));
        Eigen::VectorXcd critical_eig_vec_at_min_Ra;
        double min_Rac_for_vec = std::numeric_limits<double>::infinity();

        std::cout << "Starting parallel loop over " << num_k << " wavenumbers (kx)..." << std::endl;
        std::cout << " >> OMP Outer Threads: " << params.outer_threads << ", Eigen Inner Threads: " << params.inner_threads << std::endl;

        // Set up nested parallelism IF inner_threads > 1
        if (params.inner_threads > 1) {
            omp_set_nested(1);
            omp_set_max_active_levels(2); // Allow 2 levels
        } else {
             omp_set_nested(0); // Disable nesting if inner=1
             omp_set_max_active_levels(1);
        }
        omp_set_num_threads(params.outer_threads); // Set outer threads

        #pragma omp parallel for schedule(dynamic)
        for (int rr = 0; rr < num_k; ++rr) {
            int thread_id = omp_get_thread_num();
            // Set inner thread count for Eigen (only if nesting enabled)
            if (params.inner_threads > 1) {
                Eigen::setNbThreads(params.inner_threads);
            } else {
                 Eigen::setNbThreads(1); // Ensure serial Eigen if not nesting
            }

            double kx = params.k1[rr];
            Eigen::SparseMatrix<double> D2_kx = D2y_full + D2z_full - (kx * kx) * I_full;
            DiffMatrices diff = {I_full, I2_full, Dy_full, Dz_full, D2y_full, D2z_full, D2_kx};
            Eigen::SparseMatrix<double> A_local, B_local;

            // Binary Search
            double Ra_strt = params.Ra_start; double Ra_endd = params.Ra_end_init;
            double Ra_low_unstable = std::numeric_limits<double>::infinity();
            int flag1 = 0, flag2 = 0, flag3 = 0; int current_flag = 1;
            int iter = 0; int max_iter = 100;

            // Initial checks
            build_matrix(kx, params.Ek, params.Pr, params.Pm, els, Ra_strt, params.m, params.theta, params.mean_flow, bc, coeffs, diff, A_local, B_local);
            SolverResult res_low = solve_eigenproblem(A_local, B_local, params.p, params.sigma1, false);
            flag1 = res_low.flag;
            build_matrix(kx, params.Ek, params.Pr, params.Pm, els, Ra_endd, params.m, params.theta, params.mean_flow, bc, coeffs, diff, A_local, B_local);
            SolverResult res_high = solve_eigenproblem(A_local, B_local, params.p, params.sigma1, false);
            flag2 = res_high.flag;

            while (Ra_endd - Ra_strt > params.Ra_accuracy && iter < max_iter) {
                 iter++;
                 if (Ra_endd > params.Ra_search_limit) { /* ... warning & break ... */ current_flag = 2; break; }

                 if (flag1 == 1 && flag2 == 1) { // Both unstable
                     Ra_low_unstable = std::min(Ra_low_unstable, Ra_strt);
                     Ra_endd = Ra_strt; Ra_strt -= params.Ra_reduce_step;
                     if (Ra_strt < 0) Ra_strt = 0;
                     if (Ra_strt >= Ra_endd) { current_flag = 3; break; }
                     build_matrix(kx, params.Ek, params.Pr, params.Pm, els, Ra_strt, params.m, params.theta, params.mean_flow, bc, coeffs, diff, A_local, B_local);
                     flag1 = solve_eigenproblem(A_local, B_local, params.p, params.sigma1, false).flag;
                     flag2 = 1; // Assume end is still unstable
                     continue;

                 } else if (flag1 == 0 && flag2 == 0) { // Both stable
                     Ra_strt = Ra_endd; Ra_endd += params.Ra_extend_step;
                     build_matrix(kx, params.Ek, params.Pr, params.Pm, els, Ra_endd, params.m, params.theta, params.mean_flow, bc, coeffs, diff, A_local, B_local);
                     flag1 = 0; // Assume start is still stable
                     flag2 = solve_eigenproblem(A_local, B_local, params.p, params.sigma1, false).flag;
                     continue;

                 } else if (flag1 == 0 && flag2 == 1) { // Bracketed
                     Ra_low_unstable = std::min(Ra_low_unstable, Ra_endd);
                     double Ra_midd = (Ra_strt + Ra_endd) / 2.0;
                     if (std::abs(Ra_midd - Ra_strt) < 1e-9 || std::abs(Ra_midd - Ra_endd) < 1e-9) { current_flag = 3; break; }
                     build_matrix(kx, params.Ek, params.Pr, params.Pm, els, Ra_midd, params.m, params.theta, params.mean_flow, bc, coeffs, diff, A_local, B_local);
                     flag3 = solve_eigenproblem(A_local, B_local, params.p, params.sigma1, false).flag;
                     if (flag3 == 0) { Ra_strt = Ra_midd; flag1 = 0; }
                     else { Ra_endd = Ra_midd; flag2 = 1; Ra_low_unstable = std::min(Ra_low_unstable, Ra_endd); }

                 } else { /* ... error & break ... */ current_flag = 0; break; }
                 if (iter >= max_iter) { /* ... warning & break ... */ current_flag = (flag2 == 1) ? 3 : 2; break; }
            } 
            // end while binary search

            if (current_flag == 1) current_flag = 3; // Finished by tolerance

            // Store result
            double final_Rac = std::nan(""); 
            double final_eig_real = std::nan(""); 
            double final_eig_imag = std::nan("");
            Eigen::VectorXcd current_crit_vec;

            if (current_flag == 3) { // Success
                final_Rac = Ra_low_unstable;
                if (std::isfinite(final_Rac)) {
                     bool request_vector = (final_Rac < min_Rac_for_vec);
                     build_matrix(kx, params.Ek, params.Pr, params.Pm, els, final_Rac, params.m, params.theta, params.mean_flow, bc, coeffs, diff, A_local, B_local);
                     SolverResult final_res = solve_eigenproblem(A_local, B_local, params.p, params.sigma1, request_vector);
                     if (final_res.flag == 1) {
                        final_eig_real = final_res.max_real_critical_eig;
                         for(const auto& ev : final_res.eig_val_critical_list) if (std::abs(ev.real() - final_eig_real) < 1e-9 * std::abs(final_eig_real)){ final_eig_imag = ev.imag(); break; }
                         if (request_vector && final_res.leading_critical_eig_vec.size() > 0) current_crit_vec = final_res.leading_critical_eig_vec;
                     }
                }
            } else if (current_flag == 2) final_Rac = std::numeric_limits<double>::infinity();
            else final_Rac = -std::numeric_limits<double>::infinity();

            #pragma omp critical
            {
                Rac_results[rr] = final_Rac; binary_search_flags[rr] = current_flag;
                leading_eig_real[rr] = final_eig_real; leading_eig_imag[rr] = final_eig_imag;
                if (current_flag == 3 && final_Rac >= 0 && final_Rac < min_Rac_for_vec) {
                     min_Rac_for_vec = final_Rac;
                     if (current_crit_vec.size() > 0) critical_eig_vec_at_min_Ra = current_crit_vec;
                     else critical_eig_vec_at_min_Ra.resize(0);
                }
                 // if (rr % 10 == 0) { // Print every 10 iterations
                     std::cout << "  kx = " << std::fixed << std::setprecision(4) << kx
                               << ", Ra_c = " << std::scientific << std::setprecision(6) << final_Rac << std::defaultfloat << std::endl;
                 // }
            } // end critical

        } 
        // --- Post-processing and Saving ---
        double Racmin = std::numeric_limits<double>::infinity(); double kxc = -1.0; int index_min = -1;
        for (int rr = 0; rr < num_k; ++rr) if (std::isfinite(Rac_results[rr]) && Rac_results[rr] >= 0 && Rac_results[rr] < Racmin) { Racmin = Rac_results[rr]; kxc = params.k1[rr]; index_min = rr; }

         if (index_min != -1) {
            std::cout << std::fixed << std::setprecision(6);
            std::cout << "Minimum critical Ra_c = " << Racmin << " found at kx = " << kxc << "\n";
             if (std::abs(Racmin - min_Rac_for_vec) > 1e-6 || critical_eig_vec_at_min_Ra.size() == 0) {
                  std::cout << "Recalculating eigenvector at precise minimum   " << std::endl;
                  Eigen::SparseMatrix<double> A_crit, B_crit; Eigen::SparseMatrix<double> D2_crit = D2y_full + D2z_full - (kxc * kxc) * I_full;
                  DiffMatrices diff_crit = {I_full, I2_full, Dy_full, Dz_full, D2y_full, D2z_full, D2_crit};
                  build_matrix(kxc, params.Ek, params.Pr, params.Pm, els, Racmin, params.m, params.theta, params.mean_flow, bc, coeffs, diff_crit, A_crit, B_crit);
                  SolverResult final_res = solve_eigenproblem(A_crit, B_crit, params.p, params.sigma1, true);
                   if (final_res.flag == 1 && final_res.leading_critical_eig_vec.size() > 0) critical_eig_vec_at_min_Ra = final_res.leading_critical_eig_vec;
                   else { std::cerr << "Warning: Failed to recalculate critical eigenvector.\n"; critical_eig_vec_at_min_Ra.resize(0); }
             }
        } else { std::cout << "\n Could not find valid minimum critical Ra_c for els_base = " << els_base << ".\n"; }

        // --- Save Results ---
        std::string out_dir = "output";
        try { if (!std::filesystem::exists(out_dir)) std::filesystem::create_directory(out_dir); }
        catch (const std::exception& e) { std::cerr << "Error creating output directory: " << e.what() << "\n"; out_dir = "."; }

        std::stringstream ss_fname;
        ss_fname << out_dir << "/Racvskc_chi" << static_cast<int>(chi) << "_els" << std::scientific << std::setprecision(1) << els_base << ".txt";
        std::ofstream res_file(ss_fname.str());
        if (res_file.is_open()) { /* ... write header and results ... */ res_file.close(); std::cout << "Summary results saved to " << ss_fname.str() << "\n"; }
        else { std::cerr << "Error opening file: " << ss_fname.str() << "\n"; }

        if (index_min != -1 && critical_eig_vec_at_min_Ra.size() > 0) {
             std::stringstream ss_vec_fname;
             ss_vec_fname << out_dir << "/critical_vector_chi" << static_cast<int>(chi) << "_els" << std::scientific << std::setprecision(1) << els_base << ".txt";
             std::ofstream ev_file(ss_vec_fname.str());
             if (ev_file.is_open()) { /* ... write header and eigenvector ... */ ev_file.close(); std::cout << "Critical eigenvector saved to " << ss_vec_fname.str() << "\n"; }
             else { std::cerr << "Error opening file: " << ss_vec_fname.str() << "\n"; }
        }

        auto t_end_loop = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed_loop = t_end_loop - t_start_loop;
        std::cout << "Time for els_base = " << els_base << " loop: " << elapsed_loop.count() << " s\n";
        std::cout << "================================================" << std::endl;
    } // End loop over elsm

    auto t_end_main = std::chrono::high_resolution_clock::now();
    double omp_t_end_main = omp_get_wtime();
    std::chrono::duration<double> elapsed_main = t_end_main - t_start_main;
    std::cout << "\nTotal Wall Time: " << elapsed_main.count() << " seconds\n";
    return 0;
}