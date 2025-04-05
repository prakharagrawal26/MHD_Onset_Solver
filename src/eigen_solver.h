#ifndef EIGEN_SOLVER_H
#define EIGEN_SOLVER_H

#include <Eigen/Sparse>
#include <Eigen/Dense>
#include <vector>
#include <complex>
#include <limits>

// Structure to hold results from the eigenvalue solver
struct SolverResult {
    bool converged = false;
    int flag = 0; // 0 = no critical found, 1 = critical found

    Eigen::VectorXcd all_eig; // Store all computed eigenvalues (closest to sigma)
    std::vector<std::complex<double>> eig_val_critical_list; // Critical ones
    Eigen::VectorXcd leading_critical_eig_vec; // Eigenvector for leading critical mode
    double max_real_critical_eig = -std::numeric_limits<double>::infinity();
};

// Function to solve the generalized eigenvalue problem Ax = lambda*B*x
// Uses Spectra shift-and-invert
SolverResult solve_eigenproblem(
    const Eigen::SparseMatrix<double>& A,
    const Eigen::SparseMatrix<double>& B,
    int num_eigenvalues, // Number of eigenvalues to compute (p)
    double sigma,        // Target shift value (sigma1)
    bool find_vector     // Whether to extract the leading critical eigenvector
);

#endif // EIGEN_SOLVER_H