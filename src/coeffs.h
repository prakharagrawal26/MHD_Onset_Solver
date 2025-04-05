#ifndef COEFFS_H
#define COEFFS_H

#include <Eigen/Sparse>
#include <Eigen/Dense>

// Structure to hold variable coefficient matrices
struct VariableCoeffs {
     Eigen::SparseMatrix<double> c1, c2, c3, c4, c5, c6, c7, c8;
     Eigen::SparseMatrix<double> ux0, uz0; // Mean flow components
};

// Function declaration
VariableCoeffs calculate_coeffs(
                     double delta, double theta, double m,
                     const Eigen::MatrixXd& Y_grid,
                     const Eigen::MatrixXd& Z_grid,
                     int B_profile,
                     double els);

#endif // COEFFS_H