#ifndef MATRIX_BUILDER_H
#define MATRIX_BUILDER_H

#include "bc.h"     // Use new header name
#include "coeffs.h" // Use new header name
#include <Eigen/Sparse>

// Structure to hold differentiation matrices
struct DiffMatrices {
    const Eigen::SparseMatrix<double>& I;
    const Eigen::SparseMatrix<double>& I2;
    const Eigen::SparseMatrix<double>& Dy;
    const Eigen::SparseMatrix<double>& Dz;
    const Eigen::SparseMatrix<double>& D2y;
    const Eigen::SparseMatrix<double>& D2z;
    const Eigen::SparseMatrix<double>& D2; // Laplacian
};

// Function declaration
void build_matrix(
    double kx, double Ek, double Pr, double Pm, double els, double Ra,
    double m, double theta, int mean_flow,
    const BoundaryConditions& bc,       // Use renamed struct
    const VariableCoeffs& coeffs,     // Use renamed struct
    const DiffMatrices& diff,
    Eigen::SparseMatrix<double>& A,
    Eigen::SparseMatrix<double>& B);

#endif // MATRIX_BUILDER_H