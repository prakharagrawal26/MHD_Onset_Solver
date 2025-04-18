#ifndef MATRIX_BUILDER_H
#define MATRIX_BUILDER_H

#include "bc.h"     
#include "coeffs.h" 
#include <Eigen/Sparse>

// Structure to store differentiation matrices
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
    const BoundaryConditions& bc,       
    const VariableCoeffs& coeffs,     
    const DiffMatrices& diff,
    Eigen::SparseMatrix<double>& A,
    Eigen::SparseMatrix<double>& B);

#endif