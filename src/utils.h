#ifndef UTILS_H
#define UTILS_H

#include <Eigen/Sparse>
#include <Eigen/Dense>
#include <vector>

// Function to compute Kronecker product of two sparse matrices
Eigen::SparseMatrix<double> kron_sparse(
    const Eigen::SparseMatrix<double>& A,
    const Eigen::SparseMatrix<double>& B);

// Function to create sparse diagonal matrix from a vector view
Eigen::SparseMatrix<double> sparse_diag(const Eigen::Map<const Eigen::VectorXd>& v);

// Function to create sparse diagonal matrix from a dense vector
Eigen::SparseMatrix<double> sparse_diag(const Eigen::VectorXd& v);

// Function to emulate meshgrid (returns Z_grid, Y_grid matching usage)
void meshgrid(const Eigen::VectorXd& zz_vec, // Input vector for rows
              const Eigen::VectorXd& yy_vec, // Input vector for columns
              Eigen::MatrixXd& Z_grid,       // Output grid Z(i,j) = zz_vec(i)
              Eigen::MatrixXd& Y_grid);      // Output grid Y(i,j) = yy_vec(j)

// Helper to get sparse identity matrix
Eigen::SparseMatrix<double> sparse_identity(int n);

// Helper to create the I2 matrix variants (identity on interior points)
Eigen::SparseMatrix<double> create_I2(int ny_rows, int nz_rows);

#endif