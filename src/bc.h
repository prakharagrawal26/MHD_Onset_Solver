#ifndef BC_H
#define BC_H

#include <Eigen/Sparse>
#include <Eigen/Dense>
#include <vector>

// Structure to hold boundary condition matrices
struct BoundaryConditions {
    Eigen::SparseMatrix<double> BCUx, BCUy, BCUz, BCBx, BCBy, BCBz, BCS;
};

Eigen::SparseMatrix<double> call_BC(int n_points, int tt, const Eigen::MatrixXd& DD_dense);

// Main function to set up all boundary conditions
BoundaryConditions setup_BC(
                            int ny_points, int nz_points,
                            const Eigen::MatrixXd& Dy1d_dense,
                            const Eigen::MatrixXd& Dz1d_dense,
                            int BCzvel, int BCzmag,
                            int BCyvel, int BCymag);

#endif