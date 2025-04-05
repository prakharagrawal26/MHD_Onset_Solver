#include "bc.h"          // Use new header name
#include "utils.h"       // Use new header name
#include <stdexcept>
#include <vector>

// Helper function to build a single 1D BC matrix
// Takes N+1 size, type tt, and dense Diff Matrix
Eigen::SparseMatrix<double> call_BC(int n_points, int tt, const Eigen::MatrixXd& DD_dense) {
    // ... (implementation identical to call_BC_collocation) ...
    if (n_points < 2) throw std::runtime_error("call_BC: n_points must be at least 2.");
    if (DD_dense.rows() != n_points || DD_dense.cols() != n_points) throw std::runtime_error("call_BC: DD matrix size mismatch.");
    Eigen::SparseMatrix<double> M(n_points, n_points);
    std::vector<Eigen::Triplet<double>> triplets;
    int last_idx = n_points - 1;
    if (tt == 1) { // Dirichlet
        triplets.emplace_back(0, 0, 1.0);
        triplets.emplace_back(last_idx, last_idx, 1.0);
    } else if (tt == 2) { // Neumann
        for (int j = 0; j < n_points; ++j) if (std::abs(DD_dense(0, j)) > 1e-15) triplets.emplace_back(0, j, DD_dense(0, j));
        for (int j = 0; j < n_points; ++j) if (std::abs(DD_dense(last_idx, j)) > 1e-15) triplets.emplace_back(last_idx, j, DD_dense(last_idx, j));
    } else { throw std::runtime_error("call_BC: Invalid boundary condition type 'tt'."); }
     M.setFromTriplets(triplets.begin(), triplets.end());
     return M;
}


// Main function to set up combined BC matrices
BoundaryConditions setup_BC(
                            int ny_points, int nz_points,
                            const Eigen::MatrixXd& Dy1d_dense,
                            const Eigen::MatrixXd& Dz1d_dense,
                            int BCzvel, int BCzmag,
                            int BCyvel, int BCymag)
{
    BoundaryConditions bc_out;
    std::vector<int> tt_z(7), tt_y(7);

    // Define BC types based on input flags
    if (BCzvel == 2 && BCzmag == 1) tt_z = {2, 2, 1, 1, 1, 2, 1};
    else if (BCzvel == 1 && BCzmag == 1) tt_z = {1, 1, 1, 1, 1, 2, 1};
    else throw std::runtime_error("Unsupported BCz combination");

    if (BCyvel == 2 && BCymag == 1) tt_y = {2, 1, 2, 1, 2, 1, 1};
    else if (BCyvel == 1 && BCymag == 1) tt_y = {1, 1, 1, 1, 2, 1, 1};
    else throw std::runtime_error("Unsupported BCy combination");

    // Generate 1D BC matrices
    Eigen::SparseMatrix<double> BCzUx_1d = call_BC(nz_points, tt_z[0], Dz1d_dense);
    Eigen::SparseMatrix<double> BCzUy_1d = call_BC(nz_points, tt_z[1], Dz1d_dense);
    Eigen::SparseMatrix<double> BCzUz_1d = call_BC(nz_points, tt_z[2], Dz1d_dense);
    Eigen::SparseMatrix<double> BCzBx_1d = call_BC(nz_points, tt_z[3], Dz1d_dense);
    Eigen::SparseMatrix<double> BCzBy_1d = call_BC(nz_points, tt_z[4], Dz1d_dense);
    Eigen::SparseMatrix<double> BCzBz_1d = call_BC(nz_points, tt_z[5], Dz1d_dense);
    Eigen::SparseMatrix<double> BCzS_1d  = call_BC(nz_points, tt_z[6], Dz1d_dense);

    Eigen::SparseMatrix<double> BCyUx_1d = call_BC(ny_points, tt_y[0], Dy1d_dense);
    Eigen::SparseMatrix<double> BCyUy_1d = call_BC(ny_points, tt_y[1], Dy1d_dense);
    Eigen::SparseMatrix<double> BCyUz_1d = call_BC(ny_points, tt_y[2], Dy1d_dense);
    Eigen::SparseMatrix<double> BCyBx_1d = call_BC(ny_points, tt_y[3], Dy1d_dense);
    Eigen::SparseMatrix<double> BCyBy_1d = call_BC(ny_points, tt_y[4], Dy1d_dense);
    Eigen::SparseMatrix<double> BCyBz_1d = call_BC(ny_points, tt_y[5], Dy1d_dense);
    Eigen::SparseMatrix<double> BCyS_1d  = call_BC(ny_points, tt_y[6], Dy1d_dense);

    // Combine using Kronecker products (replicating MATLAB BC.m)
    Eigen::SparseMatrix<double> Iy_full = sparse_identity(ny_points);
    Eigen::SparseMatrix<double> Iz_full = sparse_identity(nz_points);

    // Create Iy2/Iz2 = Identity with zeroed boundary rows/cols
    Eigen::SparseMatrix<double> Iy2 = sparse_identity(ny_points);
    Eigen::SparseMatrix<double> Iz2 = sparse_identity(nz_points);
    std::vector<Eigen::Triplet<double>> triplets_y2, triplets_z2;
    triplets_y2.reserve(ny_points - 2);
    triplets_z2.reserve(nz_points - 2);
    for(int k=1; k < ny_points - 1; ++k) triplets_y2.emplace_back(k, k, 1.0);
    for(int k=1; k < nz_points - 1; ++k) triplets_z2.emplace_back(k, k, 1.0);
    Iy2.setFromTriplets(triplets_y2.begin(), triplets_y2.end());
    Iz2.setFromTriplets(triplets_z2.begin(), triplets_z2.end());

    // Apply kron products as in MATLAB
    Eigen::SparseMatrix<double> BCzUx_full = kron_sparse(Iy2, BCzUx_1d);
    Eigen::SparseMatrix<double> BCzUy_full = kron_sparse(Iy2, BCzUy_1d);
    Eigen::SparseMatrix<double> BCzUz_full = kron_sparse(Iy2, BCzUz_1d);
    Eigen::SparseMatrix<double> BCzBx_full = kron_sparse(Iy2, BCzBx_1d);
    Eigen::SparseMatrix<double> BCzBy_full = kron_sparse(Iy2, BCzBy_1d);
    Eigen::SparseMatrix<double> BCzBz_full = kron_sparse(Iy2, BCzBz_1d);
    Eigen::SparseMatrix<double> BCzS_full  = kron_sparse(Iy2, BCzS_1d);

    Eigen::SparseMatrix<double> BCyUx_full = kron_sparse(BCyUx_1d, Iz_full);
    Eigen::SparseMatrix<double> BCyUy_full = kron_sparse(BCyUy_1d, Iz_full);
    Eigen::SparseMatrix<double> BCyUz_full = kron_sparse(BCyUz_1d, Iz_full);
    Eigen::SparseMatrix<double> BCyBx_full = kron_sparse(BCyBx_1d, Iz_full);
    Eigen::SparseMatrix<double> BCyBy_full = kron_sparse(BCyBy_1d, Iz_full);
    Eigen::SparseMatrix<double> BCyBz_full = kron_sparse(BCyBz_1d, Iz_full);
    Eigen::SparseMatrix<double> BCyS_full  = kron_sparse(BCyS_1d, Iz_full);

    // Sum components
    bc_out.BCUx = BCzUx_full + BCyUx_full;
    bc_out.BCUy = BCzUy_full + BCyUy_full;
    bc_out.BCUz = BCzUz_full + BCyUz_full;
    bc_out.BCBx = BCzBx_full + BCyBx_full;
    bc_out.BCBy = BCzBy_full + BCyBy_full;
    bc_out.BCBz = BCzBz_full + BCyBz_full;
    bc_out.BCS = BCzS_full + BCyS_full;

    return bc_out;
}