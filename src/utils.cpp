#include "utils.h" // Use new header name
#include <vector>
#include <stdexcept>

// Kronecker product Implementation
Eigen::SparseMatrix<double> kron_sparse(
    const Eigen::SparseMatrix<double>& A,
    const Eigen::SparseMatrix<double>& B)
{
    // ... (implementation identical to utils_collocation.cpp) ...
    int rowsA = A.rows(), colsA = A.cols();
    int rowsB = B.rows(), colsB = B.cols();
    int rowsK = rowsA * rowsB;
    int colsK = colsA * colsB;

    Eigen::SparseMatrix<double> K(rowsK, colsK);
    std::vector<Eigen::Triplet<double>> tripletList;
    tripletList.reserve(A.nonZeros() * B.nonZeros());

    for (int kA = 0; kA < A.outerSize(); ++kA) {
        for (Eigen::SparseMatrix<double>::InnerIterator itA(A, kA); itA; ++itA) {
            for (int kB = 0; kB < B.outerSize(); ++kB) {
                for (Eigen::SparseMatrix<double>::InnerIterator itB(B, kB); itB; ++itB) {
                    int rowK = itA.row() * rowsB + itB.row();
                    int colK = itA.col() * colsB + itB.col();
                    tripletList.emplace_back(rowK, colK, itA.value() * itB.value());
                }
            }
        }
    }
    K.setFromTriplets(tripletList.begin(), tripletList.end());
    return K;
}

// Sparse Diagonal from Map
Eigen::SparseMatrix<double> sparse_diag(const Eigen::Map<const Eigen::VectorXd>& v) {
    // ... (implementation identical to utils_collocation.cpp) ...
    int n = v.size();
    Eigen::SparseMatrix<double> D(n, n);
    std::vector<Eigen::Triplet<double>> tripletList;
    tripletList.reserve(n);
    for (int i = 0; i < n; ++i) {
        if (std::abs(v(i)) > 1e-15) {
             tripletList.emplace_back(i, i, v(i));
        }
    }
    D.setFromTriplets(tripletList.begin(), tripletList.end());
    return D;
}

// Sparse Diagonal from VectorXd
Eigen::SparseMatrix<double> sparse_diag(const Eigen::VectorXd& v) {
    // ... (implementation identical to utils_collocation.cpp) ...
     int n = v.size();
    Eigen::SparseMatrix<double> D(n, n);
    std::vector<Eigen::Triplet<double>> tripletList;
    tripletList.reserve(n);
    for (int i = 0; i < n; ++i) {
        if (std::abs(v(i)) > 1e-15) {
             tripletList.emplace_back(i, i, v(i));
        }
    }
    D.setFromTriplets(tripletList.begin(), tripletList.end());
    return D;
}


// Meshgrid Implementation
void meshgrid(const Eigen::VectorXd& zz_vec, // Input vector for rows
              const Eigen::VectorXd& yy_vec, // Input vector for columns
              Eigen::MatrixXd& Z_grid,       // Output grid Z(i,j) = zz_vec(i)
              Eigen::MatrixXd& Y_grid)      // Output grid Y(i,j) = yy_vec(j)
{
    int nz_len = zz_vec.size();
    int ny_len = yy_vec.size();

    if (nz_len == 0 || ny_len == 0) {
         throw std::runtime_error("Meshgrid input vectors cannot be empty.");
    }
    Z_grid.resize(nz_len, ny_len);
    Y_grid.resize(nz_len, ny_len);
    for (int i = 0; i < nz_len; ++i) {
        for (int j = 0; j < ny_len; ++j) {
            Z_grid(i, j) = zz_vec(i);
            Y_grid(i, j) = yy_vec(j);
        }
    }
}

// Sparse Identity
Eigen::SparseMatrix<double> sparse_identity(int n) {
    if (n <= 0) {
        throw std::runtime_error("Identity matrix size must be positive.");
    }
    Eigen::SparseMatrix<double> I(n, n);
    I.setIdentity();
    return I;
}

// Create I2 variant (Identity on inner points)
Eigen::SparseMatrix<double> create_I2(int ny_rows, int nz_rows) {
    // ... (implementation identical to create_I2_collocation) ...
     if (ny_rows < 2 || nz_rows < 2) {
          throw std::runtime_error("create_I2 requires dimensions >= 2.");
     }
     int N_tot = ny_rows * nz_rows;
     Eigen::SparseMatrix<double> I2(N_tot, N_tot);
     std::vector<Eigen::Triplet<double>> triplets;
     triplets.reserve( (ny_rows-2) * (nz_rows-2) );
     for (int i = 0; i < nz_rows; ++i) { // Z loop (rows)
         for (int j = 0; j < ny_rows; ++j) { // Y loop (cols)
             int global_idx = i * ny_rows + j;
             if (j > 0 && j < ny_rows - 1 && i > 0 && i < nz_rows - 1) {
                 triplets.emplace_back(global_idx, global_idx, 1.0);
             }
         }
     }
     I2.setFromTriplets(triplets.begin(), triplets.end());
     return I2;
}