#include "eigen_solver.h" // Use new header name

// Core Spectra Includes
#include <Spectra/GenEigsRealShiftSolver.h>
#include <Spectra/MatOp/SparseGenMatProd.h>

// Eigen Includes
#include <Eigen/SparseLU>
#include <Eigen/Core>
#include <Eigen/SparseCore>

// Standard C++ Includes
#include <vector>
#include <limits>
#include <algorithm>
#include <iostream>
#include <stdexcept>
#include <complex>

namespace { // Keep helper classes internal to this file

// Comparator for sorting complex eigenvalues by real part (descending)
struct CompareEigenvaluesRealDesc {
    bool operator()(const std::complex<double>& a, const std::complex<double>& b) const {
        return a.real() > b.real();
    }
};

// Custom Operator Class for Shift-and-Invert (Ax = lambda Bx)
// Computes y = (A - sigma B)^-1 B x
class ShiftInvertGenOp {
public:
    using Scalar = double;
private:
    const Eigen::SparseMatrix<Scalar>& m_A;
    const Eigen::SparseMatrix<Scalar>& m_B;
    double m_sigma;
    mutable Eigen::SparseLU<Eigen::SparseMatrix<Scalar>, Eigen::COLAMDOrdering<int>> m_solver;
    mutable bool m_factorization_computed;

    void ensure_factorization() const {
        if (!m_factorization_computed) {
            Eigen::SparseMatrix<Scalar> M = m_A - m_sigma * m_B;
            m_solver.compute(M);
            m_factorization_computed = (m_solver.info() == Eigen::Success);
            if (!m_factorization_computed) {
                std::cerr << "ShiftInvertGenOp Warning: Eigen SparseLU factorization failed."
                          << " Eigen Info: " << m_solver.info() << std::endl;
            }
        }
    }
public:
    ShiftInvertGenOp(const Eigen::SparseMatrix<Scalar>& A,
                     const Eigen::SparseMatrix<Scalar>& B,
                     double sigma_in) :
        m_A(A), m_B(B), m_sigma(sigma_in), m_factorization_computed(false)
    {}

    int rows() const { return m_A.rows(); }
    int cols() const { return m_A.cols(); }
    void set_shift(double sigma_ignored) { } // Dummy

    void perform_op(const Scalar* x_in, Scalar* y_out) const {
        ensure_factorization();
        Eigen::Map<Eigen::Matrix<Scalar, Eigen::Dynamic, 1>> y(y_out, this->rows());
        if (!m_factorization_computed) { y.setZero(); return; }
        Eigen::Map<const Eigen::Matrix<Scalar, Eigen::Dynamic, 1>> x(x_in, this->cols());
        Eigen::Matrix<Scalar, Eigen::Dynamic, 1> z = m_B * x;
        y = m_solver.solve(z);
        if (m_solver.info() != Eigen::Success) {
            std::cerr << "ShiftInvertGenOp Warning: Eigen SparseLU solve failed! Eigen Info: " << m_solver.info() << std::endl;
            y.setZero();
        }
    }
}; // End internal class ShiftInvertGenOp

} // End anonymous namespace


// Main Solver Function using the Custom Operator
SolverResult solve_eigenproblem( // Renamed function
    const Eigen::SparseMatrix<double>& A,
    const Eigen::SparseMatrix<double>& B,
    int num_eigenvalues, // p
    double sigma,        // sigma1
    bool find_vector)
{
    SolverResult result; // Use specific result struct
    int n = A.rows();

    // --- Input Validation & Parameter Adjustment ---
    if (n == 0) { result.flag = 0; std::cerr << "Eigen Solver Error: Input matrices empty.\n"; return result; }
    if (A.cols() != n || B.rows() != n || B.cols() != n) { result.flag = 0; std::cerr << "Eigen Solver Error: Matrix dimensions mismatch.\n"; return result; }
    if (num_eigenvalues <= 0) { result.flag = 0; std::cerr << "Eigen Solver Error: Num eigenvalues must be positive.\n"; return result; }
    if (num_eigenvalues >= n) { num_eigenvalues = std::max(1, n / 2); std::cerr << "Eigen Solver Warning: Requesting " << num_eigenvalues << " eigenvalues (adjusted from >= N).\n"; }
    int ncv = std::min(std::max(2 * num_eigenvalues + 1, 20), n);
    if (ncv <= num_eigenvalues) { ncv = std::min(num_eigenvalues + 1, n); }
    if (num_eigenvalues >= ncv || ncv > n) { result.flag = 0; std::cerr << "Eigen Solver Error: Invalid config nev=" << num_eigenvalues << ", ncv=" << ncv << ", n=" << n << "\n"; return result; }

    // --- Spectra Solve ---
    try {
        ShiftInvertGenOp op(A, B, sigma); // Use operator from anonymous namespace

        Spectra::GenEigsRealShiftSolver<ShiftInvertGenOp> geigs(op, num_eigenvalues, ncv, sigma);

        geigs.init();
        int nconv = geigs.compute(Spectra::SortRule::LargestMagn);

        result.converged = (geigs.info() == Spectra::CompInfo::Successful);
        int num_computed = geigs.eigenvalues().size();

        if (result.converged || num_computed > 0) {
             if (!result.converged) { std::cerr << "Spectra Warning: Did not fully converge. Info: " << static_cast<int>(geigs.info()) << ". Using " << num_computed << " pairs.\n"; }

            Eigen::VectorXcd op_eigenvalues = geigs.eigenvalues();
            result.all_eig.resize(op_eigenvalues.size());
            for(int i=0; i<op_eigenvalues.size(); ++i) { // Convert back to original eigenvalues
                  if (std::abs(op_eigenvalues(i)) < std::numeric_limits<double>::epsilon() * 100) { result.all_eig(i) = std::complex<double>(std::numeric_limits<double>::infinity(), 0.0); }
                  else { result.all_eig(i) = sigma + 1.0 / op_eigenvalues(i); }
            }

            // Filter critical eigenvalues (Match solver.m thresholds)
            const double threshold_magnitude = 5e3;
            const double real_upper_bound = 300.0;
            int leading_critical_idx = -1;
            result.max_real_critical_eig = -std::numeric_limits<double>::infinity();
            result.eig_val_critical_list.clear();

            for (int i = 0; i < result.all_eig.size(); ++i) {
                std::complex<double> eig_val = result.all_eig(i);
                if (!std::isfinite(eig_val.real()) || !std::isfinite(eig_val.imag())) continue;
                if (eig_val.real() > 1e-9 && std::abs(eig_val) < threshold_magnitude && eig_val.real() < real_upper_bound) {
                    result.eig_val_critical_list.push_back(eig_val);
                    if (eig_val.real() > result.max_real_critical_eig) { result.max_real_critical_eig = eig_val.real(); leading_critical_idx = i; }
                }
            }

            if (!result.eig_val_critical_list.empty()) {
                result.flag = 1;
                std::sort(result.eig_val_critical_list.begin(), result.eig_val_critical_list.end(), CompareEigenvaluesRealDesc());
                if (find_vector && leading_critical_idx != -1) {
                    if(num_computed > 0) {
                         Eigen::MatrixXcd all_eig_vecs = geigs.eigenvectors(num_computed);
                         if (leading_critical_idx >= 0 && leading_critical_idx < all_eig_vecs.cols()) { result.leading_critical_eig_vec = all_eig_vecs.col(leading_critical_idx); }
                         else { std::cerr << "Eigen Solver Warning: Index " << leading_critical_idx << " invalid for computed eigenvectors (" << all_eig_vecs.cols() << ").\n"; }
                    } else { std::cerr << "Eigen Solver Warning: Cannot extract eigenvectors, none computed.\n"; }
                }
            } else { result.flag = 0; }
        } else { std::cerr << "Spectra Error: Computation did not converge. Info: " << static_cast<int>(geigs.info()) << "\n"; result.flag = 0; }
    } catch (const std::exception& e) { std::cerr << "Exception during eigenvalue computation: " << e.what() << "\n"; result.converged = false; result.flag = 0; }

    return result;
}