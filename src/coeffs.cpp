#include "coeffs.h"
#include "utils.h"   
#include <cmath>
#include <stdexcept>
#include <algorithm> // for std::max

// Calculate variable coefficients including mean flow
VariableCoeffs calculate_coeffs( // Renamed function
                     double delta, double theta, double m,
                     const Eigen::MatrixXd& Y_grid,
                     const Eigen::MatrixXd& Z_grid,
                     int B_profile,
                     double els)
{
    if (Y_grid.rows() != Z_grid.rows() || Y_grid.cols() != Z_grid.cols()) throw std::runtime_error("Y/Z grid dimensions mismatch.");
    if (Y_grid.size() == 0) throw std::runtime_error("Input grids empty.");

    int nz_len = Z_grid.rows(); int ny_len = Y_grid.cols(); int N_tot = nz_len * ny_len;
    Eigen::MatrixXd Bz_grid, Bzp_grid;
    Bz_grid.resize(nz_len, ny_len); Bzp_grid.resize(nz_len, ny_len);
    Eigen::MatrixXd Z2 = Z_grid.array().square();

    // Calculate Bz, Bzp based on profile
    if (B_profile == 1) {
        if (delta <= 0) throw std::runtime_error("Delta must be positive");
        Bz_grid = Z_grid.array() * (-Z2.array() / (delta * delta)).exp();
        double mag = Bz_grid.maxCoeff();
        if (std::abs(mag) > 1e-15) Bz_grid /= mag; else Bz_grid.setZero();
        Bzp_grid = ((-Z2.array() / (delta * delta)).exp() * (1.0 - 2.0 * Z2.array() / (delta * delta)));
        if (std::abs(mag) > 1e-15) Bzp_grid /= mag; else Bzp_grid.setZero();

    } else if (B_profile == 2) {
        if (delta <= 0) throw std::runtime_error("Delta must be positive");
        Bz_grid = 0.15 + 0.85 * (-Z2.array() / (2.0 * delta * delta)).exp();
        double mag = Bz_grid.maxCoeff();
        if (std::abs(mag) > 1e-15) Bz_grid /= mag; else Bz_grid.setZero();
        Bzp_grid = -0.85 * Z_grid.array() / (delta * delta) * (-Z2.array() / (2.0 * delta * delta)).exp();
        if (std::abs(mag) > 1e-15) Bzp_grid /= mag; else Bzp_grid.setZero();

    } else if (B_profile == 3) {
        Bz_grid = Z_grid.array() * (1.0 - Z2.array());
        double mag = Bz_grid.maxCoeff();
        if (std::abs(mag) > 1e-15) Bz_grid /= mag; else Bz_grid.setZero();
        Bzp_grid = 1.0 - 3.0 * Z2.array();
        if (std::abs(mag) > 1e-15) Bzp_grid /= mag; else Bzp_grid.setZero();

    } else if (B_profile == 4) {
        Bz_grid = Eigen::MatrixXd::Ones(nz_len, ny_len);
        Bzp_grid = Eigen::MatrixXd::Zero(nz_len, ny_len);
    } else { throw std::runtime_error("Invalid B_profile specified."); }

    // Calculate rho0 and other density/geometry related coeffs
    Eigen::MatrixXd base_rho = 1.0 + theta * Y_grid.array();
    for(int i=0; i<base_rho.size(); ++i) 
    if (base_rho(i) <= 1e-12 && std::floor(m) != m) 
    base_rho(i) = 1e-12; // Clamp base

    Eigen::MatrixXd rho0_grid = base_rho.array().pow(m);
    Eigen::MatrixXd inv_rho0_grid = rho0_grid.array().unaryExpr([](double v) {
        if (std::abs(v) > 1e-15) {
            return 1.0 / v;
        } else {
            return 0.0;
        }
    });
    Eigen::MatrixXd c1_grid_denom = 1.0 + theta * Y_grid.array();
    Eigen::MatrixXd c1_grid = c1_grid_denom.array().unaryExpr([](double v) {
        if (std::abs(v) > 1e-15) {
            return 1.0 / v;
        } else {
            return 0.0;
        }
    });

    Eigen::MatrixXd c2_grid = c1_grid.array().square();
    Eigen::MatrixXd c3_grid = Bz_grid.array() * inv_rho0_grid.array();
    Eigen::MatrixXd c4_grid = Bzp_grid.array() * inv_rho0_grid.array();
    Eigen::MatrixXd c5_grid = Bz_grid;
    Eigen::MatrixXd c6_grid = Bzp_grid;
    Eigen::MatrixXd c7_grid = base_rho.array().pow(-(m + 1.0));
    Eigen::MatrixXd c8_grid = inv_rho0_grid;

    // Calculate mean flow terms ux0, uz0
    Eigen::MatrixXd ux0_grid, uz0_grid;
    if (B_profile == 1 && std::abs(els) > 1e-15) {
        ux0_grid = -els * m * theta / 2.0 * c7_grid.array() * Z2.array() * (-2.0 * Z2.array()).exp(); // Corrected using c7
        uz0_grid = Eigen::MatrixXd::Zero(nz_len, ny_len);
    } else {
        ux0_grid = Eigen::MatrixXd::Zero(nz_len, ny_len);
        uz0_grid = Eigen::MatrixXd::Zero(nz_len, ny_len);
    }

    // Flatten and create sparse diagonal matrices
    Eigen::Map<const Eigen::VectorXd> c1_vec(c1_grid.data(), N_tot); 
    Eigen::Map<const Eigen::VectorXd> c2_vec(c2_grid.data(), N_tot);
    Eigen::Map<const Eigen::VectorXd> c3_vec(c3_grid.data(), N_tot); 
    Eigen::Map<const Eigen::VectorXd> c4_vec(c4_grid.data(), N_tot);
    Eigen::Map<const Eigen::VectorXd> c5_vec(c5_grid.data(), N_tot); 
    Eigen::Map<const Eigen::VectorXd> c6_vec(c6_grid.data(), N_tot);
    Eigen::Map<const Eigen::VectorXd> c7_vec(c7_grid.data(), N_tot); 
    Eigen::Map<const Eigen::VectorXd> c8_vec(c8_grid.data(), N_tot);
    Eigen::Map<const Eigen::VectorXd> ux0_vec(ux0_grid.data(), N_tot); 
    Eigen::Map<const Eigen::VectorXd> uz0_vec(uz0_grid.data(), N_tot);

    VariableCoeffs coeffs_out;
    coeffs_out.c1 = sparse_diag(c1_vec); 
    coeffs_out.c2 = sparse_diag(c2_vec);
    coeffs_out.c3 = sparse_diag(c3_vec); 
    coeffs_out.c4 = sparse_diag(c4_vec);
    coeffs_out.c5 = sparse_diag(c5_vec); 
    coeffs_out.c6 = sparse_diag(c6_vec);
    coeffs_out.c7 = sparse_diag(c7_vec); 
    coeffs_out.c8 = sparse_diag(c8_vec);
    coeffs_out.ux0 = sparse_diag(ux0_vec); 
    coeffs_out.uz0 = sparse_diag(uz0_vec);

    return coeffs_out;
}