#include "cheb.h"
#include <cmath>
#include <stdexcept>
#include <limits>

// Compute Chebyshev differentiation matrix D and grid points x
void cheb(int N, Eigen::MatrixXd& D, Eigen::VectorXd& x) {
    // ... (implementation identical to last working cheb_collocation.cpp) ...
    if (N < 0) throw std::runtime_error("Chebyshev N cannot be negative.");
    if (N == 0) { /* ... */ return; }
    int Np1 = N + 1;
    x.resize(Np1); D.resize(Np1, Np1);
    for (int i = 0; i <= N; ++i) {
        #ifndef M_PI
        #define M_PI 3.14159265358979323846
        #endif
        x(i) = std::cos(M_PI * static_cast<double>(i) / N);
    }
    Eigen::VectorXd c(Np1); c(0)=2.0; c(N)=2.0;
    for(int i=1; i<N; ++i) c(i)=1.0;
    for(int i=0; i<=N; ++i) if(i%2!=0) c(i)*=-1.0;
    Eigen::MatrixXd X = x.replicate(1, Np1);
    Eigen::MatrixXd dX = X - X.transpose();
    Eigen::VectorXd c_inv = c.array().inverse();
    Eigen::MatrixXd C_outer = c * c_inv.transpose();
    Eigen::MatrixXd denominator = dX + Eigen::MatrixXd::Identity(Np1, Np1);
    D = Eigen::MatrixXd::Zero(Np1, Np1);
     for(int i=0; i<Np1; ++i) {
         for(int j=0; j<Np1; ++j) {
             if (i != j) {
                  if (std::abs(denominator(i,j)) > std::numeric_limits<double>::epsilon()) {
                       D(i,j) = C_outer(i,j) / denominator(i,j);
                  } else { D(i,j) = 0; }
             }
         }
     }
    D -= D.rowwise().sum().asDiagonal();
}