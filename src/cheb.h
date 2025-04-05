#ifndef CHEB_H
#define CHEB_H

#include <Eigen/Dense>

// Compute Chebyshev differentiation matrix D and grid points x for N+1 points.
void cheb(int N, Eigen::MatrixXd& D, Eigen::VectorXd& x);

#endif // CHEB_H