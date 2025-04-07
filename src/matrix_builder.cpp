#include "matrix_builder.h" // Use new header name
#include <vector>
#include <complex>
#include <stdexcept>

void add_sparse_block(int start_row, int start_col,
                      const Eigen::SparseMatrix<double>& block,
                      std::vector<Eigen::Triplet<double>>& triplets)
{
    if (block.rows() == 0 || block.cols() == 0) return;
    for (int k = 0; k < block.outerSize(); ++k) {
        for (Eigen::SparseMatrix<double>::InnerIterator it(block, k); it; ++it) {
             if (std::abs(it.value()) > 1e-15) {
                triplets.emplace_back(start_row + it.row(), start_col + it.col(), it.value());
             }
        }
    }
}

void build_matrix(
    double kx, double Ek, double Pr, double Pm, double els, double Ra,
    double m, double theta, int mean_flow,
    const BoundaryConditions& bc,     
    const VariableCoeffs& coeffs,    
    const DiffMatrices& diff,
    Eigen::SparseMatrix<double>& A,
    Eigen::SparseMatrix<double>& B)
{
     if (kx == 0) throw std::runtime_error("build_matrix error: kx cannot be zero.");
     if (Pr == 0 || Pm == 0) throw std::runtime_error("build_matrix error: Pr/Pm cannot be zero.");

    int N_tot = diff.I.rows();
    if (N_tot == 0) throw std::runtime_error("build_matrix error: Identity matrix I zero size.");

    bool use_elsasser = (std::abs(els) > 1e-15);
    int N_vars = use_elsasser ? 7 : 4;
    int N_sys = N_vars * N_tot;

    A.resize(N_sys, N_sys); B.resize(N_sys, N_sys);
    A.reserve(Eigen::VectorXi::Constant(N_sys, 40));
    B.reserve(Eigen::VectorXi::Constant(N_sys, 5));

    std::vector<Eigen::Triplet<double>> tripletsA;
    std::vector<Eigen::Triplet<double>> tripletsB;

    const auto& c1=coeffs.c1; 
    const auto& c2=coeffs.c2; 
    const auto& c3=coeffs.c3;
    const auto& c4=coeffs.c4; 
    const auto& c5=coeffs.c5; 
    const auto& c6=coeffs.c6;
    const auto& c7=coeffs.c7; 
    const auto& c8=coeffs.c8;
    const auto& ux0=coeffs.ux0; 
    const auto& uz0=coeffs.uz0;
    const auto& I=diff.I;
    const auto& I2=diff.I2;
    const auto& Dy=diff.Dy; 
    const auto& Dz=diff.Dz;
    const auto& D2=diff.D2;

    std::vector<int> r_start(N_vars), c_start(N_vars);
    for(int i=0; i<N_vars; ++i) { r_start[i]=i*N_tot; c_start[i]=i*N_tot; }

    double cp = 1.0; double sp = 0.0; 

    if (mean_flow == 1) {
        if (!use_elsasser) {
            // A Matrix 
            add_sparse_block(r_start[0], c_start[0], I2*(-cp*I), tripletsA);
            add_sparse_block(r_start[0], c_start[0], I2*(Ek/kx*Dy*D2 + Ek*m*theta/kx*Dy*c1*Dy), tripletsA);
            add_sparse_block(r_start[0], c_start[1], I2*(Ek*D2+(5.0/3.0)*Ek*m*theta*c1*Dy + Ek*(2.0*m+1.0)*m*theta*theta/3.0*c2 - (2.0/3.0)*Ek*m*theta*Dy*c1)+bc.BCUy, tripletsA);
            add_sparse_block(r_start[0], c_start[1], I2*(1.0/kx*Dy), tripletsA);
            add_sparse_block(r_start[0], c_start[2], I2*(sp/kx*Dy), tripletsA);
            add_sparse_block(r_start[0], c_start[3], I2*(Pm/Pr*Ra), tripletsA);
            add_sparse_block(r_start[1], c_start[0], I2*(sp*I), tripletsA);
            add_sparse_block(r_start[1], c_start[0], I2*(Ek/kx*Dz*D2 + Ek*m*theta/kx*Dz*c1*Dy), tripletsA);
            add_sparse_block(r_start[1], c_start[1], I2*(1.0/kx*Dz), tripletsA);
            add_sparse_block(r_start[1], c_start[2], I2*(Ek*D2 + Ek*m*theta*c1*Dy)+bc.BCUz, tripletsA);
            add_sparse_block(r_start[1], c_start[2], -I2*(sp/kx*Dz), tripletsA);
            add_sparse_block(r_start[2], c_start[0], I2*(kx*I)+bc.BCUx, tripletsA);
            add_sparse_block(r_start[2], c_start[1], I2*(Dy + m*theta*c1), tripletsA);
            add_sparse_block(r_start[2], c_start[2], I2*Dz, tripletsA);
            add_sparse_block(r_start[3], c_start[1], I2*c1, tripletsA);
            add_sparse_block(r_start[3], c_start[3], I2*(Pm/Pr*c8*D2 + Pm/Pr*theta*c7*Dy), tripletsA);
            add_sparse_block(r_start[3], c_start[3], -I2*(kx*ux0), tripletsA);
            add_sparse_block(r_start[3], c_start[3], -I2*(uz0*Dz), tripletsA);
            add_sparse_block(r_start[3], c_start[3], bc.BCS, tripletsA);
            // B Matrix
            add_sparse_block(r_start[0], c_start[0], I2*(Ek/Pm*(1.0/kx)*Dy), tripletsB);
            add_sparse_block(r_start[0], c_start[1], I2*(Ek/Pm*I), tripletsB);
            add_sparse_block(r_start[1], c_start[0], I2*(Ek/(kx*Pm)*Dz), tripletsB);
            add_sparse_block(r_start[1], c_start[2], I2*(Ek/Pm*I), tripletsB);
            add_sparse_block(r_start[3], c_start[3], I2*I, tripletsB);
        } else { 
            // A Matrix
            add_sparse_block(r_start[0],c_start[0], I2*(-I), tripletsA);
            add_sparse_block(r_start[0],c_start[0], I2*(Ek/kx*Dy*D2 + Ek*m*theta/kx*Dy*c1*Dy), tripletsA);
            add_sparse_block(r_start[0],c_start[1], I2*(Ek*D2+(5.0/3.0)*Ek*m*theta*c1*Dy + Ek*(2.0*m+1.0)*m*theta*theta/3.0*c2 - (2.0/3.0)*Ek*m*theta*Dy*c1)+bc.BCUy, tripletsA);
            add_sparse_block(r_start[0],c_start[1], I2*(1.0/kx*Dy), tripletsA);
            add_sparse_block(r_start[0],c_start[2], -I2*(sp/kx*Dy), tripletsA);
            add_sparse_block(r_start[0],c_start[3], -I2*els*c3*Dy, tripletsA);
            add_sparse_block(r_start[0],c_start[4], I2*els*c3*kx, tripletsA);
            add_sparse_block(r_start[0],c_start[5], -I2*els*(1.0/kx)*Dy*c4, tripletsA);
            add_sparse_block(r_start[0],c_start[6], I2*(Pm/Pr*Ra), tripletsA);
            add_sparse_block(r_start[1],c_start[0], I2*(sp*I), tripletsA);
            add_sparse_block(r_start[1],c_start[0], I2*(Ek/kx*Dz*D2 + Ek*m*theta/kx*Dz*c1*Dy), tripletsA);
            add_sparse_block(r_start[1],c_start[1], I2*(1.0/kx*Dz), tripletsA);
            add_sparse_block(r_start[1],c_start[2], I2*(Ek*D2 + Ek*m*theta*c1*Dy)+bc.BCUz, tripletsA);
            add_sparse_block(r_start[1],c_start[2], -I2*(sp/kx*Dz), tripletsA);
            add_sparse_block(r_start[1],c_start[3], I2*(-els*c3*Dz - els*c4), tripletsA);
            add_sparse_block(r_start[1],c_start[5], I2*(kx*els*c3 - els*(1.0/kx)*Dz*c4), tripletsA);
            add_sparse_block(r_start[2],c_start[1], -I2*c5*Dy, tripletsA);
            add_sparse_block(r_start[2],c_start[2], -I2*(c5*Dz + c6), tripletsA);
            add_sparse_block(r_start[2],c_start[3], I2*D2+bc.BCBx, tripletsA);
            add_sparse_block(r_start[2],c_start[3], -I2*uz0*Dz, tripletsA);
            add_sparse_block(r_start[2],c_start[3], -I2*kx*ux0, tripletsA);
            add_sparse_block(r_start[3],c_start[1], I2*kx*c5, tripletsA);
            add_sparse_block(r_start[3],c_start[4], I2*D2+bc.BCBy, tripletsA);
            add_sparse_block(r_start[3],c_start[4], -I2*uz0*Dz, tripletsA);
            add_sparse_block(r_start[3],c_start[4], -I2*kx*ux0, tripletsA);
            add_sparse_block(r_start[4],c_start[2], I2*kx*c5, tripletsA);
            add_sparse_block(r_start[4],c_start[5], I2*D2+bc.BCBz, tripletsA);
            add_sparse_block(r_start[4],c_start[5], -I2*uz0*Dz, tripletsA);
            add_sparse_block(r_start[4],c_start[5], -I2*kx*ux0, tripletsA);
            add_sparse_block(r_start[5],c_start[0], I2*kx*I+bc.BCUx, tripletsA);
            add_sparse_block(r_start[5],c_start[1], I2*(Dy + m*theta*c1), tripletsA);
            add_sparse_block(r_start[5],c_start[2], I2*Dz, tripletsA);
            add_sparse_block(r_start[6],c_start[1], I2*c1, tripletsA);
            add_sparse_block(r_start[6],c_start[6], I2*(Pm/Pr*c8*D2 + Pm/Pr*theta*c7*Dy)+bc.BCS, tripletsA);
            add_sparse_block(r_start[6],c_start[6], -I2*uz0*Dz, tripletsA);
            add_sparse_block(r_start[6],c_start[6], -I2*kx*ux0, tripletsA);
            // B Matrix
            add_sparse_block(r_start[0],c_start[0], I2*(Ek/Pm*(1.0/kx)*Dy), tripletsB);
            add_sparse_block(r_start[0],c_start[1], I2*(Ek/Pm*I), tripletsB);
            add_sparse_block(r_start[1],c_start[0], I2*(Ek/(kx*Pm)*Dz), tripletsB);
            add_sparse_block(r_start[1],c_start[2], I2*(Ek/Pm*I), tripletsB);
            add_sparse_block(r_start[2],c_start[3], I2*I, tripletsB);
            add_sparse_block(r_start[3],c_start[4], I2*I, tripletsB);
            add_sparse_block(r_start[4],c_start[5], I2*I, tripletsB);
            add_sparse_block(r_start[6],c_start[6], I2*I, tripletsB);
        }
    } else { 
        if (!use_elsasser) { 
             // A Matrix
            add_sparse_block(r_start[0], c_start[0], I2*(-cp*I), tripletsA);
            add_sparse_block(r_start[0], c_start[0], I2*(Ek/kx*Dy*D2 + Ek*m*theta/kx*Dy*c1*Dy), tripletsA);
            add_sparse_block(r_start[0], c_start[1], I2*(Ek*D2+(5.0/3.0)*Ek*m*theta*c1*Dy + Ek*(2.0*m+1.0)*m*theta*theta/3.0*c2 - (2.0/3.0)*Ek*m*theta*Dy*c1)+bc.BCUy, tripletsA);
            add_sparse_block(r_start[0], c_start[1], I2*(1.0/kx*Dy), tripletsA);
            add_sparse_block(r_start[0], c_start[2], I2*(sp/kx*Dy), tripletsA);
            add_sparse_block(r_start[0], c_start[3], I2*(Pm/Pr*Ra), tripletsA);
            add_sparse_block(r_start[1], c_start[0], I2*(sp*I), tripletsA);
            add_sparse_block(r_start[1], c_start[0], I2*(Ek/kx*Dz*D2 + Ek*m*theta/kx*Dz*c1*Dy), tripletsA);
            add_sparse_block(r_start[1], c_start[1], I2*(1.0/kx*Dz), tripletsA);
            add_sparse_block(r_start[1], c_start[2], I2*(Ek*D2 + Ek*m*theta*c1*Dy)+bc.BCUz, tripletsA);
            add_sparse_block(r_start[1], c_start[2], -I2*(sp/kx*Dz), tripletsA);
            add_sparse_block(r_start[2], c_start[0], I2*(kx*I)+bc.BCUx, tripletsA);
            add_sparse_block(r_start[2], c_start[1], I2*(Dy + m*theta*c1), tripletsA);
            add_sparse_block(r_start[2], c_start[2], I2*Dz, tripletsA);
            add_sparse_block(r_start[3], c_start[1], I2*c1, tripletsA);
            add_sparse_block(r_start[3], c_start[3], I2*(Pm/Pr*c8*D2 + Pm/Pr*theta*c7*Dy)+bc.BCS, tripletsA);
            // B Matrix
            add_sparse_block(r_start[0], c_start[0], I2*(Ek/Pm*(1.0/kx)*Dy), tripletsB);
            add_sparse_block(r_start[0], c_start[1], I2*(Ek/Pm*I), tripletsB);
            add_sparse_block(r_start[1], c_start[0], I2*(Ek/(kx*Pm)*Dz), tripletsB);
            add_sparse_block(r_start[1], c_start[2], I2*(Ek/Pm*I), tripletsB);
            add_sparse_block(r_start[3], c_start[3], I2*I, tripletsB);
        } else { 
             // A Matrix 
            add_sparse_block(r_start[0],c_start[0], I2*(-I), tripletsA);
            add_sparse_block(r_start[0],c_start[0], I2*(Ek/kx*Dy*D2 + Ek*m*theta/kx*Dy*c1*Dy), tripletsA);
            add_sparse_block(r_start[0],c_start[1], I2*(Ek*D2+(5.0/3.0)*Ek*m*theta*c1*Dy + Ek*(2.0*m+1.0)*m*theta*theta/3.0*c2 - (2.0/3.0)*Ek*m*theta*Dy*c1)+bc.BCUy, tripletsA);
            add_sparse_block(r_start[0],c_start[1], I2*(1.0/kx*Dy), tripletsA);
            add_sparse_block(r_start[0],c_start[2], -I2*(sp/kx*Dy), tripletsA);
            add_sparse_block(r_start[0],c_start[3], -I2*els*c3*Dy, tripletsA);
            add_sparse_block(r_start[0],c_start[4], I2*els*c3*kx, tripletsA);
            add_sparse_block(r_start[0],c_start[5], -I2*els*(1.0/kx)*Dy*c4, tripletsA);
            add_sparse_block(r_start[0],c_start[6], I2*(Pm/Pr*Ra), tripletsA);
            add_sparse_block(r_start[1],c_start[0], I2*(sp*I), tripletsA);
            add_sparse_block(r_start[1],c_start[0], I2*(Ek/kx*Dz*D2 + Ek*m*theta/kx*Dz*c1*Dy), tripletsA);
            add_sparse_block(r_start[1],c_start[1], I2*(1.0/kx*Dz), tripletsA);
            add_sparse_block(r_start[1],c_start[2], I2*(Ek*D2 + Ek*m*theta*c1*Dy)+bc.BCUz, tripletsA);
            add_sparse_block(r_start[1],c_start[2], -I2*(sp/kx*Dz), tripletsA);
            add_sparse_block(r_start[1],c_start[3], I2*(-els*c3*Dz - els*c4), tripletsA);
            add_sparse_block(r_start[1],c_start[5], I2*(kx*els*c3 - els*(1.0/kx)*Dz*c4), tripletsA);
            add_sparse_block(r_start[2],c_start[1], -I2*c5*Dy, tripletsA);
            add_sparse_block(r_start[2],c_start[2], -I2*(c5*Dz + c6), tripletsA);
            add_sparse_block(r_start[2],c_start[3], I2*D2+bc.BCBx, tripletsA); // No mean flow term
            add_sparse_block(r_start[3],c_start[1], I2*kx*c5, tripletsA);
            add_sparse_block(r_start[3],c_start[4], I2*D2+bc.BCBy, tripletsA); // No mean flow term
            add_sparse_block(r_start[4],c_start[2], I2*kx*c5, tripletsA);
            add_sparse_block(r_start[4],c_start[5], I2*D2+bc.BCBz, tripletsA); // No mean flow term
            add_sparse_block(r_start[5],c_start[0], I2*kx*I+bc.BCUx, tripletsA);
            add_sparse_block(r_start[5],c_start[1], I2*(Dy + m*theta*c1), tripletsA);
            add_sparse_block(r_start[5],c_start[2], I2*Dz, tripletsA);
            add_sparse_block(r_start[6],c_start[1], I2*c1, tripletsA);
            add_sparse_block(r_start[6],c_start[6], I2*(Pm/Pr*c8*D2 + Pm/Pr*theta*c7*Dy)+bc.BCS, tripletsA); // No mean flow term
            // B Matrix 
            add_sparse_block(r_start[0],c_start[0], I2*(Ek/Pm*(1.0/kx)*Dy), tripletsB);
            add_sparse_block(r_start[0],c_start[1], I2*(Ek/Pm*I), tripletsB);
            add_sparse_block(r_start[1],c_start[0], I2*(Ek/(kx*Pm)*Dz), tripletsB);
            add_sparse_block(r_start[1],c_start[2], I2*(Ek/Pm*I), tripletsB);
            add_sparse_block(r_start[2],c_start[3], I2*I, tripletsB);
            add_sparse_block(r_start[3],c_start[4], I2*I, tripletsB);
            add_sparse_block(r_start[4],c_start[5], I2*I, tripletsB);
            add_sparse_block(r_start[6],c_start[6], I2*I, tripletsB);
        }
    }

    // Finalize matrices
    A.setFromTriplets(tripletsA.begin(), tripletsA.end());
    B.setFromTriplets(tripletsB.begin(), tripletsB.end());
}