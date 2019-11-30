/***************************************************************************
        Copyright (C) 2002-2019 Kentaro Kitagawa
		                   kitagawa@phys.s.u-tokyo.ac.jp
		
		This program is free software; you can redistribute it and/or
		modify it under the terms of the GNU Library General Public
		License as published by the Free Software Foundation; either
		version 2 of the License, or (at your option) any later version.
		
		You should have received a copy of the GNU Library General 
		Public License and a list of authors along with this program; 
		see the files COPYING and AUTHORS.
***************************************************************************/
#include "tikhonovreg.h"

#include <Eigen/SVD>

TikhonovRegular::TikhonovRegular(const Matrix &matrixA, double sv_cond_cutoff) {
    m_A = matrixA;
    m_xlen = matrixA.cols();
    m_ylen = matrixA.rows();
    auto svd = Eigen::BDCSVD<Matrix>(matrixA, Eigen::ComputeFullU | Eigen::ComputeFullV);
    m_sigma = svd.singularValues();
    double sigma_max = m_sigma.maxCoeff();
    double cutoff = sigma_max / sv_cond_cutoff;
    long rank = (m_sigma.array() > cutoff).count();
    rank = std::max(rank, 2L);
    m_sigma = m_sigma.topRows(rank);
    m_V = svd.matrixV().leftCols(rank);
    m_UT = svd.matrixU().leftCols(rank).transpose();
    m_sv_cutoff = cutoff;
    m_AinvReg = m_V * Eigen::VectorXd(1.0 / m_sigma.array()).asDiagonal() * m_UT;
    fprintf(stderr, "Rank=%ld, sigma_max=%.3g, sigma_min=%.3g\n", rank, sigma_max, m_sigma.minCoeff());
}

TikhonovRegular::Vector
TikhonovRegular::chooseLambda(Method method, const Vector &vec_y, double error_sq) {
    assert(vec_y.size() == m_ylen);
    Vector vec_x_lambda(m_xlen);
    Eigen::VectorXd vec_x = vec_x_lambda;
    double kappa_max = -1e20;
    double gcv_min = 1e20;
    double xi_prev = 0.0;
    double lambda_prev = 0.0;
    for(double lambda = m_sigma.maxCoeff() * 1.0; lambda > m_sv_cutoff; lambda *= 0.9) {
        auto slambda = Eigen::VectorXd((m_sigma.array() / (m_sigma.array().square() + lambda*lambda)));
//        m_AinvReg = m_V * slambda.asDiagonal() * m_UT; //speed limiting line
        m_AinvReg = m_V;
        for(int i = 0; i < m_sigma.size(); ++i)
            m_AinvReg.col(i) *= slambda.coeff(i);
        m_AinvReg *= m_UT;

        vec_x_lambda = m_AinvReg * vec_y;
        auto dy = m_A * vec_x_lambda - vec_y;
        double dy_sqnorm = dy.squaredNorm();
        switch (method) {
        case Method::L_Curve:
            {
                double rho = dy_sqnorm;
                double xi = vec_x_lambda.squaredNorm();
                if(xi_prev != 0.0) {
                    double dxi_dl = (xi - xi_prev) / (lambda - lambda_prev);
                    //curvature
                    double kappa = 2 * xi*rho/dxi_dl* (pow(lambda,2)*dxi_dl*rho+2*lambda*xi*rho+pow(lambda,4)*xi*dxi_dl)
                        / pow(pow(lambda,4)*xi*xi+rho*rho, 1.5);
                    fprintf(stderr, "kappa=%.3g, lambda=%.3g; ", kappa, lambda);
                    if(kappa_max < kappa) {
                        kappa_max = kappa;
                        vec_x = vec_x_lambda;
                        m_lambda = lambda;
                    }
                }
                xi_prev = xi;
            }
            break;
        case Method::MinGCV:
            {
            double gcv = dy_sqnorm / pow((Eigen::MatrixXd::Identity(m_ylen, m_ylen) - (m_A * m_AinvReg)).trace(), 2.0);
            fprintf(stderr, "gcv=%.3g, lambda=%.3g; ", gcv, lambda);
            if(gcv_min > gcv) {
                gcv_min = gcv;
                m_lambda = lambda;
                vec_x = vec_x_lambda;
            }
            }
            break;
        case Method::KnownError:
            fprintf(stderr, "dy_sqnorm=%.3g, lambda=%.3g; ", dy_sqnorm, lambda);
            if(dy_sqnorm / m_ylen > error_sq) {
                //stores before error becomes smaller than expected.
                m_lambda = lambda;
                vec_x = vec_x_lambda;
            }
            break;
        }
    }
    fprintf(stderr, "lambda = %g\n", m_lambda);
    return vec_x_lambda;
}

