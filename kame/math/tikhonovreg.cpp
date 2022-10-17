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

#include <Eigen/LU>
#include <Eigen/SVD>

TikhonovRegular::TikhonovRegular(const Matrix &matrixA, TikhonovMatrix matStype, double sv_cond_cutoff) {
    m_A = matrixA;
    m_matStype = matStype;
    m_xlen = matrixA.cols();
    m_ylen = matrixA.rows();
    auto svd = Eigen::BDCSVD<Matrix>(matrixA, Eigen::ComputeFullU | Eigen::ComputeFullV);
    m_sigma = svd.singularValues();
    double sigma_max = m_sigma.maxCoeff();
    double cutoff = sigma_max / sv_cond_cutoff;
    long rank = (m_sigma.array() > cutoff).count();
    rank = std::max(rank, 2L);
    m_sigma = m_sigma.topRows(rank);
    m_sv_cutoff = cutoff;
    switch(matStype) {
    case TikhonovMatrix::I: {
        m_V = svd.matrixV().leftCols(rank);
        m_UT = svd.matrixU().leftCols(rank).transpose();
        m_AinvReg = m_V * Eigen::VectorXd(1.0 / m_sigma.array()).asDiagonal() * m_UT;
        fprintf(stderr, "Rank=%ld, sigma_max=%.3g, sigma_min=%.3g\n", rank, sigma_max, m_sigma.minCoeff());
        }
        break;
    case TikhonovMatrix::D2: {
        Eigen::MatrixXd matS = Eigen::MatrixXd::Identity(m_xlen, m_xlen);
        for(int i = 0; i < matS.cols() - 1; ++i)
            matS.col(i) -= matS.col(i + 1);
        matS += matS.transpose();
        matS *= 0.5; //[1 -0.5 0...0; -0.5 1 -0.5 0...0;....
        m_S = matS;
        m_STS = matS.transpose() * matS;
        m_ATA = matrixA.transpose() * matrixA;
        m_AinvReg = m_ATA.inverse() * m_A.transpose();
        }
        break;
    }

}

bool
TikhonovRegular::testLambda(double lambda, Method method, const Vector &vec_y, Vector &vec_x, double &index, double error_sq, double lambda_prev, double &xi_prev) {
    switch(m_matStype) {
    case TikhonovMatrix::I: {
        auto slambda = Eigen::VectorXd((m_sigma.array() / (m_sigma.array().square() + lambda*lambda)));
//        m_AinvReg = m_V * slambda.asDiagonal() * m_UT; //speed limiting line
        m_AinvReg = m_V;
        for(int i = 0; i < m_sigma.size(); ++i)
            m_AinvReg.col(i) *= slambda.coeff(i);
        m_AinvReg *= m_UT;
        }
        break;
    case TikhonovMatrix::D2: {
        auto m = m_ATA + lambda*lambda * m_STS;
        Eigen::PartialPivLU<Eigen::MatrixXd> lu(m);
        m_AinvReg = lu.inverse() * m_A.transpose();
        }
        break;
    }

    vec_x = m_AinvReg * vec_y;
    auto dy = m_A * vec_x - vec_y;
    switch (method) {
    case Method::L_Curve:
        {
        double rho = dy.squaredNorm();
        double xi;
        if(m_matStype == TikhonovMatrix::I)
            xi = vec_x.squaredNorm();
        else
            xi = (m_S * vec_x).squaredNorm();
        double dxi_dl = (xi - xi_prev) / (lambda - lambda_prev);
        xi_prev = xi;
        //curvature
        double kappa = 2 * xi*rho/dxi_dl* (pow(lambda,2)*dxi_dl*rho+2*lambda*xi*rho+pow(lambda,4)*xi*dxi_dl)
            / pow(pow(lambda,4)*xi*xi+rho*rho, 1.5);
        fprintf(stderr, "kappa=%.3g, lambda=%.3g; ", kappa, lambda);
        bool ret = (index < kappa);
        index = kappa;
        return ret;
        }
    case Method::MinGCV:
        {
        double gcv = dy.squaredNorm() / pow((Eigen::MatrixXd::Identity(m_ylen, m_ylen) - (m_A * m_AinvReg)).trace(), 2.0);
        fprintf(stderr, "gcv=%.3g, lambda=%.3g; ", gcv, lambda);
        bool ret = (index > gcv);
        index = gcv;
        return ret;
        }
    case Method::KnownError: {
        double dy_sqnorm = dy.squaredNorm();
        fprintf(stderr, "dy_sqnorm=%.3g, lambda=%.3g; ", dy_sqnorm, lambda);
        return dy_sqnorm / m_ylen < error_sq;
        }
    case Method::AllNonNegative:
        fprintf(stderr, "min x=%.3g, lambda=%.3g; ", vec_x.minCoeff(), lambda);
        return (vec_x.array() < 0.0).any();
    }
}

TikhonovRegular::Vector
TikhonovRegular::chooseLambda(Method method, const Vector &vec_y, double error_sq) {
    assert(vec_y.size() == m_ylen);
    Vector vec_x(m_xlen);
    Eigen::VectorXd vec_x_lambda = vec_x;
    double xi_prev = 0.0;
    double index_best = 0.0;

    double lambda_max = m_sigma.maxCoeff() * 0.1;
    if(m_matStype == TikhonovMatrix::D2)
        lambda_max *= 10000.0;
    switch (method) {
    case Method::L_Curve:
    case Method::MinGCV: {
        double lambda_prev = 0.0;
        //serach by lambda reduction
        int cnt = 0;
        for(double lambda = lambda_max; lambda > lambda_max * 0.00001; lambda *= 0.9) {
            double index = index_best;
            if(testLambda(lambda, method, vec_y, vec_x_lambda, index, error_sq, lambda_prev, xi_prev) || (cnt < 2)) {
                index_best = index;
                vec_x = vec_x_lambda;
                m_lambda = lambda;
            }
            cnt++;
            lambda_prev = lambda;
        }
        }
        break;
    case Method::KnownError:
    case Method::AllNonNegative:
        //bisection algorithm for determination.
        double lambda_min = 0.0;
        double thres = lambda_max * 1e-4;
        bool firsttime = true;
        for(;lambda_max - lambda_min > thres;) {
            double lambda = (lambda_max + lambda_min) / 2;
            if(testLambda(lambda, method, vec_y, vec_x_lambda, index_best, error_sq, 0.0, xi_prev) && !firsttime) {
                lambda_min = lambda;
            }
            else {
                lambda_max = lambda;
                vec_x = vec_x_lambda;
                m_lambda = lambda;
            }
            firsttime = false;
        }
        break;
    }

    fprintf(stderr, "lambda = %g\n", m_lambda);
    return vec_x;
}

