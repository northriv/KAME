/***************************************************************************
        Copyright (C) 2002-2019 Kentaro Kitagawa
		                   kitag@issp.u-tokyo.ac.jp
		
		This program is free software; you can redistribute it and/or
		modify it under the terms of the GNU General Public
		License as published by the Free Software Foundation; either
		version 2 of the License, or (at your option) any later version.
		
		You should have received a copy of the GNU General 
		Public License and a list of authors along with this program; 
		see the files COPYING and AUTHORS.
***************************************************************************/
#include "tikhonovreg.h"

#include <Eigen/LU>
#include <Eigen/SVD>

TikhonovRegular::TikhonovRegular(const Matrix &matrixA, TikhonovMatrix matStype, double sv_cond_cutoff, unsigned int max_rank) {
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
    rank = std::min(rank, (long)max_rank);
    Vector sigma = m_sigma.topRows(rank);
    m_sigma = sigma; //do not directly subst.
    m_sv_cutoff = cutoff;
    m_ATA = matrixA.transpose() * matrixA; //both solvers' normal equations
    switch(matStype) {
    case TikhonovMatrix::I: {
        m_V = svd.matrixV().leftCols(rank);
        m_UT = svd.matrixU().leftCols(rank).transpose();
        m_AinvReg = m_V * Eigen::VectorXd(1.0 / m_sigma.array()).asDiagonal() * m_UT;
        dbgPrint(formatString("Tikhonov: rank=%ld, sigma_max=%.3g, sigma_min=%.3g",
            rank, sigma_max, m_sigma.minCoeff()));
        }
        break;
    case TikhonovMatrix::D2: {
        Eigen::MatrixXd matS = Eigen::MatrixXd::Identity(m_xlen, m_xlen);
        for(int i = 0; i < matS.cols() - 1; ++i)
            matS.col(i) -= matS.col(i + 1);
        //eval(): adding a matrix's own transpose in place is aliasing, which
        //Eigen asserts on in a debug build.  Release builds gave the intended
        //matrix only by the luck of column-major traversal over this sparsity.
        matS += matS.transpose().eval();
        matS *= 0.5; //[1 -0.5 0...0; -0.5 1 -0.5 0...0;....
        m_S = matS;
        m_STS = matS.transpose() * matS;
        m_AinvReg = m_ATA.inverse() * m_A.transpose();
        }
        break;
    }

}

void
TikhonovRegular::setLambda(double lambda) {
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
}

bool
TikhonovRegular::testLambda(double lambda, Method method, const Vector &vec_y, Vector &vec_x, double &index, double error_sq, double lambda_prev, double &xi_prev) {
    if(method == Method::AllNonNegative) {
        //The discrepancy principle on the constrained solution.  Its residual
        //is nondecreasing in lambda -- the feasible set is fixed and the
        //penalty only grows -- so the bisection below applies unchanged.
        vec_x = solveNonNeg(vec_y, lambda);
        double dy_sqnorm = (m_A * vec_x - vec_y).squaredNorm();
        dbgPrint(formatString("Tikhonov: nnls dy_sqnorm=%.3g, lambda=%.3g", dy_sqnorm, lambda));
        return dy_sqnorm / m_ylen < error_sq;
    }
    setLambda(lambda);
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
        dbgPrint(formatString("Tikhonov: kappa=%.3g, lambda=%.3g", kappa, lambda));
        bool ret = (index < kappa);
        index = kappa;
        return ret;
        }
    case Method::MinGCV:
        {
        double gcv = dy.squaredNorm() / pow((Eigen::MatrixXd::Identity(m_ylen, m_ylen) - (m_A * m_AinvReg)).trace(), 2.0);
        dbgPrint(formatString("Tikhonov: gcv=%.3g, lambda=%.3g", gcv, lambda));
        bool ret = (index > gcv);
        index = gcv;
        return ret;
        }
    case Method::KnownError: {
        double dy_sqnorm = dy.squaredNorm();
        dbgPrint(formatString("Tikhonov: dy_sqnorm=%.3g, lambda=%.3g", dy_sqnorm, lambda));
        return dy_sqnorm / m_ylen < error_sq;
        }
    case Method::AllNonNegative:
        break; //handled above; the linear solution has no say in it
    }
    // Exhaustive over Method, but GCC still warns "control reaches end of
    // non-void function" and at -O2 treats the path as unreachable; an
    // out-of-range Method (odmr2danalysis.cpp casts an int combo index) would
    // then fall off the end.  clang does not warn.
    return false;
}

void
TikhonovRegular::prepareNonNeg_(double lambda) {
    if((m_G.rows() == m_xlen) && (m_lambdaG == lambda))
        return;
    m_G = m_ATA;
    if(m_matStype == TikhonovMatrix::I)
        m_G.diagonal().array() += lambda * lambda;
    else
        m_G += lambda * lambda * m_STS;
    //lambda = 0 leaves AtA alone, whose small eigenvalues are what made the
    //problem ill-posed; a whisper on the diagonal keeps every subsystem solvable.
    m_G.diagonal().array() += 1e-12 * m_G.trace() / m_xlen;
    m_lambdaG = lambda;
}

TikhonovRegular::Vector
TikhonovRegular::solveOnSupport_(const Vector &h, const std::vector<char> &inP) const {
    std::vector<long> idx;
    for(long j = 0; j < m_xlen; ++j)
        if(inP[j])
            idx.push_back(j);
    Vector z = Vector::Zero(m_xlen);
    if(idx.empty())
        return z;
    long k = (long)idx.size();
    Matrix Gpp(k, k);
    Vector hp(k);
    for(long a = 0; a < k; ++a) {
        hp[a] = h[idx[a]];
        for(long b = 0; b < k; ++b)
            Gpp(a, b) = m_G(idx[a], idx[b]);
    }
    Vector zp = Gpp.ldlt().solve(hp); //symmetric, positive (semi)definite
    for(long a = 0; a < k; ++a)
        z[idx[a]] = zp[a];
    return z;
}

TikhonovRegular::Vector
TikhonovRegular::solveNonNeg(const Vector &y, double lambda, const Vector *warm) {
    assert(y.size() == m_ylen);
    prepareNonNeg_(lambda);
    const long n = m_xlen;
    Vector h = m_A.transpose() * y;
    Vector x = Vector::Zero(n);
    double hmax = h.cwiseAbs().maxCoeff();
    if( !(hmax > 0.0))
        return x; //nothing to explain
    const double tol = 1e-10 * hmax;
    std::vector<char> inP(n, 0); //the passive set P: columns free to be positive

    //Lawson-Hanson's inner loop.  z minimises over the current support but may
    //have gone negative somewhere; move from x towards it as far as the
    //constraint allows, drop from the support whatever was run into, and
    //re-minimise, until the support's own minimiser is feasible.
    auto settle = [&](Vector z) {
        for(long guard = 0; guard <= n; ++guard) {
            double alpha = 1.0;
            long block = -1;
            for(long k = 0; k < n; ++k) {
                if( !inP[k] || (z[k] > 0.0))
                    continue;
                double d = x[k] - z[k];
                double a = (d > 0.0) ? x[k] / d : 0.0; //x_k >= 0 >= z_k: 0 <= a <= 1
                if(a < alpha) {
                    alpha = a;
                    block = k;
                }
            }
            if(block < 0) {
                x = z;
                return;
            }
            x += alpha * (z - x);
            x[block] = 0.0;
            inP[block] = 0;
            for(long k = 0; k < n; ++k)
                if(inP[k] && !(x[k] > 0.0)) {
                    x[k] = 0.0;
                    inP[k] = 0;
                }
            z = solveOnSupport_(h, inP);
        }
        x = z.cwiseMax(0.0); //guard exhausted, which the theory says cannot happen
    };

    if(warm && (warm->size() == n)) {
        for(long j = 0; j < n; ++j)
            inP[j] = ((*warm)[j] > 0.0);
        settle(solveOnSupport_(h, inP));
    }
    //w = -gradient/2: how much each column still correlates with what is
    //unexplained.  A column outside the support with w > 0 would lower the
    //residual if let in; when none would, the KKT conditions hold and x is it.
    Vector w = h - m_G * x;
    for(long iter = 0; iter < 3 * n; ++iter) {
        long jmax = -1;
        double wmax = tol;
        for(long j = 0; j < n; ++j)
            if( !inP[j] && (w[j] > wmax)) {
                wmax = w[j];
                jmax = j;
            }
        if(jmax < 0)
            break;
        inP[jmax] = 1;
        settle(solveOnSupport_(h, inP));
        w = h - m_G * x;
    }
    return x;
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

    //One line per trial of lambda, and the search runs every record: a scan
    //(L-curve, GCV) spends ~110 of them.  The chosen value is not lost with
    //them -- it is put on the graph itself, where the measurement can be read
    //against it (\sa lambda(), drawRelaxDensityMap()).
    dbgPrint(formatString("Tikhonov: lambda = %g", m_lambda));
    return vec_x;
}

