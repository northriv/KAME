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
#ifndef TIKHONOVREG_H_
#define TIKHONOVREG_H_

#include "support.h"

#include <Eigen/Core>
#include <vector>

//! Tikhonov Regularization Method
//!
//! Two solvers over one kernel.  solve() is the unconstrained, linear
//! minimiser -- what the lambda criteria are defined on, and what a covariance
//! can be written down for.  solveNonNeg() is the minimiser over x >= 0, which
//! is what a distribution of relaxation times actually is.
class DECLSPEC_KAME TikhonovRegular {
public:
    using Vector = Eigen::VectorXd;
    using Matrix = Eigen::MatrixXd;
    //! Regular(identity), Second derivartive op.
    enum class TikhonovMatrix {I = 0, D2 = 1};
    //! y = A x.
    //! \arg sv_cond_cutoff cutoff value for truncated SVD inside regular Tikhonov problem (\a matStype = I).
    TikhonovRegular(const Matrix &matrixA, TikhonovMatrix matStype = TikhonovMatrix::I, double sv_cond_cutoff = 2000.0, unsigned int max_rank = 100);
    ~TikhonovRegular() {}
    //! Criteria for lambda selection.
    //! L-curve criterion, Generalized cross validation and the known error
    //! level for <dy^2> are evaluated on the LINEAR solution, whose dependence
    //! on y they assume.  AllNonNegative is the known-error criterion evaluated
    //! on the non-negative solution instead: the lambda at which the NNLS
    //! residual meets the noise, which is the criterion native to that solver.
    enum class Method {L_Curve, MinGCV, KnownError, AllNonNegative};
    //! \arg error_sq estimated noise level squared per \a y data point.
    Vector chooseLambda(Method method, const Vector &y, double error_sq = 0.0);
    //! Prepares solve() for \a lambda: the regularized inverse A#lambda.
    //! chooseLambda() leaves solve() at whatever lambda it TRIED last, not the
    //! one it settled on -- a scan ends at its smallest -- so anything solving
    //! rows with the chosen lambda must call this first.  Does not touch lambda().
    void setLambda(double lambda);
    //! \return \a x_lambda
    Vector solve(const Vector &y) const {
        assert(y.size() == m_ylen);
        Vector ret = m_AinvReg * y; //direct return is buggy.
        assert(ret.size() == m_xlen);
        return ret;
    }
    //! The x >= 0 minimising ||A x - y||^2 + lambda^2 ||S x||^2.
    //!
    //! Amplitudes of relaxation components are populations and cannot be
    //! negative.  The unconstrained minimiser explains y with large terms of
    //! alternating sign -- neighbouring columns of an exponential kernel are
    //! nearly parallel -- and the constraint forbids that outright, so far less
    //! lambda is needed to hold the solution still and sharp features survive.
    //! Lawson-Hanson's active-set method on the normal equations, after Bro and
    //! de Jong (FNNLS): G = AtA + lambda^2 StS is formed once per lambda and
    //! shared by every right-hand side, and each step solves a subsystem of it.
    //! \arg warm a previous solution for this y or a neighbour's; its support
    //! seeds the active set, and a map that changes little between records
    //! converges in a step or two.
    Vector solveNonNeg(const Vector &y, double lambda, const Vector *warm = nullptr);
    double xlen() const {return m_xlen;}
    double ylen() const {return m_ylen;}
    //! The regularization parameter chooseLambda() settled on.
    double lambda() const {return m_lambda;}
    //! \return ||A x - y||^2, what \a x leaves unexplained of \a y.  Against
    //! the known noise level it says whether the choice of lambda has fitted
    //! the data, the noise, or neither.
    double residualSq(const Vector &y, const Vector &x) const {
        return (m_A * x - y).squaredNorm();
    }
private:
    long m_xlen, m_ylen;
    Matrix m_A;
    TikhonovMatrix m_matStype;
    Matrix m_UT, m_V;
    Vector m_sigma; //SVD solutions during regular problem.
    Matrix m_S, m_ATA, m_STS; //during general problem.
    Matrix m_AinvReg; //!< regularized inverse, A#lambda = (AtA + lambda^2 StS)^-1 At
    double m_lambda;
    double m_sv_cutoff;
    Matrix m_G; //!< AtA + lambda^2 StS for m_lambdaG.  \sa solveNonNeg()
    double m_lambdaG = -1.0;
    void prepareNonNeg_(double lambda);
    //! Unconstrained minimiser over the columns marked in \a inP, zero elsewhere.
    Vector solveOnSupport_(const Vector &h, const std::vector<char> &inP) const;
    //\return true if larger lambda is preferable for bi-sect search, true if best so far.
    bool testLambda(double lambda, Method method, const Vector &y, Vector &vec_x, double &index, double error_sq, double lambda_prev, double &xi_prev);
};

#endif /*TIKHONOV_H_*/
