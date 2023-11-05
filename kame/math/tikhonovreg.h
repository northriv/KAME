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
#ifndef TIKHONOVREG_H_
#define TIKHONOVREG_H_

#include "support.h"

#include <Eigen/Core>

//! Tikhonov Regularization Method
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
    //! L-curve criterion, Genererized cross validation, Known error level for <dy^2>, All Non-negative values for x.
    enum class Method {L_Curve, MinGCV, KnownError, AllNonNegative};
    //! \arg error_sq estimated noise level squared per \a y data point.
    Vector chooseLambda(Method method, const Vector &y, double error_sq = 0.0);
    //! \return \a x_lambda
    Vector solve(const Vector &y) const {
        assert(y.size() == m_ylen);
        Vector ret = m_AinvReg * y; //direct return is buggy.
        assert(ret.size() == m_xlen);
        return ret;
    }
    double xlen() const {return m_xlen;}
    double ylen() const {return m_ylen;}
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
    //\return true if larger lambda is preferable for bi-sect search, true if best so far.
    bool testLambda(double lambda, Method method, const Vector &y, Vector &vec_x, double &index, double error_sq, double lambda_prev, double &xi_prev);
};

#endif /*TIKHONOV_H_*/
