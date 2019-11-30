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
    //! prepares SVD from a matrix.
    //! y = A x.
    //! \arg sv_cond_cutoff cutoff value for truncated SVD.
    TikhonovRegular(const Matrix &matrixA, double sv_cond_cutoff = 2000.0);
    ~TikhonovRegular() {}
    //! L-curve criterion, Genererized cross validation, Known error level.
    enum class Method {L_Curve, MinGCV, KnownError};
    //! \arg error_sq estimated noise level squared per \a y data point.
    Vector chooseLambda(Method method, const Vector &y, double error_sq = 0.0);
    //! \return \a x_lambda
    Vector solve(const Vector &y) const {
        assert(y.size() == m_ylen);
        return m_AinvReg * y;
    }
    double xlen() const {return m_xlen;}
    double ylen() const {return m_ylen;}
private:
    long m_xlen, m_ylen;
    Matrix m_A, m_UT, m_V, m_sigma;
    Matrix m_AinvReg; //!< regularized inverse, A#lambda = (AtA + lambda^2 I)^-1 At
    double m_lambda;
    double m_sv_cutoff;
};

#endif /*TIKHONOV_H_*/
