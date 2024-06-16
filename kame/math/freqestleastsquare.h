/***************************************************************************
		Copyright (C) 2002-2015 Kentaro Kitagawa
		                   kitag@issp.u-tokyo.ac.jp
		
		This program is free software; you can redistribute it and/or
		modify it under the terms of the GNU Library General Public
		License as published by the Free Software Foundation; either
		version 2 of the License, or (at your option) any later version.
		
		You should have received a copy of the GNU Library General 
		Public License and a list of authors along with this program; 
		see the files COPYING and AUTHORS.
***************************************************************************/
#ifndef FREQESTLEASTSQUARE_H_
#define FREQESTLEASTSQUARE_H_

#include "spectrumsolver.h"

//! Frequency estimation by least square fit.
//! Number of signals is determined by information criterion.
class DECLSPEC_KAME FreqEstLeastSquare : public SpectrumSolver {
public:
	FreqEstLeastSquare(tfuncIC ic) : 
		SpectrumSolver(), m_funcIC(ic) {}
protected:
	virtual void genSpectrum(const std::vector<std::complex<double> >& memin,
		std::vector<std::complex<double> >& memout,
		int t0, double tol, FFT::twindowfunc windowfunc, double windowlength);
	virtual bool hasWeighting() const {return true;}
private:
	const tfuncIC m_funcIC;
};

#include <Eigen/Core>

//! Fourier-series expansion solved by truncated SVD.
class TSVDFourierSeries : public SpectrumSolver {
public:
protected:
    virtual void genSpectrum(const std::vector<std::complex<double> >& memin,
        std::vector<std::complex<double> >& memout,
        int t0, double tol, FFT::twindowfunc windowfunc, double windowlength);
private:
    long m_xlen = 0, m_ylen = 0;
    Eigen::MatrixXcd m_AinvReg; //SVD solutions.
};

#endif /*FREQESTLEASTSQUARE_H_*/
