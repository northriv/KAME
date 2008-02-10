/***************************************************************************
		Copyright (C) 2002-2008 Kentaro Kitagawa
		                   kitag@issp.u-tokyo.ac.jp
		
		This program is free software; you can redistribute it and/or
		modify it under the terms of the GNU Library General Public
		License as published by the Free Software Foundation; either
		version 2 of the License, or (at your option) any later version.
		
		You should have received a copy of the GNU Library General 
		Public License and a list of authors along with this program; 
		see the files COPYING and AUTHORS.
***************************************************************************/
#ifndef FREQEST_H_
#define FREQEST_H_

#include "spectrumsolver.h"

//! Base class for MUSIC and EIG.
class FreqEstimation : public SpectrumSolver {
public:
	FreqEstimation(tfuncIC ic, bool eigenvalue_method, bool mvdl) : 
		SpectrumSolver(), m_eigenvalue_method(eigenvalue_method), m_mvdl_method(mvdl), m_funcIC(ic) {}
protected:
	virtual bool genSpectrum(const std::vector<std::complex<double> >& memin,
		std::vector<std::complex<double> >& memout,
		int t0, double tol, FFT::twindowfunc windowfunc, double windowlength);
	const bool m_eigenvalue_method;
	const bool m_mvdl_method;
	const tfuncIC m_funcIC;
};

//! MUltiple SIgnal Classification.
class MUSIC : public FreqEstimation {
public:
	MUSIC(tfuncIC ic = &icMDL) : FreqEstimation(ic, false, false) {}
protected:
};

class EigenVectorMethod : public FreqEstimation {
public:
	EigenVectorMethod(tfuncIC ic = &icMDL) : FreqEstimation(ic, true, false) {}
protected:
};

//! Capon MLM / MVDL.
class MVDL : public FreqEstimation {
public:
	MVDL() : FreqEstimation(&SpectrumSolver::icAIC, true, true) {}
protected:
};

#endif /*FREQEST_H_*/
