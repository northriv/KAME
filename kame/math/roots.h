/***************************************************************************
		Copyright (C) 2002-2007 Kentaro Kitagawa
		                   kitag@issp.u-tokyo.ac.jp
		
		This program is free software; you can redistribute it and/or
		modify it under the terms of the GNU Library General Public
		License as published by the Free Software Foundation; either
		version 2 of the License, or (at your option) any later version.
		
		You should have received a copy of the GNU Library General 
		Public License and a list of authors along with this program; 
		see the files COPYING and AUTHORS.
***************************************************************************/
#ifndef ROOTS_H_
#define ROOTS_H_

#include "support.h"
#include <vector>
#include <complex>

//! Durand-Kerner-Aberth method.
//! \arg polynominal coeff. a_i, where a_n has to be 1. f(x) = a_n x^n + a_n-1 x^n-1 + ... + a^0.
//! \arg roots Pass initial values. The roots will be returned.
//! \arg eps Errors.
//! \arg max_it Max. num. of iterations.
//! \return Error.
double rootsDKA(const std::vector<std::complex<double> > &polynominal, std::vector<std::complex<double> > &roots,
	double eps, int max_it);

#endif /*ROOTS_H_*/
