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
#ifndef MATRIX_H_
#define MATRIX_H_
//---------------------------------------------------------------------------
#include "support.h"

#include <vector>
#include <complex>
#include <boost/numeric/ublas/matrix.hpp>
#include <boost/numeric/ublas/triangular.hpp>
#include <boost/numeric/ublas/symmetric.hpp>
using namespace boost::numeric::ublas;

void householderQR(const symmetric_matrix<double> &a,
	matrix<double> &q, triangular_matrix<double, upper> &r);
void modifiedGramSchmidt(const matrix<std::complex<double> > &a,
	matrix<std::complex<double> > &q, triangular_matrix<std::complex<double>, upper> &r);

void eigHermiteRRR(const matrix<std::complex<double> > &a,
	vector<double> &lambda, matrix<std::complex<double> > &v,
	double tol);

#endif /*MATRIX_H_*/
