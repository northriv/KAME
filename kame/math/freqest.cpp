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
#include "freqest.h"
#include "matrix.h"
#include <numeric>
#include <boost/numeric/ublas/matrix_proxy.hpp>
#include <boost/numeric/ublas/io.hpp>

bool
FreqEstimation::genSpectrum(const std::vector<std::complex<double> >& memin,
		std::vector<std::complex<double> >& memout,
		int t0, double tol, FFT::twindowfunc /*windowfunc*/, double windowlength) {
	FFT::twindowfunc windowfunc = &FFT::windowFuncTri;
	int t = memin.size();
	int n = memout.size();
	
	double tpoworg = 0.0;
	for(int i = 0; i < t; i++) {
		tpoworg += std::norm(memin[i]);
	}

	if(t0 < 0)
		t0 += (-t0 / n + 1) * n;	
	std::vector<std::complex<double> > memphaseout(n);	
	MEMStrict::genSpectrum(memin, memphaseout, t0, tol, &FFT::windowFuncRect, 1.0);	
	
	std::vector<std::complex<double> > rx(t, 0.0);
	for(unsigned int p = 0; p <= t - 1; p++) {
		for(unsigned int i = 0; i < t - p; i++) {
			rx[p] += std::conj(memin[i]) * memin[i+p];
		}
		rx[p] *= windowfunc(0.5*p/(double)t) * t / (t - p);
	}
	matrix<std::complex<double> > r(t, t);
	for(int i = 0; i < t; i++) {
		matrix_row<matrix<std::complex<double> > > rrow(r, i);
		for(int j = i; j < t; j++) {
			rrow(j) = rx[j - i];
		}
	}
	matrix<std::complex<double> > eigv;
	vector<double> lambda;
	eigHermiteRRR(r, lambda, eigv, tol * std::abs(rx[0]));
	int p;
	double thres = lambda[0] + (lambda[t-1] - lambda[0]) * windowlength / 2;
	for(int i = t - 1; i >= 0; i--) {
		if(lambda[i] < thres) {
			p = t - 1 - i;
			break;
		}
	}
	p = std::max(p, 1);
//	std::cout << lambda << std::endl;
//	std::cout << eigv << std::endl;
	std::vector<double> ip(n, 0.0), dy(n, 0.0);
	std::vector<std::complex<double> > fftin(n, 0.0), fftout(n), fftin2(n, 0.0), fftout2(n);
	for(int i = 0; i < t - p; i++) {
		matrix_column<matrix<std::complex<double> > > eigvcol(eigv, i);
		double z = (m_eigenvalue_method) ? (sqrt(1.0 / lambda[i])) : 1.0;
		ASSERT(fabs(norm_2(eigvcol) - 1.0) < 0.1);
		for(int j = 0; j < t; j++) {
			fftin[j] = eigvcol(j) * z;
			fftin2[j] = eigvcol(j) * (double)j * std::complex<double>(0, z);
		}
		m_ifftN->exec(fftin, fftout);
		m_ifftN->exec(fftin2, fftout2);
		for(int k = 0; k < n; k++) {
			ip[k] += std::norm(fftout[k]);
			dy[k] += std::real(fftout2[k] * std::conj(fftout[k]));
		}
	}
	double tpow = 0.0;
	for(int i = 0; i < n; i++) {
		tpow += 1.0 / ip[i];
		memout[i] = memphaseout[i] * sqrt(1.0 / (ip[i] * std::norm(memphaseout[i])));
	}
	double normalize = sqrt(n * tpoworg / tpow);
	for(int i = 0; i < n; i++) {
		memout[i] *= normalize;
	}
	
	for(int i = 1; i < n; i++) {
		if((dy[i - 1] < 0) && (dy[i] > 0)) {
			double t = - dy[i - 1] / (dy[i] - dy[i - 1]);
			t = std::max(0.0, std::min(t, 1.0));
			double r = std::abs(memout[i - 1]) * (1 - t) + std::abs(memout[i]) * t;
			m_peaks.push_back(std::pair<double, double>(t + i - 1, r));
		}
	}
	return true;
}
