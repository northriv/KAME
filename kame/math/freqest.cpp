/***************************************************************************
		Copyright (C) 2002-2010 Kentaro Kitagawa
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

void
FreqEstimation::genSpectrum(const std::vector<std::complex<double> >& memin,
		std::vector<std::complex<double> >& memout,
		int t0, double tol, FFT::twindowfunc windowfunc, double windowlength) {
	int t = memin.size();
	int n = memout.size();
	
	if(t > 1024)
		throw XKameError(i18n("Too large size to allocate a matrix."), __FILE__, __LINE__);
	
	int wpoints = lrint(numberOfNoises(memin)); //# of fittable data in freq. domain.
	wpoints = std::min(std::max(wpoints, t/100 + 1), t);
//	fprintf(stderr, "# of data points = %d\n", wpoints);
	
	double tpoworg = 0.0;
	for(int i = 0; i < t; i++) {
		tpoworg += std::norm(memin[i]);
	}

	std::vector<std::complex<double> > rx(t);
	autoCorrelation(memin, rx);
	//# of signal space.
	int p = t; // / 2 - 1;
	rx.resize(p);
	// Correlation matrix.
	ublas::matrix<std::complex<double> > r(p, p);
	for(int i = 0; i < p; i++) {
		ublas::matrix_row<ublas::matrix<std::complex<double> > > rrow(r, i);
		for(int j = i; j < p; j++) {
			rrow(j) = rx[j - i];
		}
	}
	ublas::matrix<std::complex<double> > eigv;
	ublas::vector<double> lambda;
	double tol_lambda = tol * std::abs(rx[0]) * 0.1;
	eigHermiteRRR(r, lambda, eigv, tol_lambda);

	//# of signals.
	int numsig = 0;
	if(!m_mvdl_method) {
		//Minimum IC.
		double minic = 1e99;
		double sumlambda = 0.0, sumloglambda = 1.0;
		for(int i = 0; i < p; i++) {
			sumlambda += lambda[i];
			sumloglambda += log(lambda[i]);
			int q = p - 1 - i;
			double logl = t * (p - q) * (sumloglambda / (double)(p - q) - log(sumlambda / (double)(p - q)));
			double ic = m_funcIC(logl, q * (2*p - q), wpoints);
			if(ic < minic) {
				minic = ic;
				numsig = q;
			}
		}
		numsig *= windowlength;
		numsig = std::max(std::min(numsig, p - 1), 0);
		
//		fprintf(stderr, "MinIC=%g, # of signal=%d, p=%d\n", minic, numsig, p);

//		std::cout << lambda << std::endl;
//		std::cout << eigv << std::endl;
	}
	std::vector<std::complex<double> > fftin(t, 0.0), fftout(t), acsum(t, 0.0);
	for(int i = 0; i < p - numsig; i++) {
		ublas::matrix_column<ublas::matrix<std::complex<double> > > eigvcol(eigv, i);
		ASSERT(fabs(norm_2(eigvcol) - 1.0) < 0.1);
		for(int j = 0; j < p; j++) {
			fftin[j] = eigvcol(j);
		}
		autoCorrelation(fftin, fftout);
		double z = lambda[i];
		z = std::max(z, tol_lambda);
		z = (m_eigenvalue_method) ? (1.0 / z) : 1.0;
		for(int k = 0; k < p; k++) {
			acsum[k] += fftout[k] * z;
		}
	}
	std::vector<std::complex<double> > zffftin(n, 0.0), zffftout(n);
	std::vector<double> ip(n), dy(n);
	acsum[0] /= (double)2;
	std::copy(acsum.begin(), acsum.end(), zffftin.begin());
	m_ifftN->exec(zffftin, zffftout);
	for(int i = 0; i < n; i++)
		ip[i] = std::real(zffftout[i]);
	for(int i = 0; i < p; i++)
		zffftin[i] = (double)((i >= n/2) ? (i - n) : i) * acsum[i] * std::complex<double>(0, 1);
	m_ifftN->exec(zffftin, zffftout);
	for(int i = 0; i < n; i++)
		dy[i] = std::real(zffftout[i]);
	
	//Power spectrum density.
	std::vector<double> psd(n);
	double tpow = 0.0;
	for(int i = 0; i < n; i++) {
		double z = 1.0 / ip[i];
		tpow += z;
		psd[i] = z;
	}
	double normalize = n * tpoworg / tpow;
	for(int i = 0; i < n; i++) {
		psd[i] *= normalize;
	}
	//Least-Square Phase Estimation.
	double coeff = lspe(memin, t0, psd, memout, tol, true, windowfunc);	
	normalize *= coeff * coeff;
	
	//Peak detection. Sub-resolution detection for smooth curves.
	for(int ip = 0; ip < n; ip++) {
		int in = (ip + 1) % n;
		if((dy[ip] < 0) && (dy[in] > 0)) {
			double dx = - dy[ip] / (dy[in] - dy[ip]);
			if((dx >= 0) && (dx <= 1.0)) {
				std::complex<double> z = 0.0, xn = 1.0,
					x = std::polar(1.0, 2 * M_PI * (dx + ip) / (double)n);
				for(int j = 0; j < p; j++) {
					z += acsum[j] * xn;
					xn *= x;
				}
				double r = sqrt(normalize * std::max(1.0 / std::real(z), 0.0));
				m_peaks.push_back(std::pair<double, double>(r, dx + ip));
			}
		}
	}
}
