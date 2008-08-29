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
#include "freqestleastsquare.h"

void
FreqEstLeastSquare::genSpectrum(const std::vector<std::complex<double> >& memin,
		std::vector<std::complex<double> >& memout,
		int t0, double tol, FFT::twindowfunc windowfunc, double windowlength) {
	int t = memin.size();
	int n = memout.size();
	int t0a = t0;
	if(t0a < 0)
		t0a += (-t0a / n + 1) * n;
	
	std::vector<double> weight;
	window(t, t0, windowfunc, windowlength, weight);
	double wsum = 0.0, wsqrtsum = 0.0;
	for(int i = 0; i < t; i++) {
		wsqrtsum += weight[i];
		weight[i] = weight[i] * weight[i];
		wsum += weight[i];
	}
	double wpoints = wsqrtsum*wsqrtsum/wsum; //# of fittable data.
	wpoints *= 0.3; //For reduction by filtering.

	double sigma2 = 0.0;
	for(int i = 0; i < t; i++) {
		sigma2 += std::norm(memin[i]) * weight[i];
	}
	sigma2 /= wsum;
	
	// For peak search by ZF-FFT;
	std::fill(m_ifft.begin(), m_ifft.end(), 0.0);
	
	std::vector<std::complex<double> > wave(memin);

	std::deque<std::complex<double> > zlist;
	double ic = 1e99;
	for(int npeaks = 0; npeaks < 32; npeaks++) {
		double loglikelifood = -wpoints * (log(2*M_PI) + 1.0 + log(sigma2));
		double ic_new = m_funcIC(loglikelifood, npeaks * 3.0, wpoints * 2.0);
		if(ic_new > ic) {
			if(m_peaks.size()) {
				m_peaks.pop_back();
				zlist.pop_back();
			}
			ASSERT(m_peaks.size() == npeaks - 1);
			break;
		}
		ic = ic_new;
		
		double freq = 0.0;
		std::complex<double> z(0.0);
		// Peak search by ZF-FFT;
		for(int i = 0; i < t; i++) {
			m_ifft[(t0a + i) % n] = wave[i] * sqrt(weight[i]);
		}
		m_fftN->exec(m_ifft, memout);
		for(int i = 0; i < n; i++) {
			if(std::norm(z) < std::norm(memout[i])) {
				freq = i;
				z = memout[i];
			}
		}
		freq *= t / (double)n;
		z /= wsqrtsum;

		std::vector<std::complex<double> > coeff(t);
		for(int i = 0; i < t; i++) {
			double k = 2.0 * M_PI / (double)t * (i + t0);
			coeff[i] = std::polar(1.0, - freq * k);
		}
		sigma2 = 0.0;
		for(int i = 0; i < t; i++) {
			sigma2 += std::norm(wave[i] - z * std::conj(coeff[i])) * weight[i];
		}
		sigma2 /= wsum;
		
//		fprintf(stderr, "NPeak = %d, sigma2 = %g, freq = %g, z = %g, ph = %g, ic = %g\n", 
//			npeaks, sigma2, freq, std::abs(z), std::arg(z), ic);
		
		bool it_success = true;
		// Newton's method for non-linear least square fit.
		for(int it = 0; it < 4; it++) {
			double ds2df = 0.0;
			double ds2dx = 0.0;
			double ds2dy = 0.0;
			double d2s2df2 = 0.0;
//			double d2s2dx2 = 1.0;
//			double d2s2dy2 = 1.0;
			double d2s2dfx = 0.0;
			double d2s2dfy = 0.0;
//			double d2s2dxy = 0.0;
			for(int i = 0; i < t; i++) {
				double k = 2.0 * M_PI / (double)t * (i + t0);
				std::complex<double> yew = wave[i] * coeff[i] * weight[i];
				std::complex<double> yewzk = std::conj(z) * yew * k; 
				ds2df -= std::imag(yewzk);
				ds2dx += std::real(-yew + z * weight[i]);
				ds2dy += std::imag(-yew + z * weight[i]);
				d2s2df2 += std::real(yewzk * k);
				d2s2dfx -= std::imag(yew * k);
				d2s2dfy += std::real(yew * k);
			}
			ds2df /= wsum;
			ds2dx /= wsum;
			ds2dy /= wsum;
			d2s2df2 /= wsum;
			d2s2dfx /= wsum;
			d2s2dfy /= wsum;
			double detJ = d2s2df2 - d2s2dfx*d2s2dfx - d2s2dfy*d2s2dfy;
			double df = -ds2df + ds2dx * d2s2dfx + ds2dy * d2s2dfy;
			df /= detJ;
			double dx = ds2df * d2s2dfx - ds2dx * (d2s2df2 - d2s2dfy*d2s2dfy) - ds2dy * d2s2dfx*d2s2dfy;
			dx /= detJ;
			double dy = ds2df * d2s2dfy - ds2dx * d2s2dfx*d2s2dfy - ds2dy * (d2s2df2 - d2s2dfx*d2s2dfx);
			dy /= detJ;
			
			if(fabs(df) > 0.4) {
				double s = 0.4 / fabs(df);
				df *= s;
				dx *= s;
				dy *= s;
			}
			freq += df;
			z += std::complex<double>(dx, dy);
			for(int i = 0; i < t; i++) {
				double k = 2.0 * M_PI / (double)t * (i + t0);
				coeff[i] = std::polar(1.0, - freq * k);
			}
			double sigma2_new = 0.0;
			for(int i = 0; i < t; i++) {
				sigma2_new += std::norm(wave[i] - z * std::conj(coeff[i])) * weight[i];
			}
			sigma2_new /= wsum;

//			fprintf(stderr, "It = %d, sigma2 = %g, freq = %g, z = %g, ph = %g\n",
//				it, sigma2_new, freq, std::abs(z), std::arg(z));
			
			if(sigma2 < sigma2_new) {
				it_success = false;
				break;
			}
			if(sigma2_new / sigma2 > 0.9999){ //1.0 - tol) {
				sigma2 = sigma2_new;
				break;
			}
			sigma2 = sigma2_new;			
		}
		if(!it_success) {
			break;
		}
		
		m_peaks.push_back(std::pair<double, double>(std::abs(z) * n, freq / t * n));
		zlist.push_back(z);
		
		for(int i = 0; i < t; i++) {
			wave[i] -= z * std::conj(coeff[i]);
		}
		
	}
	std::fill(m_ifft.begin(), m_ifft.end(), 0.0);
	for(int i = 0; i < m_peaks.size(); i++) {
		double freq = m_peaks[i].second;
		std::complex<double> z(zlist[i]);
		double p = 2.0*M_PI/n;
		for(int i = 0; i < n; i++) {
			int j = (i + n/2) % n;
			m_ifft[j] += z * std::polar(1.0, p*freq*(i + n/2 - n));
		}
	}
	m_fftN->exec(m_ifft, memout);
}
