/***************************************************************************
		Copyright (C) 2002-2015 Kentaro Kitagawa
		                   kitagawa@phys.s.u-tokyo.ac.jp
		
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
	
	//Fitting with weights.
	std::vector<double> weight;
	window(t, t0, windowfunc, windowlength, weight);
	double wsum = 0.0, w2sum = 0.0;
	for(int i = 0; i < t; i++) {
		wsum += weight[i];
		w2sum += weight[i] * weight[i];
	}
	int wpoints = wsum*wsum/w2sum; //# of fittable data in time domain.
	wpoints = lrint(wpoints * numberOfNoises(memin) / (double)t); //# of fittable data in freq. domain.
	wpoints = std::min(std::max(wpoints, t/100 + 1), t);
//	fprintf(stderr, "# of data points = %d\n", wpoints);

	//Standard error.
	double sigma2 = 0.0;
	for(int i = 0; i < t; i++) {
		sigma2 += std::norm(memin[i]) * weight[i];
	}
	sigma2 /= wsum;
	
	// Peak search by ZF-FFT;
	std::fill(m_ifft.begin(), m_ifft.end(), std::complex<double>(0.0));
	std::vector<std::complex<double> > convwnd(m_ifft);
	for(int i = 0; i < t; i++) {
		m_ifft[(t0a + i) % n] = memin[i] * weight[i];
	}
	m_fftN->exec(m_ifft, memout);
	for(int i = 0; i < n; i++) {
		memout[i] /= wsum;
	}
	
	std::vector<std::complex<double> > wave(memin);
	std::deque<std::complex<double> > zlist;
	
	double ic = 1e99;
	for(int lp = 0; lp < 32; lp++) {
		int npeaks = m_peaks.size();
		double loglikelifood = -wpoints * (log(2*M_PI) + 1.0 + log(sigma2));
		double ic_new = m_funcIC(loglikelifood, npeaks * 3.0, wpoints * 2.0);
		if(ic_new > ic) {
			if(m_peaks.size()) {
				m_peaks.pop_back();
				zlist.pop_back();
			}
			break;
		}
		ic = ic_new;
		
		double freq = 0.0;
		std::complex<double> z(0.0);
		double normz = 0;
		for(int i = 0; i < n; i++) {
			if(normz < std::norm(memout[i])) {
				freq = i;
				z = memout[i];
				normz = std::norm(z);
			}
		}
		freq *= t / (double)n;

		//Prepare exp(-omega t)
		std::vector<std::complex<double> > coeff(t);
		double p = -2.0 * M_PI / (double)t * freq;
		for(int i = 0; i < t;) {
			std::complex<double> x = std::polar(1.0, (i + t0) * p);
			std::complex<double> y = std::polar(1.0, p);
			for(int j = 0; (j < 1024) && (i < t); j++) {
				coeff[i] = x;
				x *= y;
				i++;
			}
		}
		//Standard error.
		sigma2 = 0.0;
		for(int i = 0; i < t; i++) {
			sigma2 += std::norm(wave[i] - z * std::conj(coeff[i])) * weight[i];
		}
		sigma2 /= wsum;
		
//		fprintf(stderr, "NPeak = %d, sigma2 = %g, freq = %g, z = %g, ph = %g, ic = %g\n", 
//			npeaks, sigma2, freq, std::abs(z), std::arg(z), ic);
		
		// Newton's method for non-linear least square fit.
		for(int it = 0; it < 10; it++) {
			// Derivertive.
			double ds2df = 0.0;
			double ds2dx = 0.0;
			double ds2dy = 0.0;
			// Jacobian.
			double d2s2df2 = 0.0;
//			double d2s2dx2 = 1.0;
//			double d2s2dy2 = 1.0;
			double d2s2dfx = 0.0;
			double d2s2dfy = 0.0;
//			double d2s2dxy = 0.0;
			double kstep = 2.0 * M_PI / (double)t;
			double k = kstep * t0;
			for(int i = 0; i < t; i++) {
				k += kstep;
				std::complex<double> yew = wave[i] * coeff[i] * weight[i];
				std::complex<double> yewzk = std::conj(z) * yew * k; 
				ds2df -= std::imag(yewzk);
				std::complex<double> a = -yew + z * weight[i];
				ds2dx += std::real(a);
				ds2dy += std::imag(a);
				d2s2df2 += std::real(yewzk * k);
				a = yew * k;
				d2s2dfx -= std::imag(a);
				d2s2dfy += std::real(a);
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
			
//			fprintf(stderr, "Ds2 = %g,%g,%g\nJ=[%g,%g,%g;%g,1,0;%g,0,1]\nDx=%g,%g,%g\n",
//				ds2df,ds2dx,ds2dy,d2s2df2,d2s2dfx,d2s2dfy,d2s2dfx,d2s2dfy,df,dx,dy);
			
			double sor = 1.0;
			sor = std::min(sor, 0.4 / fabs(df));
			sor = std::min(sor, 0.2*sqrt(std::norm(z)/fabs(dx*dx+dy*dy)));
			df *= sor;
			dx *= sor;
			dy *= sor;
			freq += df;
			z += std::complex<double>(dx, dy);
			//Prepare exp(-omega t)
			double p = -2.0 * M_PI / (double)t * freq;
			for(int i = 0; i < t;) {
				std::complex<double> x = std::polar(1.0, (i + t0) * p);
				std::complex<double> y = std::polar(1.0, p);
				for(int j = 0; (j < 1024) && (i < t); j++) {
					coeff[i] = x;
					x *= y;
					i++;
				}
			}

//			fprintf(stderr, "It = %d, freq = %g, z = %g, ph = %g\n",
//				it, freq, std::abs(z), std::arg(z));
			
			if((df < 0.001) && (dx*dx+dy*dy < tol*tol*std::norm(z))) {
				break;
			}
		}

		//Standard error.
		double sigma2 = 0.0;
		for(int i = 0; i < t; i++) {
			sigma2 += std::norm(wave[i] - z * std::conj(coeff[i])) * weight[i];
		}
		sigma2 /= wsum;
		
		m_peaks.push_back(std::pair<double, double>(std::abs(z) * n, freq / t * n));
		zlist.push_back(z);
		//Subtract the wave.
		for(int i = 0; i < t; i++) {
			wave[i] -= z * std::conj(coeff[i]);
		}
		
		// Recalculate ZF-FFT.
		std::fill(m_ifft.begin(), m_ifft.end(), std::complex<double>(0.0));
		for(int i = 0; i < t; i++) {
			m_ifft[(t0a + i) % n] = wave[i] * weight[i];
		}
		m_fftN->exec(m_ifft, memout);
		for(int i = 0; i < n; i++) {
			memout[i] /= wsum;
		}
	}
	std::fill(m_ifft.begin(), m_ifft.end(), std::complex<double>(0.0));
	for(int i = 0; i < m_peaks.size(); i++) {
		double freq = m_peaks[i].second;
		std::complex<double> z(zlist[i]);
		double p = 2.0 * M_PI / (double)n * freq;
		for(int i = 0; i < n;) {
			std::complex<double> x = std::polar(1.0, (i + n/2 - n) * p);
			std::complex<double> y = std::polar(1.0, p);
			x *= z;
			for(int j = 0; (j < 1024) && (i < n); j++) {
				m_ifft[(i + n/2) % n] += x;
				x *= y;
				i++;
			}
		}
	}
	m_fftN->exec(m_ifft, memout);
}
