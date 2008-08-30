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
	
	std::vector<double> wnd;
	window(t, t0, windowfunc, windowlength, wnd);
	std::vector<double> weight(wnd);
	double wsum = 0.0, wndsum = 0.0;
	for(int i = 0; i < t; i++) {
		wndsum += wnd[i];
		weight[i] = weight[i] * weight[i];
		wsum += weight[i];
	}
	double wpoints = wndsum*wndsum/wsum; //# of fittable data.
	wpoints *= 0.3; //For reduction by filtering.

	double sigma2 = 0.0;
	for(int i = 0; i < t; i++) {
		sigma2 += std::norm(memin[i]) * weight[i];
	}
	sigma2 /= wsum;
	
	// Peak search by ZF-FFT;
	std::fill(m_ifft.begin(), m_ifft.end(), 0.0);
	std::vector<std::complex<double> > convwnd(m_ifft);
	for(int i = 0; i < t; i++) {
		m_ifft[(t0a + i) % n] = memin[i] * wnd[i];
	}
	m_fftN->exec(m_ifft, memout);
	for(int i = 0; i < n; i++) {
		memout[i] /= wndsum;
	}
	// Prepare convolution.
	for(int i = 0; i < t; i++) {
		m_ifft[(t0a + i) % n] = wnd[i];
	}
	m_fftN->exec(m_ifft, convwnd);
	for(int i = 0; i < n; i++) {
		convwnd[i] /= wndsum;
	}
	const int convwnd_blk_size = 8;
	const int convwnd_blk_cnt = (n + convwnd_blk_size - 1) / convwnd_blk_size;
	std::vector<double> convwnd_max_in_blk(convwnd_blk_cnt);
	double convwnd_max = 0.0;
	for(int blk = 0; blk < convwnd_blk_cnt; blk++) {
		int i = blk * convwnd_blk_size;
		convwnd_max_in_blk[blk] = 0.0;
		for(int k = 0; (k < convwnd_blk_size) && (i < n); k++) {
			convwnd_max_in_blk[blk] = std::max(convwnd_max_in_blk[blk], std::abs(convwnd[i]));
			i++;
		}
		convwnd_max = std::max(convwnd_max, convwnd_max_in_blk[blk]);
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
		for(int i = 0; i < n; i++) {
			if(std::norm(z) < std::norm(memout[i])) {
				freq = i;
				z = memout[i];
			}
		}
		freq *= t / (double)n;

		std::vector<std::complex<double> > coeff(t);
		double p = -2.0 * M_PI / (double)t * freq;
		for(int i = 0; i < t; i++) {
			coeff[i] = std::polar(1.0, (i + t0) * p);
		}
		sigma2 = 0.0;
		for(int i = 0; i < t; i++) {
			sigma2 += std::norm(wave[i] - z * std::conj(coeff[i])) * weight[i];
		}
		sigma2 /= wsum;
		
		fprintf(stderr, "NPeak = %d, sigma2 = %g, freq = %g, z = %g, ph = %g, ic = %g\n", 
			npeaks, sigma2, freq, std::abs(z), std::arg(z), ic);
		
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
			for(int i = 0; i < t; i++) {
				double k = 2.0 * M_PI / (double)t * (i + t0);
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
			
			double sor = 1.0;
			sor = std::min(sor, 0.4 / fabs(df));
			sor = std::min(sor, 0.2*sqrt(std::norm(z)/fabs(dx*dx+dy*dy)));
			df *= sor;
			dx *= sor;
			dy *= sor;
			freq += df;
			z += std::complex<double>(dx, dy);
			double p = -2.0 * M_PI / (double)t * freq;
			for(int i = 0; i < t; i++) {
				coeff[i] = std::polar(1.0, (i + t0) * p);
			}

			fprintf(stderr, "It = %d, freq = %g, z = %g, ph = %g\n",
				it, freq, std::abs(z), std::arg(z));
			
			if((df < 0.001) && (dx*dx+dy*dy < tol*tol*std::norm(z))) {
				break;
			}
		}

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
		
		double freqn = freq / t * n;
		std::complex<double> convz0 = std::polar(1.0, M_PI * (freqn - rint(freqn)));
		convz0 -= std::conj(convz0);
		double approx_convz_size = std::abs(convz0) / sqrt(tol);
		int approx_convwnd_size = 0;
		for(int blk = 0; blk < convwnd_blk_cnt; blk++) {
			if(convwnd_max_in_blk[blk] > sqrt(tol) *  convwnd_max)
				approx_convwnd_size += convwnd_blk_size;
		}
		double approx_conv_size = approx_convz_size * approx_convwnd_size * 2;
		if(0) {//(approx_conv_size * 3 < n * log((double)n))) {
			// Subtraction in freq. domain by convolution.
			int convz_size = lrint(std::abs(convz0) / tol + 0.5);
			for(int df = - convz_size; df <= convz_size; df++) {
				int f = lrint(freqn) + df;
				double p = 2.0 * M_PI * (freqn - f);
				std::complex<double> convz0 = std::polar(1.0, p/2.0);
				convz0 -= std::conj(convz0);
				std::complex<double> convz = convz0 / std::complex<double>(0.0, p);
				double th = tol * convwnd_max / std::abs(convz);
				convz *= z;
				
				for(int blk = 0; blk < convwnd_blk_cnt; blk++) {
					if(convwnd_max_in_blk[blk] > th) {
						int i = blk * convwnd_blk_size;
						for(int k = 0; (k < convwnd_blk_size) && (i < n); k++) {
							memout[(i + f + n) % n] -= convz * convwnd[i];					
							i++;
						}
					}
				}
			}
		}
		else {
			// Recalculate ZF-FFT.
			std::fill(m_ifft.begin(), m_ifft.end(), 0.0);
			for(int i = 0; i < t; i++) {
				m_ifft[(t0a + i) % n] = wave[i] * wnd[i];
			}
			m_fftN->exec(m_ifft, memout);
			for(int i = 0; i < n; i++) {
				memout[i] /= wndsum;
			}
		}
	}
	std::fill(m_ifft.begin(), m_ifft.end(), 0.0);
	for(int i = 0; i < m_peaks.size(); i++) {
		double freq = m_peaks[i].second;
		std::complex<double> z(zlist[i]);
		double p = 2.0 * M_PI / (double)n * freq;
		for(int i = 0; i < n; i++) {
			int j = (i + n/2) % n;
			m_ifft[j] += z * std::polar(1.0, p * (i + n/2 - n));
		}
	}
	m_fftN->exec(m_ifft, memout);
}
