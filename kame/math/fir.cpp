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
#include "fir.h"
#include "support.h"
#include <algorithm> 

#include "fft.h"

FIR::FIR(int taps, double bandwidth, double center) :
	m_taps(taps), m_bandWidth(bandwidth), m_centerFreq(center) {
	if(taps < 3) taps = 2;
	taps = taps/2;

	int taplen = 2 * taps + 1;
	m_tapLen = taplen;
	int fftlen = lrint(pow(2.0, ceil(log(taplen * 5) / log(2.0))));
	fftlen = std::max(64, fftlen);
	m_fftLen = fftlen;
	m_pBufR = (double*)fftw_malloc(sizeof(double) * fftlen);
	m_pBufC = (fftw_complex*)fftw_malloc(sizeof(fftw_complex) * (fftlen / 2 + 1));
	m_firWnd.resize(fftlen / 2 + 1);
	m_rdftplan = fftw_plan_dft_r2c_1d(fftlen, m_pBufR, m_pBufC, FFTW_ESTIMATE);
	m_ridftplan = fftw_plan_dft_c2r_1d(fftlen, m_pBufC, m_pBufR, FFTW_ESTIMATE);
	
	double omega = M_PI * bandwidth;
	for(int i = 0; i < fftlen; i++)
		m_pBufR[i] = 0.0;
	double z = 0.0;
	for(int i = -taps; i <= taps; i++) {
		double x = i * omega;
		//sinc(x) * Hamming window
		double y = (i == 0) ? 1.0 : (sin(x)/x);
		y *= 0.54 + 0.46*cos(M_PI*(double)i/taps);
		m_pBufR[(fftlen + i) % fftlen] = y;
		z += y;
	}
	//scaling sum into unity
	//shift center freq
	double omega_c = 2 * M_PI * center;
	for(int i = -taps; i <= taps; i++) {
		m_pBufR[(fftlen + i) % fftlen] *= cos(omega_c * i) / (z * (double)(fftlen));
	}
	
	fftw_execute(m_rdftplan);
	
	for(int i = 0; i < (int)m_firWnd.size(); i++) {
		m_firWnd[i] = m_pBufC[i][0];
	}
}
FIR::~FIR() {
	fftw_destroy_plan(m_rdftplan);
	fftw_destroy_plan(m_ridftplan);
	fftw_free(m_pBufR);
	fftw_free(m_pBufC);
}
void
FIR::exec(const double *src, double *dst, int len) {
	for(int ss = 0; ss < len; ss += (int)m_fftLen - m_tapLen * 2) {
		for(int i = 0; i < m_fftLen; i++) {
			int j = ss + i - m_tapLen;
			if(j < 0)
				j = std::min(-j - 1, len - 1);
			if(j >= len)
				j = std::max(2 * len - 1 - j, 0);
			m_pBufR[i] = src[j];
		}
		fftw_execute(m_rdftplan);
		for(int i = 0; i < (int)m_firWnd.size(); i++) {
			m_pBufC[i][0] = m_pBufC[i][0] * m_firWnd[i];
			m_pBufC[i][1] = m_pBufC[i][1] * m_firWnd[i];
		}
		fftw_execute(m_ridftplan);
		for(int i = m_tapLen; i < m_fftLen - m_tapLen; i++) {
			int j = ss + i - m_tapLen;
			if((j < 0) || (j >= len))
				continue;
			else
				dst[j] = m_pBufR[i];
		}
	}
}
