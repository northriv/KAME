/***************************************************************************
		Copyright (C) 2002-2015 Kentaro Kitagawa
		                   kitag@issp.u-tokyo.ac.jp
		
		This program is free software; you can redistribute it and/or
		modify it under the terms of the GNU General Public
		License as published by the Free Software Foundation; either
		version 2 of the License, or (at your option) any later version.
		
		You should have received a copy of the GNU General 
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
	//Scratch for plan creation and the one-off coefficient FFT; freed at the
	//end of the ctor.  exec() later runs these immutable plans on its own
	//per-call buffers (matching fftw_malloc alignment).
	double *pbufR = (double*)fftw_malloc(sizeof(double) * fftlen);
	fftw_complex *pbufC = (fftw_complex*)fftw_malloc(sizeof(fftw_complex) * (fftlen / 2 + 1));
	m_firWnd.resize(fftlen / 2 + 1);
	m_rdftplan.reset(fftw_plan_dft_r2c_1d(fftlen, pbufR, pbufC, FFTW_ESTIMATE), fftw_destroy_plan);
	m_ridftplan.reset(fftw_plan_dft_c2r_1d(fftlen, pbufC, pbufR, FFTW_ESTIMATE), fftw_destroy_plan);

	double omega = M_PI * bandwidth;
	for(int i = 0; i < fftlen; i++)
		pbufR[i] = 0.0;
	double z = 0.0;
	for(int i = -taps; i <= taps; i++) {
		double x = i * omega;
		//sinc(x) * Hamming window
		double y = (i == 0) ? 1.0 : (sin(x)/x);
		y *= 0.54 + 0.46*cos(M_PI*(double)i/taps);
		pbufR[(fftlen + i) % fftlen] = y;
		z += y;
	}
	//scaling sum into unity
	//shift center freq
	double omega_c = 2 * M_PI * center;
	for(int i = -taps; i <= taps; i++) {
		pbufR[(fftlen + i) % fftlen] *= cos(omega_c * i) / (z * (double)(fftlen));
	}

	fftw_execute_dft_r2c(m_rdftplan.get(), pbufR, pbufC);

	for(int i = 0; i < (int)m_firWnd.size(); i++) {
		m_firWnd[i] = pbufC[i][0];
	}
	fftw_free(pbufR);
	fftw_free(pbufC);
}
void
FIR::exec(const double *src, double *dst, int len) const {
	double *pbufR = (double*)fftw_malloc(sizeof(double) * m_fftLen);
	fftw_complex *pbufC = (fftw_complex*)fftw_malloc(sizeof(fftw_complex) * (m_fftLen / 2 + 1));
	for(int ss = 0; ss < len; ss += (int)m_fftLen - m_tapLen * 2) {
		for(int i = 0; i < m_fftLen; i++) {
			int j = ss + i - m_tapLen;
			if(j < 0)
				j = std::min(-j - 1, len - 1);
			if(j >= len)
				j = std::max(2 * len - 1 - j, 0);
			pbufR[i] = src[j];
		}
		fftw_execute_dft_r2c(m_rdftplan.get(), pbufR, pbufC);
		for(int i = 0; i < (int)m_firWnd.size(); i++) {
			pbufC[i][0] = pbufC[i][0] * m_firWnd[i];
			pbufC[i][1] = pbufC[i][1] * m_firWnd[i];
		}
		fftw_execute_dft_c2r(m_ridftplan.get(), pbufC, pbufR);
		for(int i = m_tapLen; i < m_fftLen - m_tapLen; i++) {
			int j = ss + i - m_tapLen;
			if((j < 0) || (j >= len))
				continue;
			else
				dst[j] = pbufR[i];
		}
	}
	fftw_free(pbufR);
	fftw_free(pbufC);
}
