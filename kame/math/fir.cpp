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
#include "fir.h"
#include "support.h"
#include <algorithm> 

#include "fft.h"
 
FIR::FIR()
{
	m_Taps = -1;
	m_BandWidth = -1.0;
	m_Center = 0.0;
}
FIR::~FIR()
{
}
float
FIR::oneFIR(float x, float *z, float *m_Coeff, int taps)
{
	float zo = x;
	float y = 0;
	for(int i = -taps; i <= taps; i++) {
		float z1;
		z1 = *z;
		*(z++) = zo;
		y += zo * *(m_Coeff++);
		zo = z1;
	}
	return y;
}
int
FIR::setupBPF(int taps, double bandwidth, double center)
{
	if(taps < 3) taps = 2;
	taps = taps/2;
	if((m_Taps != taps) || (m_BandWidth != bandwidth) || (m_Center != center)) {
		m_Taps = taps;
		m_BandWidth = bandwidth;
		m_Center = center;
		m_Coeff.resize(taps*2 + 1);
		fill(m_Coeff.begin(), m_Coeff.end(), 0.0);
		m_Z.resize(taps*2 + 1);
		m_Coeff[taps] = 1.0;
		if(bandwidth < 1.0) {
			float omega = PI*bandwidth;
			float omega2 = 2 * PI * center;
			for(int i = -taps; i < 0; i++) {
				float x = i*omega;
				//sinc(x) * Hamming window
				m_Coeff[taps + i] = sinf(x)/x*(0.54 + 0.46*cosf(PI*(float)i/taps));
			}
			//mirroring
			for(int i = 1; i <= taps; i++) {
				m_Coeff[taps + i] = m_Coeff[taps - i];
			}
			//scaling sum into unity
			float z = 0.0;
			for(int i = 0; i <= 2*taps; i++) {
				z += m_Coeff[i];
			}
			for(int i = 0; i <= 2*taps; i++) {
				m_Coeff[i] /= z;
			}
			//shift center freq
			for(int i = 0; i <= 2*taps; i++) {
				m_Coeff[i] *= cos(omega2 * i);
			}
		}
	}
	return 0;
}
int
FIR::doFIR(double *src, double *dst, int len)
{
	fill(m_Z.begin(), m_Z.end(), 0.0);
	int taps = m_Taps;
	//fill buffer with mirror image
	int k = 0;
   	for(; k < taps; k++) {
		oneFIR(src[(taps - k) % len], &m_Z[0], &m_Coeff[0], taps);
	}
	//wait until the center of taps
   	for(k = 0; k < taps; k++) {
		oneFIR(src[k % len], &m_Z[0], &m_Coeff[0], taps);
	}
	//do FIR
   	for(; k < len; k++) {
		*(dst++) = oneFIR(src[k % len], &m_Z[0], &m_Coeff[0], taps);
	}
	//the rest using mirror
   	for(k = 0; k < taps; k++) {
		*(dst++) = oneFIR(src[(unsigned int)(len -k - 2) % len], &m_Z[0], &m_Coeff[0], taps);
	}
	return 0;
}


FIRMDCT::FIRMDCT(int taps, double bandwidth, double center) :
	m_taps(taps), m_bandWidth(bandwidth), m_centerFreq(center) {
	if(taps < 3) taps = 2;
	taps = taps/2;

	int taplen = 2 * taps + 1;
	m_tapLen = taplen;
	int mdctlen = std::max(32, (int)lrint(pow(2.0, ceil(log(taplen * 8) / log(2.0)))));
	m_mdctLen = mdctlen;
	m_pBufR = (double*)fftw_malloc(sizeof(double) * mdctlen);
	m_pBufC = (fftw_complex*)fftw_malloc(sizeof(fftw_complex) * (mdctlen / 2 + 1));
	m_firWnd.resize(mdctlen / 2 + 1);
	m_mdctWnd.resize(mdctlen - 2 * taplen);
	m_rdftplan = fftw_plan_dft_r2c_1d(mdctlen, m_pBufR, m_pBufC, FFTW_ESTIMATE);
	m_ridftplan = fftw_plan_dft_c2r_1d(mdctlen, m_pBufC, m_pBufR, FFTW_ESTIMATE);
	
	double omega = M_PI * bandwidth;
	for(int i = 0; i < mdctlen; i++)
		m_pBufR[i] = 0.0;
	double z = 0.0;
	for(int i = -taps; i <= taps; i++) {
		double x = i * omega;
		//sinc(x) * Hamming window
		double y = (i == 0) ? 1.0 : (sin(x)/x);
		y *= 0.54 + 0.46*cos(M_PI*(double)i/taps);
		m_pBufR[(mdctlen + i) % mdctlen] = y;
		z += y;
	}
	//scaling sum into unity
	//shift center freq
	double omega_c = 2 * M_PI * center;
	for(int i = -taps; i <= taps; i++) {
		m_pBufR[(mdctlen + i) % mdctlen] *= cos(omega_c * i) / (z * (double)(mdctlen));
	}
	
	fftw_execute(m_rdftplan);
	
	for(int i = 0; i < (int)m_firWnd.size(); i++) {
		m_firWnd[i] = std::complex<double>(m_pBufC[i][0], m_pBufC[i][1]);
	}
	for(int i = 0; i < (int)m_mdctWnd.size(); i++) {
		m_mdctWnd[i] = sin(M_PI * (i + 0.5) / (double)m_mdctWnd.size());
	}
}
FIRMDCT::~FIRMDCT() {
	fftw_destroy_plan(m_rdftplan);
	fftw_destroy_plan(m_ridftplan);
	fftw_free(m_pBufR);
	fftw_free(m_pBufC);
}
void
FIRMDCT::exec(const double *src, double *dst, int len) {
	for(int i = 0; i < len; i++) {
		dst[i] = 0.0;
	}
	for(int ss = -(int)m_mdctWnd.size() / 2; ss < len; ss += (int)m_mdctWnd.size() / 2) {
		for(int i = 0; i < m_tapLen; i++)
			m_pBufR[i] = 0.0;
		for(int i = m_mdctLen - m_tapLen; i < m_mdctLen; i++)
			m_pBufR[i] = 0.0;
		for(int i = m_tapLen; i < m_mdctLen - m_tapLen; i++) {
			int j = ss + i - m_tapLen;
			if((j < 0) || (j >= len))
				m_pBufR[i] = 0.0;
			else
				m_pBufR[i] = m_mdctWnd[i - m_tapLen] * src[j];
		}
		fftw_execute(m_rdftplan);
		for(int i = 0; i < (int)m_firWnd.size(); i++) {
			std::complex<double> z = std::complex<double>(m_pBufC[i][0], m_pBufC[i][1]) * m_firWnd[i];
			m_pBufC[i][0] = std::real(z);
			m_pBufC[i][1] = std::imag(z);
		}
		fftw_execute(m_ridftplan);
		for(int i = m_tapLen; i < m_mdctLen - m_tapLen; i++) {
			int j = ss + i - m_tapLen;
			if((j < 0) || (j >= len))
				continue;
			else
				dst[j] += m_mdctWnd[i - m_tapLen] * m_pBufR[i];
		}
	}
}
