/***************************************************************************
		Copyright (C) 2002-2007 Kentaro Kitagawa
		                   kitagawa@scphys.kyoto-u.ac.jp
		
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
