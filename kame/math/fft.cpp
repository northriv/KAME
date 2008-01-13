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
#include "fft.h"

#include <gsl/gsl_sf.h>
#define bessel_i0 gsl_sf_bessel_I0

int
FFT::fitLength(int length0) {
	int length = lrint(pow(2.0, (ceil(log(length0) / log(2.0)))));		
	length = std::min(length, (int)lrint(pow(2.0, (ceil(log(length0 / 3.0) / log(2.0))))) * 3);		
	length = std::min(length, (int)lrint(pow(2.0, (ceil(log(length0 / 5.0) / log(2.0))))) * 5);		
	length = std::min(length, (int)lrint(pow(2.0, (ceil(log(length0 / 7.0) / log(2.0))))) * 7);		
	length = std::min(length, (int)lrint(pow(2.0, (ceil(log(length0 / 9.0) / log(2.0))))) * 9);		
	ASSERT(length0 <= length);
	dbgPrint(formatString("FFT using L=%d\n", length));
	return length;
}

FFT::FFT(int sign, int length) {
	m_fftlen = length;
	m_pBufin = (fftw_complex*)fftw_malloc(sizeof(fftw_complex) * length);
	m_pBufout = (fftw_complex*)fftw_malloc(sizeof(fftw_complex) * length);
	m_fftplan.reset(new fftw_plan);
	*m_fftplan = fftw_plan_dft_1d(length, m_pBufin, m_pBufout,
		(sign > 0) ? FFTW_BACKWARD : FFTW_FORWARD, FFTW_ESTIMATE);
}
FFT::~FFT() {
	fftw_destroy_plan(*m_fftplan);
	fftw_free(m_pBufin);
	fftw_free(m_pBufout);
}
void
FFT::exec(const std::vector<std::complex<double> >& wavein,
		std::vector<std::complex<double> >& waveout) {
	int size = wavein.size();
	ASSERT(size == length());
	waveout.resize(size);
	const std::complex<double> *pin = &wavein[0];
	fftw_complex *pout = m_pBufin;
	for(int i = 0; i < size; i++) {
		(*pout)[0] = pin->real();
		(*pout)[1] = pin->imag();
		pout++;
		pin++;
	}
	fftw_execute(*m_fftplan);
	const fftw_complex *pin2 = m_pBufout;
	std::complex<double> *pout2 = &waveout[0];
	for(int i = 0; i < size; i++) {
		*pout2 = std::complex<double>((*pin2)[0], (*pin2)[1]);
		pout2++;
		pin2++;
	}
}

double FFT::windowFuncRect(double x) {
	return (fabs(x) <= 0.5) ? 1 : 0;
//	return 1.0;
}
double FFT::windowFuncTri(double x) {
	return std::max(0.0, 1.0 - 2.0 * fabs(x));
}
double FFT::windowFuncHanning(double x) {
	if (fabs(x) >= 0.5)
		return 0.0;
	return 0.5 + 0.5*cos(2*PI*x);
}
double FFT::windowFuncHamming(double x) {
	if (fabs(x) >= 0.5)
		return 0.0;
	return 0.54 + 0.46*cos(2*PI*x);
}
double FFT::windowFuncBlackman(double x) {
	if (fabs(x) >= 0.5)
		return 0.0;
	return 0.42323+0.49755*cos(2*PI*x)+0.07922*cos(4*PI*x);
}
double FFT::windowFuncBlackmanHarris(double x) {
	if (fabs(x) >= 0.5)
		return 0.0;
	return 0.35875+0.48829*cos(2*PI*x)+0.14128*cos(4*PI*x)+0.01168*cos(6*PI*x);
}
double FFT::windowFuncFlatTop(double x) {
	return windowFuncHamming(x)*((fabs(x) < 1e-4) ? 1 : sin(4*PI*x)/(4*PI*x));
}
double FFT::windowFuncKaiser(double x, double alpha) {
	if (fabs(x) >= 0.5)
		return 0.0;
	x *= 2;
	x = sqrt(std::max(1 - x*x, 0.0));
	return bessel_i0(PI*alpha*x) / bessel_i0(PI*alpha);
}
double FFT::windowFuncKaiser1(double x) {
	return windowFuncKaiser(x, 3.0);
}
double FFT::windowFuncKaiser2(double x) {
	return windowFuncKaiser(x, 7.2);
}
double FFT::windowFuncKaiser3(double x) {
	return windowFuncKaiser(x, 15.0);
}

double FFT::windowFuncFlatTopLong(double x) {
	return windowFuncHamming(x)*((fabs(x) < 1e-4) ? 1 : sin(6*PI*x)/(6*PI*x));
}
double FFT::windowFuncFlatTopLongLong(double x) {
	return windowFuncHamming(x)*((fabs(x) < 1e-4) ? 1 : sin(8*PI*x)/(8*PI*x));
}
double FFT::windowFuncHalfSin(double x) {
	if (fabs(x) >= 0.5)
		return 0.0;
    return cos(PI*x);
}
