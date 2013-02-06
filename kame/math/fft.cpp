/***************************************************************************
		Copyright (C) 2002-2013 Kentaro Kitagawa
		                   kitag@kochi-u.ac.jp

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
FFTBase::fitLength(int length0) {
	int length = lrint(pow(2.0, (ceil(log(length0) / log(2.0)))));		
	length = std::min(length, (int)lrint(pow(2.0, (ceil(log(length0 / 3.0) / log(2.0))))) * 3);		
	length = std::min(length, (int)lrint(pow(2.0, (ceil(log(length0 / 5.0) / log(2.0))))) * 5);		
	length = std::min(length, (int)lrint(pow(2.0, (ceil(log(length0 / 7.0) / log(2.0))))) * 7);		
	length = std::min(length, (int)lrint(pow(2.0, (ceil(log(length0 / 9.0) / log(2.0))))) * 9);		
	assert(length0 <= length);
//	dbgPrint(formatString("FFT using L=%d\n", length));
	return length;
}

double FFTBase::windowFuncRect(double x) {
	return (fabs(x) <= 0.5) ? 1 : 0;
//	return 1.0;
}
double FFTBase::windowFuncTri(double x) {
	return std::max(0.0, 1.0 - 2.0 * fabs(x));
}
double FFTBase::windowFuncHanning(double x) {
	if (fabs(x) >= 0.5)
		return 0.0;
	return 0.5 + 0.5*cos(2*M_PI*x);
}
double FFTBase::windowFuncHamming(double x) {
	if (fabs(x) >= 0.5)
		return 0.0;
	return 0.54 + 0.46*cos(2*M_PI*x);
}
double FFTBase::windowFuncBlackman(double x) {
	if (fabs(x) >= 0.5)
		return 0.0;
	return 0.42323+0.49755*cos(2*M_PI*x)+0.07922*cos(4*M_PI*x);
}
double FFTBase::windowFuncBlackmanHarris(double x) {
	if (fabs(x) >= 0.5)
		return 0.0;
	return 0.35875+0.48829*cos(2*M_PI*x)+0.14128*cos(4*M_PI*x)+0.01168*cos(6*M_PI*x);
}
double FFTBase::windowFuncFlatTop(double x) {
	return windowFuncHamming(x)*((fabs(x) < 1e-4) ? 1 : sin(4*M_PI*x)/(4*M_PI*x));
}
double FFTBase::windowFuncKaiser(double x, double alpha) {
	if (fabs(x) >= 0.5)
		return 0.0;
	x *= 2;
	x = sqrt(std::max(1 - x*x, 0.0));
	return bessel_i0(M_PI*alpha*x) / bessel_i0(M_PI*alpha);
}
double FFTBase::windowFuncKaiser1(double x) {
	return windowFuncKaiser(x, 3.0);
}
double FFTBase::windowFuncKaiser2(double x) {
	return windowFuncKaiser(x, 7.2);
}
double FFTBase::windowFuncKaiser3(double x) {
	return windowFuncKaiser(x, 15.0);
}

double FFTBase::windowFuncFlatTopLong(double x) {
	return windowFuncHamming(x)*((fabs(x) < 1e-4) ? 1 : sin(6*M_PI*x)/(6*M_PI*x));
}
double FFTBase::windowFuncFlatTopLongLong(double x) {
	return windowFuncHamming(x)*((fabs(x) < 1e-4) ? 1 : sin(8*M_PI*x)/(8*M_PI*x));
}
double FFTBase::windowFuncHalfSin(double x) {
	if (fabs(x) >= 0.5)
		return 0.0;
    return cos(M_PI*x);
}


FFTBase::FFTBase(int length) {
	m_fftlen = length;
	m_fftplan.reset(new fftw_plan);
}
FFTBase::~FFTBase() {
	fftw_destroy_plan(*m_fftplan);
}
FFT::FFT(int sign, int length) : FFTBase(length) {
	m_pBufin = (fftw_complex*)fftw_malloc(sizeof(fftw_complex) * length);
	m_pBufout = (fftw_complex*)fftw_malloc(sizeof(fftw_complex) * length);
	*m_fftplan = fftw_plan_dft_1d(length, m_pBufin, m_pBufout,
		(sign > 0) ? FFTW_BACKWARD : FFTW_FORWARD, FFTW_ESTIMATE);
}
FFT::~FFT() {
	fftw_free(m_pBufin);
	fftw_free(m_pBufout);
}
void
FFT::exec(const std::vector<std::complex<double> >& wavein,
		std::vector<std::complex<double> >& waveout) {
	int size = wavein.size();
	assert(size == length());
	waveout.resize(size);
	const std::complex<double> *pin = &wavein[0];
	fftw_complex *pout = m_pBufin;
	for(int i = 0; i < size; i++) {
		( *pout)[0] = pin->real();
		( *pout)[1] = pin->imag();
		pout++;
		pin++;
	}
	fftw_execute( *m_fftplan);
	const fftw_complex *pin2 = m_pBufout;
	std::complex<double> *pout2 = &waveout[0];
	for(int i = 0; i < size; i++) {
		*pout2 = std::complex<double>(( *pin2)[0], ( *pin2)[1]);
		pout2++;
		pin2++;
	}
}

RFFT::RFFT(int length) : FFTBase(length) {
	m_pBufin = (double*)fftw_malloc(sizeof(double) * length);
	m_pBufout = (fftw_complex*)fftw_malloc(sizeof(fftw_complex) * (length / 2 + 1));
	*m_fftplan = fftw_plan_dft_r2c_1d(length, m_pBufin, m_pBufout, FFTW_ESTIMATE);
}
RFFT::~RFFT() {
	fftw_free(m_pBufin);
	fftw_free(m_pBufout);
}
void
RFFT::exec(const std::vector<double>& wavein,
		std::vector<std::complex<double> >& waveout) {
	int size = wavein.size();
	assert(size == length());
	waveout.resize(size / 2 + 1);
	const double *pin = &wavein[0];
	double *pout = m_pBufin;
	for(int i = 0; i < size; i++) {
		*pout++ = *pin++;
	}
	fftw_execute(*m_fftplan);
	const fftw_complex *pin2 = m_pBufout;
	std::complex<double> *pout2 = &waveout[0];
	for(int i = 0; i < size / 2 + 1; i++) {
		*pout2 = std::complex<double>(( *pin2)[0], ( *pin2)[1]);
		pout2++;
		pin2++;
	}
}

RIFFT::RIFFT(int length) : FFTBase(length) {
	m_pBufin = (fftw_complex*)fftw_malloc(sizeof(fftw_complex) * (length / 2 + 1));
	m_pBufout = (double*)fftw_malloc(sizeof(double) * length);
	*m_fftplan = fftw_plan_dft_c2r_1d(length, m_pBufin, m_pBufout, FFTW_ESTIMATE);
}
RIFFT::~RIFFT() {
	fftw_free(m_pBufin);
	fftw_free(m_pBufout);
}
void
RIFFT::exec(const std::vector<std::complex<double> >& wavein,
		std::vector<double>& waveout) {
	int size = length();
	assert((int)wavein.size() == length() / 2 + 1);
	waveout.resize(size);
	const std::complex<double> *pin = &wavein[0];
	fftw_complex *pout = m_pBufin;
	for(int i = 0; i < size / 2 + 1; i++) {
		( *pout)[0] = pin->real();
		( *pout)[1] = pin->imag();
		pout++;
		pin++;
	}
	fftw_execute( *m_fftplan);
	const double *pin2 = m_pBufout;
	double *pout2 = &waveout[0];
	for(int i = 0; i < size; i++) {
		*pout2++ = *pin2++;
	}
}
