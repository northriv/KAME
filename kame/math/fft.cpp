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
}
FFTBase::~FFTBase() {
}
FFT::FFT(int sign, int length) : FFTBase(length) {
	//Plan creation uses scratch buffers only for addressing/alignment
	//(FFTW_ESTIMATE never reads them); freed right after.  exec() then runs
	//the plan on per-call buffers of matching (fftw_malloc) alignment.
	fftw_complex *pbufin = (fftw_complex*)fftw_malloc(sizeof(fftw_complex) * length);
	fftw_complex *pbufout = (fftw_complex*)fftw_malloc(sizeof(fftw_complex) * length);
	m_fftplan.reset(fftw_plan_dft_1d(length, pbufin, pbufout,
		(sign > 0) ? FFTW_BACKWARD : FFTW_FORWARD, FFTW_ESTIMATE), fftw_destroy_plan);
	fftw_free(pbufin);
	fftw_free(pbufout);
}
void
FFT::exec(const std::vector<std::complex<double> >& wavein,
		std::vector<std::complex<double> >& waveout) const {
	int size = wavein.size();
	assert(size == length());
	waveout.resize(size);
	fftw_complex *pbufin = (fftw_complex*)fftw_malloc(sizeof(fftw_complex) * size);
	fftw_complex *pbufout = (fftw_complex*)fftw_malloc(sizeof(fftw_complex) * size);
	const std::complex<double> *pin = &wavein[0];
	fftw_complex *pout = pbufin;
	for(int i = 0; i < size; i++) {
		( *pout)[0] = pin->real();
		( *pout)[1] = pin->imag();
		pout++;
		pin++;
	}
	fftw_execute_dft( m_fftplan.get(), pbufin, pbufout);
	const fftw_complex *pin2 = pbufout;
	std::complex<double> *pout2 = &waveout[0];
	for(int i = 0; i < size; i++) {
		*pout2 = std::complex<double>(( *pin2)[0], ( *pin2)[1]);
		pout2++;
		pin2++;
	}
	fftw_free(pbufin);
	fftw_free(pbufout);
}

RFFT::RFFT(int length) : FFTBase(length) {
	double *pbufin = (double*)fftw_malloc(sizeof(double) * length);
	fftw_complex *pbufout = (fftw_complex*)fftw_malloc(sizeof(fftw_complex) * (length / 2 + 1));
	m_fftplan.reset(fftw_plan_dft_r2c_1d(length, pbufin, pbufout, FFTW_ESTIMATE), fftw_destroy_plan);
	fftw_free(pbufin);
	fftw_free(pbufout);
}
void
RFFT::exec(const std::vector<double>& wavein,
		std::vector<std::complex<double> >& waveout) const {
	int size = wavein.size();
	assert(size == length());
	waveout.resize(size / 2 + 1);
	double *pbufin = (double*)fftw_malloc(sizeof(double) * size);
	fftw_complex *pbufout = (fftw_complex*)fftw_malloc(sizeof(fftw_complex) * (size / 2 + 1));
	const double *pin = &wavein[0];
	double *pout = pbufin;
	for(int i = 0; i < size; i++) {
		*pout++ = *pin++;
	}
	fftw_execute_dft_r2c( m_fftplan.get(), pbufin, pbufout);
	const fftw_complex *pin2 = pbufout;
	std::complex<double> *pout2 = &waveout[0];
	for(int i = 0; i < size / 2 + 1; i++) {
		*pout2 = std::complex<double>(( *pin2)[0], ( *pin2)[1]);
		pout2++;
		pin2++;
	}
	fftw_free(pbufin);
	fftw_free(pbufout);
}

RIFFT::RIFFT(int length) : FFTBase(length) {
	fftw_complex *pbufin = (fftw_complex*)fftw_malloc(sizeof(fftw_complex) * (length / 2 + 1));
	double *pbufout = (double*)fftw_malloc(sizeof(double) * length);
	m_fftplan.reset(fftw_plan_dft_c2r_1d(length, pbufin, pbufout, FFTW_ESTIMATE), fftw_destroy_plan);
	fftw_free(pbufin);
	fftw_free(pbufout);
}
void
RIFFT::exec(const std::vector<std::complex<double> >& wavein,
		std::vector<double>& waveout) const {
	int size = length();
	assert((int)wavein.size() == length() / 2 + 1);
	waveout.resize(size);
	fftw_complex *pbufin = (fftw_complex*)fftw_malloc(sizeof(fftw_complex) * (size / 2 + 1));
	double *pbufout = (double*)fftw_malloc(sizeof(double) * size);
	const std::complex<double> *pin = &wavein[0];
	fftw_complex *pout = pbufin;
	for(int i = 0; i < size / 2 + 1; i++) {
		( *pout)[0] = pin->real();
		( *pout)[1] = pin->imag();
		pout++;
		pin++;
	}
	fftw_execute_dft_c2r( m_fftplan.get(), pbufin, pbufout);
	const double *pin2 = pbufout;
	double *pout2 = &waveout[0];
	for(int i = 0; i < size; i++) {
		*pout2++ = *pin2++;
	}
	fftw_free(pbufin);
	fftw_free(pbufout);
}
