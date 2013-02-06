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
#ifndef fftH
#define fftH
//---------------------------------------------------------------------------
#include "support.h"

#include <vector>
#include <complex>

#include <fftw3.h>

//! Wrapper class for fast Fourier transformation by FFTW.
class FFTBase {
public:
	FFTBase(int length);
	virtual ~FFTBase();
	//! Expand to appropriate length for better O(n log n) computation.
	static int fitLength(int length); 
	int length() const {return m_fftlen;}

	//for Window Func.
	typedef double (*twindowfunc)(double x);
	static double windowFuncRect(double x);
	static double windowFuncTri(double x);
	static double windowFuncHanning(double x);
	static double windowFuncHamming(double x);
	static double windowFuncFlatTop(double x);
	static double windowFuncBlackman(double x);
	static double windowFuncBlackmanHarris(double x);
	static double windowFuncKaiser(double x, double alpha);
	static double windowFuncKaiser1(double x);
	static double windowFuncKaiser2(double x);
	static double windowFuncKaiser3(double x);
	static double windowFuncFlatTopLong(double x);
	static double windowFuncFlatTopLongLong(double x);
	static double windowFuncHalfSin(double x);
protected:
	int m_fftlen;
	shared_ptr<fftw_plan> m_fftplan;
};

//! Wrapper class for FFTW.
class FFT : public FFTBase {
public:
	//! Create FFT plan.
	//! \param sign -1:FFT, 1:IFFT.
	//! \param length FFT length.
	FFT(int sign, int length);
	virtual ~FFT();

	void exec(const std::vector<std::complex<double> >& wavein,
		std::vector<std::complex<double> >& waveout);
private:
	fftw_complex *m_pBufin, *m_pBufout;
};

//! Read Data FFT(DFT).
class RFFT : public FFTBase {
public:
	//! Create real data FFT plan.
	//! \param length FFT length.
	RFFT(int length);
	virtual ~RFFT();

	void exec(const std::vector<double>& wavein,
		std::vector<std::complex<double> >& waveout);
private:
	double *m_pBufin;
	fftw_complex *m_pBufout;
};

//! Read Data IFFT(IDFT).
class RIFFT : public FFTBase {
public:
	//! Create real data IFFT plan.
	//! \param length FFT length.
	RIFFT(int length);
	virtual ~RIFFT();

	void exec(const std::vector<std::complex<double> >& wavein,
		std::vector<double>& waveout);
private:
	double *m_pBufout;
	fftw_complex *m_pBufin;
};
#endif
