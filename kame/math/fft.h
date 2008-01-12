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
#ifndef fftH
#define fftH
//---------------------------------------------------------------------------
#include "support.h"

#include <vector>
#include <complex>

#include <fftw3.h>
//! Wrapper class for FFTW.
class FFT {
public:
	//! Create FFT plan.
	//! \arg sign -1:FFT, 1:IFFT.
	//! \arg length FFT length.
	//! \arg fit_length Expand to appropriate length for O(n log n) computation.
	FFT(int sign, int length);
	~FFT();
	static int fitLength(int length); 
	int length() const {return m_fftlen;}
	void exec(const std::vector<std::complex<double> >& wavein,
		std::vector<std::complex<double> >& waveout);

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
private:
	int m_fftlen;
	shared_ptr<fftw_plan> m_fftplan;
	fftw_complex *m_pBufin, *m_pBufout;
};

#endif
