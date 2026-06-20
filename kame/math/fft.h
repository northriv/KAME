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
#ifndef fftH
#define fftH
//---------------------------------------------------------------------------
#include "support.h"

#include <vector>
#include <complex>
#include <type_traits>

#include <fftw3.h>

//! Wrapper class for fast Fourier transformation by FFTW.
class DECLSPEC_KAME FFTBase {
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
	//! FFTW plan: shared and immutable after construction; destroyed exactly
	//! once via the shared_ptr deleter (fftw_destroy_plan).  Holding it by
	//! shared_ptr lets exec() stay const (the plan is only read), so an FFT
	//! object carries no mutable state and is safe to share across STM
	//! snapshots / run from multiple threads concurrently.
	shared_ptr<std::remove_pointer<fftw_plan>::type> m_fftplan;
};

//! Wrapper class for FFTW.
class DECLSPEC_KAME FFT : public FFTBase {
public:
	//! Create FFT plan.
	//! \param sign -1:FFT, 1:IFFT.
	//! \param length FFT length.
	FFT(int sign, int length);

	//! Transform \a wavein into \a waveout.
	//! \a const: scratch buffers are allocated per call and the immutable
	//! plan is executed via fftw_execute_dft, so no per-object state is
	//! mutated (FFTW guarantees the new-array execute is thread-safe on a
	//! shared plan as long as the buffers differ per call).
	void exec(const std::vector<std::complex<double> >& wavein,
		std::vector<std::complex<double> >& waveout) const;
};

//! Read Data FFT(DFT).
class DECLSPEC_KAME RFFT : public FFTBase {
public:
	//! Create real data FFT plan.
	//! \param length FFT length.
	RFFT(int length);

	void exec(const std::vector<double>& wavein,
		std::vector<std::complex<double> >& waveout) const;
};

//! Read Data IFFT(IDFT).
class DECLSPEC_KAME RIFFT : public FFTBase {
public:
	//! Create real data IFFT plan.
	//! \param length FFT length.
	RIFFT(int length);

	void exec(const std::vector<std::complex<double> >& wavein,
		std::vector<double>& waveout) const;
};
#endif
