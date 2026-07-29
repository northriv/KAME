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
/*
  Finite Impulse Response Filter
*/


#ifndef FIR_H
#define FIR_H

#include "support.h"
#include <vector>
#include <complex>
#include <type_traits>
#include <fftw3.h>

//! FIR (Finite Impulse Response) Digital Filter.
//! Accelerated by FFT.
class DECLSPEC_KAME FIR {
public:
	//! makes coeff. for BPF. Window func. method.
	//! \param taps odd num. a number of taps
	//! \param bandwidth 0 to 1.0. the unit is sampling freq.
	//! \param center 0.0 to 1.0. the unit is sampling freq.
	FIR(int taps, double bandwidth, double center);
	//! Apply the filter to \a src (length \a len) into \a dst.
	//! \a const: scratch buffers are allocated per call and the immutable
	//! r2c/c2r plans are run via fftw_execute_dft_r2c/c2r, so the object
	//! carries no mutable state and is safe to share across STM snapshots
	//! and to run from multiple threads concurrently.
	void exec(const double *src, double *dst, int len) const;
	int taps() const {return m_taps;}
	double bandWidth() const {return m_bandWidth;}
	double centerFreq() const {return m_centerFreq;}
private:
	//! r2c / c2r plans: shared & immutable, destroyed once via shared_ptr deleter.
	shared_ptr<std::remove_pointer<fftw_plan>::type> m_rdftplan, m_ridftplan;
	std::vector<double> m_firWnd; //!< frequency-domain filter coefficients (immutable after ctor).
	int m_fftLen, m_tapLen;
	const int m_taps;
	const double m_bandWidth;
	const double m_centerFreq;
};

#endif //FIR_H
