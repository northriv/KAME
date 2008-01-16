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
/*
  Finite Impulse Response Filter
*/


#ifndef FIR_H
#define FIR_H

#include <vector>
#include <complex>
#include <fftw3.h>

//! FIR (Finite Impulse Response) Digital Filter.
//! Accelerated by FFT.
class FIR
{
public:
	//! makes coeff. for BPF. Window func. method.
	//! \param taps odd num. a number of taps
	//! \param bandwidth 0 to 1.0. the unit is sampling freq.
	//! \param center 0.0 to 1.0. the unit is sampling freq.
	FIR(int taps, double bandwidth, double center);
	~FIR();
	void exec(const double *src, double *dst, int len);
	int taps() const {return m_taps;}
	double bandWidth() const {return m_bandWidth;}
	double centerFreq() const {return m_centerFreq;}
private:
	fftw_plan m_rdftplan, m_ridftplan;
	double *m_pBufR;
	fftw_complex *m_pBufC;
	std::vector<double> m_firWnd;
	int m_fftLen, m_tapLen;
	const int m_taps;
	const double m_bandWidth;
	const double m_centerFreq;
};

#endif //FIR_H
