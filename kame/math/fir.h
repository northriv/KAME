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

class FIR
{
public:
 	FIR();
	~FIR();
	//! makes coeff. for BPF. Window func. method.
	//! \param taps odd num. a number of taps
	//! \param bandwidth 0 to 1.0. the unit is sampling freq.
	//! \param center 0.0 to 1.0. the unit is sampling freq.
	int setupBPF(int taps, double bandwidth, double center);
	//! \param src a pointer to src
	//! \param dst a pointer to dst
	//! \param len length
	int doFIR(double *src, double *dst, int len);
private:
 	float oneFIR(float x, float *z, float *coeff, int taps);
	int m_Taps; ///< a number of taps divided by 2
	double m_BandWidth;
	double m_Center;
	std::vector<float> m_Coeff, m_Z; ///< coeff. buffer
};

#endif //FIR_H
