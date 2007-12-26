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
#ifndef nmrmemH
#define nmrmemH
//---------------------------------------------------------------------------
#include "support.h"

#include <fftw.h>
#include <vector>
#include <complex>

class NMRMEM {
public:
	~NMRMEM();
	bool exec(const std::vector<fftw_complex>& memin, std::vector<fftw_complex>& memout,
		int t0, double torr);
	const std::vector<fftw_complex>& ifft() const {return m_ifft;}
private:
	void setup(unsigned int t, unsigned int n);
	void clearFTBuf(std::vector<fftw_complex> &buf);
	double stepMEM(const std::vector<fftw_complex>& memin, std::vector<fftw_complex>& memout,
		double alpha, double sigma, int t0, double torr);
	void solveZ(double torr);

	std::vector<fftw_complex> m_lambda, m_ifft, m_accumDY, m_accumDYFT;
	fftw_plan m_ifftplanN, m_fftplanN, m_fftplanT;
	std::vector<double> m_accumG2;
	double m_accumZ;
};

#endif
