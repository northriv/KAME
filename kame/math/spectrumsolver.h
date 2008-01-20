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
#ifndef spectrumsolverH
#define spectrumsolverH
//---------------------------------------------------------------------------
#include "support.h"

#include <vector>
#include <complex>
#include <deque>

#include <fft.h>

class SpectrumSolver {
public:
	SpectrumSolver();
	virtual ~SpectrumSolver();
	
	bool exec(const std::vector<std::complex<double> >& memin, std::vector<std::complex<double> >& memout,
		int t0, double tol, FFT::twindowfunc windowfunc, double windowlength);
	const std::vector<std::complex<double> >& ifft() const {return m_ifft;}
	const std::vector<std::pair<double, double> >& peaks() const {return m_peaks;}
	
	typedef double (*tfuncIC)(double sigma2, int p, int t);

	//! Akachi's information criterion.
	//! \arg loglikelifood ln(L).
	//! \arg k # of parameters.
	//! \arg n # of samples.
	static double icAIC(double loglikelifood, int k, int n);
	//! Corrected Akachi's information criterion.
	//! \sa icAIC
	static double icAICc(double loglikelifood, int k, int n);
	//! Hannan-Quinn information criterion.
	//! \sa icAIC
	static double icHQ(double loglikelifood, int k, int n);	
	//! Minimum Description Length.
	//! \sa icAIC
	static double icMDL(double loglikelifood, int k, int n);	
protected:
	virtual bool genSpectrum(const std::vector<std::complex<double> >& memin,
		std::vector<std::complex<double> >& memout,
		int t0, double tol, FFT::twindowfunc windowfunc, double windowlength) = 0;
	std::vector<std::complex<double> > m_ifft;
	std::vector<std::pair<double, double> > m_peaks;
	void genIFFT(std::vector<std::complex<double> >& wavein);

	shared_ptr<FFT> m_fftN, m_ifftN;
	int fftlen() const {return m_ifftN->length();}
	
	//! For autocorrelation.
	shared_ptr<FFT> m_fftRX, m_ifftRX;
	void autoCorrelation(const std::vector<std::complex<double> >&wave,
		std::vector<std::complex<double> >&corr);
};

class FFTSolver : public SpectrumSolver {
public:
	FFTSolver() : SpectrumSolver()  {}
	virtual ~FFTSolver() {}
	
	static double windowLength(int tdlen, int t0, double windowlength);
protected:
	virtual bool genSpectrum(const std::vector<std::complex<double> >& memin,
		std::vector<std::complex<double> >& memout,
		int t0, double tol, FFT::twindowfunc windowfunc, double windowlength);
};

class MEMStrict : public SpectrumSolver {
public:
	virtual ~MEMStrict();
protected:
	virtual bool genSpectrum(const std::vector<std::complex<double> >& memin,
		std::vector<std::complex<double> >& memout,
		int t0, double tol, FFT::twindowfunc windowfunc, double windowlength);
private:
	void setup(unsigned int t, unsigned int n);
	double stepMEM(const std::vector<std::complex<double> >& memin, std::vector<std::complex<double> >& memout,
		double alpha, double sigma, int t0, double tol);
	void solveZ(double tol);

	std::vector<std::complex<double> > m_lambda, m_accumDY, m_accumDYFT;
	shared_ptr<FFT> m_fftT;
	std::vector<double> m_accumG2;
	double m_accumZ;
};

#endif
