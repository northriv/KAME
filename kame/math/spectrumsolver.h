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
	//! \return (power, index) in descending order.
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
	
	static double windowLength(int tdlen, int t0, double windowlength);
protected:
	virtual bool genSpectrum(const std::vector<std::complex<double> >& memin,
		std::vector<std::complex<double> >& memout,
		int t0, double tol, FFT::twindowfunc windowfunc, double windowlength) = 0;
	std::vector<std::complex<double> > m_ifft;
	std::vector<std::pair<double, double> > m_peaks;
	
	void window(int t, int t0, FFT::twindowfunc windowfunc,
		double windowlength, std::vector<double> &window);
	
	void genIFFT(const std::vector<std::complex<double> >& wavein);
	//! Least-square phase estimation.
	//! \return coeff.
	double lspe(const std::vector<std::complex<double> >& wavein, int origin,
		const std::vector<double>& psd, std::vector<std::complex<double> >& waveout,
		double tol, bool powfit, FFT::twindowfunc windowfunc);
	//! \return err.
	double stepLSPE(const std::vector<std::complex<double> >& wavein, int origin,
		const std::vector<double>& psd, std::vector<std::complex<double> >& waveout,
		bool powfit, double &coeff, const std::vector<double> &weight);

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

template <class T, class X, bool IsXConfigurable = false>
class CompositeSpectrumSolver : public T {
public:
protected:
	virtual bool genSpectrum(const std::vector<std::complex<double> >& memin,
		std::vector<std::complex<double> >& memout,
		int t0, double tol, FFT::twindowfunc windowfunc, double windowlength) {
		m_preprocessor.exec(memin, memout, t0, tol,
			IsXConfigurable ? windowfunc : &FFT::windowFuncRect,
			IsXConfigurable ? windowlength : 1.0);
		int t = memin.size();
		int n = memout.size();
		int t0a = t0;
		if(t0a < 0)
			t0a += (-t0a / n + 1) * n;
		std::vector<std::complex<double> > nsin(t);
		for(int i = 0; i < t; i++) {
			int j = (t0a + i) % n;
			nsin[i] = m_preprocessor.ifft()[j];
		}
		return T::genSpectrum(nsin, memout, t0, tol,
			!IsXConfigurable ? windowfunc : &FFT::windowFuncRect,
			!IsXConfigurable ? windowlength : 1.0);
	}
	X m_preprocessor;
};

#endif
