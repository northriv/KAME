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
#ifndef memH
#define memH
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
		int t0, double torr, FFT::twindowfunc windowfunc, double windowlength);
	const std::vector<std::complex<double> >& ifft() const {return m_ifft;}
	const std::vector<std::pair<double, double> >& poles() const {return m_poles;}
protected:
	virtual bool genSpectrum(const std::vector<std::complex<double> >& memin,
		std::vector<std::complex<double> >& memout,
		int t0, double torr, FFT::twindowfunc windowfunc, double windowlength) = 0;
	std::vector<std::complex<double> > m_ifft;
	std::vector<std::pair<double, double> > m_poles;
	void genIFFT(std::vector<std::complex<double> >& wavein);

	shared_ptr<FFT> m_fftN, m_ifftN;
	int fftlen() const {return m_ifftN->length();}
};

class FFTSolver : public SpectrumSolver {
public:
	FFTSolver() : SpectrumSolver()  {}
	virtual ~FFTSolver() {}
	
	static double windowLength(int tdlen, int t0, double windowlength);
protected:
	virtual bool genSpectrum(const std::vector<std::complex<double> >& memin,
		std::vector<std::complex<double> >& memout,
		int t0, double torr, FFT::twindowfunc windowfunc, double windowlength);
};

class MEMStrict : public SpectrumSolver {
public:
	virtual ~MEMStrict();
protected:
	virtual bool genSpectrum(const std::vector<std::complex<double> >& memin,
		std::vector<std::complex<double> >& memout,
		int t0, double torr, FFT::twindowfunc windowfunc, double windowlength);
private:
	void setup(unsigned int t, unsigned int n);
	double stepMEM(const std::vector<std::complex<double> >& memin, std::vector<std::complex<double> >& memout,
		double alpha, double sigma, int t0, double torr);
	void solveZ(double torr);

	std::vector<std::complex<double> > m_lambda, m_accumDY, m_accumDYFT;
	shared_ptr<FFT> m_fftT;
	std::vector<double> m_accumG2;
	double m_accumZ;
};

template <class Context>
class YuleWalkerCousin : public MEMStrict {
public:
	typedef double (*tfuncARIC)(double sigma2, int p, int t);

	YuleWalkerCousin(tfuncARIC ic);
	virtual ~YuleWalkerCousin() {}

	//! Akachi's information criterion.
	static double arAIC(double sigma2, int p, int t);
	//! Corrected Akachi's information criterion.
	static double arAICc(double sigma2, int p, int t);
	//! Hannan-Quinn information criterion.
	static double arHQ(double sigma2, int p, int t);	
	//! Final Prediction Error criterion.
	static double arFPE(double sigma2, int p, int t);
	//! Minimum Description Length.
	static double arMDL(double sigma2, int p, int t);
protected:
	virtual bool genSpectrum(const std::vector<std::complex<double> >& memin, std::vector<std::complex<double> >& memout,
		int t0, double torr, FFT::twindowfunc windowfunc, double windowlength);

	virtual void first(
		const std::vector<std::complex<double> >& memin, const shared_ptr<Context> &context, FFT::twindowfunc windowfunc) = 0;
	virtual void step(const shared_ptr<Context> &context) = 0;
	std::deque<shared_ptr<Context> > m_contexts;
private:
	const tfuncARIC m_funcARIC;
};

struct ARContext {
	ARContext() {}
	ARContext(const ARContext &c) : a(c.a), sigma2(c.sigma2), p(c.p), t(c.t) {}
	std::vector<std::complex<double> > a;
	double sigma2;
	unsigned int p;
	unsigned int t;
};
struct MEMBurgContext : public ARContext {
	MEMBurgContext() : ARContext() {}
	MEMBurgContext(const MEMBurgContext &c) : ARContext(c), eta(c.eta), epsilon(c.epsilon) {}
	std::vector<std::complex<double> > eta, epsilon;
};

class MEMBurg : public YuleWalkerCousin<MEMBurgContext> {
public:
	MEMBurg(tfuncARIC ic) : YuleWalkerCousin<MEMBurgContext>(ic) {}
	virtual ~MEMBurg() {}
protected:
	virtual void first(
		const std::vector<std::complex<double> >& memin, const shared_ptr<MEMBurgContext> &context, FFT::twindowfunc windowfunc);
	virtual void step(const shared_ptr<MEMBurgContext> &context);
private:

};

class YuleWalkerAR : public YuleWalkerCousin<ARContext> {
public:
	YuleWalkerAR(tfuncARIC ic) : YuleWalkerCousin<ARContext>(ic) {}
	virtual ~YuleWalkerAR() {}
protected:
	virtual void first(
		const std::vector<std::complex<double> >& memin, const shared_ptr<ARContext> &context, FFT::twindowfunc windowfunc);
	virtual void step(const shared_ptr<ARContext> &context);
private:
	std::vector<std::complex<double> > m_rx;
};

#endif
