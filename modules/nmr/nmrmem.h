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

#include <vector>
#include <complex>
#include <deque>
#include <fftw.h>

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
private:
	int m_fftlen;
	shared_ptr<fftw_plan> m_fftplan;
	shared_ptr<std::vector<fftw_complex> > m_bufin, m_bufout;
};

class SpectrumSolver {
public:
	SpectrumSolver();
	virtual ~SpectrumSolver();
	
	bool exec(const std::vector<std::complex<double> >& memin, std::vector<std::complex<double> >& memout,
		int t0, double torr, FFT::twindowfunc windowfunc, double windowlength);
	const std::vector<std::complex<double> >& ifft() const {return m_ifft;}
protected:
	virtual bool genSpectrum(const std::vector<std::complex<double> >& memin,
		std::vector<std::complex<double> >& memout,
		int t0, double torr, FFT::twindowfunc windowfunc, double windowlength) = 0;
	std::vector<std::complex<double> > m_ifft;
	void genIFFT(std::vector<std::complex<double> >& wavein);

	shared_ptr<FFT> m_fftN, m_ifftN;
	int fftlen() const {return m_ifftN->length();}
};

class FFTSolver : public SpectrumSolver {
public:
	FFTSolver() : SpectrumSolver()  {}
	virtual ~FFTSolver() {}
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
