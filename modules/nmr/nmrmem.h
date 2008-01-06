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
#include <deque>

class SpectrumSolver {
public:
	SpectrumSolver();
	virtual ~SpectrumSolver();

	//for Window Func.
	typedef double (*twindowfunc)(double x);
	
	bool exec(const std::vector<fftw_complex>& memin, std::vector<fftw_complex>& memout,
		int t0, double torr, twindowfunc windowfunc, double windowlength);
	const std::vector<fftw_complex>& ifft() const {return m_ifft;}

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
	static void fftw2std(const std::vector<fftw_complex>& wavein, std::vector<std::complex<double> > &waveout);
	static void std2fftw(const std::vector<std::complex<double> >& wavein, std::vector<fftw_complex> &waveout);
protected:
	virtual bool genSpectrum(const std::vector<fftw_complex>& memin, std::vector<fftw_complex>& memout,
		int t0, double torr, twindowfunc windowfunc, double windowlength) = 0;
	void clearFTBuf(std::vector<fftw_complex> &buf);
	void genIFFT(std::vector<fftw_complex>& wavein);
	std::vector<fftw_complex> m_ifft;
	fftw_plan m_fftplanN, m_ifftplanN;	
	int fftlen() const {return m_ifft.size();}
};

class FFTSolver : public SpectrumSolver {
public:
	FFTSolver() : SpectrumSolver()  {}
	virtual ~FFTSolver() {}
protected:
	virtual bool genSpectrum(const std::vector<fftw_complex>& memin, std::vector<fftw_complex>& memout,
		int t0, double torr, twindowfunc windowfunc, double windowlength);
};

class MEMStrict : public SpectrumSolver {
public:
	virtual ~MEMStrict();
protected:
	virtual bool genSpectrum(const std::vector<fftw_complex>& memin, std::vector<fftw_complex>& memout,
		int t0, double torr, twindowfunc windowfunc, double windowlength);
private:
	void setup(unsigned int t, unsigned int n);
	double stepMEM(const std::vector<fftw_complex>& memin, std::vector<fftw_complex>& memout,
		double alpha, double sigma, int t0, double torr);
	void solveZ(double torr);

	std::vector<fftw_complex> m_lambda, m_accumDY, m_accumDYFT;
	fftw_plan m_fftplanT;
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
	virtual bool genSpectrum(const std::vector<fftw_complex>& memin, std::vector<fftw_complex>& memout,
		int t0, double torr, twindowfunc windowfunc, double windowlength);

	virtual void first(
		const std::vector<std::complex<double> >& memin, const shared_ptr<Context> &context, twindowfunc windowfunc) = 0;
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
		const std::vector<std::complex<double> >& memin, const shared_ptr<MEMBurgContext> &context, twindowfunc windowfunc);
	virtual void step(const shared_ptr<MEMBurgContext> &context);
private:

};

class YuleWalkerAR : public YuleWalkerCousin<ARContext> {
public:
	YuleWalkerAR(tfuncARIC ic) : YuleWalkerCousin<ARContext>(ic) {}
	virtual ~YuleWalkerAR() {}
protected:
	virtual void first(
		const std::vector<std::complex<double> >& memin, const shared_ptr<ARContext> &context, twindowfunc windowfunc);
	virtual void step(const shared_ptr<ARContext> &context);
private:
	std::vector<std::complex<double> > m_rx;
};

#endif
