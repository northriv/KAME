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
#ifndef AR_H_
#define AR_H_

#include "spectrumsolver.h"

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
		int t0, double tol, FFT::twindowfunc windowfunc, double windowlength);

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

#endif /*AR_H_*/
