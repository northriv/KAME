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
#include "ar.h"


template <class Context>
YuleWalkerCousin<Context>::YuleWalkerCousin(tfuncARIC ic) : MEMStrict(), m_funcARIC(ic) {	
}

template <class Context>
double
YuleWalkerCousin<Context>::arAIC(double sigma2, int p, int t) {
	return log(sigma2) + 2 * (p + 1) / (double)t;
}
template <class Context>
double
YuleWalkerCousin<Context>::arAICc(double sigma2, int p, int t) {
	return log(sigma2) + 2 * (p + 2) / (double)(t - p - 3);
}
template <class Context>
double
YuleWalkerCousin<Context>::arHQ(double sigma2, int p, int t) {
	return log(sigma2) + 2.0 * (p + 1) * log(log(t) / log(2.0)) / (double)t;
}
template <class Context>
double
YuleWalkerCousin<Context>::arFPE(double sigma2, int p, int t) {
	return sigma2 * (t + p + 1) / (t - p - 1);
	
}
template <class Context>
double
YuleWalkerCousin<Context>::arMDL(double sigma2, int p, int t) {
	return t * log(sigma2) + (p + 1) * log(t);
}
template <class Context>
bool
YuleWalkerCousin<Context>::genSpectrum(const std::vector<std::complex<double> >& memin, std::vector<std::complex<double> >& memout,
	int t0, double torr, FFT::twindowfunc /*windowfunc*/, double windowlength) {
	unsigned int t = memin.size();
	unsigned int n = memout.size();

	if(t0 < 0)
		t0 += (-t0 / n + 1) * n;	
	std::vector<std::complex<double> > memphaseout(n);	
	MEMStrict::genSpectrum(memin, memphaseout, t0, torr, &FFT::windowFuncRect, 1.0);	
	
	int taps_div = t - 1;
	taps_div = std::max(taps_div / 10, 1);
	shared_ptr<Context> context(new Context);
	context->a.resize(t);
	std::fill(context->a.begin(), context->a.end(), 0.0);
	context->a[0] = 1.0;
	context->t = t;
	context->sigma2 = 0.0;
	for(unsigned int i = 0; i < t; i++) {
		context->sigma2 += std::norm(memin[i]) / (double)t;
	}
	context->p = 0;
	first(memin, context);
	unsigned int taps = 0;
	double ic = 1e99;
	for(unsigned int p = 0; p < t - 2; p++) {
		if(p % taps_div == 0) {
			m_contexts.push_back(shared_ptr<Context>(new Context(*context)));
			if(taps + taps_div/2 < p)
				break;
		}
		step(context);
		context->p++;
		if(context->sigma2 < 0)
			break;
		double new_ic = m_funcARIC(context->sigma2, context->p, t);
		if(new_ic < ic) {
			taps = p + 1;
			ic = new_ic;
		}
	}
	taps = std::min((int)lrint(windowlength * taps), (int)t - 2);
	int cidx = std::min((int)taps / taps_div, (int)m_contexts.size() - 1);
	context = m_contexts[cidx];
	m_contexts.clear();
	for(unsigned int p = cidx * taps_div; p < taps; p++) {
		ASSERT(context->p == p);
		step(context);
		context->p++;		
	}
	dbgPrint(formatString("MEM/AR: t=%d, taps=%d, IC_min=%g, IC=%g\n", t, taps, ic, m_funcARIC(context->sigma2, context->p, t)));

	std::vector<std::complex<double> > zfbuf(n), fftbuf(n);
	std::fill(zfbuf.begin(), zfbuf.end(), 0.0);
	for(int i = 0; i < taps + 1; i++) {
		zfbuf[i] = context->a[i];
	}
	m_fftN->exec(zfbuf, fftbuf);

	for(int i = 0; i < n; i++) {
		double z = t * context->sigma2 / (std::norm(fftbuf[i]));
		z = sqrt(std::max(z, 0.0) / std::norm(memphaseout[i]));
		memout[i] = memphaseout[i] * z;
	}
	genIFFT(memout);

	// Derivative of denominator.
	std::fill(zfbuf.begin(), zfbuf.end(), 0.0);
	for(int i = 0; i < taps + 1; i++) {
		zfbuf[i] = context->a[i] * (double)i * std::complex<double>(0, -1);
	}
	std::vector<std::complex<double> > fftbuf2(n);
	m_fftN->exec(zfbuf, fftbuf2);
	double dy_old = 0.0;
	for(int i = 0; i < n; i++) {
		double dy = std::real(fftbuf2[i] * std::conj(fftbuf[i]));
		if((dy_old < 0) && (dy > 0)) {
			double dx = - dy_old / (dy - dy_old);
			dx = std::max(0.0, std::min(dx, 1.0));
			std::complex<double> z = 0.0, xn = 1.0,
				x = std::polar(1.0, -2 * M_PI * (dx + i - 1) / (double)n);
			for(int i = 0; i < taps + 1; i++) {
				z += context->a[i] * xn;
				xn *= x;
			}
			double r = sqrt(std::max(t * context->sigma2 / std::norm(z), 0.0));
			m_peaks.push_back(std::pair<double, double>(dx + i - 1, r));
		}
		dy_old = dy;
	}
	return true;
}

template class YuleWalkerCousin<ARContext>;
template class YuleWalkerCousin<MEMBurgContext>;

void
MEMBurg::first(
	const std::vector<std::complex<double> >& memin, const shared_ptr<MEMBurgContext> &context) {
	unsigned int t = context->t;
	context->epsilon.resize(t);
	context->eta.resize(t);
	std::copy(memin.begin(), memin.end(), context->epsilon.begin());
	std::copy(memin.begin(), memin.end(), context->eta.begin());
}
void
MEMBurg::step(const shared_ptr<MEMBurgContext> &context) {
	unsigned int t = context->t;
	unsigned int p = context->p;
	std::complex<double> x = 0.0, y = 0.0;
	for(unsigned int i = p + 1; i < t; i++) {
		x += context->epsilon[i] * std::conj(context->eta[i-1]);
		y += std::norm(context->epsilon[i]) + std::norm(context->eta[i-1]);
	}
	std::complex<double> alpha = x / y;
	alpha *= -2;
	for(unsigned int i = t - 1; i >= p + 1; i--) {
		context->eta[i] = context->eta[i-1] + std::conj(alpha) * context->epsilon[i];
		context->epsilon[i] += alpha * context->eta[i-1];
	}
	std::vector<std::complex<double> > a_next(p + 2);
	for(unsigned int i = 0; i < p + 2; i++) {
		a_next[i] = context->a[i] + alpha * std::conj(context->a[p + 1 - i]);
	}
	std::copy(a_next.begin(), a_next.end(), context->a.begin());
	context->sigma2 *= 1 - std::norm(alpha);
}
void
YuleWalkerAR::first(
	const std::vector<std::complex<double> >& memin, const shared_ptr<ARContext> &context) {
	unsigned int t = context->t;
	m_rx.resize(t);
	autoCorrelation(memin, m_rx);
}
void
YuleWalkerAR::step(const shared_ptr<ARContext> &context) {
	unsigned int p = context->p;
	std::complex<double> delta = 0.0;
	for(unsigned int i = 0; i < p + 1; i++) {
		delta += context->a[i] * m_rx[p + 1 - i];
	}
	std::vector<std::complex<double> > a_next(p + 2);
	for(unsigned int i = 0; i < p + 2; i++) {
		a_next[i] = context->a[i] - delta/context->sigma2 * std::conj(context->a[p + 1 - i]);
	}
	std::copy(a_next.begin(), a_next.end(), context->a.begin());
	context->sigma2 += - std::norm(delta) / context->sigma2;
}

