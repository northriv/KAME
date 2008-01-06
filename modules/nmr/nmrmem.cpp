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
#include "nmrmem.h"

#include <fftw.h>
#include <gsl/gsl_sf.h>
#define lambertW0 gsl_sf_lambert_W0
#define bessel_i0 gsl_sf_bessel_I0

int
FFT::fitLength(int length0) {
	int length = lrint(pow(2.0, (ceil(log(length0) / log(2.0)))));		
	length = std::min(length, (int)lrint(pow(2.0, (ceil(log(length0 / 3.0) / log(2.0))))) * 3);		
	length = std::min(length, (int)lrint(pow(2.0, (ceil(log(length0 / 5.0) / log(2.0))))) * 5);		
	length = std::min(length, (int)lrint(pow(2.0, (ceil(log(length0 / 7.0) / log(2.0))))) * 7);		
	length = std::min(length, (int)lrint(pow(2.0, (ceil(log(length0 / 9.0) / log(2.0))))) * 9);		
	length = std::min(length, (int)lrint(pow(2.0, (ceil(log(length0 / 11.0) / log(2.0))))) * 11);		
	length = std::min(length, (int)lrint(pow(2.0, (ceil(log(length0 / 13.0) / log(2.0))))) * 13);		
	length = std::min(length, (int)lrint(pow(2.0, (ceil(log(length0 / 15.0) / log(2.0))))) * 15);		
	length = std::min(length, (int)lrint(pow(2.0, (ceil(log(length0 / 21.0) / log(2.0))))) * 21);		
	length = std::min(length, (int)lrint(pow(2.0, (ceil(log(length0 / 25.0) / log(2.0))))) * 25);		
	length = std::min(length, (int)lrint(pow(2.0, (ceil(log(length0 / 27.0) / log(2.0))))) * 27);
	length = std::min(length, (int)lrint(pow(2.0, (ceil(log(length0 / 33.0) / log(2.0))))) * 33);
	length = std::min(length, (int)lrint(pow(2.0, (ceil(log(length0 / 35.0) / log(2.0))))) * 35);
	ASSERT(length0 <= length);
	dbgPrint(formatString("FFT using L=%d\n", length));
	return length;
}

FFT::FFT(int sign, int length) {
	m_fftlen = length;
	m_bufin.reset(new std::vector<fftw_complex>(length));
	m_bufout.reset(new std::vector<fftw_complex>(length));
	m_fftplan.reset(new fftw_plan);
	*m_fftplan = fftw_create_plan(length, (sign > 0) ? FFTW_BACKWARD : FFTW_FORWARD, FFTW_ESTIMATE);
}
FFT::~FFT() {
	fftw_destroy_plan(*m_fftplan);
}
void
FFT::exec(const std::vector<std::complex<double> >& wavein,
		std::vector<std::complex<double> >& waveout) {
	int size = wavein.size();
	ASSERT(size == length());
	waveout.resize(size);
	const std::complex<double> *pin = &wavein[0];
	fftw_complex *pout = &m_bufin->at(0);
	for(int i = 0; i < size; i++) {
		pout->re = pin->real();
		pout->im = pin->imag();
		pout++;
		pin++;
	}
	fftw_one(*m_fftplan, &m_bufin->at(0), &m_bufout->at(0));
	const fftw_complex *pin2 = &m_bufout->at(0);
	std::complex<double> *pout2 = &waveout[0];
	for(int i = 0; i < size; i++) {
		*pout2 = std::complex<double>(pin2->re, pin2->im);
		pout2++;
		pin2++;
	}
}

double FFT::windowFuncRect(double x) {
	return (fabs(x) <= 0.5) ? 1 : 0;
//	return 1.0;
}
double FFT::windowFuncTri(double x) {
	return std::max(0.0, 1.0 - 2.0 * fabs(x));
}
double FFT::windowFuncHanning(double x) {
	if (fabs(x) >= 0.5)
		return 0.0;
	return 0.5 + 0.5*cos(2*PI*x);
}
double FFT::windowFuncHamming(double x) {
	if (fabs(x) >= 0.5)
		return 0.0;
	return 0.54 + 0.46*cos(2*PI*x);
}
double FFT::windowFuncBlackman(double x) {
	if (fabs(x) >= 0.5)
		return 0.0;
	return 0.42323+0.49755*cos(2*PI*x)+0.07922*cos(4*PI*x);
}
double FFT::windowFuncBlackmanHarris(double x) {
	if (fabs(x) >= 0.5)
		return 0.0;
	return 0.35875+0.48829*cos(2*PI*x)+0.14128*cos(4*PI*x)+0.01168*cos(6*PI*x);
}
double FFT::windowFuncFlatTop(double x) {
	return windowFuncHamming(x)*((fabs(x) < 1e-4) ? 1 : sin(4*PI*x)/(4*PI*x));
}
double FFT::windowFuncKaiser(double x, double alpha) {
	if (fabs(x) >= 0.5)
		return 0.0;
	x *= 2;
	x = sqrt(std::max(1 - x*x, 0.0));
	return bessel_i0(PI*alpha*x) / bessel_i0(PI*alpha);
}
double FFT::windowFuncKaiser1(double x) {
	return windowFuncKaiser(x, 3.0);
}
double FFT::windowFuncKaiser2(double x) {
	return windowFuncKaiser(x, 7.2);
}
double FFT::windowFuncKaiser3(double x) {
	return windowFuncKaiser(x, 15.0);
}

SpectrumSolver::SpectrumSolver() {} 
SpectrumSolver::~SpectrumSolver() {}

void
SpectrumSolver::genIFFT(std::vector<std::complex<double> >& wavein) {
	m_ifftN->exec(wavein, m_ifft);
	int n = wavein.size();
	double k = 1.0 / n;
	std::complex<double> *pifft = &m_ifft[0];
	for(unsigned int i = 0; i < n; i++) {
		*pifft *= k;
		pifft++;
	}	
}

bool
SpectrumSolver::exec(const std::vector<std::complex<double> >& memin, std::vector<std::complex<double> >& memout,
	int t0, double torr, FFT::twindowfunc windowfunc, double windowlength) {
	unsigned int t = memin.size();
	unsigned int n = memout.size();
	ASSERT(n == FFT::fitLength(n));
	if (m_ifft.size() != n) {
		m_fftN.reset(new FFT(-1, n));		
		m_ifftN.reset(new FFT(1, n));		
		m_ifft.resize(n);
	}
	return genSpectrum(memin, memout, t0, torr, windowfunc, windowlength);
}


bool
FFTSolver::genSpectrum(const std::vector<std::complex<double> >& fftin, std::vector<std::complex<double> >& fftout,
	int t0, double /*torr*/, FFT::twindowfunc windowfunc, double windowlength) {
	unsigned int t = fftin.size();
	unsigned int n = fftout.size();
	int t0a = t0;
	if(t0a < 0)
		t0a += (-t0a / n + 1) * n;

	double wk = 0.5 / (std::max(-t0, (int)t + t0) * windowlength);
	std::fill(m_ifft.begin(), m_ifft.end(), 0.0);
	for(int i = 0; i < t; i++) {
		double w = windowfunc((i + t0) * wk);
		m_ifft[(t0a + i) % n] = fftin[i] * w;
	}
	m_fftN->exec(m_ifft, fftout);
	return true;
}

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
	int t0, double torr, FFT::twindowfunc windowfunc, double windowlength) {
	windowfunc = &FFT::windowFuncTri;
	
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
		context->sigma2 += std::norm(memin[i]);
	}
	context->p = 0;
	first(memin, context, windowfunc);
	unsigned int taps = 0;
//	sigma2 /= t;
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
	std::complex<double> *pout = &zfbuf[0];
	for(int i = 0; i < taps + 1; i++) {
		*pout = context->a[i];
		pout++;
	}
	m_fftN->exec(zfbuf, fftbuf);
	
	std::complex<double> *pin = &fftbuf[0];
	for(unsigned int i = 0; i < n; i++) {
		double z = context->sigma2 / (std::norm(*pin));
		z = sqrt(std::max(z, 0.0) / std::norm(memphaseout[i]));
		memout[i] = memphaseout[i] * z;
		pin++;
	}
	genIFFT(memout);
	return true;
}

template class YuleWalkerCousin<ARContext>;
template class YuleWalkerCousin<MEMBurgContext>;

void
MEMBurg::first(
	const std::vector<std::complex<double> >& memin, const shared_ptr<MEMBurgContext> &context, FFT::twindowfunc /*windowfunc*/) {
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
	const std::vector<std::complex<double> >& memin, const shared_ptr<ARContext> &context, FFT::twindowfunc windowfunc) {
	unsigned int t = context->t;
	m_rx.resize(t);
	std::fill(m_rx.begin(), m_rx.end(), 0.0);
	for(unsigned int p = 0; p <= t - 1; p++) {
		for(unsigned int i = 0; i < t - p; i++) {
			m_rx[p] += std::conj(memin[i]) * memin[i+p];
		}
		m_rx[p] *= windowfunc(0.5*p/(double)t) * t / (t - p);
	}		
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

MEMStrict::~MEMStrict() {
}

void
MEMStrict::setup(unsigned int t, unsigned int n) {
	if (m_lambda.size() != t) {
		m_fftT.reset(new FFT(-1, t));
		m_accumDY.resize(t);
		m_accumDYFT.resize(t);
		m_lambda.resize(t);
		m_accumG2.resize(t);
	}
}
void
MEMStrict::solveZ(double torr) {
	unsigned int size = m_accumDYFT.size();
	std::vector<double> dy2(size);
	std::vector<double> &g2(m_accumG2);

	for(unsigned int i = 0; i < size; i++) {
		dy2[i] = std::norm(m_accumDYFT[i]);
	}
	double nsumz;
	for(unsigned int it = 0; it < lrint(log(size) + 2); it++) {
		double k = 2 * m_accumZ * m_accumZ;
		for(unsigned int i = 0; i < size; i++) {
			g2[i] = lambertW0(k * dy2[i]) * 0.5;
		}
		nsumz = 0.0;
		for(unsigned int i = 0; i < size; i++) {
			nsumz += exp(g2[i]);
		}
		double err = fabs(nsumz - m_accumZ) / nsumz;
		m_accumZ = nsumz;
		if(err < torr) {
//			fprintf(stderr, "MEM: Z solved w/ it=%u,err=%g\n", it, err);
			break;
		}
	}
}

bool
MEMStrict::genSpectrum(const std::vector<std::complex<double> >& memin0, std::vector<std::complex<double> >& memout,
	int t0, double torr, FFT::twindowfunc /*windowfunc*/, double windowlength) {
	std::vector<std::complex<double> > memin(std::min((int)lrint(windowlength * memin0.size()), (int)memin0.size()));
	unsigned int tshift = (memin0.size() - memin.size()) / 2;
	for(unsigned int i = 0; i < memin.size(); i++)
		memin[i] = memin0[i + tshift];
	t0 += (int)tshift;
	unsigned int t = memin.size();
	unsigned int n = memout.size();
	if(t0 < 0)
		t0 += (-t0 / n + 1) * n;
	setup(t, n);
	double sqrtpow = 0.0;
	for(unsigned int i = 0; i < memin.size(); i++)
		sqrtpow += std::norm(memin[i]);
	sqrtpow = sqrt(sqrtpow);
	double err;
	double alpha = 0.3;
	for(double sigma = sqrtpow / 4.0; sigma < sqrtpow; sigma *= 1.2) {
		//	fprintf(stderr, "MEM: Using T=%u,N=%u,sigma=%g\n", t,n,sigma);
		std::fill(m_accumDYFT.begin(), m_accumDYFT.end(), 0.0);
		std::fill(m_ifft.begin(), m_ifft.end(), 0.0);
		std::fill(m_lambda.begin(), m_lambda.end(), 0.0);
		std::fill(m_accumDY.begin(), m_accumDY.end(), 0.0);
		std::fill(m_accumG2.begin(), m_accumG2.end(), 0.0);
		m_accumZ = t;
		double oerr = sqrtpow;
		unsigned int it;
		for(it = 0; it < 50; it++) {
			err = stepMEM(memin, memout, alpha, sigma, t0, torr);
			if(err < torr * sqrtpow) {
				break;
			}
			if(err > sqrtpow * 1.1) {
				break;
			}
			if(err > oerr * 1.0) {
				break;
			}
			oerr = err;
		}
		if(err < torr * sqrtpow) {
			dbgPrint(formatString("MEM: Converged w/ sigma=%g, alpha=%g, err=%g, it=%u\n", sigma, alpha, err, it));
			double osqrtpow = 0.0;
			for(unsigned int i = 0; i < memout.size(); i++)
				osqrtpow += std::norm(memout[i]);
			osqrtpow = sqrt(osqrtpow / n);
			dbgPrint(formatString("MEM: Pout/Pin=%g\n", osqrtpow/sqrtpow));
			return true;
		}
		else {
			dbgPrint(formatString("MEM: Failed w/ sigma=%g, alpha=%g, err=%g, it=%u\n", sigma, alpha, err, it));
		}
	}
	dbgPrint(formatString("MEM: Use ZF-FFT instead.\n"));
	std::fill(m_ifft.begin(), m_ifft.end(), 0.0);
	for(unsigned int i = 0; i < t; i++) {
		m_ifft[(t0 + i) % n] = memin[i];
	}
	m_fftN->exec(m_ifft, memout);			
	return false;
}
double
MEMStrict::stepMEM(const std::vector<std::complex<double> >& memin, std::vector<std::complex<double> >& memout, 
	double alpha, double sigma, int t0, double torr) {
	unsigned int n = m_ifft.size();
	unsigned int t = memin.size();
	double isigma = 1.0 / sigma;
	std::vector<std::complex<double> > &lambdaZF(m_ifft);
	std::fill(lambdaZF.begin(), lambdaZF.end(), 0.0);
	std::complex<double> *plambda = &m_lambda[0];
	for(unsigned int i = 0; i < t; i++) {
		lambdaZF[(t0 + i) % n] = *plambda * isigma;
		plambda++;
	}
	m_ifftN->exec(lambdaZF, memout);
	std::vector<double> pfz(n);
	double sumz = 0.0;
	double *ppfz = &pfz[0];
	std::complex<double> *pmemout = &memout[0];
	for(unsigned int i = 0; i < n; i++) {
		*ppfz = exp(std::norm(*pmemout));
		sumz += *ppfz++;
		pmemout++;
	}
	double k = 2.0 * sigma / sumz * n;
	ppfz = &pfz[0];
	pmemout = &memout[0];
	for(unsigned int i = 0; i < n; i++) {
		double p = k * *ppfz++;
		*pmemout = std::conj(*pmemout) * p;
		pmemout++;
	}
	genIFFT(memout);

	k = alpha / t / sigma / 2;
	double err = 0.0;
	const std::complex<double> *pmemin = &memin[0];
	std::complex<double> *pout = &m_accumDY[0];
	for(unsigned int i = 0; i < t; i++) {
		std::complex<double> *pifft = &m_ifft[(t0 + i) % n];
		std::complex<double> dy = *pmemin - *pifft;
		pmemin++;
		err += std::norm(dy);
		*pout += dy * k;
		pout++;
	}
	err = sqrt(err);
	
	m_fftT->exec(m_accumDY, m_accumDYFT);
	solveZ(torr);
	k = sigma / t;
	pout = &m_accumDYFT[0];
	for(unsigned int i = 0; i < t; i++) {
		double p = k * sqrt(m_accumG2[i] / (std::norm(*pout)));
		*pout = std::conj(*pout) * p;
		pout++;
	}
	m_fftT->exec(m_accumDYFT, m_lambda);
	return err;
}
