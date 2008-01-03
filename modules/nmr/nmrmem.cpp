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

#include <gsl/gsl_sf.h>
#define lambertW0 gsl_sf_lambert_W0
#define bessel_i0 gsl_sf_bessel_I0

double SpectrumSolver::windowFuncRect(double x) {
	return (fabs(x) <= 0.5) ? 1 : 0;
//	return 1.0;
}
double SpectrumSolver::windowFuncTri(double x) {
	return std::max(0.0, 1.0 - 2.0 * fabs(x));
}
double SpectrumSolver::windowFuncHanning(double x) {
	if (fabs(x) >= 0.5)
		return 0.0;
	return 0.5 + 0.5*cos(2*PI*x);
}
double SpectrumSolver::windowFuncHamming(double x) {
	if (fabs(x) >= 0.5)
		return 0.0;
	return 0.54 + 0.46*cos(2*PI*x);
}
double SpectrumSolver::windowFuncBlackman(double x) {
	if (fabs(x) >= 0.5)
		return 0.0;
	return 0.42323+0.49755*cos(2*PI*x)+0.07922*cos(4*PI*x);
}
double SpectrumSolver::windowFuncBlackmanHarris(double x) {
	if (fabs(x) >= 0.5)
		return 0.0;
	return 0.35875+0.48829*cos(2*PI*x)+0.14128*cos(4*PI*x)+0.01168*cos(6*PI*x);
}
double SpectrumSolver::windowFuncFlatTop(double x) {
	return windowFuncHamming(x)*((fabs(x) < 1e-4) ? 1 : sin(4*PI*x)/(4*PI*x));
}
double SpectrumSolver::windowFuncKaiser(double x, double alpha) {
	if (fabs(x) >= 0.5)
		return 0.0;
	x *= 2;
	x = sqrt(std::max(1 - x*x, 0.0));
	return bessel_i0(PI*alpha*x) / bessel_i0(PI*alpha);
}
double SpectrumSolver::windowFuncKaiser1(double x) {
	return windowFuncKaiser(x, 3.0);
}
double SpectrumSolver::windowFuncKaiser2(double x) {
	return windowFuncKaiser(x, 7.2);
}
double SpectrumSolver::windowFuncKaiser3(double x) {
	return windowFuncKaiser(x, 15.0);
}

SpectrumSolver::SpectrumSolver() {} 
SpectrumSolver::~SpectrumSolver() {
	if(m_ifft.size()) {
		fftw_destroy_plan(m_fftplanN);
		fftw_destroy_plan(m_ifftplanN);
	}
}

void
SpectrumSolver::fftw2std(const std::vector<fftw_complex>& wavein, std::vector<std::complex<double> > &waveout) {
	int size = wavein.size();
	waveout.resize(size);
	const fftw_complex *pin = &wavein[0];
	std::complex<double> *pout = &waveout[0];
	for(int i = 0; i < size; i++) {
		*pout = std::complex<double>(pin->re, pin->im);
		pout++;
		pin++;
	}
}
void
SpectrumSolver::std2fftw(const std::vector<std::complex<double> >& wavein, std::vector<fftw_complex> &waveout) {
	int size = wavein.size();
	waveout.resize(size);
	const std::complex<double> *pin = &wavein[0];
	fftw_complex *pout = &waveout[0];
	for(int i = 0; i < size; i++) {
		pout->re = pin->real();
		pout->im = pin->imag();
		pout++;
		pin++;
	}
}
void
SpectrumSolver::clearFTBuf(std::vector<fftw_complex> &buf) {
	for(unsigned int i = 0; i < buf.size(); i++) {
		buf[i].re = 0.0;
		buf[i].im = 0.0;
	}
}
void
SpectrumSolver::genIFFT(std::vector<fftw_complex>& wavein) {
	fftw_one(m_ifftplanN, &wavein[0], &m_ifft[0]);
	int n = wavein.size();
	double k = 1.0 / n;
	fftw_complex *pifft = &m_ifft[0];
	for(unsigned int i = 0; i < n; i++) {
		pifft->re *= k;	
		pifft->im *= k;
		pifft++;
	}	
}

bool
SpectrumSolver::exec(const std::vector<fftw_complex>& memin, std::vector<fftw_complex>& memout,
	int t0, double torr, twindowfunc windowfunc, double windowlength) {
	unsigned int t = memin.size();
	unsigned int n = memout.size();
	if (m_ifft.size() != n) {
		if(m_ifft.size()) {
			fftw_destroy_plan(m_fftplanN);
			fftw_destroy_plan(m_ifftplanN);
		}
		m_fftplanN = fftw_create_plan(n, FFTW_FORWARD, FFTW_ESTIMATE);		
		m_ifftplanN = fftw_create_plan(n, FFTW_BACKWARD, FFTW_ESTIMATE);		
		m_ifft.resize(n);
	}
	genSpectrum(memin, memout, t0, torr, windowfunc, windowlength);
}


bool
FFTSolver::genSpectrum(const std::vector<fftw_complex>& fftin, std::vector<fftw_complex>& fftout,
	int t0, double /*torr*/, twindowfunc windowfunc, double windowlength) {
	unsigned int t = fftin.size();
	unsigned int n = fftout.size();
	int t0a = t0;
	if(t0a < 0)
		t0a += (-t0a / n + 1) * n;

	double wk = 0.5 / (std::max(-t0, (int)t + t0) * windowlength);
	clearFTBuf(m_ifft);
	for(int i = 0; i < t; i++) {
		fftw_complex *pout = &m_ifft[(t0a + i) % n];
		double w = windowfunc((i + t0) * wk);
		pout->re = fftin[i].re * w;
		pout->im = fftin[i].im * w;
	}
	fftw_one(m_fftplanN, &m_ifft[0], &fftout[0]);
	return true;
}

bool
YuleWalkerCousin::genSpectrum(const std::vector<fftw_complex>& memin, std::vector<fftw_complex>& memout,
	int t0, double torr, twindowfunc windowfunc, double windowlength) {
	unsigned int t = memin.size();
	unsigned int n = memout.size();
	unsigned int taps = std::min((int)lrint(windowlength / 2.0 * t), (int)t - 2);
	std::vector<std::complex<double> > bufin(t);
	fftw2std(memin, bufin);

	if(t0 < 0)
		t0 += (-t0 / n + 1) * n;	
	std::vector<fftw_complex> zfbuf(n), fftout(n);
	clearFTBuf(zfbuf);
	for(int i = 0; i < t; i++) {
		fftw_complex *pout = &zfbuf[(t0 + i) % n];
		pout->re = bufin[i].real();
		pout->im = bufin[i].imag();
	}
	fftw_one(m_fftplanN, &zfbuf[0], &fftout[0]);
	std::vector<std::complex<double> > bufzffft(n);
	fftw2std(fftout, bufzffft);
		
	std::vector<std::complex<double> > a(taps + 1);
	std::fill(a.begin(), a.end(), 0.0);
	a[0] = 1.0;
	double sigma2 = 0.0;
	for(unsigned int i = 0; i < t; i++) {
		sigma2 += std::norm(bufin[i]);
	}
//	sigma2 /= t;
	int taps_conv = exec(bufin, a, sigma2, torr, windowfunc);
	fprintf(stderr, "MEM/AR: taps=%d, taps_conv=%d\n", taps, taps_conv);
	taps = taps_conv;

	clearFTBuf(zfbuf);
	fftw_complex *pout = &zfbuf[0];
	for(int i = 0; i < taps + 1; i++) {
		pout->re = a[i].real();
		pout->im = a[i].imag();
		pout++;
	}
	fftw_one(m_fftplanN, &zfbuf[0], &fftout[0]);
	
	fftw_complex *pin = &fftout[0];
	for(unsigned int i = 0; i < n; i++) {
		double z = sigma2 / (pin->re*pin->re + pin->im*pin->im);
		z = sqrt(std::max(z, 0.0) / std::norm(bufzffft[i]));
		memout[i].re = bufzffft[i].real() * z;
		memout[i].im = bufzffft[i].imag() * z;
		pin++;
	}
	genIFFT(memout);
	return true;
}

int
MEMBurg::exec(const std::vector<std::complex<double> >& memin, std::vector<std::complex<double> >& a,
	double &sigma2, double /*torr*/, twindowfunc /*windowfunc*/) {
	unsigned int t = memin.size();
	unsigned int m = a.size() - 1;
	std::vector<std::complex<double> > epsilon(t), eta(t), epsilon_next(t), eta_next(t), a_next(m+1);
	std::copy(memin.begin(), memin.end(), epsilon.begin());
	std::copy(memin.begin(), memin.end(), eta.begin());
	std::copy(memin.begin(), memin.end(), epsilon_next.begin());
	std::copy(memin.begin(), memin.end(), eta_next.begin());
	std::fill(a_next.begin(), a_next.end(), 0.0);

	for(unsigned int p = 0; p < m; p++) {
		std::complex<double> x = 0.0, y = 0.0;
		for(unsigned int i = p + 1; i < t; i++) {
			x += epsilon[i] * std::conj(eta[i-1]);
			y += std::norm(epsilon[i]) + std::norm(eta[i-1]);
		}
		std::complex<double> alpha = x / y;
		alpha *= -2;
		for(unsigned int i = p + 1; i < t; i++) {
			eta_next[i] = eta[i-1] + std::conj(alpha) * epsilon[i];
			epsilon_next[i] = epsilon[i] + alpha * eta[i-1];
		}
		std::copy(eta_next.begin(), eta_next.end(), eta.begin());
		std::copy(epsilon_next.begin(), epsilon_next.end(), epsilon.begin());
		for(unsigned int i = 0; i < p + 2; i++) {
			a_next[i] = a[i] + alpha * std::conj(a[p + 1 - i]);
		}
		std::copy(a_next.begin(), a_next.end(), a.begin());
		sigma2 *= 1 - std::norm(alpha);
	}
	return m;
}

int
YuleWalkerAR::exec(const std::vector<std::complex<double> >& memin, std::vector<std::complex<double> >& a,
	double &sigma2, double /*torr*/, twindowfunc windowfunc) {
	unsigned int t = memin.size();
	unsigned int m = a.size() - 1;
	std::vector<std::complex<double> > rx(t), a_next(m+1);
	std::fill(rx.begin(), rx.end(), 0.0);
	std::fill(a_next.begin(), a_next.end(), 0.0);
	for(unsigned int p = 0; p <= m; p++) {
		for(unsigned int i = 0; i < t - p; i++) {
			rx[p] += std::conj(memin[i]) * memin[i+p];
		}
		rx[p] *= windowfunc(0.5*p/(double)t) * t / (t - p);
	}

	for(unsigned int p = 0; p < m; p++) {
		std::complex<double> delta = 0.0;
		for(unsigned int i = 0; i < p + 1; i++) {
			delta += a[i] * rx[p + 1 - i];
		}
		for(unsigned int i = 0; i < p + 2; i++) {
			a_next[i] = a[i] - delta/sigma2 * std::conj(a[p + 1 - i]);
		}
		double sigma_next = sigma2 - std::norm(delta) / sigma2;
		if(sigma_next < 0) {
			return p;
		}
		std::copy(a_next.begin(), a_next.end(), a.begin());
		sigma2 = sigma_next;
	}
	return m;
}

MEMStrict::~MEMStrict() {
	if(m_lambda.size())
		fftw_destroy_plan(m_fftplanT);
}

void
MEMStrict::setup(unsigned int t, unsigned int n) {
	if (m_lambda.size() != t) {
		if(m_lambda.size())
			fftw_destroy_plan(m_fftplanT);
		m_fftplanT = fftw_create_plan(t, FFTW_FORWARD, FFTW_ESTIMATE);
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
		dy2[i] = m_accumDYFT[i].re*m_accumDYFT[i].re + m_accumDYFT[i].im*m_accumDYFT[i].im;
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
MEMStrict::genSpectrum(const std::vector<fftw_complex>& memin0, std::vector<fftw_complex>& memout,
	int t0, double torr, twindowfunc /*windowfunc*/, double windowlength) {
	std::vector<fftw_complex> memin(std::min((int)lrint(windowlength * memin0.size()), (int)memin0.size()));
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
		sqrtpow += memin[i].re*memin[i].re + memin[i].im*memin[i].im;
	sqrtpow = sqrt(sqrtpow);
	double err;
	double alpha = 0.3;
	for(double sigma = sqrtpow / 4.0; sigma < sqrtpow; sigma *= 1.414) {
		//	fprintf(stderr, "MEM: Using T=%u,N=%u,sigma=%g\n", t,n,sigma);
		clearFTBuf(m_accumDYFT);
		clearFTBuf(m_ifft);
		clearFTBuf(memout);
		clearFTBuf(m_lambda);
		clearFTBuf(m_accumDY);
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
			fprintf(stderr, "MEM: Converged w/ sigma=%g, alpha=%g, err=%g, it=%u\n", sigma, alpha, err, it);
			double osqrtpow = 0.0;
			for(unsigned int i = 0; i < memout.size(); i++)
				osqrtpow += memout[i].re*memout[i].re + memout[i].im*memout[i].im;
			osqrtpow = sqrt(osqrtpow / n);
			fprintf(stderr, "MEM: Pout/Pin=%g\n", osqrtpow/sqrtpow);
			return true;
		}
		else {
			fprintf(stderr, "MEM: Failed w/ sigma=%g, alpha=%g, err=%g, it=%u\n", sigma, alpha, err, it);
		}
	}
	fprintf(stderr, "MEM: Use ZF-FFT instead.\n");
	clearFTBuf(m_ifft);
	for(unsigned int i = 0; i < t; i++) {
		fftw_complex *pout = &m_ifft[(t0 + i) % n];
		pout->re = memin[i].re;
		pout->im = memin[i].im;
	}
	fftw_one(m_fftplanN, &m_ifft[0], &memout[0]);			
	return false;
}
double
MEMStrict::stepMEM(const std::vector<fftw_complex>& memin, std::vector<fftw_complex>& memout, 
	double alpha, double sigma, int t0, double torr) {
	unsigned int n = m_ifft.size();
	unsigned int t = memin.size();
	double isigma = 1.0 / sigma;
	std::vector<fftw_complex> &lambdaZF(m_ifft);
	clearFTBuf(lambdaZF);
	fftw_complex *plambda = &m_lambda[0];
	for(unsigned int i = 0; i < t; i++) {
		fftw_complex *pout = &lambdaZF[(t0 + i) % n];
		pout->re = plambda->re * isigma;
		pout->im = plambda->im * isigma;
		plambda++;
	}
	fftw_one(m_ifftplanN, &lambdaZF[0], &memout[0]);
	std::vector<double> pfz(n);
	double sumz = 0.0;
	double *ppfz = &pfz[0];
	fftw_complex *pmemout = &memout[0];
	for(unsigned int i = 0; i < n; i++) {
		*ppfz = exp(pmemout->re*pmemout->re + pmemout->im*pmemout->im);
		sumz += *ppfz++;
		pmemout++;
	}
	double k = 2.0 * sigma / sumz * n;
	ppfz = &pfz[0];
	pmemout = &memout[0];
	for(unsigned int i = 0; i < n; i++) {
		double p = k * *ppfz++;
		pmemout->re *= p;
		pmemout->im *= -p;
		pmemout++;
	}
	genIFFT(memout);

	k = alpha / t / sigma / 2;
	double err = 0.0;
	const fftw_complex *pmemin = &memin[0];
	fftw_complex *pout = &m_accumDY[0];
	for(unsigned int i = 0; i < t; i++) {
		fftw_complex *pifft = &m_ifft[(t0 + i) % n];
		fftw_complex dy;
		dy.re = pmemin->re - pifft->re;
		dy.im = pmemin->im - pifft->im;
		pmemin++;
		err += dy.re*dy.re + dy.im*dy.im;
		pout->re += dy.re * k;
		pout->im += dy.im * k;
		pout++;
	}
	err = sqrt(err);
	
	fftw_one(m_fftplanT, &m_accumDY[0], &m_accumDYFT[0]);
	solveZ(torr);
	k = sigma / t;
	pout = &m_accumDYFT[0];
	for(unsigned int i = 0; i < t; i++) {
		fftw_complex z1 = *pout;
		double p = k * sqrt(m_accumG2[i] / (z1.re*z1.re + z1.im*z1.im));
		pout->re *= p;
		pout->im *= -p;
		pout++;
	}
	fftw_one(m_fftplanT, &m_accumDYFT[0], &m_lambda[0]);
	return err;
}
