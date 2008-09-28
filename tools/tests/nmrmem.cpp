/***************************************************************************
 Copyright (C) 2002-2008 Kentaro Kitagawa
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

NMRMEM::~NMRMEM() {
	if(m_ifft.size()) {
		fftw_destroy_plan(m_fftplanN);
		fftw_destroy_plan(m_ifftplanN);
	}
	if(m_lambda.size())
		fftw_destroy_plan(m_fftplanT);
}
void
NMRMEM::clearFTBuf(std::vector<fftw_complex> &buf) {
	for(unsigned int i = 0; i < buf.size(); i++) {
		buf[i].re = 0.0;
		buf[i].im = 0.0;
	}
}
void
NMRMEM::setup(unsigned int t, unsigned int n) {
	if (m_ifft.size() != n) {
		if(m_ifft.size()) {
			fftw_destroy_plan(m_fftplanN);
			fftw_destroy_plan(m_ifftplanN);
		}
		m_fftplanN = fftw_create_plan(n, FFTW_FORWARD, FFTW_ESTIMATE);		
		m_ifftplanN = fftw_create_plan(n, FFTW_BACKWARD, FFTW_ESTIMATE);		
		m_ifft.resize(n);
	}
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
NMRMEM::solveZ(double torr) {
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
NMRMEM::exec(const std::vector<fftw_complex>& memin, std::vector<fftw_complex>& memout,
	int t0, double torr) {
	unsigned int t = memin.size();
	unsigned int n = memout.size();
	if(t0 < 0)
		t0 += (-t0 / n + 1) * n;
	setup(t, n);
	double sqrtpow = 0.0;
	for(unsigned int i = 0; i < memin.size(); i++)
		sqrtpow += memin[i].re*memin[i].re + memin[i].im*memin[i].im;
	sqrtpow = sqrt(sqrtpow);
	double sigma = sqrtpow * 0.5;
	fprintf(stderr, "MEM: Using T=%u,N=%u,sigma=%g\n", t,n,sigma);
	double err;
	for(double alpha = 0.3; alpha > 0.1; alpha -= 0.1) {
		clearFTBuf(m_accumDYFT);
		clearFTBuf(m_ifft);
		clearFTBuf(memout);
		clearFTBuf(m_lambda);
		clearFTBuf(m_accumDY);
		std::fill(m_accumG2.begin(), m_accumG2.end(), 0.0);
		m_accumZ = t;
		double oerr = sqrtpow;
		for(unsigned it = 0; it < 50; it++) {
			err = stepMEM(memin, memout, alpha, sigma, t0, torr);
			if(err < torr * sqrtpow) {
				break;
			}
			if(err > sqrtpow * 1.1) {
				break;
			}
			if(err > oerr * 1.5) {
				break;
			}
			oerr = err;
		}
		if(err < torr * sqrtpow) {
			fprintf(stderr, "MEM: Converged w/ alpha=%g, err=%g\n", alpha, err);
			double osqrtpow = 0.0;
			for(unsigned int i = 0; i < memout.size(); i++)
				osqrtpow += memout[i].re*memout[i].re + memout[i].im*memout[i].im;
			osqrtpow = sqrt(osqrtpow / n);
			fprintf(stderr, "MEM: Pout/Pin=%g\n", osqrtpow/sqrtpow);
			return true;
		}
		else {
			fprintf(stderr, "MEM: Failed w/ alpha=%g, err=%g\n", alpha, err);
		}
	}
	fprintf(stderr, "MEM: Use ZF-FFT instead.\n");
	clearFTBuf(m_ifft);
	for(unsigned int i = 0; i < t; i++) {
		fftw_complex *pout = &m_ifft[(t0 + i) % n];
		pout->re = memin[i].re;
		pout->im = memin[i].im;
	}
	clearFTBuf(memout);
	fftw_one(m_fftplanN, &m_ifft[0], &memout[0]);			
	return false;
}
double
NMRMEM::stepMEM(const std::vector<fftw_complex>& memin, std::vector<fftw_complex>& memout, 
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
	fftw_one(m_ifftplanN, &memout[0], &m_ifft[0]);
	k = 1.0 / n;
	fftw_complex *pifft = &m_ifft[0];
	for(unsigned int i = 0; i < n; i++) {
		pifft->re *= k;	
		pifft->im *= k;
		pifft++;
	}
	
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
