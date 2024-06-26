/***************************************************************************
		Copyright (C) 2002-2015 Kentaro Kitagawa
		                   kitag@issp.u-tokyo.ac.jp

		This program is free software; you can redistribute it and/or
		modify it under the terms of the GNU Library General Public
		License as published by the Free Software Foundation; either
		version 2 of the License, or (at your option) any later version.

		You should have received a copy of the GNU Library General
		Public License and a list of authors along with this program;
		see the files COPYING and AUTHORS.
***************************************************************************/
#include "spectrumsolver.h"
#include <algorithm>
#include <gsl/gsl_sf.h>
#define lambertW0 gsl_sf_lambert_W0

SpectrumSolver::SpectrumSolver() {} 
SpectrumSolver::~SpectrumSolver() {}

double
SpectrumSolver::icAIC(double loglikelifood, int k, int /*n*/) {
	return -2 * loglikelifood + 2 * k;
}
double
SpectrumSolver::icAICc(double loglikelifood, int k, int n) {
	return -2 * loglikelifood + 2 * (k + 1) * n / (double)(n - k - 2);
}
double
SpectrumSolver::icHQ(double loglikelifood, int k, int n) {
	return -2 * loglikelifood + 2 * k * log(log(n) / log(2.0));
}
double
SpectrumSolver::icMDL(double loglikelifood, int k, int n) {
	return -loglikelifood + k * log(n) / 2.0;
}

void
SpectrumSolver::genIFFT(const std::vector<std::complex<double> >& wavein) {
	m_ifftN->exec(wavein, m_ifft);
	int n = wavein.size();
	double k = 1.0 / n;
	std::complex<double> *pifft = &m_ifft[0];
	for(unsigned int i = 0; i < n; i++) {
		*pifft *= k;
		pifft++;
	}	
}

void
SpectrumSolver::window(int t, int t0, FFT::twindowfunc windowfunc,
	double windowlength, std::vector<double> &window) {
	double wk = 1.0 / windowLength(t, t0, windowlength);
	window.resize(t);
	for(int i = 0; i < t; i++) {
		window[i] = windowfunc((i + t0) * wk);
	}
}
void
SpectrumSolver::exec(const std::vector<std::complex<double> >& memin, std::vector<std::complex<double> >& memout,
	int t0, double torr, FFT::twindowfunc windowfunc, double windowlength) {
	windowlength = std::max(0.0, windowlength);
	unsigned int t = memin.size();
	unsigned int n = memout.size();
	if (m_ifft.size() != n) {
		m_fftN.reset(new FFT(-1, n));		
		m_ifftN.reset(new FFT(1, n));		
		m_ifft.resize(n);
	}
	m_peaks.clear();

	if(hasWeighting()) {
		genSpectrum(memin, memout, t0, torr, windowfunc, windowlength);		
	}
	else {
	//Rectangular window.	
		double wlen = windowLength(t, t0, windowlength);
		int idx1 = std::max(0, std::min((int)memin.size() - 1, -t0 + (int)lrint(wlen / 2.0)));
		int idx0 = std::min(std::max(0, -t0 - (int)lrint(wlen / 2.0)), idx1);
		std::vector<std::complex<double> > memin1(idx1 + 1 - idx0);
		for(unsigned int i = 0; i < memin1.size(); i++)
			memin1[i] = memin[i + idx0];
		t0 += idx0;
		genSpectrum(memin1, memout, t0, torr, windowfunc, 1.0);		
	}
	
	std::sort(m_peaks.begin(), m_peaks.end(), std::greater<std::pair<double, double> >());
//	std::reverse(m_peaks.begin(), m_peaks.end());
}

void
SpectrumSolver::autoCorrelation(const std::vector<std::complex<double> >&wave,
	std::vector<std::complex<double> >&corr) {
	if(!m_fftRX || (m_fftRX->length() != wave.size())) {
		int len = FFT::fitLength(wave.size() * 2);
		m_fftRX.reset(new FFT(-1, len));
		m_ifftRX.reset(new FFT(1, len));
	}
	std::vector<std::complex<double> > wavezf(m_fftRX->length(), 0.0), corrzf(m_fftRX->length());
	std::copy(wave.begin(), wave.end(), wavezf.begin());
	m_fftRX->exec(wavezf, corrzf);
	for(int i = 0; i < corrzf.size(); i++)
		wavezf[i] = std::norm(corrzf[i]);
	m_ifftRX->exec(wavezf, corrzf);
	corr.resize(wave.size());
	double normalize = 1.0 / (corr.size() * m_fftRX->length());
	for(int i = 0; i < corr.size(); i++) {
		corr[i] = corrzf[i] * normalize;
	}
}
double
SpectrumSolver::lspe(const std::vector<std::complex<double> >& wavein, int origin,
	const std::vector<double>& psd, std::vector<std::complex<double> >& waveout,
	double tol, bool powfit, FFT::twindowfunc windowfunc) {
	int t = wavein.size();
	int n = waveout.size();
	int t0a = origin;
	if(t0a < 0)
		t0a += (-t0a / n + 1) * n;
	
	std::vector<std::complex<double> > wavezf(n, 0.0);
	for(int i = 0; i < t; i++) {
		int j = (t0a + i) % n;
		wavezf[j] = wavein[i];
	}
	m_fftN->exec(wavezf, waveout);
	
	// Initial values.
	for(int i = 0; i < n; i++) {
		std::complex<double> z = waveout[i];
		waveout[i] = z * sqrt(psd[i] / std::norm(z));
	}
	genIFFT(waveout);
	
	std::vector<double> weight;
	window(t, origin, windowfunc, 1.0, weight);
	
	double sigma20 = 0.0, sigma2 = 0.0, coeff = 1.0;
	for(int it = 0; it < 20; it++) {
		double ns2 = stepLSPE(wavein, origin, psd, waveout, powfit, coeff, weight);
		if(it == 0)
			sigma20 = ns2;
		dbgPrint(formatString("LSPE: err=%g, coeff=%g, it=%u\n", ns2, coeff, it));
		genIFFT(waveout);
		if((it > 3) && (sigma2 - ns2  < sigma20 * tol)) {
			break;
		}
		sigma2 = ns2;
	}
	return coeff;
}
double
SpectrumSolver::stepLSPE(const std::vector<std::complex<double> >& wavein, int origin,
	const std::vector<double>& psd, std::vector<std::complex<double> >& waveout,
	bool powfit, double &coeff, const std::vector<double> &weight) {
	int t = wavein.size();
	int n = waveout.size();
	int t0a = origin;
	if(t0a < 0)
		t0a += (-t0a / n + 1) * n;
	double dcoeff = 1.0;
	if(powfit) {
		double den = 0.0;
		double nom = 0.0;
		for(int i = 0; i < t; i++) {
			double w = weight[i];
			int j = (t0a + i) % n;
			std::complex<double> z = m_ifft[j];
			den += std::norm(z) * w;
			nom += std::real(std::conj(wavein[i]) * z) * w;
		}
		dcoeff = nom / den;
		coeff *= dcoeff;
	}
	std::vector<std::complex<double> > zfin(n, 0.0), zfout(n);
	double sigma2 = 0.0;
	double sumw = 0.0;
	for(int i = 0; i < t; i++) {
		double w = weight[i];
		sumw += w;
		int j = (t0a + i) % n;
		std::complex<double> z = dcoeff * m_ifft[j] - wavein[i];
		zfin[j] = z * w;
		sigma2 += std::norm(z) * w;
	}
	m_fftN->exec(zfin, zfout);
	double sor = 1.0 / sumw;
	double wdiag = dcoeff * sor * sumw;
	for(int i = 0; i < n; i++) {
		std::complex<double> z = - (zfout[i] - waveout[i] * wdiag);
		waveout[i] = coeff * z * sqrt(psd[i] / std::norm(z));
	}
	return sigma2;
}

double
SpectrumSolver::windowLength(int t, int t0, double windowlength) {
	return 2.0 * (std::max(-t0, (int)t + t0) * windowlength);
}

double
SpectrumSolver::numberOfNoises(const std::vector<std::complex<double> >& memin) {
	int t = memin.size();
	int n = fftlen();
	
	std::vector<std::complex<double> > zfin(n), zfft(n);
	std::fill(zfin.begin(), zfin.end(), std::complex<double>(0.0));
	for(int i = 0; i < t; i++) {
		zfin[i % n] = memin[i];
	}
	m_fftN->exec(zfin, zfft);

	double ssum = 0.0, s2sum = 0.0;
	for(int i = 0; i < n; i++) {
		double s = std::abs(zfft[i]);
		ssum += s;
		s2sum += s*s;
	}
	double num = ssum*ssum/s2sum/n*t;
//	fprintf(stderr, "# of noises = %g, ratio=%f\n", num, num/t);
	return num;
}

void
FFTSolver::genSpectrum(const std::vector<std::complex<double> >& fftin, std::vector<std::complex<double> >& fftout,
	int t0, double /*torr*/, FFT::twindowfunc windowfunc, double windowlength) {
	unsigned int t = fftin.size();
	unsigned int n = fftout.size();
	int t0a = t0;
	if(t0a < 0)
		t0a += (-t0a / n + 1) * n;

	std::vector<double> weight;
	window(t, t0, windowfunc, windowlength, weight);
	std::fill(m_ifft.begin(), m_ifft.end(), std::complex<double>(0.0));
	std::vector<std::complex<double> > zffftin(n, 0.0), fftout2(n);
	for(int i = 0; i < t; i++) {
		int j = (t0a + i) % n;
		m_ifft[j] = fftin[i] * weight[i];
		zffftin[j] = m_ifft[j] * (double)(t0 + i) * std::complex<double>(0, -1);
	}
	m_fftN->exec(m_ifft, fftout);
	m_fftN->exec(zffftin, fftout2);

	//Peak detection. Sub-resolution detection for smooth curves.
	std::vector<double> dy(n); //Derivative.
	for(int i = 0; i < n; i++) {
		dy[i] = std::real(fftout2[i] * std::conj(fftout[i]));
	}
	for(int ip = 0; ip < n; ip++) {
		int in = (ip + 1) % n;
		if((dy[ip] > 0) && (dy[in] < 0)) {
			double dx = - dy[ip] / (dy[in] - dy[ip]);
//			dx = std::max(0.0, std::min(dx, 1.0));
			if((dx >= 0) && (dx <= 1.0)) {
/*
			std::complex<double> z = 0.0, xn = 1.0,
				x = std::polar(1.0, -2 * M_PI * (dx + i - 1) / (double)n);
			for(int j = 0; j < t; j++) {
				z += fftin[j] * w * xn;
				xn *= x;
			}
			double r = std::abs(z);
*/
				double r = std::abs(fftout[ip] * (1 - dx) + fftout[in] * dx);
				m_peaks.push_back(std::pair<double, double>(r, dx + ip));
			}
		}
	}
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
MEMStrict::solveZ(double tol) {
	unsigned int size = m_accumDYFT.size();
	std::vector<double> dy2(size);
	std::vector<double> &g2(m_accumG2);

	for(unsigned int i = 0; i < size; i++) {
		dy2[i] = std::norm(m_accumDYFT[i]);
	}
	for(unsigned int it = 0; it < lrint(log(size) + 2); it++) {
		double k = 2 * m_accumZ * m_accumZ;
		for(unsigned int i = 0; i < size; i++) {
			g2[i] = lambertW0(k * dy2[i]) * 0.5;
		}
		double nsumz = 0.0;
		for(unsigned int i = 0; i < size; i++) {
			nsumz += exp(g2[i]);
		}
		double err = fabs(nsumz - m_accumZ) / nsumz;
		m_accumZ = nsumz;
		if(err < tol) {
//			fprintf(stderr, "MEM: Z solved w/ it=%u,err=%g\n", it, err);
			break;
		}
	}
}

void
MEMStrict::genSpectrum(const std::vector<std::complex<double> >& memin, std::vector<std::complex<double> >& memout,
	int t0, double tol, FFT::twindowfunc /*windowfunc*/, double /*windowlength*/) {
	int t = memin.size();
	int n = memout.size();
	if(t0 < 0)
		t0 += (-t0 / n + 1) * n;
	setup(t, n);
	double sqrtpow = 0.0;
	for(unsigned int i = 0; i < memin.size(); i++)
		sqrtpow += std::norm(memin[i]);
	sqrtpow = sqrt(sqrtpow);
	double err = sqrtpow;
	double alpha = 0.3;
	for(double sigma = sqrtpow / 4.0; sigma < sqrtpow; sigma *= 1.2) {
		//	fprintf(stderr, "MEM: Using T=%u,N=%u,sigma=%g\n", t,n,sigma);
		std::fill(m_accumDYFT.begin(), m_accumDYFT.end(), std::complex<double>(0.0));
		std::fill(m_ifft.begin(), m_ifft.end(), std::complex<double>(0.0));
		std::fill(m_lambda.begin(), m_lambda.end(), std::complex<double>(0.0));
		std::fill(m_accumDY.begin(), m_accumDY.end(), std::complex<double>(0.0));
		std::fill(m_accumG2.begin(), m_accumG2.end(), 0.0);
		m_accumZ = t;
		double oerr = sqrtpow;
		unsigned int it;
		for(it = 0; it < 50; it++) {
			err = stepMEM(memin, memout, alpha, sigma, t0, tol);
			if(err < tol * sqrtpow) {
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
		if(err < tol * sqrtpow) {
			dbgPrint(formatString("MEM: Converged w/ sigma=%g, alpha=%g, err=%g, it=%u\n", sigma, alpha, err, it));
			double osqrtpow = 0.0;
			for(unsigned int i = 0; i < memout.size(); i++)
				osqrtpow += std::norm(memout[i]);
			osqrtpow = sqrt(osqrtpow / n);
			dbgPrint(formatString("MEM: Pout/Pin=%g\n", osqrtpow/sqrtpow));
			break;
		}
		else {
			dbgPrint(formatString("MEM: Failed w/ sigma=%g, alpha=%g, err=%g, it=%u\n", sigma, alpha, err, it));
		}
	}
	if(err >= tol * sqrtpow) {
		dbgPrint(formatString("MEM: Use ZF-FFT instead.\n"));
		std::fill(m_ifft.begin(), m_ifft.end(), std::complex<double>(0.0));
		for(unsigned int i = 0; i < t; i++) {
			m_ifft[(t0 + i) % n] = memin[i];
		}
		m_fftN->exec(m_ifft, memout);			
	}

	std::vector<std::complex<double> > zffftin(n), fftout2(n);
	for(int i = 0; i < n; i++) {
		zffftin[i] = m_ifft[i] * (double)((i >= n/2) ? (i - n) : i) * std::complex<double>(0, -1);
	}
	
	//Peak detection. Sub-resolution detection for smooth curves.
	m_fftN->exec(zffftin, fftout2);
	std::vector<double> dy(n);
	for(int i = 0; i < n; i++) {
		dy[i] = std::real(fftout2[i] * std::conj(memout[i])); //Derivative.
	}
	for(int ip = 0; ip < n; ip++) {
		int in = (ip + 1) % n;
		if((dy[ip] > 0) && (dy[in] < 0)) {
			double dx = - dy[ip] / (dy[in] - dy[ip]);
			if((dx >= 0) && (dx <= 1.0)) {
				double r = std::abs(memout[ip] * (1 - dx) + memout[in] * dx);
				m_peaks.push_back(std::pair<double, double>(r, dx + ip));
			}
		}
	}	
}
double
MEMStrict::stepMEM(const std::vector<std::complex<double> >& memin, std::vector<std::complex<double> >& memout, 
	double alpha, double sigma, int t0, double tol) {
	unsigned int n = m_ifft.size();
	unsigned int t = memin.size();
	double isigma = 1.0 / sigma;
	std::vector<std::complex<double> > &lambdaZF(m_ifft);
	std::fill(lambdaZF.begin(), lambdaZF.end(), std::complex<double>(0.0));
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
	solveZ(tol);
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
