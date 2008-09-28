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
#include "roots.h"

double rootsDKA(const std::vector<std::complex<double> > &polynomial, std::vector<std::complex<double> > &roots,
	double eps, int max_it) {
	int n = polynomial.size() - 1;
	ASSERT(abs(polynomial[n] - 1.0) < 0.01);
	ASSERT(polynomial.size() == roots.size() + 1);
	std::vector<bool> conv(n, false);
	std::vector<std::complex<double> > roots_next(n);
	double err;	
	int it;
	for(it = 0; it < max_it; it++) {
		err = 0.0;	
		std::copy(roots.begin(), roots.end(), roots_next.begin());
		for(int i = 0; i < n; i++) {
			if(conv[i])
				continue;
			std::complex<double> f = 0.0;
			std::complex<double> x = roots[i];
			std::complex<double> xn = 1.0;
			for(int j = 0; j <= n; j++) {
				f += polynomial[j] * xn;
				xn *= x;
			}
			std::complex<double> deno = 1.0;
			for(int j = 0; j < n; j++) {
				if(i == j) continue;
				deno *= (x - roots[j]);
			}
			if(std::abs(deno) < eps) {
				conv[i] = true;
				continue;
			}
			std::complex<double> dx = - f / deno;
			err = std::max(std::abs(dx), err);
			roots_next[i] = roots[i] + dx;
		}
		std::copy(roots_next.begin(), roots_next.end(), roots.begin());
		if(err < eps)
			break;
	}
	dbgPrint(formatString("Roots-DKA: it=%d, err=%g", it, err));
	return err;
}
