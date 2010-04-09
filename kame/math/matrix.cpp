/***************************************************************************
		Copyright (C) 2002-2010 Kentaro Kitagawa
		                   kitag@issp.u-tokyo.ac.jp
		
		This program is free software; you can redistribute it and/or
		modify it under the terms of the GNU Library General Public
		License as published by the Free Software Foundation; either
		version 2 of the License, or (at your option) any later version.
		
		You should have received a copy of the GNU Library General 
		Public License and a list of authors along with this program; 
		see the files COPYING and AUTHORS.
***************************************************************************/
#include "matrix.h"
#include <boost/numeric/ublas/io.hpp>
#include <boost/numeric/ublas/matrix_proxy.hpp>

typedef int LPKint;
typedef double LPKdoublereal;
typedef struct {double r, i;} LPKdoublecomplex;
extern "C" int zheevr_(char *jobz, char *range, char *uplo, LPKint *n, 
	LPKdoublecomplex *a, LPKint *lda, LPKdoublereal *vl, LPKdoublereal *vu, 
	LPKint *il, LPKint *iu, LPKdoublereal *abstol, LPKint *m, LPKdoublereal *
	w, LPKdoublecomplex *z, LPKint *ldz, LPKint *isuppz, LPKdoublecomplex *
	work, LPKint *lwork, LPKdoublereal *rwork, LPKint *lrwork, LPKint *
	iwork, LPKint *liwork, LPKint *info);

template <typename T, typename A>
inline void
subst(T &y, const A &x) {
	y = x;
}
inline void
subst(LPKdoublecomplex &y, const std::complex<double> &x) {
	y.r = std::real(x);
	y.i = std::imag(x);
}
inline void
subst(std::complex<double> &y, const LPKdoublecomplex &x) {
	y = std::complex<double>(x.r, x.i);
}
	
template <typename T, typename LPKT>
void cmat2lpk(const ublas::matrix<T> &a, ublas::vector<LPKT>& lpk) {
	lpk.resize(a.size1() * a.size2());
	LPKT *plpk = &lpk[0];
	for(int i = 0; i < a.size2(); i++) {
		ublas::matrix_column<const ublas::matrix<T> > acol(a, i);
		for(int j = 0; j < acol.size(); j++)
			subst(*plpk++, acol(j));
	}
}
template <typename T, typename LPKT>
void lpk2cmat(const ublas::vector<LPKT>& lpk, ublas::matrix<T> &a) {
	ASSERT(a.size1() * a.size2() == lpk.size());
	const LPKT *plpk = &lpk[0];
	for(int i = 0; i < a.size2(); i++) {
		ublas::matrix_column<ublas::matrix<std::complex<double> > > acol(a, i);
		for(int j = 0; j < acol.size(); j++)
			subst(acol(j), *plpk++);
	}
}
template <typename T, typename LPKT>
void lpk2cvec(const ublas::vector<LPKT>& lpk, ublas::vector<T> &a) {
	a.resize(lpk.size());
	const LPKT *plpk = &lpk[0];
	for(int i = 0; i < a.size(); i++) {
		subst(a[i], *plpk++);
	}
}

void eigHermiteRRR(const ublas::matrix<std::complex<double> > &a_org,
	ublas::vector<double> &lambda, ublas::matrix<std::complex<double> > &v,
	double tol) {
	LPKint n = a_org.size2();
	LPKint lda = a_org.size1();
	ASSERT(lda >= n);
	LPKint ldz = n;
	ublas::vector<LPKdoublecomplex> a(n*lda), z(n*ldz);
	ublas::vector<LPKdoublereal> w(n);
	ublas::vector<LPKint> isuppz(2*n);
	
	cmat2lpk(a_org, a);
	
	LPKint lwork = -1, liwork = -1, lrwork = -1;
	ublas::vector<LPKdoublecomplex> work(1);
	ublas::vector<LPKdoublereal> rwork(1);
	ublas::vector<LPKint> iwork(1);
	
	LPKint info = 0, numret;
	LPKint il, iu;
	LPKdoublereal vl, vu;
	char cv = 'V', ca = 'A', cu = 'U';
	int ret = zheevr_(&cv, &ca, &cu, &n, &a[0], &lda,
		 &vl, &vu, &il, &iu, &tol, &numret, &w[0], &z[0], &ldz, 
		 &isuppz[0], &work[0], &lwork, &rwork[0], &lrwork, &iwork[0], &liwork, &info);
	ASSERT(info == 0);
	lwork = lrint(work[0].r);
	work.resize(lwork);
	lrwork = lrint(rwork[0]);
	rwork.resize(lrwork);
	liwork = iwork[0];
	iwork.resize(liwork);
	ret = zheevr_(&cv, &ca, &cu, &n, &a[0], &lda,
		&vl, &vu, &il, &iu, &tol, &numret, &w[0], &z[0], &ldz, 
		 &isuppz[0], &work[0], &lwork, &rwork[0], &lrwork, &iwork[0], &liwork, &info);
	ASSERT(info == 0);
	
	lpk2cvec(w, lambda);
	v.resize(n, ldz);
	lpk2cmat(z, v);
}

