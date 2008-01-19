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
#include "matrix.h"
#include <boost/numeric/ublas/io.hpp>
#include <boost/numeric/ublas/matrix_proxy.hpp>
#include <cblas.h>
#include <clapack.h>

typedef __CLPK_integer LPKint;
typedef __CLPK_doublereal LPKdoublereal;
typedef __CLPK_doublecomplex LPKdoublecomplex;

template <class T, class A>
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
	
template <class T, class LPKT>
void cmat2lpk(matrix<T> &a, vector<LPKT>& lpk) {
	lpk.resize(a.size1() * a.size2());
	LPKT *plpk = &lpk[0];
	for(int i = 0; i < a.size2(); i++) {
		matrix_column<matrix<T> > acol(a, i);
		for(int j = 0; j < acol.size(); j++)
			subst(*plpk++, acol(j));
	}
}
template <class T, class LPKT>
void lpk2cmat(const vector<LPKT>& lpk, matrix<T> &a) {
	ASSERT(a.size1() * a.size2() == lpk.size());
	const LPKT *plpk = &lpk[0];
	for(int i = 0; i < a.size2(); i++) {
		matrix_column<matrix<std::complex<double> > > acol(a, i);
		for(int j = 0; j < acol.size(); j++)
			subst(acol(j), *plpk++);
	}
}
template <class T, class LPKT>
void lpk2cvec(const vector<LPKT>& lpk, vector<T> &a) {
	a.resize(lpk.size());
	const LPKT *plpk = &lpk[0];
	for(int i = 0; i < a.size(); i++) {
		subst(a[i], *plpk++);
	}
}

void eigHermiteRRR(const matrix<std::complex<double> > &a_corg,
	vector<double> &lambda, matrix<std::complex<double> > &v,
	double tol) {
	matrix<std::complex<double> > a_org(a_corg);
	LPKint n = a_org.size2();
	LPKint lda = a_org.size1();
	ASSERT(lda >= n);
	LPKint ldz = n;
	vector<LPKdoublecomplex> a(n*lda), z(n*ldz);
	vector<LPKdoublereal> w(n);
	vector<LPKint> isuppz(2*n);
	
	cmat2lpk(a_org, a);
	
	LPKint lwork = -1, liwork = -1, lrwork = -1;
	vector<LPKdoublecomplex> work(1);
	vector<LPKdoublereal> rwork(1);
	vector<LPKint> iwork(1);
	
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

void householderQR(const symmetric_matrix<double> &a_org,
	matrix<double> &q, triangular_matrix<double, upper> &r) {
	symmetric_matrix<double> a(a_org);
	int n = a.size1();
	ASSERT(n == (int)a.size2());
	r.resize(n, n);
	q = identity_matrix<double>(n);
	for(int i = 0; i < n - 1; i++) {
		matrix_vector_range<symmetric_matrix<double> > aran(a, range(i, n - 1), range(i, i) );
		double normaran = norm_2(aran);
		double alpha = -((aran(0) > 0) ? 1 : -1) * normaran;
		aran(0) -= alpha;
		matrix<double> v(n - i, 1);
		matrix_column<matrix<double> > vcol(v, 0); 
		vcol = aran;
		v /= normaran;
		matrix<double> V = identity_matrix<double>(n - i);
		V -= 2 * prod(v, trans(v));
		matrix_range<symmetric_matrix<double> > aran2(a, range(i, n - 1), range(i, n - 1) );
		aran2 = prod(V, aran2);
		matrix_range<matrix<double> > qran(q, range(i, n - 1), range(0, n - 1));
		qran = prod(V, qran);
	}
	q = trans(q);
	r = a;
}

void modifiedGramSchmidt(const matrix<std::complex<double> > &a_org,
	matrix<std::complex<double> > &q, triangular_matrix<std::complex<double>, upper> &r) {
	matrix<std::complex<double> > a(a_org);
	int n = a.size1();
	ASSERT(n == (int)a.size2());
	r.resize(n, n);
	q.resize(n, n);
	for(int i = 0; i < n; i++) {
		matrix_column<matrix<std::complex<double> > > acol(a, i);
		matrix_column<matrix<std::complex<double> > > qcol(q, i);
		for(int j = 0; j < i; j++) {
			r(i, j) = 0;
		}
		r(i, i) = norm_2(acol);
		qcol = acol / r(i, i);
		for(int j = i + 1; j < n; j++) {
			matrix_column<matrix<std::complex<double> > > acol(a, j);
			r(i, j) = inner_prod(qcol, acol);
			acol -= r(i, j) * qcol;
		}
	}
}
