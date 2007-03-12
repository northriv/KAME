/***************************************************************************
		Copyright (C) 2002-2007 Kentaro Kitagawa
		                   kitagawa@scphys.kyoto-u.ac.jp
		
		This program is free software; you can redistribute it and/or
		modify it under the terms of the GNU Library General Public
		License as published by the Free Software Foundation; either
		version 2 of the License, or (at your option) any later version.
		
		You should have received a copy of the GNU Library General 
		Public License and a list of authors along with this program; 
		see the files COPYING and AUTHORS.
***************************************************************************/
#include "nmrrelax.h"
#include "nmrrelaxfit.h"

#include <klocale.h>

#include <gsl/gsl_multifit_nlin.h>

typedef int (exp_f) (const gsl_vector * X, void * PARAMS, gsl_vector * F);
typedef int (exp_df) (const gsl_vector * X, void * PARAMS, gsl_matrix * J);
typedef int (exp_fdf) (const gsl_vector * X, void * PARAMS, gsl_vector * F, gsl_matrix * J);

static int
do_nlls(int n, int p, double *param, double *err, double *det, void *user, exp_f  *ef, exp_df *edf, exp_fdf *efdf
		, int itercnt);

#include <gsl/gsl_blas.h>
#include <gsl/gsl_ieee_utils.h>

//---------------------------------------------------------------------------

class XRelaxFuncPoly : public XRelaxFunc
{
    
public:
//! define a term in a relaxation function
//! a*exp(-p*t/T1)
	struct Term
	{
		int p;
	 double a;
	};
 
 const struct Term *m_terms; 
 
 XNODE_OBJECT
protected:
 XRelaxFuncPoly(const char *name, bool runtime, const Term *terms)
 : XRelaxFunc(name, runtime), m_terms(terms)
		{
		}    
public:  
 virtual ~XRelaxFuncPoly() {}
 
 //! called during fitting
 //! \param f f(t, it1) will be passed
 //! \param dfdt df/d(it1) will be passed
 //! \param t a time P1 or 2tau
 //! \param it1 1/T1 or 1/T2   
 virtual void relax(double *f, double *dfdt, double t, double it1)
		{
 double rf = 0, rdf = 0;
	 double x = -t * it1;
		 x = std::min(5.0, x);
			 for(const Term *term = m_terms; term->p != 0; term++)
				 {
 double a = term->a * exp(x*term->p);
	 rf += a;
		 rdf += a * term->p;
			 }
																	  rdf *= -t;
																		  *f = 1.0 - rf;
																			  *dfdt = -rdf;
																				  }
 
};

class XRelaxFuncSqrt : public XRelaxFunc
{
	XNODE_OBJECT
protected:
 XRelaxFuncSqrt(const char *name, bool runtime)
 : XRelaxFunc(name, runtime)
		{
		}    
public:  
 
 virtual ~XRelaxFuncSqrt() {}
 
 //! called during fitting
 //! \param f f(t, it1) will be passed
 //! \param dfdt df/d(it1) will be passed
 //! \param t a time P1 or 2tau
 //! \param it1 1/T1 or 1/T2   
 virtual void relax(double *f, double *dfdt, double t, double it1)
		{
 it1 = std::max(0.0, it1);
	 double rt = sqrt(t * it1);
		 double a = exp(-rt);
			 *f = 1.0 - a;
				 *dfdt = t/rt/2.0 * a;
					 }
};

//NQR I=1
static const struct XRelaxFuncPoly::Term s_relaxdata_nqr2[] = {
	{3, 1.0}, {0, 0}
};
//NQR I=3/2
static const struct XRelaxFuncPoly::Term s_relaxdata_nqr3[] = {
	{3, 1.0}, {0, 0}
};
//NQR I=5/2, 5/2
static const struct XRelaxFuncPoly::Term s_relaxdata_nqr5_5[] = {
    {3, 3.0/7}, {10, 4.0/7}, {0, 0}
};
//NQR I=5/2, 3/2
static const struct XRelaxFuncPoly::Term  s_relaxdata_nqr5_3[] = {
    {3, 3.0/28}, {10, 25.0/28}, {0, 0}
};
//NQR I=3, 3
static const struct XRelaxFuncPoly::Term  s_relaxdata_nqr6_6[] = {
	{21,0.05303}, {10,0.64935}, {3,0.29762}, {0, 0}
};
//NQR I=3, 2
static const struct XRelaxFuncPoly::Term  s_relaxdata_nqr6_4[] = {
	{21,0.47727}, {10,0.41558}, {3,0.10714}, {0, 0}
};
//NQR I=3, 1
static const struct XRelaxFuncPoly::Term  s_relaxdata_nqr6_2[] = {
	{21,0.88384}, {10,0.10823}, {3,0.0079365}, {0, 0}
};
//NQR I=7/2, 7/2
static const struct XRelaxFuncPoly::Term  s_relaxdata_nqr7_7[] = {
    {3, 3.0/14}, {10, 50.0/77}, {21, 3.0/22}, {0, 0}
};
//NQR I=7/2, 5/2
static const struct XRelaxFuncPoly::Term  s_relaxdata_nqr7_5[] = {
    {3, 2.0/21}, {10, 25.0/154}, {21, 49.0/66}, {0, 0}
};
//NQR I=7/2, 3/2
static const struct XRelaxFuncPoly::Term  s_relaxdata_nqr7_3[] = {
    {3, 1.0/42}, {10, 18.0/77}, {21, 49.0/66}, {0, 0}
};
//NQR I=9/2, 9/2
static const struct XRelaxFuncPoly::Term  s_relaxdata_nqr9_9[] = {
    {3, 4.0/33}, {10, 80.0/143}, {21, 49.0/165}, {36, 16.0/715}, {0, 0}
};
//NQR I=9/2, 7/2
static const struct XRelaxFuncPoly::Term  s_relaxdata_nqr9_7[] = {
    {3, 9.0/132}, {10, 5.0/572}, {21, 441.0/660}, {36, 72.9/2860}, {0, 0}
};
//NQR I=9/2, 5/2
static const struct XRelaxFuncPoly::Term  s_relaxdata_nqr9_5[] = {
    {3, 1.0/33}, {10, 20.0/143}, {21, 4.0/165}, {36, 576.0/715}, {0, 0}
};
//NQR I=9/2, 3/2
static const struct XRelaxFuncPoly::Term  s_relaxdata_nqr9_3[] = {
    {3, 1.0/132}, {10, 45.0/572}, {21, 49.0/165}, {36, 441.0/715}, {0, 0}
};
//NMR I=1/2
static const struct XRelaxFuncPoly::Term  s_relaxdata_nmr1[] = {
    {1, 1}, {0, 0}
};
//NMR I=1
static const struct XRelaxFuncPoly::Term  s_relaxdata_nmr2[] = {
	{3,0.75}, {1,0.25}, {0, 0}
};
//NMR I=3/2 center
static const struct XRelaxFuncPoly::Term  s_relaxdata_nmr3ca[] = {
    {1, 0.1}, {6, 0.9}, {0, 0}
};
static const struct XRelaxFuncPoly::Term  s_relaxdata_nmr3cb[] = {
    {1, 0.4}, {6, 0.6}, {0, 0}
};
//NMR I=3/2 satellite
static const struct XRelaxFuncPoly::Term  s_relaxdata_nmr3s[] = {
    {1, 0.1}, {3, 0.5}, {6, 0.4}, {0, 0}
};
//NMR I=5/2 center
static const struct XRelaxFuncPoly::Term  s_relaxdata_nmr5ca[] = {
    {1, 0.02857}, {6, 0.17778}, {15, 0.793667}, {0, 0}
};
static const struct XRelaxFuncPoly::Term  s_relaxdata_nmr5cb[] = {
    {1, 0.25714}, {6, 0.266667}, {15, 0.4762}, {0, 0}
};
//NMR I=5/2 satellite
static const struct XRelaxFuncPoly::Term  s_relaxdata_nmr5s[] = {
    {1, 0.028571}, {3, 0.05357}, {6, 0.0249987}, {10, 0.4464187}, {15, 0.4463875}, {0, 0}
};
//NMR I=5/2 satellite 3/2-5/2
static const struct XRelaxFuncPoly::Term  s_relaxdata_nmr5s2[] = {
    {1, 0.028571}, {3, 0.2143}, {6, 0.3996}, {10, 0.2857}, {15, 0.0714}, {0, 0}
};
//NMR I=3 1--1
static const struct XRelaxFuncPoly::Term  s_relaxdata_nmr6c[] = {  
	{21,0.66288}, {15,0.14881}, {10,0.081169}, {6,0.083333}, {3,0.0059524}, {1,0.017857}, {0,0}
};
//NMR I=3 2-1
static const struct XRelaxFuncPoly::Term  s_relaxdata_nmr6s1[] = {  
	{21,0.23864}, {15,0.48214}, {10,0.20779}, {3,0.053571}, {1,0.017857}, {0,0}
};
//NMR I=3 3-2
static const struct XRelaxFuncPoly::Term  s_relaxdata_nmr6s2[] = {  
	{21,0.026515}, {15,0.14881}, {10,0.32468}, {6,0.33333}, {3,0.14881}, {1,0.017857}, {0,0}
};
//NMR I=7/2 center
static const struct XRelaxFuncPoly::Term  s_relaxdata_nmr7c[] = {
    {1, 0.0119}, {6, 0.06818}, {15, 0.20605}, {28, 0.7137375}, {0, 0}
};
//NMR I=7/2 satellite 3/2-1/2
static const struct XRelaxFuncPoly::Term  s_relaxdata_nmr7s1[] = {
    {1, 0.01191}, {3, 0.05952}, {6, 0.030305}, {10, 0.17532}, {15, 0.000915}, {21, 0.26513}, {28, 0.45678}, {0, 0}
};
//NMR I=7/2 satellite 5/2-3/2
static const struct XRelaxFuncPoly::Term  s_relaxdata_nmr7s2[] = {
	{28,0.11422}, {21,0.37121}, {15,0.3663}, {10,0.081169}, {6,0.0075758}, {3,0.047619}, {1,0.011905} , {0, 0}
};
//NMR I=7/2 satellite 7/2-5/2
static const struct XRelaxFuncPoly::Term  s_relaxdata_nmr7s3[] = {
	{28,0.009324}, {21,0.068182}, {15,0.20604}, {10,0.32468}, {6,0.27273}, {3,0.10714}, {1,0.011905}, {0, 0}
};
//NMR I=9/2 center
static const struct XRelaxFuncPoly::Term  s_relaxdata_nmr9c[] = {
	{45,0.65306}, {28,0.215}, {15,0.092308}, {6,0.033566}, {1,0.0060606}, {0, 0}
};
//NMR I=9/2 satellite 3/2-1/2
static const struct XRelaxFuncPoly::Term  s_relaxdata_nmr9s1[] = {
	{45,0.45352}, {36,0.30839}, {28,0.0033594}, {21,0.14848}, {15,0.016026}, {10,0.039336}, {6,0.021037}, {3,0.0037879}, {1,0.0060606}, {0, 0}
};
//NMR I=9/2 satellite 5/2-3/2
static const struct XRelaxFuncPoly::Term  s_relaxdata_nmr9s2[] = {
	{45,0.14809}, {36,0.4028}, {28,0.28082}, {21,0.012121}, {15,0.064103}, {10,0.06993}, {6,0.0009324}, {3,0.015152}, {1,0.0060606}, {0, 0}
};
//NMR I=9/2 satellite 7/2-5/2
static const struct XRelaxFuncPoly::Term  s_relaxdata_nmr9s3[] = {
	{45,0.020825}, {36,0.12745}, {28,0.30318}, {21,0.33409}, {15,0.14423}, {10,0.0043706}, {6,0.025699}, {3,0.034091}, {1,0.0060606}, {0, 0}
};
//NMR I=9/2 satellite 9/2-7/2
static const struct XRelaxFuncPoly::Term  s_relaxdata_nmr9s4[] = {
	{45,0.0010284}, {36,0.011189}, {28,0.05375}, {21,0.14848}, {15,0.25641}, {10,0.27972}, {6,0.18275}, {3,0.060606}, {1,0.0060606}, {0, 0}
};

XRelaxFuncList::XRelaxFuncList(const char *name, bool runtime)
: XAliasListNode<XRelaxFunc>(name, runtime)
{
 create<XRelaxFuncPoly>("NMR I=1/2", true, s_relaxdata_nmr1);
	 create<XRelaxFuncPoly>("NMR I=1", true, s_relaxdata_nmr2);
		 create<XRelaxFuncPoly>("NMR I=3/2 center a", true, s_relaxdata_nmr3ca);
			 create<XRelaxFuncPoly>("NMR I=3/2 center b", true, s_relaxdata_nmr3cb);
				 create<XRelaxFuncPoly>("NMR I=3/2 satellite", true, s_relaxdata_nmr3s);
					 create<XRelaxFuncPoly>("NMR I=5/2 center a", true, s_relaxdata_nmr5ca);
						 create<XRelaxFuncPoly>("NMR I=5/2 center b", true, s_relaxdata_nmr5cb);
							 create<XRelaxFuncPoly>("NMR I=5/2 satellite 3/2-1/2", true, s_relaxdata_nmr5s);
								 create<XRelaxFuncPoly>("NMR I=5/2 satellite 5/2-3/2", true, s_relaxdata_nmr5s2);
									 create<XRelaxFuncPoly>("NMR I=3 1-0", true, s_relaxdata_nmr6c);
										 create<XRelaxFuncPoly>("NMR I=3 2-1", true, s_relaxdata_nmr6s1);
											 create<XRelaxFuncPoly>("NMR I=3 3-2", true, s_relaxdata_nmr6s2);
												 create<XRelaxFuncPoly>("NMR I=7/2 center", true, s_relaxdata_nmr7c);
													 create<XRelaxFuncPoly>("NMR I=7/2 satellite 3/2-1/2", true, s_relaxdata_nmr7s1);
														 create<XRelaxFuncPoly>("NMR I=7/2 satellite 5/2-3/2", true, s_relaxdata_nmr7s2);
															 create<XRelaxFuncPoly>("NMR I=7/2 satellite 7/2-5/2", true, s_relaxdata_nmr7s3);
																 create<XRelaxFuncPoly>("NMR I=9/2 center", true, s_relaxdata_nmr9c);
																	 create<XRelaxFuncPoly>("NMR I=9/2 satellite 3/2-1/2", true, s_relaxdata_nmr9s1);
																		 create<XRelaxFuncPoly>("NMR I=9/2 satellite 5/2-3/2", true, s_relaxdata_nmr9s2);
																			 create<XRelaxFuncPoly>("NMR I=9/2 satellite 7/2-5/2", true, s_relaxdata_nmr9s3);
																				 create<XRelaxFuncPoly>("NMR I=9/2 satellite 9/2-7/2", true, s_relaxdata_nmr9s4);
																					 create<XRelaxFuncPoly>("NQR I=1", true, s_relaxdata_nqr2);
																						 create<XRelaxFuncPoly>("NQR I=3/2", true, s_relaxdata_nqr3);
																							 create<XRelaxFuncPoly>("NQR I=5/2 5/2-3/2", true, s_relaxdata_nqr5_5);
																								 create<XRelaxFuncPoly>("NQR I=5/2 3/2-1/2", true, s_relaxdata_nqr5_3);
																									 create<XRelaxFuncPoly>("NQR I=3 3-2", true, s_relaxdata_nqr6_6);
																										 create<XRelaxFuncPoly>("NQR I=3 2-1", true, s_relaxdata_nqr6_4);
																											 create<XRelaxFuncPoly>("NQR I=3 1-0", true, s_relaxdata_nqr6_2);
																												 create<XRelaxFuncPoly>("NQR I=7/2 7/2-5/2", true, s_relaxdata_nqr7_7);
																													 create<XRelaxFuncPoly>("NQR I=7/2 5/2-3/2", true, s_relaxdata_nqr7_5);
																														 create<XRelaxFuncPoly>("NQR I=7/2 3/2-2/1", true, s_relaxdata_nqr7_3);
																															 create<XRelaxFuncPoly>("NQR I=9/2 9/2-7/2", true, s_relaxdata_nqr9_9);
																																 create<XRelaxFuncPoly>("NQR I=9/2 7/2-5/2", true, s_relaxdata_nqr9_7);
																																	 create<XRelaxFuncPoly>("NQR I=9/2 5/2-3/2", true, s_relaxdata_nqr9_5);
																																		 create<XRelaxFuncPoly>("NQR I=9/2 3/2-2/1", true, s_relaxdata_nqr9_3);
																																			 create<XRelaxFuncSqrt>("Random Spins; exp(-sqrt(t/tau))", true);   
																																				 }


int
XRelaxFunc::relax_f (const gsl_vector * x, void *params,
					 gsl_vector * f)
{
 XNMRT1::NLLS *data = ((XNMRT1::NLLS *)params);
	 double iT1 = gsl_vector_get (x, 0);
		 double c = gsl_vector_get (x, 1);
			 
			 double a = gsl_vector_get (x, 2);
				 
				 int i = 0;
					 for(std::deque<XNMRT1::Pt>::iterator it = data->pts->begin();
							 it != data->pts->end(); it++)
						 {
 if(it->isigma == 0) continue;
						 double t = it->p1;
							 double yi = 0, dydt = 0;
								 data->func->relax(&yi, &dydt, t, iT1);
									 double y = it->var;
										 gsl_vector_set (f, i, (c * yi + a - y) * it->isigma);
											 i++;
												 }
															  return GSL_SUCCESS;
																  }
int
XRelaxFunc::relax_df (const gsl_vector * x, void *params,
					  gsl_matrix * J)
{
 
 XNMRT1::NLLS *data = ((XNMRT1::NLLS *)params);
	 double iT1 = gsl_vector_get (x, 0);
		 double c = gsl_vector_get (x, 1);
//  double a = gsl_vector_get (x, 2);
			 
			 int i = 0;
				 for(std::deque<XNMRT1::Pt>::iterator it = data->pts->begin();
						 it != data->pts->end(); it++)
					 {
 if(it->isigma == 0) continue;
						 double t = it->p1;
							 double yi = 0, dydt = 0;
								 data->func->relax(&yi, &dydt, t, iT1);
									 gsl_matrix_set (J, i, 0, (c * dydt) * it->isigma);
										 gsl_matrix_set (J, i, 1, yi * it->isigma);
											 gsl_matrix_set (J, i, 2, it->isigma);
												 i++;
													 }
														  return GSL_SUCCESS;
															  }
int
XRelaxFunc::relax_fdf (const gsl_vector * x, void *params,
					   gsl_vector * f, gsl_matrix * J)
{
 XNMRT1::NLLS *data = ((XNMRT1::NLLS *)params);
	 double iT1 = gsl_vector_get (x, 0);
		 
		 double c = gsl_vector_get (x, 1);
			 double a = gsl_vector_get (x, 2);
				 
				 int i = 0;
					 for(std::deque<XNMRT1::Pt>::iterator it = data->pts->begin();
							 it != data->pts->end(); it++)
						 {
 if(it->isigma == 0) continue;
						 double t = it->p1;
							 double yi = 0, dydt = 0;
								 data->func->relax(&yi, &dydt, t, iT1);
									 double y = it->var;
										 gsl_vector_set (f, i, (c * yi + a - y) * it->isigma);
											 gsl_matrix_set (J, i, 0, (c * dydt) * it->isigma);
												 gsl_matrix_set (J, i, 1, yi * it->isigma);
													 gsl_matrix_set (J, i, 2, it->isigma);
														 i++;
															 }
															  return GSL_SUCCESS;
																  }
std::string
XNMRT1::iterate(shared_ptr<XRelaxFunc> &func,
				int itercnt)
{
 int n = 0;
	 for(std::deque<Pt>::iterator it = m_sumpts.begin(); it != m_sumpts.end(); it++)
		 {
 if(it->isigma > 0)  n++;
						 }    
																						struct NLLS nlls = {
																							&m_sumpts,
																							func
																						};
																							int p = 3;
																								if(n <= p) return formatString("%d",n) + KAME::i18n(" points, more points needed.");
																											   int status;
																												   double norm = 0;
																													   XTime firsttime = XTime::now();
																														   for(;;)
																															   {
 status = do_nlls(n, p, m_params, m_errors, &norm,
				  &nlls, &XRelaxFunc::relax_f, &XRelaxFunc::relax_df, &XRelaxFunc::relax_fdf, itercnt);
	 if(!status) break;
					 if(XTime::now() - firsttime < 0.01) continue;
															 if(XTime::now() - firsttime > 0.05) break;
																									 double p1max = *p1Max();
																										 double p1min = *p1Min();
																											 m_params[0] = 1.0 / exp(log(p1max/p1min) * (((double)KAME::rand())/RAND_MAX) + log(p1min));
																												 m_params[1] = 0.1 + 0.5*(((double)KAME::rand())/RAND_MAX);
																													 m_params[2] = 0.0;
																														 status = do_nlls(n, p, m_params, m_errors, &norm,
																																		  &nlls, &XRelaxFunc::relax_f, &XRelaxFunc::relax_df, &XRelaxFunc::relax_fdf, itercnt);
																															 }
																																	  m_errors[0] *= norm / sqrt((double)n);
																																		  m_errors[1] *= norm / sqrt((double)n);
																																			  m_errors[2] *= norm / sqrt((double)n);
																																				  
																																				  double t1 = 0.001 / m_params[0];
																																					  double t1err = 0.001 / pow(m_params[0], 2.0) * m_errors[0];
																																						  QString buf = "";
																																							  if(*t2Mode())
																																								  {
 buf += QString().sprintf("1/T2[1/ms] = %.5f +/- %.5f (%.2f%%)\n",
						  1000.0 * m_params[0], 1000.0 * m_errors[0], fabs(100.0 * m_errors[0]/m_params[0]));
	 buf += QString().sprintf("T2[ms] = %.6f +/- %.6f (%.2f%%)\n",
							  t1, t1err, fabs(100.0 * t1err/t1));
		 }
																																											   else
																																												   {
 buf += QString().sprintf("1/T1[1/s] = %.5f +/- %.5f (%.2f%%)\n",
						  1000.0 * m_params[0], 1000.0 * m_errors[0], fabs(100.0 * m_errors[0]/m_params[0]));
	 buf += QString().sprintf("T1[s] = %.7f +/- %.7f (%.2f%%)\n",
							  t1, t1err, fabs(100.0 * t1err/t1));
		 }
																																											   buf += QString().sprintf("c[V] = %.5g +/- %.5g (%.3f%%)\n",
																																																		m_params[1], m_errors[1], fabs(100.0 * m_errors[1]/m_params[1]));
																																												   buf += QString().sprintf("a[V] = %.5g +/- %.5g (%.3f%%)\n",
																																																			m_params[2], m_errors[2], fabs(100.0 * m_errors[2]/m_params[2]));
																																													   buf += QString().sprintf("status = %s\n", gsl_strerror (status));
																																														   buf += QString().sprintf("rms of residuals = %.3g\n", norm / sqrt((double)n));
																																															   buf += QString().sprintf("elapsed time = %.2f ms\n", 1000.0 * (XTime::now() - firsttime));
																																																   return buf;
																																																	   }

int
do_nlls(int n, int p, double *param, double *err, double *det, void *user, exp_f  *ef, exp_df *edf, exp_fdf *efdf
		, int itercnt)
{
 const gsl_multifit_fdfsolver_type *T;
	 T = gsl_multifit_fdfsolver_lmsder;
		 gsl_multifit_fdfsolver *s;
			 int iter = 0;
				 int status;
					 int i;
						 double c;
							 gsl_multifit_function_fdf f;
								 
								 gsl_ieee_env_setup ();
									 
									 f.f = ef;
										 f.df = edf;
											 f.fdf = efdf;
												 f.n = n;
													 f.p = p;
														 f.params = user;
															 s = gsl_multifit_fdfsolver_alloc (T, n, p);
																 gsl_vector_view x = gsl_vector_view_array (param, p);
																	 gsl_multifit_fdfsolver_set (s, &f, &x.vector);
																		 
																		 
																		 do
																			 {
 iter++;
	 status = gsl_multifit_fdfsolver_iterate (s);
		 
		 if (status)
						break;
						
						status = gsl_multifit_test_delta (s->dx, s->x,
														  1e-4, 1e-4);
							}
																			   while (status == GSL_CONTINUE && iter < itercnt);
																			   
																			   if(det) *det = gsl_blas_dnrm2 (s->f);
																						   for(i = 0; i < p; i++)
																													 param[i] = gsl_vector_get (s->x, i);
																													 
																													 gsl_matrix *covar = gsl_matrix_alloc (p, p);
																														 gsl_multifit_covar (s->J, 0.0, covar);
																															 for(i = 0; i < p; i++)
																																 {
 c = gsl_matrix_get(covar,i,i);
	 
	 err[i] = (c > 0) ? sqrt(c) : -1.0;
		 }
																																					   gsl_matrix_free(covar);
																																						   gsl_multifit_fdfsolver_free (s);
																																							   
																																							   return status;
																																								   }

