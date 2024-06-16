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
#include "nmrrelax.h"
#include "nmrrelaxfit.h"
#include "rand.h"

#include "nllsfit.h"
//---------------------------------------------------------------------------

class XRelaxFuncPoly : public XRelaxFunc {
public:
//! define a term in a relaxation function
//! a*exp(-p*t/T1)
    struct Term {
        int p;
        double a;
    };

    XRelaxFuncPoly(const char *name, bool runtime, const Term *terms)
        : XRelaxFunc(name, runtime), m_terms(terms) {
    }
    virtual ~XRelaxFuncPoly() {}

    //! called during fitting
    //! \param f f(t, it1) will be passed
    //! \param dfdt df/d(it1) will be passed
    //! \param t a time P1 or 2tau
    //! \param it1 1/T1 or 1/T2
    virtual void relax(double *f, double *dfdt, double t, double it1) {
        double rf = 0, rdf = 0;
        double x = -t * it1;
        x = std::min(5.0, x);
        for(const Term *term = m_terms; term->p != 0; term++) {
            double a = term->a * exp(x*term->p);
            rf += a;
            rdf += a * term->p;
        }
        rdf *= -t;
        *f = 1.0 - rf;
        *dfdt = -rdf;
    }
private:
    const struct Term *m_terms;
};
//! Power exponential.
class XRelaxFuncPowExp : public XRelaxFunc {
public:
    XRelaxFuncPowExp(const char *name, bool runtime, double pow)
        : XRelaxFunc(name, runtime), m_pow(pow)	{}
    virtual ~XRelaxFuncPowExp() {}

    //! called during fitting
    //! \param f f(t, it1) will be passed
    //! \param dfdt df/d(it1) will be passed
    //! \param t a time P1 or 2tau
    //! \param it1 1/T1 or 1/T2
    virtual void relax(double *f, double *dfdt, double t, double it1) {
        it1 = std::max(0.0, it1);
        double rt = pow(t * it1, m_pow);
        double a = exp(-rt);
        *f = 1.0 - a;
        *dfdt = t*rt/(t*it1)*m_pow * a;
    }
private:
    const double m_pow;
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
    {3, 9.0/132}, {10, 5.0/572}, {21, 441.0/660}, {36, 729.0/2860}, {0, 0}
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
    : XAliasListNode<XRelaxFunc>(name, runtime) {
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
    create<XRelaxFuncPowExp>("Pow.Exp.0.5: exp(-t^0.5)", true, 0.5);
    create<XRelaxFuncPowExp>("Pow.Exp.0.6: exp(-t^0.6)", true, 0.6);
    create<XRelaxFuncPowExp>("Pow.Exp.0.7: exp(-t^0.7)", true, 0.7);
    create<XRelaxFuncPowExp>("Pow.Exp.0.8: exp(-t^0.8)", true, 0.8);
    create<XRelaxFuncPowExp>("Gaussian: exp(-t^2)", true, 2.0);
    create<XRelaxFuncPowExp>("Exp.: exp(-t)", true, 1.0);
}

XString
XNMRT1::iterate(Transaction &tr, shared_ptr<XRelaxFunc> &func, int itercnt) {
    const Snapshot &shot_this(tr);
    //# of samples.
    int n = 0;
    double max_var = -1e90, min_var = 1e90;
    for(auto it = shot_this[ *this].m_sumpts.begin(); it != shot_this[ *this].m_sumpts.end(); it++) {
        if(it->isigma > 0)  n++;
        max_var = std::max(max_var, it->var);
        min_var = std::min(min_var, it->var);
    }
    //# of indep. params.
    int p = shot_this[ *mInftyFit()] ? 3 : 2;
    if(n <= p) return formatString("%d",n) + i18n(" points, more points needed.");

    const auto &values = tr[ *this].m_sumpts;
    auto relax_f = [&values, &func](const double*params, size_t n, size_t p,
            double *f, std::vector<double *> &df) -> bool {
        double iT1 = params[0];
        double c = params[1];
        double a = (p == 3) ? params[2] : -c;

        int i = 0;
        for(auto &&pt : values) {
            if(pt.isigma == 0) continue;
            double t = pt.p1;
            double yi = 0, dydt = 0;
            func->relax(&yi, &dydt, t, iT1);
            if(f) {
                f[i] = (c * yi + a - pt.var) * pt.isigma;
            }
            df[0][i] = (c * dydt) * pt.isigma;
            df[1][i] = yi * pt.isigma;
            if(p == 3)
                df[2][i] = pt.isigma;
            i++;
        }
        return true;
    };

    XTime firsttime = XTime::now();
    NonLinearLeastSquare nlls;
    for(;;) {
        std::valarray<double> init_params(p);
        for(int i = 0; i < p; ++i)
            init_params[i] = tr[ *this].m_params[i];
        nlls = NonLinearLeastSquare(relax_f, init_params, n, itercnt);
        for(int i = 0; i < p; ++i) {
            tr[ *this].m_params[i] = nlls.params()[i];
            tr[ *this].m_errors[i] = nlls.errors()[i];
        }
        if(nlls.isSuccessful() && (fabs(tr[ *this].m_params[1]) < (max_var - min_var) * 10))
            break;
        if(XTime::now() - firsttime < 0.02) continue;
        if(XTime::now() - firsttime > 0.07) break;
        double p1max = shot_this[ *p1Max()];
        double p1min = shot_this[ *p1Min()];
        tr[ *this].m_params[0] = 1.0 / exp(log(p1max/p1min) * randMT19937() + log(p1min));
        tr[ *this].m_params[1] = (max_var - min_var) * (randMT19937() * 2.0 + 0.9) * ((randMT19937() < 0.5) ? 1 :  -1);
        tr[ *this].m_params[2] = 0.0;
    }

    if( !shot_this[ *mInftyFit()])
        tr[ *this].m_params[2] = -tr[ *this].m_params[1];

    double t1 = 0.001 / shot_this[ *this].m_params[0];
    double t1err = 0.001 / pow(shot_this[ *this].m_params[0], 2.0) * shot_this[ *this].m_errors[0];
    XString buf = "";
    switch((MeasMode)(int)shot_this[ *mode()]) {
    case MeasMode::ST_E:
    case MeasMode::T1:
        buf += formatString("1/T1[1/s] = %.5g +- %.3g(%.2f%%)\n",
                                 1000.0 * shot_this[ *this].m_params[0],
                                 1000.0 * shot_this[ *this].m_errors[0],
                                 fabs(100.0 * shot_this[ *this].m_errors[0]/shot_this[ *this].m_params[0]));
        buf += formatString("T1[s] = %.5g +- %.3g(%.2f%%)\n",
                                 t1, t1err, fabs(100.0 * t1err/t1));
        break;
    case MeasMode::T2:
    case MeasMode::T2_Multi:
        buf += formatString("1/T2[1/ms] = %.5g +- %.3g(%.2f%%)\n",
                                 1000.0 * shot_this[ *this].m_params[0],
                                 1000.0 * shot_this[ *this].m_errors[0],
                                 fabs(100.0 * shot_this[ *this].m_errors[0]/shot_this[ *this].m_params[0]));
        buf += formatString("T2[ms] = %.5g +- %.3g(%.2f%%)\n",
                                 t1, t1err, fabs(100.0 * t1err/t1));
        break;
    }
    buf += formatString("c[V] = %.5g +- %.3g(%.3f%%)\n",
        shot_this[ *this].m_params[1], shot_this[ *this].m_errors[1],
        fabs(100.0 * shot_this[ *this].m_errors[1]/shot_this[ *this].m_params[1]));
    buf += formatString("a[V] = %.5g +- %.3g(%.3f%%)\n",
        shot_this[ *this].m_params[2], shot_this[ *this].m_errors[2],
        fabs(100.0 * shot_this[ *this].m_errors[2]/shot_this[ *this].m_params[2]));
    buf += formatString("status = %s\n", nlls.status().c_str());
    buf += formatString("rms of residuals = %.3g\n", sqrt(nlls.chiSquare() / n));
    buf += formatString("elapsed time = %.2f ms\n", 1000.0 * (XTime::now() - firsttime));
    return buf;
}

