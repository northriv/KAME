/***************************************************************************
        Copyright (C) 2002-2024 Kentaro Kitagawa
		                   kitag@issp.u-tokyo.ac.jp
		
		This program is free software; you can redistribute it and/or
		modify it under the terms of the GNU Library General Public
		License as published by the Free Software Foundation; either
		version 2 of the License, or (at your option) any later version.
		
		You should have received a copy of the GNU Library General 
		Public License and a list of authors along with this program; 
		see the files COPYING and AUTHORS.
***************************************************************************/
//---------------------------------------------------------------------------

#ifndef graphmathfittoolH
#define graphmathfittoolH
//---------------------------------------------------------------------------

#include "graphmathtool.h"
#include "nllsfit.h"
#include "rand.h"

template <class F, unsigned int P>
class XGraph1DMathFitToolX: public XGraph1DMathTool {
public:
    XGraph1DMathFitToolX(const char *name, bool runtime, Transaction &tr_meas,
                      const shared_ptr<XScalarEntryList> &entries, const shared_ptr<XDriver> &driver,
                      const shared_ptr<XPlot> &plot) :
        XGraph1DMathTool(name, runtime, ref(tr_meas), entries, driver, plot) {
         m_entry = create<XScalarEntry>(getName().c_str(), false, driver);
         m_entry_err = create<XScalarEntry>((getName() + "_err").c_str(), false, driver);
         entries->insert(tr_meas, m_entry);
         entries->insert(tr_meas, m_entry_err);
    }
    virtual ~XGraph1DMathFitToolX() {}
    virtual void update(Transaction &tr, XQGraph *graphwidget, cv_iterator xbegin, cv_iterator xend, cv_iterator ybegin, cv_iterator yend) override {
        using namespace std::placeholders;
        auto func = std::bind(&F::fitFunc, xbegin, xend, _1, _4, _5);

        XTime firsttime = XTime::now();
        NonLinearLeastSquare nlls;
        double v = 0.0;
        double v_err = 0.0;
        for(;;) {
            std::valarray<double> init_params
                    = F::initParams(xbegin, xend, ybegin, yend);
            nlls = NonLinearLeastSquare(func, init_params, (size_t)(xend - xbegin));
            std::tie(v, v_err) = F::result(P, &nlls.params()[0], &nlls.errors()[0]);
            if(nlls.isSuccessful())
                break;
            if(XTime::now() - firsttime < 0.02) continue;
            if(XTime::now() - firsttime > 0.07) break;
        }
        m_entry->value(tr, v);
        m_entry_err->value(tr, v_err);
        updateOnScreenObjects(tr, graphwidget);
    }
    const shared_ptr<XScalarEntry> entry() const {return m_entry;}
    virtual void releaseEntries(Transaction &tr) override {
        entries()->release(tr, m_entry_err);
        entries()->release(tr, m_entry);}
private:
    shared_ptr<XScalarEntry> m_entry, m_entry_err;
};

template <class F>
class XGraph2DMathFitToolX: public XGraph2DMathTool {
public:
    XGraph2DMathFitToolX(const char *name, bool runtime, Transaction &tr_meas,
                      const shared_ptr<XScalarEntryList> &entries, const shared_ptr<XDriver> &driver,
                      const shared_ptr<XPlot> &plot) :
        XGraph2DMathTool(name, runtime, ref(tr_meas), entries, driver, plot) {
        m_entry = create<XScalarEntry>(getName().c_str(), false, driver);
        entries->insert(tr_meas, m_entry);
    }
    virtual ~XGraph2DMathFitToolX() {}
    virtual void update(Transaction &tr, XQGraph *graphwidget, const uint32_t *leftupper, unsigned int width,
        unsigned int stride, unsigned int numlines, double coefficient) override {
        double v = F()(leftupper, width, stride, numlines, coefficient);
        m_entry->value(tr, v);
        updateOnScreenObjects(tr, graphwidget);
    }
    const shared_ptr<XScalarEntry> entry() const {return m_entry;}
    virtual void releaseEntries(Transaction &tr) override {entries()->release(tr, m_entry);}
private:
    shared_ptr<XScalarEntry> m_entry;
};

struct FuncGraph1DMathGaussianFitTool{
    using cv_iterator = std::vector<XGraph::VFloat>::const_iterator;
    static bool fitFunc(cv_iterator xbegin, cv_iterator xend,
        const double*params, double *f, std::vector<double *> &df){
        double x0 = params[0];
        double isigma = params[1]; //1/sigma
        double isigma_sq = isigma * isigma;
        double height = params[2];
        double y0 = params[3];
        unsigned int j = 0;
        for(auto xit = xbegin; xit != xend; ++xit) {
            double dx = *xit - x0;
            double dy = height * exp( - dx * dx * isigma_sq / 2);
            *f++ = dy + y0;
            df[j][0] = dx * isigma_sq * dy; //dy/dx0
            df[j][1] = -dx * dx * isigma * dy; //dy/d(1/sigma)
            df[j][2] = dy / height; //dy/da
            df[j][3] = 1;
            j++;
        }
        return true;
    }
    static std::array<double, 2> result(unsigned int p, const double*params, const double*errors) {
        switch(p) {
        default:
        case 0: return {params[0], errors[0]};
        case 1: {
            double fwhm = 2*std::sqrt(2.0*std::log(2.0))/params[1];
            return {fwhm, fwhm * errors[1] / params[1]}; //FWHM
        }
        case 2: return {params[2], errors[2]};
        case 3: return {params[3], errors[3]};
        }
    }
    static std::valarray<double> initParams(cv_iterator xbegin, cv_iterator xend, cv_iterator ybegin, cv_iterator yend){
        return {( *std::max_element(xbegin, xend) - *std::min_element(xbegin, xend)) * randMT19937() + *std::min_element(xbegin, xend),
                                ( *std::max_element(xbegin, xend) - *std::min_element(xbegin, xend)) * randMT19937() * 2,
                                ( *std::max_element(ybegin, yend) - *std::min_element(ybegin, yend)) * randMT19937() * 2,
                                *std::min_element(ybegin, yend) * 2.0 * randMT19937(),
        };
    }
};

using XGraph1DMathGaussianPositionTool = XGraph1DMathFitToolX<FuncGraph1DMathGaussianFitTool, 0>;
using XGraph1DMathGaussianFWHMTool = XGraph1DMathFitToolX<FuncGraph1DMathGaussianFitTool, 1>;
using XGraph1DMathGaussianHeightTool = XGraph1DMathFitToolX<FuncGraph1DMathGaussianFitTool, 2>;
using XGraph1DMathGaussianBaselineTool = XGraph1DMathFitToolX<FuncGraph1DMathGaussianFitTool, 3>;
#endif
