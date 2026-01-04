/***************************************************************************
        Copyright (C) 2002-2025 Kentaro Kitagawa
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
#include "graphpainter.h"
#include "nllsfit.h"
#include "rand.h"


template <class F>
struct DECLSPEC_KAME OnXAxisMathFitToolObject : public OnAxisFuncObject<true> {
    OnXAxisMathFitToolObject(XQGraphPainter* p, const shared_ptr<XNode> &pickable_node) : OnAxisFuncObject<true>(p, pickable_node) {}

    NonLinearLeastSquare nlls_result;
protected:
    virtual std::vector<XGraph::VFloat> func(const std::vector<XGraph::VFloat> &x,
                                             std::vector<XGraph::VFloat>&& y) {
        if( !nlls_result.isSuccessful())
            return {};
        F::func( &nlls_result.params()[0], x.begin(), x.end(), y.begin(), y.end());
        return std::move(y);
    }
};

template <class F, unsigned int P>
class DECLSPEC_KAME XGraph1DMathFitToolX: public XGraph1DMathTool {
public:
    XGraph1DMathFitToolX(const char *name, bool runtime, Transaction &tr_meas,
                      const shared_ptr<XScalarEntryList> &entries, const shared_ptr<XDriver> &driver,
                      const shared_ptr<XPlot> &plot, const shared_ptr<XNode> &parentList, const std::vector<std::string> &entrynames) :
        XGraph1DMathTool(name, runtime, ref(tr_meas), entries, driver, plot, parentList) {
         m_entry = create<XScalarEntry>(
            entrynames[0].c_str(), false, driver);
         m_entry_err = create<XScalarEntry>(
            (XString(entrynames[0]) + "_err").c_str(), false, driver);
         entries->insert(tr_meas, m_entry);
         entries->insert(tr_meas, m_entry_err);
    }
    virtual ~XGraph1DMathFitToolX() {}
    virtual void update(Transaction &tr, const shared_ptr<XQGraphPainter> &painter, cv_iterator xbegin, cv_iterator xend, cv_iterator ybegin, cv_iterator yend) override {
        using namespace std::placeholders;
        auto func = std::bind(&F::fitFunc, xbegin, xend, ybegin, yend, _1, _4, _5);

        for(auto it = ybegin; it != yend; ++it)
            if(std::isnan( *it))
                return; //gsl may crash if nan included.

        XTime firsttime = XTime::now();
        NonLinearLeastSquare nlls;
        double v = 0.0;
        double v_err = 0.0;
        double cost_min = 1e20;
        for(int retry = 0 ;retry < 60; retry++) {
            std::valarray<double> init_params
                    = F::initParams(xbegin, xend, ybegin, yend);
            if( !init_params.size()) return;
            for(auto x: init_params)
                if(std::isnan(x))
                    return;
            auto nllsnew = NonLinearLeastSquare(func, init_params, std::distance(xbegin, xend), 100);
            if(nllsnew.isSuccessful()) {
                if(cost_min > nllsnew.chiSquare()) {
                    cost_min = nllsnew.chiSquare();
                    nlls = std::move(nllsnew);
                    if((retry > 2) && (cost_min / nlls.chiSquare() < 1.01))
                        break; //enough good
                }
            }
            if(XTime::now() - firsttime < 0.01) continue;
            if(XTime::now() - firsttime > 0.07) break;
        }

        if( !nlls.isSuccessful())
            return;

        std::tie(v, v_err) = F::result(P, &nlls.params()[0], &nlls.errors()[0]);
        m_entry->value(tr, v);
        m_entry_err->value(tr, v_err);
        XString msg = tr[ *m_entry->value()].to_str() + " +- " + tr[ *m_entry_err->value()].to_str();

        if(auto plot = m_plot.lock()) {
            double bgx = tr[ *begin()];
            double edx = tr[ *end()];
            double bgy = 0.0;
            double edy = 1.0;
            if(auto oso = m_osoFitCurve.lock()) {
                oso->nlls_result = std::move(nlls);
                oso->setBaseColor(tr[ *baseColor()]);
                oso->placeObject(plot, bgx, edx, bgy, edy, {0.0, 0.0, -0.001});
            }
        }
        updateOnScreenObjects(tr, painter, std::move(msg));
    }
    const shared_ptr<XScalarEntry> entry() const {return m_entry;}
    virtual bool releaseEntries(Transaction &tr) override {
        if( !entries()->release(tr, m_entry_err))
            return false;//transaction has failed.
        return entries()->release(tr, m_entry);
    }
protected:
    virtual void updateAdditionalOnScreenObjects(const Snapshot &shot, const shared_ptr<XQGraphPainter> &painter, const XString &msg) override {
        XGraph1DMathTool::updateAdditionalOnScreenObjects(shot, painter, msg);
    }
    virtual std::deque<shared_ptr<OnScreenObject>> createAdditionalOnScreenObjects(const shared_ptr<XQGraphPainter> &painter) override {
        auto osos = XGraph1DMathTool::createAdditionalOnScreenObjects(painter);
        auto oso = painter->createOnScreenObjectWeakly<OnXAxisMathFitToolObject<F>>(shared_from_this());
        m_osoFitCurve = oso;
        osos.push_back(oso);
        return osos;
    }
private:
    shared_ptr<XScalarEntry> m_entry, m_entry_err;
    weak_ptr<OnXAxisMathFitToolObject<F>> m_osoFitCurve;
};

struct DECLSPEC_KAME FuncGraph1DMathGaussianFitTool{
    using v_iterator = std::vector<XGraph::VFloat>::iterator;
    using cv_iterator = std::vector<XGraph::VFloat>::const_iterator;
    static bool fitFunc(cv_iterator xbegin, cv_iterator xend, cv_iterator ybegin, cv_iterator yend,
        const double*params, double *f, std::vector<double *> &df){
        double x0 = params[0];
        double isigma = params[1]; //1/sigma
        double isigma_sq = isigma * isigma;
        double height = params[2];
        double y0 = params[3];
        unsigned int j = 0;
        auto yit = ybegin;
        for(auto xit = xbegin; xit != xend; ++xit) {
            double dx = *xit - x0;
            double dy = height * exp( - dx * dx * isigma_sq / 2);
            if(f)
                *f++ = dy + y0 - *yit++;
            df[0][j] = dx * isigma_sq * dy; //dy/dx0
            df[1][j] = -dx * dx * isigma * dy; //dy/d(1/sigma)
            df[2][j] = dy / height; //dy/da
            df[3][j] = 1;
            j++;
        }
        return true;
    }
    static void func(const double*params, cv_iterator xbegin, cv_iterator xend, v_iterator ybegin, v_iterator yend) {
        double x0 = params[0];
        double isigma = params[1]; //1/sigma
        double isigma_sq = isigma * isigma;
        double height = params[2];
        double y0 = params[3];
        auto yit = ybegin;
        for(auto xit = xbegin; xit != xend; ++xit) {
            double dx = *xit - x0;
            double dy = height * exp( - dx * dx * isigma_sq / 2);
            *yit++ = dy + y0;
        }
    }
    static std::tuple<double, double> result(unsigned int p, const double*params, const double*errors) {
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
        if(std::distance(xbegin, xend) < 4)
            return {};
        double xmax = *std::max_element(xbegin, xend);
        double xmin = *std::min_element(xbegin, xend);
        double ymax = *std::max_element(ybegin, yend);
        double ymin = *std::min_element(ybegin, yend);
        return {(xmax - xmin) * randMT19937() + xmin,
            5.0/(xmax - xmin) * randMT19937(),
            (ymax - ymin) * (2 * randMT19937() - 1),
            (ymax - ymin) * (2 * randMT19937() - 1),
        };
    }
};

using XGraph1DMathGaussianPositionTool = XGraph1DMathFitToolX<FuncGraph1DMathGaussianFitTool, 0>;
using XGraph1DMathGaussianFWHMTool = XGraph1DMathFitToolX<FuncGraph1DMathGaussianFitTool, 1>;
using XGraph1DMathGaussianHeightTool = XGraph1DMathFitToolX<FuncGraph1DMathGaussianFitTool, 2>;
using XGraph1DMathGaussianBaselineTool = XGraph1DMathFitToolX<FuncGraph1DMathGaussianFitTool, 3>;

struct DECLSPEC_KAME FuncGraph1DMathLorenzianFitTool{
    using v_iterator = std::vector<XGraph::VFloat>::iterator;
    using cv_iterator = std::vector<XGraph::VFloat>::const_iterator;
    static bool fitFunc(cv_iterator xbegin, cv_iterator xend, cv_iterator ybegin, cv_iterator yend,
        const double*params, double *f, std::vector<double *> &df){
        double x0 = params[0];
        double igamma = params[1]; //1/gamma
        double height = params[2];
        double y0 = params[3];
        unsigned int j = 0;
        auto yit = ybegin;
        for(auto xit = xbegin; xit != xend; ++xit) {
            double dx = (*xit - x0) * igamma;
            double dy = height / (dx * dx + 1);
            if(f)
                *f++ = dy + y0 - *yit++;
            df[0][j] = 2 * dx * igamma * dy * dy / height; //dy/dx0
            df[1][j] = -2 * dx * (*xit - x0) * dy * dy / height; //dy/d(1/gamma)
            df[2][j] = dy / height; //dy/da
            df[3][j] = 1;
            j++;
        }
        return true;
    }
    static void func(const double*params, cv_iterator xbegin, cv_iterator xend, v_iterator ybegin, v_iterator yend) {
        double x0 = params[0];
        double igamma = params[1]; //1/gamma
        double height = params[2];
        double y0 = params[3];
        unsigned int j = 0;
        auto yit = ybegin;
        for(auto xit = xbegin; xit != xend; ++xit) {
            double dx = (*xit - x0) * igamma;
            double dy = height / (dx * dx + 1);
            *yit++ = dy + y0;
        }
    }
    static std::tuple<double, double> result(unsigned int p, const double*params, const double*errors) {
        switch(p) {
        default:
        case 0: return {params[0], errors[0]};
        case 1: {
            double fwhm = 2/params[1];
            return {fwhm, fwhm * errors[1] / params[1]}; //FWHM
        }
        case 2: return {params[2], errors[2]};
        case 3: return {params[3], errors[3]};
        }
    }
    static std::valarray<double> initParams(cv_iterator xbegin, cv_iterator xend, cv_iterator ybegin, cv_iterator yend){
        if(std::distance(xbegin, xend) < 4)
            return {};
        double xmax = *std::max_element(xbegin, xend);
        double xmin = *std::min_element(xbegin, xend);
        double ymax = *std::max_element(ybegin, yend);
        double ymin = *std::min_element(ybegin, yend);
        return {(xmax - xmin) * randMT19937() + xmin,
            5.0 / (xmax - xmin) * randMT19937(),
            (ymax - ymin) * (2 * randMT19937() - 1),
            (ymax - ymin) * (2 * randMT19937() - 1),
        };
    }
};

using XGraph1DMathLorenzianPositionTool = XGraph1DMathFitToolX<FuncGraph1DMathLorenzianFitTool, 0>;
using XGraph1DMathLorenzianFWHMTool = XGraph1DMathFitToolX<FuncGraph1DMathLorenzianFitTool, 1>;
using XGraph1DMathLorenzianHeightTool = XGraph1DMathFitToolX<FuncGraph1DMathLorenzianFitTool, 2>;
using XGraph1DMathLorenzianBaselineTool = XGraph1DMathFitToolX<FuncGraph1DMathLorenzianFitTool, 3>;

#endif
