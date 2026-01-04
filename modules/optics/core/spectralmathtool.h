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

#ifndef spectralmathtoolH
#define spectralmathtoolH
//---------------------------------------------------------------------------

#include "graphmathtool.h"
#include "graphmathfittool.h"

template <class Func, unsigned int lambda0_pm = 694210u> //290K
struct FuncSpectral1DMathToolRubyScalePiermarini : public Func {
    static double lambdaToGPa(double lambda) {
        return (lambda - lambda0_pm / 1000.0) / 0.365; //GPa
    }
    using cv_iterator = std::vector<XGraph::VFloat>::const_iterator;
    double operator()(cv_iterator xbegin, cv_iterator xend, cv_iterator ybegin, cv_iterator yend){
        double lambda = Func{}(xbegin, xend, ybegin, yend); //nm
        return lambdaToGPa(lambda);
    }
    //for fittool, scaling params.
    static std::tuple<double, double> result(unsigned int p, const double*params, const double*errors) {
        return {lambdaToGPa(params[0]), lambdaToGPa(params[1] + params[0]) - lambdaToGPa(params[0])};
    }
};
using XSpectral1DMathToolRubyScalePiermariniRT = XGraph1DMathToolX<FuncSpectral1DMathToolRubyScalePiermarini<FuncGraph1DMathToolMaxPosition>>;
using XSpectral1DMathToolRubyScalePiermarini77K = XGraph1DMathToolX<FuncSpectral1DMathToolRubyScalePiermarini<FuncGraph1DMathToolMaxPosition, 693420u>>;
using XSpectral1DMathToolRubyScalePiermariniGaussianRT = XGraph1DMathFitToolX<FuncSpectral1DMathToolRubyScalePiermarini<FuncGraph1DMathGaussianFitTool>, 0>;
using XSpectral1DMathToolRubyScalePiermariniGaussian77K = XGraph1DMathFitToolX<FuncSpectral1DMathToolRubyScalePiermarini<FuncGraph1DMathGaussianFitTool, 693420u>, 0>;


template <class Func, unsigned int mao_exponent_1000 = 5000u, unsigned int lambda0_pm = 694210u> //290K
struct FuncSpectral1DMathToolRubyScaleMao : public Func {
    static double lambdaToGPa(double lambda) {
        constexpr double mao_exponent = mao_exponent_1000 / 1000.0; //5 for std., 7.665 for Ar, 7.715 for He.
        return 1904 * (std::pow(lambda / (lambda0_pm / 1000.0), mao_exponent) - 1) / mao_exponent; //GPa
    }
    using cv_iterator = std::vector<XGraph::VFloat>::const_iterator;
    double operator()(cv_iterator xbegin, cv_iterator xend, cv_iterator ybegin, cv_iterator yend){
        double lambda = Func{}(xbegin, xend, ybegin, yend);//nm
        return lambdaToGPa(lambda);
    }
    //for fittool, scaling params.
    static std::tuple<double, double> result(unsigned int p, const double*params, const double*errors) {
        return {lambdaToGPa(params[0]), lambdaToGPa(params[1] + params[0]) - lambdaToGPa(params[0])};
    }
};
using XSpectral1DMathToolRubyScaleMaoRT = XGraph1DMathToolX<FuncSpectral1DMathToolRubyScaleMao<FuncGraph1DMathToolMaxPosition>>;
using XSpectral1DMathToolRubyScaleMaoArRT = XGraph1DMathToolX<FuncSpectral1DMathToolRubyScaleMao<FuncGraph1DMathToolMaxPosition, 7665u>>;
using XSpectral1DMathToolRubyScaleMaoHeRT = XGraph1DMathToolX<FuncSpectral1DMathToolRubyScaleMao<FuncGraph1DMathToolMaxPosition, 7715u>>;
using XSpectral1DMathToolRubyScaleMao77K = XGraph1DMathToolX<FuncSpectral1DMathToolRubyScaleMao<FuncGraph1DMathToolMaxPosition, 693420u>>;
using XSpectral1DMathToolRubyScaleMaoAr77K = XGraph1DMathToolX<FuncSpectral1DMathToolRubyScaleMao<FuncGraph1DMathToolMaxPosition, 7665u,693420u>>;
using XSpectral1DMathToolRubyScaleMaoHe77K = XGraph1DMathToolX<FuncSpectral1DMathToolRubyScaleMao<FuncGraph1DMathToolMaxPosition, 7715u,693420u>>;
using XSpectral1DMathToolRubyScaleMaoGaussianRT = XGraph1DMathFitToolX<FuncSpectral1DMathToolRubyScaleMao<FuncGraph1DMathGaussianFitTool>, 0>;
using XSpectral1DMathToolRubyScaleMaoArGaussianRT = XGraph1DMathFitToolX<FuncSpectral1DMathToolRubyScaleMao<FuncGraph1DMathGaussianFitTool, 7665u>, 0>;
using XSpectral1DMathToolRubyScaleMaoHeGaussianRT = XGraph1DMathFitToolX<FuncSpectral1DMathToolRubyScaleMao<FuncGraph1DMathGaussianFitTool, 7715u>, 0>;
using XSpectral1DMathToolRubyScaleMaoGaussian77K = XGraph1DMathFitToolX<FuncSpectral1DMathToolRubyScaleMao<FuncGraph1DMathGaussianFitTool, 693420u>, 0>;
using XSpectral1DMathToolRubyScaleMaoArGaussian77K = XGraph1DMathFitToolX<FuncSpectral1DMathToolRubyScaleMao<FuncGraph1DMathGaussianFitTool, 7665u,693420u>, 0>;
using XSpectral1DMathToolRubyScaleMaoHeGaussian77K = XGraph1DMathFitToolX<FuncSpectral1DMathToolRubyScaleMao<FuncGraph1DMathGaussianFitTool, 7715u,693420u>, 0>;

class XSpectral1DMathToolList : public XGraph1DMathToolList {
public:
    using XGraph1DMathToolList::XGraph1DMathToolList;

    DEFINE_TYPE_HOLDER(
        std::reference_wrapper<Transaction>, const shared_ptr<XScalarEntryList> &,
        const shared_ptr<XDriver> &, const shared_ptr<XPlot> &, const shared_ptr<XNode> &, const std::vector<std::string> &
        )
private:
};


#endif
