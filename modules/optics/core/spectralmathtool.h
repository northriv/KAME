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

#ifndef spectralmathtoolH
#define spectralmathtoolH
//---------------------------------------------------------------------------

#include "graphmathtool.h"

struct FuncSpectral1DMathToolRubyScalePiermariniRT {
    using cv_iterator = std::vector<XGraph::VFloat>::const_iterator;
    double operator()(cv_iterator xbegin, cv_iterator xend, cv_iterator ybegin, cv_iterator yend){
        double lambda = FuncGraph1DMathToolMinPosition{}(xbegin, xend, ybegin, yend);
        constexpr double lambda0 = 694.35; //nm
        return (lambda - lambda0) / 0.365; //GPa
    }
};
using XSpectral1DMathToolRubyScalePiermariniRT = XGraph1DMathToolX<FuncSpectral1DMathToolRubyScalePiermariniRT>;


struct FuncSpectral1DMathToolRubyScaleMaoRT {
    using cv_iterator = std::vector<XGraph::VFloat>::const_iterator;
    double operator()(cv_iterator xbegin, cv_iterator xend, cv_iterator ybegin, cv_iterator yend){
        double lambda = FuncGraph1DMathToolMinPosition{}(xbegin, xend, ybegin, yend);
        constexpr double lambda0 = 694.35; //nm
        return 1904 * (std::pow(lambda / lambda0, 5) - 1) / 5; //GPa
    }
};
using XSpectral1DMathToolRubyScaleMaoRT = XGraph1DMathToolX<FuncSpectral1DMathToolRubyScaleMaoRT>;

class XSpectral1DMathToolList : public XGraph1DMathToolList {
public:
    XSpectral1DMathToolList(const char *name, bool runtime,
        const shared_ptr<XMeasure> &meas, const shared_ptr<XDriver> &driver,
        const shared_ptr<XPlot> &plot);
    virtual ~XSpectral1DMathToolList() {}

    DEFINE_TYPE_HOLDER(
        std::reference_wrapper<Transaction>, const shared_ptr<XScalarEntryList> &,
        const shared_ptr<XDriver> &, const shared_ptr<XPlot> &
        )
private:
};


#endif
