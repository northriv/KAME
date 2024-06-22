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
#include "spectralmathtool.h"

//---------------------------------------------------------------------------
DECLARE_TYPE_HOLDER(XSpectral1DMathToolList)

REGISTER_TYPE(XSpectral1DMathToolList, Spectral1DMathToolRubyScalePiermariniRT, "RubyScalePiermariniRT");
REGISTER_TYPE(XSpectral1DMathToolList, Spectral1DMathToolRubyScaleMaoRT, "RubyScaleMaoRT");
REGISTER_TYPE(XSpectral1DMathToolList, Spectral1DMathToolRubyScaleMaoArRT, "RubyScaleMaoArRT");
REGISTER_TYPE(XSpectral1DMathToolList, Spectral1DMathToolRubyScaleMaoHeRT, "RubyScaleMaoHeRT");
//REGISTER_TYPE(XGraph1DMathToolList, Graph1DMathToolSum, "Sum");
//REGISTER_TYPE(XGraph1DMathToolList, Graph1DMathToolAverage, "Average");
//REGISTER_TYPE(XGraph1DMathToolList, Graph1DMathToolCoG, "CoG");
//REGISTER_TYPE(XGraph1DMathToolList, Graph1DMathToolMaxValue, "MaxValue");
//REGISTER_TYPE(XGraph1DMathToolList, Graph1DMathToolMinValue, "MinValue");
//REGISTER_TYPE(XGraph1DMathToolList, Graph1DMathToolMaxPosition, "MaxPosition");
//REGISTER_TYPE(XGraph1DMathToolList, Graph1DMathToolMinPosition, "MinPosition");

//REGISTER_TYPE(XGraph2DMathToolList, Graph2DMathToolSum, "Sum");
//REGISTER_TYPE(XGraph2DMathToolList, Graph2DMathToolAverage, "Average");


XSpectral1DMathToolList::XSpectral1DMathToolList(const char *name, bool runtime,
                         const shared_ptr<XMeasure> &meas, const shared_ptr<XDriver> &driver, const shared_ptr<XPlot> &plot) :
    XGraph1DMathToolList(name, runtime, meas, driver, plot) {
}


