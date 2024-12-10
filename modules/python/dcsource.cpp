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
#include "pythondriver.h"

#include "dcsource.h"

#ifdef USE_PYBIND11
PyDriverExporter<XDCSource, XPrimaryDriver> dcsource([](auto node, auto payload){
    node.def("changeFunction", &XDCSource::changeFunction);
    node.def("changeOutput", &XDCSource::changeOutput);
    node.def("changeValue", &XDCSource::changeValue);
    node.def("changeRange", &XDCSource::changeRange);
    node.def("queryStatus", &XDCSource::queryStatus);
    node.def("max", &XDCSource::max);
});
#endif
