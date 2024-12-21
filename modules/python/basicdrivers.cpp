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

#ifdef USE_PYBIND11

#include "primarydriverwiththread.h"
#include "charinterface.h"
#include "chardevicedriver.h"
PyDriverExporter<XCharDeviceDriver<XPrimaryDriverWithThread>, XPrimaryDriverWithThread>
    chardriver([](auto node, auto payload){
});
PyDriverExporter<XPythonDriver<XCharDeviceDriver<XPrimaryDriverWithThread>>, XCharDeviceDriver<XPrimaryDriverWithThread>>
    pychardriverbase([](auto node, auto payload){
});
PyDriverExporterWithTrampoline<XPythonCharDeviceDriverWithThread<>,
    XPythonDriver<XCharDeviceDriver<XPrimaryDriverWithThread>>,
    XPythonCharDeviceDriverWithThreadHelper<>>
    pychardriver("XPythonCharDeviceDriverWithThread", [](auto node, auto payload){
    node
    .def("finishWritingRaw", &XPythonCharDeviceDriverWithThread<>::finishWritingRaw);
});

#include "dmm.h"
PyDriverExporter<XDMM, XPrimaryDriver> dmm([](auto node, auto payload){
    payload.def("value", [](shared_ptr<XDMM::Payload> &self, unsigned int i){return self->value(i);});
});

#include "dcsource.h"
PyDriverExporter<XDCSource, XPrimaryDriver> dcsource([](auto node, auto payload){
    node.def("changeFunction", &XDCSource::changeFunction)
        .def("changeOutput", &XDCSource::changeOutput)
        .def("changeValue", &XDCSource::changeValue)
        .def("changeRange", &XDCSource::changeRange)
        .def("queryStatus", &XDCSource::queryStatus)
        .def("max", &XDCSource::max);
});

#include "motor.h"
PyDriverExporter<XMotorDriver, XPrimaryDriver> motordriver([](auto node, auto payload){
    node.def("runSequentially", &XMotorDriver::runSequentially);
});

#include "signalgenerator.h"
PyDriverExporter<XSG, XPrimaryDriver> sg([](auto node, auto payload){
    payload.def("freq", &XSG::Payload::freq);
});


#endif //USE_PYBIND11
