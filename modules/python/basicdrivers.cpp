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
    .def("finishWritingRaw", [](shared_ptr<XPythonCharDeviceDriverWithThread<>> &self,
         const shared_ptr<const XPrimaryDriver::RawData> &rawdata,
         const XTime &time_awared, const XTime &time_recorded) {
        pybind11::gil_scoped_release unguard; //maybe time-consuming.
        self->finishWritingRaw(rawdata, time_awared, time_recorded);
    });
});

#include "dmm.h"
struct XPythonDMM : public XPythonDriver<XCharDeviceDriver<XDMM>> {
    using tBaseDriver = XPythonDriver<XCharDeviceDriver<XDMM>>;
    using tBaseDriver::tBaseDriver; //inherits constructors.

    //open to public, previously protected ones.
    //! one-shot reading
    virtual double oneShotRead() override = 0;
    //! is called when m_function is changed
    virtual void changeFunction() override = 0;
    //! one-shot multi-channel reading
    virtual std::deque<double> oneShotMultiRead() override {return {};}

    struct Payload : public tBaseDriver::Payload {};
};

struct XPythonDMMHelper : public XPythonDMM {
    using tBaseDriver = XPythonDMM;
    using tBaseDriver::tBaseDriver; //inherits constructors.

    //! one-shot reading
    virtual double oneShotRead() override {
        PYBIND11_OVERRIDE_PURE(
            double, tBaseDriver, oneShotRead);
    }
    //! one-shot multi-channel reading
    virtual std::deque<double> oneShotMultiRead() override {
        PYBIND11_OVERRIDE_PURE(
            std::deque<double>, tBaseDriver, oneShotMultiRead);
    }
    //! called when m_function is changed
    virtual void changeFunction() override {
        PYBIND11_OVERRIDE(
            void, tBaseDriver, changeFunction);
    }
    struct Payload : public tBaseDriver::Payload {};
};
PyDriverExporter<XDMM, XPrimaryDriver> dmm("XDMM", [](auto node, auto payload){
    payload.def("value", [](shared_ptr<XDMM::Payload> &self, unsigned int i){return self->value(i);});
});
PyDriverExporter<XCharDeviceDriver<XDMM>, XDMM> chardmm([](auto node, auto payload){
});
PyDriverExporter<XPythonDriver<XCharDeviceDriver<XDMM>>, XCharDeviceDriver<XDMM>> pydmmbase([](auto node, auto payload){
});
PyDriverExporterWithTrampoline<XPythonDMM, XDMM, XPythonDMMHelper> pydmm("XPythonDMM", [](auto node, auto payload){
    node
        .def("oneShotRead", &XPythonDMM::oneShotRead)
        .def("oneShotMultiRead", &XPythonDMM::oneShotMultiRead)
        .def("changeFunction", &XPythonDMM::changeFunction);
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
