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

PyXNodeExporter<XCustomCharInterface, XInterface>
    customcharinterface([](auto node){
    node
    .def("query", [](shared_ptr<XCustomCharInterface> &self, const char *str){
        pybind11::gil_scoped_release unguard;
        self->query(str);
    })
    .def("send", [](shared_ptr<XCustomCharInterface> &self, const char *str){
        pybind11::gil_scoped_release unguard;
        self->send(str);
    })
    .def("receive", [](shared_ptr<XCustomCharInterface> &self){
        pybind11::gil_scoped_release unguard;
        self->receive();
    })
    .def("buffer", &XCustomCharInterface::buffer)
    .def("toDouble", &XCustomCharInterface::toDouble)
    .def("toInt", &XCustomCharInterface::toInt)
    .def("toUInt", &XCustomCharInterface::toUInt)
    .def("toStr", &XCustomCharInterface::toStr)
    .def("toStrSimplified", &XCustomCharInterface::toStrSimplified)
    .def("eos", &XCustomCharInterface::eos)
    .def("setEOS", &XCustomCharInterface::setEOS);
});
PyXNodeExporter<XCharInterface, XCustomCharInterface>
    charinterface([](auto node){
    node
    .def("write", [](shared_ptr<XCharInterface> &self, const char *sendbuf, int size){
        pybind11::gil_scoped_release unguard;
        self->write(sendbuf, size);
    })
    .def("receive", [](shared_ptr<XCharInterface> &self, unsigned int length){
        pybind11::gil_scoped_release unguard;
        self->receive(length);
    })
    .def("setGPIBUseSerialPollOnWrite", &XCharInterface::setGPIBUseSerialPollOnWrite)
    .def("setGPIBUseSerialPollOnRead", &XCharInterface::setGPIBUseSerialPollOnRead)
    .def("setGPIBWaitBeforeWrite", &XCharInterface::setGPIBWaitBeforeWrite)
    .def("setGPIBWaitBeforeRead", &XCharInterface::setGPIBWaitBeforeRead)
    .def("setGPIBWaitBeforeSPoll", &XCharInterface::setGPIBWaitBeforeSPoll)
    .def("setGPIBMAVbit", &XCharInterface::setGPIBMAVbit)
    .def("setSerialBaudRate", &XCharInterface::setSerialBaudRate)
    .def("setSerialStopBits", &XCharInterface::setSerialStopBits)
    .def("setSerialParity", &XCharInterface::setSerialParity)
    .def("setSerial7Bits", &XCharInterface::setSerial7Bits)
    .def("setSerialFlushBeforeWrite", &XCharInterface::setSerialFlushBeforeWrite)
    .def("setSerialEOS", &XCharInterface::setSerialEOS)
    .def("setSerialHasEchoBack", &XCharInterface::setSerialHasEchoBack);
});


PyDriverExporter<XCharDeviceDriver<XPrimaryDriverWithThread>, XPrimaryDriverWithThread>
    chardriver;
PyDriverExporter<XPythonDriver<XCharDeviceDriver<XPrimaryDriverWithThread>>, XCharDeviceDriver<XPrimaryDriverWithThread>>
    pychardriverbase;
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
        PYBIND11_OVERRIDE(
            std::deque<double>, tBaseDriver, oneShotMultiRead);
    }
    //! called when m_function is changed
    virtual void changeFunction() override {
        PYBIND11_OVERRIDE_PURE(
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
        .def("oneShotRead", [](shared_ptr<XPythonDMM> &self){
            pybind11::gil_scoped_release unguard;
            self->oneShotRead();
        })
        .def("oneShotMultiRead", [](shared_ptr<XPythonDMM> &self){
            pybind11::gil_scoped_release unguard;
            self->oneShotMultiRead();
        })
        .def("changeFunction", [](shared_ptr<XPythonDMM> &self){
            pybind11::gil_scoped_release unguard;
            self->changeFunction();
        });
});

#include "dcsource.h"
PyDriverExporter<XDCSource, XPrimaryDriver> dcsource([](auto node, auto payload){
    node.def("changeFunction", [](shared_ptr<XDCSource> &self, int ch, int x){
            pybind11::gil_scoped_release unguard;
            self->changeFunction(ch, x);
        })
        .def("changeOutput", [](shared_ptr<XDCSource> &self, int ch, bool x){
            pybind11::gil_scoped_release unguard;
            self->changeOutput(ch, x);
        })
        .def("changeValue", [](shared_ptr<XDCSource> &self, int ch, double x, bool autorange){
            pybind11::gil_scoped_release unguard;
            self->changeValue(ch, x, autorange);
        })
        .def("changeRange", [](shared_ptr<XDCSource> &self, int ch, int x){
            pybind11::gil_scoped_release unguard;
            self->changeRange(ch, x);
        })
        .def("queryStatus", [](shared_ptr<XDCSource> &self, Transaction &tr, int ch){
            pybind11::gil_scoped_release unguard;
            self->queryStatus(tr, ch);
        })
        .def("max", [](shared_ptr<XDCSource> &self, int ch, bool autorange){
            pybind11::gil_scoped_release unguard;
            self->max(ch, autorange);
        });
});

#include "motor.h"
PyDriverExporter<XMotorDriver, XPrimaryDriver> motordriver([](auto node, auto payload){
    //TODO         pybind11::gil_scoped_release unguard;
    node.def("runSequentially", &XMotorDriver::runSequentially);
});

#include "signalgenerator.h"
PyDriverExporter<XSG, XPrimaryDriver> sg([](auto node, auto payload){
    payload.def("freq", &XSG::Payload::freq);
});

#include "dso.h"
PyDriverExporter<XDSO, XPrimaryDriver> dso([](auto node, auto payload){
    payload.def("trigPos", &XDSO::Payload::trigPos)
        .def("numChannels", &XDSO::Payload::numChannels)
        .def("timeInterval", &XDSO::Payload::timeInterval)
    .def("length", &XDSO::Payload::length)
    .def("wave", [](XDSO::Payload &self, unsigned int ch){
        using namespace Eigen;
        auto cvector = Map<const VectorXd, 0>(
            self.wave(ch), self.length());
        return Ref<const VectorXd>(cvector);
    })
    .def("setParameters", &XDSO::Payload::setParameters)
    .def("lengthDisp", &XDSO::Payload::lengthDisp)
    .def("waveDisp", [](XDSO::Payload &self, unsigned int ch){
        using namespace Eigen;
        auto cvector = Map<const VectorXd, 0>(
            self.waveDisp(ch), self.lengthDisp());
        return Ref<const VectorXd>(cvector);
    })
    .def("trigPosDisp", &XDSO::Payload::trigPosDisp)
    .def("numChannelsDisp", &XDSO::Payload::numChannelsDisp)
    .def("timeIntervalDisp", &XDSO::Payload::timeIntervalDisp)
    .def("shotDescription", &XDSO::Payload::shortDescription);
});


#endif //USE_PYBIND11
