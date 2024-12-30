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

#include "pulserdriver.h"
#include "nmrpulse.h"

PyDriverExporter<XNMRPulseAnalyzer, XSecondaryDriver> nmrpulse("XNMRPulseAnalyzer", [](auto node, auto payload){
    payload.def("wave", [](XNMRPulseAnalyzer::Payload &self){
        using namespace Eigen;
        auto cvector = Map<const VectorXcd, 0>(
            &self.wave()[0], self.waveWidth());
        return Ref<const VectorXcd>(cvector);
    })
    .def("darkPSD", [](XNMRPulseAnalyzer::Payload &self){
        using namespace Eigen;
        auto cvector = Map<const VectorXd, 0>(
            &self.darkPSD()[0], self.ftWidth());
        return Ref<const VectorXd>(cvector);
    })
    .def("darkPSDFactorToVoltSq", [](XNMRPulseAnalyzer::Payload &self){return self.darkPSDFactorToVoltSq();})
    .def("echoesT2", [](XNMRPulseAnalyzer::Payload &self){return self.echoesT2();})
    .def("dFreq", [](XNMRPulseAnalyzer::Payload &self){return self.dFreq();})
    .def("interval", [](XNMRPulseAnalyzer::Payload &self){return self.interval();})
    .def("startTime", [](XNMRPulseAnalyzer::Payload &self){return self.startTime();})
    .def("waveWidth", [](XNMRPulseAnalyzer::Payload &self){return self.waveWidth();})
    .def("waveFTPos", [](XNMRPulseAnalyzer::Payload &self){return self.waveFTPos();})
    .def("ftWave", [](XNMRPulseAnalyzer::Payload &self){
        using namespace Eigen;
        auto cvector = Map<const VectorXcd, 0>(
            &self.ftWave()[0], self.ftWidth());
        return Ref<const VectorXcd>(cvector);
    })
    .def("ftWidth", [](XNMRPulseAnalyzer::Payload &self){return self.ftWidth();});
});

#include "nmrspectrum.h"
PyDriverExporter<XNMRSpectrum, XSecondaryDriver> nmrspectrum([](auto node, auto payload){
    payload.def("wave", &XNMRSpectrum::Payload::wave)
        .def("weights", &XNMRSpectrum::Payload::weights)
        .def("darkPSD", &XNMRSpectrum::Payload::darkPSD)
        .def("res", &XNMRSpectrum::Payload::res)
        .def("min", &XNMRSpectrum::Payload::min);
});

#include "nmrfspectrum.h"
PyDriverExporter<XNMRFSpectrum, XSecondaryDriver> nmrfspectrum([](auto node, auto payload){
    payload.def("wave", &XNMRFSpectrum::Payload::wave)
        .def("weights", &XNMRFSpectrum::Payload::weights)
        .def("darkPSD", &XNMRFSpectrum::Payload::darkPSD)
        .def("res", &XNMRFSpectrum::Payload::res)
        .def("min", &XNMRFSpectrum::Payload::min);
});

#include "nmrrelax.h"
PyDriverExporter<XNMRT1, XSecondaryDriver> nmrt1([](auto node, auto payload){
});

#include "autolctuner.h"
PyDriverExporter<XAutoLCTuner, XSecondaryDriver> autolctuner([](auto node, auto payload){
});

#endif //USE_PYBIND11
