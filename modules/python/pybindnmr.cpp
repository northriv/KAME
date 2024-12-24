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
    payload.def("wave", [](shared_ptr<XNMRPulseAnalyzer::Payload> &self){return self->wave();})
    .def("darkPSD", [](shared_ptr<XNMRPulseAnalyzer::Payload> &self){return self->darkPSD();})
    .def("darkPSDFactorToVoltSq", [](shared_ptr<XNMRPulseAnalyzer::Payload> &self){return self->darkPSDFactorToVoltSq();})
    .def("echoesT2", [](shared_ptr<XNMRPulseAnalyzer::Payload> &self){return self->echoesT2();})
    .def("dFreq", [](shared_ptr<XNMRPulseAnalyzer::Payload> &self){return self->dFreq();})
    .def("interval", [](shared_ptr<XNMRPulseAnalyzer::Payload> &self){return self->interval();})
    .def("startTime", [](shared_ptr<XNMRPulseAnalyzer::Payload> &self){return self->startTime();})
    .def("waveWidth", [](shared_ptr<XNMRPulseAnalyzer::Payload> &self){return self->waveWidth();})
    .def("waveFTPos", [](shared_ptr<XNMRPulseAnalyzer::Payload> &self){return self->waveFTPos();})
    .def("ftWave", [](shared_ptr<XNMRPulseAnalyzer::Payload> &self){return self->ftWave();})
    .def("ftWidth", [](shared_ptr<XNMRPulseAnalyzer::Payload> &self){return self->ftWidth();});
});

#endif //USE_PYBIND11
