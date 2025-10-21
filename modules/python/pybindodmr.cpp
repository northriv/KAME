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
#include "pythondriver.h"

#ifdef USE_PYBIND11

#include "primarydriverwiththread.h"
#include "charinterface.h"
#include "chardevicedriver.h"

#include "lasermodule.h"
PyDriverExporter<XLaserModule, XPrimaryDriver> lasermodule([](auto node, auto payload){
    payload.def("temperature", &XLaserModule::Payload::temperature)
        .def("current", &XLaserModule::Payload::current)
        .def("power", &XLaserModule::Payload::power)
        .def("voltage", &XLaserModule::Payload::voltage);
});


#include "digitalcamera.h"
PyDriverExporter<XDigitalCamera, XPrimaryDriver> digitalcamera([](auto node, auto payload){
    payload.def("blackLvlOffset", &XDigitalCamera::Payload::blackLvlOffset)
        .def("exposureTime", &XDigitalCamera::Payload::exposureTime)
        .def("electricDark", &XDigitalCamera::Payload::electricDark)
        .def("width", &XDigitalCamera::Payload::width)
        .def("height", &XDigitalCamera::Payload::height)
        .def("stride", &XDigitalCamera::Payload::stride)
        .def("firstPixel", &XDigitalCamera::Payload::firstPixel)
        .def("rawCounts", [](XDigitalCamera::Payload &self){
            using namespace Eigen;
            using RMatrixXu32 = Matrix<uint32_t, Dynamic, Dynamic, RowMajor>;
            auto cmatrix = Map<const RMatrixXu32, 0, Stride<Dynamic, 1>>(
                &self.rawCounts()->at(self.firstPixel()),
                self.height(), self.width(),
                Stride<Dynamic, 1>(self.stride(), 1));
            return Ref<const RMatrixXu32>(cmatrix);
        })
        .def("darkCounts", [](XDigitalCamera::Payload &self){
            using namespace Eigen;
            using RMatrixXu32 = Matrix<uint32_t, Dynamic, Dynamic, RowMajor>;
            auto cmatrix = Map<const RMatrixXu32, 0, Stride<Dynamic, 1>>(
                &self.darkCounts()->at(self.firstPixel()),
                self.height(), self.width(),
                Stride<Dynamic, 1>(self.stride(), 1));
            return Ref<const RMatrixXu32>(cmatrix);
        });
});

#include "odmrfspectrum.h"
PyDriverExporter<XODMRFSpectrum, XSecondaryDriver> odmrfspectrum([](auto node, auto payload){
    node.def("clear", &XODMRFSpectrum::clear);
    payload.def("wave", &XODMRFSpectrum::Payload::wave)
        .def("weights", &XODMRFSpectrum::Payload::weights)
        .def("numChannels", &XODMRFSpectrum::Payload::numChannels)
        .def("res", &XODMRFSpectrum::Payload::res)
        .def("min", &XODMRFSpectrum::Payload::min);
});

#include "odmrimaging.h"
PyDriverExporter<XODMRImaging, XSecondaryDriver> odmrimaging([](auto node, auto payload){
    payload.def("sampleIntensities", &XODMRImaging::Payload::sampleIntensities)
    .def("sampleIntensitiesCorrected", &XODMRImaging::Payload::sampleIntensitiesCorrected)
    .def("referenceIntensities", &XODMRImaging::Payload::referenceIntensities)
    .def("plRaw", &XODMRImaging::Payload::plRaw)
    .def("plCorr", &XODMRImaging::Payload::plCorr)
    .def("pl0", &XODMRImaging::Payload::pl0)
    .def("dPL", &XODMRImaging::Payload::dPL)
    .def("numSamples", &XODMRImaging::Payload::numSamples)
    .def("gainForDisp", &XODMRImaging::Payload::gainForDisp)
    .def("width", &XODMRImaging::Payload::width)
    .def("height", &XODMRImaging::Payload::height)
    .def("sequenceLength", &XODMRImaging::Payload::sequenceLength)
    .def("sequence", &XODMRImaging::Payload::sequence);
});

#include "filterwheel.h"
PyDriverExporter<XFilterWheel, XSecondaryDriver> filterwheel([](auto node, auto payload){
    payload.def("dwellIndex", &XFilterWheel::Payload::dwellIndex)
        .def("wheelIndexOfFrame", &XFilterWheel::Payload::wheelIndexOfFrame);
});

#include "imageprocessor.h"
PyDriverExporter<XImageProcessor, XSecondaryDriver> imageprocessor([](auto node, auto payload){
    payload.def("intensities", [](XImageProcessor::Payload &self, unsigned int ch){
            using namespace Eigen;
            using RMatrixXd = Matrix<double, Dynamic, Dynamic, RowMajor>;
            auto cmatrix = Map<const RMatrixXd, 0, Stride<Dynamic, 1>>(
                &self.intensities(ch).at(0),
                self.height(), self.width(),
                Stride<Dynamic, 1>(self.width(), 1));
            return Ref<const RMatrixXd>(cmatrix);
        })
        .def("raw", &XImageProcessor::Payload::raw)
        .def("numSamples", &XImageProcessor::Payload::numSamples)
        .def("gainForDisp", &XImageProcessor::Payload::gainForDisp)
        .def("width", &XImageProcessor::Payload::width)
        .def("height", &XImageProcessor::Payload::height);
});

#include "opticalspectrometer.h"
PyDriverExporter<XOpticalSpectrometer, XPrimaryDriver> opticalspectrometer([](auto node, auto payload){
    payload.def("integrationTime", &XOpticalSpectrometer::Payload::integrationTime)
        .def("counts", [](XOpticalSpectrometer::Payload &self){
            if( !self.isCountsValid())
                throw std::length_error("No valid counts");
            using namespace Eigen;
            auto cvector = Map<const VectorXd, 0>(
                &self.counts()[0], self.accumLength());
            return Ref<const VectorXd>(cvector);
        })
        .def("isCountsValid", &XOpticalSpectrometer::Payload::isCountsValid)
        .def("darkCounts", [](XOpticalSpectrometer::Payload &self){
            if( !self.isDarkValid())
                throw std::length_error("No valid counts");
            using namespace Eigen;
            auto cvector = Map<const VectorXd, 0>(
                &self.darkCounts()[0], self.accumLength());
            return Ref<const VectorXd>(cvector);
        })
        .def("isDarkValid", &XOpticalSpectrometer::Payload::isDarkValid)
        .def("accumCounts", [](XOpticalSpectrometer::Payload &self){
            using namespace Eigen;
            auto cvector = Map<const VectorXd, 0>(
                &self.accumCounts()[0], self.accumLength());
            return Ref<const VectorXd>(cvector);
        })
        .def("accumLength", &XOpticalSpectrometer::Payload::accumLength)
        .def("waveLengths", [](XOpticalSpectrometer::Payload &self){
            using namespace Eigen;
            auto cvector = Map<const VectorXd, 0>(
                &self.waveLengths()[0], self.accumLength());
            return Ref<const VectorXd>(cvector);
        });
});

#endif //USE_PYBIND11
