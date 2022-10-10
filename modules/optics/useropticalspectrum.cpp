/***************************************************************************
        Copyright (C) 2002-2018 Kentaro Kitagawa
		                   kitagawa@phys.s.u-tokyo.ac.jp
		
		This program is free software; you can redistribute it and/or
		modify it under the terms of the GNU Library General Public
		License as published by the Free Software Foundation; either
		version 2 of the License, or (at your option) any later version.
		
		You should have received a copy of the GNU Library General 
		Public License and a list of authors along with this program; 
		see the files COPYING and AUTHORS.
***************************************************************************/
#include "useropticalspectrum.h"
#include "charinterface.h"
#include "analyzer.h"

REGISTER_TYPE(XDriverList, OceanOpticsSpectrometer, "OceanOptics/Seabreeze USB/HR2000+ spectrometer");

//---------------------------------------------------------------------------
XOceanOpticsSpectrometer::XOceanOpticsSpectrometer(const char *name, bool runtime,
	Transaction &tr_meas, const shared_ptr<XMeasure> &meas) :
    XCharDeviceDriver<XOpticalSpectrometer, XOceanOpticsUSBInterface>(name, runtime, ref(tr_meas), meas) {
//    startWavelen()->disable();
//    stopWavelen()->disable();
}

void
XOceanOpticsSpectrometer::open() {
    interface()->initDevice();

    auto config = interface()->readConfigurations();
    gMessagePrint(formatString("S/N:%s; %s; %s", config.serialNo.c_str(), config.opticalBenchConfig.c_str(), config.spectrometerConfig.c_str()));
    int nlpoly;
    m_wavelenCalibCoeffs.resize(4);
    for(unsigned int i = 0; i < 4; ++i)
        if(sscanf(config.wavelenCalib[i].c_str(), "%lf", &m_wavelenCalibCoeffs[i]) != 1)
            throw XInterface::XConvError(__FILE__, __LINE__);
    m_strayLightCoeffs.resize(1);
    if(sscanf(config.strayLightConst.c_str(), "%lf", &m_strayLightCoeffs[0]) != 1)
        throw XInterface::XConvError(__FILE__, __LINE__);

    if(sscanf(config.nlpoly.c_str(), "%d", &nlpoly) != 1)
        throw XInterface::XConvError(__FILE__, __LINE__);
    nlpoly++;
    m_nonlinCorrCoeffs.resize(nlpoly);
    for(unsigned int i = 0; i < nlpoly; ++i) {
        if(sscanf(config.nonlinCorr[i].c_str(), "%lf", &m_nonlinCorrCoeffs[i]) != 1)
            throw XInterface::XConvError(__FILE__, __LINE__);
    }

	start();
}
void
XOceanOpticsSpectrometer::onAverageChanged(const Snapshot &shot, XValueNodeBase *) {
	unsigned int avg = shot[ *average()];
}
void
XOceanOpticsSpectrometer::onIntegrationTimeChanged(const Snapshot &shot, XValueNodeBase *) {
    interface()->setIntegrationTime(lrint(shot[ *integrationTime()] * 1e6));
}

void
XOceanOpticsSpectrometer::acquireSpectrum(shared_ptr<RawData> &writer) {
    XScopedLock<XOceanOpticsUSBInterface> lock( *interface());

    auto status = interface()->readInstrumStatus();
    writer->push((uint8_t)status.size());
    writer->insert(writer->end(), status.begin(), status.end());
    writer->push((uint8_t)m_wavelenCalibCoeffs.size());
    for(double x:  m_wavelenCalibCoeffs)
        writer->push(x);
    writer->push((uint8_t)m_strayLightCoeffs.size());
    for(double x:  m_strayLightCoeffs)
        writer->push(x);
    writer->push((uint8_t)m_nonlinCorrCoeffs.size());
    for(double x:  m_nonlinCorrCoeffs)
        writer->push(x);

    int len = interface()->readSpectrum(m_spectrumBuffer);
    if( !len)
        throw XSkippedRecordError(__FILE__, __LINE__);
    writer->push((uint32_t)len); //be actual pixels + 1(end delimiter 0x69).
    writer->insert(writer->end(),
                     m_spectrumBuffer.begin(), m_spectrumBuffer.begin() + len);
}
void
XOceanOpticsSpectrometer::convertRawAndAccum(RawDataReader &reader, Transaction &tr) {
    uint8_t statussize = reader.pop<uint8_t>();
    uint16_t pixels = reader.pop<uint16_t>();
    tr[ *this].m_integrationTime = reader.pop<uint32_t>() * 1e-6; //sec
    uint8_t lamp_enabled = reader.pop<uint8_t>();
    uint8_t trigger_mode = reader.pop<uint8_t>();
    uint8_t acq_status = reader.pop<uint8_t>();
    uint8_t packets_in_spectrum = reader.pop<uint8_t>();
    uint8_t power_down = reader.pop<uint8_t>();
    uint8_t packets_in_ep = reader.pop<uint8_t>();
    reader.pop<uint8_t>();
    reader.pop<uint8_t>();
    uint8_t usb_speed = reader.pop<uint8_t>();
    for(unsigned int i = 15; i < statussize; ++i)
        reader.pop<uint8_t>();

    std::vector<double> wavelenCalibCoeffs(4); //polynominal func. coeff.
    wavelenCalibCoeffs.resize(reader.pop<uint8_t>());
    for(unsigned int i = 0; i < wavelenCalibCoeffs.size(); ++i)
        wavelenCalibCoeffs[i] = reader.pop<double>();
    std::vector<double> strayLightCoeffs(2); //polynominal func. coeff.
    strayLightCoeffs.resize(reader.pop<uint8_t>());
    for(unsigned int i = 0; i < strayLightCoeffs.size(); ++i)
        strayLightCoeffs[i] = reader.pop<double>();
    std::vector<double> nonlinCorrCoeffs(8); //polynominal func. coeff.
    nonlinCorrCoeffs.resize(reader.pop<uint8_t>());
    for(unsigned int i = 0; i < nonlinCorrCoeffs.size(); ++i)
        nonlinCorrCoeffs[i] = reader.pop<double>();

    auto fn_poly = [](const auto &coeffs, double v) {
        double y = 0.0, x = 1.0;
        for(auto coeff: coeffs) {
            y += coeff * x;
            x *= v;
        }
        return y;
    };

    unsigned int samples = reader.pop<uint32_t>();
    samples /= 2; //uint16_t each + 0x69
    tr[ *this].waveLengths_().resize(samples);
    tr[ *this].accumCounts_().resize(samples, 0.0);
    for(unsigned int i = 0; i < samples; ++i) {
        double lambda = fn_poly(wavelenCalibCoeffs, i);
        tr[ *this].waveLengths_()[i] = lambda;
        double v = 0x100 * reader.pop<uint8_t>();
        v += reader.pop<uint8_t>(); //little endian
        v = fn_poly(nonlinCorrCoeffs, v);
        tr[ *this].accumCounts_()[i] += v;
    }
    if(reader.pop<uint8_t>() != 0x69)
        throw XInterface::XConvError(__FILE__, __LINE__);
    tr[ *this].m_accumulated++;

//    tr[ *this].counts_()[0] = tr[ *this].counts_()[1];

}
