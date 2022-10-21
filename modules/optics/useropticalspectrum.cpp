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

REGISTER_TYPE(XDriverList, OceanOpticsSpectrometer, "OceanOptics/Seabreeze HR2000+/4000 spectrometer");

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
    try {
        interface()->setIntegrationTime(lrint(shot[ *integrationTime()] * 1e6));
    }
    catch (XKameError &e) {
        e.print(getLabel() + " " + i18n(" Error"));
    }
}

void
XOceanOpticsSpectrometer::acquireSpectrum(shared_ptr<RawData> &writer) {
    XScopedLock<XOceanOpticsUSBInterface> lock( *interface());

    auto status = interface()->readInstrumStatus();
    uint16_t pixels = status[0] + status[1] * 0x100u;
    uint8_t packets_in_spectrum = status[9];
    uint8_t packets_in_ep = status[11];
    uint8_t usb_speed = status[14]; //0x80 if highspeed
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

    //ugly hack
    uint32_t integration_time_us = status[2] + status[3] * 0x100u + status[4] * 0x10000u + status[5] * 0x1000000uL;

    if(integration_time_us > 300000)
        msecsleep(integration_time_us / 1000 - 100);
    int len = interface()->readSpectrum(m_spectrumBuffer, pixels, usb_speed == 0x80u);
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
    uint8_t usb_speed = reader.pop<uint8_t>(); //0x80 if highspeed
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
    tr[ *this].m_nonLinCorrCoeffs.resize(reader.pop<uint8_t>());
    for(unsigned int i = 0; i < tr[ *this].m_nonLinCorrCoeffs.size(); ++i)
        tr[ *this].m_nonLinCorrCoeffs[i] = reader.pop<double>();

    auto fn_poly = [](const std::vector<double> &coeffs, double v) {
        double y = 0.0, x = 1.0;
        for(auto coeff: coeffs) {
            y += coeff * x;
            x *= v;
        }
        return y;
    };

    unsigned int samples = reader.pop<uint32_t>();
    samples /= 2; //uint16_t each + 0x69
    //Sony ILX511B CCD
    unsigned int dark_pixel_begin = 0;
    unsigned int dark_pixel_end = 18;
    unsigned int active_pixel_begin = 20;
    unsigned int active_pixel_end = 2048;
    if(samples > 2048) {
        //Toshiba TCD1304AP CCD
        dark_pixel_begin = 5;
        dark_pixel_end = 18;
        active_pixel_begin = 21;
        active_pixel_end = 3669;
    }
    tr[ *this].waveLengths_().resize(active_pixel_end - active_pixel_begin);
    tr[ *this].accumCounts_().resize(active_pixel_end - active_pixel_begin, 0.0);
    //todo HR2000 bundles LSB/MSB every one USB packet.
    double dark = 0.0;
    int dark_cnt = 0;
    uint32_t xor_bit = 0;
    for(unsigned int i = 0; i < active_pixel_begin; ++i) {
        uint32_t v = reader.pop<uint16_t>(); //little endian
        if(i == 0) {
            //detecting bit13 flip for HR2000+
            xor_bit = lrint(std::pow(2.0, floor(std::log2((double)v))));
            if(xor_bit < 0x1000uL)
                xor_bit = 0;
        }
        if((i >= dark_pixel_begin) && (i < dark_pixel_end)) {
            dark_cnt++;
            dark += v ^ xor_bit;
        }
    }
    dark /= dark_cnt;
    tr[ *this].m_electric_dark = dark;
    auto &poly_coeff = tr[ *this].m_nonLinCorrCoeffs;
    for(unsigned int i = 0; i < active_pixel_end - active_pixel_begin; ++i) {
        double lambda = fn_poly(wavelenCalibCoeffs, i + active_pixel_begin);
        tr[ *this].waveLengths_()[i] = lambda;
        uint32_t v = reader.pop<uint16_t>(); //little endian
        v = v ^ xor_bit;
        double efficiency = fn_poly(poly_coeff, v);
        tr[ *this].accumCounts_()[i] += (v - dark) / efficiency + dark;
    }
    for(unsigned int i = active_pixel_end; i < samples; ++i) {
        reader.pop<uint16_t>();
    }
    if(reader.pop<uint8_t>() != 0x69)
        throw XInterface::XConvError(__FILE__, __LINE__);
    tr[ *this].m_accumulated++;
}
