/***************************************************************************
        Copyright (C) 2002-2026 Kentaro Kitagawa
                           kitag@issp.u-tokyo.ac.jp

        This program is free software; you can redistribute it and/or
        modify it under the terms of the GNU General Public
        License as published by the Free Software Foundation; either
        version 2 of the License, or (at your option) any later version.

        You should have received a copy of the GNU General
        Public License and a list of authors along with this program;
        see the files COPYING and AUTHORS.
***************************************************************************/
#include "useropticalspectrum.h"
#include "charinterface.h"
#include "analyzer.h"

#if defined USE_OCEANOPTICS_USB


REGISTER_TYPE(XDriverList, OceanOpticsSpectrometer, "OceanOptics/Insight USB/HR2000(+)/4000 spectrometer");

//---------------------------------------------------------------------------
XOceanOpticsSpectrometer::XOceanOpticsSpectrometer(const char *name, bool runtime,
    Transaction &tr_meas, const shared_ptr<XMeasure> &meas) :
    XCharDeviceDriver<XOpticalSpectrometer, XOceanOpticsUSBInterface>(name, runtime, ref(tr_meas), meas) {
//    startWavelen()->disable();
//    stopWavelen()->disable();
    //! The combo index IS the raw SET_TRIG_MODE value, and the label says so. Only 0 and 1 are
    //! unambiguous across firmware generations; what 2/3/4 mean depends on the FPGA firmware
    //! version (see XOceanOpticsUSBInterface::TrigMode), so they are named by value rather than
    //! by a guessed semantic — a mislabelled external mode previously made the driver sit in a
    //! mode that returns un-integrated (dark) readouts. open() reports the firmware version.
    trans( *trigMode()).add({"Free Run (0)", "Software Trig. (1)",
        "Ext. Trig. 2", "Ext. Trig. 3", "Ext. Trig. 4"});
}

void
XOceanOpticsSpectrometer::open() {
    m_statusCacheValid = false; //nothing cached yet for this device session.
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

    try {
        auto status = interface()->readInstrumStatus();
        uint16_t ver_raw = interface()->readRegInfo(XOceanOpticsUSBInterface::Register::FPGAFirmwareVersion);
        uint16_t ver = ver_raw / 0x1000; //major version.
        //Report the version, but do NOT assert what the external values mean: an HR4000
        //reporting 0x3000 was measured to follow NEITHER vendor document — value 2 behaved as
        //Synchronization (exposure tracked the trigger PERIOD, independent of duty), value 3
        //returned un-integrated frames, and value 4 was ignored (device stayed free-running).
        //So the candidate meanings are printed as candidates, to be confirmed on the bench.
        gMessagePrint(formatString("FPGA firmware 0x%04x (major %u). External TrigMode values are "
            "raw SET_TRIG_MODE payloads; verify on the bench — vendor docs disagree and at least "
            "one unit follows neither. Candidates: doc<3.0 => 2=Synchronization, "
            "3=Hardware(edge, exposure=IntegrationTime); doc>=3.0 => 2=Hardware LEVEL"
            "(exposure=trigger HIGH width), 3=Synchronous, 4=Hardware EDGE. Bench test: at fixed "
            "frequency, counts that track DUTY mean LEVEL, counts that track PERIOD mean "
            "SYNCHRONIZATION.", (unsigned)ver_raw, (unsigned)ver));
        uint16_t div = interface()->readRegInfo(XOceanOpticsUSBInterface::Register::MasterClockCounterDivisor);
        uint16_t delay = interface()->readRegInfo(XOceanOpticsUSBInterface::Register::HardwareTriggerDelay);
        uint16_t time_to_strobe = interface()->readRegInfo(XOceanOpticsUSBInterface::Register::SingleStrobeHighClockTransition);
        uint16_t strobe_duration = interface()->readRegInfo(XOceanOpticsUSBInterface::Register::SingleStrobeLowClockTransition);
        iterate_commit([=](Transaction &tr){
            uint32_t integration_time_us = status[2] + status[3] * 0x100u + status[4] * 0x10000u + status[5] * 0x1000000uL;
            tr[ *integrationTime()] = integration_time_us * 1e-6;
            tr[ *enableStrobe()] = status[6];
            tr[ *trigMode()] = status[7];
            tr[ *timeToStrobeSignal()] = time_to_strobe * 1e-3;
            tr[ *strobeSignalDuration()] = strobe_duration * 1e-3;
            double delay_sec = (ver < 3) ? delay / (48e6 / div) : delay * 500e-9;
            tr[ *delayFromExtTrig()] = delay_sec;
        });
    }
    catch (XInterface::XUnsupportedFeatureError &e) {
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
        m_statusCacheValid = false; //integration time is one of the cached status fields.
        interface()->setIntegrationTime(lrint(shot[ *integrationTime()] * 1e6));
    }
    catch (XKameError &e) {
        e.print(getLabel() + " " + i18n(" Error"));
    }
}
void
XOceanOpticsSpectrometer::onEnableStrobeChnaged(const Snapshot &shot, XValueNodeBase *) {
    try {
        interface()->enableStrobe(shot[ *enableStrobe()]);
    }
    catch (XKameError &e) {
        e.print(getLabel() + " " + i18n(" Error"));
    }
}
void
XOceanOpticsSpectrometer::onStrobeCondChnaged(const Snapshot &, XValueNodeBase *) {
    try {
        Snapshot shot( *this);
        interface()->setupStrobeCond(shot[ *timeToStrobeSignal()], shot[ *strobeSignalDuration()]);
    }
    catch (XKameError &e) {
        e.print(getLabel() + " " + i18n(" Error"));
    }
}
void
XOceanOpticsSpectrometer::onTrigCondChnaged(const Snapshot &, XValueNodeBase *) {
    try {
        m_statusCacheValid = false; //trigger mode and the device reset below both stale it.
        Snapshot shot( *this);
        if( !interface()->isUSB2000()) {
            //HR4000-class only. Leaving a trigger mode can latch the FPGA acquisition state
            //(status[8] stuck at 3 "acquiring", so Free Run never reports ready). Neither
            //SET_TRIG_MODE nor CMD::INIT clears that — only a USB port reset does (the same as
            //a manual interface Control off/on). So on any trigger-mode change: reset the
            //device (close+reopen the handle), re-init, then re-apply the current settings
            //before applying the new trigger mode. OceanOptics firmware is persistent, so the
            //reset does not require a firmware reload. USB2000 lacks these FPGA registers and
            //is left to the original (no-op-ish) path below.
            interface()->resetDevice();
            msecsleep(100); //let the port reset settle before talking to the device again.
            interface()->initDevice();
            interface()->clearSpectrumEndpoints();
            interface()->setIntegrationTime(lrint(shot[ *integrationTime()] * 1e6));
            interface()->enableStrobe(shot[ *enableStrobe()]);
            interface()->setupStrobeCond(shot[ *timeToStrobeSignal()], shot[ *strobeSignalDuration()]);
        }
        unsigned int requested = (unsigned int)shot[ *trigMode()];
        interface()->setupTrigCond((XOceanOpticsUSBInterface::TrigMode)requested,
            shot[ *delayFromExtTrig()]);
        //Read the mode back from the device: status[7] is the trigger mode the firmware actually
        //holds. If it does not echo what we sent, the value is unsupported on this firmware —
        //which is the failure that silently returns un-integrated (dark) frames. Purely
        //diagnostic, so a failed read must not abort the mode change.
        try {
            auto st = interface()->readInstrumStatus();
            if(st.size() > 8)
                gMessagePrint(formatString(
                    "%s: TrigMode requested %u -> device reports %u (acq. status %u)",
                    getLabel().c_str(), requested, (unsigned)st[7], (unsigned)st[8]));
            //Deliberately NOT filled into m_cachedStatus: that vector is written only by the
            //acquisition thread (this runs on the caller's), and m_statusCacheValid stays false
            //above, so the acquisition thread refreshes it itself.
        }
        catch (XKameError &) {
            //device busy/armed right after the change; the acquisition loop will re-read it.
        }
    }
    catch (XKameError &e) {
        e.print(getLabel() + " " + i18n(" Error"));
    }
}
void
XOceanOpticsSpectrometer::onAnalogOutputChnaged(const Snapshot &shot, XValueNodeBase *) {
    try {
        Snapshot shot( *this);
        interface()->setAnalogOutput(shot[ *analogOutput()]);
    }
    catch (XKameError &e) {
        e.print(getLabel() + " " + i18n(" Error"));
    }
}


void
XOceanOpticsSpectrometer::acquireSpectrum(shared_ptr<RawData> &writer, const atomic<bool> &terminated) {
    // Take the commanded trigger mode BEFORE the interface lock: a device mutex must never be
    // held across a Snapshot. It also replaces the former status[7] probe — deciding this from
    // a status query is what forced a USB round trip to an armed device on every cycle.
    const bool in_trig_mode = ((unsigned int)Snapshot( *this)[ *trigMode()] != 0);

    XScopedLock<XOceanOpticsUSBInterface> lock( *interface());
    bool isusb2000 = interface()->isUSB2000();

    if(isusb2000) //USB2000 can respond control commands even after requestSpectrum().
        interface()->requestSpectrum();

    // External/software trigger mode (HR4000-class): the spectrometer yields a spectrum only
    // after a trigger edge, so we arm (requestSpectrum) and read with an interruptible, polled
    // async read (readSpectrumInterruptible): it returns as soon as the triggered data is
    // ready, aborts on thread termination, and only times out (then skips) if no trigger ever
    // arrives.
    bool trig_mode = !isusb2000 && in_trig_mode;

    uint16_t pixels = 2048u;
    uint8_t usb_speed = 0u;
    bool acq_ready = true;
    uint32_t integration_time_us = 0;
    std::vector<uint8_t> status;
    if(interface()->hasStatusQuery()) {
        if(trig_mode && m_statusCacheValid && !m_cachedStatus.empty()) {
            // Armed and waiting for an edge: reuse the cached status rather than polling the
            // device. A status read issued while it is armed can block for the entire USB
            // timeout and come back short, which readInstrumStatus() reports as XConvError.
            // acq_ready is only consulted on the non-trigger path, so a stale copy is fine.
            status = m_cachedStatus;
        }
        else {
            try {
                status = interface()->readInstrumStatus();
            }
            catch (XInterface::XConvError &) {
                // Short read: the transfer timed out against a busy/armed device. Recoverable,
                // so skip this cycle quietly instead of reporting a communication failure.
                throw XSkippedRecordError(__FILE__, __LINE__);
            }
            m_cachedStatus = status;
            m_statusCacheValid = true;
        }

        pixels = isusb2000 ? status[0] * 0x100u + status[1] : status[0] + status[1] * 0x100u;
    //        uint8_t packets_in_spectrum = status[9];
    //        uint8_t packets_in_ep = status[11];
        usb_speed = status[14]; //0x80 if highspeed
        acq_ready = isusb2000 ? (status[8] != 0) : (status[8] == 0);

        integration_time_us = isusb2000 ? (status[2] * 0x100u + status[3]) * 1000u:
                    status[2] + status[3] * 0x100u + status[4] * 0x10000u + status[5] * 0x1000000uL;
    }

    if( !trig_mode && !acq_ready) {
        //waits for completion
        msecsleep(std::min(100.0, integration_time_us * 1e-3 / 4));
        throw XSkippedRecordError(__FILE__, __LINE__);
    }

    if( !isusb2000)
        interface()->requestSpectrum();

    if(isusb2000 && interface()->hasStatusQuery())
        status.resize(14); //to distinguish USB2000.
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

    int len;
    if(trig_mode) {
        //On-demand: wait for the requested trigger (its edge then the exposure). Generous,
        //since the loop is idle between requests; thread stop aborts immediately.
        double trig_timeout = std::max(15.0, integration_time_us * 1e-6 * 2 + 10.0);
        len = interface()->readSpectrumInterruptible(m_spectrumBuffer, pixels,
            usb_speed == 0x80u, terminated, trig_timeout);
    }
    else
        len = interface()->readSpectrum(m_spectrumBuffer, pixels, usb_speed == 0x80u);
    if( !len)
        throw XSkippedRecordError(__FILE__, __LINE__);
    writer->push((uint32_t)len); //be actual pixels + 1(end delimiter 0x69).
    writer->insert(writer->end(),
                     m_spectrumBuffer.begin(), m_spectrumBuffer.begin() + len);

}
void
XOceanOpticsSpectrometer::convertRawAndAccum(RawDataReader &reader, Transaction &tr) {
    uint8_t statussize = reader.pop<uint8_t>();
    bool isusb2000 = statussize < 16;
    uint16_t pixels = 2048u;
    if(statussize) {
        if(isusb2000)
            pixels = reader.pop<uint8_t>() * 0x100u + reader.pop<uint8_t>(); //MSB,LSB
        else
            pixels = reader.pop<uint16_t>();
        tr[ *this].m_integrationTime = isusb2000 ?
            (reader.pop<uint8_t>() * 0x100u + reader.pop<uint8_t>()) * 1e-3 : reader.pop<uint32_t>() * 1e-6; //sec
        uint8_t lamp_enabled = reader.pop<uint8_t>();
        uint8_t trigger_mode = reader.pop<uint8_t>();
        uint8_t acq_status = reader.pop<uint8_t>(); //in USB2000, is request spectra.
        uint8_t packets_in_spectrum = reader.pop<uint8_t>(); //in USB2000, is timer swap.
        uint8_t power_down = reader.pop<uint8_t>(); //in USB2000, is spectra data ready.
        uint8_t packets_in_ep = reader.pop<uint8_t>(); //in USB2000, reserve = 0.
        reader.pop<uint8_t>();
        reader.pop<uint8_t>();
        uint8_t usb_speed = reader.pop<uint8_t>(); //0x80 if highspeed,  //in USB2000, is researve = 0.
        reader.pop<uint8_t>();
        for(unsigned int i = 16; i < statussize; ++i)
            reader.pop<uint8_t>(); //for future?
    }
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
    if(tr[ *this].m_nonLinCorrCoeffs.size() <= 1) {
        tr[ *this].m_nonLinCorrCoeffs = {1.0};
    }

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
    if(isusb2000) {
        //Sony ILX511 CCD
        dark_pixel_begin = 2;
        dark_pixel_end = 25;
        active_pixel_begin = 26;
        active_pixel_end = 2048;
    }
    if(samples > 2048) {
        //Toshiba TCD1304AP CCD
        dark_pixel_begin = 5;
        dark_pixel_end = 18;
        active_pixel_begin = 21;
        active_pixel_end = 3669;
    }
    tr[ *this].waveLengths_().resize(active_pixel_end - active_pixel_begin);
    tr[ *this].accumCounts_().resize(active_pixel_end - active_pixel_begin, 0.0);

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
#endif // OCEANOPTICSUSB_H
