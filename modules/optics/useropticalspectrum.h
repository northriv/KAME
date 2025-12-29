/***************************************************************************
        Copyright (C) 2002-2018 Kentaro Kitagawa
		                   kitag@issp.u-tokyo.ac.jp
		
		This program is free software; you can redistribute it and/or
		modify it under the terms of the GNU Library General Public
		License as published by the Free Software Foundation; either
		version 2 of the License, or (at your option) any later version.
		
		You should have received a copy of the GNU Library General 
		Public License and a list of authors along with this program; 
		see the files COPYING and AUTHORS.
***************************************************************************/
#ifndef usernetworkanalyerH
#define usernetworkanalyerH

#include "opticalspectrometer.h"
#include "chardevicedriver.h"
//---------------------------------------------------------------------------

#if defined USE_OCEANOPTICS_USB
#include "oceanopticsusb.h"

//! OceanOptics/Seabreeze spectrometer, HR2000+
class XOceanOpticsSpectrometer : public XCharDeviceDriver<XOpticalSpectrometer, XOceanOpticsUSBInterface> {
public:
    XOceanOpticsSpectrometer(const char *name, bool runtime,
		Transaction &tr_meas, const shared_ptr<XMeasure> &meas);
    virtual ~XOceanOpticsSpectrometer() {}
protected:
    virtual void onStartWavelenChanged(const Snapshot &shot, XValueNodeBase *) override {}
    virtual void onStopWavelenChanged(const Snapshot &shot, XValueNodeBase *) override {}
    virtual void onAverageChanged(const Snapshot &shot, XValueNodeBase *) override;
    virtual void onIntegrationTimeChanged(const Snapshot &shot, XValueNodeBase *) override;
    virtual void onEnableStrobeChnaged(const Snapshot &shot, XValueNodeBase *) override;
    virtual void onStrobeCondChnaged(const Snapshot &shot, XValueNodeBase *) override;
    virtual void onTrigCondChnaged(const Snapshot &shot, XValueNodeBase *) override;

    virtual void convertRawAndAccum(RawDataReader &reader, Transaction &tr) override;

	//! Be called just after opening interface. Call start() inside this routine appropriately.
    virtual void open() override;

    virtual void acquireSpectrum(shared_ptr<RawData> &) override;
private:
    std::vector<double> m_wavelenCalibCoeffs; //polynominal func. coeff.
    std::vector<double> m_nonlinCorrCoeffs; //polynominal func. coeff.
    std::vector<double> m_strayLightCoeffs; //polynominal func. coeff.
    std::vector<uint8_t> m_spectrumBuffer;
};
#endif //USE_OCEANOPTICS_USB

#endif
