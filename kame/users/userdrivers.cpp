/***************************************************************************
		Copyright (C) 2002-2007 Kentaro Kitagawa
		                   kitagawa@scphys.kyoto-u.ac.jp
		
		This program is free software; you can redistribute it and/or
		modify it under the terms of the GNU Library General Public
		License as published by the Free Software Foundation; either
		version 2 of the License, or (at your option) any later version.
		
		You should have received a copy of the GNU Library General 
		Public License and a list of authors along with this program; 
		see the files COPYING and AUTHORS.
 ***************************************************************************/
#include "users/testdriver.h"
#include "users/dmm/userdmm.h"
#include "users/dso/tds.h"
#include "users/dso/nidaqdso.h"
#include "tempcontrol/usertempcontrol.h"
#include "magnetps/usermagnetps.h"
#include "nmr/pulserdriverh8.h"
#include "nmr/pulserdriversh.h"
#include "nmr/pulserdrivernidaq.h"
#include "nmr/nmrpulse.h"
#include "nmr/nmrspectrum.h"
#include "nmr/nmrfspectrum.h"
#include "nmr/nmrrelax.h"
#include "nmr/signalgenerator.h"
#include "lia/userlockinamp.h"
#include "dcsource/dcsource.h"
#include "funcsynth/userfuncsynth.h"
#include "montecarlo/kamemontecarlo.h"
//---------------------------------------------------------------------------

#define LIST XDriverList
DECLARE_TYPE_HOLDER
  
REGISTER_TYPE(KE2000, "Keithley 2000/2001 DMM");
REGISTER_TYPE(KE2182, "Keithley 2182 nanovolt meter");
REGISTER_TYPE(HP34420A, "Agilent 34420A nanovolt meter");
REGISTER_TYPE(CryoconM32, "Cryocon M32 temp. controller");
REGISTER_TYPE(CryoconM62, "Cryocon M62 temp. controller");
REGISTER_TYPE(LakeShore340, "LakeShore 340 temp. controller");
REGISTER_TYPE(AVS47IB, "Picowatt AVS-47 bridge");
REGISTER_TYPE(ITC503, "Oxford ITC-503 temp. controller");
REGISTER_TYPE(PS120, "Oxford PS-120 magnet power supply");
REGISTER_TYPE(IPS120, "Oxford IPS-120 magnet power supply");
REGISTER_TYPE(SR830, "Stanford Research SR830 lock-in amp.");
REGISTER_TYPE(AH2500A, "Andeen-Hagerling 2500A capacitance bridge");
REGISTER_TYPE(YK7651, "YOKOGAWA 7651 dc source");
REGISTER_TYPE(WAVEFACTORY, "NF WAVE-FACTORY pulse generator");
REGISTER_TYPE(SG7130, "KENWOOD SG7130 signal generator");
REGISTER_TYPE(SG7200, "KENWOOD SG7200 signal generator");
REGISTER_TYPE(HP8643, "Agilent 8643 signal generator");
REGISTER_TYPE(HP8648, "Agilent 8648 signal generator");
REGISTER_TYPE(TDS, "Tektronix DSO");
#ifdef HAVE_NI_DAQMX
    REGISTER_TYPE(NIDAQmxDSO, "National Instruments DAQ as DSO");
    REGISTER_TYPE(NIDAQSSeriesPulser, "NMR pulser NI-DAQ S Series");
    REGISTER_TYPE(NIDAQDOPulser, "NMR pulser NI-DAQ digital port only");
    REGISTER_TYPE(NIDAQMSeriesWithSSeriesPulser, "NMR pulser NI-DAQ M Series with S Series");
#endif
REGISTER_TYPE(H8Pulser, "NMR pulser handmade-H8");
REGISTER_TYPE(SHPulser, "NMR pulser handmade-SH2");
REGISTER_TYPE(NMRPulseAnalyzer, "NMR FID/echo analyzer");
REGISTER_TYPE(NMRSpectrum, "NMR field-swept spectrum measurement");
REGISTER_TYPE(NMRFSpectrum, "NMR frequency-swept spectrum measurement");
REGISTER_TYPE(NMRT1, "NMR relaxation measurement");
REGISTER_TYPE(TestDriver, "Test driver: random number generation");
REGISTER_TYPE(MonteCarloDriver, "Monte-Carlo simulation");

shared_ptr<XNode>
XDriverList::createByTypename(const std::string &type, const std::string& name) {
    shared_ptr<XNode> ptr = (*creator(type))
        (name.c_str(), false, m_scalarentries, m_interfaces, m_thermometers,
                dynamic_pointer_cast<XDriverList>(shared_from_this()));
    if(ptr) insert(ptr);
    return ptr;
}
