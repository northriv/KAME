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
  
REGISTER_TYPE(KE2000);
REGISTER_TYPE(KE2182);
REGISTER_TYPE(HP34420A);
REGISTER_TYPE(CryoconM32);
REGISTER_TYPE(CryoconM62);
REGISTER_TYPE(LakeShore340);
REGISTER_TYPE(AVS47IB);
REGISTER_TYPE(ITC503);
REGISTER_TYPE(PS120);
REGISTER_TYPE(IPS120);
REGISTER_TYPE(SR830);
REGISTER_TYPE(AH2500A);
REGISTER_TYPE(YK7651);
REGISTER_TYPE(WAVEFACTORY);
REGISTER_TYPE(TDS);
#ifdef HAVE_NI_DAQMX
    REGISTER_TYPE_N_NAME(XNIDAQmxDSO, "DSO-NIDAQ");
    REGISTER_TYPE_N_NAME(XNIDAQSSeriesPulser, "NMRPulser-NIDAQ-SSeries");
    REGISTER_TYPE_N_NAME(XNIDAQDOPulser, "NMRPulser-NIDAQ-DOonly");
    REGISTER_TYPE_N_NAME(XNIDAQMSeriesWithSSeriesPulser, "NMRPulser-NIDAQ-MSeriesWithSSeries");
#endif
REGISTER_TYPE(H8Pulser);
REGISTER_TYPE(SHPulser);
REGISTER_TYPE(NMRPulseAnalyzer);
REGISTER_TYPE(NMRSpectrum);
REGISTER_TYPE(NMRFSpectrum);
REGISTER_TYPE(NMRT1);
REGISTER_TYPE(SG7130);
REGISTER_TYPE(SG7200);
REGISTER_TYPE(HP8643);
REGISTER_TYPE(HP8648);
REGISTER_TYPE(TestDriver);
REGISTER_TYPE(MonteCarloDriver);

shared_ptr<XNode>
XDriverList::createByTypename(const std::string &type, const std::string& name) {
    shared_ptr<XNode> ptr = (*creator(type))
        (name.c_str(), false, m_scalarentries, m_interfaces, m_thermometers,
                dynamic_pointer_cast<XDriverList>(shared_from_this()));
    if(ptr) insert(ptr);
    return ptr;
}
