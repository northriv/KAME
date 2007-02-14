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
  
REGISTER_TYPE(TestDriver);
REGISTER_TYPE(KE2000);
REGISTER_TYPE(KE2182);
REGISTER_TYPE(HP34420A);
REGISTER_TYPE(TDS);
#ifdef HAVE_NI_DAQMX
    REGISTER_TYPE(NIDAQmxDSO);
    REGISTER_TYPE(NIDAQSSeriesPulser);
    REGISTER_TYPE(NIDAQMSeriesPulser);
    REGISTER_TYPE(NIDAQMSeriesWithSSeriesPulser);
#endif
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
REGISTER_TYPE(H8Pulser); //nmr stuff, remove me in lite version.
REGISTER_TYPE(SHPulser); //nmr stuff, remove me in lite version.
REGISTER_TYPE(NMRPulseAnalyzer); //nmr stuff, remove me in lite version.
REGISTER_TYPE(NMRSpectrum); //nmr stuff, remove me in lite version.
REGISTER_TYPE(NMRFSpectrum); //nmr stuff, remove me in lite version.
REGISTER_TYPE(NMRT1); //nmr stuff, remove me in lite version.
REGISTER_TYPE(SG7130); //nmr stuff, remove me in lite version.
REGISTER_TYPE(SG7200); //nmr stuff, remove me in lite version.
REGISTER_TYPE(HP8643); //nmr stuff, remove me in lite version.
REGISTER_TYPE(HP8648); //nmr stuff, remove me in lite version.
REGISTER_TYPE(MonteCarloDriver);

shared_ptr<XNode>
XDriverList::createByTypename(const std::string &type, const std::string& name) {
    shared_ptr<XNode> ptr = (*creator(type))
        (name.c_str(), false, m_scalarentries, m_interfaces, m_thermometers,
                dynamic_pointer_cast<XDriverList>(shared_from_this()));
    if(ptr) insert(ptr);
    return ptr;
}
