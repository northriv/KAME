/***************************************************************************
        Copyright (C) 2002-2023 Kentaro Kitagawa
                           kitagawa@phys.s.u-tokyo.ac.jp

        This program is free software; you can redistribute it and/or
        modify it under the terms of the GNU Library General Public
        License as published by the Free Software Foundation; either
        version 2 of the License, or (at your option) any later version.

        You should have received a copy of the GNU Library General
        Public License and a list of authors along with this program;
        see the files COPYING and AUTHORS.
***************************************************************************/
//---------------------------------------------------------------------------
#include "userarbfunc.h"
#include "charinterface.h"

REGISTER_TYPE(XDriverList, ArbFuncGenSCPI, "LXI 3390 arbitrary function generator");

XArbFuncGenSCPI::XArbFuncGenSCPI(const char *name, bool runtime,
    Transaction &tr_meas, const shared_ptr<XMeasure> &meas) : XCharDeviceDriver<XArbFuncGen>(name, runtime, ref(tr_meas), meas) {
    trans( *waveform()).add({"SINUSOID", "SQUARE", "RAMP", "PULSE", "NOISE", "DC", "USER"});
}
void
XArbFuncGenSCPI::changeOutput(bool active) {
    if(active)
        interface()->send("OUTPUT ON");
    else
        interface()->send("OUTPUT OFF");
}
void
XArbFuncGenSCPI::changePulseCond() {
    XScopedLock<XInterface> lock( *interface());
    Snapshot shot( *this);
    changeOutput(false);
    interface()->send("FUNC " + shot[ *waveform()].to_str());
    interface()->sendf("FREQ %g", (double)shot[ *freq()]);
    interface()->sendf("VOLT %g", (double)shot[ *ampl()]);
    interface()->sendf("VOLT:OFFSET %g", (double)shot[ *offset()]);
    interface()->sendf("FUNC:SQU:DCYC %g", (double)shot[ *duty()]);
    interface()->sendf("PULSE:PER %g", (double)shot[ *pulsePeriod()]);
    interface()->sendf("FUNC:PULSE:WIDTH %g", (double)shot[ *pulseWidth()]);
    interface()->sendf("FUNC:PULSE:DCYC %g", (double)shot[ *duty()]);
    changeOutput(shot[ *output()]);
}

void
XArbFuncGenSCPI::open() {

    start();
}

