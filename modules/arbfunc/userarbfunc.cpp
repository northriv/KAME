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
    trans( *waveform()).add({"SIN", "SQU", "RAMP", "PULS", "NOIS", "DC", "USER", "PATT"});
//    interface()->setGPIBMAVbit(0x10);
    interface()->setGPIBUseSerialPollOnWrite(false);
    interface()->setGPIBUseSerialPollOnRead(false);
    interface()->setGPIBWaitBeforeSPoll(50);
    interface()->setGPIBWaitBeforeWrite(50);
    interface()->setGPIBWaitBeforeRead(50);
    interface()->setEOS("\n");
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
    if(shot[ *burst()])
        interface()->send("BURST:STAT ON");
    else
        interface()->send("BURST:STAT OFF");
    interface()->sendf("BURST:PHASE %g", (double)shot[ *burstPhase()]);
    changeOutput(shot[ *output()]);
//    if(shot[ *burst()]) {
//        interface()->query("BURST:NCYC?");
//        if(interface()->toStr() == "INF") {
//            interface()->query("TRIG:SOUR?");
//            if(interface()->toStr() == "IMM")

//        }
//    }
}

void
XArbFuncGenSCPI::open() {
    XString __func;
    bool __burst = false;
    double __freq, __ampl, __offset, __duty, __period, __width, __burstphase;
    interface()->query("BURST:STAT?");
    if(interface()->toStr() == "ON")
        __burst = true;
    interface()->query("BURST:PHASE?");
    __burstphase = interface()->toDouble();
    interface()->query("FUNC?");
    __func = interface()->toStr();
    interface()->query("FREQ?");
    __freq = interface()->toDouble();
    interface()->query("VOLT?");
    __ampl = interface()->toDouble();
    interface()->query("VOLT:OFFSET?");
    __offset = interface()->toDouble();
    if(__func == "SQU")
        interface()->query("FUNC:SQU:DCYC?");
    else
        interface()->query("FUNC:PULSE:DCYC?");
    __duty = interface()->toDouble();
    interface()->query("FUNC:PULSE:WIDTH?");
    __width = interface()->toDouble();
    interface()->query("PULSE:PER?");
    __period = interface()->toDouble();

    iterate_commit([=](Transaction &tr){
        tr[ *burst()] = __burst;
        tr[ *burstPhase()] = __burstphase;
        tr[ *freq()] = __freq;
        tr[ *ampl()] = __ampl;
        tr[ *offset()] = __offset;
        tr[ *duty()] = __duty;
        tr[ *pulsePeriod()] = __period;
        tr[ *pulseWidth()] = __width;
        tr[ *waveform()].str(__func);
    });

    start();
}
