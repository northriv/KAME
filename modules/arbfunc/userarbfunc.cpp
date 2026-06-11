/***************************************************************************
        Copyright (C) 2002-2023 Kentaro Kitagawa
                           kitag@issp.u-tokyo.ac.jp

        This program is free software; you can redistribute it and/or
        modify it under the terms of the GNU General Public
        License as published by the Free Software Foundation; either
        version 2 of the License, or (at your option) any later version.

        You should have received a copy of the GNU General
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
    trans( *trigSrc()).add({"IMM", "EXT", "BUS"});
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
//    changeOutput(false);
    interface()->send("BURST:STAT OFF");
    interface()->sendf("APPL:%s %g, %g, %g",
        shot[ *waveform()].to_str().c_str(),
        (double)shot[ *freq()],
        (double)shot[ *ampl()],
        (double)shot[ *offset()]);
    interface()->sendf("FUNC:SQU:DCYC %g", (double)shot[ *duty()]);
    double period = shot[ *pulsePeriod()];
    if(period > 0)
        interface()->sendf("PULSE:PER %g", period); //overrides period given by 1/Freq
    double width = shot[ *pulseWidth()];
    if(width > 0)
        interface()->sendf("FUNC:PULSE:WIDTH %g", width); //width and duty are exclusive; width takes over
    else
        interface()->sendf("FUNC:PULSE:DCYC %g", (double)shot[ *duty()]); //width = 0: specify by duty
    if(shot[ *burst()]) {
    //    changeOutput(shot[ *output()]);
        interface()->sendf("BURS:PHAS %g", (double)shot[ *burstPhase()]);
        unsigned int cyc = shot[ *burstCycles()];
        if(cyc == 0)
            interface()->send("BURS:NCYC INF");
        else
            interface()->sendf("BURS:NCYC %u", cyc);
        interface()->send("TRIG:SOUR " + shot[ *trigSrc()].to_str());
        interface()->send("BURST:STAT ON");
        if(shot[ *output()] && (cyc == 0)) {
            if(shot[ *trigSrc()].to_str() == "BUS") {
                interface()->send("*OPC;*TRG"); //issue a trigger
            }
        }
    }
    else {
        //hack for studpid LXI3390
        interface()->sendf("PHAS %g", (double)shot[ *burstPhase()]);
    }
}

void
XArbFuncGenSCPI::open() {
    interface()->send("*CLS");
    XString __func, __trigsrc;
    bool __burst = false;
    double __freq, __ampl, __offset, __duty, __period, __width, __burstphase;
    interface()->query("BURST:STAT?");
    if(interface()->toInt() == 1)
        __burst = true;
    interface()->query("BURST:PHASE?");
    __burstphase = interface()->toDouble();
    unsigned int __cycles = 0; //0 = INFinity
    interface()->query("BURST:NCYC?");
    if(interface()->toStrSimplified() != "INF") {
        double __ncyc = interface()->toDouble();
        if((__ncyc > 0.5) && (__ncyc < 1e9))
            __cycles = (unsigned int)(__ncyc + 0.5);
    }
    interface()->query("FUNC?");
    __func = interface()->toStrSimplified();
    interface()->query("TRIG:SOUR?");
    __trigsrc = interface()->toStrSimplified();
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
    //Node conventions: PulseWidth 0 = specify the pulse shape by Duty;
    //PulsePeriod 0 = follow Freq (period = 1/Freq). Start in the legacy
    //duty/Freq-driven mode; entering a value takes over.
    __width = 0.0;
    __period = 0.0;

    iterate_commit([=](Transaction &tr){
        tr[ *burst()] = __burst;
        tr[ *burstPhase()] = __burstphase;
        tr[ *burstCycles()] = __cycles;
        tr[ *freq()] = __freq;
        tr[ *ampl()] = __ampl;
        tr[ *offset()] = __offset;
        tr[ *duty()] = __duty;
        tr[ *pulsePeriod()] = __period;
        tr[ *pulseWidth()] = __width;
        tr[ *waveform()].str(__func);
        tr[ *trigSrc()].str(__trigsrc);
    });

    start();
}

