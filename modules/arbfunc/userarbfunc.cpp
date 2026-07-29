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
REGISTER_TYPE(XDriverList, Agilent33250A, "Agilent/Keysight 33250A arbitrary function generator");

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
XAgilent33250A::XAgilent33250A(const char *name, bool runtime,
    Transaction &tr_meas, const shared_ptr<XMeasure> &meas)
    : XArbFuncGenSCPI(name, runtime, ref(tr_meas), meas) {
    //33250A shares most of the Agilent 33xxx SCPI command set with the 3390; only the
    //per-update command sequence (changePulseCond) needs 33250A-specific care.
}
void
XAgilent33250A::changePulseCond() {
    XScopedLock<XInterface> lock( *interface());
    Snapshot shot( *this);
    interface()->send("*CLS"); //clear stale errors so the front-panel ERR reflects this update only
    XString wave = shot[ *waveform()].to_str();
    bool is_burst = shot[ *burst()];
    //CRITICAL: never put the generator into *continuous* output while reconfiguring. The
    //base driver uses APPLy (and toggles BURST:STAT OFF), both of which un-burst the 33250A
    //and emit a continuous waveform. Because the 33250A output is wired to the HR4000
    //external-trigger input, that transient continuous output fires spurious exposures
    //(extra frames). Instead, arm the burst first (BURST:STAT ON + BUS/EXT trigger) — then
    //the generator emits ONLY on a trigger — and stage the waveform with the individual
    //FUNC/FREQ/VOLT commands, which do not un-burst and do not emit. Also avoids the
    //33220A-only FUNC:PULSe:* commands the 33250A rejects.
    if(is_burst) {
        interface()->send("TRIG:SOUR " + shot[ *trigSrc()].to_str());
        unsigned int cyc = shot[ *burstCycles()];
        if(cyc == 0)
            interface()->send("BURS:NCYC INF");
        else
            interface()->sendf("BURS:NCYC %u", cyc);
        interface()->sendf("BURS:PHAS %g", (double)shot[ *burstPhase()]);
        interface()->send("BURST:STAT ON"); //armed: no output until a trigger arrives
    }
    interface()->sendf("FUNC %s", wave.c_str());
    interface()->sendf("FREQ %g", (double)shot[ *freq()]);
    interface()->sendf("VOLT %g", (double)shot[ *ampl()]);
    interface()->sendf("VOLT:OFFSET %g", (double)shot[ *offset()]);
    if(wave == "SQU")
        interface()->sendf("FUNC:SQU:DCYC %g", (double)shot[ *duty()]);
    else if(wave == "PULS") {
        double period = shot[ *pulsePeriod()];
        if(period > 0)
            interface()->sendf("PULS:PER %g", period);
        double width = shot[ *pulseWidth()];
        if(width > 0)
            interface()->sendf("PULS:WIDT %g", width);
    }
    if( !is_burst)
        interface()->send("BURST:STAT OFF");
    //Drain the error queue: clears the front-panel ERR and surfaces any command the 33250A
    //still rejects (so an incompatibility shows up here instead of silently).
    for(int i = 0; i < 8; ++i) {
        interface()->query("SYST:ERR?");
        XString e = interface()->toStrSimplified();
        if(e.empty() || (e[0] == '+' && e.size() >= 2 && e[1] == '0') || (e[0] == '0'))
            break;
        gWarnPrint(getLabel() + " 33250A SCPI: " + e);
    }
}
void
XArbFuncGenSCPI::sendSoftwareTrigger() {
    //Standard IEEE-488.2 bus trigger; fires a burst when TRIG:SOUR is BUS.
    interface()->send("*TRG");
}
void
XAgilent33250A::open() {
    interface()->send("*CLS");
    XString __func, __trigsrc;
    bool __burst = false;
    double __freq, __ampl, __offset, __duty = 50.0, __period = 0.0, __width = 0.0, __burstphase;
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
    //33250A: query only the square duty cycle (it has no FUNC:PULSe:DCYCle); the stored
    //value is returned regardless of the active function. Guard against any query hiccup so
    //a single read never fails the whole connection.
    try {
        interface()->query("FUNC:SQU:DCYC?");
        __duty = interface()->toDouble();
    }
    catch (XKameError &) {
        __duty = 50.0;
    }
    //Node conventions: PulseWidth 0 = specify by Duty; PulsePeriod 0 = follow Freq.
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

    //clear any error queued during the read-back so the front-panel ERR starts clean.
    interface()->send("*CLS");

    start();
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
    bool is_burst = shot[ *burst()];
    //Do NOT use APPLy here: APPLy forces OUTPUT ON and emits a *continuous* waveform, which
    //leaks to whatever the output drives (here a laser) during reconfiguration — the
    //spurious bursts seen after each exposure. Use the individual FUNC/FREQ/VOLT commands
    //instead (they do not turn the output on). Order matters: setting FUNC clears burst on
    //the 3390, so stage the waveform FIRST and (re-)arm the burst LAST. As long as the
    //output is OFF during reconfiguration (the caller turns it on afterwards), nothing is
    //emitted; once armed, the generator emits only on a trigger.
    interface()->sendf("FUNC %s", shot[ *waveform()].to_str().c_str());
    interface()->sendf("FREQ %g", (double)shot[ *freq()]);
    interface()->sendf("VOLT %g", (double)shot[ *ampl()]);
    interface()->sendf("VOLT:OFFSET %g", (double)shot[ *offset()]);
    interface()->sendf("FUNC:SQU:DCYC %g", (double)shot[ *duty()]);
    double period = shot[ *pulsePeriod()];
    if(period > 0)
        interface()->sendf("PULSE:PER %g", period); //overrides period given by 1/Freq
    double width = shot[ *pulseWidth()];
    if(width > 0)
        interface()->sendf("FUNC:PULSE:WIDTH %g", width); //width and duty are exclusive; width takes over
    else
        interface()->sendf("FUNC:PULSE:DCYC %g", (double)shot[ *duty()]); //width = 0: specify by duty
    if(is_burst) {
        interface()->sendf("BURS:PHAS %g", (double)shot[ *burstPhase()]);
        unsigned int cyc = shot[ *burstCycles()];
        if(cyc == 0)
            interface()->send("BURS:NCYC INF");
        else
            interface()->sendf("BURS:NCYC %u", cyc);
        interface()->send("TRIG:SOUR " + shot[ *trigSrc()].to_str());
        interface()->send("BURST:STAT ON"); //arm LAST: armed, no output until a trigger
        if(shot[ *output()] && (cyc == 0) && (shot[ *trigSrc()].to_str() == "BUS"))
            interface()->send("*OPC;*TRG"); //continuous(INF) BUS burst: issue a trigger
    }
    else {
        interface()->send("BURST:STAT OFF");
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

