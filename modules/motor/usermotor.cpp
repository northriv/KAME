/***************************************************************************
        Copyright (C) 2002-2024 Kentaro Kitagawa
                           kitag@issp.u-tokyo.ac.jp

        This program is free software; you can redistribute it and/or
        modify it under the terms of the GNU Library General Public
        License as published by the Free Software Foundation; either
        version 2 of the License, or (at your option) any later version.

        You should have received a copy of the GNU Library General
        Public License and a list of authors along with this program;
        see the files COPYING and AUTHORS.
***************************************************************************/

#include "usermotor.h"
//---------------------------------------------------------------------------

REGISTER_TYPE(XDriverList, FlexCRK, "OrientalMotor FLEX CRK motor controller");
REGISTER_TYPE(XDriverList, OrientalMotorCVD2B, "OrientalMotor CVD2B motor controller");
REGISTER_TYPE(XDriverList, OrientalMotorCVD5B, "OrientalMotor CVD5B motor controller");
REGISTER_TYPE(XDriverList, FlexAR, "OrientalMotor FLEX AR/DG2 motor controller");
REGISTER_TYPE(XDriverList, EMP401, "OrientalMotor EMP401 motor controller");
REGISTER_TYPE(XDriverList, SigmaPAMC104, "SigmaOptics PAMC-104 piezo-assited motor controller");

const std::vector<uint32_t> XOrientalMotorCVD2B::s_resolutions_2B = {200,400,800,1000,1600,2000,3200,5000,6400,10000,12800,20000,25000,25600,50000,51200};
const std::vector<uint32_t> XOrientalMotorCVD2B::s_resolutions_5B = {500,1000,1250,2000,2500,4000,5000,10000,12500,20000,25000,40000,50000,62500, 100000, 125000};

XOrientalMotorCVD2B::XOrientalMotorCVD2B(const char *name, bool runtime,
    Transaction &tr_meas, const shared_ptr<XMeasure> &meas) :
    XModbusRTUDriver<XMotorDriver>(name, runtime, ref(tr_meas), meas) {
    interface()->setSerialBaudRate(115200);
    interface()->setSerialStopBits(1);
    interface()->setSerialParity(XCharInterface::PARITY_EVEN);
    pushing()->disable();
    round()->disable();
    roundBy()->disable();
}
void
XOrientalMotorCVD2B::storeToROM() {
    XScopedLock<XInterface> lock( *interface());
    interface()->presetTwoResistors(0x192, 1); //RAM to NV.
    interface()->presetTwoResistors(0x192, 0);
}
void
XOrientalMotorCVD2B::clearPosition() {
    XScopedLock<XInterface> lock( *interface());
    if(static_cast<int32_t>(interface()->readHoldingTwoResistors(0xc6)) == 0) {
        //ugly hack
        //already 0 pos, to HOME
//        interface()->presetSingleResistor(0x7d, 0x0010u); //HOME
//        interface()->presetSingleResistor(0x7d, 0x0000u);
    }
    else {
        interface()->presetTwoResistors(0x018a, 1); //counter clear.
        interface()->presetTwoResistors(0x018a, 0);
    }
}
void
XOrientalMotorCVD2B::getStatus(const Snapshot &shot, double *position, bool *slipping, bool *ready) {
    XScopedLock<XInterface> lock( *interface());
    uint32_t output = interface()->readHoldingTwoResistors(0x178);
//    *slipping = output & 0x20u;
    if(output & 0x2) { //ALM-A
        uint16_t alarm = interface()->readHoldingTwoResistors(0x80);
        gErrPrint(getLabel() + i18n(" Alarm %1 has been emitted").arg((int)alarm));
        interface()->presetTwoResistors(0x180, 1); //clears alarm.
        interface()->presetTwoResistors(0x180, 0);
    }
    if(output & 0x80) { //INFO
        uint16_t alarm = interface()->readHoldingTwoResistors(0xf6);
        if(alarm) {
            gErrPrint(getLabel() + i18n(" Info %1 has been emitted").arg((int)alarm));
            interface()->presetTwoResistors(0x1a6, 1); //clears info.
            interface()->presetTwoResistors(0x1a6, 0);
        }
    }
    *ready = (output & 0x10u);
//    if(shot[ *hasEncoder()])
//        *position = static_cast<int32_t>(interface()->readHoldingTwoResistors(0xc6))
//            * 360.0 / (double)shot[ *stepEncoder()];
//    else
        *position = static_cast<int32_t>(interface()->readHoldingTwoResistors(0xc6))
            * 360.0 / (double)shot[ *stepMotor()];
}
void
XOrientalMotorCVD2B::changeConditions(const Snapshot &shot) {
    XScopedLock<XInterface> lock( *interface());
    interface()->presetTwoResistors(0x24c,  lrint(shot[ *currentRunning()] * 10));
    interface()->presetTwoResistors(0x180a,  lrint(shot[ *currentRunning()] * 10));
    interface()->presetTwoResistors(0x250,  lrint(shot[ *currentStopping()] * 10));
    switch(interface()->readHoldingTwoResistors(0x28e)) {
    case 0:
        //kHz/s
        interface()->presetTwoResistors(0x280,  lrint((1e3  / shot[ *timeAcc()]) * 1e3));
        interface()->presetTwoResistors(0x282,  lrint((1e3 / shot[ *timeDec()]) * 1e3));
        break;
    case 1:
        //s
        break;
    case 2:
        //ms/kHz
        interface()->presetTwoResistors(0x280,  lrint(shot[ *timeAcc()] * 1e3));
        interface()->presetTwoResistors(0x282,  lrint(shot[ *timeDec()] * 1e3));
    }
    interface()->presetTwoResistors(0x28c,  0); //common setting

//    interface()->presetTwoResistors(0x312,  lrint((double)shot[ *stepEncoder()]));
    bool phase2 = interface()->readHoldingTwoResistors(0x39c) == 0;
    if(interface()->readHoldingTwoResistors(0x39c) >= 2)
        phase2 = isPresetTo2Phase();
    auto &resolutions = phase2 ? s_resolutions_2B : s_resolutions_5B;
    auto it = std::lower_bound(resolutions.begin(), resolutions.end(), lrint((double)shot[ *stepMotor()] * 1.02));
    interface()->presetTwoResistors(0x39e,  it - resolutions.begin());
    interface()->presetTwoResistors(0x1804,  lrint(shot[ *speed()]));
    unsigned int microstep = shot[ *microStep()] ? 1 : 0;
    //    if(interface()->readHoldingTwoResistors(0x258) != microstep) {
    interface()->presetTwoResistors(0x258, microstep);
    //    }
    if(interface()->readHoldingSingleResistor(0x17e) & 0x4000u) { //INFO-CFG
        gWarnPrint(i18n("Store settings to NV memory and restart."));
        interface()->presetTwoResistors(0x018c, 1); //configuration.
        interface()->presetTwoResistors(0x018c, 0);
    }
}
void
XOrientalMotorCVD2B::getConditions() {
    double crun, cstop, mstep, tacc = 0, tdec = 0, senc = 0, smotor, spd, tgt;
    bool atv;
    {
        XScopedLock<XInterface> lock( *interface());
        interface()->diagnostics();
        crun = interface()->readHoldingTwoResistors(0x24c) * 0.1;
        cstop = interface()->readHoldingTwoResistors(0x250) * 0.1;
        mstep = (interface()->readHoldingTwoResistors(0x258) != 0);
        switch(interface()->readHoldingTwoResistors(0x28e)) {
        case 0:
            //kHz/s
            tacc = 1e3 / (interface()->readHoldingTwoResistors(0x280) * 1e-3);
            tdec = 1e3 / (interface()->readHoldingTwoResistors(0x282) * 1e-3);
            break;
        case 1:
            //s
            break;
        case 2:
            //ms/kHz
            tacc = interface()->readHoldingTwoResistors(0x280) * 1e-3;
            tdec = interface()->readHoldingTwoResistors(0x282) * 1e-3;
        }
//        senc = interface()->readHoldingTwoResistors(0x312);
        bool phase2 = interface()->readHoldingTwoResistors(0x39c) == 0;
        if(interface()->readHoldingTwoResistors(0x39c) >= 2)
            phase2 = isPresetTo2Phase();
        auto &resolutions = phase2 ? s_resolutions_2B : s_resolutions_5B;
        try {
            smotor = resolutions.at(interface()->readHoldingTwoResistors(0x39e));
        }
        catch(std::out_of_range &) {
            smotor = 0;
        }
        spd = interface()->readHoldingTwoResistors(0x1804);
        tgt = static_cast<int32_t>(interface()->readHoldingTwoResistors(0x1802)) * 360.0 / smotor;
        atv = (interface()->readHoldingSingleResistor(0x7d) & 0x40u) == 0; //not AWC
    }
    iterate_commit([=](Transaction &tr){
        tr[ *currentRunning()] = crun;
        tr[ *currentStopping()] = cstop;
        tr[ *microStep()] = mstep;
        tr[ *timeAcc()] = tacc;
        tr[ *timeDec()] = tdec;
        tr[ *stepEncoder()] = senc;
        tr[ *stepMotor()] = smotor;
        tr[ *speed()] = spd;
        tr[ *target()] = tgt;
        tr[ *round()].setUIEnabled(false);
        tr[ *roundBy()].setUIEnabled(false);
        tr[ *pushing()].setUIEnabled(false);
        tr[ *active()] = atv;
    });
}
void
XOrientalMotorCVD2B::stopRotation() {
    sendStopSignal(false);
}
void
XOrientalMotorCVD2B::sendStopSignal(bool wait) {
    for(int i = 0;; ++i) {
        {
            XScopedLock<XInterface> lock( *interface());
            uint32_t output = interface()->readHoldingTwoResistors(0x178);
            bool isready = (output & 0x10u);
            if(isready) break;
            if(i == 0) {
                uint32_t netin = interface()->readHoldingTwoResistors(0x7c);
                interface()->presetTwoResistors(0x7c, (netin & ~0xc020u) | 0x20u); //STOP
                interface()->presetTwoResistors(0x7c, netin & ~0xc020u);
                if( !wait)
                    break;
            }
        }
        msecsleep(100);
        if(i > 10) {
            throw XInterface::XInterfaceError(i18n("Motor is still not ready"), __FILE__, __LINE__);
        }
    }
}
void
XOrientalMotorCVD2B::toHomePosition() {
    XScopedLock<XInterface> lock( *interface());
    uint32_t netin = interface()->readHoldingTwoResistors(0x7c);
    interface()->presetTwoResistors(0x7c, (netin & ~0xc000u) | 0x0010u); //HOME
    interface()->presetTwoResistors(0x7c, netin & ~0xc0010u);
}
void
XOrientalMotorCVD2B::setForward() {
    XScopedLock<XInterface> lock( *interface());
    uint32_t netin = interface()->readHoldingTwoResistors(0x7c);
    interface()->presetTwoResistors(0x7c, (netin & ~0xc000u) | 0x4000u); //FW-POS
}
void
XOrientalMotorCVD2B::setReverse() {
    XScopedLock<XInterface> lock( *interface());
    uint32_t netin = interface()->readHoldingTwoResistors(0x7c);
    interface()->presetTwoResistors(0x7c, (netin & ~0xc000u) | 0x8000u); //RV-POS
}
void
XOrientalMotorCVD2B::setTarget(const Snapshot &shot, double target) {
    XScopedLock<XInterface> lock( *interface());
    sendStopSignal(true);
    interface()->presetTwoResistors(0x1800, 1); //absolute pos.
    interface()->presetTwoResistors(0x1802, lrint(target / 360.0 * shot[ *stepMotor()]));
    uint32_t netin = interface()->readHoldingTwoResistors(0x7c);
    interface()->presetTwoResistors(0x7c, (netin & ~0xc008u) | 0x08u); //START
    interface()->presetTwoResistors(0x7c, netin & ~0xc008u);
}
void
XOrientalMotorCVD2B::setActive(bool active) {
    XScopedLock<XInterface> lock( *interface());
    uint32_t netin = interface()->readHoldingTwoResistors(0x7c);
    if( !active) {
        sendStopSignal(true);
        interface()->presetTwoResistors(0x7c, (netin & ~0xc040u) | 0x40u); //AWO
    }
    else {
        interface()->presetTwoResistors(0x7c, netin & ~0x40u);
    }
}
void
XOrientalMotorCVD2B::setAUXBits(unsigned int bits) {
    XScopedLock<XInterface> lock( *interface());

    //routing R-IN8,9 to DOUT0,1
    interface()->presetTwoResistors(0x1210, 80); //RIN8 to R0_R
    interface()->presetTwoResistors(0x1212, 81); //RIN9 to R1_R
    interface()->presetTwoResistors(0x10C0, 80); //DOUT0 to R0_R
    interface()->presetTwoResistors(0x10C2, 81); //DOUT1 to R1_R
    if(interface()->readHoldingSingleResistor(0x17e) & 0x4000u) { //INFO-CFG
        gWarnPrint(i18n("Configuration issued."));
        interface()->presetTwoResistors(0x018c, 1); //configuration.
        interface()->presetTwoResistors(0x018c, 0);
    }
    uint32_t netin = interface()->readHoldingTwoResistors(0x7c);
    interface()->presetTwoResistors(0x7c, (netin & ~0x300uL) | ((bits & 0x03uL) * 0x100uL)); //R-IN8-10 as R0,1,(2)
}

XFlexCRK::XFlexCRK(const char *name, bool runtime,
    Transaction &tr_meas, const shared_ptr<XMeasure> &meas) :
    XModbusRTUDriver<XMotorDriver>(name, runtime, ref(tr_meas), meas) {
    interface()->setSerialBaudRate(57600);
    interface()->setSerialStopBits(1);
    interface()->setSerialParity(XCharInterface::PARITY_EVEN);
}
void
XFlexCRK::storeToROM() {
    XScopedLock<XInterface> lock( *interface());
    interface()->presetSingleResistor(0x45, 1); //RAM to NV.
    interface()->presetSingleResistor(0x45, 0);
}
void
XFlexCRK::clearPosition() {
    XScopedLock<XInterface> lock( *interface());
    interface()->presetSingleResistor(0x4b, 1); //counter clear.
    interface()->presetSingleResistor(0x4b, 0);
}
void
XFlexCRK::getStatus(const Snapshot &shot, double *position, bool *slipping, bool *ready) {
    XScopedLock<XInterface> lock( *interface());
    uint32_t output = interface()->readHoldingTwoResistors(0x20); //reading status1:status2
    *slipping = output & 0x2000000u;
    if(output & 0x80) {
        uint16_t alarm = interface()->readHoldingSingleResistor(0x100);
        gErrPrint(getLabel() + i18n(" Alarm %1 has been emitted").arg((int)alarm));
        interface()->presetSingleResistor(0x40, 1); //clears alarm.
        interface()->presetSingleResistor(0x40, 0);
    }
    if(output & 0x40) {
        uint16_t warn = interface()->readHoldingSingleResistor(0x10b);
        gWarnPrint(getLabel() + i18n(" Code = %1").arg((int)warn));
    }
//	uint32_t ierr = interface()->readHoldingTwoResistors(0x128);
//	if(ierr) {
//		gErrPrint(getLabel() + i18n(" Interface error %1 has been emitted").arg((int)ierr));
//	}
    *ready = (output & 0x20000000u);
//	fprintf(stderr, "0x20:%x\n", (unsigned int)output);
    if(shot[ *hasEncoder()])
        *position = static_cast<int32_t>(interface()->readHoldingTwoResistors(0x11e))
            * 360.0 / (double)shot[ *stepEncoder()];
    else
        *position = static_cast<int32_t>(interface()->readHoldingTwoResistors(0x118))
            * 360.0 / (double)shot[ *stepMotor()];
}
void
XFlexCRK::changeConditions(const Snapshot &shot) {
    XScopedLock<XInterface> lock( *interface());
    interface()->presetSingleResistor(0x21e,  lrint(shot[ *currentRunning()]));
    interface()->presetSingleResistor(0x21f,  lrint(shot[ *currentStopping()]));
    interface()->presetSingleResistor(0x236, 0); //common setting for acc/dec.
    interface()->presetTwoResistors(0x224,  lrint(shot[ *timeAcc()] * 1e3));
    interface()->presetTwoResistors(0x226,  lrint(shot[ *timeDec()] * 1e3));
    interface()->presetTwoResistors(0x312,  lrint((double)shot[ *stepEncoder()]));
    interface()->presetTwoResistors(0x314,  lrint((double)shot[ *stepMotor()]));
    interface()->presetTwoResistors(0x502,  lrint(shot[ *speed()]));
    unsigned int microstep = shot[ *microStep()] ? 6 : 0;
    if(interface()->readHoldingSingleResistor(0x311) != microstep) {
        gWarnPrint(i18n("Store settings to NV memory and restart, microstep div.=10."));
        interface()->presetSingleResistor(0x311, microstep); //division = 10.
    }
}
void
XFlexCRK::getConditions() {
    double crun, cstop, mstep, tacc, tdec, senc, smotor, spd, tgt;
    bool atv;
    {
        XScopedLock<XInterface> lock( *interface());
        interface()->diagnostics();
        crun = interface()->readHoldingSingleResistor(0x21e);
        cstop = interface()->readHoldingSingleResistor(0x21f);
        mstep = (interface()->readHoldingSingleResistor(0x311) != 0);
        tacc = interface()->readHoldingTwoResistors(0x224) * 1e-3;
        tdec = interface()->readHoldingTwoResistors(0x226) * 1e-3;
        senc = interface()->readHoldingTwoResistors(0x312);
        smotor = interface()->readHoldingTwoResistors(0x314);
        spd = interface()->readHoldingTwoResistors(0x502);
        tgt = static_cast<int32_t>(interface()->readHoldingTwoResistors(0x402)) * 360.0 / smotor;
        atv = (interface()->readHoldingSingleResistor(0x1e) & 0x2000u) == 1; //CON
        interface()->presetSingleResistor(0x203, 0); //STOP I/O normally open.
        interface()->presetSingleResistor(0x200, 0); //START by RS485.
        interface()->presetSingleResistor(0x20b, 0); //C-ON by RS485.
        interface()->presetSingleResistor(0x20c, 0); //HOME/FWD/RVS by RS485.
        interface()->presetSingleResistor(0x20d, 0); //No. by RS485.
        interface()->presetSingleResistor(0x202, 1); //Dec. after STOP.
        interface()->presetSingleResistor(0x601, 1); //Absolute.
    }
    iterate_commit([=](Transaction &tr){
        tr[ *currentRunning()] = crun;
        tr[ *currentStopping()] = cstop;
        tr[ *microStep()] = mstep;
        tr[ *timeAcc()] = tacc;
        tr[ *timeDec()] = tdec;
        tr[ *stepEncoder()] = senc;
        tr[ *stepMotor()] = smotor;
        tr[ *speed()] = spd;
        tr[ *target()] = tgt;
        tr[ *round()].setUIEnabled(false);
        tr[ *roundBy()].setUIEnabled(false);
        tr[ *pushing()].setUIEnabled(false);
        tr[ *active()] = atv;
    });
}
void
XFlexCRK::stopRotation() {
    sendStopSignal(false);
}
void
XFlexCRK::sendStopSignal(bool wait) {
    for(int i = 0;; ++i) {
        {
            XScopedLock<XInterface> lock( *interface());
            uint32_t output = interface()->readHoldingTwoResistors(0x20); //reading status1:status2
            bool isready = (output & 0x20000000u);
            if(isready) break;
            if(i ==0) {
                interface()->presetSingleResistor(0x1e, 0x3001u); //C-ON, STOP, M0
                interface()->presetSingleResistor(0x1e, 0x2001u); //C-ON, M0
                if( !wait)
                    break;
            }
        }
        msecsleep(100);
        if(i > 10) {
            throw XInterface::XInterfaceError(i18n("Motor is still not ready"), __FILE__, __LINE__);
        }
    }
}
void
XFlexCRK::toHomePosition() {
    XScopedLock<XInterface> lock( *interface());
    interface()->presetSingleResistor(0x1e, 0x2800u); //C-ON, HOME
    interface()->presetSingleResistor(0x1e, 0x2000u); //C-ON
}
void
XFlexCRK::setForward() {
    XScopedLock<XInterface> lock( *interface());
    interface()->presetSingleResistor(0x1e, 0x2201u); //C-ON, FWD, M0
}
void
XFlexCRK::setReverse() {
    XScopedLock<XInterface> lock( *interface());
    interface()->presetSingleResistor(0x1e, 0x2401u); //C-ON, RVS, M0
}
void
XFlexCRK::setTarget(const Snapshot &shot, double target) {
    XScopedLock<XInterface> lock( *interface());
    sendStopSignal(true);
    interface()->presetTwoResistors(0x402, lrint(target / 360.0 * shot[ *stepMotor()]));
    interface()->presetSingleResistor(0x1e, 0x2101u); //C-ON, START, M0
    interface()->presetSingleResistor(0x1e, 0x2001u); //C-ON, M0
}
void
XFlexCRK::setActive(bool active) {
    XScopedLock<XInterface> lock( *interface());
    if(active) {
        interface()->presetSingleResistor(0x1e, 0x2001u); //C-ON, M0
    }
    else {
        sendStopSignal(true);
        interface()->presetSingleResistor(0x1e, 0x0001u); //M0
    }
}
void
XFlexCRK::setAUXBits(unsigned int bits) {
    interface()->presetSingleResistor(0x206, 11); //OUT1 to R-OUT1
    interface()->presetSingleResistor(0x207, 12); //OUT2 to R-OUT2
    interface()->presetSingleResistor(0x208, 15); //OUT3 to R-OUT3
    interface()->presetSingleResistor(0x209, 16); //OUT4 to R-OUT4
    interface()->presetSingleResistor(0x1f, bits & 0xfu);
}

void
XFlexAR::storeToROM() {
    XScopedLock<XInterface> lock( *interface());
    interface()->presetTwoResistors(0x192, 1); //RAM to NV.
    interface()->presetTwoResistors(0x192, 0);
}
void
XFlexAR::clearPosition() {
    XScopedLock<XInterface> lock( *interface());
    interface()->presetTwoResistors(0x18a, 1); //counter clear.
    interface()->presetTwoResistors(0x18a, 0);
}
void
XFlexAR::getStatus(const Snapshot &shot, double *position, bool *slipping, bool *ready) {
    XScopedLock<XInterface> lock( *interface());
    uint32_t output = interface()->readHoldingTwoResistors(0x7e);
    *ready = output & 0x20;
    *slipping = output & 0x8000;
    if(output & 0x80) {
        uint32_t alarm = interface()->readHoldingTwoResistors(0x80);
        gErrPrint(getLabel() + i18n(" Alarm %1 has been emitted").arg((int)alarm));
        interface()->presetTwoResistors(0x180, 1); //clears alarm.
        interface()->presetTwoResistors(0x180, 0);
        interface()->presetTwoResistors(0x182, 1); //clears abs. pos. alarm.
        interface()->presetTwoResistors(0x182, 0);
        interface()->presetTwoResistors(0x184, 1); //clears alarm history.
        interface()->presetTwoResistors(0x184, 0);
    }
    if(output & 0x40) {
        uint32_t warn = interface()->readHoldingTwoResistors(0x96);
        gWarnPrint(getLabel() + i18n(" Code = %1").arg((int)warn));
    }
    if(shot[ *hasEncoder()])
        *position = static_cast<int32_t>(interface()->readHoldingTwoResistors(0xcc))
            * 360.0 / (double)shot[ *stepEncoder()];
    else
        *position = static_cast<int32_t>(interface()->readHoldingTwoResistors(0xc6))
            * 360.0 / (double)shot[ *stepMotor()];
}
void
XFlexAR::changeConditions(const Snapshot &shot) {
    XScopedLock<XInterface> lock( *interface());
    interface()->presetTwoResistors(0x200,  1); //slowing down on STOP
    interface()->presetTwoResistors(0x240,  lrint(shot[ *currentRunning()] * 10.0));
    interface()->presetTwoResistors(0x1380,  lrint(shot[ *currentRunning()] * 10.0)); //pushing mode
    interface()->presetTwoResistors(0x242,  lrint(shot[ *currentStopping()] * 10.0));
    interface()->presetTwoResistors(0x28c, 0); //common setting for acc/dec.
    interface()->presetTwoResistors(0x280,  lrint(shot[ *timeAcc()] * 1e3));
    interface()->presetTwoResistors(0x282,  lrint(shot[ *timeDec()] * 1e3));
    interface()->presetTwoResistors(0x284,  0); //starting speed.
    interface()->presetTwoResistors(0x480,  lrint(shot[ *speed()]));
    interface()->presetTwoResistors(0x580, shot[ *pushing()] ? 3 : 0);

    bool conf_needed = false;
    //electric gear
    {
        int smotor = lrint(interface()->readHoldingTwoResistors(0x382) * 1000.0 /  interface()->readHoldingTwoResistors(0x380));
        int a = 1000;
        int b = shot[ *stepMotor()];
        if(b != smotor) {
            conf_needed = true;
            interface()->presetTwoResistors(0x380,  a); //A
            interface()->presetTwoResistors(0x382,  b); //B, rot=1000B/A
        }
    }
    interface()->presetTwoResistors(0x1002, shot[ *stepEncoder()] / shot[ *stepMotor()]); //Multiplier is stored in MS2 No.
    int b_micro = shot[ *microStep()] ? 1 : 0;
    if(interface()->readHoldingTwoResistors(0x1028) != b_micro) {
        conf_needed = true;
        interface()->presetTwoResistors(0x1028, b_micro);
    }
    int b_round = shot[ *round()] ? 1 : 0;
    if(interface()->readHoldingTwoResistors(0x38e) != b_round) {
        conf_needed = true;
        interface()->presetTwoResistors(0x38e,  b_round);
    }
    int num_round = std::max(lrint((double)shot[ *roundBy()]), 1L);
    if(b_round && (interface()->readHoldingTwoResistors(0x390) != num_round)) {
        conf_needed = true;
        interface()->presetTwoResistors(0x390,  num_round);
        interface()->presetTwoResistors(0x20a,  lrint((double)shot[ *roundBy()]) / 2); //AREA1+
        interface()->presetTwoResistors(0x20c,  0); //AREA1-
    }

    if(conf_needed) {
//        sendStopSignal(true);
        setActive(false);
        interface()->presetTwoResistors(0x18c, 1);
        interface()->presetTwoResistors(0x18c, 0);
    }
}
void
XFlexAR::getConditions() {
    double crun, cstop, mstep, tacc, tdec, senc, smotor, spd, tgt, rnd, rndby;
    bool atv, push;
    {
        XScopedLock<XInterface> lock( *interface());
        interface()->diagnostics();
        crun = interface()->readHoldingTwoResistors(0x240) * 0.1;
        cstop = interface()->readHoldingTwoResistors(0x242) * 0.1;
        mstep = (interface()->readHoldingTwoResistors(0x1028) != 0);
        tacc = interface()->readHoldingTwoResistors(0x280) * 1e-3;
        tdec = interface()->readHoldingTwoResistors(0x282) * 1e-3;
        //EGear 1000B/A
        smotor = lrint(interface()->readHoldingTwoResistors(0x382) * 1000.0 /  interface()->readHoldingTwoResistors(0x380));
        senc = smotor * interface()->readHoldingTwoResistors(0x1002); //Multiplier is stored in MS2 No.
        spd = interface()->readHoldingTwoResistors(0x480);
        tgt = static_cast<int32_t>(interface()->readHoldingTwoResistors(0x400)) * 360.0 / smotor;
        rnd = (interface()->readHoldingTwoResistors(0x38e) == 1);
        rndby = interface()->readHoldingTwoResistors(0x390);
        atv = (interface()->readHoldingTwoResistors(0x7c) & 0x40u) == 0; //FREE
        push = interface()->readHoldingTwoResistors(0x580) == 3; //Pushing Mode or not
        interface()->presetTwoResistors(0x500, 1); //Absolute.
        interface()->presetTwoResistors(0x119e, 71); //NET-OUT15 = TLC
        interface()->presetTwoResistors(0x1140, 32); //OUT0 to R0
        interface()->presetTwoResistors(0x1142, 33); //OUT1 to R1
    //	interface()->presetTwoResistors(0x1144, 34); //OUT2 to R2
        interface()->presetTwoResistors(0x1146, 35); //OUT3 to R3
        interface()->presetTwoResistors(0x1148, 36); //OUT4 to R4
        interface()->presetTwoResistors(0x114a, 37); //OUT5 to R5
        interface()->presetTwoResistors(0x1160, 32); //NET-IN0 to R0
        interface()->presetTwoResistors(0x1162, 33); //NET-IN1 to R1
    //	interface()->presetTwoResistors(0x1164, 34); //NET-IN2 to R2
        interface()->presetTwoResistors(0x1166, 35); //NET-IN3 to R3
        interface()->presetTwoResistors(0x1168, 36); //NET-IN4 to R4
////        interface()->presetTwoResistors(0x116a, 37); //NET-IN5 to R5
//        interface()->presetTwoResistors(0x1160, 48); //NET-IN0 to M0
//        interface()->presetTwoResistors(0x1162, 49); //NET-IN1 to M1
//        interface()->presetTwoResistors(0x1164, 50); //NET-IN2 to M2
//        interface()->presetTwoResistors(0x1166, 4); //NET-IN3 to START
//        interface()->presetTwoResistors(0x1168, 3); //NET-IN4 to HOME
        interface()->presetTwoResistors(0x116a, 18); //NET-IN5 to STOP
    }
    iterate_commit([=](Transaction &tr){
        tr[ *currentRunning()] = crun;
        tr[ *currentStopping()] = cstop;
        tr[ *microStep()] = mstep;
        tr[ *timeAcc()] = tacc;
        tr[ *timeDec()] = tdec;
        tr[ *stepMotor()] = smotor;
        tr[ *stepEncoder()] = senc;
        tr[ *speed()] = spd;
        tr[ *target()] = tgt;
        tr[ *hasEncoder()] = true;
        tr[ *pushing()] = push;
        tr[ *round()] = rnd;
        tr[ *roundBy()] = rndby;
        tr[ *active()] = atv;
    });
}
void
XFlexAR::setTarget(const Snapshot &shot, double target) {
    XScopedLock<XInterface> lock( *interface());
    sendStopSignal(false);
    int steps = shot[ *hasEncoder()] ? shot[ *stepEncoder()] : shot[ *stepMotor()];
    interface()->presetTwoResistors(0x400, lrint(target / 360.0 * steps));
    uint32_t netin = interface()->readHoldingTwoResistors(0x7c);
    netin &= ~(0x4000uL | 0x8000uL | 0x20uL); //FWD | RVS | STOP
    interface()->presetTwoResistors(0x7c, netin);
    msecsleep(4);
//    interface()->presetTwoResistors(0x7c, netin | 0x8uL); //START
    interface()->presetTwoResistors(0x7c, netin | 0x100uL); //MS0
    msecsleep(4);
//    interface()->presetTwoResistors(0x7c, netin & ~0x8uL);
    interface()->presetTwoResistors(0x7c, netin & ~0x100uL);
}
void
XFlexAR::prepairSequence(const std::vector<double> &points, const std::vector<double> &speeds) {
    Snapshot shot( *this);
    sendStopSignal(false);
    int steps = shot[ *hasEncoder()] ? shot[ *stepEncoder()] : shot[ *stepMotor()];
    if(points.size() > 63) {
        throw XInterface::XInterfaceError(getLabel() +
            i18n(": Too many points."), __FILE__, __LINE__);
    }
    uint32_t acc, dec;
    acc = interface()->readHoldingTwoResistors(0x600);
    dec = interface()->readHoldingTwoResistors(0x680);
    for(unsigned int i = 0; i < points.size(); ++i) {
        interface()->presetTwoResistors(0x482 + 2 * i,  lrint(speeds[i]));
        interface()->presetTwoResistors(0x402 + 2 * i, lrint(points[i] / 360.0 * steps));
        interface()->presetTwoResistors(0x502 + 2 * i,  1); //absolute
        interface()->presetTwoResistors(0x602 + 2 * i,  acc);
        interface()->presetTwoResistors(0x682 + 2 * i,  dec);
        interface()->presetTwoResistors(0x582 + 2 * i,  1); //sequential
    }
    interface()->presetTwoResistors(0x582 + 2 * (points.size() - 1),  0);
    interface()->presetTwoResistors(0x1002,  1); //MS1 -> No.1--
}

void
XFlexAR::runSequentially(const std::vector<std::vector<double>> &points,
                         const std::vector<std::vector<double>> &speeds, const std::vector<shared_ptr<XMotorDriver>> &slaves) {
    XScopedLock<XInterface> lock( *interface());
    if( !interface()->isOpened())
        throw XInterface::XInterfaceError(getLabel() +
            i18n(": not opened."), __FILE__, __LINE__);
    Snapshot shot( *this);
    prepairSequence(points[0], speeds[0]);
    int addr_master = shot[ *interface()->address()];
    for(unsigned int i = 0; i < slaves.size(); ++i) {
        auto slave = dynamic_pointer_cast<XFlexAR>(slaves[i]);
        if( !slave)
            throw XInterface::XInterfaceError(getLabel() +
                i18n(": Different motor driver."), __FILE__, __LINE__);
        slave->prepairSequence(points[i + 1], speeds[i + 1]);
        slave->interface()->presetTwoResistors(0x30,  addr_master); //grouped
    }
    //ignites.
    uint32_t netin = interface()->readHoldingTwoResistors(0x7c);
    netin &= ~(0x4000uL | 0x8000uL | 0x20uL); //FWD | RVS | STOP
    interface()->presetTwoResistors(0x7c, netin);
    msecsleep(4);
    interface()->presetTwoResistors(0x7c, netin | 0x200uL); //MS1
    msecsleep(4);
    interface()->presetTwoResistors(0x7c, netin & ~0x200uL);
    for(unsigned int i = 0; i < slaves.size(); ++i) {
        auto slave = dynamic_pointer_cast<XFlexAR>(slaves[i]);
        slave->interface()->presetTwoResistors(0x30,  -1); //ungroup
    }
}
void
XFlexAR::stopRotation() {
     sendStopSignal(false);
}
void
XFlexAR::sendStopSignal(bool wait) {
    for(int i = 0;; ++i) {
        {
            XScopedLock<XInterface> lock( *interface());
            uint32_t output = interface()->readHoldingTwoResistors(0x7e);
            bool isready = output & 0x20;
            if(isready) break;
            if(i ==0) {
                uint32_t netin = interface()->readHoldingTwoResistors(0x7c);
                netin &= ~(0x4000uL | 0x8000u); //FWD | RVS
                interface()->presetTwoResistors(0x7c, netin | 0x20uL); //STOP
    //            fprintf(stderr, "STOP%u\n", netin);
                msecsleep(4);
                interface()->presetTwoResistors(0x7c, netin & ~0x20uL);
                if( !wait)
                    break;
            }
        }
        msecsleep(150);
        if(i > 10) {
            throw XInterface::XInterfaceError(i18n("Motor is still not ready"), __FILE__, __LINE__);
        }
    }
}
void
XFlexAR::toHomePosition() {
    XScopedLock<XInterface> lock( *interface());
    uint32_t netin = interface()->readHoldingTwoResistors(0x7c);
    netin &= ~(0x4000uL | 0x8000uL | 0x20uL); //FWD | RVS | STOP
    interface()->presetTwoResistors(0x7c, netin | 0x0010uL); //HOME
    interface()->presetTwoResistors(0x7c, netin);
}
void
XFlexAR::setForward() {
    XScopedLock<XInterface> lock( *interface());
    uint32_t netin = interface()->readHoldingTwoResistors(0x7c);
    netin &= ~(0x4000uL | 0x8000uL | 0x20uL); //FWD | RVS | STOP
    interface()->presetTwoResistors(0x7c, netin | 0x4000uL); //FWD
}
void
XFlexAR::setReverse() {
    XScopedLock<XInterface> lock( *interface());
    uint32_t netin = interface()->readHoldingTwoResistors(0x7c);
    netin &= ~(0x4000uL | 0x8000uL | 0x20uL); //FWD | RVS | STOP
    interface()->presetTwoResistors(0x7c, netin | 0x8000uL); //RVS
}
void
XFlexAR::setActive(bool active) {
    XScopedLock<XInterface> lock( *interface());
    uint32_t netin = interface()->readHoldingTwoResistors(0x7c);
    if(active) {
        interface()->presetTwoResistors(0x7c, netin & ~0x40uL);
    }
    else {
        sendStopSignal(true);
        interface()->presetTwoResistors(0x7c, netin | 0x40uL); //FREE
    }
}
void
XFlexAR::setAUXBits(unsigned int bits) {
    XScopedLock<XInterface> lock( *interface());
    uint32_t netin = interface()->readHoldingTwoResistors(0x7c);
    if((bits < 0x20uL) || (bits == 0xffuL)) {
        interface()->presetTwoResistors(0x7c, (netin & ~0x1fuL) | (bits & 0x1fuL));
    }
//    else {
//        //debug use
//        if(bits > 0x10000uL) {
//            uint32_t addr = bits / 0x10000uL;
//            interface()->presetTwoResistors(addr, bits % 0x10000uL);
//        }
//        else {
//            uint32_t res = interface()->readHoldingTwoResistors(bits);
//            fprintf(stderr, "%x: 0x%x(%d)\n", bits, res, res);
//        }
//        fprintf(stderr, "7e: %x\n", interface()->readHoldingTwoResistors(0x7e));
//    }
}

XEMP401::XEMP401(const char *name, bool runtime,
    Transaction &tr_meas, const shared_ptr<XMeasure> &meas) :
    XCharDeviceDriver<XMotorDriver>(name, runtime, ref(tr_meas), meas) {
    interface()->setSerialBaudRate(9600);
//	interface()->setSerialStopBits(1);
    interface()->setSerialParity(XCharInterface::PARITY_NONE);
    interface()->setSerialHasEchoBack(true);
    interface()->setEOS("\r\n");
    stepEncoder()->disable();
    hasEncoder()->disable();
    timeDec()->disable();
    round()->disable();
    roundBy()->disable();
    currentRunning()->disable();
    currentStopping()->disable();
    store()->disable();
    pushing()->disable();
    goHomeMotor()->disable();
}
void
XEMP401::storeToROM() {
}
void
XEMP401::clearPosition() {
    XScopedLock<XInterface> lock( *interface());
    interface()->send("RTNCR");
    waitForCursor();
}
void
XEMP401::waitForCursor() {
    for(;;) {
        interface()->receive(1);
        if(interface()->buffer()[0] == '>')
            break;
    }
    msecsleep(10);
}
void
XEMP401::getStatus(const Snapshot &shot, double *position, bool *slipping, bool *ready) {
    XScopedLock<XInterface> lock( *interface());
    interface()->send("R");
    for(;;) {
        interface()->receive();
        int x;
        if(interface()->scanf(" PC =%d", &x) == 1) {
            *position = x; // / (double)shot[ *stepMotor()];
            break;
        }
        if(interface()->scanf(" Ready =%d", &x) == 1) {
            *ready = (x != 0);
        }
        *slipping = false;
    }
    waitForCursor();
}
void
XEMP401::changeConditions(const Snapshot &shot) {
    XScopedLock<XInterface> lock( *interface());
    interface()->queryf("T,%d", (int)lrint(shot[ *timeAcc()] * 10));
    double n2 = 1.0;
    if(shot[ *microStep()])
         n2 = 10;
    waitForCursor();
    interface()->queryf("UNIT,%.4f,%.1f", 1.0 / shot[ *stepMotor()], n2);
    waitForCursor();
    interface()->queryf("V,%d", (int)lrint(shot[ *speed()]));
    waitForCursor();
    interface()->queryf("VS,%d", (int)lrint(shot[ *speed()]));
    waitForCursor();
}
void
XEMP401::getConditions() {
    double mstep, spd, tacc, smotor;
    {
        XScopedLock<XInterface> lock( *interface());
        interface()->query("T");
        int x;
        if(interface()->scanf("%*d: T = %d", &x) != 1)
            throw XInterface::XConvError(__FILE__, __LINE__);
        tacc = x * 0.1;

        waitForCursor();
        interface()->query("V");
        if(interface()->scanf("%*d: V = %d", &x) != 1)
            throw XInterface::XConvError(__FILE__, __LINE__);
        spd = x;

        waitForCursor();
        interface()->query("UNIT");
        double n1,n2;
        if(interface()->scanf("%*d: UNIT = %lf,%lf", &n1, &n2) != 2)
            throw XInterface::XConvError(__FILE__, __LINE__);
        mstep = (n2 > 1.1);
        smotor = 1.0 / n1;
        waitForCursor();
    }

    iterate_commit([=](Transaction &tr){
        tr[ *microStep()] = mstep;
        tr[ *timeAcc()] = tacc;
        tr[ *stepMotor()] = smotor;
        tr[ *speed()] = spd;
    });
}
void
XEMP401::stopRotation() {
    XScopedLock<XInterface> lock( *interface());
    interface()->setSerialHasEchoBack(false);
    interface()->write("\x1b", 1); //ESC.
    interface()->setSerialHasEchoBack(true);
    waitForCursor();
    interface()->send("S");
    waitForCursor();
    interface()->query("ACTL 0,0,0,0");
    waitForCursor();
}
void
XEMP401::setForward() {
    XScopedLock<XInterface> lock( *interface());
    stopRotation();
    interface()->query("H,+");
    waitForCursor();
    interface()->send("SCAN");
    waitForCursor();
}
void
XEMP401::setReverse() {
    XScopedLock<XInterface> lock( *interface());
    stopRotation();
    interface()->query("H,-");
    waitForCursor();
    interface()->send("SCAN");
    waitForCursor();
}
void
XEMP401::setTarget(const Snapshot &shot, double target) {
    XScopedLock<XInterface> lock( *interface());
    stopRotation();
    interface()->queryf("D,%+.2f", target);
    waitForCursor();
    interface()->send("ABS");
    waitForCursor();
}
void
XEMP401::setActive(bool active) {
    XScopedLock<XInterface> lock( *interface());
    if(active) {
    }
    else {
        stopRotation();
    }
}
void
XEMP401::setAUXBits(unsigned int bits) {
    interface()->queryf("OUT,%1u%1u%1u%1u%1u%1u",
        (bits / 32u) % 2u, (bits / 16u) % 2u, (bits / 8u) % 2u, (bits / 4u) % 2u, (bits / 2u) % 2u, bits % 2u);
    waitForCursor();
}

XSigmaPAMC104::XSigmaPAMC104(const char *name, bool runtime,
        Transaction &tr_meas, const shared_ptr<XMeasure> &meas) :
    XSharedSerialPortDriver<XMotorDriver>(name, runtime, ref(tr_meas), meas) {
    interface()->setSerialBaudRate(115200);
    interface()->setSerialStopBits(1);
    interface()->setSerialParity(XCharInterface::PARITY_NONE);
    interface()->setEOS("\r\n");
    // interface()->setSerialEOS("\r\n");

    trans( *speed()) = 1000; //Hz

    currentRunning()->disable();
    currentStopping()->disable();
    slipping()->disable();
    stepMotor()->disable();
    stepEncoder()->disable();
    timeAcc()->disable();
    timeDec()->disable();
    active()->disable();
    microStep()->disable();
    hasEncoder()->disable();
    pushing()->disable();
    store()->disable();
    roundBy()->disable();
    round()->disable();
    goHomeMotor()->disable();
    auxBits()->disable();
    m_pulsesTotal = 0;
}
char
XSigmaPAMC104::channelChar(const Snapshot &shot) {
    return 'A' + (unsigned int)shot[ *interface()->address()];
}
void
XSigmaPAMC104::clearPosition() {
    m_pulsesTotal = 0;
}
void
XSigmaPAMC104::getStatus(const Snapshot &shot, double *position, bool *slipping, bool *ready) {
    *position = m_pulsesTotal;
}
void
XSigmaPAMC104::stopRotation() {
    XScopedLock<XInterface> lock( *interface());
    interface()->send("S");
    interface()->receive(); //expecting "FIN"
//    if(interface()->toStrSimplified() != "FIN")
//        throw XInterface::XConvError(__FILE__, __LINE__);
}
void
XSigmaPAMC104::setForward() {
    XScopedLock<XInterface> lock( *interface());
//    stopRotation();
    Snapshot shot( *this);
    interface()->sendf("NR%4u%4u%c", (unsigned int)shot[ *speed()], 0, channelChar(shot));
    interface()->receive(); //expecting "OK"
    if(interface()->toStrSimplified() != "OK")
        throw XInterface::XConvError(__FILE__, __LINE__);
}
void
XSigmaPAMC104::setReverse() {
    XScopedLock<XInterface> lock( *interface());
//    stopRotation();
    Snapshot shot( *this);
    interface()->sendf("RR%4u%4u%c", (unsigned int)shot[ *speed()], 0, channelChar(shot));
    interface()->receive(); //expecting "OK"
    if(interface()->toStrSimplified() != "OK")
        throw XInterface::XConvError(__FILE__, __LINE__);
}
void
XSigmaPAMC104::setTarget(const Snapshot &, double target) {
    XScopedLock<XInterface> lock( *interface());
//    stopRotation();
    Snapshot shot( *this);
    long dx = lrint(target - m_pulsesTotal);
    if(dx > 0)
        interface()->sendf("NR%4u%4u%c", (unsigned int)shot[ *speed()], (unsigned int)dx, channelChar(shot));
    else
        interface()->sendf("RR%4u%4u%c", (unsigned int)shot[ *speed()], (unsigned int)(-dx), channelChar(shot));
    m_pulsesTotal = target; //ugly hack.
    interface()->receive(); //expecting "OK"
    if(interface()->toStrSimplified() != "OK")
        throw XInterface::XConvError(__FILE__, __LINE__);
}
