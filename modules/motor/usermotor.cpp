/***************************************************************************
        Copyright (C) 2002-2015 Kentaro Kitagawa
                           kitagawa@phys.s.u-tokyo.ac.jp

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
REGISTER_TYPE(XDriverList, FlexAR, "OrientalMotor FLEX AR/DG2 motor controller");
REGISTER_TYPE(XDriverList, EMP401, "OrientalMotor EMP401 motor controller");

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
        uint32_t output = interface()->readHoldingTwoResistors(0x20); //reading status1:status2
        bool isready = (output & 0x20000000u);
        if(isready) break;
        if(i ==0) {
            interface()->presetSingleResistor(0x1e, 0x3001u); //C-ON, STOP, M0
            interface()->presetSingleResistor(0x1e, 0x2001u); //C-ON, M0
            if( !wait)
                break;
        }
        msecsleep(100);
        if(i > 10) {
            throw XInterface::XInterfaceError(i18n("Motor is still not ready"), __FILE__, __LINE__);
        }
    }
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
//        smotor = lrint(interface()->readHoldingTwoResistors(0x382) * 1000.0 /  interface()->readHoldingTwoResistors(0x380));
        int a = 1000;
        int b = shot[ *stepMotor()];
        if((a != interface()->readHoldingTwoResistors(0x380)) ||
            (b != interface()->readHoldingTwoResistors(0x382))) {
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
    netin &= ~(0x4000uL | 0x8000uL | 0x20uL | 0xfuL); //FWD | RVS | STOP | START | M0-2
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
                         const std::vector<std::vector<double>> &speeds, const std::vector<const shared_ptr<XMotorDriver>> &slaves) {
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
    netin &= ~(0x4000uL | 0x8000uL | 0x20uL | 0xfuL); //FWD | RVS | STOP | START | M0-2
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
        uint32_t output = interface()->readHoldingTwoResistors(0x7e);
        bool isready = output & 0x20;
        if(isready) break;
        if(i ==0) {
            uint32_t netin = interface()->readHoldingTwoResistors(0x7c);
            netin &= ~(0x4000uL | 0x8000u | 0x8uL); //FWD | RVS | START
            interface()->presetTwoResistors(0x7c, netin | 0x20uL); //STOP
//            fprintf(stderr, "STOP%u\n", netin);
            msecsleep(4);
            interface()->presetTwoResistors(0x7c, netin & ~0x20uL);
            if( !wait)
                break;
        }
        msecsleep(150);
        if(i > 10) {
            throw XInterface::XInterfaceError(i18n("Motor is still not ready"), __FILE__, __LINE__);
        }
    }
}
void
XFlexAR::setForward() {
    XScopedLock<XInterface> lock( *interface());
    uint32_t netin = interface()->readHoldingTwoResistors(0x7c);
    netin &= ~(0x4000uL | 0x8000uL | 0x20uL | 0x8uL); //FWD | RVS | STOP | START
    interface()->presetTwoResistors(0x7c, netin | 0x4000uL); //FWD
}
void
XFlexAR::setReverse() {
    XScopedLock<XInterface> lock( *interface());
    uint32_t netin = interface()->readHoldingTwoResistors(0x7c);
    netin &= ~(0x4000uL | 0x8000uL | 0x20uL | 0x8uL); //FWD | RVS | STOP | START
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
    if((bits < 0x40uL) || (bits == 0xffuL)) {
        interface()->presetTwoResistors(0x7c, (netin & ~0x3fuL) | (bits & 0x3fuL));
    }
    else {
        //debug use
        if(bits > 0x10000uL) {
            uint32_t addr = bits / 0x10000uL;
            interface()->presetTwoResistors(addr, bits % 0x10000uL);
        }
        else {
            uint32_t res = interface()->readHoldingTwoResistors(bits);
            fprintf(stderr, "%x: 0x%x(%d)\n", bits, res, res);
        }
        fprintf(stderr, "7e: %x\n", interface()->readHoldingTwoResistors(0x7e));
    }
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
