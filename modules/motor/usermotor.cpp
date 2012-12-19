/***************************************************************************
		Copyright (C) 2002-2012 Kentaro Kitagawa
		                   kitag@kochi-u.ac.jp

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

XFlexCRK::XFlexCRK(const char *name, bool runtime,
	Transaction &tr_meas, const shared_ptr<XMeasure> &meas) :
    XModbusRTUDriver<XMotorDriver>(name, runtime, ref(tr_meas), meas) {
	interface()->setSerialBaudRate(115200);
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
	interface()->presetTwoResistors(0x312,  lrint(shot[ *stepEncoder()]));
	interface()->presetTwoResistors(0x314,  lrint(shot[ *stepMotor()]));
	interface()->presetTwoResistors(0x502,  lrint(shot[ *speed()]));
	unsigned int microstep = shot[ *microStep()] ? 6 : 0;
	if(interface()->readHoldingSingleResistor(0x311) != microstep) {
		gWarnPrint(i18n("Store settings to NV memory and restart, microstep div.=10."));
		interface()->presetSingleResistor(0x311, microstep); //division = 10.
	}
}
void
XFlexCRK::getConditions(Transaction &tr) {
	XScopedLock<XInterface> lock( *interface());
	interface()->diagnostics();
	tr[ *currentRunning()] = interface()->readHoldingSingleResistor(0x21e);
	tr[ *currentStopping()] = interface()->readHoldingSingleResistor(0x21f);
	tr[ *microStep()] = (interface()->readHoldingSingleResistor(0x311) != 0);
	tr[ *timeAcc()] = interface()->readHoldingTwoResistors(0x224) * 1e-3;
	tr[ *timeDec()] = interface()->readHoldingTwoResistors(0x226) * 1e-3;
	tr[ *stepEncoder()] = interface()->readHoldingTwoResistors(0x312);
	tr[ *stepMotor()] = interface()->readHoldingTwoResistors(0x314);
	tr[ *speed()] = interface()->readHoldingTwoResistors(0x502);
	tr[ *target()] = static_cast<int32_t>(interface()->readHoldingTwoResistors(0x402))
			* 360.0 / tr[ *stepMotor()];
	interface()->presetSingleResistor(0x203, 0); //STOP I/O normally open.
	interface()->presetSingleResistor(0x200, 0); //START by RS485.
	interface()->presetSingleResistor(0x20b, 0); //C-ON by RS485.
	interface()->presetSingleResistor(0x20d, 0); //No. by RS485.
	interface()->presetSingleResistor(0x202, 1); //Dec. after STOP.
	interface()->presetSingleResistor(0x601, 1); //Absolute.
}
void
XFlexCRK::stopMotor() {
	for(int i = 0;; ++i) {
		uint32_t output = interface()->readHoldingTwoResistors(0x20); //reading status1:status2
		bool isready = (output & 0x20000000u);
		if(isready) break;
		if(i ==0) {
			interface()->presetSingleResistor(0x1e, 0x3001u); //C-ON, STOP, M1
			interface()->presetSingleResistor(0x1e, 0x2001u); //C-ON, M1
		}
		msecsleep(100);
		if(i > 10) {
			gErrPrint(getLabel() + i18n(", Motor is still running"));
		}
	}
}
void
XFlexCRK::setTarget(const Snapshot &shot, double target) {
	XScopedLock<XInterface> lock( *interface());
	stopMotor();
	interface()->presetTwoResistors(0x402, lrint(target / 360.0 * shot[ *stepMotor()]));
	interface()->presetSingleResistor(0x1e, 0x2101u); //C-ON, START, M1
	interface()->presetSingleResistor(0x1e, 0x2001u); //C-ON, M1
}
void
XFlexCRK::setActive(bool active) {
	XScopedLock<XInterface> lock( *interface());
	if(active) {
		interface()->presetSingleResistor(0x1e, 0x2001u); //C-ON, M1
	}
	else {
		stopMotor();
		interface()->presetSingleResistor(0x1e, 0x0001u); //M1
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
		interface()->presetTwoResistors(0x184, 1); //clears alarm.
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
	interface()->presetTwoResistors(0x240,  lrint(shot[ *currentRunning()] * 10.0));
	interface()->presetTwoResistors(0x242,  lrint(shot[ *currentStopping()] * 10.0));
	interface()->presetTwoResistors(0x28c, 0); //common setting for acc/dec.
	interface()->presetTwoResistors(0x280,  lrint(shot[ *timeAcc()] * 1e3));
	interface()->presetTwoResistors(0x282,  lrint(shot[ *timeDec()] * 1e3));
	int b = 1;
	if(int x = shot[ *stepMotor()] % 1000)
		b = 1000 / x;
	b = std::min(b, 10);
	int a = lrint(shot[ *stepMotor()]/1000.0*b);
	interface()->presetTwoResistors(0x380,  a); //A
	interface()->presetTwoResistors(0x382,  b); //B, rot=1000B/A
	interface()->presetTwoResistors(0x1002, shot[ *stepEncoder()] / shot[ *stepMotor()]); //Multiplier is stored in MS2 No.
	interface()->presetTwoResistors(0x480,  lrint(shot[ *speed()]));
	interface()->presetTwoResistors(0x1028, shot[ *microStep()] ? 1 : 0);
}
void
XFlexAR::getConditions(Transaction &tr) {
	XScopedLock<XInterface> lock( *interface());
	interface()->diagnostics();
	tr[ *currentRunning()] = interface()->readHoldingTwoResistors(0x240) * 0.1;
	tr[ *currentStopping()] = interface()->readHoldingTwoResistors(0x242) * 0.1;
	tr[ *microStep()] = (interface()->readHoldingTwoResistors(0x1028) != 0);
	tr[ *timeAcc()] = interface()->readHoldingTwoResistors(0x280) * 1e-3;
	tr[ *timeDec()] = interface()->readHoldingTwoResistors(0x282) * 1e-3;
	tr[ *stepMotor()] = interface()->readHoldingTwoResistors(0x380) * 1000.0 /  interface()->readHoldingTwoResistors(0x382);
	tr[ *stepEncoder()] = tr[ *stepMotor()] * interface()->readHoldingTwoResistors(0x1002); //Multiplier is stored in MS2 No.
	tr[ *hasEncoder()] = true;
	tr[ *speed()] = interface()->readHoldingTwoResistors(0x480);
	tr[ *target()] = static_cast<int32_t>(interface()->readHoldingTwoResistors(0x400))
			* 360.0 / tr[ *stepMotor()];
	interface()->presetTwoResistors(0x200, 3); //Inactive after stop.
	interface()->presetTwoResistors(0x500, 1); //Absolute.
	interface()->presetTwoResistors(0x119e, 71); //NET-OUT15 = TLC
}
void
XFlexAR::setTarget(const Snapshot &shot, double target) {
	XScopedLock<XInterface> lock( *interface());
	stopMotor();
	int steps = shot[ *hasEncoder()] ? shot[ *stepEncoder()] : shot[ *stepMotor()];
	interface()->presetTwoResistors(0x400, lrint(target / 360.0 * steps));
	interface()->presetTwoResistors(0x7c, 0x100u); //MS0
	interface()->presetTwoResistors(0x7c, 0x0u);
}
void
XFlexAR::stopMotor() {
	for(int i = 0;; ++i) {
		uint32_t output = interface()->readHoldingTwoResistors(0x7e);
		bool isready = output & 0x20;
		if(isready) break;
		if(i ==0) {
			interface()->presetTwoResistors(0x7c, 0x20u); //STOP
			interface()->presetTwoResistors(0x7c, 0x0u);
		}
		msecsleep(100);
		if(i > 10) {
			gErrPrint(getLabel() + i18n(", Motor is still running"));
		}
	}
}
void
XFlexAR::setActive(bool active) {
	XScopedLock<XInterface> lock( *interface());
	if(active) {
		interface()->presetTwoResistors(0x7c, 0x0u);
	}
	else {
		stopMotor();
		interface()->presetTwoResistors(0x7c, 0x40u); //FREE
	}
}
void
XFlexAR::setAUXBits(unsigned int bits) {
	interface()->presetTwoResistors(0x1140, 32); //OUT0 to R0
	interface()->presetTwoResistors(0x1142, 33); //OUT1 to R1
	interface()->presetTwoResistors(0x1144, 34); //OUT2 to R2
	interface()->presetTwoResistors(0x1146, 35); //OUT3 to R3
	interface()->presetTwoResistors(0x1148, 36); //OUT4 to R4
	interface()->presetTwoResistors(0x114a, 37); //OUT5 to R5
	interface()->presetTwoResistors(0x1160, 32); //NET-IN0 to R0
	interface()->presetTwoResistors(0x1162, 33); //NET-IN1 to R1
	interface()->presetTwoResistors(0x1164, 34); //NET-IN2 to R2
	interface()->presetTwoResistors(0x1166, 35); //NET-IN3 to R3
	interface()->presetTwoResistors(0x1168, 36); //NET-IN4 to R4
	interface()->presetTwoResistors(0x116a, 37); //NET-IN5 to R5
	interface()->presetSingleResistor(0x7d, bits & 0x3fu);
}
