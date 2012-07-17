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

REGISTER_TYPE(XDriverList, FlexAR, "OrientalMotor FLEX AR motor controler");

XFlexAR::XFlexAR(const char *name, bool runtime,
	Transaction &tr_meas, const shared_ptr<XMeasure> &meas) :
    XModbusRTUDriver<XMotorDriver>(name, runtime, ref(tr_meas), meas) {
	interface()->setSerialBaudRate(115200);
	interface()->setSerialStopBits(1);
	interface()->setSerialParity(XCharInterface::PARITY_EVEN);
}
void
XFlexAR::getStatus(const Snapshot &shot, double *position, bool *slipping, bool *ready) {
	uint32_t alarm = interface()->readHoldingTwoResistors(0x80);
	if(alarm) {
		throw XInterface::XInterfaceError(i18n("Alarm %1 has been emitted").arg((int)alarm), __FILE__, __LINE__);
	}
	uint32_t warn = interface()->readHoldingTwoResistors(0x96);
	*slipping = (warn == 0x30);
	if(warn != 0x30) {
		gWarnPrint(i18n("Code = %1").arg((int)warn));
	}
	uint32_t ierr = interface()->readHoldingTwoResistors(0xac);
	if(ierr) {
		throw XInterface::XInterfaceError(i18n("Interface error %1 has been emitted").arg((int)ierr), __FILE__, __LINE__);
	}
	*position = interface()->readHoldingTwoResistors(0xcc) / (double)shot[ *step()];
	uint32_t output = interface()->readHoldingTwoResistors(0x7e);
	*ready = output & 0x10;
}
void
XFlexAR::changeConditions(const Snapshot &shot) {
	interface()->presetTwoResistors(0x240,  lrint(shot[ *currentRunning()] * 10.0));
	interface()->presetTwoResistors(0x242,  lrint(shot[ *currentStopping()] * 10.0));
	interface()->presetTwoResistors(0x1028,  shot[ *microStep()] ? 1 : 0);
	interface()->presetTwoResistors(0x28c, 0); //common setting for acc/dec.
	interface()->presetTwoResistors(0x280,  lrint(shot[ *timeAcc()] * 1e3));
	interface()->presetTwoResistors(0x282,  lrint(shot[ *timeDec()] * 1e3));
	interface()->presetTwoResistors(0x380,  lrint(shot[ *step()]));
	interface()->presetTwoResistors(0x480,  lrint(shot[ *speed()]));
}
void
XFlexAR::getConditions(Transaction &tr) {
	interface()->diagnostics();
	tr[ *currentRunning()] = interface()->readHoldingTwoResistors(0x240) * 0.1;
	tr[ *currentStopping()] = interface()->readHoldingTwoResistors(0x242) * 0.1;
	tr[ *microStep()] = (interface()->readHoldingTwoResistors(0x1028) == 1);
	tr[ *timeAcc()] = interface()->readHoldingTwoResistors(0x280) * 1e-3;
	tr[ *timeDec()] = interface()->readHoldingTwoResistors(0x282) * 1e-3;
	tr[ *step()] = interface()->readHoldingTwoResistors(0x380);
	tr[ *speed()] = interface()->readHoldingTwoResistors(0x480);
}
void
XFlexAR::setTarget(const Snapshot &shot, double target) {
	interface()->presetTwoResistors(0x400, lrint(target * shot[ *step()]));
}
void
XFlexAR::setActive(bool active) {
	uint32_t input = interface()->readHoldingTwoResistors(0x7c);
	interface()->presetTwoResistors(0x7c, (input & ~0x10u) + ( !active ? 0x10u : 0));
}
