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
    XModbusRTUDriver<XFlexAR>(name, runtime, ref(tr_meas), meas) {
	setSerialBaudRate(9600);
	setSerialStopBits(1);
}
void
XFlexAR::getStatus(const Snapshot &shot, double *position, bool *slipping, bool *ready) {
	uint32_t alarm = readHoldingTwoResistors(0x80);
	if(alarm) {
		throw XInterfaceError(formatString(i18n("Alarm %d has been emitted"), (int)alarm), __FILE__, __LINE__);
	}
	uint32_t warn = readHoldingTwoResistors(0x96);
	*slipping = (warn == 0x30);
	if(warn != 0x30) {
		gWarnPrint(formatString(i18n("Code = %d"), (int)warn));
	}
	uint32_t ierr = readHoldingTwoResistors(0xac);
	if(ierr) {
		throw XInterfaceError(formatString(i18n("Interface error %d has been emitted"), (int)ierr), __FILE__, __LINE__);
	}
	*position = readHoldingTwoResistors(0xcc) / (double)shot[ *step()];
	uint32_t output = readHoldingTwoResistors(0x7e);
	*ready = output & 0x10;
}
void
XFlexAR::changeConditions(const Snapshot &shot) {
	presetTwoResistors(0x380,  lrint(shot[ *step()]));
	presetTwoResistors(0x480,  lrint(shot[ *speed()]));
	presetTwoResistors(0x240,  lrint(shot[ *currentRunning()] * 10.0));
	presetTwoResistors(0x242,  lrint(shot[ *currentStopping()] * 10.0));
	presetTwoResistors(0x1028,  shot[ *microStep()] ? 1 : 0);
	presetTwoResistors(0x28c, 0); //common setting for acc/dec.
	presetTwoResistors(0x280,  lrint(shot[ *timeAcc()] * 1e3));
	presetTwoResistors(0x282,  lrint(shot[ *timeDec()] * 1e3));
}
void
XFlexAR::getConditions(Transaction &tr) {
	tr[ *step()] = readHoldingTwoResistors(0x380);
	tr[ *speed()] = readHoldingTwoResistors(0x480);
	tr[ *currentRunning()] = readHoldingTwoResistors(0x240) * 0.1;
	tr[ *currentStopping()] = readHoldingTwoResistors(0x242) * 0.1;
	tr[ *microStep()] = (readHoldingTwoResistors(0x1028) == 1);
	tr[ *timeAcc()] = readHoldingTwoResistors(0x280) * 1e-3;
	tr[ *timeDec()] = readHoldingTwoResistors(0x282) * 1e-3;
}
void
XFlexAR::setTarget(const Snapshot &shot, double target) {
	presetTwoResistors(0x400, lrint(target * shot[ *step()]));
}
void
XFlexAR::setActive(bool active) {
	uint32_t input = readHoldingTwoResistors(0x7c);
	presetTwoResistors(0x7c, (input & ~0x10u) + ( !active ? 0x10u : 0));
}
