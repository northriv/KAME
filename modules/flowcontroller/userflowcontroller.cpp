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

#include "userflowcontroller.h"
//---------------------------------------------------------------------------

REGISTER_TYPE(XDriverList, FCST1000, "FCST1000 Series Mass Flow Controllers");

XFlowControllerDriver::XFlowControllerDriver(const char *name, bool runtime,
	Transaction &tr_meas, const shared_ptr<XMeasure> &meas) :
	XFujikinProtocolDriver<XFlowControllerDriver>(name, runtime, ref(tr_meas), meas) {
	interface()->setSerialBaudRate(38400);
	interface()->setSerialStopBits(1);
	interface()->setSerialParity(XCharInterface::PARITY_NONE);
}
bool
XFlowControllerDriver::isUnitInSLM() {
	XString unit = interface()->query<XString>(ValveDriverClass, 1, 0x03);
	return (unit == "SLM");
}
bool
XFlowControllerDriver::isController() {
	unsigned int type = interface()->query<uint8_t>(GasCalibrationClass, 1, 0xa0);
	return (type != 0);
}
double
XFlowControllerDriver::getFullScale() {
	return interface()->query<uint16_t>(GasCalibrationClass, 1, 0x02);
}
void
XFlowControllerDriver::getStatus(double &flow, double &valve_v, bool &alarm, bool &warning)  {
	flow = interface()->query<uint16_t>(ValveDriverClass, 1, 0xa9);
	flow = (flow - 0x4000) / 0x8000;
	flow *= getFullScale();
	valve_v = interface()->query<uint16_t>(ValveDriverClass, 1, 0xa9);
	valve_v = (valve_v - 0x4000) / 0x8000 * 100.0;

	int bits = interface()->query<uint8_t>(ValveDriverClass, 1, 0xa0);
	alarm = bits & 2;
	warning = bits & 32;
}
void
XFlowControllerDriver::setValveState(bool open) {
	interface()->send(ValveDriverClass, 1, 0x01, open ? (uint8_t)0x02 : (uint8_t)0x01);
}
void
XFlowControllerDriver::changeControl(bool ctrl) {
	interface()->send(ValveDriverClass, 1, 0x01, (uint8_t)0x00);
	if(ctrl)
		interface()->send(FlowControllerClass, 1, 0x03, (uint8_t)0x01); //digital mode.
	else
		interface()->send(FlowControllerClass, 1, 0x03, (uint8_t)0x02); //analog mode.
}
void
XFlowControllerDriver::changeSetPoint(double target) {
	target  = target / getFullScale();
	target = std::max(0.0, target);
	target = std::min(1.0, target);
	uint16_t x = lrint(target * 0x8000 + 0x4000);
	interface()->send(FlowControllerClass, 1, 0xa4, x);
}
void
XFlowControllerDriver::setRampTime(double time) {
	interface()->send(ValveDriverClass, 1, 0xa4, (uint32_t)lrint(time));
}
