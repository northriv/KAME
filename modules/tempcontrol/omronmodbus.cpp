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
//---------------------------------------------------------------------------
#include "omronmodbus.h"
#include "charinterface.h"

REGISTER_TYPE(XDriverList, OmronE5_CModbus, "OMRON E5*C controller via modbus");

XOmronE5_CModbus::XOmronE5_CModbus(const char *name, bool runtime,
	Transaction &tr_meas, const shared_ptr<XMeasure> &meas) :
    XModbusRTUDriver<XTempControl> (name, runtime, ref(tr_meas), meas) {
    interface()->setSerialBaudRate(19200);
    interface()->setSerialStopBits(1);
    interface()->setSerialParity(XCharInterface::PARITY_EVEN);

    const char *channels_create[] = { "1", 0L };
	const char *excitations_create[] = { 0L };
    const char *loops_create[] = { "Loop1", 0L };
    createChannels(ref(tr_meas), meas, true, channels_create,
        excitations_create, loops_create);
}
void XOmronE5_CModbus::open() throw (XKameError &) {
	start();

    Snapshot shot_ch( *channels());
    const XNode::NodeList &list( *shot_ch.list());
    trans( *currentChannel(0)) = list.at[0];

    interface()->presetSingleResistor(0x0, 0x00u + 1u); //Writing on

    double digit = pow(10.0, -(double)static_cast<int32_t>(interface()->readHoldingTwoResistors(0x0420)));
    double target = static_cast<int32_t>(interface()->readHoldingTwoResistors(0x106)) * digit;
    trans( *targetTemp(0)) = target;
    double manpow = static_cast<int32_t>(interface()->readHoldingTwoResistors(0x600)) * 0.1;
    trans( *manualPower(0)) = manpow;
    double p = static_cast<int32_t>(interface()->readHoldingTwoResistors(0x0a00)) * 0.1;
    double id_digit = static_cast<int32_t>(interface()->readHoldingTwoResistors(0x1312)) ? 0.1 : 1.0;
    double i = static_cast<int32_t>(interface()->readHoldingTwoResistors(0x0a04)) * id_digit;
    double d = static_cast<int32_t>(interface()->readHoldingTwoResistors(0x0a08)) * id_digit;
    trans( *prop(0)) = p;
    trans( *interval(0)) = i;
    trans( *deriv(0)) = d;

    uint32_t status = static_cast<uint32_t>(interface()->readHoldingTwoResistors(0x2));
    bool isrunning = status & 0x01000000uL;
    bool isman = status & 0x04000000uL;

	for(Transaction tr( *this);; ++tr) {
		const Snapshot &shot(tr);
        for(unsigned int idx = 0; idx < numOfLoops(); ++idx) {
            if( !hasExtDevice(shot, idx)) {
                tr[ *heaterMode(idx)].clear();
                tr[ *heaterMode(idx)].add("OFF");
                tr[ *heaterMode(idx)].add("AUTO");
                tr[ *heaterMode(idx)].add("MAN");
                tr[ *powerMax(idx)].setUIEnabled(false);
                tr[ *powerMin(idx)].setUIEnabled(false);
                tr[ *currentChannel(idx)].setUIEnabled(false);
                tr[ *heaterMode(idx)] = isrunning ? (isman ? 2 : 1) : 0;
            }
            tr[ *powerRange(idx)].setUIEnabled(false);
        }
        if(tr.commit())
			break;
	}

}
double XOmronE5_CModbus::getRaw(shared_ptr<XChannel> &) {
    double digit = pow(10.0, -(double)static_cast<int32_t>(interface()->readHoldingTwoResistors(0x0420)));
    return static_cast<int32_t>(interface()->readHoldingTwoResistors(0x0)) * digit;
}
double XOmronE5_CModbus::getTemp(shared_ptr<XChannel> &) {
    double digit = pow(10.0, -(double)static_cast<int32_t>(interface()->readHoldingTwoResistors(0x0420)));
    return static_cast<int32_t>(interface()->readHoldingTwoResistors(0x0)) * digit;
}
double XOmronE5_CModbus::getHeater(unsigned int loop) {
    return static_cast<int32_t>(interface()->readHoldingTwoResistors(0x8)) * 0.1;
}
void XOmronE5_CModbus::onPChanged(unsigned int, double p) {
    interface()->presetTwoResistors(0x0a00, static_cast<uint32_t>(lrint(p / 0.1)));
}
void XOmronE5_CModbus::onIChanged(unsigned int, double i) {
    double id_digit = static_cast<int32_t>(interface()->readHoldingTwoResistors(0x1312)) ? 0.1 : 1.0;
    interface()->presetTwoResistors(0x0a04, static_cast<uint32_t>(lrint(i / id_digit)));
}
void XOmronE5_CModbus::onDChanged(unsigned int, double d) {
    double id_digit = static_cast<int32_t>(interface()->readHoldingTwoResistors(0x1312)) ? 0.1 : 1.0;
    interface()->presetTwoResistors(0x0a08, static_cast<uint32_t>(lrint(d / id_digit)));
}
void XOmronE5_CModbus::onTargetTempChanged(unsigned int, double temp) {
    double digit = pow(10.0, -(double)static_cast<int32_t>(interface()->readHoldingTwoResistors(0x0420)));
    interface()->presetTwoResistors(0x106, static_cast<uint32_t>(lrint(temp / digit)));
}
void XOmronE5_CModbus::onManualPowerChanged(unsigned int, double pow) {
    interface()->presetTwoResistors(0x600, static_cast<uint32_t>(lrint(pow / 0.1)));
}
void XOmronE5_CModbus::onHeaterModeChanged(unsigned int, int) {
    bool isman = ( **heaterMode(0))->to_str() == "MAN";
    bool isrunning = (( **heaterMode(0))->to_str() == "AUTO") || isman;
    interface()->presetSingleResistor(0x0, 0x0100u + (isrunning ? 0u : 1u));
    interface()->presetSingleResistor(0x0, 0x0900u + (isman ? 1u : 0u));
}
void XOmronE5_CModbus::onPowerRangeChanged(unsigned int /*loop*/, int) {
}
void XOmronE5_CModbus::onCurrentChannelChanged(unsigned int , const shared_ptr<XChannel> &) {
}
void XOmronE5_CModbus::onExcitationChanged(const shared_ptr<XChannel> &, int) {
}
