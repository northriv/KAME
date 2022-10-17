/***************************************************************************
        Copyright (C) 2002-2020 Kentaro Kitagawa
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
#include "usergauge.h"
//---------------------------------------------------------------------------

REGISTER_TYPE(XDriverList, TPG362, "Pfeiffer TPG361/362 Gauge Measurement & Control Unit");

XTPG362::XTPG362(const char *name, bool runtime,
	Transaction &tr_meas, const shared_ptr<XMeasure> &meas) :
    XPfeifferProtocolDriver<XGauge>(name, runtime, ref(tr_meas), meas) {
    trans( *interface()->device()) = "SERIAL";
    trans( *interface()->address()) = 1; //Pfeiffer protocol address = 10,11,12
    const char *channels_create[] = {"Ch1", "Ch2", 0L};
	createChannels(ref(tr_meas), meas, channels_create);
}

double
XTPG362::getPressure(unsigned int ch) {
    double res = interface()->requestReal(Snapshot( *this)[ *interface()->address()] * 10 + 1 + ch, XPfeifferProtocolInterface::DATATYPE::U_EXPO_NEW, 740);
    return res * 100.0; //hPa -> Pa
}
