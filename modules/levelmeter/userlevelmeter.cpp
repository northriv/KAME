/***************************************************************************
		Copyright (C) 2002-2014 Kentaro Kitagawa
		                   kitag@kochi-u.ac.jp
		
		This program is free software; you can redistribute it and/or
		modify it under the terms of the GNU Library General Public
		License as published by the Free Software Foundation; either
		version 2 of the License, or (at your option) any later version.
		
		You should have received a copy of the GNU Library General 
		Public License and a list of authors along with this program; 
		see the files COPYING and AUTHORS.
***************************************************************************/
//---------------------------------------------------------------------------
#include "userlevelmeter.h"
//---------------------------------------------------------------------------

REGISTER_TYPE(XDriverList, ILM, "Oxford ILM Helium Level Meter");
REGISTER_TYPE(XDriverList, LM500, "Cryomagnetics LM-500 Level Meter");

XILM::XILM(const char *name, bool runtime,
	Transaction &tr_meas, const shared_ptr<XMeasure> &meas) :
    XOxfordDriver<XLevelMeter>(name, runtime, ref(tr_meas), meas) {
	const char *channels_create[] = {"He", 0L};
	createChannels(ref(tr_meas), meas, channels_create);
}

double
XILM::getLevel(unsigned int ch) {
	return read(ch + 1) / 10.0;
}

XLM500::XLM500(const char *name, bool runtime,
	Transaction &tr_meas, const shared_ptr<XMeasure> &meas) :
    XCharDeviceDriver<XLevelMeter>(name, runtime, ref(tr_meas), meas) {
	const char *channels_create[] = {"1", "2", 0L};
	createChannels(ref(tr_meas), meas, channels_create);

	interface()->setEOS("");
	interface()->setGPIBUseSerialPollOnWrite(false);
	interface()->setGPIBUseSerialPollOnRead (false);
	interface()->setGPIBWaitBeforeWrite(40);
	//    ExclusiveWaitAfterWrite = 10;
	interface()->setGPIBWaitBeforeRead(40);		
	interface()->setSerialEOS("\r\n");
	interface()->setSerialBaudRate(9600);
	interface()->setSerialHasEchoBack(true);
}

double
XLM500::getLevel(unsigned int ch) {
	interface()->queryf("MEAS? %u", ch + 1);
	return interface()->toDouble();
}
