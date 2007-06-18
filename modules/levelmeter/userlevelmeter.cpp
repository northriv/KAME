/***************************************************************************
		Copyright (C) 2002-2007 Kentaro Kitagawa
		                   kitag@issp.u-tokyo.ac.jp
		
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
#include <klocale.h>
//---------------------------------------------------------------------------

REGISTER_TYPE(XDriverList, ILM, "Oxford ILM Helium Level Meter");

XILM::XILM(const char *name, bool runtime,
			   const shared_ptr<XScalarEntryList> &scalarentries,
			   const shared_ptr<XInterfaceList> &interfaces,
			   const shared_ptr<XThermometerList> &thermometers,
			   const shared_ptr<XDriverList> &drivers) :
    XOxfordDriver<XLevelMeter>(name, runtime, scalarentries, interfaces, thermometers, drivers)
{
	const char *channels_create[] = {"He", 0L};
	createChannels(scalarentries, channels_create);
}

double
XILM::getLevel(unsigned int ch)
{
	return read(1) / 10.0;
}
