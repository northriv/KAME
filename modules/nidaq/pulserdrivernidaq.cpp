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
#include "pulserdrivernidaq.h"

#include "interface.h"

REGISTER_TYPE(XDriverList, NIDAQAODOPulser, "NMR pulser NI-DAQ analog/digital output");
REGISTER_TYPE(XDriverList, NIDAQDOPulser, "NMR pulser NI-DAQ digital output only");
REGISTER_TYPE(XDriverList, NIDAQMSeriesWithSSeriesPulser, "NMR pulser NI-DAQ M Series with S Series");

XNIDAQMSeriesWithSSeriesPulser::XNIDAQMSeriesWithSSeriesPulser(const char *name, bool runtime,
	Transaction &tr_meas, const shared_ptr<XMeasure> &meas) :
    XNIDAQmxPulser(name, runtime, ref(tr_meas), meas),
	m_ao_interface(XNode::create<XNIDAQmxInterface>("SubInterface", false,
													dynamic_pointer_cast<XDriver>(this->shared_from_this()))) {
    meas->interfaces()->insert(tr_meas, m_ao_interface);
    m_ao_interface->control()->disable();
}
void
XNIDAQAODOPulser::open() {
 	openAODO();
	this->start();	
}
void
XNIDAQDOPulser::open() {
//	if(XString(interface()->productSeries()) != "M")
//		throw XInterface::XInterfaceError(i18n("Product-type mismatch."), __FILE__, __LINE__);
 	openDO();
	this->start();	
}
void
XNIDAQMSeriesWithSSeriesPulser::open() {
	if(XString(interface()->productSeries()) != "M")
		throw XInterface::XInterfaceError(i18n("Product-type mismatch."), __FILE__, __LINE__);

	intfAO()->start();
	if(!intfAO()->isOpened())
		throw XInterface::XInterfaceError(i18n("Opening M series device failed."), __FILE__, __LINE__);

	if(XString(intfAO()->productSeries()) != "S")
		throw XInterface::XInterfaceError(i18n("Product-type mismatch."), __FILE__, __LINE__);
 	openAODO();
	this->start();	
}
