/***************************************************************************
		Copyright (C) 2002-2007 Kentaro Kitagawa
		                   kitagawa@scphys.kyoto-u.ac.jp
		
		This program is free software; you can redistribute it and/or
		modify it under the terms of the GNU Library General Public
		License as published by the Free Software Foundation; either
		version 2 of the License, or (at your option) any later version.
		
		You should have received a copy of the GNU Library General 
		Public License and a list of authors along with this program; 
		see the files COPYING and AUTHORS.
 ***************************************************************************/
#include "pulserdrivernidaq.h"

#ifdef HAVE_NI_DAQMX

#include "interface.h"
#include <klocale.h>

XNIDAQMSeriesWithSSeriesPulser::XNIDAQMSeriesWithSSeriesPulser(const char *name, bool runtime,
   const shared_ptr<XScalarEntryList> &scalarentries,
   const shared_ptr<XInterfaceList> &interfaces,
   const shared_ptr<XThermometerList> &thermometers,
   const shared_ptr<XDriverList> &drivers) :
    XNIDAQmxPulser(name, runtime, scalarentries, interfaces, thermometers, drivers),
	m_ao_interface(XNode::create<XNIDAQmxInterface>("SubInterface", false,
            dynamic_pointer_cast<XDriver>(this->shared_from_this())))
{
	setPausingGateTerm(formatString("/%s/PFI4", intfCtr()->devName()).c_str());
    interfaces->insert(m_ao_interface);
    m_ao_interface->control()->setUIEnabled(false);
}
void
XNIDAQSSeriesPulser::open() throw (XInterface::XInterfaceError &)
{
	if(std::string(interface()->productSeries()) != "S")
		throw XInterface::XInterfaceError(KAME::i18n("Product-type mismatch."), __FILE__, __LINE__);
 	openAODO();
	this->start();	
}
void
XNIDAQDOPulser::open() throw (XInterface::XInterfaceError &)
{
//	if(std::string(interface()->productSeries()) != "M")
//		throw XInterface::XInterfaceError(KAME::i18n("Product-type mismatch."), __FILE__, __LINE__);
 	openDO();
	this->start();	
}
void
XNIDAQMSeriesWithSSeriesPulser::open() throw (XInterface::XInterfaceError &)
{
	if(std::string(interface()->productSeries()) != "M")
		throw XInterface::XInterfaceError(KAME::i18n("Product-type mismatch."), __FILE__, __LINE__);

	intfAO()->start();
	//override auto-setup.
	if(std::string(intfAO()->productType()) == "PCI-6111") {
		m_resolutionAO = 1e-3;
		m_resolutionDO = 1e-3;
	}
	if(std::string(intfAO()->productSeries()) != "S")
		throw XInterface::XInterfaceError(KAME::i18n("Product-type mismatch."), __FILE__, __LINE__);
 	openAODO();
	this->start();	
}

#endif //HAVE_NI_DAQMX
