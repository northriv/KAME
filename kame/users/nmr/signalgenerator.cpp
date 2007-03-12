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
#include <klocale.h>
#include <qcheckbox.h>
#include <qstatusbar.h>

#include "analyzer.h"
#include "charinterface.h"
#include "signalgenerator.h"
#include "forms/signalgeneratorform.h"

XSG::XSG(const char *name, bool runtime,
		 const shared_ptr<XScalarEntryList> &scalarentries,
		 const shared_ptr<XInterfaceList> &interfaces,
		 const shared_ptr<XThermometerList> &thermometers,
		 const shared_ptr<XDriverList> &drivers)
    : XPrimaryDriver(name, runtime, scalarentries, interfaces, thermometers, drivers),
	  m_freq(create<XDoubleNode>("Freq", true)),
	  m_oLevel(create<XDoubleNode>("OutputLevel", true)),
	  m_fmON(create<XBoolNode>("FMON", true)),
	  m_amON(create<XBoolNode>("AMON", true)),
	  m_form(new FrmSG(g_pFrmMain))
{
	m_form->statusBar()->hide();
	m_form->setCaption(KAME::i18n("Signal Gen. Control - ") + getLabel() );

	m_conOLevel = xqcon_create<XQLineEditConnector>(m_oLevel, m_form->m_edOLevel);
	m_conFreq = xqcon_create<XQLineEditConnector>(m_freq, m_form->m_edFreq);
	m_conAMON = xqcon_create<XQToggleButtonConnector>(m_amON, m_form->m_ckbAMON);
	m_conFMON = xqcon_create<XQToggleButtonConnector>(m_fmON, m_form->m_ckbFMON);
      
	oLevel()->setUIEnabled(false);
	freq()->setUIEnabled(false);
	amON()->setUIEnabled(false);
	fmON()->setUIEnabled(false);
}
void
XSG::showForms()
{
	m_form->show();
	m_form->raise();
}

void
XSG::start()
{
	m_oLevel->setUIEnabled(true);
	m_freq->setUIEnabled(true);
	m_amON->setUIEnabled(true);
	m_fmON->setUIEnabled(true);
        
	m_lsnOLevel = oLevel()->onValueChanged().connectWeak(
		shared_from_this(), &XSG::onOLevelChanged);
	m_lsnFreq = freq()->onValueChanged().connectWeak(
		shared_from_this(), &XSG::onFreqChanged);
	m_lsnAMON = amON()->onValueChanged().connectWeak(
		shared_from_this(), &XSG::onAMONChanged);
	m_lsnFMON = fmON()->onValueChanged().connectWeak(
		shared_from_this(), &XSG::onFMONChanged);
}
void
XSG::stop()
{        
	m_lsnOLevel.reset();
	m_lsnFreq.reset();
	m_lsnAMON.reset();
	m_lsnFMON.reset();
  
	m_oLevel->setUIEnabled(false);
	m_freq->setUIEnabled(false);
	m_amON->setUIEnabled(false);
	m_fmON->setUIEnabled(false);
  
	afterStop();
}

void
XSG::analyzeRaw() throw (XRecordError&)
{
    m_freqRecorded = pop<double>();
}
void
XSG::visualize()
{
	//! impliment extra codes which do not need write-lock of record
	//! record is read-locked
}
void
XSG::onFreqChanged(const shared_ptr<XValueNodeBase> &)
{
    double _freq = *freq();
    if(_freq <= 0) {
        gErrPrint(getLabel() + " " + KAME::i18n("Positive Value Needed."));
        return;
    }

    XTime time(XTime::now());
    
    try {
        changeFreq(_freq);
    }
	catch (XKameError &e) {
        e.print(getLabel() + " " + KAME::i18n("SG Error."));
        return;
    }
    clearRaw();
    push(_freq);
    finishWritingRaw(time, XTime::now());
}


XSG7200::XSG7200(const char *name, bool runtime,
				 const shared_ptr<XScalarEntryList> &scalarentries,
				 const shared_ptr<XInterfaceList> &interfaces,
				 const shared_ptr<XThermometerList> &thermometers,
				 const shared_ptr<XDriverList> &drivers)
    : XCharDeviceDriver<XSG>(name, runtime, scalarentries, interfaces, thermometers, drivers)
{
	interface()->setGPIBUseSerialPollOnWrite(false);
	interface()->setGPIBUseSerialPollOnRead(false);
}
XSG7130::XSG7130(const char *name, bool runtime,
				 const shared_ptr<XScalarEntryList> &scalarentries,
				 const shared_ptr<XInterfaceList> &interfaces,
				 const shared_ptr<XThermometerList> &thermometers,
				 const shared_ptr<XDriverList> &drivers)
    : XSG7200(name, runtime, scalarentries, interfaces, thermometers, drivers)
{
}
void
XSG7200::changeFreq(double mhz)
{
	interface()->sendf("FR%fMHZ", mhz);
	msecsleep(50); //wait stabilization of PLL
}
void
XSG7200::onOLevelChanged(const shared_ptr<XValueNodeBase> &)
{
	interface()->sendf("LE%fDBM", (double)*oLevel());
}
void
XSG7200::onFMONChanged(const shared_ptr<XValueNodeBase> &)
{
	interface()->send(*fmON() ? "FMON" : "FMOFF");
}
void
XSG7200::onAMONChanged(const shared_ptr<XValueNodeBase> &)
{
	interface()->send(*amON() ? "AMON" : "AMOFF");
}

XHP8643::XHP8643(const char *name, bool runtime,
				 const shared_ptr<XScalarEntryList> &scalarentries,
				 const shared_ptr<XInterfaceList> &interfaces,
				 const shared_ptr<XThermometerList> &thermometers,
				 const shared_ptr<XDriverList> &drivers)
    : XCharDeviceDriver<XSG>(name, runtime, scalarentries, interfaces, thermometers, drivers)
{
}
void
XHP8643::changeFreq(double mhz)
{
	interface()->sendf("FREQ:CW %f MHZ", mhz);
	msecsleep(50); //wait stabilization of PLL
}
void
XHP8643::onOLevelChanged(const shared_ptr<XValueNodeBase> &)
{
	interface()->sendf("AMPL:LEV %f DBM", (double)*oLevel());
}
void
XHP8643::onFMONChanged(const shared_ptr<XValueNodeBase> &)
{
	interface()->sendf("FMSTAT %s", *fmON() ? "ON" : "OFF");
}
void
XHP8643::onAMONChanged(const shared_ptr<XValueNodeBase> &)
{
	interface()->sendf("AMSTAT %s", *amON() ? "ON" : "OFF");
}

XHP8648::XHP8648(const char *name, bool runtime,
				 const shared_ptr<XScalarEntryList> &scalarentries,
				 const shared_ptr<XInterfaceList> &interfaces,
				 const shared_ptr<XThermometerList> &thermometers,
				 const shared_ptr<XDriverList> &drivers)
    : XHP8643(name, runtime, scalarentries, interfaces, thermometers, drivers)
{
}
void
XHP8648::onOLevelChanged(const shared_ptr<XValueNodeBase> &)
{
	interface()->sendf("POW:AMPL %f DBM", (double)*oLevel());
}
