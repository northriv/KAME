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
#ifndef signalgeneratorH
#define signalgeneratorH

#include "primarydriver.h"
#include "xnodeconnector.h"

class FrmSG;

class XSG : public XPrimaryDriver
{
	XNODE_OBJECT
protected:
	XSG(const char *name, bool runtime,
		const shared_ptr<XScalarEntryList> &scalarentries,
		const shared_ptr<XInterfaceList> &interfaces,
		const shared_ptr<XThermometerList> &thermometers,
		const shared_ptr<XDriverList> &drivers);
public:
	//! usually nothing to do
	virtual ~XSG() {}
	//! show all forms belonging to driver
	virtual void showForms();
protected:
	//! Start up your threads, connect GUI, and activate signals
	virtual void start();
	//! Shut down your threads, unconnect GUI, and deactivate signals
	//! this may be called even if driver has already stopped.
	virtual void stop();
  
	//! this is called when raw is written 
	//! unless dependency is broken
	//! convert raw to record
	virtual void analyzeRaw() throw (XRecordError&);
	//! this is called after analyze() or analyzeRaw()
	//! record is readLocked
	virtual void visualize();
public:
	//! driver specific part below
	const shared_ptr<XDoubleNode> &freq() const {return m_freq;} //!< freq [MHz]
	const shared_ptr<XDoubleNode> &oLevel() const {return m_oLevel;} //!< Output Level [dBm]
	const shared_ptr<XBoolNode> &fmON() const {return m_fmON;} //!< Activate FM
	const shared_ptr<XBoolNode> &amON() const {return m_amON;} //!< Activate AM
    
	double freqRecorded() const {return m_freqRecorded;} //!< freq [MHz]
protected:
	virtual void changeFreq(double mhz) = 0;
	virtual void onOLevelChanged(const shared_ptr<XValueNodeBase> &) = 0;
	virtual void onFMONChanged(const shared_ptr<XValueNodeBase> &) = 0;
	virtual void onAMONChanged(const shared_ptr<XValueNodeBase> &) = 0;
private:
	void onFreqChanged(const shared_ptr<XValueNodeBase> &);  

	const shared_ptr<XDoubleNode> m_freq;
	const shared_ptr<XDoubleNode> m_oLevel;
	const shared_ptr<XBoolNode> m_fmON;
	const shared_ptr<XBoolNode> m_amON;
  
	double m_freqRecorded;
  
	xqcon_ptr m_conFreq, m_conOLevel, m_conFMON, m_conAMON;
	shared_ptr<XListener> m_lsnFreq, m_lsnOLevel, m_lsnFMON, m_lsnAMON;
  
	const qshared_ptr<FrmSG> m_form;
};//---------------------------------------------------------------------------
#endif
