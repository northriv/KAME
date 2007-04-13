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
#ifndef dcsourceH
#define dcsourceH

#include "primarydriver.h"
#include "xnodeconnector.h"

class FrmDCSource;

class XDCSource : public XPrimaryDriver
{
	XNODE_OBJECT
protected:
 XDCSource(const char *name, bool runtime,
		   const shared_ptr<XScalarEntryList> &scalarentries,
		   const shared_ptr<XInterfaceList> &interfaces,
		   const shared_ptr<XThermometerList> &thermometers,
		   const shared_ptr<XDriverList> &drivers);
public:
 //! usually nothing to do
 virtual ~XDCSource() {}
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
 
 //! driver specific part below
 const shared_ptr<XComboNode> &function() const {return m_function;}
 const shared_ptr<XBoolNode> &output() const {return m_output;}
 const shared_ptr<XDoubleNode> &value() const {return m_value;}
 const shared_ptr<XUIntNode> &channel() const {return m_channel;}
 
protected:
 virtual void changeFunction(int x) = 0;
 virtual void changeOutput(bool x) = 0;
 virtual void changeValue(double x) = 0;
private:
 
 xqcon_ptr m_conFunction, m_conOutput, m_conValue, m_conChannel;
 const shared_ptr<XComboNode> m_function;
 const shared_ptr<XBoolNode> m_output;
 const shared_ptr<XDoubleNode> m_value;
 const shared_ptr<XUIntNode> m_channel;
 shared_ptr<XListener> m_lsnFunction, m_lsnOutput, m_lsnValue;
 
 virtual void onFunctionChanged(const shared_ptr<XValueNodeBase> &);
 virtual void onOutputChanged(const shared_ptr<XValueNodeBase> &);
 virtual void onValueChanged(const shared_ptr<XValueNodeBase> &);
 
 const qshared_ptr<FrmDCSource> m_form;
};

#endif

