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
//----------------------------------------------------------------------------
#ifndef caltableH
#define caltableH
//----------------------------------------------------------------------------

#include "thermometer.h"
#include "xnodeconnector.h"
//----------------------------------------------------------------------------

class Ui_FrmCalTable;
typedef QForm<QWidget, Ui_FrmCalTable> FrmCalTable;
class Ui_FrmGraphNURL;
typedef QForm<QWidget, Ui_FrmGraphNURL> FrmGraphNURL;
class XWaveNGraph;

class XConCalTable : public XQConnector {
	Q_OBJECT
public:
	XConCalTable(const shared_ptr<XThermometerList> &list, FrmCalTable *form);
	virtual ~XConCalTable() {}

	const shared_ptr<XTouchableNode> &display() const {return m_display;}
	const shared_ptr<XDoubleNode> &temp() const {return m_temp;}
	const shared_ptr<XDoubleNode> &value() const {return m_value;}  
	const shared_ptr<XItemNode<XThermometerList, XThermometer> >&thermometer() const {
		return m_thermometer;
	}
  
private:
	shared_ptr<XThermometerList> m_list; 
 
	const shared_ptr<XTouchableNode> m_display;
	const shared_ptr<XDoubleNode> m_temp, m_value;
	shared_ptr<XItemNode<XThermometerList, XThermometer> > m_thermometer;
	xqcon_ptr m_conThermo, m_conTemp, m_conValue, m_conDisplay;
  
	shared_ptr<Listener> m_lsnTemp, m_lsnValue;
	shared_ptr<Listener> m_lsnDisplay;
  
	void onTempChanged(const Snapshot &shot, XValueNodeBase *);
	void onValueChanged(const Snapshot &shot, XValueNodeBase *);  
	void onDisplayTouched(const Snapshot &shot, XTouchableNode *);
	FrmCalTable *const m_pForm;
	qshared_ptr<FrmGraphNURL> m_waveform;
	const shared_ptr<XWaveNGraph> m_wave;
};

//----------------------------------------------------------------------------
#endif
