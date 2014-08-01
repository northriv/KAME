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
#ifndef driverlistconnectorH
#define driverlistconnectorH

#include "driver.h"
#include "xnodeconnector.h"
//---------------------------------------------------------------------------

class Ui_FrmDriver;
typedef QForm<QWidget, Ui_FrmDriver> FrmDriver;

class QTableWidget;
class QLabel;

class XDriverListConnector : public XListQConnector {
	Q_OBJECT
public:
	XDriverListConnector
	(const shared_ptr<XDriverList> &node, FrmDriver *item);
	virtual ~XDriverListConnector() {}
protected:
	virtual void onCatch(const Snapshot &shot, const XListNodeBase::Payload::CatchEvent &e);
	virtual void onRelease(const Snapshot &shot, const XListNodeBase::Payload::ReleaseEvent &e);
protected slots:
void cellClicked ( int row, int col);
private:
	shared_ptr<XTouchableNode> m_create, m_release;
  
	struct tcons {
		QLabel *label;
		shared_ptr<XDriver> driver;
		shared_ptr<XListener> lsnOnRecord;
	};
	typedef std::deque<shared_ptr<tcons> > tconslist;
	tconslist m_cons;
  
	shared_ptr<XListener> m_lsnOnCreateTouched, m_lsnOnReleaseTouched;
  
	const xqcon_ptr m_conCreate, m_conRelease;
	void onRecord(const Snapshot &shot, XDriver *);
	void onCreateTouched(const Snapshot &shot, XTouchableNode *);
	void onReleaseTouched(const Snapshot &shot, XTouchableNode *);
};


#endif
