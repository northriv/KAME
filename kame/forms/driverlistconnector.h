/***************************************************************************
		Copyright (C) 2002-2008 Kentaro Kitagawa
		                   kitag@issp.u-tokyo.ac.jp
		
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

class Q3Table;
class QLabel;

class XDriverListConnector : public XListQConnector
{
	Q_OBJECT
	XQCON_OBJECT
protected:
	XDriverListConnector
	(const shared_ptr<XDriverList> &node, FrmDriver *item);
public:
	virtual ~XDriverListConnector() {}
protected:
	virtual void onCatch(const shared_ptr<XNode> &node);
	virtual void onRelease(const shared_ptr<XNode> &node);
protected slots:
void clicked ( int row, int col, int button, const QPoint& );
private:
	shared_ptr<XNode> m_create, m_release;
  
	struct tcons {
		struct tlisttext {
			QLabel *label;
			shared_ptr<XString> str;
		};
		QLabel *label;
		shared_ptr<XDriver> driver;
		shared_ptr<XTalker<tlisttext> > tlkOnRecordRedirected;
		shared_ptr<XListener> lsnOnRecordRedirected;
		void onRecordRedirected(const tlisttext &);
	};
	typedef std::deque<shared_ptr<tcons> > tconslist;
	tconslist m_cons;
  
	shared_ptr<XListener> m_lsnOnRecord;
	shared_ptr<XListener> m_lsnOnCreateTouched, m_lsnOnReleaseTouched;
  
	const xqcon_ptr m_conCreate, m_conRelease;
	void onRecord(const shared_ptr<XDriver> &driver);
	void onCreateTouched(const shared_ptr<XNode> &);
	void onReleaseTouched(const shared_ptr<XNode> &);
};


#endif
