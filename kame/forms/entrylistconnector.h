/***************************************************************************
		Copyright (C) 2002-2015 Kentaro Kitagawa
		                   kitag@issp.u-tokyo.ac.jp
		
		This program is free software; you can redistribute it and/or
		modify it under the terms of the GNU Library General Public
		License as published by the Free Software Foundation; either
		version 2 of the License, or (at your option) any later version.
		
		You should have received a copy of the GNU Library General 
		Public License and a list of authors along with this program; 
		see the files COPYING and AUTHORS.
***************************************************************************/
#ifndef entrylistconnectorH
#define entrylistconnectorH

#include "xnodeconnector.h"
#include "driver.h"
//---------------------------------------------------------------------------

class QTableWidget;
class XScalarEntry;
class XChartList;
class XScalarEntryList;
class XDriver;

class XEntryListConnector : public XListQConnector {
	Q_OBJECT
public:
	XEntryListConnector
    (const shared_ptr<XScalarEntryList> &node, QTableWidget *item, const shared_ptr<XChartList> &chartlist);
	virtual ~XEntryListConnector() {}
protected:
	virtual void onCatch(const Snapshot &shot, const XListNodeBase::Payload::CatchEvent &e);
	virtual void onRelease(const Snapshot &shot, const XListNodeBase::Payload::ReleaseEvent &e);
protected slots:
void cellClicked ( int row, int col);
private:
	const shared_ptr<XChartList> m_chartList;

	struct tcons {
		xqcon_ptr constore, condelta;
		QLabel *label;
		shared_ptr<XScalarEntry> entry;
		shared_ptr<XDriver> driver;
		shared_ptr<Listener> lsnOnRecord;
	};
	typedef std::deque<shared_ptr<tcons> > tconslist;
	tconslist m_cons;
	void onRecord(const Snapshot &shot, XDriver*);
};

#endif
