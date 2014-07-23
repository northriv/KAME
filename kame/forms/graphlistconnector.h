/***************************************************************************
		Copyright (C) 2002-2013 Kentaro Kitagawa
		                   kitag@kochi-u.ac.jp
		
		This program is free software; you can redistribute it and/or
		modify it under the terms of the GNU Library General Public
		License as published by the Free Software Foundation; either
		version 2 of the License, or (at your option) any later version.
		
		You should have received a copy of the GNU Library General 
		Public License and a list of authors along with this program; 
		see the files COPYING and AUTHORS.
***************************************************************************/
//---------------------------------------------------------------------------

#ifndef graphlistconnectorH
#define graphlistconnectorH

#include "xnodeconnector.h"
//---------------------------------------------------------------------------

class QPushButton;
class QTableWidget;
class XGraphList;

class XGraphListConnector : public XListQConnector {
	Q_OBJECT
public:
    XGraphListConnector(const shared_ptr<XGraphList> &node, QTableWidget *item,
						QPushButton *btnnew, QPushButton *btndelete);
	virtual ~XGraphListConnector() {}
protected:
	virtual void onCatch(const Snapshot &shot, const XListNodeBase::Payload::CatchEvent &e);
	virtual void onRelease(const Snapshot &shot, const XListNodeBase::Payload::ReleaseEvent &e);
protected slots:
void cellClicked ( int row, int col);
private:
	const shared_ptr<XGraphList> m_graphlist;
  
	const shared_ptr<XTouchableNode> m_newGraph;
	const shared_ptr<XTouchableNode> m_deleteGraph;
	struct tcons {
		xqcon_ptr conx, cony1, conz;
		shared_ptr<XNode> node;
		QWidget *widget;
	};
	typedef std::deque<tcons> tconslist;
	tconslist m_cons;
    
  
	const xqcon_ptr m_conNewGraph, m_conDeleteGraph;
	shared_ptr<XListener> m_lsnNewGraph, m_lsnDeleteGraph;
  
	void onNewGraph (const Snapshot &shot, XTouchableNode *);
	void onDeleteGraph (const Snapshot &shot, XTouchableNode *);
};

#endif
