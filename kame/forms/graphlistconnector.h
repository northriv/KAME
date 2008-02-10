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
//---------------------------------------------------------------------------

#ifndef graphlistconnectorH
#define graphlistconnectorH

#include "xnodeconnector.h"
//---------------------------------------------------------------------------

class QPushButton;
class QTable;
class XGraphList;

class XGraphListConnector : public XListQConnector
{
	Q_OBJECT
	XQCON_OBJECT
protected:
	XGraphListConnector(const shared_ptr<XGraphList> &node, QTable *item,
						QPushButton *btnnew, QPushButton *btndelete);
public:
	virtual ~XGraphListConnector() {}
protected:
	virtual void onCatch(const shared_ptr<XNode> &node);
	virtual void onRelease(const shared_ptr<XNode> &node);
protected slots:
void clicked ( int row, int col, int button, const QPoint& );
private:
	const shared_ptr<XGraphList> m_graphlist;
  
	const shared_ptr<XNode> m_newGraph;
	const shared_ptr<XNode> m_deleteGraph;
	struct tcons {
		xqcon_ptr conx, cony1, conz;
		shared_ptr<XNode> node;
		QWidget *widget;
	};
	typedef std::deque<tcons> tconslist;
	tconslist m_cons;
    
  
	const xqcon_ptr m_conNewGraph, m_conDeleteGraph;
	shared_ptr<XListener> m_lsnNewGraph, m_lsnDeleteGraph;
  
	void onNewGraph (const shared_ptr<XNode> &);
	void onDeleteGraph (const shared_ptr<XNode> &);
};

#endif
