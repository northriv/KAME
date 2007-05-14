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
#ifndef INTERFACELISTCONNECTOR_H_
#define INTERFACELISTCONNECTOR_H_

#include "interface.h"
#include "xnodeconnector.h"

class QTable;
class QPushButton;

class XInterfaceListConnector : public XListQConnector
{
	Q_OBJECT
	XQCON_OBJECT
protected:
	XInterfaceListConnector(const shared_ptr<XInterfaceList> &node, QTable *item);
public:
	virtual ~XInterfaceListConnector() {}
protected:
	virtual void onCatch(const shared_ptr<XNode> &node);
	virtual void onRelease(const shared_ptr<XNode> &node);
protected slots:
void clicked ( int row, int col, int button, const QPoint& );
private:
	struct tcons {
		xqcon_ptr condev, concontrol, conport, conaddr;
		shared_ptr<XInterface> interface;
		QPushButton *btn;
		shared_ptr<XListener> lsnOnControlChanged;
	};
	typedef std::deque<tcons> tconslist;
	tconslist m_cons;

	const shared_ptr<XInterfaceList> m_interfaceList;
	void onControlChanged(const shared_ptr<XValueNodeBase> &);
};

#endif /*INTERFACELISTCONNECTOR_H_*/
