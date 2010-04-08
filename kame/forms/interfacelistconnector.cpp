/***************************************************************************
		Copyright (C) 2002-2010 Kentaro Kitagawa
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
#include "interfacelistconnector.h"
#include "driver.h"

#include <q3table.h>
#include <QLineEdit>
#include <QComboBox>
#include <QPushButton>
#include <QSpinBox>
#include <kiconloader.h>

XInterfaceListConnector::XInterfaceListConnector(
    const shared_ptr<XInterfaceList> &node, Q3Table *item)
	: XListQConnector(node, item), m_interfaceList(node) {
	connect(m_pItem, SIGNAL( clicked( int, int, int, const QPoint& )),
			this, SLOT(clicked( int, int, int, const QPoint& )) );
	item->setNumCols(5);
	double def = 50;
	item->setColumnWidth(0, (int)(def * 1.5));
	item->setColumnWidth(1, (int)(def * 1.2));
	item->setColumnWidth(2, (int)(def * 2));
	item->setColumnWidth(3, (int)(def * 1));
	item->setColumnWidth(4, (int)(def * 1));
	QStringList labels;
	labels += i18n("Driver");
	labels += i18n("Control");
	labels += i18n("Device");
	labels += i18n("Port");
	labels += i18n("Addr");
	item->setColumnLabels(labels);

	Snapshot shot( *node);
	if(shot.size()) {
		for(int idx = 0; idx < shot.size(); ++idx) {
			XListNodeBase::Payload::CatchEvent e;
			e.emitter = node.get();
			e.caught = shot.list()->at(idx);
			e.index = idx;
			onCatch(shot, e);
		}
	}
}

void
XInterfaceListConnector::onControlChanged(const Snapshot &shot, XValueNodeBase *node) {
	for(tconslist::iterator it = m_cons.begin(); it != m_cons.end(); it++) {
		if(it->interface->control().get() == node) {
		    KIconLoader *loader = KIconLoader::global();
			if(shot[ *it->interface->control()]) {
				it->btn->setIcon( loader->loadIcon("stop",
					KIconLoader::Toolbar, KIconLoader::SizeSmall, true ) );
				it->btn->setText(i18n("&STOP"));
			}
			else {
				it->btn->setIcon( loader->loadIcon("run",
					KIconLoader::Toolbar, KIconLoader::SizeSmall, true ) );
				it->btn->setText(i18n("&RUN"));
			}
		}
	}
}
void
XInterfaceListConnector::onCatch(const Snapshot &shot, const XListNodeBase::Payload::CatchEvent &e) {
	shared_ptr<XInterface> interface = static_pointer_cast<XInterface>(e.caught);
	int i = m_pItem->numRows();
	m_pItem->insertRows(i);
	m_pItem->setText(i, 0, interface->getLabel().c_str());
	struct tcons con;
	con.interface = interface;
	con.btn = new QPushButton(m_pItem);
	con.btn->setCheckable(true);
	con.btn->setAutoDefault(false);
	con.btn->setFlat(true);
	con.concontrol = xqcon_create<XQToggleButtonConnector>(interface->control(), con.btn);    
	m_pItem->setCellWidget(i, 1, con.btn);
	QComboBox *cmbdev(new QComboBox(m_pItem) );
	con.condev = xqcon_create<XQComboBoxConnector>(interface->device(), cmbdev, Snapshot( *interface->device()));
	m_pItem->setCellWidget(i, 2, cmbdev);
	QLineEdit *edPort(new QLineEdit(m_pItem) );
	con.conport = xqcon_create<XQLineEditConnector>(interface->port(), edPort, false);
	m_pItem->setCellWidget(i, 3, edPort);
	QSpinBox *numAddr(new QSpinBox(m_pItem) );
	numAddr->setRange(0, 32);
	numAddr->setSingleStep(1);
	con.conaddr = xqcon_create<XQSpinBoxConnector>(interface->address(), numAddr);
	m_pItem->setCellWidget(i, 4, numAddr);
	m_cons.push_back(con);
	for(Transaction tr( *interface);; ++tr) {
		con.lsnOnControlChanged = tr[ *interface->control()].onValueChanged().connectWeakly(
			shared_from_this(), &XInterfaceListConnector::onControlChanged,
			XListener::FLAG_MAIN_THREAD_CALL);
		if(tr.commit()) {
			onControlChanged(tr, interface->control().get());
			break;
		}
	}
}
void
XInterfaceListConnector::onRelease(const Snapshot &shot, const XListNodeBase::Payload::ReleaseEvent &e) {
	for(tconslist::iterator it = m_cons.begin(); it != m_cons.end();) {
		if(it->interface == e.released) {
			for(int i = 0; i < m_pItem->numRows(); i++) {
				if(m_pItem->cellWidget(i, 1) == it->btn) m_pItem->removeRow(i);
			}
			it = m_cons.erase(it);
		}
		else {
			it++;
		}    
	}  
}
void
XInterfaceListConnector::clicked ( int row, int , int , const QPoint& ) {
	for(tconslist::iterator it = m_cons.begin(); it != m_cons.end(); it++)
	{
		if(m_pItem->cellWidget(row, 1) == it->btn)
			it->interface->driver()->showForms();
	}
}
