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
//---------------------------------------------------------------------------
#include "interfacelistconnector.h"
#include "driver.h"
#include "icon.h"

#include <QLineEdit>
#include <QComboBox>
#include <QPushButton>
#include <QSpinBox>
#include <QTableWidget>
#include <QApplication>

XInterfaceListConnector::XInterfaceListConnector(
    const shared_ptr<XInterfaceList> &node, QTableWidget *item)
	: XListQConnector(node, item), m_interfaceList(node) {
    connect(m_pItem, SIGNAL(cellClicked( int, int)),
            this, SLOT(cellClicked( int, int)) );
	item->setColumnCount(5);
	double def = 45;
	item->setColumnWidth(0, (int)(def * 1.5));
    item->setColumnWidth(1, (int)(def * 1.5));
	item->setColumnWidth(2, (int)(def * 2));
	item->setColumnWidth(3, (int)(def * 2));
	item->setColumnWidth(4, (int)(def * 1));
	QStringList labels;
	labels += i18n("Driver");
	labels += i18n("Control");
	labels += i18n("Device");
	labels += i18n("Port");
	labels += i18n("Addr");
	item->setHorizontalHeaderLabels(labels);

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
	for(auto it = m_cons.begin(); it != m_cons.end(); it++) {
		if(it->interface->control().get() == node) {
            if(shot[ *it->interface->control()]) {
                if(shot[ *node].isUIEnabled()) {
                    it->btn->setIcon( QIcon( *g_pIconRotate) );
                    it->btn->setText(i18n("&STOP"));
                }
                else {
                    it->btn->setIcon(QApplication::style()->standardIcon(QStyle::SP_MediaStop));
                    it->btn->setText(i18n("&STOP"));
                }
			}
			else {
                it->btn->setIcon(
                    QApplication::style()->standardIcon(QStyle::SP_MediaPlay));
                it->btn->setText(i18n("&RUN"));
			}
		}
	}
}
void
XInterfaceListConnector::onCatch(const Snapshot &shot, const XListNodeBase::Payload::CatchEvent &e) {
	auto interface = static_pointer_cast<XInterface>(e.caught);
	int i = m_pItem->rowCount();
	m_pItem->insertRow(i);
    m_pItem->setItem(i, 0, new QTableWidgetItem(interface->getLabel().c_str()));
	m_cons.push_back(tcons());
	tcons &con(m_cons.back());
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
    //Ranges should be preset in prior to connectors.
    numAddr->setRange(0, 255);
	numAddr->setSingleStep(1);
    con.conaddr = xqcon_create<XQSpinBoxUnsignedConnector>(interface->address(), numAddr);
	m_pItem->setCellWidget(i, 4, numAddr);
    {
        Snapshot shot = interface->iterate_commit([=, &con](Transaction &tr){
            con.lsnOnControlChanged = tr[ *interface->control()].onValueChanged().connectWeakly(
                shared_from_this(), &XInterfaceListConnector::onControlChanged,
                Listener::FLAG_MAIN_THREAD_CALL);
        });
        onControlChanged(shot, interface->control().get());
    }
}
void
XInterfaceListConnector::onRelease(const Snapshot &shot, const XListNodeBase::Payload::ReleaseEvent &e) {
	for(auto it = m_cons.begin(); it != m_cons.end();) {
		if(it->interface == e.released) {
			for(int i = 0; i < m_pItem->rowCount(); i++) {
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
XInterfaceListConnector::cellClicked ( int row, int ) {
	for(auto it = m_cons.begin(); it != m_cons.end(); it++)
	{
		if(m_pItem->cellWidget(row, 1) == it->btn)
			it->interface->driver()->showForms();
	}
}
