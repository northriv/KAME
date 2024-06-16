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
#include "entrylistconnector.h"
#include "analyzer.h"
#include "driver.h"
#include <QLabel>
#include <QPushButton>
#include <QCheckBox>
#include <QTableWidget>
#include <QDoubleSpinBox>

//---------------------------------------------------------------------------

XEntryListConnector::XEntryListConnector
(const shared_ptr<XScalarEntryList> &node, QTableWidget *item, const shared_ptr<XChartList> &chartlist)
	: XListQConnector(node, item),
	  m_chartList(chartlist) {
    connect(item, SIGNAL( cellClicked( int, int)),
            this, SLOT(cellClicked( int, int)) );
	m_pItem->setColumnCount(4);
	const double def = 50;
	m_pItem->setColumnWidth(0, (int)(def * 2.5));
	m_pItem->setColumnWidth(1, (int)(def * 2.0));
	m_pItem->setColumnWidth(2, (int)(def * 0.8));
	m_pItem->setColumnWidth(3, (int)(def * 2.5));
	QStringList labels;
	labels += i18n("Entry");
	labels += i18n("Value");
	labels += i18n("Store");
	labels += i18n("Delta");
	m_pItem->setHorizontalHeaderLabels(labels);
  
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
XEntryListConnector::onRecord(const Snapshot &shot, XDriver *driver) {
	for(tconslist::iterator it = m_cons.begin(); it != m_cons.end(); it++) {
		if(( *it)->entry->driver().get() == driver) {
            try {
                ( *it)->label->setText(shot.at( *( *it)->entry->value()).to_str());
            }
            catch (XNode::NodeNotFoundError &e) {
            }
		}
	}
}

void
XEntryListConnector::cellClicked ( int row, int col) {
	switch(col) {
	case 0:
	case 1: {
			Snapshot shot( *m_chartList);
			if(shot.size()) {
				if((row >= 0) && (row < (int)shot.list()->size())) {
					dynamic_pointer_cast<XValChart>(shot.list()->at(row))->showChart();
				}
			}
		}
	break;
	default:
		break;
	}
}
void
XEntryListConnector::onRelease(const Snapshot &shot, const XListNodeBase::Payload::ReleaseEvent &e) {
	for(tconslist::iterator it = m_cons.begin(); it != m_cons.end();) {
		assert(m_pItem->rowCount() == (int)m_cons.size());
		if(( *it)->entry == e.released) {
			for(int i = 0; i < m_pItem->rowCount(); i++) {
				if(m_pItem->cellWidget(i, 1) == ( *it)->label) m_pItem->removeRow(i);
			}
			it = m_cons.erase(it);
		}
		else {
			it++;
		}
	}
}
void
XEntryListConnector::onCatch(const Snapshot &shot, const XListNodeBase::Payload::CatchEvent &e) {
	shared_ptr<XScalarEntry> entry = static_pointer_cast<XScalarEntry>(e.caught);
	int i = m_pItem->rowCount();
	m_pItem->insertRow(i);
    m_pItem->setItem(i, 0, new QTableWidgetItem(entry->getLabel().c_str()));

	shared_ptr<XDriver> driver = entry->driver();

    m_cons.push_back(std::make_shared<tcons>());
	m_cons.back()->entry = entry;
	m_cons.back()->label = new QLabel(m_pItem);
	m_pItem->setCellWidget(i, 1, m_cons.back()->label);
	QCheckBox *ckbStore = new QCheckBox(m_pItem);
	m_cons.back()->constore = xqcon_create<XQToggleButtonConnector>(entry->store(), ckbStore);
	m_pItem->setCellWidget(i, 2, ckbStore);
	QDoubleSpinBox *numDelta = new QDoubleSpinBox(m_pItem);
    //Ranges should be preset in prior to connectors.
    numDelta->setRange(-1, 1e4);
	numDelta->setSingleStep(1);
	numDelta->setValue(0);
	numDelta->setDecimals(5);
	m_cons.back()->condelta = xqcon_create<XQDoubleSpinBoxConnector>(entry->delta(), numDelta);
	m_pItem->setCellWidget(i, 3, numDelta);
	m_cons.back()->driver = driver;
    driver->iterate_commit([=](Transaction &tr){
		m_cons.back()->lsnOnRecord = tr[ *driver].onRecord().connectWeakly(
				shared_from_this(), &XEntryListConnector::onRecord,
				Listener::FLAG_MAIN_THREAD_CALL | Listener::FLAG_AVOID_DUP | Listener::FLAG_DELAY_ADAPTIVE);
    });

	assert(m_pItem->rowCount() == (int)m_cons.size());
}

