/***************************************************************************
		Copyright (C) 2002-2011 Kentaro Kitagawa
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
#include "driverlistconnector.h"
#include "driver.h"
#include "measure.h"
#include <qlineedit.h>
#include <qpushbutton.h>
#include <q3table.h>
#include <q3listbox.h>
#include <qlabel.h>
#include <kiconloader.h>
#include "ui_drivertool.h"
#include "ui_drivercreate.h"

typedef QForm<QDialog, Ui_DlgCreateDriver> DlgCreateDriver;

XDriverListConnector::XDriverListConnector
(const shared_ptr<XDriverList> &node, FrmDriver *item)
	: XListQConnector(node, item->m_tblDrivers),
	  m_create(XNode::createOrphan<XTouchableNode>("Create", true)),
	  m_release(XNode::createOrphan<XTouchableNode>("Release", true)),
	  m_conCreate(xqcon_create<XQButtonConnector>(m_create, item->m_btnNew)),
	  m_conRelease(xqcon_create<XQButtonConnector>(m_release, item->m_btnDelete))   {

    KIconLoader *loader = KIconLoader::global();
	item->m_btnNew->setIcon( loader->loadIcon("filenew",
					KIconLoader::Toolbar, KIconLoader::SizeSmall, true ) );
	item->m_btnDelete->setIcon( loader->loadIcon("fileclose",
					KIconLoader::Toolbar, KIconLoader::SizeSmall, true ) );
    
	connect(m_pItem, SIGNAL( clicked( int, int, int, const QPoint& )),
			this, SLOT(clicked( int, int, int, const QPoint& )) );
  
	m_pItem->setNumCols(3);
	double def = 50;
	m_pItem->setColumnWidth(0, (int)(def * 1.5));
	m_pItem->setColumnWidth(1, (int)(def * 1.0));
	m_pItem->setColumnWidth(2, (int)(def * 4.5));
	QStringList labels;
	labels += i18n("Driver");
	labels += i18n("Type");
	labels += i18n("Recorded Time");
	m_pItem->setColumnLabels(labels);

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

	for(Transaction tr( *m_create);; ++tr) {
		m_lsnOnCreateTouched = tr[ *m_create].onTouch().connectWeakly(shared_from_this(),
			&XDriverListConnector::onCreateTouched, XListener::FLAG_MAIN_THREAD_CALL);
		if(tr.commit())
			break;
	}
	for(Transaction tr( *m_release);; ++tr) {
		m_lsnOnReleaseTouched = tr[ *m_release].onTouch().connectWeakly(shared_from_this(),
			&XDriverListConnector::onReleaseTouched, XListener::FLAG_MAIN_THREAD_CALL);
		if(tr.commit())
			break;
	}
}

void
XDriverListConnector::onCatch(const Snapshot &shot, const XListNodeBase::Payload::CatchEvent &e) {
	shared_ptr<XDriver> driver(static_pointer_cast<XDriver>(e.caught));
  
	int i = m_pItem->numRows();
	m_pItem->insertRows(i);
	m_pItem->setText(i, 0, driver->getLabel().c_str());
	// typename is not set at this moment
	m_pItem->setText(i, 1, driver->getTypename().c_str());

	m_cons.push_back(shared_ptr<tcons>(new tcons));
	m_cons.back()->label = new QLabel(m_pItem);
	m_pItem->setCellWidget(i, 2, m_cons.back()->label);
	m_cons.back()->driver = driver;
	for(Transaction tr( *driver);; ++tr) {
		m_cons.back()->lsnOnRecord = tr[ *driver].onRecord().connectWeakly(
				shared_from_this(), &XDriverListConnector::onRecord,
				XListener::FLAG_MAIN_THREAD_CALL | XListener::FLAG_AVOID_DUP | XListener::FLAG_DELAY_ADAPTIVE);
		if(tr.commit())
			break;
	}

	ASSERT(m_pItem->numRows() == (int)m_cons.size());
}
void
XDriverListConnector::onRelease(const Snapshot &shot, const XListNodeBase::Payload::ReleaseEvent &e) {
	for(tconslist::iterator it = m_cons.begin(); it != m_cons.end();) {
		ASSERT(m_pItem->numRows() == (int)m_cons.size());
		if(( *it)->driver == e.released) {
			for(int i = 0; i < m_pItem->numRows(); i++) {
				if(m_pItem->cellWidget(i, 2) == ( *it)->label)
					m_pItem->removeRow(i);
			}
			it = m_cons.erase(it);
		}
		else
			it++;
	}
}
void
XDriverListConnector::clicked ( int row, int col, int , const QPoint& ) {
	for(tconslist::iterator it = m_cons.begin(); it != m_cons.end(); it++) {
		if(m_pItem->cellWidget(row, 2) == ( *it)->label) {
			if(col < 3) ( *it)->driver->showForms();
		}
	}
}

void
XDriverListConnector::onRecord(const Snapshot &shot, XDriver *driver) {
	for(tconslist::iterator it = m_cons.begin(); it != m_cons.end(); it++) {
		if(( *it)->driver.get() == driver) {
			( *it)->label->setText(shot[ *driver].time().getTimeStr());
		}
	}
}
void
XDriverListConnector::onCreateTouched(const Snapshot &shot, XTouchableNode *) {
	qshared_ptr<DlgCreateDriver> dlg(new DlgCreateDriver(g_pFrmMain));
	dlg->setModal(true);
	static int num = 0;
	num++;
	dlg->m_edName->setText(QString("NewDriver%1").arg(num));
   
	dlg->m_lstType->clear();
	for(unsigned int i = 0; i < XDriverList::typelabels().size(); i++) {
        dlg->m_lstType->insertItem(XDriverList::typelabels()[i].c_str());
	}
   
	dlg->m_lstType->setCurrentItem(-1);
	if(dlg->exec() == QDialog::Rejected) {
		return;
	}
	int idx = dlg->m_lstType->currentItem();
	shared_ptr<XNode> driver;
	if((idx >= 0) && (idx < (int)XDriverList::typenames().size())) {
		if(m_list->getChild(dlg->m_edName->text().toUtf8().data())) {
	        gErrPrint(i18n("Duplicated name."));
		}
		else {
	       driver = m_list->createByTypename(XDriverList::typenames()[idx],
											  dlg->m_edName->text().toUtf8().data());
		}
	}
	if( !driver)
        gErrPrint(i18n("Driver creation failed."));
}
void
XDriverListConnector::onReleaseTouched(const Snapshot &shot, XTouchableNode *) {
    shared_ptr<XDriver> driver;
	for(tconslist::iterator it = m_cons.begin(); it != m_cons.end(); it++) {
		if(( *it)->label == m_pItem->cellWidget(m_pItem->currentRow(), 2)) {
			driver = ( *it)->driver;
		}
	}    
    if(driver) m_list->release(driver);
}
