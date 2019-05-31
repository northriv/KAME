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
#include "driverlistconnector.h"
#include "driver.h"
#include "measure.h"
#include <QLineEdit>
#include <QPushButton>
#include <QTableWidget>
#include <QLabel>
#include "ui_drivertool.h"
#include "ui_drivercreate.h"
#include "icon.h"
#include <QPainter>
#include <QPixmap>

typedef QForm<QDialog, Ui_DlgCreateDriver> DlgCreateDriver;

XDriverListConnector::XDriverListConnector
(const shared_ptr<XDriverList> &node, FrmDriver *item)
	: XListQConnector(node, item->m_tblDrivers),
	  m_create(XNode::createOrphan<XTouchableNode>("Create", true)),
	  m_release(XNode::createOrphan<XTouchableNode>("Release", true)),
	  m_conCreate(xqcon_create<XQButtonConnector>(m_create, item->m_btnNew)),
	  m_conRelease(xqcon_create<XQButtonConnector>(m_release, item->m_btnDelete))   {

    item->m_btnNew->setIcon(
        QApplication::style()->standardIcon(QStyle::SP_FileDialogStart));
    item->m_btnDelete->setIcon(
        QApplication::style()->standardIcon(QStyle::SP_DialogCloseButton));
    
    connect(m_pItem, SIGNAL( cellClicked( int, int)),
            this, SLOT(cellClicked( int, int)) );
  
    m_pItem->setColumnCount(3);
	double def = 50;
	m_pItem->setColumnWidth(0, (int)(def * 1.5));
	m_pItem->setColumnWidth(1, (int)(def * 1.0));
	m_pItem->setColumnWidth(2, (int)(def * 4.5));
	QStringList labels;
	labels += i18n("Driver");
	labels += i18n("Type");
	labels += i18n("Recorded Time");
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

    m_create->iterate_commit([=](Transaction &tr){
		m_lsnOnCreateTouched = tr[ *m_create].onTouch().connectWeakly(shared_from_this(),
			&XDriverListConnector::onCreateTouched, Listener::FLAG_MAIN_THREAD_CALL);
    });
    m_release->iterate_commit([=](Transaction &tr){
		m_lsnOnReleaseTouched = tr[ *m_release].onTouch().connectWeakly(shared_from_this(),
			&XDriverListConnector::onReleaseTouched, Listener::FLAG_MAIN_THREAD_CALL);
    });
}

void
XDriverListConnector::onCatch(const Snapshot &shot, const XListNodeBase::Payload::CatchEvent &e) {
	shared_ptr<XDriver> driver(static_pointer_cast<XDriver>(e.caught));
  
    int i = m_pItem->rowCount();
    m_pItem->insertRow(i);
    m_pItem->setItem(i, 0, new QTableWidgetItem(driver->getLabel().c_str()));
	// typename is not set at this moment
    m_pItem->setItem(i, 1, new QTableWidgetItem(driver->getTypename().c_str()));

    m_cons.push_back(std::make_shared<tcons>());
    m_cons.back()->label = new QLabel(m_pItem);
	m_pItem->setCellWidget(i, 2, m_cons.back()->label);
	m_cons.back()->driver = driver;
    driver->iterate_commit([=](Transaction &tr){
		m_cons.back()->lsnOnRecord = tr[ *driver].onRecord().connectWeakly(
				shared_from_this(), &XDriverListConnector::onRecord,
				Listener::FLAG_MAIN_THREAD_CALL | Listener::FLAG_AVOID_DUP | Listener::FLAG_DELAY_ADAPTIVE);
    });

    assert(m_pItem->rowCount() == (int)m_cons.size());
}
void
XDriverListConnector::onRelease(const Snapshot &shot, const XListNodeBase::Payload::ReleaseEvent &e) {
	for(auto it = m_cons.begin(); it != m_cons.end();) {
        assert(m_pItem->rowCount() == (int)m_cons.size());
		if(( *it)->driver == e.released) {
            for(int i = 0; i < m_pItem->rowCount(); i++) {
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
XDriverListConnector::cellClicked ( int row, int col) {
	for(auto it = m_cons.begin(); it != m_cons.end(); it++) {
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
    qshared_ptr<DlgCreateDriver> dlg(new DlgCreateDriver(m_pItem));
	dlg->setModal(true);
    static int num = 0;
	num++;
	dlg->m_edName->setText(QString("NewDriver%1").arg(num));
   
    auto iconMaker = [](const QString &str, QColor clr = 0x808080u){
        QPixmap pixmap(96, 96);
        pixmap.fill(Qt::transparent);
        QPainter painter( &pixmap);
        QFont font(painter.font());
        font.setPixelSize(std::min(48, 92 / str.length()));
        painter.setFont(font);
        font.setBold(true);
        QPen pen(clr);
        painter.setPen(pen);
        painter.setRenderHint(QPainter::Antialiasing);
        painter.drawText(pixmap.rect(), str, QTextOption(Qt::AlignVCenter|Qt::AlignHCenter));
        return pixmap;
    };
	dlg->m_lstType->clear();
    for(auto &&label: XDriverList::typelabels()) {
        QPixmap icon;
        if(label.find("temp") != std::string::npos)
            icon = iconMaker("TEMP", 0xa00000u);
        if(label.find("magnet power") != std::string::npos)
            icon = iconMaker("MAG", 0x800080u);
        if(label.find("DMM") != std::string::npos)
            icon = iconMaker("DMM", 0x000000u);
        if(label.find("Network Analyzer") != std::string::npos)
            icon = iconMaker("NA", 0x008080u);
        if(label.find("signal generator") != std::string::npos)
            icon = iconMaker("SG", 0x00a080u);
        if(label.find("DSO") != std::string::npos)
            icon = iconMaker("DSO", 0xa0a000u);
        if(label.find("NMR") != std::string::npos || label.find("Thamway") != std::string::npos)
            icon = iconMaker("NMR", 0x000080u);
        if(icon.isNull())
            icon = iconMaker(label.substr(0, 1).c_str());
        new QListWidgetItem(icon, label.c_str(), dlg->m_lstType);
    }
   
    dlg->m_lstType->setCurrentRow(-1);
	if(dlg->exec() == QDialog::Rejected) {
		return;
	}
    int idx = dlg->m_lstType->currentRow();
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
