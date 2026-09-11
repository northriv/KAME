/***************************************************************************
		Copyright (C) 2002-2015 Kentaro Kitagawa
		                   kitag@issp.u-tokyo.ac.jp
		
		This program is free software; you can redistribute it and/or
		modify it under the terms of the GNU General Public
		License as published by the Free Software Foundation; either
		version 2 of the License, or (at your option) any later version.
		
		You should have received a copy of the GNU General 
		Public License and a list of authors along with this program; 
		see the files COPYING and AUTHORS.
***************************************************************************/
//---------------------------------------------------------------------------
#ifdef USE_PYBIND11
    #include <pybind11/pybind11.h>
#endif
#include "driverlistconnector.h"
#include "driver.h"
#include "measure.h"
#include "interface.h"
#include "kame.h"
#include <QLineEdit>
#include <QListWidget>
#include <QPushButton>
#include <QTableWidget>
#include <QHeaderView>
#include <QLabel>
#include "ui_drivertool.h"
#include "ui_drivercreate.h"
#include "icon.h"
#include <QPainter>
#include <QPixmap>

#include <iostream>
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
    //Every section must stay Interactive (Qt's default) to be draggable by the
    //user; ResizeToContents / Fixed / Stretch all disable mouse resizing, which
    //is why this list alone could not be adjusted.  The last section still
    //fills the remaining width, and onCatch() grows column 0 to fit long
    //driver names (grow-only, so a manual width is never taken back).
    m_pItem->horizontalHeader()->setSectionResizeMode(QHeaderView::Interactive);
    m_pItem->horizontalHeader()->setStretchLastSection(true);

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

    //Widens the name column when a long driver name does not fit; never
    //narrows it, so a width chosen by the user with the mouse survives.
    int hint = m_pItem->fontMetrics().horizontalAdvance(driver->getLabel().c_str()) + 24;
    if(hint > m_pItem->columnWidth(0))
        m_pItem->setColumnWidth(0, hint);

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
			if(col < 3) {
				( *it)->driver->showForms();
				//The toolbox has just done its job and is now standing in
				//front of the result.
				if(auto *frm = dynamic_cast<FrmKameMain *>(g_pFrmMain))
					frm->foldToolboxes();
			}
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
        font.setPixelSize(std::min(48, 92 / (int)str.length()));
        painter.setFont(font);
        font.setBold(true);
        QPen pen(clr);
        painter.setPen(pen);
        painter.setRenderHint(QPainter::Antialiasing);
        painter.drawText(pixmap.rect(), str, QTextOption(Qt::AlignVCenter|Qt::AlignHCenter));
        return pixmap;
    };
	dlg->m_lstType->clear();
    auto labels_unsort = static_pointer_cast<XDriverList>(m_list)->typelabels();
    auto typenames_unsort = static_pointer_cast<XDriverList>(m_list)->typenames();
    std::map<std::string, std::string> map;//sorts by label.
    for(unsigned int i = 0; i < std::min(typenames_unsort.size(), labels_unsort.size()); ++i) {
        map.insert(std::make_pair(labels_unsort[i], typenames_unsort[i]));
    }
    for(auto &&x: map) {
        auto &label = x.first;
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
        if(label.find("ODMR") != std::string::npos)
            icon = iconMaker("ODMR", 0x000080u);
        if(icon.isNull())
            icon = iconMaker(label.substr(0, 1).c_str());
        auto *item = new QListWidgetItem(icon, label.c_str(), dlg->m_lstType);
        //Carry the type name on the item itself.  Resolving the choice by row
        //number would break the moment the list is filtered, and the search
        //below matches against this too — a driver may be known by the name
        //that appears in a .kam file rather than by its label.
        item->setData(Qt::UserRole, QString(x.second.c_str()));
    }

    //Live search: every space-separated word must appear somewhere in the
    //label or the type name, so "agilent dmm" narrows to one line however the
    //label happens to be worded.
    DlgCreateDriver *d = dlg.get();
    connect(d->m_edSearch, &QLineEdit::textChanged, d, [d](const QString &text){
        QStringList words = text.simplified().split(' ', Qt::SkipEmptyParts);
        int first_shown = -1;
        for(int i = 0; i < d->m_lstType->count(); ++i) {
            QListWidgetItem *item = d->m_lstType->item(i);
            QString hay = item->text() + " " + item->data(Qt::UserRole).toString();
            bool shown = true;
            for(auto &&w: words)
                if( !hay.contains(w, Qt::CaseInsensitive)) {shown = false; break;}
            item->setHidden( !shown);
            if(shown && (first_shown < 0)) first_shown = i;
        }
        //Keeps Enter meaningful while typing: with a search in progress the
        //first match is selected, so the dialog's default button creates it.
        //An empty box goes back to demanding a deliberate choice.
        QListWidgetItem *cur = d->m_lstType->currentItem();
        if(words.isEmpty())
            d->m_lstType->setCurrentRow(-1);
        else if( !cur || cur->isHidden())
            d->m_lstType->setCurrentRow(first_shown);
    });
    d->m_edSearch->setFocus();
   
    dlg->m_lstType->setCurrentRow(-1);
	if(dlg->exec() == QDialog::Rejected) {
		return;
	}
    QListWidgetItem *chosen = dlg->m_lstType->currentItem();
	shared_ptr<XNode> driver;
    if(chosen && !chosen->isHidden()) {
        XString type = chosen->data(Qt::UserRole).toString().toUtf8().data();
        if(m_list->getChild(dlg->m_edName->text().toUtf8().data())) {
	        gErrPrint(i18n("Duplicated name."));
		}
		else {
            try {
               driver = m_list->createByTypename(type,
											  dlg->m_edName->text().toUtf8().data());
            }
#ifdef USE_PYBIND11
            catch (pybind11::error_already_set& e) {
                pybind11::gil_scoped_acquire guard;
                gErrPrint(i18n("Python error: ") + e.what());
            }
#endif
            catch (std::runtime_error &e) {
                gErrPrint(i18n("Python KAME binding error: ") + e.what());
            }
            catch (...) {
                gErrPrint(i18n("Unknown python error."));
            }
        }
	}
	if( !driver) {
        gErrPrint(i18n("Driver creation failed."));
        return;
    }
    //Creating a driver by hand means configuring it next, so open its own
    //window.  Loading a .kam deliberately does not: it would throw open a
    //window for every driver in the file.
    if(auto drv = dynamic_pointer_cast<XDriver>(driver))
        drv->showForms();

    //A driver that came with an interface cannot be started until its port is
    //set, so put the Interface pane in front — after the driver's own window,
    //so that is what ends up with the keyboard.  A driver with no interface
    //leaves its own window in front instead, there being no port to set.  One that talks to no hardware
    //of its own — an analysis or management driver — leaves the layout alone.
    //An interface is a child node of its driver (XCharDeviceDriver and
    //XDummyDriver both create it as one), so the driver itself can be asked.
    Snapshot shot_driver( *driver);
    bool has_interface = false;
    if(shot_driver.size())
        for(auto &&child: *shot_driver.list())
            if(dynamic_pointer_cast<XInterface>(child)) {has_interface = true; break;}
    if(has_interface)
        if(auto *frm = dynamic_cast<FrmKameMain *>(g_pFrmMain))
            frm->revealInterfacePane();
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
