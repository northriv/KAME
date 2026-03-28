/***************************************************************************
        Copyright (C) 2002-2026 Kentaro Kitagawa
		                   kitag@issp.u-tokyo.ac.jp

		This program is free software; you can redistribute it and/or
		modify it under the terms of the GNU Library General Public
		License as published by the Free Software Foundation; either
		version 2 of the License, or (at your option) any later version.

		You should have received a copy of the GNU Library General
		Public License and a list of authors along with this program;
		see the files COPYING and AUTHORS.
***************************************************************************/
#include "calibentryconnector.h"
#include "analyzer.h"
#include "thermometer.h"

#include <QComboBox>
#include <QPushButton>
#include <QTableWidget>
#include <QApplication>

XCalibratedEntryListConnector::XCalibratedEntryListConnector(
    const shared_ptr<XCalibratedEntryList> &list,
    QTableWidget *table, QPushButton *btnNew, QPushButton *btnDelete)
    : XListQConnector(list, table),
      m_list(list),
      m_newEntry(XNode::createOrphan<XTouchableNode>("NewCalibEntry", true)),
      m_deleteEntry(XNode::createOrphan<XTouchableNode>("DeleteCalibEntry", true)),
      m_conNew(xqcon_create<XQButtonConnector>(m_newEntry, btnNew)),
      m_conDelete(xqcon_create<XQButtonConnector>(m_deleteEntry, btnDelete)) {

    btnNew->setIcon(QApplication::style()->standardIcon(QStyle::SP_FileDialogStart));
    btnDelete->setIcon(QApplication::style()->standardIcon(QStyle::SP_DialogCloseButton));
    btnNew->setEnabled(true);
    btnDelete->setEnabled(true);

    m_pItem->setColumnCount(3);
    m_pItem->setColumnWidth(0, 100);
    m_pItem->setColumnWidth(1, 100);
    m_pItem->setColumnWidth(2, 100);
    QStringList labels;
    labels += i18n("Name");
    labels += i18n("Source Entry");
    labels += i18n("Calibration");
    m_pItem->setHorizontalHeaderLabels(labels);

    connect(table, SIGNAL(cellClicked(int,int)), this, SLOT(cellClicked(int,int)));

    Snapshot shot( *list);
    if(shot.size()) {
        for(int idx = 0; idx < (int)shot.size(); ++idx) {
            XListNodeBase::Payload::CatchEvent e;
            e.emitter = list.get();
            e.caught = shot.list()->at(idx);
            e.index = idx;
            onCatch(shot, e);
        }
    }

    m_newEntry->iterate_commit([=](Transaction &tr){
        m_lsnNew = tr[ *m_newEntry].onTouch().connectWeakly(
            shared_from_this(), &XCalibratedEntryListConnector::onNew,
            Listener::FLAG_MAIN_THREAD_CALL);
    });
    m_deleteEntry->iterate_commit([=](Transaction &tr){
        m_lsnDelete = tr[ *m_deleteEntry].onTouch().connectWeakly(
            shared_from_this(), &XCalibratedEntryListConnector::onDelete,
            Listener::FLAG_MAIN_THREAD_CALL);
    });
}

void
XCalibratedEntryListConnector::onNew(const Snapshot &, XTouchableNode *) {
    static int num = 1;
    m_list->createByTypename("", formatString("CalibEntry%d", num++));
}

void
XCalibratedEntryListConnector::onDelete(const Snapshot &, XTouchableNode *) {
    int n = m_pItem->currentRow();
    Snapshot shot( *m_list);
    if(shot.size() && n >= 0 && n < (int)shot.list()->size())
        m_list->release(shot.list()->at(n));
}

void
XCalibratedEntryListConnector::cellClicked(int, int) {}

void
XCalibratedEntryListConnector::onCatch(const Snapshot &, const XListNodeBase::Payload::CatchEvent &e) {
    auto entry = static_pointer_cast<XCalibratedEntry>(e.caught);
    int i = m_pItem->rowCount();
    m_pItem->insertRow(i);
    m_pItem->setItem(i, 0, new QTableWidgetItem(entry->getLabel().c_str()));

    Snapshot shot_entries( *m_list->entries());
    Snapshot shot_curves( *m_list->curves());

    tcons con;
    con.node = e.caught;

    QComboBox *cmbSrc = new QComboBox(m_pItem);
    con.consrc = xqcon_create<XQComboBoxConnector>(entry->source(), cmbSrc, shot_entries);
    m_pItem->setCellWidget(i, 1, cmbSrc);

    QComboBox *cmbCurve = new QComboBox(m_pItem);
    con.concurve = xqcon_create<XQComboBoxConnector>(entry->curve(), cmbCurve, shot_curves);
    m_pItem->setCellWidget(i, 2, cmbCurve);

    con.widget = cmbSrc;
    m_cons.push_back(con);
}

void
XCalibratedEntryListConnector::onRelease(const Snapshot &, const XListNodeBase::Payload::ReleaseEvent &e) {
    for(auto it = m_cons.begin(); it != m_cons.end(); ++it) {
        if(it->node == e.released) {
            for(int i = 0; i < m_pItem->rowCount(); ++i) {
                if(m_pItem->cellWidget(i, 1) == it->widget) {
                    m_pItem->removeRow(i);
                    break;
                }
            }
            m_cons.erase(it);
            return;
        }
    }
}
