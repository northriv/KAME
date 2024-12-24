/***************************************************************************
        Copyright (C) 2002-2024 Kentaro Kitagawa
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
#ifdef USE_PYBIND11
    #include <pybind11/pybind11.h>
#endif

#include "graphlistconnector.h"

#include <QComboBox>
#include <QPushButton>
#include <QTableWidget>
#include <QApplication>

#include "recorder.h"
#include "analyzer.h"

//---------------------------------------------------------------------------

XGraphListConnector::XGraphListConnector(const shared_ptr<XGraphList> &node, QTableWidget *item,
										 QPushButton *btnnew, QPushButton *btndelete) :
    XListQConnector(node, item),
    m_graphlist(node),
    m_newGraph(XNode::createOrphan<XTouchableNode>("NewGraph", true)),
    m_deleteGraph(XNode::createOrphan<XTouchableNode>("DeleteGraph", true)),
    m_conNewGraph(xqcon_create<XQButtonConnector>(m_newGraph, btnnew)),
    m_conDeleteGraph(xqcon_create<XQButtonConnector>(m_deleteGraph, btndelete)) {

    btnnew->setIcon(
        QApplication::style()->standardIcon(QStyle::SP_FileDialogStart));
    btndelete->setIcon(
        QApplication::style()->standardIcon(QStyle::SP_DialogCloseButton));
               
    connect(item, SIGNAL( cellClicked( int, int)),
            this, SLOT(cellClicked( int, int)) );
	m_pItem->setColumnCount(4);
	const double def = 50;
	m_pItem->setColumnWidth(0, (int)(def * 2.0));
	m_pItem->setColumnWidth(1, (int)(def * 2.0));
	m_pItem->setColumnWidth(2, (int)(def * 2.0));
	m_pItem->setColumnWidth(3, (int)(def * 2.0));
	QStringList labels;
	labels += i18n("Name");
	labels += i18n("Axis X");
	labels += i18n("Axis Y");
	labels += i18n("Axis Z");
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
  
    m_newGraph->iterate_commit([=](Transaction &tr){
		m_lsnNewGraph = tr[ *m_newGraph].onTouch().connectWeakly(
			shared_from_this(), &XGraphListConnector::onNewGraph, Listener::FLAG_MAIN_THREAD_CALL);
    });
    m_deleteGraph->iterate_commit([=](Transaction &tr){
		m_lsnDeleteGraph = tr[ *m_deleteGraph].onTouch().connectWeakly(
			shared_from_this(), &XGraphListConnector::onDeleteGraph, Listener::FLAG_MAIN_THREAD_CALL);
    });
}

void
XGraphListConnector::onNewGraph (const Snapshot &shot, XTouchableNode *) {
	static int graphidx = 1;
    try {
        m_graphlist->createByTypename("", formatString("Graph-%d", graphidx++));
    }
#ifdef USE_PYBIND11
    catch (pybind11::error_already_set& e) {
        gErrPrint(std::string("Python error: ") + e.what());
    }
#endif
    catch (std::runtime_error &e) {
        gErrPrint(std::string("Python KAME binding error: ") + e.what());
    }
    catch (...) {
        gErrPrint(std::string("Unknown python error."));
    }
}
void
XGraphListConnector::onDeleteGraph (const Snapshot &shot, XTouchableNode *) {
	int n = m_pItem->currentRow();
	Snapshot shot_this( *m_graphlist);
	if(shot_this.size()) {
		if((n >= 0) && (n < (int)shot_this.list()->size())) {
			shared_ptr<XNode> node = shot_this.list()->at(n);
			m_graphlist->release(node);
		}
	}
}
void
XGraphListConnector::cellClicked ( int row, int col) {
	switch(col) {
	case 0: {
			Snapshot shot( *m_graphlist);
			if(shot.size()) {
				if((row >= 0) && (row < (int)shot.list()->size())) {
					dynamic_pointer_cast<XValGraph>(shot.list()->at(row))->showGraph();
				}
			}
		}
	break;
	default:
        break;
	}
}
void
XGraphListConnector::onRelease(const Snapshot &shot, const XListNodeBase::Payload::ReleaseEvent &e) {
	for(auto it = m_cons.begin(); it != m_cons.end();) {
		if(it->node == e.released) {
			for(int i = 0; i < m_pItem->rowCount(); i++) {
				if(m_pItem->cellWidget(i, 1) == it->widget) m_pItem->removeRow(i);
			}
			it = m_cons.erase(it);
		}
		else {
			it++;
		}
	}
}
void
XGraphListConnector::onCatch(const Snapshot &shot, const XListNodeBase::Payload::CatchEvent &e) {
	shared_ptr<XValGraph> graph = static_pointer_cast<XValGraph>(e.caught);
	int i = m_pItem->rowCount();
	m_pItem->insertRow(i);
    m_pItem->setItem(i, 0, new QTableWidgetItem(graph->getLabel().c_str()));

	Snapshot shot_entries( *m_graphlist->entries());
	struct tcons con;
	con.node = e.caught;
	QComboBox *cmbX = new QComboBox(m_pItem);
	con.conx = xqcon_create<XQComboBoxConnector>(graph->axisX(), cmbX, shot_entries);
	m_pItem->setCellWidget(i, 1, cmbX);
	QComboBox *cmbY1 = new QComboBox(m_pItem);
	con.cony1 = xqcon_create<XQComboBoxConnector>(graph->axisY1(), cmbY1, shot_entries);
	m_pItem->setCellWidget(i, 2, cmbY1);
	QComboBox *cmbZ = new QComboBox(m_pItem);
	con.conz = xqcon_create<XQComboBoxConnector>(graph->axisZ(), cmbZ, shot_entries);
	m_pItem->setCellWidget(i, 3, cmbZ);

	con.widget = m_pItem->cellWidget(i, 1);
	m_cons.push_back(con);
}
