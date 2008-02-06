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
//---------------------------------------------------------------------------

#include "graphlistconnector.h"

#include <qtable.h>
#include <qcombobox.h>
#include <qpushbutton.h>
#include <kiconloader.h>
#include <kapplication.h>

#include "recorder.h"
#include "analyzer.h"

//---------------------------------------------------------------------------

XGraphListConnector::XGraphListConnector(const shared_ptr<XGraphList> &node, QTable *item,
										 QPushButton *btnnew, QPushButton *btndelete) :
    XListQConnector(node, item),
    m_graphlist(node),
    m_newGraph(createOrphan<XNode>("NewGraph", true)),
    m_deleteGraph(createOrphan<XNode>("DeleteGraph", true)),
    m_conNewGraph(xqcon_create<XQButtonConnector>(m_newGraph, btnnew)),
    m_conDeleteGraph(xqcon_create<XQButtonConnector>(m_deleteGraph, btndelete))
{
	btnnew->setIconSet( KApplication::kApplication()->iconLoader()->loadIconSet("filenew", 
																				KIcon::Toolbar, KIcon::SizeSmall, true ) );  
	btndelete->setIconSet( KApplication::kApplication()->iconLoader()->loadIconSet("fileclose", 
																				   KIcon::Toolbar, KIcon::SizeSmall, true ) ); 
               
	connect(item, SIGNAL( clicked( int, int, int, const QPoint& )),
			this, SLOT(clicked( int, int, int, const QPoint& )) );
	m_pItem->setNumCols(4);
	const double def = 50;
	m_pItem->setColumnWidth(0, (int)(def * 2.0));
	m_pItem->setColumnWidth(1, (int)(def * 2.0));
	m_pItem->setColumnWidth(2, (int)(def * 2.0));
	m_pItem->setColumnWidth(3, (int)(def * 2.0));
	QStringList labels;
	labels += KAME::i18n("Name");
	labels += KAME::i18n("Axis X");
	labels += KAME::i18n("Axis Y");
	labels += KAME::i18n("Axis Z");
	m_pItem->setColumnLabels(labels);

	atomic_shared_ptr<const XNode::NodeList> list(node->children());
	if(list) {  
		for(XNode::NodeList::const_iterator it = list->begin(); it != list->end(); it++)
			onCatch(*it);
	}

  
	m_lsnNewGraph = m_newGraph->onTouch().connectWeak(
        shared_from_this(), &XGraphListConnector::onNewGraph, XListener::FLAG_MAIN_THREAD_CALL);
	m_lsnDeleteGraph = m_deleteGraph->onTouch().connectWeak(
        shared_from_this(), &XGraphListConnector::onDeleteGraph, XListener::FLAG_MAIN_THREAD_CALL);
}

void
XGraphListConnector::onNewGraph (const shared_ptr<XNode> &) {
	static int graphidx = 1;
    m_graphlist->createByTypename("", std::string(QString().sprintf("Graph-%d", graphidx++).utf8()));
}
void
XGraphListConnector::onDeleteGraph (const shared_ptr<XNode> &) {
	int n = m_pItem->currentRow();
	atomic_shared_ptr<const XNode::NodeList> list(m_graphlist->children());
	if(list) {    
		if((n >= 0) && (n < (int)list->size())) {
			shared_ptr<XNode> node = list->at(n);
			m_graphlist->releaseChild(node);
		}
	}
}
void
XGraphListConnector::clicked ( int row, int col, int, const QPoint& ) {
	switch(col) {
	case 0:
	{
		atomic_shared_ptr<const XNode::NodeList> list(m_graphlist->children());
		if(list) { 
			if((row >= 0) && (row < (int)list->size())) {
				dynamic_pointer_cast<XValGraph>(list->at(row))->showGraph();
			}
		}
	}
	break;
	default:
        break;
	}
}
void
XGraphListConnector::onRelease(const shared_ptr<XNode> &node)
{
	for(tconslist::iterator it = m_cons.begin(); it != m_cons.end();)
	{
		if(it->node == node)
		{
			for(int i = 0; i < m_pItem->numRows(); i++)
			{
				if(m_pItem->cellWidget(i, 1) == it->widget) m_pItem->removeRow(i);
			}
			it = m_cons.erase(it);
		}
		else
		{
			it++;
		}
	}
}
void
XGraphListConnector::onCatch(const shared_ptr<XNode> &node)
{
	shared_ptr<XValGraph> graph = dynamic_pointer_cast<XValGraph>(node);
	int i = m_pItem->numRows();
	m_pItem->insertRows(i);
	m_pItem->setText(i, 0, graph->getLabel());

	struct tcons con;
	con.node = node;
	QComboBox *cmbX = new QComboBox(m_pItem);
	con.conx = xqcon_create<XQComboBoxConnector>(graph->axisX(), cmbX);
	m_pItem->setCellWidget(i, 1, cmbX);
	QComboBox *cmbY1 = new QComboBox(m_pItem);
	con.cony1 = xqcon_create<XQComboBoxConnector>(graph->axisY1(), cmbY1);
	m_pItem->setCellWidget(i, 2, cmbY1);
	QComboBox *cmbZ = new QComboBox(m_pItem);
	con.conz = xqcon_create<XQComboBoxConnector>(graph->axisZ(), cmbZ);
	m_pItem->setCellWidget(i, 3, cmbZ);

	con.widget = m_pItem->cellWidget(i, 1);
	m_cons.push_back(con);
}
