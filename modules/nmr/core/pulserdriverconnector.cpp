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
#include "pulserdriverconnector.h"
#include "pulserdriver.h"
#include "analyzer.h"
#include <qtable.h>
#include "graph.h"
#include "graphwidget.h"

XQPulserDriverConnector::XQPulserDriverConnector(
    const shared_ptr<XPulser> &node, QTable *item, XQGraph *qgraph)
	: XQConnector(node, item),
      m_pTable(item),
      m_pulser(node),
      m_graph(createOrphan<XGraph>(node->getName().c_str(), false))
{
	shared_ptr<XPulser> pulser(node);    
	m_lsnOnPulseChanged = pulser->onRecord().connectWeak(
		shared_from_this(), &XQPulserDriverConnector::onPulseChanged,
		XListener::FLAG_MAIN_THREAD_CALL | XListener::FLAG_AVOID_DUP | XListener::FLAG_DELAY_ADAPTIVE);
  
	m_pTable->setNumCols(3);
	double def = 50;
	m_pTable->setColumnWidth(0, (int)(def * 1.5));
	m_pTable->setColumnWidth(1, (int)(def * 1.5));
	m_pTable->setColumnWidth(2, (int)(def * 3.0));
	QStringList labels;
	labels += "Time [ms]";
	labels += "Diff [ms]";
	labels += "Pattern (Port 0, 1, ...)";
	m_pTable->setColumnLabels(labels);
	m_pTable->setReadOnly(true);
	m_pTable->setSelectionMode(QTable::MultiRow);

	QHeader *header = m_pTable->verticalHeader();
	header->setResizeEnabled(false);
      
	connect(m_pTable, SIGNAL( selectionChanged()), this, SLOT(selectionChanged()) );
	connect(m_pTable, SIGNAL( clicked( int, int, int, const QPoint& )), this,
			SLOT( clicked( int, int, int, const QPoint& )));

    qgraph->setGraph(m_graph);
    
    atomic_shared_ptr<const XNode::NodeList> axes_list(m_graph->axes()->children());
    shared_ptr<XAxis> axisx = dynamic_pointer_cast<XAxis>(axes_list->at(0));
    shared_ptr<XAxis> axisy = dynamic_pointer_cast<XAxis>(axes_list->at(1));

    axisy->ticLabelFormat()->value("%.0f");
    
    m_graph->backGround()->value(QColor(0x0A, 0x05, 0x45).rgb());
    m_graph->titleColor()->value(clWhite);
    m_graph->drawLegends()->value(false);
    axisx->label()->value("Time [ms]");
    axisx->ticColor()->value(clWhite);
    axisx->labelColor()->value(clWhite);
    axisx->ticLabelColor()->value(clWhite);  
    axisy->label()->value("Port");
    axisy->majorTicScale()->value(1);
    axisy->autoFreq()->value(false);  
    axisy->displayMinorTics()->value(false);  
    axisy->ticColor()->value(clWhite);
    axisy->labelColor()->value(clWhite);
    axisy->ticLabelColor()->value(clWhite);
    for(int i=0; i < XPulser::NUM_DO_PORTS; i++)
	{
		shared_ptr<XXYPlot> plot = m_graph->plots()->create<XXYPlot>(
			QString().sprintf("Port%d", i), true, m_graph);
		plot->label()->value(QString().sprintf(KAME::i18n("Port%d"), i));
		plot->axisX()->value(axisx);
		plot->axisY()->value(axisy);
		m_plots.push_back(plot);
		plot->drawPoints()->value(false);
		plot->displayMajorGrid()->value(false);
		plot->lineColor()->value(QColor(0x4e, 0xff, 0x10).rgb());
		plot->clearPoints()->setUIEnabled(false);
		plot->maxCount()->setUIEnabled(false);
	}
    m_barPlot = m_graph->plots()->create<XXYPlot>("Bars", true, m_graph);
    m_barPlot->label()->value(KAME::i18n("Bars"));
    m_barPlot->axisX()->value(axisx);
    m_barPlot->axisY()->value(axisy);
    m_barPlot->drawBars()->value(true);
    m_barPlot->drawLines()->value(false);
    m_barPlot->drawPoints()->value(false);
    m_barPlot->barColor()->value(QColor(0x4A, 0x3D, 0x87).rgb());
    m_barPlot->displayMajorGrid()->value(true);
    m_barPlot->majorGridColor()->value(QColor(0x4A, 0x4A, 0).rgb());
    m_barPlot->drawLines()->setUIEnabled(false);
    m_barPlot->drawPoints()->setUIEnabled(false);
    m_barPlot->lineColor()->setUIEnabled(false);
    m_barPlot->pointColor()->setUIEnabled(false);
    m_barPlot->clearPoints()->setUIEnabled(false);
    m_barPlot->maxCount()->setUIEnabled(false);

    m_graph->label()->value(KAME::i18n("Pulse Patterns"));
}

XQPulserDriverConnector::~XQPulserDriverConnector()
{
}

void
XQPulserDriverConnector::clicked( int , int , int, const QPoint & )
{
}

void
XQPulserDriverConnector::selectionChanged()
{
    shared_ptr<XPulser> pulser(m_pulser);
    pulser->readLockRecord();    
    updateGraph(true);
    pulser->readUnlockRecord();
}
void
XQPulserDriverConnector::updateGraph(bool checkselection)
{
    shared_ptr<XPulser> pulser(m_pulser);
    XScopedLock<XGraph> lock(*m_graph);
    
    std::deque<XGraph::ValPoint> & barplot_points(m_barPlot->points());
    m_barPlot->maxCount()->value(pulser->m_relPatList.size());
    barplot_points.clear();
    std::deque<std::deque<XGraph::ValPoint> *> plots_points;
    for(std::deque<shared_ptr<XXYPlot> >::iterator it = m_plots.begin();
		it != m_plots.end(); it++)
	{
		(*it)->maxCount()->value(pulser->m_relPatList.size() * 2);
		(*it)->points().clear();
		plots_points.push_back(&(*it)->points());
	}
    uint32_t lastpat = pulser->m_relPatList.empty() ? 0 :
        pulser->m_relPatList[pulser->m_relPatList.size() - 1].pattern;
    double firsttime = -0.001, lasttime = 100;
    
    int i = 0;
    for(XPulser::RelPatListIterator it = pulser->m_relPatList.begin(); 
		it != pulser->m_relPatList.end(); it++)
	{
		double time = it->time * pulser->resolution();
		if(m_pTable->isRowSelected(i))
		{
			if(firsttime < 0) firsttime = time;
			lasttime = time;
		}
		barplot_points.push_back(XGraph::ValPoint(time, m_plots.size()));
		for(int j = 0; j < (int)plots_points.size(); j++)
		{
			plots_points[j]->push_back(XGraph::ValPoint(time, j + 0.7 * ((lastpat >> j) % 2)));
			plots_points[j]->push_back(XGraph::ValPoint(time, j + 0.7 * ((it->pattern >> j) % 2)));
		}
		lastpat = it->pattern;
		i++;
	}
    if(checkselection)
	{
		if(lasttime == firsttime) {
			firsttime -= 0.5;
			lasttime += 0.5;
		}
		double width = lasttime - firsttime;
		firsttime -= width / 10;
		lasttime += width / 10;
		shared_ptr<XAxis> axisx = *m_barPlot->axisX();
		axisx->autoScale()->value(false);
		axisx->minValue()->value(firsttime);
		axisx->maxValue()->value(lasttime);
	}        
    m_graph->requestUpdate();
}

void
XQPulserDriverConnector::onPulseChanged(const shared_ptr<XDriver> &)
{
    shared_ptr<XPulser> pulser(m_pulser);

    pulser->readLockRecord();
    
    if(pulser->time()) {
        
        m_pTable->blockSignals(true);
        m_pTable->setNumRows(pulser->m_relPatList.size());
        int i = 0;
        for(XPulser::RelPatListIterator it = pulser->m_relPatList.begin();
			it != pulser->m_relPatList.end(); it++)
		{
			//        Form->tblPulse->insertRows(i);
			m_pTable->setText(i, 0, QString().sprintf("%.4f", it->time * pulser->resolution()));
			m_pTable->setText(i, 1, QString().sprintf("%.4f", it->toappear * pulser->resolution()));
			QString s;
			uint32_t pat = it->pattern;
			for(int j = 0; j < XPulser::NUM_DO_PORTS; j++) {
				//            if(j != 0) s+= ",";
				s += (pat % 2) ? "1" : "0";
				pat /= 2;
			}
			m_pTable->setText(i, 2, s);
			i++;
		}
        m_pTable->blockSignals(false);
        
        updateGraph(false);
    }
    else {
        m_pTable->setNumRows(0);
        XScopedLock<XGraph> lock(*m_graph);
        for(std::deque<shared_ptr<XXYPlot> >::iterator it = m_plots.begin();
			it != m_plots.end(); it++) {
            (*it)->clearAllPoints();
        }
        m_barPlot->clearAllPoints();
        m_graph->requestUpdate();
    }
    
    pulser->readUnlockRecord();    
}
