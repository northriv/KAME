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
#include "pulserdriverconnector.h"
#include "pulserdriver.h"
#include "analyzer.h"
#include <QTableWidget>
#include <QHeaderView>
#include "graph.h"
#include "graphwidget.h"

XQPulserDriverConnector::XQPulserDriverConnector(
    const shared_ptr<XPulser> &node, QTableWidget *item, XQGraph *qgraph)
	: XQConnector(node, item),
      m_pTable(item),
      m_pulser(node),
      m_graph(XNode::createOrphan<XGraph>(node->getName().c_str(), false)) {

	shared_ptr<XPulser> pulser(node);
    pulser->iterate_commit([=](Transaction &tr){
		m_lsnOnPulseChanged = tr[ *pulser].onRecord().connectWeakly(
			shared_from_this(), &XQPulserDriverConnector::onPulseChanged,
			Listener::FLAG_MAIN_THREAD_CALL | Listener::FLAG_AVOID_DUP | Listener::FLAG_DELAY_ADAPTIVE);
    });
	m_pTable->setColumnCount(3);
	double def = 50;
	m_pTable->setColumnWidth(0, (int)(def * 1.5));
	m_pTable->setColumnWidth(1, (int)(def * 1.5));
	m_pTable->setColumnWidth(2, (int)(def * 3.0));
	QStringList labels;
	labels += "Time [ms]";
	labels += "Diff [ms]";
	labels += "Pattern (Port 0, 1, ...)";
	m_pTable->setHorizontalHeaderLabels(labels);
//	m_pTable->setReadOnly(true);
//	m_pTable->setSelectionModel(QItemSelectionModel::Rows);

	QHeaderView *header = m_pTable->verticalHeader();
#if QT_VERSION >= 0x50000
    header->setSectionResizeMode(QHeaderView::Fixed);
#else
    header->setResizeMode(QHeaderView::Fixed);
#endif
    connect(m_pTable, SIGNAL( cellClicked( int, int)),
            this, SLOT(cellClicked( int, int)) );
    connect(m_pTable, SIGNAL( itemSelectionChanged()), this, SLOT(selectionChanged()) );

    qgraph->setGraph(m_graph);
    
    m_graph->iterate_commit([=](Transaction &tr){
		const XNode::NodeList &axes_list( *tr.list(m_graph->axes()));
		shared_ptr<XAxis> axisx = static_pointer_cast<XAxis>(axes_list.at(0));
		shared_ptr<XAxis> axisy = static_pointer_cast<XAxis>(axes_list.at(1));

		tr[ *axisy->ticLabelFormat()] = "%.0f";

		tr[ *m_graph->backGround()] = QColor(0x0A, 0x05, 0x45).rgb();
		tr[ *m_graph->titleColor()] = clWhite;
		tr[ *m_graph->drawLegends()] = false;
		tr[ *axisx->label()] = "Time [ms]";
		tr[ *axisx->ticColor()] = clWhite;
		tr[ *axisx->labelColor()] = clWhite;
		tr[ *axisx->ticLabelColor()] = clWhite;
		tr[ *axisy->label()] = "Port";
		tr[ *axisy->majorTicScale()] = 1;
		tr[ *axisy->autoFreq()] = false;
		tr[ *axisy->displayMinorTics()] = false;
		tr[ *axisy->ticColor()] = clWhite;
		tr[ *axisy->labelColor()] = clWhite;
		tr[ *axisy->ticLabelColor()] = clWhite;
		m_plots.clear();
		for(int i=0; i < XPulser::NUM_DO_PORTS; i++) {
			shared_ptr<XXYPlot> plot = m_graph->plots()->create<XXYPlot>(
				tr, formatString("Port%d", i).c_str(), true, tr, m_graph);
			tr[ *plot->label()] = i18n("Port%1").arg(i);
			tr[ *plot->axisX()] = axisx;
			tr[ *plot->axisY()] = axisy;
			m_plots.push_back(plot);
			tr[ *plot->drawPoints()] = false;
			tr[ *plot->displayMajorGrid()] = false;
//			tr[ *plot->lineColor()] = QColor(0x4e, 0xff, 0x10).rgb();
			tr[ *plot->clearPoints()].setUIEnabled(false);
			tr[ *plot->maxCount()].setUIEnabled(false);
		}
		m_barPlot = m_graph->plots()->create<XXYPlot>(tr, "Bars", true, tr, m_graph);
		tr[ *m_barPlot->label()] = i18n("Bars");
		tr[ *m_barPlot->axisX()] = axisx;
		tr[ *m_barPlot->axisY()] = axisy;
		tr[ *m_barPlot->drawBars()] = true;
		tr[ *m_barPlot->drawLines()] = false;
		tr[ *m_barPlot->drawPoints()] = false;
		tr[ *m_barPlot->barColor()] = QColor(0x4A, 0x3D, 0x87).rgb();
		tr[ *m_barPlot->displayMajorGrid()] = true;
		tr[ *m_barPlot->majorGridColor()] = QColor(0x4A, 0x4A, 0).rgb();
		tr[ *m_barPlot->drawLines()].setUIEnabled(false);
		tr[ *m_barPlot->drawPoints()].setUIEnabled(false);
		tr[ *m_barPlot->lineColor()].setUIEnabled(false);
		tr[ *m_barPlot->pointColor()].setUIEnabled(false);
		tr[ *m_barPlot->clearPoints()].setUIEnabled(false);
		tr[ *m_barPlot->maxCount()].setUIEnabled(false);

		tr[ *m_graph->label()] = i18n("Pulse Patterns");
    });
}

XQPulserDriverConnector::~XQPulserDriverConnector() {
}

void
XQPulserDriverConnector::cellClicked( int , int ) {
}

void
XQPulserDriverConnector::selectionChanged() {
    shared_ptr<XPulser> pulser(m_pulser);
    Snapshot shot( *pulser);
    updateGraph(shot, true);
}
void
XQPulserDriverConnector::updateGraph(const Snapshot &shot, bool checkselection) {
    shared_ptr<XPulser> pulser(m_pulser);
    const XPulser::Payload::RelPatList &relpatlist(shot[ *pulser].relPatList());
	m_graph->iterate_commit([=](Transaction &tr){
        auto &barplot_points(tr[ *m_barPlot].points());
		tr[ *m_barPlot->maxCount()] = relpatlist.size();
		barplot_points.clear();
        std::vector<decltype(&barplot_points)> plots_points;
        for(auto it = m_plots.begin();
			it != m_plots.end(); it++) {
			tr[ *(*it)->maxCount()] = relpatlist.size() * 2;
			tr[ **it].points().clear();
			plots_points.push_back(&tr[ **it].points());
		}
		uint32_t lastpat = relpatlist.empty() ? 0 :
			relpatlist[relpatlist.size() - 1].pattern;
		double firsttime = -0.001, lasttime = 100;

		int i = 0;
		for(XPulser::Payload::RelPatList::const_iterator it = relpatlist.begin();
			it != relpatlist.end(); it++) {
			double time = it->time * pulser->resolution();
            auto tableitem = m_pTable->item(i + 1, 0);
            if(tableitem && tableitem->isSelected()) {
				if(firsttime < 0) firsttime = time;
				lasttime = time;
			}
			barplot_points.push_back(XGraph::ValPoint(time, m_plots.size()));
			for(int j = 0; j < (int)plots_points.size(); j++) {
				plots_points[j]->push_back(XGraph::ValPoint(time, j + 0.7 * ((lastpat >> j) % 2)));
				plots_points[j]->push_back(XGraph::ValPoint(time, j + 0.7 * ((it->pattern >> j) % 2)));
			}
			lastpat = it->pattern;
			i++;
		}
		if(checkselection) {
			if(lasttime == firsttime) {
				firsttime -= 0.5;
				lasttime += 0.5;
			}
			double width = lasttime - firsttime;
			firsttime -= width / 10;
			lasttime += width / 10;
			shared_ptr<XAxis> axisx = tr[ *m_barPlot->axisX()];
			tr[ *axisx->autoScale()] = false;
			tr[ *axisx->minValue()] = firsttime;
			tr[ *axisx->maxValue()] = lasttime;
		}
		tr.mark(tr[ *m_graph].onUpdate(), m_graph.get());
    });
}

void
XQPulserDriverConnector::onPulseChanged(const Snapshot &shot, XDriver *) {
    shared_ptr<XPulser> pulser(m_pulser);
    if(shot[ *pulser].time()) {
        m_pTable->blockSignals(true);
        m_pTable->setRowCount(shot[ *pulser].relPatList().size());
        int i = 0;
        for(XPulser::Payload::RelPatList::const_iterator it = shot[ *pulser].relPatList().begin();
			it != shot[ *pulser].relPatList().end(); it++) {
			//        Form->tblPulse->insertRow(i);
			m_pTable->setItem(i, 0, new QTableWidgetItem(formatString("%.4f", it->time * pulser->resolution())));
			m_pTable->setItem(i, 1, new QTableWidgetItem(formatString("%.4f", it->toappear * pulser->resolution())));
			QString s;
			uint32_t pat = it->pattern;
			for(int j = 0; j < XPulser::NUM_DO_PORTS; j++) {
				//            if(j != 0) s+= ",";
				s += (pat % 2) ? "1" : "0";
				pat /= 2;
			}
			m_pTable->setItem(i, 2, new QTableWidgetItem(s));
			i++;
		}
        m_pTable->blockSignals(false);
        
        updateGraph(shot, false);
    }
    else {
        m_pTable->clear();
    	m_graph->iterate_commit([=](Transaction &tr){
            for(auto it = m_plots.begin();
				it != m_plots.end(); it++) {
				tr[ **it].points().clear();
			}
			tr[ *m_barPlot].points().clear();
			tr.mark(tr[ *m_graph].onUpdate(), m_graph.get());
        });
    }
}
