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
#include "ui_graphform.h"
#include "graphwidget.h"
#include "analyzer.h"
#include "graph.h"
#include "driver.h"
#include "measure.h"

#include <QStatusBar>

//---------------------------------------------------------------------------
XScalarEntry::XScalarEntry(const char *name, bool runtime, const shared_ptr<XDriver> &driver,
						   const char *format)
	: XNode(name, runtime),
	  m_driver(driver),
	  m_delta(create<XDoubleNode>("Delta", false)),
	  m_store(create<XBoolNode>("Store", false)),
	  m_value(create<XDoubleNode>("Value", true)),
	  m_storedValue(create<XDoubleNode>("StoredValue", true)) {

	m_delta->setFormat(format);
	m_storedValue->setFormat(format);
	m_value->setFormat(format);
	trans( *store()) = false;
}
void
XScalarEntry::storeValue(Transaction &tr) {
    tr[ *storedValue()] = (double)tr[ *value()];
    tr[ *this].m_bTriggered = false;
}

XString
XScalarEntry::getLabel() const {
	return driver()->getLabel() + "-" + XNode::getLabel();
}

void
XScalarEntry::value(Transaction &tr, double val) {
	Snapshot &shot(tr);
	if((shot[ *delta()] != 0) && (fabs(val - shot[ *storedValue()]) > shot[ *delta()])) {
		tr[ *this].m_bTriggered = true;
	}
	tr[ *value()] = val;
}

XValChart::XValChart(const char *name, bool runtime,
	const shared_ptr<XScalarEntry> &entry)
	: XNode(name, runtime),
	  m_entry(entry),
	  m_graph(create<XGraph>(name, false)),
      m_graphForm(new FrmGraph(g_pFrmMain, Qt::Window)) {

    m_graphForm->m_graphwidget->setGraph(m_graph);
    
    m_graph->iterate_commit([=](Transaction &tr){
		m_chart= m_graph->plots()->create<XXYPlot>(tr, entry->getLabel().c_str(), true, tr, m_graph);
		tr[ *m_chart->label()] = entry->getLabel();
		const XNode::NodeList &axes_list( *tr.list(m_graph->axes()));
		shared_ptr<XAxis> axisx = static_pointer_cast<XAxis>(axes_list.at(0));
		shared_ptr<XAxis> axisy = static_pointer_cast<XAxis>(axes_list.at(1));

		tr[ *m_chart->axisX()] = axisx;
		tr[ *m_chart->axisY()] = axisy;
		tr[ *m_chart->maxCount()] = 600;
		tr[ *axisx->length()] = 0.95 - tr[ *axisx->x()];
		tr[ *axisy->length()] = 0.90 - tr[ *axisy->y()];
		tr[ *axisx->label()] = "Time";
        tr[ *axisx->ticLabelFormat()] = "TIME:%H:%M:%S";
		tr[ *axisy->label()] = entry->getLabel();
		tr[ *axisy->ticLabelFormat()] = entry->value()->format();
		tr[ *axisx->minValue()].setUIEnabled(false);
		tr[ *axisx->maxValue()].setUIEnabled(false);
		tr[ *axisx->autoScale()].setUIEnabled(false);
		tr[ *axisx->logScale()].setUIEnabled(false);
		tr[ *m_graph->label()] = entry->getLabel();

        m_graph->applyTheme(tr, true);
    });

    entry->driver()->iterate_commit([=](Transaction &tr){
		m_lsnOnRecord = tr[ *entry->driver()].onRecord().connectWeakly(
			shared_from_this(), &XValChart::onRecord);
    });
}
void
XValChart::onRecord(const Snapshot &shot, XDriver *driver) {
    XTime time = shot[ *driver].time();
    if(time.isSet()) {
        try {
            double val = shot.at( *m_entry->value());
            iterate_commit([=](Transaction &tr){
                m_chart->addPoint(tr, time.sec() + time.usec() * 1e-6, val);
    //            tr[ *m_graph->osdStrings()] = time.getTimeStr();
            });
        }
        catch (XNode::NodeNotFoundError &e) {
        }
    }
}
void
XValChart::showChart(void) {
	m_graphForm->setWindowTitle(i18n("Chart - ") + getLabel() );
    m_graphForm->showNormal();
    m_graphForm->raise();
}

XChartList::XChartList(const char *name, bool runtime, const shared_ptr<XScalarEntryList> &entries)
	: XAliasListNode<XValChart>(name, runtime),
	  m_entries(entries) {
    entries->iterate_commit([=](Transaction &tr){
	    m_lsnOnCatchEntry = tr[ *entries].onCatch().connectWeakly(shared_from_this(), &XChartList::onCatchEntry);
	    m_lsnOnReleaseEntry = tr[ *entries].onRelease().connectWeakly(shared_from_this(), &XChartList::onReleaseEntry);
    });
}

void
XChartList::onCatchEntry(const Snapshot &shot, const XListNodeBase::Payload::CatchEvent &e) {
    shared_ptr<XScalarEntry> entry = static_pointer_cast<XScalarEntry>(e.caught);
    create<XValChart>(entry->getLabel().c_str(), true, entry);
}
void
XChartList::onReleaseEntry(const Snapshot &shot, const XListNodeBase::Payload::ReleaseEvent &e) {
	shared_ptr<XScalarEntry> entry = dynamic_pointer_cast<XScalarEntry>(e.released);
    iterate_commit_while([=](Transaction &tr)->bool{
		shared_ptr<XValChart> valchart;
		if(tr.size()) {
			const XNode::NodeList &list( *tr.list());
			for(auto it = list.begin(); it != list.end(); it++) {
				auto chart = dynamic_pointer_cast<XValChart>( *it);
				if(chart->entry() == entry) valchart = chart;
			}
		}
		if( !valchart)
            return false;
		if( !release(tr, valchart))
            return true;//will fail.
        return true;
    });
}

XValGraph::XValGraph(const char *name, bool runtime,
					 Transaction &tr_entries, const shared_ptr<XScalarEntryList> &entries)
	: XNode(name, runtime),
	  m_graphForm(),
	  m_axisX(create<tAxis>("AxisX", false, ref(tr_entries), entries)),
	  m_axisY1(create<tAxis>("AxisY1", false, ref(tr_entries), entries)),
	  m_axisZ(create<tAxis>("AxisZ", false, ref(tr_entries), entries)),
	  m_entries(entries) {
	iterate_commit([=](Transaction &tr){
	    m_lsnAxisChanged = tr[ *axisX()].onValueChanged().connectWeakly(
	        shared_from_this(), &XValGraph::onAxisChanged,
            Listener::FLAG_MAIN_THREAD_CALL | Listener::FLAG_AVOID_DUP);
	    tr[ *axisY1()].onValueChanged().connect(m_lsnAxisChanged);
	    tr[ *axisZ()].onValueChanged().connect(m_lsnAxisChanged);
    });
}
void
XValGraph::onAxisChanged(const Snapshot &shot, XValueNodeBase *) {
    shared_ptr<XScalarEntry> entryx;
    shared_ptr<XScalarEntry> entryy1;
    shared_ptr<XScalarEntry> entryz;
    shared_ptr<XGraph> graph;
    iterate_commit([=, &entryx, &entryy1, &entryz, &graph](Transaction &tr){
		const Snapshot &shot_this(tr);
	    entryx = shot_this[ *axisX()];
	    entryy1 = shot_this[ *axisY1()];
	    entryz = shot_this[ *axisZ()];

		if(tr[ *this].m_graph) release(tr, tr[ *this].m_graph);
		tr[ *this].m_graph = create<XGraph>(tr, getName().c_str(), false);
        graph = tr[ *this].m_graph;

		if( !entryx || !entryy1) return;

		tr[ *this].m_livePlot =
			graph->plots()->create<XXYPlot>(tr,
				(graph->getName() + "-Live").c_str(), false, tr, graph);
		tr[ *shot_this[ *this].m_livePlot->label()] = graph->getLabel() + " Live";
		tr[ *this].m_storePlot =
			graph->plots()->create<XXYPlot>(tr,
				(graph->getName() + "-Stored").c_str(), false, tr, graph);
		tr[ *shot_this[ *this].m_storePlot->label()] = graph->getLabel() + " Stored";

		const XNode::NodeList &axes_list( *tr.list(graph->axes()));
		auto axisx = static_pointer_cast<XAxis>(axes_list.at(0));
		auto axisy = static_pointer_cast<XAxis>(axes_list.at(1));

		tr[ *axisx->ticLabelFormat()] = entryx->value()->format();
		tr[ *axisy->ticLabelFormat()] = entryy1->value()->format();
		tr[ *shot_this[ *this].m_livePlot->axisX()] = axisx;
		tr[ *shot_this[ *this].m_livePlot->axisY()] = axisy;
		tr[ *shot_this[ *this].m_storePlot->axisX()] = axisx;
		tr[ *shot_this[ *this].m_storePlot->axisY()] = axisy;

		tr[ *axisx->length()] = 0.95 - shot_this[ *axisx->x()];
		tr[ *axisy->length()] = 0.90 - shot_this[ *axisy->y()];
		if(entryz) {
			shared_ptr<XAxis> axisz = graph->axes()->create<XAxis>(
                tr, "Z Axis", false, XAxis::AxisDirection::Z, false, tr, graph);
			tr[ *axisz->ticLabelFormat()] = entryz->value()->format();
			tr[ *shot_this[ *this].m_livePlot->axisZ()] = axisz;
			tr[ *shot_this[ *this].m_storePlot->axisZ()] = axisz;
	//	axisz->label()] = "Z Axis";
			tr[ *axisz->label()] = entryz->getLabel();
		}

		tr[ *shot_this[ *this].m_storePlot->displayMajorGrid()] = false;
        tr[ *shot_this[ *this].m_livePlot->maxCount()] = 10000;
        tr[ *shot_this[ *this].m_storePlot->maxCount()] = 10000;
		tr[ *axisx->label()] = entryx->getLabel();
		tr[ *axisy->label()] = entryy1->getLabel();
		tr[ *graph->label()] = getLabel();

        graph->applyTheme(tr, true);
    });
    m_graphForm.reset(new FrmGraph(g_pFrmMain, Qt::Window));
    m_graphForm->m_graphwidget->setGraph(graph);
    m_entries.lock()->iterate_commit([=](Transaction &tr){
		if( !tr.isUpperOf( *entryx)) return;
		if( !tr.isUpperOf( *entryy1)) return;
		if(entryz && !tr.isUpperOf( *entryz)) return;
		m_lsnLiveChanged = tr[ *entryx->value()].onValueChanged().connectWeakly(
			shared_from_this(), &XValGraph::onLiveChanged);
		tr[ *entryy1->value()].onValueChanged().connect(m_lsnLiveChanged);
		if(entryz) tr[ *entryz->value()].onValueChanged().connect(m_lsnLiveChanged);

		m_lsnStoreChanged = tr[ *entryx->storedValue()].onValueChanged().connectWeakly(
			shared_from_this(), &XValGraph::onStoreChanged);
		tr[ *entryy1->storedValue()].onValueChanged().connect(m_lsnStoreChanged);
		if(entryz) tr[ *entryz->storedValue()].onValueChanged().connect(m_lsnStoreChanged);
    });

	showGraph();
}

void
XValGraph::clearAllPoints() {
	iterate_commit([=](Transaction &tr){
		if( !tr[ *this].m_graph) return;
		tr[ *this].m_storePlot->clearAllPoints(tr);
		tr[ *this].m_livePlot->clearAllPoints(tr);
    });
}
void
XValGraph::onLiveChanged(const Snapshot &shot, XValueNodeBase *) {
	Snapshot shot_this( *this);
	shared_ptr<XScalarEntry> entryx = shot_this[ *axisX()];
	shared_ptr<XScalarEntry> entryy1 = shot_this[ *axisY1()];
	shared_ptr<XScalarEntry> entryz = shot_this[ *axisZ()];
	if( !entryx || !entryy1) return;
	Snapshot shot_entries( *m_entries.lock());
	if( !shot_entries.isUpperOf( *entryx)) return;
	if( !shot_entries.isUpperOf( *entryy1)) return;
	if(entryz && !shot_entries.isUpperOf( *entryz)) return;

	double x, y, z = 0.0;
	x = shot_entries[ *entryx->value()];
	y = shot_entries[ *entryy1->value()];
	if(entryz) z = shot_entries[ *entryz->value()];

	iterate_commit([=](Transaction &tr){
		tr[ *this].m_livePlot->addPoint(tr, x, y, z);
    });
}
void
XValGraph::onStoreChanged(const Snapshot &shot, XValueNodeBase *) {
	Snapshot shot_this( *this);
	shared_ptr<XScalarEntry> entryx = shot_this[ *axisX()];
	shared_ptr<XScalarEntry> entryy1 = shot_this[ *axisY1()];
	shared_ptr<XScalarEntry> entryz = shot_this[ *axisZ()];
	if( !entryx || !entryy1) return;
	Snapshot shot_entries( *m_entries.lock());
	if( !shot_entries.isUpperOf( *entryx)) return;
	if( !shot_entries.isUpperOf( *entryy1)) return;
	if(entryz && !shot_entries.isUpperOf( *entryz)) return;

	double x, y, z = 0.0;
	x = shot_entries[ *entryx->storedValue()];
	y = shot_entries[ *entryy1->storedValue()];
	if(entryz) z = shot_entries[ *entryz->storedValue()];

	iterate_commit([=](Transaction &tr){
		tr[ *this].m_storePlot->addPoint(tr, x, y, z);
    });
}
void
XValGraph::showGraph() {
	if(m_graphForm) {
		m_graphForm->setWindowTitle(i18n("Graph - ") + getLabel() );
        m_graphForm->showNormal();
        m_graphForm->raise();
    }
}

XGraphList::XGraphList(const char *name, bool runtime, const shared_ptr<XScalarEntryList> &entries)
	: XCustomTypeListNode<XValGraph>(name, runtime),
	  m_entries(entries) {
}

shared_ptr<XNode>
XGraphList::createByTypename(const XString &, const XString& name)  {
    shared_ptr<XValGraph> x;
    m_entries->iterate_commit([=, &x](Transaction &tr){
        if(x) release(x);
        x = create<XValGraph>(name.c_str(), false, tr, m_entries);
    });
    return x;
}
