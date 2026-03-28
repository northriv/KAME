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
//---------------------------------------------------------------------------
#include "ui_graphform.h"
#include "graphwidget.h"
#include "analyzer.h"
#include "graph.h"
#include "driver.h"
#include "measure.h"
#include "thermometer.h"

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
    auto drv = driver();
    if( !drv) return XNode::getLabel();
	return drv->getLabel() + "-" + XNode::getLabel();
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
    m_graphForm->m_axisSelector->hide();

    m_graph->iterate_commit([=](Transaction &tr){
        m_chart = m_graph->plots()->create<XXYPlot>(tr, entry->getLabel().c_str(), true, tr, m_graph);
        if( !m_chart) return; //transaction has failed.
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

    if(auto drv = entry->driver()) {
        drv->iterate_commit([=](Transaction &tr){
            m_lsnOnRecord = tr[ *drv].onVisualization().connectWeakly(
                shared_from_this(), &XValChart::onVisualization);
        });
    }
}
void
XValChart::onVisualization(const Snapshot &shot, bool afterRecorded, XDriver *driver) {
    XTime time = shot[ *driver].time();
    if(afterRecorded && time.isSet()) {
        try {
            double val;
            try {
                val = shot.at( *m_entry->value());
            }
            catch (XNode::NodeNotFoundError &) {
                // Entry (e.g. calibrated proxy) not in driver's snapshot;
                // use a fresh snapshot of the entry's own subtree instead.
                Snapshot shot_entry( *m_entry);
                val = shot_entry[ *m_entry->value()];
            }
            iterate_commit([=](Transaction &tr){
                m_chart->addPoint(tr, time.sec() + time.usec() * 1e-6, val);
    //            tr[ *m_graph->onScreenStrings()] = time.getTimeStr();
            });
        }
        catch (XNode::NodeNotFoundError &e) {
            //Entry has been already released from the list.
            static XMutex s_mutex;
            XScopedLock<XMutex> lock(s_mutex);
            m_lsnOnRecord.reset();
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
        m_lsnOnCatchEntry = tr[ *entries].onCatch().connectWeakly(shared_from_this(), &XChartList::onCatchEntry, Listener::FLAG_MAIN_THREAD_CALL);
        m_lsnOnReleaseEntry = tr[ *entries].onRelease().connectWeakly(shared_from_this(), &XChartList::onReleaseEntry);
    });
}

void
XChartList::onCatchEntry(const Snapshot &shot, const XListNodeBase::Payload::CatchEvent &e) {
    shared_ptr<XScalarEntry> entry = static_pointer_cast<XScalarEntry>(e.caught);
    if( !entry->driver()) return; // skip driver-less entries (e.g. XCalibratedEntry)
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

FrmGraph *
XValGraph::graphForm() {
    if( !m_graphForm)
        m_graphForm.reset(new FrmGraph(g_pFrmMain, Qt::Window));
    return m_graphForm.get();
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
        if( !tr[ *this].m_graph) return; //transaction has failed.
        graph = tr[ *this].m_graph;

		if( !entryx || !entryy1) {
			tr[ *this].m_livePlot.reset();
			tr[ *this].m_storePlot.reset();
			return;
		}

		tr[ *this].m_livePlot =
			graph->plots()->create<XXYPlot>(tr,
				(graph->getName() + "-Live").c_str(), false, tr, graph);
        if( !tr[ *this].m_livePlot) return;
		tr[ *shot_this[ *this].m_livePlot->label()] = graph->getLabel() + " Live";
		tr[ *this].m_storePlot =
			graph->plots()->create<XXYPlot>(tr,
				(graph->getName() + "-Stored").c_str(), false, tr, graph);
        if( !tr[ *this].m_storePlot) return;
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
    graphForm()->m_graphwidget->setGraph(graph);
    m_lsnOnVisualization.reset();
    if(auto drv = entryx ? entryx->driver() : shared_ptr<XDriver>()) {
        drv->iterate_commit([=](Transaction &tr){
            m_lsnOnVisualization = tr[ *drv].onVisualization().connectWeakly(
                shared_from_this(), &XValGraph::onVisualization);
        });
    }
    m_entries.lock()->iterate_commit([=](Transaction &tr){
		if( !tr.isUpperOf( *entryx)) return;
		if( !tr.isUpperOf( *entryy1)) return;
		if(entryz && !tr.isUpperOf( *entryz)) return;
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
XValGraph::onVisualization(const Snapshot &shot, bool afterRecorded, XDriver *driver) {
    if( !afterRecorded || !shot[ *driver].time().isSet()) return;
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
		if( !tr[ *this].m_livePlot) return;
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
		if( !tr[ *this].m_storePlot) return;
		tr[ *this].m_storePlot->addPoint(tr, x, y, z);
    });
}
void
XValGraph::showGraph() {
    if(m_graphForm && Snapshot( *this)[ *this].m_graph) {
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

XCalibratedEntry::XCalibratedEntry(const char *name, bool runtime,
    const shared_ptr<XScalarEntryList> &entries,
    const shared_ptr<XCalibrationCurveList> &curves)
    : XNode(name, runtime),
      m_proxy(XNode::createOrphan<XScalarEntry>(name, false, shared_ptr<XDriver>())),
      m_entries(entries) {
    entries->iterate_commit([=](Transaction &tr){
        m_source = create<tSource>("Source", false, tr, entries);
    });
    curves->iterate_commit([=](Transaction &tr){
        m_curve = create<tCurve>("Curve", false, tr, curves);
    });
    iterate_commit([=](Transaction &tr){
        m_lsnSelectionChanged = tr[ *m_source].onValueChanged().connectWeakly(
            shared_from_this(), &XCalibratedEntry::onSelectionChanged,
            Listener::FLAG_MAIN_THREAD_CALL);
        tr[ *m_curve].onValueChanged().connect(m_lsnSelectionChanged);
    });
}
void
XCalibratedEntry::onSelectionChanged(const Snapshot &, XValueNodeBase *) {
    Snapshot shot( *this);
    shared_ptr<XScalarEntry> src = shot[ *m_source];
    shared_ptr<XCalibrationCurve> curve = shot[ *m_curve];
    auto curSrc = m_currentSource.lock();
    if(src != curSrc) {
        m_lsnSourceValueChanged.reset();
        m_currentSource = src;
        // Release old proxy and recreate with source's driver so the proxy
        // participates in driver-based recording (textwriter, entrylistconnector).
        auto entries = m_entries.lock();
        if(m_proxyInserted && entries) {
            Snapshot shot(*entries);
            if(shot.isUpperOf(*m_proxy))
                entries->release(m_proxy);
            m_proxyInserted = false;
        }
        m_proxy = XNode::createOrphan<XScalarEntry>(
            getName().c_str(), false, src ? src->driver() : shared_ptr<XDriver>());
        if(src) {
            src->value()->iterate_commit([=](Transaction &tr){
                m_lsnSourceValueChanged = tr[ *src->value()].onValueChanged().connectWeakly(
                    shared_from_this(), &XCalibratedEntry::onSourceValueChanged);
            });
        }
    }
    auto entries = m_entries.lock();
    if(entries) {
        if(src && curve && src->driver()) {
            if( !m_proxyInserted) {
                entries->iterate_commit([=](Transaction &tr){
                    entries->insert(tr, m_proxy);
                });
                m_proxyInserted = true;
            }
            double raw = ***src->value();
            try {
                double out = curve->getOutput(raw);
                m_proxy->iterate_commit([=](Transaction &tr){ m_proxy->value(tr, out); });
            } catch(...) {}
        } else {
            if(m_proxyInserted) {
                Snapshot shot(*entries);
                if(shot.isUpperOf(*m_proxy))
                    entries->release(m_proxy);
                m_proxyInserted = false;
            }
        }
    }
}
void
XCalibratedEntry::onSourceValueChanged(const Snapshot &shot, XValueNodeBase *node) {
    auto src = m_currentSource.lock();
    auto proxy = m_proxy; // capture in case proxy is recreated concurrently
    if( !src || !proxy) return;
    Snapshot shot_this( *this);
    shared_ptr<XCalibrationCurve> curve = shot_this[ *m_curve];
    if( !curve) return;
    double raw = shot[ *static_cast<XDoubleNode*>(node)];
    try {
        double out = curve->getOutput(raw);
        proxy->iterate_commit([=](Transaction &tr){ proxy->value(tr, out); });
    } catch(...) {}
}

XCalibratedEntryList::XCalibratedEntryList(const char *name, bool runtime,
    const shared_ptr<XScalarEntryList> &entries,
    const shared_ptr<XCalibrationCurveList> &curves)
    : XCustomTypeListNode<XCalibratedEntry>(name, runtime),
      m_entries(entries), m_curves(curves) {
    iterate_commit([=](Transaction &tr){
        m_lsnOnCatch = tr[ *this].onCatch().connectWeakly(
            shared_from_this(), &XCalibratedEntryList::onCatch);
        m_lsnOnRelease = tr[ *this].onRelease().connectWeakly(
            shared_from_this(), &XCalibratedEntryList::onRelease);
    });
}
shared_ptr<XNode>
XCalibratedEntryList::createByTypename(const XString &, const XString &name) {
    auto entry = XNode::createOrphan<XCalibratedEntry>(
        name.c_str(), false, m_entries, m_curves);
    insert(entry);
    return entry;
}
void
XCalibratedEntryList::onCatch(const Snapshot &, const XListNodeBase::Payload::CatchEvent &) {
    // proxy insertion deferred to XCalibratedEntry::onSelectionChanged when both source and curve are set
}
void
XCalibratedEntryList::onRelease(const Snapshot &, const XListNodeBase::Payload::ReleaseEvent &e) {
    auto entry = static_pointer_cast<XCalibratedEntry>(e.released);
    if(!entry->proxyInserted()) return;
    auto proxy = entry->proxy();
    Snapshot shot(*m_entries);
    if(shot.isUpperOf(*proxy))
        m_entries->release(proxy);
}
