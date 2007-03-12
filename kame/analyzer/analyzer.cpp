/***************************************************************************
		Copyright (C) 2002-2007 Kentaro Kitagawa
		                   kitagawa@scphys.kyoto-u.ac.jp
		
		This program is free software; you can redistribute it and/or
		modify it under the terms of the GNU Library General Public
		License as published by the Free Software Foundation; either
		version 2 of the License, or (at your option) any later version.
		
		You should have received a copy of the GNU Library General 
		Public License and a list of authors along with this program; 
		see the files COPYING and AUTHORS.
***************************************************************************/
//---------------------------------------------------------------------------
#include "graphform.h"
#include "graphwidget.h"
#include "analyzer.h"
#include "graph.h"
#include "driver.h"
#include "measure.h"

#include <klocale.h>
#include <qstatusbar.h>

//---------------------------------------------------------------------------
XScalarEntry::XScalarEntry(const char *name, bool runtime, const shared_ptr<XDriver> &driver,
						   const char *format)
	: XNode(name, runtime),
	  m_driver(driver),
	  m_delta(create<XDoubleNode>("Delta", false)),
	  m_store(create<XBoolNode>("Store", false)),
	  m_value(create<XDoubleNode>("Value", true)),
	  m_storedValue(create<XDoubleNode>("StoredValue", true)),
	  m_bTriggered(false)
{
	m_delta->setFormat(format);
	m_storedValue->setFormat(format);
	m_value->setFormat(format);
	store()->value(false);
}
bool
XScalarEntry::isTriggered() const
{
    return m_bTriggered;
}
void
XScalarEntry::storeValue()
{
    storedValue()->value(*value());
    m_bTriggered = false;
}

std::string
XScalarEntry::getLabel() const
{
	return driver()->getLabel() + "-" + XNode::getLabel();
}

void
XScalarEntry::value(double val)
{
	if((*delta() != 0) && (fabs(val - *storedValue()) > *delta()))
    {
        m_bTriggered = true;
    }
	value()->value(val);
}

XValChart::XValChart(const char *name, bool runtime, const shared_ptr<XScalarEntry> &entry)
	: XNode(name, runtime),
	  m_entry(entry),
	  m_graph(create<XGraph>(name, false)),
	  m_graphForm(new FrmGraph(g_pFrmMain))
{
    m_graphForm->statusBar()->hide();
    m_graphForm->m_graphwidget->setGraph(m_graph);
    
    m_chart= m_graph->plots()->create<XXYPlot>(entry->getName().c_str(), true, m_graph);
	m_graph->persistence()->value(0.0);
    m_chart->label()->value(entry->getLabel());
    atomic_shared_ptr<const XNode::NodeList> axes_list(m_graph->axes()->children());
    shared_ptr<XAxis> axisx = dynamic_pointer_cast<XAxis>(axes_list->at(0));
    shared_ptr<XAxis> axisy = dynamic_pointer_cast<XAxis>(axes_list->at(1));
      
    m_chart->axisX()->value(axisx);
    m_chart->axisY()->value(axisy);
    m_chart->maxCount()->value(300);
    axisx->length()->value(0.95 - *axisx->x());
    axisy->length()->value(0.90 - *axisy->y());
    axisx->label()->value("Time");
    axisx->ticLabelFormat()->value("TIME:%T");
    axisy->label()->value(entry->getLabel());
    axisy->ticLabelFormat()->value(m_entry->value()->format());
    axisx->minValue()->setUIEnabled(false);
    axisx->maxValue()->setUIEnabled(false);
    axisx->autoScale()->setUIEnabled(false);
    axisx->logScale()->setUIEnabled(false);
    m_graph->label()->value(entry->getLabel());

    m_lsnOnRecord = entry->driver()->onRecord().connectWeak(
        shared_from_this(), &XValChart::onRecord);
}
void
XValChart::onRecord(const shared_ptr<XDriver> &driver)
{
	double val;
    val = *m_entry->value();
    XTime time = driver->time();
    if(time)
        m_chart->addPoint(time.sec() + time.usec() * 1e-6, val);
}
void
XValChart::showChart(void)
{
	m_graphForm->setCaption(KAME::i18n("Chart - ") + getLabel() );
	m_graphForm->show();
}

XChartList::XChartList(const char *name, bool runtime, const shared_ptr<XScalarEntryList> &entries)
	: XAliasListNode<XValChart>(name, runtime),
	  m_entries(entries)
{
    m_lsnOnCatchEntry = entries->onCatch().connectWeak(
        shared_from_this(), &XChartList::onCatchEntry);
    m_lsnOnReleaseEntry = entries->onRelease().connectWeak(
        shared_from_this(), &XChartList::onReleaseEntry);
}

void
XChartList::onCatchEntry(const shared_ptr<XNode> &node)
{
    shared_ptr<XScalarEntry> entry = dynamic_pointer_cast<XScalarEntry>(node);
    create<XValChart>(entry->getName().c_str(), true, entry);
}
void
XChartList::onReleaseEntry(const shared_ptr<XNode> &node)
{
	shared_ptr<XScalarEntry> entry = dynamic_pointer_cast<XScalarEntry>(node);

	shared_ptr<XValChart> valchart;
	atomic_shared_ptr<const XNode::NodeList> list(children());
	if(list) {
		for(XNode::NodeList::const_iterator it = list->begin(); it != list->end(); it++) {
			shared_ptr<XValChart> chart = dynamic_pointer_cast<XValChart>(*it);
			if(chart->entry() == entry) valchart = chart;
		}
	}
	if(valchart) releaseChild(valchart);
}

XValGraph::XValGraph(const char *name, bool runtime,
					 const shared_ptr<XScalarEntryList> &entries)
	: XNode(name, runtime),
	  m_graph(),
	  m_graphForm(),
	  m_axisX(create<tAxis>("AxisX", false, entries)),
	  m_axisY1(create<tAxis>("AxisY1", false, entries)),
	  m_axisZ(create<tAxis>("AxisZ", false, entries))
{
    m_lsnAxisChanged = axisX()->onValueChanged().connectWeak(
        shared_from_this(), &XValGraph::onAxisChanged,
		XListener::FLAG_MAIN_THREAD_CALL | XListener::FLAG_AVOID_DUP);
    axisY1()->onValueChanged().connect(m_lsnAxisChanged);
    axisZ()->onValueChanged().connect(m_lsnAxisChanged);
}
void
XValGraph::onAxisChanged(const shared_ptr<XValueNodeBase> &)
{
    shared_ptr<XScalarEntry> entryx = *axisX();
    shared_ptr<XScalarEntry> entryy1 = *axisY1();
    shared_ptr<XScalarEntry> entryz = *axisZ();
    
	if(m_graph) releaseChild(m_graph);
	m_graph = create<XGraph>(getName().c_str(), false);
	m_graphForm.reset(new FrmGraph(g_pFrmMain));
	m_graphForm->statusBar()->hide();
	m_graphForm->m_graphwidget->setGraph(m_graph);

	if(!entryx || !entryy1) return;
  
	m_livePlot = 
		m_graph->plots()->create<XXYPlot>((m_graph->getName() + "-Live").c_str(), false, m_graph);
	m_livePlot->label()->value(m_graph->getLabel() + " Live");
	m_storePlot = 
		m_graph->plots()->create<XXYPlot>((m_graph->getName() + "-Stored").c_str(), false, m_graph);
	m_storePlot->label()->value(m_graph->getLabel() + " Stored");

	atomic_shared_ptr<const XNode::NodeList> axes_list(m_graph->axes()->children());
	shared_ptr<XAxis> axisx = dynamic_pointer_cast<XAxis>(axes_list->at(0));
	shared_ptr<XAxis> axisy = dynamic_pointer_cast<XAxis>(axes_list->at(1));
  
	axisx->ticLabelFormat()->value(entryx->value()->format());
	axisy->ticLabelFormat()->value(entryy1->value()->format());
	m_livePlot->axisX()->value(axisx);
	m_livePlot->axisY()->value(axisy);
	m_storePlot->axisX()->value(axisx);
	m_storePlot->axisY()->value(axisy);
  
	axisx->length()->value(0.95 - *axisx->x());
	axisy->length()->value(0.90 - *axisy->y());
	if(entryz) {
		shared_ptr<XAxis> axisz = m_graph->axes()->create<XAxis>(
            "Z Axis", false, XAxis::DirAxisZ, false, m_graph);
		axisz->ticLabelFormat()->value(entryz->value()->format());
		m_livePlot->axisZ()->value(axisz);
		m_storePlot->axisZ()->value(axisz);
//	axisz->label()->value("Z Axis");
		axisz->label()->value(entryz->getLabel());
	}
  
	m_storePlot->pointColor()->value(clGreen);
	m_storePlot->lineColor()->value(clGreen);
	m_storePlot->barColor()->value(clGreen);
	m_storePlot->displayMajorGrid()->value(false);
	m_livePlot->maxCount()->value(4000);
	m_storePlot->maxCount()->value(4000);
	axisx->label()->value(entryx->getLabel());
	axisy->label()->value(entryy1->getLabel());
	m_graph->label()->value(getLabel());

	m_lsnLiveChanged = entryx->value()->onValueChanged().connectWeak(
		shared_from_this(), &XValGraph::onLiveChanged);
	entryy1->value()->onValueChanged().connect(m_lsnLiveChanged);
	if(entryz) entryz->value()->onValueChanged().connect(m_lsnLiveChanged);
  
	m_lsnStoreChanged = entryx->storedValue()->onValueChanged().connectWeak(
		shared_from_this(), &XValGraph::onStoreChanged);
	entryy1->storedValue()->onValueChanged().connect(m_lsnStoreChanged);
	if(entryz) entryz->storedValue()->onValueChanged().connect(m_lsnStoreChanged);
  
	showGraph();
}

void
XValGraph::clearAllPoints()
{
	if(!m_graph) return;
	m_storePlot->clearAllPoints();
	m_livePlot->clearAllPoints();
}
void
XValGraph::onLiveChanged(const shared_ptr<XValueNodeBase> &)
{
	double x, y, z = 0.0;
    shared_ptr<XScalarEntry> entryx = *axisX();
    shared_ptr<XScalarEntry> entryy1 = *axisY1();
    shared_ptr<XScalarEntry> entryz = *axisZ();
    
    if(!entryx || !entryy1) return;
    x = *entryx->value();
    y = *entryy1->value();
    if(entryz) z = *entryz->value();
  
    m_livePlot->addPoint(x, y, z);
}
void
XValGraph::onStoreChanged(const shared_ptr<XValueNodeBase> &)
{
	double x, y, z = 0.0;
    shared_ptr<XScalarEntry> entryx = *axisX();
    shared_ptr<XScalarEntry> entryy1 = *axisY1();
    shared_ptr<XScalarEntry> entryz = *axisZ();
    
    if(!entryx || !entryy1) return;
    x = *entryx->storedValue();
    y = *entryy1->storedValue();
    if(entryz) z = *entryz->storedValue();
  
    m_storePlot->addPoint(x, y, z);
}
void
XValGraph::showGraph()
{
	if(m_graphForm) {
		m_graphForm->setCaption(KAME::i18n("Graph - ") + getLabel() );
		m_graphForm->show();
	}
}

XGraphList::XGraphList(const char *name, bool runtime, const shared_ptr<XScalarEntryList> &entries)
	: XCustomTypeListNode<XValGraph>(name, runtime),
	  m_entries(entries)
{
}

