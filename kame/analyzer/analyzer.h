/***************************************************************************
		Copyright (C) 2002-2014 Kentaro Kitagawa
		                   kitag@kochi-u.ac.jp
		
		This program is free software; you can redistribute it and/or
		modify it under the terms of the GNU Library General Public
		License as published by the Free Software Foundation; either
		version 2 of the License, or (at your option) any later version.
		
		You should have received a copy of the GNU Library General 
		Public License and a list of authors along with this program; 
		see the files COPYING and AUTHORS.
***************************************************************************/
//---------------------------------------------------------------------------

#ifndef analyzerH
#define analyzerH

#include "xnode.h"
#include "xitemnode.h"
#include "xlistnode.h"
#include "support.h"
#include "xnodeconnector.h"

//Retain 'small' analyzed data capable to be handled by graphs, charts,...
class XJournal;
class XDriver;

class DECLSPEC_KAME XScalarEntry : public XNode {
public:
	XScalarEntry(const char *name, bool runtime, const shared_ptr<XDriver> &driver,
				 const char *format = 0L);
	virtual ~XScalarEntry() {}

	//A condition for determining a trigger of storing.
	//0: never
	//negative: allways
	//positive: when the difference from old value exceeds 'Delta'.
	const shared_ptr<XDoubleNode> &delta() const {return m_delta;}
	//if false, one line should not include this.
	const shared_ptr<XBoolNode> &store() const {return m_store;}

	const shared_ptr<XDoubleNode> &value() const {return m_value;}
	const shared_ptr<XDoubleNode> &storedValue() const {return m_storedValue;}
  
	void storeValue(Transaction &tr);

	shared_ptr<XDriver> driver() const {return m_driver.lock();}
  
	virtual XString getLabel() const;
  
	void value(Transaction &tr, double val);

	struct Payload : public XNode::Payload {
		Payload() : m_bTriggered(false) {}
		bool isTriggered() const {return m_bTriggered;}
	private:
		friend class XScalarEntry;
		bool m_bTriggered;
	};
protected:
private:
	const weak_ptr<XDriver> m_driver;
 
	const shared_ptr<XDoubleNode> m_delta;
	const shared_ptr<XBoolNode> m_store;

	const shared_ptr<XDoubleNode> m_value;
	const shared_ptr<XDoubleNode> m_storedValue;
};

class XDriverList;

class DECLSPEC_KAME XScalarEntryList : public XAliasListNode<XScalarEntry> {
public:
	XScalarEntryList(const char *name, bool runtime) : XAliasListNode<XScalarEntry>(name, runtime) {}
};

class Ui_FrmGraph;
class QMainWindow;
typedef QForm<QWidget, Ui_FrmGraph> FrmGraph;
class XGraph;
class XXYPlot;

class XValChart : public XNode {
public:
	XValChart(const char *name, bool runtime, const shared_ptr<XScalarEntry> &entry);
	virtual ~XValChart() {}
	void showChart();
	const shared_ptr<XScalarEntry> &entry() const {return m_entry;}
private:
	shared_ptr<XListener> m_lsnOnRecord;
	//callback from Driver
	void onRecord(const Snapshot &shot, XDriver *driver);

	const shared_ptr<XScalarEntry> m_entry;
	shared_ptr<XGraph> m_graph;
	qshared_ptr<FrmGraph> m_graphForm;
	shared_ptr<XXYPlot> m_chart;
};

class XChartList : public XAliasListNode<XValChart> {
public:
	XChartList(const char *name, bool runtime, const shared_ptr<XScalarEntryList> &entries);
	virtual ~XChartList() {}
private:
	shared_ptr<XListener> m_lsnOnCatchEntry;
	shared_ptr<XListener> m_lsnOnReleaseEntry;
	void onCatchEntry(const Snapshot &shot, const XListNodeBase::Payload::CatchEvent &e);
	void onReleaseEntry(const Snapshot &shot, const XListNodeBase::Payload::ReleaseEvent &e);

	const shared_ptr<XScalarEntryList> m_entries;
};

class XValGraph : public XNode {
public:
	XValGraph(const char *name, bool runtime,
		Transaction &tr_entries, const shared_ptr<XScalarEntryList> &entries);
	virtual ~XValGraph() {}

	void showGraph();
	void clearAllPoints();

	typedef XItemNode<XScalarEntryList, XScalarEntry> tAxis;
  
	const shared_ptr<tAxis> &axisX() const {return m_axisX;}
	const shared_ptr<tAxis> &axisY1() const {return m_axisY1;}
	const shared_ptr<tAxis> &axisZ() const {return m_axisZ;}

	struct Payload : public XNode::Payload {
	private:
		friend class XValGraph;
		shared_ptr<XGraph> m_graph;
		shared_ptr<XXYPlot> m_livePlot, m_storePlot;
	};
protected:
private:
	qshared_ptr<FrmGraph> m_graphForm;

	shared_ptr<tAxis> m_axisX, m_axisY1, m_axisZ;
	shared_ptr<XListener> m_lsnAxisChanged;
	void onAxisChanged(const Snapshot &shot, XValueNodeBase *node);

	shared_ptr<XListener> m_lsnLiveChanged;
	shared_ptr<XListener> m_lsnStoreChanged;
	void onLiveChanged(const Snapshot &shot, XValueNodeBase *node);
	void onStoreChanged(const Snapshot &shot, XValueNodeBase *node);

	weak_ptr<XScalarEntryList> m_entries;
};

class XGraphList : public XCustomTypeListNode<XValGraph> {
public:
	XGraphList(const char *name, bool runtime, const shared_ptr<XScalarEntryList> &entries);
	virtual ~XGraphList() {}

	virtual shared_ptr<XNode> createByTypename(const XString &, const XString& name);
	const shared_ptr<XScalarEntryList> &entries() const {return m_entries;}
private:
	const shared_ptr<XScalarEntryList> m_entries;
};
//---------------------------------------------------------------------------
#endif

