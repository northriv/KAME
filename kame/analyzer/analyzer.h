/***************************************************************************
		Copyright (C) 2002-2008 Kentaro Kitagawa
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

class XScalarEntry : public XNode
{
	XNODE_OBJECT
protected:
	XScalarEntry(const char *name, bool runtime, const shared_ptr<XDriver> &driver,
				 const char *format = 0L);
public:
	virtual ~XScalarEntry() {}

	//a criterion that determine a trigger of storing
	//0: never
	//negative: allways
	//positive: when the difference from old value exceeds 'Delta'
	const shared_ptr<XDoubleNode> &delta() const {return m_delta;}
	//if false, one line does't contain this
	const shared_ptr<XBoolNode> &store() const {return m_store;}

	const shared_ptr<XDoubleNode> &value() const {return m_value;}
	const shared_ptr<XDoubleNode> &storedValue() const {return m_storedValue;}
  
	bool isTriggered() const;
	void storeValue();

	shared_ptr<XDriver> driver() const {return m_driver.lock();}
  
	virtual std::string getLabel() const;
  
	virtual void value(double val);
protected:
private:
	const weak_ptr<XDriver> m_driver;
 
	const shared_ptr<XDoubleNode> m_delta;
	const shared_ptr<XBoolNode> m_store;

	const shared_ptr<XDoubleNode> m_value;
	const shared_ptr<XDoubleNode> m_storedValue;
  
	bool m_bTriggered;
};

class XDriverList;

class XScalarEntryList : public XAliasListNode<XScalarEntry> {
	XNODE_OBJECT
protected:
    XScalarEntryList(const char *name, bool runtime) : XAliasListNode<XScalarEntry>(name, runtime) {}
};

class FrmGraph;
class XGraph;
class XXYPlot;

class XValChart : public XNode
{
	XNODE_OBJECT
protected:
	XValChart(const char *name, bool runtime, const shared_ptr<XScalarEntry> &entry);
public:
	virtual ~XValChart() {}
	void showChart();
	const shared_ptr<XScalarEntry> &entry() const {return m_entry;}
private:
	shared_ptr<XListener> m_lsnOnRecord;
	//callback from Driver
	void onRecord(const shared_ptr<XDriver> &);

	const shared_ptr<XScalarEntry> m_entry;
	shared_ptr<XGraph> m_graph;
	qshared_ptr<FrmGraph> m_graphForm;
	shared_ptr<XXYPlot> m_chart;
};

class XChartList : public XAliasListNode<XValChart>
{
	XNODE_OBJECT
protected:
	XChartList(const char *name, bool runtime, const shared_ptr<XScalarEntryList> &entries);
public:
	virtual ~XChartList() {}
private:
	shared_ptr<XListener> m_lsnOnCatchEntry;
	shared_ptr<XListener> m_lsnOnReleaseEntry;
	void onCatchEntry(const shared_ptr<XNode> &node);
	void onReleaseEntry(const shared_ptr<XNode> &node);

	const shared_ptr<XScalarEntryList> m_entries;
};

class XValGraph : public XNode
{
	XNODE_OBJECT
protected:
	XValGraph(const char *name, bool runtime,
			  const shared_ptr<XScalarEntryList> &entries);
public:
	virtual ~XValGraph() {}

	void showGraph();
	void clearAllPoints();

	typedef XItemNode<XScalarEntryList, XScalarEntry> tAxis;
  
	const shared_ptr<tAxis> &axisX() const {return m_axisX;}
	const shared_ptr<tAxis> &axisY1() const {return m_axisY1;}
	const shared_ptr<tAxis> &axisZ() const {return m_axisZ;}
protected:
private:
	shared_ptr<XGraph> m_graph;
	qshared_ptr<FrmGraph> m_graphForm;

	shared_ptr<tAxis> m_axisX, m_axisY1, m_axisZ;
	shared_ptr<XXYPlot> m_livePlot, m_storePlot;
	shared_ptr<XListener> m_lsnAxisChanged;
	void onAxisChanged(const shared_ptr<XValueNodeBase> &node);

	shared_ptr<XListener> m_lsnLiveChanged;
	shared_ptr<XListener> m_lsnStoreChanged;
	void onLiveChanged(const shared_ptr<XValueNodeBase> &node);
	void onStoreChanged(const shared_ptr<XValueNodeBase> &node);
};

class XGraphList : public XCustomTypeListNode<XValGraph>
{
	XNODE_OBJECT
protected:
	XGraphList(const char *name, bool runtime, const shared_ptr<XScalarEntryList> &entries);
public:
	virtual ~XGraphList() {}

	virtual shared_ptr<XNode> createByTypename(const std::string &, const std::string& name)  {
		return XNode::create<XValGraph>(name.c_str(), false, m_entries);
	}
private:
	const shared_ptr<XScalarEntryList> m_entries;
};
//---------------------------------------------------------------------------
#endif

