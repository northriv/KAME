/***************************************************************************
		Copyright (C) 2002-2010 Kentaro Kitagawa
		                   kitag@issp.u-tokyo.ac.jp

		This program is free software; you can redistribute it and/or
		modify it under the terms of the GNU Library General Public
		License as published by the Free Software Foundation; either
		version 2 of the License, or (at your option) any later version.

		You should have received a copy of the GNU Library General 
		Public License and a list of authors along with this program; 
		see the files COPYING and AUTHORS.
 ***************************************************************************/
#ifndef SECONDARYDRIVER_H_
#define SECONDARYDRIVER_H_

#include "driver.h"
#include "xitemnode.h"

//! Base class for all instrument drivers
class XSecondaryDriver : public XDriver {
public:
	XSecondaryDriver(const char *name, bool runtime, Transaction &tr_meas, const shared_ptr<XMeasure> &meas);
	virtual ~XSecondaryDriver();

	//! Shows all forms belonging to driver
	virtual void showForms() = 0;

	struct Payload : public XDriver::Payload {
	private:
		friend class XSecondaryDriver;
		struct Connection {
			shared_ptr<XListener> m_lsnOnRecord;
			shared_ptr<XPointerItemNode<XDriverList> > m_selecter;
			bool operator==(const XItemNodeBase *p) const {return p == m_selecter.get();}
		};
		typedef std::vector<Connection> ConnectionList;
		ConnectionList m_connections;
	};
protected:
	//! Call this to receive signal/data.
	void connect(const shared_ptr<XPointerItemNode<XDriverList> > &selecter);
	//! check dependencies and lock all records and analyze
	//! null pointer will be passed to analyze()
	//! emitter is driver itself.
	//! \sa analyze(), checkDependency()
	void requestAnalysis();

	//! This function is called when a connected driver emit a signal
	virtual void analyze(Transaction &tr, const Snapshot &shot_emitter, const Snapshot &shot_others,
		XDriver *emitter) throw (XRecordError&) = 0;
	//! This function is called inside analyze() or analyzeRaw()
	//! this must be reentrant unlike analyze()
	virtual void visualize(const Snapshot &shot) = 0;
	//! Checks if the connected drivers have valid time stamps.
	//! \return true if dependency is resolved.
	//! This function must be reentrant unlike analyze().
	virtual bool checkDependency(const Snapshot &shot_this,
		const Snapshot &shot_emitter, const Snapshot &shot_others,
		XDriver *emitter) const = 0;

	//! usually nothing to do
	virtual void start() {}
	//! usually nothing to do
	virtual void stop() {}
private:
	shared_ptr<XListener> m_lsnOnItemChanged;
	//! called by connected drivers,
	//! checks dependency, takes snapshot for drivers,
	//! and finally calls purely virtual function analyze();
	void onConnectedRecorded(const Snapshot &shot, XDriver *driver);

	void onItemChanged(const Snapshot &shot, XValueNodeBase *item);

	weak_ptr<XDriverList> m_drivers;
};

#endif /*SECONDARYDRIVER_H_*/
