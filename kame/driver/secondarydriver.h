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
#ifndef SECONDARYDRIVER_H_
#define SECONDARYDRIVER_H_

#include "driver.h"
#include "xitemnode.h"

//! Base class for all instrument drivers
class XSecondaryDriver : public XDriver
{
	XNODE_OBJECT
protected:
	XSecondaryDriver(const char *name, bool runtime,
		const shared_ptr<XScalarEntryList> &scalarentries,
		const shared_ptr<XInterfaceList> &interfaces,
		const shared_ptr<XThermometerList> &thermometers,
		const shared_ptr<XDriverList> &drivers);
public:
	virtual ~XSecondaryDriver() {}  

	//! show all forms belonging to driver
	virtual void showForms() = 0;

	virtual const shared_ptr<XRecordDependency> dependency() const {return m_dependency;}
protected:
	//! call this to receive signal/data
	void connect(const shared_ptr<XItemNodeBase> &item, bool check_deep_dep = true);
	//! check dependencies and lock all records and analyze
	//! null pointer will be passed to analyze()
	//! emitter is driver itself.
	//! \sa analyze(), checkDependency()
	void requestAnalysis();
	//! unlock one of connections in order to change the state of that instrument.
	void unlockConnection(const shared_ptr<XDriver> &connected);

	//! this is called when connected driver emit a signal
	//! unless dependency is broken
	//! all connected drivers are readLocked
	virtual void analyze(const shared_ptr<XDriver> &emitter) throw (XRecordError&) = 0;
	//! this is called after analyze() or analyzeRaw()
	//! record is readLocked
	//! this must be reentrant unlike analyze()
	virtual void visualize() = 0;
	//! check connected drivers have valid time
	//! \return true if dependency is resolved
	//! this must be reentrant unlike analyze()
	virtual bool checkDependency(const shared_ptr<XDriver> &emitter) const = 0;

	//! usually nothing to do
	virtual void start() {}
	//! usually nothing to do
	virtual void stop() {}
private:
	void readLockAllConnections();
	void readUnlockAllConnections();

	//! \ret true if dependency is resolved
	bool checkDeepDependency(shared_ptr<XRecordDependency> &) const;

	shared_ptr<XRecordDependency> m_dependency;

	//! holds connections
	std::vector<shared_ptr<const XDriver> > m_connections;
	std::vector<shared_ptr<const XDriver> > m_connections_check_deep_dep;
	std::vector<shared_ptr<const XDriver> > m_locked_connections;
	XRWLock m_connection_mutex;
	typedef std::vector<shared_ptr<const XDriver> >::const_iterator tConnection_it;
	shared_ptr<XListener> m_lsnOnRecord;
	shared_ptr<XListener> m_lsnBeforeItemChanged;
	shared_ptr<XListener> m_lsnOnItemChangedCheckDeepDep;
	shared_ptr<XListener> m_lsnOnItemChanged;
	//! called by connected drivers
	//! does dependency checks, readLock all connected drivers, write
	//! and finally call purely virtual function analyze();
	void onConnectedRecorded(const shared_ptr<XDriver> &);
	void beforeItemChanged(const shared_ptr<XValueNodeBase> &item);
	void onItemChangedCheckDeepDep(const shared_ptr<XValueNodeBase> &item);
	void onItemChanged(const shared_ptr<XValueNodeBase> &item);
};

#endif /*SECONDARYDRIVER_H_*/
