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
#include "secondarydriver.h"
#include <klocale.h>

static XThreadLocal<std::vector<std::pair<shared_ptr<const XDriver>, XSecondaryDriver* > > > stl_locked_connections;

XSecondaryDriver::XSecondaryDriver(const char *name, bool runtime, 
								   const shared_ptr<XScalarEntryList> &scalarentries,
								   const shared_ptr<XInterfaceList> &interfaces,
								   const shared_ptr<XThermometerList> &thermometers,
								   const shared_ptr<XDriverList> &drivers) :
    XDriver(name, runtime, scalarentries, interfaces, thermometers, drivers),
    m_dependency(new XRecordDependency())
{
}

bool
XSecondaryDriver::checkDeepDependency(shared_ptr<XRecordDependency> &dep) const
{
    bool sane = true;
    dep->clear();
    for(tConnection_it it = m_connections_check_deep_dep.begin(); it != m_connections_check_deep_dep.end(); it++) {
        bool is_conflict = dep->merge(*it);
        if(is_conflict) {
            sane = false;
            break;
        }
    }
    return sane;
}


void
XSecondaryDriver::readLockAllConnections() {
    m_connection_mutex.readLock();
    for(tConnection_it it = m_connections.begin(); it != m_connections.end(); it++) {
    	stl_locked_connections->push_back(std::pair<shared_ptr<const XDriver>, XSecondaryDriver* >(*it, this));
        (*it)->readLockRecord();
    }
}
void
XSecondaryDriver::readUnlockAllConnections() {
    for(std::vector<std::pair<shared_ptr<const XDriver>, XSecondaryDriver* > >::iterator
    	it = stl_locked_connections->begin(); it != stl_locked_connections->end();) {
    	if(it->second == this) {
    		it->first->readUnlockRecord();
    		it = stl_locked_connections->erase(it);
    	}
    	else {
    		it++;
    	}
    }
    m_connection_mutex.readUnlock();
}
void
XSecondaryDriver::unlockConnection(const shared_ptr<XDriver> &connected) {
	stl_locked_connections->erase(std::find(
		stl_locked_connections->begin(), stl_locked_connections->end(),
		std::pair<shared_ptr<const XDriver>, XSecondaryDriver* >(connected, this)));
	connected->readUnlockRecord();
}

void
XSecondaryDriver::requestAnalysis()
{
    onConnectedRecorded(dynamic_pointer_cast<XDriver>(shared_from_this()));
}
void
XSecondaryDriver::onConnectedRecorded(const shared_ptr<XDriver> &driver)
{
	for(unsigned int i = 0;; i++) {
		readLockAllConnections();
	    //! check if emitter has already connected or if self-emission
	    if((std::find(m_connections.begin(), m_connections.end(), driver)
			!= m_connections.end()) 
	       || (driver == shared_from_this())) {
	        //! driver-side dependency check
	        if(checkDependency(driver)) {
	            shared_ptr<XRecordDependency> dep(new XRecordDependency);
	            //! check if recorded times don't contradict
	            if(checkDeepDependency(dep)) {
	            	bool skipped = false;
	                if(!tryStartRecording()) {
	                	readUnlockAllConnections();
	                	usleep(5000);
                    	if(i > 100) {
                    		gErrPrint(formatString(KAME::i18n(
                    				"Dead lock deteceted on %s. Operation canceled.\nReport this bug to author(s)."),
                    				getName().c_str()));
                    		return;
                    	}
	                	continue;
	                }
	                XTime time_recorded = driver->time();
	                try {
	                    analyze(driver);
	                }
	                catch (XSkippedRecordError&) {
	                	skipped = true;
	                }
	                catch (XRecordError& e) {
						time_recorded = XTime(); //record is invalid
						e.print(getLabel() + ": " + KAME::i18n("Record Error, because "));
	                }
	                readUnlockAllConnections();
	                if(skipped)
	                	abortRecordingNReadLock();
	            	else {
		                m_dependency = dep;
		                finishRecordingNReadLock(driver->timeAwared(), time_recorded);
	            	}
	                visualize();
	                readUnlockRecord();
	            	return;
	            }
	        }
	    }
	    readUnlockAllConnections();
	    return;
	}
}
void
XSecondaryDriver::connect(const shared_ptr<XItemNodeBase> &item, bool check_deep_dep)
{
    if(m_lsnBeforeItemChanged)
        item->beforeValueChanged().connect(m_lsnBeforeItemChanged);
    else
        m_lsnBeforeItemChanged = item->beforeValueChanged().connectWeak(
			shared_from_this(), &XSecondaryDriver::beforeItemChanged);
	if(check_deep_dep) {
	    if(m_lsnOnItemChangedCheckDeepDep)
	        item->onValueChanged().connect(m_lsnOnItemChangedCheckDeepDep);
	    else
	        m_lsnOnItemChangedCheckDeepDep = item->onValueChanged().connectWeak(
				shared_from_this(), &XSecondaryDriver::onItemChangedCheckDeepDep);
	}
	else {
	    if(m_lsnOnItemChanged)
	        item->onValueChanged().connect(m_lsnOnItemChanged);
	    else
	        m_lsnOnItemChanged = item->onValueChanged().connectWeak(
				shared_from_this(), &XSecondaryDriver::onItemChanged);
	}
}
void
XSecondaryDriver::beforeItemChanged(const shared_ptr<XValueNodeBase> &node) {
    //! changes in items are not allowed while onRecord() is emitting
    m_connection_mutex.writeLock();

    shared_ptr<XPointerItemNode<XDriverList> > item =
		dynamic_pointer_cast<XPointerItemNode<XDriverList> >(node);
    shared_ptr<XNode> nd = *item;
    shared_ptr<XDriver> driver = dynamic_pointer_cast<XDriver>(nd);

    if(driver) {
        driver->onRecord().disconnect(m_lsnOnRecord);
        ASSERT(std::find(m_connections.begin(), m_connections.end(), driver) != m_connections.end());
        m_connections.erase(std::find(m_connections.begin(), m_connections.end(), driver));
        std::vector<shared_ptr<const XDriver> >::iterator it = 
        	std::find(m_connections_check_deep_dep.begin(), m_connections_check_deep_dep.end(), driver);
        if(it != m_connections_check_deep_dep.end())
        	m_connections_check_deep_dep.erase(it);
    }
}
void
XSecondaryDriver::onItemChangedCheckDeepDep(const shared_ptr<XValueNodeBase> &node) {
    shared_ptr<XPointerItemNode<XDriverList> > item = 
        dynamic_pointer_cast<XPointerItemNode<XDriverList> >(node);
    shared_ptr<XNode> nd = *item;
    shared_ptr<XDriver> driver = dynamic_pointer_cast<XDriver>(nd);

    if(driver) {
        m_connections_check_deep_dep.push_back(driver);
    }

    onItemChanged(node);
}
void
XSecondaryDriver::onItemChanged(const shared_ptr<XValueNodeBase> &node) {
    shared_ptr<XPointerItemNode<XDriverList> > item = 
        dynamic_pointer_cast<XPointerItemNode<XDriverList> >(node);
    shared_ptr<XNode> nd = *item;
    shared_ptr<XDriver> driver = dynamic_pointer_cast<XDriver>(nd);

    if(driver) {
        m_connections.push_back(driver);
        if(m_lsnOnRecord)
            driver->onRecord().connect(m_lsnOnRecord);
        else
            m_lsnOnRecord = driver->onRecord().connectWeak(
                shared_from_this(), &XSecondaryDriver::onConnectedRecorded);
    }

    m_connection_mutex.writeUnlock();
}
