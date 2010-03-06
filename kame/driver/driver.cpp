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
#include "driver.h"
#include "interface.h"

DECLARE_TYPE_HOLDER(XDriverList)

XDriverList::XDriverList(const char *name, bool runtime,
						 const shared_ptr<XMeasure> &measure) :
	XCustomTypeListNode<XDriver>(name, runtime),
	m_measure(measure) {
}

shared_ptr<XNode>
XDriverList::createByTypename(const XString &type, const XString& name) {
	shared_ptr<XMeasure> measure(m_measure.lock());
	shared_ptr<XNode> ptr;
	for(Transaction tr( *measure);; ++tr) {
		ptr = creator(type)
			(name.c_str(), false, ref(tr), measure);
		if(ptr)
			if( !insert(tr, ptr))
				continue;
		if(tr.commit())
			break;
	}
    return ptr;
}

XDriver::XBufferUnderflowRecordError::XBufferUnderflowRecordError(const char *file, int line) : 
    XRecordError(i18n("Buffer Underflow."), file, line) {}

XDriver::XDriver(const char *name, bool runtime, Transaction &tr_meas, const shared_ptr<XMeasure> &meas) :
    XNode(name, runtime) {
}

void
XDriver::readUnlockRecord() const {
    m_recordLock.readUnlock();
}
void
XDriver::readLockRecord() const {
    m_recordLock.readLock();
}

bool
XDriver::tryStartRecording() {
    return m_recordLock.tryWriteLock();
}
void
XDriver::finishRecordingNReadLock(const XTime &time_awared, const XTime &time_recorded) {
    m_awaredTime = time_awared;
    m_recordTime = time_recorded;
    m_recordLock.writeUnlockNReadLock();
    onRecord().talk(dynamic_pointer_cast<XDriver>(shared_from_this()));
}
void
XDriver::abortRecording() {
    m_recordLock.writeUnlock();
}

void
XDriver::abortRecordingNReadLock() {
    m_recordLock.writeUnlockNReadLock();
}

XRecordDependency::tDependency::tDependency(const shared_ptr<const XDriver> &d, const XTime &time) :
    driver(d), time(time)
{}

XRecordDependency::XRecordDependency() :
    m_bConflict(false)
{}
XRecordDependency::XRecordDependency(const shared_ptr<XRecordDependency> &dep) :
    m_bConflict(dep->m_bConflict), m_dependency(dep->m_dependency)
{}
bool
XRecordDependency::tDependency::operator<(const tDependency &d) const
{
    return driver.get() < d.driver.get();
}

bool
XRecordDependency::merge(const shared_ptr<const XDriver> &driver)
{
    if(m_bConflict) return true;
    bool conflicted = false;
    if(!driver->time()) {
        conflicted = true;
    }
    else {
        m_dependency.insert(tDependency(driver, driver->time()));
        shared_ptr<XRecordDependency> dep = driver->dependency();
        if(dep) {
            if(dep->isConflict()) {
                conflicted = true;
            }
            else {
                for(tDependency_it dit = dep->m_dependency.begin(); 
                    dit != dep->m_dependency.end(); dit++) {
                    //! search for entry which has the same driver
                    tDependency_it fit = m_dependency.find(*dit);
                    if(fit != m_dependency.end()) {
                        ASSERT(fit->driver == dit->driver);
                        if(fit->time != dit->time) {
                            //! the same driver and different times 
                            conflicted = true;
                            break;
                        }
                    }
                    else {
                        m_dependency.insert(*dit);
                    }
                }
            }
        }
    }
    m_bConflict = conflicted;
    return conflicted;
}
void
XRecordDependency::clear()
{
    m_dependency.clear();
    m_bConflict = false;
}
