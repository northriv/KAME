#include "driver.h"
#include "interface.h"
#include <klocale.h>

XDriver::XBufferUnderflowRecordError::XBufferUnderflowRecordError(const char *file, int line) : 
    XRecordError(KAME::i18n("Buffer Underflow."), file, line) {}

XDriverList::XDriverList(const char *name, bool runtime,
   const shared_ptr<XScalarEntryList> &scalarentries,
   const shared_ptr<XInterfaceList> &interfaces,
   const shared_ptr<XThermometerList> &thermometers) :
        XCustomTypeListNode<XDriver>(name, runtime), 
        m_scalarentries(scalarentries),
        m_interfaces(interfaces),
        m_thermometers(thermometers)
{
}

XDriver::XDriver(const char *name, bool runtime, 
   const shared_ptr<XScalarEntryList> &,
   const shared_ptr<XInterfaceList> &,
   const shared_ptr<XThermometerList> &,
   const shared_ptr<XDriverList> &) :
    XNode(name, runtime)
{
}

void
XDriver::startMeas() {
    //! clear time
    m_recordTime = XTime();
    shared_ptr<XRecordDependency> dep(dependency());
    if(dep) dep->clear();
    try {
        start();
    }
    catch (XKameError &e) {
        e.print(getLabel() + "; ");
    }
}
void
XDriver::stopMeas() {
    try {
        stop();
    }
    catch (XKameError &e) {
        e.print(getLabel() + "; ");
    }
}

void
XDriver::readUnlockRecord() const {
    m_recordLock.readUnlock();
}
void
XDriver::readLockRecord() const {
    m_recordLock.readLock();
}

void
XDriver::startRecording() {
    m_recordLock.writeLock();
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
