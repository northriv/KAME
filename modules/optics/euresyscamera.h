/***************************************************************************
        Copyright (C) 2002-2024 Kentaro Kitagawa
		                   kitag@issp.u-tokyo.ac.jp
		
		This program is free software; you can redistribute it and/or
		modify it under the terms of the GNU Library General Public
		License as published by the Free Software Foundation; either
		version 2 of the License, or (at your option) any later version.
		
		You should have received a copy of the GNU Library General 
		Public License and a list of authors along with this program; 
		see the files COPYING and AUTHORS.
***************************************************************************/
#ifndef euresyscameraH
#define euresyscameraH

#include "digitalcamera.h"
//---------------------------------------------------------------------------

#if defined USE_EURESYS_EGRABBER
#include <EGrabber.h>

class XEGrabberInterface : public XInterface {
public:
    XEGrabberInterface(const char *name, bool runtime, const shared_ptr<XDriver> &driver);
    virtual ~XEGrabberInterface();

    virtual bool isOpened() const override {return !!m_camera;}

    void lock() {s_mutex.lock();} //!<overrides XInterface::lock().
    void unlock() {s_mutex.unlock();}
    bool isLocked() const {return s_mutex.isLockedByCurrentThread();}

    const shared_ptr<Euresys::EGrabber<>> &camera() const {return m_camera;}
protected:
    virtual void open() override;
    //! This can be called even if has already closed.
    virtual void close() override;
private:
    static XRecursiveMutex s_mutex;
    shared_ptr<Euresys::EGrabber<>> m_camera;
    static unique_ptr<Euresys::EGenTL> s_gentl;
    static unique_ptr<Euresys::EGrabberDiscovery> s_discovery;
    static int s_refcnt;
};

template<class tDriver>
class XEGrabberDriver : public tDriver {
public:
    XEGrabberDriver(const char *name, bool runtime,
        Transaction &tr_meas, const shared_ptr<XMeasure> &meas);
protected:
    const shared_ptr<XEGrabberInterface> &interface() const {return m_interface;}
    //! Be called just after opening interface. Call start() inside this routine appropriately.
    virtual void open() {this->start();}
    //! Be called during stopping driver. Call interface()->stop() inside this routine.
    virtual void close() {interface()->stop();}
    void onOpen(const Snapshot &shot, XInterface *);
    void onClose(const Snapshot &shot, XInterface *);
    //! This should not cause an exception.
    virtual void closeInterface() override;
private:
    shared_ptr<Listener> m_lsnOnOpen, m_lsnOnClose;

    const shared_ptr<XEGrabberInterface> m_interface;
};
template<class tDriver>
XEGrabberDriver<tDriver>::XEGrabberDriver(const char *name, bool runtime,
    Transaction &tr_meas, const shared_ptr<XMeasure> &meas) :
    tDriver(name, runtime, tr_meas, meas),
    m_interface(XNode::create<XEGrabberInterface>("Interface", false,
                                                 dynamic_pointer_cast<XDriver>(this->shared_from_this()))) {
    meas->interfaces()->insert(tr_meas, m_interface);
    this->iterate_commit([=](Transaction &tr){
        m_lsnOnOpen = tr[ *interface()].onOpen().connectWeakly(
            this->shared_from_this(), &XEGrabberDriver<tDriver>::onOpen);
        m_lsnOnClose = tr[ *interface()].onClose().connectWeakly(
            this->shared_from_this(), &XEGrabberDriver<tDriver>::onClose);
    });
}
template<class tDriver>
void
XEGrabberDriver<tDriver>::onOpen(const Snapshot &shot, XInterface *) {
    try {
        open();
    }
    catch (XInterface::XInterfaceError& e) {
        e.print(this->getLabel() + i18n(": Opening driver failed, because "));
        onClose(shot, NULL);
    }
}
template<class tDriver>
void
XEGrabberDriver<tDriver>::onClose(const Snapshot &shot, XInterface *) {
    try {
        this->stop();
    }
    catch (XInterface::XInterfaceError& e) {
        e.print(this->getLabel() + i18n(": Stopping driver failed, because "));
        closeInterface();
    }
}
template<class tDriver>
void
XEGrabberDriver<tDriver>::closeInterface() {
    try {
        this->close();
    }
    catch (XInterface::XInterfaceError &e) {
        e.print();
    }
}


//! Camralink/coaxpress camera via Euresys egrabber
class XEGrabberCamera : public XEGrabberDriver<XDigitalCamera> {
public:
    XEGrabberCamera(const char *name, bool runtime,
		Transaction &tr_meas, const shared_ptr<XMeasure> &meas);
    virtual ~XEGrabberCamera() {}
protected:
    virtual void setVideoMode(unsigned int mode, unsigned int roix = 0, unsigned int roiy = 0, unsigned int roiw = 0, unsigned int roih = 0) override;
    virtual void setTriggerMode(TriggerMode mode) override;
    virtual void setBrightness(unsigned int gain) override;
    virtual void setCameraGain(double db) override;
    virtual void setExposureTime(double time) override;

	//! Be called just after opening interface. Call start() inside this routine appropriately.
    virtual void open() override;

    virtual void analyzeRaw(RawDataReader &reader, Transaction &tr) override;
    virtual XTime acquireRaw(shared_ptr<RawData> &) override;
private:
    void stopTransmission();
    atomic<bool> m_isTrasmitting;
};
#endif //USE_EURESYS_EGRABBER

#endif
