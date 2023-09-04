/***************************************************************************
        Copyright (C) 2002-2023 Kentaro Kitagawa
		                   kitagawa@phys.s.u-tokyo.ac.jp
		
		This program is free software; you can redistribute it and/or
		modify it under the terms of the GNU Library General Public
		License as published by the Free Software Foundation; either
		version 2 of the License, or (at your option) any later version.
		
		You should have received a copy of the GNU Library General 
		Public License and a list of authors along with this program; 
		see the files COPYING and AUTHORS.
***************************************************************************/
#ifndef iidccameraH
#define iidccameraH

#include "digitalcamera.h"
#include "dc1394/dc1394.h"
//---------------------------------------------------------------------------

#if defined USE_LIBDC1394
class XDC1394Interface : public XInterface {
public:
    XDC1394Interface(const char *name, bool runtime, const shared_ptr<XDriver> &driver);

    virtual bool isOpened() const override {return m_devname.length();}

    //! e.g. "Dev1".
    const char*devName() const {return m_devname.c_str();}
protected:
    virtual void open() override;
    //! This can be called even if has already closed.
    virtual void close() override;
private:
    XString m_devname;
};

template<class tDriver>
class XDC1394Driver : public tDriver {
public:
    XDC1394Driver(const char *name, bool runtime,
        Transaction &tr_meas, const shared_ptr<XMeasure> &meas);
protected:
    const shared_ptr<XDC1394Interface> &interface() const {return m_interface;}
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

    const shared_ptr<XDC1394Interface> m_interface;
};
template<class tDriver>
XDC1394Driver<tDriver>::XDC1394Driver(const char *name, bool runtime,
    Transaction &tr_meas, const shared_ptr<XMeasure> &meas) :
    tDriver(name, runtime, tr_meas, meas),
    m_interface(XNode::create<XDC1394Interface>("Interface", false,
                                                 dynamic_pointer_cast<XDriver>(this->shared_from_this()))) {
    meas->interfaces()->insert(tr_meas, m_interface);
    this->iterate_commit([=](Transaction &tr){
        m_lsnOnOpen = tr[ *interface()].onOpen().connectWeakly(
            this->shared_from_this(), &XDC1394Driver<tDriver>::onOpen);
        m_lsnOnClose = tr[ *interface()].onClose().connectWeakly(
            this->shared_from_this(), &XDC1394Driver<tDriver>::onClose);
    });
}
template<class tDriver>
void
XDC1394Driver<tDriver>::onOpen(const Snapshot &shot, XInterface *) {
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
XDC1394Driver<tDriver>::onClose(const Snapshot &shot, XInterface *) {
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
XDC1394Driver<tDriver>::closeInterface() {
    try {
        this->close();
    }
    catch (XInterface::XInterfaceError &e) {
        e.print();
    }
}


//! IIDC camera via libdc1394
class XIIDCCamera : public XDC1394Driver<XDigitalCamera> {
public:
    XIIDCCamera(const char *name, bool runtime,
		Transaction &tr_meas, const shared_ptr<XMeasure> &meas);
    virtual ~XIIDCCamera() {}
protected:
    virtual void onAverageChanged(const Snapshot &shot, XValueNodeBase *) override;
    virtual void onExposureChanged(const Snapshot &shot, XValueNodeBase *) override;

	//! Be called just after opening interface. Call start() inside this routine appropriately.
    virtual void open() override;
    virtual unique_ptr<QImage> acquireRaw(shared_ptr<RawData> &) override;
private:
};
#endif //USE_LIBDC1394

#endif
