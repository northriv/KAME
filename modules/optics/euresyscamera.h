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
#include "charinterface.h"
//---------------------------------------------------------------------------

#if defined USE_EURESYS_EGRABBER
#include <EGrabber.h>

class XEGrabberInterface : public XCustomCharInterface {
public:
    XEGrabberInterface(const char *name, bool runtime, const shared_ptr<XDriver> &driver, bool grablink = false);
    virtual ~XEGrabberInterface();

    virtual bool isOpened() const override {return !!m_camera;}

    virtual void lock() override; //!<overrides XInterface::lock().
    virtual void unlock() override;
    virtual bool isLocked() const override {return s_mutex.isLockedByCurrentThread();}

    //For cameralink cameras.
    virtual void send(const char *) override;
    virtual void receive() override;
    void flush();

    using Camera = Euresys::EGrabber<>;

    const shared_ptr<Camera> &camera() const {return m_camera;}

    //For cameralink cameras.
    void setSerialBaudRate(unsigned int rate) {m_serialBaudRate = rate;}
    void setSerialEOS(const char *str) {m_serialEOS = str;} //!< be overridden by \a setEOS().
protected:
    virtual void open() override;
    //! This can be called even if has already closed.
    virtual void close() override;

private:
    bool m_bIsSerialPortOpened;
    void checkAndOpenSerialPort();
    void closeSerialPort();

    static XRecursiveMutex s_mutex;
    shared_ptr<Camera> m_camera;
    static unique_ptr<Euresys::EGenTL> s_gentl;
    static unique_ptr<Euresys::EGrabberDiscovery> s_discovery;
    static int s_refcnt;

    XString m_serialEOS;
    unsigned int m_serialBaudRate;
};

template<class tDriver>
class XEGrabberDriver : public tDriver {
public:
    XEGrabberDriver(const char *name, bool runtime,
        Transaction &tr_meas, const shared_ptr<XMeasure> &meas, bool grablink = false);
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
    Transaction &tr_meas, const shared_ptr<XMeasure> &meas, bool grablink) :
    tDriver(name, runtime, tr_meas, meas),
    m_interface(XNode::create<XEGrabberInterface>("Interface", false,
        dynamic_pointer_cast<XDriver>(this->shared_from_this()), grablink)) {
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


//! Coaxpress camera via Euresys egrabber
class XEGrabberCamera : public XEGrabberDriver<XDigitalCamera> {
public:
    XEGrabberCamera(const char *name, bool runtime,
        Transaction &tr_meas, const shared_ptr<XMeasure> &meas, bool grablink = false);
    virtual ~XEGrabberCamera() {}
protected:
    virtual void setVideoMode(unsigned int mode, unsigned int roix = 0, unsigned int roiy = 0, unsigned int roiw = 0, unsigned int roih = 0) override;
    virtual void setTriggerMode(TriggerMode mode) override;
    virtual void setTriggerSrc(const Snapshot &) override {}
    virtual void setBlackLevelOffset(unsigned int v) override;
    virtual void setGain(unsigned int g, unsigned int emgain) override;
    virtual void setExposureTime(double time) override;

	//! Be called just after opening interface. Call start() inside this routine appropriately.
    virtual void open() override;

    virtual void analyzeRaw(RawDataReader &reader, Transaction &tr) override;
    virtual XTime acquireRaw(shared_ptr<RawData> &) override;

    void stopTransmission();
    atomic<bool> m_isTrasmitting;

    virtual bool pushFeatureSerialCommand(shared_ptr<RawData> &, const std::string &featname) {return false;}
    virtual std::pair<unsigned int, unsigned int> setVideoModeViaSerial(unsigned int roix, unsigned int roiw, unsigned int roiy, unsigned int roih) {return {};}
    virtual void setTriggerModeViaSerial(TriggerMode mode) {};
private:
    std::vector<std::string> m_featuresInRemoteModule;
    bool isFeatureAvailableInRemoteModule(const std::string &s) const {
        return
            std::find(m_featuresInRemoteModule.begin(), m_featuresInRemoteModule.end(), s) != m_featuresInRemoteModule.end();
    }
};
//! Cameralink camera via Euresys egrabber
class XGrablinkCamera : public XEGrabberCamera {
public:
    XGrablinkCamera(const char *name, bool runtime,
        Transaction &tr_meas, const shared_ptr<XMeasure> &meas);
    virtual ~XGrablinkCamera() {}
};

class XHamamatsuCameraOverGrablink : public XEGrabberCamera {
public:
    XHamamatsuCameraOverGrablink(const char *name, bool runtime,
        Transaction &tr_meas, const shared_ptr<XMeasure> &meas);
    virtual ~XHamamatsuCameraOverGrablink() {}
protected:
    virtual void setBlackLevelOffset(unsigned int v) override;
    virtual void setGain(unsigned int g, unsigned int emgain) override;
    virtual void setExposureTime(double time) override;

    virtual void afterOpen() override;

    virtual bool pushFeatureSerialCommand(shared_ptr<RawData> &, const std::string &featname) override;
    virtual std::pair<unsigned int, unsigned int> setVideoModeViaSerial(unsigned int roix, unsigned int roiw, unsigned int roiy, unsigned int roih) override;
    virtual void setTriggerModeViaSerial(TriggerMode mode) override;

    void checkSerialError(const char *file, unsigned int line);

    bool m_bIsEM = false;
    bool m_bIsCooled = false;
    unsigned int m_xdummypx, m_xdatapx, m_ydummypx, m_ydatapx; //dummy px, available px
    unsigned int m_maxBitsFast, m_maxBitsSlow;
    bool m_bHasSlowScan = false;
    unsigned int m_offsetx, m_offsety;
    double m_rat;
    unsigned int m_emg, m_ceg;
};

class XJAICameraOverGrablink : public XEGrabberCamera {
public:
    XJAICameraOverGrablink(const char *name, bool runtime,
        Transaction &tr_meas, const shared_ptr<XMeasure> &meas);
    virtual ~XJAICameraOverGrablink() {}
protected:
    virtual void setBlackLevelOffset(unsigned int offset) override;
    virtual void setGain(unsigned int g, unsigned int emgain) override;
    virtual void setExposureTime(double time) override;
    virtual void setTriggerSrc(const Snapshot &) override;

    virtual void afterOpen() override;

    virtual bool pushFeatureSerialCommand(shared_ptr<RawData> &, const std::string &) override {return false;}
    virtual std::pair<unsigned int, unsigned int> setVideoModeViaSerial(unsigned int roix, unsigned int roiw, unsigned int roiy, unsigned int roih) override;
    virtual void setTriggerModeViaSerial(TriggerMode mode) override;

    void checkSerialError(const char *file, unsigned int line);

    unsigned int m_sensorWidth, m_sensorHeight;
};
#endif //USE_EURESYS_EGRABBER

#endif
