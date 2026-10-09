/***************************************************************************
        Copyright (C) 2002-2026 Kentaro Kitagawa
                           kitag@issp.u-tokyo.ac.jp

        This program is free software; you can redistribute it and/or
        modify it under the terms of the GNU General Public
        License as published by the Free Software Foundation; either
        version 2 of the License, or (at your option) any later version.

        You should have received a copy of the GNU General
        Public License and a list of authors along with this program;
        see the files COPYING and AUTHORS.
***************************************************************************/
//---------------------------------------------------------------------------

#ifndef webcameraH
#define webcameraH

#include "digitalcamera.h"
#include "webcambackend.h"
//---------------------------------------------------------------------------

//! Lists the cameras the platform sees in device(); opening one makes camera() non-null.
class XWebCamInterface : public XInterface {
public:
    XWebCamInterface(const char *name, bool runtime, const shared_ptr<XDriver> &driver);
    virtual ~XWebCamInterface() = default;

    virtual bool isOpened() const override {return !!camera();}

    //! nullptr unless opened.  Needs no interface lock, and its callers hold
    //! none while using it: the backend serializes its own calls.
    shared_ptr<WebCam::Camera> camera() const {
        XScopedLock<XMutex> lock(m_cameraMutex);
        return m_camera;
    }
protected:
    virtual void open() override;
    //! This can be called even if has already closed.
    virtual void close() override;
private:
    //! Guards m_camera alone.  Not the interface lock: XInterface::start()
    //! holds that across open(), which on Qt Multimedia waits for the main
    //! thread, and the main thread reaches camera() from GUI listeners.
    mutable XMutex m_cameraMutex;
    shared_ptr<WebCam::Camera> m_camera;
};

//! Built-in or USB video class camera, through AVFoundation or Qt Multimedia
//! (whichever optics.pro found).  Delivers 8-bit luminance only: a webcam's
//! output is gamma-encoded and auto-exposed, which makes it a monitor rather
//! than a measuring instrument.
class XWebCamera : public XDigitalCamera {
public:
    XWebCamera(const char *name, bool runtime,
        Transaction &tr_meas, const shared_ptr<XMeasure> &meas);
    virtual ~XWebCamera() = default;
protected:
    const shared_ptr<XWebCamInterface> &interface() const {return m_interface;}

    //! A ROI is cropped in software: webcams stream whole frames.
    virtual void setVideoMode(unsigned int mode, unsigned int roix = 0, unsigned int roiy = 0,
        unsigned int roiw = 0, unsigned int roih = 0) override;
    //! Continuous or single-shot only; webcams have no trigger input.
    virtual void setTriggerMode(TriggerMode mode) override;
    virtual void setTriggerSrc(const Snapshot &) override {}
    virtual void setBlackLevelOffset(unsigned int) override {}
    virtual void setGain(unsigned int g, unsigned int emgain) override;
    virtual void setExposureTime(double time) override;

    virtual void analyzeRaw(RawDataReader &reader, Transaction &tr) override;
    virtual XTime acquireRaw(shared_ptr<RawData> &) override;
    //! This should not cause an exception.
    virtual void closeInterface() override;
private:
    //! Tag at the head of each raw record, so a later layout can coexist with journals of this one.
    static constexpr uint32_t RAW_MONO8 = 1;

    //! Be called just after opening interface.
    void open();
    void onOpen(const Snapshot &shot, XInterface *);
    void onClose(const Snapshot &shot, XInterface *);
    void onFrameRateChanged(const Snapshot &shot, XValueNodeBase *);
    //! Throws when not opened.
    shared_ptr<WebCam::Camera> camera() const;
    //! Replaces the FrameRate items, keeping this driver's own listener quiet.
    void setFrameRateItems(Transaction &tr, const std::vector<double> &rates);

    const shared_ptr<XWebCamInterface> m_interface;
    shared_ptr<Listener> m_lsnOnOpen, m_lsnOnClose, m_lsnOnFrameRateChanged;

    //! Guards the members below, up to m_roi; held only briefly, never across a transaction.
    XMutex m_mutex;
    std::vector<WebCam::Format> m_formats;
    int m_formatIndex = -1;
    std::vector<double> m_rates; //!< FrameRate items for the current format; [0] is its maximum.
    struct ROI {unsigned int x = 0, y = 0, w = 0, h = 0;} m_roi; //!< w == 0: whole frame.

    atomic<double> m_minFrameInterval; //!< [s]; 0 takes every frame the camera delivers.
    atomic<bool> m_singleShot, m_singleShotPending;

    //! Acquisition thread only.
    WebCam::Frame m_frame;
    int64_t m_lastDelivered_us = 0;
    double m_fpsMeasured = 0;
};

#endif
