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
//---------------------------------------------------------------------------

#ifndef digitalcameraH
#define digitalcameraH
//---------------------------------------------------------------------------
#include "primarydriverwiththread.h"
#include "xnodeconnector.h"

class XScalarEntry;
class QMainWindow;
class Ui_FrmDigitalCamera;
typedef QForm<QMainWindow, Ui_FrmDigitalCamera> FrmDigitalCamera;

class XGraph;;
class X2DImage;

//! Base class for scientific/machine vision digital camera.
class DECLSPEC_SHARED XDigitalCamera : public XPrimaryDriverWithThread {
public:
    XDigitalCamera(const char *name, bool runtime,
		Transaction &tr_meas, const shared_ptr<XMeasure> &meas);
	//! usually nothing to do.
    virtual ~XDigitalCamera() {}
	//! Shows all forms belonging to driver.
    virtual void showForms() override;

    const shared_ptr<XGraph> &graph() const {return m_graph;}

    //! driver specific part below
    const shared_ptr<XUIntNode> &brightness() const {return m_brightness;} //!< gain
    const shared_ptr<XUIntNode> &shutter() const {return m_shutter;} //!< [s]
    const shared_ptr<XUIntNode> &average() const {return m_average;} //
    const shared_ptr<XTouchableNode> &storeDark() const {return m_storeDark;}
    const shared_ptr<XBoolNode> &subtractDark() const {return m_subtractDark;}
    const shared_ptr<XComboNode> &videoMode() const {return m_videoMode;}
    const shared_ptr<XComboNode> &frameRate() const {return m_frameRate;}
    const shared_ptr<XBoolNode> &autoGainForAverage() const {return m_autoGainForAverage;}
    const shared_ptr<XDoubleNode> &gainForAverage() const {return m_gainForAverage;}
    const shared_ptr<XStringNode> &status() const {return m_status;}

    const shared_ptr<X2DImage> &liveImage() const {return m_liveImage;}
    const shared_ptr<X2DImage> &processedImage() const {return m_processedImage;}

    struct Payload : public XPrimaryDriver::Payload {
        unsigned int brightness() const {return m_brightness;}
        double shutter() const {return m_shutter;} //! [s]
        unsigned int accumulated() const {return m_accumulated;}
        double electricDark() const {return m_electric_dark;} //dark count
        double gainForAverage() const {return m_gainForAverage;}
        shared_ptr<QImage> liveImage() const {return m_liveImage;}
        shared_ptr<QImage> processedImage() const {return m_processedImage;}
//    private:
//        friend class XDigitalCamera;
        unsigned int m_brightness;
        double m_shutter;
        double m_gainForAverage;
        XString m_status;
        unsigned int m_accumulated;
        double m_electric_dark;
        shared_ptr<std::vector<int32_t>> m_darkCounts;
        shared_ptr<std::vector<int32_t>> m_summedCounts;
        shared_ptr<QImage> m_liveImage, m_processedImage;
    };
protected:

	//! This function is called after committing XPrimaryDriver::analyzeRaw() or XSecondaryDriver::analyze().
	//! This might be called even if the record is invalid (time() == false).
    virtual void visualize(const Snapshot &shot) override;
  
protected:
    virtual void setVideoMode(unsigned int mode) = 0;
    virtual void setBrightness(unsigned int brightness) = 0;
    virtual void setShutter(unsigned int shutter) = 0;

    //! This function will be called when raw data are written.
    //! Implement this function to convert the raw data to the record (Payload).
    //! \sa analyze()
    virtual void analyzeRaw(RawDataReader &reader, Transaction &tr) override;

    virtual unique_ptr<QImage> acquireRaw(shared_ptr<RawData> &) = 0;

private:
    const shared_ptr<XUIntNode> m_brightness;
    const shared_ptr<XUIntNode> m_shutter;
    const shared_ptr<XUIntNode> m_average;
    const shared_ptr<XTouchableNode> m_storeDark;
    const shared_ptr<XBoolNode> m_subtractDark;
    const shared_ptr<XComboNode> m_videoMode;
    const shared_ptr<XComboNode> m_frameRate;
    const shared_ptr<XBoolNode> m_autoGainForAverage;
    const shared_ptr<XDoubleNode> m_gainForAverage;
    const shared_ptr<XStringNode> m_status;


    const qshared_ptr<FrmDigitalCamera> m_form;
    const shared_ptr<X2DImage> m_liveImage, m_processedImage;

    shared_ptr<Listener> m_lsnOnVideoModeChanged;
    shared_ptr<Listener> m_lsnOnBrightnessChanged;
    shared_ptr<Listener> m_lsnOnShutterChanged;
    shared_ptr<Listener> m_lsnOnStoreDarkTouched;

    void onVideoModeChanged(const Snapshot &shot, XValueNodeBase *);
    void onBrightnessChanged(const Snapshot &shot, XValueNodeBase *);
    void onShutterChanged(const Snapshot &shot, XValueNodeBase *);

    std::deque<xqcon_ptr> m_conUIs;

	shared_ptr<XGraph> m_graph;

    void onStoreDarkTouched(const Snapshot &shot, XTouchableNode *);

    bool m_storeDarkInvoked;
    constexpr static unsigned int NumAverageCountsPool = 2;
    std::shared_ptr<std::vector<int32_t>> m_averageCountsPool[NumAverageCountsPool];

    void *execute(const atomic<bool> &) override;
};

//---------------------------------------------------------------------------

#endif
