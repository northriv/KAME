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
    const shared_ptr<XUIntNode> &gain() const {return m_gain;} //!< gain for device
    const shared_ptr<XUIntNode> &shutter() const {return m_shutter;} //!< [s]
    const shared_ptr<XUIntNode> &average() const {return m_average;} //
    const shared_ptr<XTouchableNode> &storeDark() const {return m_storeDark;}
    const shared_ptr<XBoolNode> &subtractDark() const {return m_subtractDark;}
    const shared_ptr<XBoolNode> &incrementalAverage() const {return m_incrementalAverage;}
    const shared_ptr<XComboNode> &videoMode() const {return m_videoMode;}
    enum class TriggerMode {CONTINUEOUS = 0, SINGLE = 1, EXT_POS_EDGE = 2, EXT_NEG_EDGE = 3};
    const shared_ptr<XComboNode> &triggerMode() const {return m_triggerMode;}
    enum class ColoringMethod {MONO = 0, RGBWHEEL = 1, DPL_PL = 2};
    constexpr static unsigned int MAX_COLORS = 3;
    const shared_ptr<XComboNode> &coloringMethod() const {return m_coloringMethod;}
    const shared_ptr<XComboNode> &frameRate() const {return m_frameRate;}
    const shared_ptr<XBoolNode> &autoGainForAverage() const {return m_autoGainForAverage;}
    const shared_ptr<XUIntNode> &colorIndex() const {return m_colorIndex;} //!< For color wheel or Delta PL/PL measurement, 0 for off-resonance.
    const shared_ptr<XDoubleNode> &gainForAverage() const {return m_gainForAverage;}

    const shared_ptr<X2DImage> &liveImage() const {return m_liveImage;}
    const shared_ptr<X2DImage> &processedImage() const {return m_processedImage;}

    struct Payload : public XPrimaryDriver::Payload {
        unsigned int gain() const {return m_gain;}
        double shutter() const {return m_shutter;} //! [s]
        unsigned int accumulated(unsigned int cidx) const {return m_accumulated[cidx];}
        unsigned int accumulated() const {return m_accumulated[m_maxColorIndex];}
        double electricDark() const {return m_electric_dark;} //dark count
        double gainForAverage() const {return m_gainForAverage;}
        shared_ptr<QImage> liveImage() const {return m_liveImage;}
        shared_ptr<QImage> processedImage() const {return m_processedImage;}
//    private:
//        friend class XDigitalCamera;
        unsigned int m_gain;
        double m_shutter;
        double m_gainForAverage;
        XString m_status;
        unsigned int m_accumulated[MAX_COLORS];
        double m_electric_dark;
        unsigned int m_darkAccumulated;
        unsigned int m_colorIndex;
        unsigned int m_maxColorIndex;
        local_shared_ptr<std::vector<uint32_t>> m_darkCounts;
        local_shared_ptr<std::vector<uint32_t>> m_summedCounts[MAX_COLORS];
        shared_ptr<QImage> m_liveImage, m_processedImage;
        int32_t m_avMin, m_avMax;
    };
protected:

	//! This function is called after committing XPrimaryDriver::analyzeRaw() or XSecondaryDriver::analyze().
	//! This might be called even if the record is invalid (time() == false).
    virtual void visualize(const Snapshot &shot) override;
  
protected:
    virtual void setVideoMode(unsigned int mode) = 0;
    virtual void setTriggerMode(TriggerMode mode) = 0;
    virtual void setGain(unsigned int gain ) = 0;
    virtual void setShutter(unsigned int shutter) = 0;

    virtual XTime acquireRaw(shared_ptr<RawData> &) = 0;

    void setGrayImage(RawDataReader &reader, Transaction &tr, uint32_t width, uint32_t height, bool big_endian = false, bool mono16 = false);
private:
    const shared_ptr<XUIntNode> m_gain;
    const shared_ptr<XUIntNode> m_shutter;
    const shared_ptr<XUIntNode> m_average;
    const shared_ptr<XTouchableNode> m_storeDark;
    const shared_ptr<XTouchableNode> m_clearAverage;
    const shared_ptr<XBoolNode> m_subtractDark;
    const shared_ptr<XComboNode> m_videoMode;
    const shared_ptr<XComboNode> m_triggerMode;
    const shared_ptr<XComboNode> m_frameRate;
    const shared_ptr<XComboNode> m_coloringMethod;
    const shared_ptr<XBoolNode> m_autoGainForAverage;
    const shared_ptr<XBoolNode> m_incrementalAverage;
    const shared_ptr<XUIntNode> m_colorIndex;
    const shared_ptr<XDoubleNode> m_gainForAverage;


    const qshared_ptr<FrmDigitalCamera> m_form;
    const shared_ptr<X2DImage> m_liveImage, m_processedImage;

    shared_ptr<Listener> m_lsnOnVideoModeChanged;
    shared_ptr<Listener> m_lsnOnTriggerModeChanged;
    shared_ptr<Listener> m_lsnOnGainChanged;
    shared_ptr<Listener> m_lsnOnShutterChanged;
    shared_ptr<Listener> m_lsnOnStoreDarkTouched;
    shared_ptr<Listener> m_lsnOnClearAverageTouched;

    void onVideoModeChanged(const Snapshot &shot, XValueNodeBase *);
    void onTriggerModeChanged(const Snapshot &shot, XValueNodeBase *);
    void onGainChanged(const Snapshot &shot, XValueNodeBase *);
    void onShutterChanged(const Snapshot &shot, XValueNodeBase *);

    std::deque<xqcon_ptr> m_conUIs;

	shared_ptr<XGraph> m_graph;

    void onStoreDarkTouched(const Snapshot &shot, XTouchableNode *);
    void onClearAverageTouched(const Snapshot &shot, XTouchableNode *);

    bool m_storeDarkInvoked;
    bool m_clearAverageInvoked; //todo time
    constexpr static unsigned int NumSummedCountsPool = MAX_COLORS + 1;
    atomic_shared_ptr<std::vector<uint32_t>> m_summedCountsPool[NumSummedCountsPool];
    local_shared_ptr<std::vector<uint32_t>> summedCountsFromPool(int imagebytes);

    void *execute(const atomic<bool> &) override;
};

//---------------------------------------------------------------------------

#endif
