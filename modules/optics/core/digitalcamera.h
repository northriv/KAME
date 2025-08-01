/***************************************************************************
        Copyright (C) 2002-2025 Kentaro Kitagawa
		                   kitag@issp.u-tokyo.ac.jp
		
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
#include "xwavengraph.h"

class XScalarEntry;
class XGraph;
template<class T> struct Vector4;
class XGraph2DMathToolList;
class QMainWindow;
class Ui_FrmDigitalCamera;
typedef QForm<QMainWindow, Ui_FrmDigitalCamera> FrmDigitalCamera;
class XQGraph;
class X2DImage;
class OnScreenObjectWithMarker;

template <unsigned int NumPool = 5>
struct ImageSpacePoolAllocator {
    local_shared_ptr<std::vector<uint32_t>> allocate(int imagebytes);
private:
    atomic_shared_ptr<std::vector<uint32_t>> m_pool[NumPool];
};

template <unsigned int NumPool>
local_shared_ptr<std::vector<uint32_t>>
ImageSpacePoolAllocator<NumPool>::allocate(int imagesize) {
    local_shared_ptr<std::vector<uint32_t>> pNext, p;
    for(int i = 0; i < NumPool; ++i) {
        if( !m_pool[i])
            m_pool[i] = make_local_shared<std::vector<uint32_t>>(imagesize);
        p.swap(m_pool[i]); //atomic swap
        if(p && p.unique()) { //confirmed uniquness.
            m_pool[i].compareAndSet({}, p); //sharing me for later use.
            pNext = p;
            p->resize(imagesize);
            break;
        }
        m_pool[i].compareAndSet({}, p); //restoring busy one for later use.
    }
    if( !pNext)
        pNext = make_local_shared<std::vector<uint32_t>>(imagesize);
    return pNext;
}

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
    const shared_ptr<XDoubleNode> &cameraGain() const {return m_cameraGain;} //! integer, typ. 0--255
    const shared_ptr<XDoubleNode> &emGain() const {return m_emGain;} //! integer, typ. 0--255
    const shared_ptr<XUIntNode> &blackLvlOffset() const {return m_blackLvlOffset;} //!< offset black for device
    const shared_ptr<XDoubleNode> &exposureTime() const {return m_exposureTime;} //!< [s]
    const shared_ptr<XTouchableNode> &storeDark() const {return m_storeDark;}
    const shared_ptr<XBoolNode> &subtractDark() const {return m_subtractDark;}
    const shared_ptr<XComboNode> &videoMode() const {return m_videoMode;}
    enum class TriggerMode {CONTINUEOUS = 0, SINGLE = 1, EXT_POS_EDGE = 2, EXT_NEG_EDGE = 3, EXT_POS_EXPOSURE = 4, EXT_NEG_EXPOSURE = 5};
    const shared_ptr<XComboNode> &triggerMode() const {return m_triggerMode;}
    const shared_ptr<XComboNode> &triggerSrc() const {return m_triggerSrc;}
    const shared_ptr<XComboNode> &frameRate() const {return m_frameRate;}
    const shared_ptr<XUIntNode> &antiShakePixels() const {return m_antiShakePixels;}
    const shared_ptr<XBoolNode> &autoGainForDisp() const {return m_autoGainForDisp;}
    const shared_ptr<XDoubleNode> &gainForDisp() const {return m_gainForDisp;}

    struct Payload : public XPrimaryDriver::Payload {
//        double cameraGain() const {return m_cameraGain;}
        unsigned int blackLvlOffset() const {return m_blackLvlOffset;}
        double exposureTime() const {return m_exposureTime;} //! [s]
        double electricDark() const {return m_electric_dark;} //dark count
        unsigned int width() const {return m_width;}
        unsigned int height() const {return m_height;}
        unsigned int stride() const {return m_stride;} //stride != width when antishake is on.
        unsigned int firstPixel() const {return m_firstPixel;} //not zero when antishake is on.
        local_shared_ptr<std::vector<uint32_t>> rawCounts() const {return m_rawCounts;}
        local_shared_ptr<std::vector<uint32_t>> darkCounts() const {return m_darkCounts;}
//    private:
//        friend class XDigitalCamera;
//        double m_cameraGain;
        unsigned int m_stride;
        unsigned int m_firstPixel;
        unsigned int m_blackLvlOffset;
        double m_exposureTime;
        XString m_status;
        double m_electric_dark;
        unsigned int m_dark;
        unsigned int m_width, m_height;
        local_shared_ptr<std::vector<uint32_t>> m_darkCounts;
        local_shared_ptr<std::vector<uint32_t>> m_rawCounts;
        shared_ptr<QImage> m_qimage;
        double m_cogXOrig, m_cogYOrig; //for antishake.
        std::vector<uint32_t> m_histogram;
        double m_maxIntensity, m_minIntensity, m_modeIntensity;
        unsigned int m_antishake_pixels = {};
        bool m_storeDarkInvoked = false, m_storeAntiShakeInvoked = false;
    };
protected:

	//! This function is called after committing XPrimaryDriver::analyzeRaw() or XSecondaryDriver::analyze().
	//! This might be called even if the record is invalid (time() == false).
    virtual void visualize(const Snapshot &shot) override;
  
protected:
    virtual void afterOpen() {};

    virtual void setVideoMode(unsigned int mode, unsigned int roix = 0, unsigned int roiy = 0,
        unsigned int roiw = 0, unsigned int roih = 0) = 0;
    virtual void setTriggerMode(TriggerMode mode) = 0;
    virtual void setTriggerSrc(const Snapshot &) = 0;
    virtual void setBlackLevelOffset(unsigned int lv) = 0;
    virtual void setGain(unsigned int g, unsigned int emgain) = 0;
    virtual void setExposureTime(double time) = 0;

    virtual XTime acquireRaw(shared_ptr<RawData> &) = 0;

    void setGrayImage(RawDataReader &reader, Transaction &tr, uint32_t width, uint32_t height, bool big_endian = false, bool mono16 = false);
private:
    const shared_ptr<XDoubleNode> m_cameraGain;
    const shared_ptr<XDoubleNode> m_emGain;
    const shared_ptr<XUIntNode> m_blackLvlOffset;
    const shared_ptr<XDoubleNode> m_exposureTime;
    const shared_ptr<XTouchableNode> m_storeDark;
    const shared_ptr<XTouchableNode> m_roiSelectionTool;
    const shared_ptr<XUIntNode> m_antiShakePixels;
    const shared_ptr<XBoolNode> m_subtractDark;
    const shared_ptr<XComboNode> m_videoMode;
    const shared_ptr<XComboNode> m_triggerMode;
    const shared_ptr<XComboNode> m_triggerSrc;
    const shared_ptr<XComboNode> m_frameRate;
    const shared_ptr<XBoolNode> m_autoGainForDisp;
    const shared_ptr<XDoubleNode> m_gainForDisp;

    const qshared_ptr<FrmDigitalCamera> m_form;
    const shared_ptr<X2DImage> m_liveImage;

    shared_ptr<Listener> m_lsnOnVideoModeChanged;
    shared_ptr<Listener> m_lsnOnTriggerModeChanged;
    shared_ptr<Listener> m_lsnOnTriggerSrcChanged;
    shared_ptr<Listener> m_lsnOnGainChanged;
    shared_ptr<Listener> m_lsnOnBlackLevelOffsetChanged;
    shared_ptr<Listener> m_lsnOnExposureTimeChanged;
    shared_ptr<Listener> m_lsnOnStoreDarkTouched;
    shared_ptr<Listener> m_lsnOnROISelectionToolTouched;
    shared_ptr<Listener> m_lsnOnROISelectionToolFinished;
    shared_ptr<Listener> m_lsnOnAntiShakeChanged;

    void onVideoModeChanged(const Snapshot &shot, XValueNodeBase *);
    void onTriggerModeChanged(const Snapshot &shot, XValueNodeBase *);
    void onTriggerSrcChanged(const Snapshot &shot, XValueNodeBase *);
    void onGainChanged(const Snapshot &shot, XValueNodeBase *);
    void onBlackLevelOffsetChanged(const Snapshot &shot, XValueNodeBase *);
    void onExposureTimeChanged(const Snapshot &shot, XValueNodeBase *);

    std::deque<xqcon_ptr> m_conUIs;

    shared_ptr<XGraph> m_graph;

    shared_ptr<XGraph2DMathToolList> m_graphToolList;

    const shared_ptr<XWaveNGraph> m_waveHist;

    void onStoreDarkTouched(const Snapshot &shot, XTouchableNode *);
    void onAntiShakeChanged(const Snapshot &shot, XValueNodeBase *);

    void onROISelectionToolTouched(const Snapshot &shot, XTouchableNode *);
    void onROISelectionToolFinished(const Snapshot &shot,
        const std::tuple<XString, Vector4<double>, Vector4<double>, XQGraph*>&);

    ImageSpacePoolAllocator<3> m_pool;

    void *execute(const atomic<bool> &) override;
};

//---------------------------------------------------------------------------

#endif
