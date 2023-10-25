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

#ifndef odmrImagingH
#define odmrImagingH
//---------------------------------------------------------------------------
#include "secondarydriver.h"
#include "xnodeconnector.h"

class XDigitalCamera;
class QMainWindow;
class Ui_FrmODMRImaging;
typedef QForm<QMainWindow, Ui_FrmODMRImaging> FrmODMRImaging;

class X2DImage;
class OnScreenObjectWithMarker;
class XGraph2DMathTool;
class XGraph2DMathToolList;
class XQGraph2DMathToolConnector;
class XScalarEntry;

//! ODMR Imaging from camera capture images.
class DECLSPEC_SHARED XODMRImaging : public XSecondaryDriver {
public:
    XODMRImaging(const char *name, bool runtime,
		Transaction &tr_meas, const shared_ptr<XMeasure> &meas);
	//! usually nothing to do.
    virtual ~XODMRImaging();
	//! Shows all forms belonging to driver.
    virtual void showForms() override;

    //! driver specific part below
    const shared_ptr<XUIntNode> &average() const {return m_average;} //
    const shared_ptr<XTouchableNode> &clearAverage() const {return m_clearAverage;}
    const shared_ptr<XBoolNode> &incrementalAverage() const {return m_incrementalAverage;}
    const shared_ptr<XBoolNode> &autoGainForDisp() const {return m_autoGainForDisp;}
    const shared_ptr<XUIntNode> &wheelIndex() const {return m_wheelIndex;} //!< For color wheel or Delta PL/PL measurement, 0 for off-resonance.
    const shared_ptr<XDoubleNode> &gainForDisp() const {return m_gainForDisp;}
    const shared_ptr<XDoubleNode> &minDPLoPLForDisp() const {return m_minDPLoPLForDisp;}//!< [%]
    const shared_ptr<XDoubleNode> &maxDPLoPLForDisp() const {return m_maxDPLoPLForDisp;}//!< [%]
    const shared_ptr<XComboNode> &dispMethod() const {return m_dispMethod;}
    const shared_ptr<XUIntNode> &refIntensFrames() const {return m_refIntensFrames;}

    const shared_ptr<X2DImage> &processedImage() const {return m_processedImage;}

    struct Payload : public XSecondaryDriver::Payload {
        const std::vector<double> &sampleIntensities(bool is_mw_on) const {return m_sampleIntensities[is_mw_on ? 1 : 0];}
        const std::vector<double> &referenceIntensities(bool is_mw_on) const {return m_referenceIntensities[is_mw_on ? 1 : 0];}
        double pl(bool is_mw_on, unsigned int i) const {
            double pl__ = sampleIntensities(is_mw_on)[i];
            return pl__;
        }
        double dPL(unsigned int i) const {
            double pl_off = pl(false, i);
            double pl_on = pl(true, i);
            return pl_on - pl_off;
        }
        double dPLoPL(unsigned int i) const {
            return dPL(i) / m_pl0[i];
        }
        unsigned int numSamples() const {return m_sampleIntensities[0].size();}
        double gainForDisp() const {return m_gainForDisp;}
        unsigned int width() const {return m_width;}
        unsigned int height() const {return m_height;}
    protected:
        friend class XODMRImaging;
        double m_gainForDisp;
        unsigned int m_accumulated[2];
        unsigned int m_wheelIndex;
        local_shared_ptr<std::vector<uint32_t>> m_summedCounts[2];//MW off and on.
        double m_coefficients[2];
        std::vector<double> m_sampleIntensities[2];
        std::vector<double> m_pl0;
        std::vector<double> m_referenceIntensities[2];
        XTime m_timeClearRequested;
        unsigned int m_width, m_height;
        bool isCurrentImageMWOn() const {return m_accumulated[0] > m_accumulated[1];}
        unsigned int currentIndex() const {return isCurrentImageMWOn() ? 1 : 0;}
        shared_ptr<QImage> m_qimage;
    };
protected:
    //! This function is called when a connected driver emit a signal
    virtual void analyze(Transaction &tr, const Snapshot &shot_emitter, const Snapshot &shot_others,
         XDriver *emitter) override;
    //! This function is called after committing XPrimaryDriver::analyzeRaw() or XSecondaryDriver::analyze().
    //! This might be called even if the record is invalid (time() == false).
    virtual void visualize(const Snapshot &shot) override;
    //! Checks if the connected drivers have valid time stamps.
    //! \return true if dependency is resolved.
    //! This function must be reentrant unlike analyze().
    virtual bool checkDependency(const Snapshot &shot_this,
        const Snapshot &shot_emitter, const Snapshot &shot_others,
        XDriver *emitter) const override;

    const shared_ptr<XItemNode<XDriverList, XDigitalCamera> > &camera() const {return m_camera;}

    virtual void analyzeIntensities(Transaction &tr) {};
private:
    const shared_ptr<XItemNode<XDriverList, XDigitalCamera> > m_camera;
    const shared_ptr<XUIntNode> m_average;
    const shared_ptr<XTouchableNode> m_clearAverage;
    const shared_ptr<XBoolNode> m_autoGainForDisp;
    const shared_ptr<XBoolNode> m_incrementalAverage;
    const shared_ptr<XUIntNode> m_wheelIndex;
    const shared_ptr<XDoubleNode> m_gainForDisp;
    const shared_ptr<XDoubleNode> m_minDPLoPLForDisp; //!< [%]
    const shared_ptr<XDoubleNode> m_maxDPLoPLForDisp; //!< [%]
    const shared_ptr<XComboNode> m_dispMethod;
    const shared_ptr<XUIntNode> m_refIntensFrames;

    const qshared_ptr<FrmODMRImaging> m_form;
    const shared_ptr<X2DImage> m_processedImage;
    std::deque<shared_ptr<XGraph2DMathToolList>> m_sampleToolLists, m_referenceToolLists, m_darkToolLists; //PL for MW off and on.

    shared_ptr<Listener> m_lsnOnClearAverageTouched, m_lsnOnCondChanged;

    std::map<XNode*, shared_ptr<XScalarEntry>> m_samplePLEntries, m_sampleDPLoPLEntries;
    weak_ptr<XScalarEntryList> m_entries;

    std::deque<xqcon_ptr> m_conUIs;

    std::deque<shared_ptr<XQGraph2DMathToolConnector>> m_conTools;

    void onClearAverageTouched(const Snapshot &shot, XTouchableNode *);
    void onCondChanged(const Snapshot &shot, XValueNodeBase *);

    constexpr static unsigned int NumSummedCountsPool = 3;
    atomic_shared_ptr<std::vector<uint32_t>> m_summedCountsPool[NumSummedCountsPool];
    local_shared_ptr<std::vector<uint32_t>> summedCountsFromPool(int imagebytes);
};

//---------------------------------------------------------------------------

#endif
