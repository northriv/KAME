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
//---------------------------------------------------------------------------

#ifndef imageProcessorH
#define imageProcessorH
//---------------------------------------------------------------------------
#include "secondarydriver.h"
#include "xnodeconnector.h"

class XDigitalCamera;
class XFilterWheel;
class QMainWindow;
class Ui_FrmImageProcessor;
typedef QForm<QMainWindow, Ui_FrmImageProcessor> FrmImageProcessor;

class X2DImage;
class XScalarEntry;

//! ODMR Imaging from camera capture images.
class DECLSPEC_SHARED XImageProcessor : public XSecondaryDriver {
public:
    XImageProcessor(const char *name, bool runtime,
		Transaction &tr_meas, const shared_ptr<XMeasure> &meas);
	//! usually nothing to do.
    virtual ~XImageProcessor();
	//! Shows all forms belonging to driver.
    virtual void showForms() override;

    //! driver specific part below
    const shared_ptr<XUIntNode> &average() const {return m_average;} //
    const shared_ptr<XTouchableNode> &clearAverage() const {return m_clearAverage;}
    const shared_ptr<XBoolNode> &incrementalAverage() const {return m_incrementalAverage;}
    const shared_ptr<XBoolNode> &autoGain() const {return m_autoGain;}
    const shared_ptr<XUIntNode> &filterIndexR() const {return m_filterIndexR;}
    const shared_ptr<XUIntNode> &filterIndexG() const {return m_filterIndexG;}
    const shared_ptr<XUIntNode> &filterIndexB() const {return m_filterIndexB;}
    const shared_ptr<XDoubleNode> &colorGainR() const {return m_colorGainR;}
    const shared_ptr<XDoubleNode> &colorGainG() const {return m_colorGainG;}
    const shared_ptr<XDoubleNode> &colorGainB() const {return m_colorGainB;}
    const shared_ptr<XDoubleNode> &gainForDisp() const {return m_gainForDisp;}

    const shared_ptr<X2DImage> &rgbImage() const {return m_rgbImage;}

    struct Payload : public XSecondaryDriver::Payload {
        const std::vector<double> &intensities(unsigned int i) const {
            assert(i < 3); return m_intensities[i];
        }
        double raw(unsigned int idx_in_seq, unsigned int i) const {
            double pl__ = intensities(idx_in_seq)[i];
            return pl__;
        }
        unsigned int numSamples() const {return m_intensities[0].size();}
        double gainForDisp() const {return m_gainForDisp;}
        unsigned int width() const {return m_width;}
        unsigned int height() const {return m_height;}
    protected:
        friend class XImageProcessor;
        double m_gainForDisp;
        unsigned int m_accumulated[3];
        double m_colorGains[3];
        local_shared_ptr<std::vector<uint32_t>> m_summedCounts[4];//MW off and on.
        double m_coefficients[3];
        std::vector<double> m_intensities[3];
        XTime m_timeClearRequested = {};
        unsigned int m_width, m_height;
        unsigned int currentIndex() const {
            for(unsigned int i = 1; i < 3; ++i) {
                if(m_accumulated[i] < m_accumulated[0])
                    return i;
            }
            return 0;
        }
        unsigned int m_filterIndice[3];
        unsigned int m_indiceForRGB[3];
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
    const shared_ptr<XItemNode<XDriverList, XFilterWheel> > &filterWheel() const {return m_filterWheel;}

    virtual void analyzeIntensities(Transaction &tr) {};
private:
    const shared_ptr<XItemNode<XDriverList, XDigitalCamera> > m_camera;
    const shared_ptr<XItemNode<XDriverList, XFilterWheel> > m_filterWheel;
    const shared_ptr<XUIntNode> m_average;
    const shared_ptr<XTouchableNode> m_clearAverage;
    const shared_ptr<XBoolNode> m_autoGain;
    const shared_ptr<XBoolNode> m_incrementalAverage;
    const shared_ptr<XUIntNode> m_filterIndexR, m_filterIndexG, m_filterIndexB;
    const shared_ptr<XDoubleNode> m_colorGainR, m_colorGainG, m_colorGainB;
    const shared_ptr<XDoubleNode> m_gainForDisp;

    const qshared_ptr<FrmImageProcessor> m_form;
    const shared_ptr<X2DImage> m_rgbImage;

    shared_ptr<Listener> m_lsnOnClearAverageTouched, m_lsnOnCondChanged;

    std::deque<xqcon_ptr> m_conUIs;

    void onClearAverageTouched(const Snapshot &shot, XTouchableNode *);
    void onCondChanged(const Snapshot &shot, XValueNodeBase *);

    constexpr static unsigned int NumSummedCountsPool = 4;
    atomic_shared_ptr<std::vector<uint32_t>> m_summedCountsPool[NumSummedCountsPool];
    local_shared_ptr<std::vector<uint32_t>> summedCountsFromPool(int imagebytes);
};

//---------------------------------------------------------------------------

#endif
