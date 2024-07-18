/***************************************************************************
        Copyright (C) 2002-2023 Kentaro Kitagawa
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

#ifndef odmrImagingH
#define odmrImagingH
//---------------------------------------------------------------------------
#include "secondarydriver.h"
#include "xnodeconnector.h"

class XDigitalCamera;
class XFilterWheel;
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
    const shared_ptr<XUIntNode> &precedingSkips() const {return m_precedingSkips;} //skips some OFF/ON sequences before averaging
    const shared_ptr<XTouchableNode> &clearAverage() const {return m_clearAverage;}
    const shared_ptr<XBoolNode> &incrementalAverage() const {return m_incrementalAverage;}
    const shared_ptr<XBoolNode> &autoGainForDisp() const {return m_autoGainForDisp;}
    const shared_ptr<XUIntNode> &filterIndex() const {return m_filterIndex;}
    const shared_ptr<XDoubleNode> &gainForDisp() const {return m_gainForDisp;}
    const shared_ptr<XDoubleNode> &gamma() const {return m_gamma;}
    const shared_ptr<XDoubleNode> &minDPLoPLForDisp() const {return m_minDPLoPLForDisp;}//!< [%]
    const shared_ptr<XDoubleNode> &maxDPLoPLForDisp() const {return m_maxDPLoPLForDisp;}//!< [%]
    const shared_ptr<XComboNode> &dispMethod() const {return m_dispMethod;}
    const shared_ptr<XUIntNode> &refIntensFrames() const {return m_refIntensFrames;}
    const shared_ptr<XComboNode> &sequence() const {return m_sequence;}

    const shared_ptr<X2DImage> &processedImage() const {return m_processedImage;}

    enum class Sequence {OFF_ON = 2, OFF_OFF_ON = 3, OFF_OFF_OFF_ON = 4};
    struct Payload : public XSecondaryDriver::Payload {
        const std::vector<double> &sampleIntensities(unsigned int i) const {
            assert(i < sequenceLength()); return m_sampleIntensities[i];
        }
        const std::vector<double> &sampleIntensitiesCorrected(unsigned int i) const {
            assert(i < sequenceLength()); return m_sampleIntensitiesCorrected[i];
        }
        const std::vector<double> &referenceIntensities(unsigned int i) const {
            assert(i < sequenceLength()); return m_referenceIntensities[i];
        }
        double plRaw(unsigned int idx_in_seq, unsigned int i) const {
            double pl__ = sampleIntensities(idx_in_seq)[i];
            return pl__;
        }
        double plCorr(unsigned int idx_in_seq, unsigned int i) const {
            double pl__ = sampleIntensitiesCorrected(idx_in_seq)[i];
            return pl__;
        }
        double pl0(unsigned int i) const {
            return plRaw(sequenceLength() - 2, i);
        }
        double dPL(unsigned int i) const {
            double pl_off = plCorr(sequenceLength() - 2, i);
//            pl_off = 0;
//            for(unsigned int j = 0; j < sequenceLength() - 1; ++j) {
//                pl_off += plCorr(j, i);
//            }
//            pl_off /= sequenceLength() - 1;
            double pl_on = plCorr(sequenceLength() - 1, i);
            return pl_on - pl_off;
        }
        double dPLoPL(unsigned int i) const {
            return dPL(i) / pl0(i);
        }
        unsigned int numSamples() const {return m_sampleIntensities[0].size();}
        double gainForDisp() const {return m_gainForDisp;}
        unsigned int width() const {return m_width;}
        unsigned int height() const {return m_height;}
        unsigned int sequenceLength() const {return (unsigned int)m_sequence;}
        Sequence sequence() const {return m_sequence;}
    protected:
        friend class XODMRImaging;
        Sequence m_sequence;
        double m_gainForDisp;
        double m_gamma;
        unsigned int m_accumulated[4];
        unsigned int m_skippedFrames; //sa precedingSkips()
        local_shared_ptr<std::vector<uint32_t>> m_summedCounts[4];//MW off and on.
        double m_coefficients[4];
        std::vector<double> m_sampleIntensities[4];
        std::vector<double> m_sampleIntensitiesCorrected[4];
        std::vector<double> m_referenceIntensities[4];
        XTime m_timeClearRequested = {};
        unsigned int m_width, m_height;
        unsigned int currentIndex() const {
            for(unsigned int i = 1; i < sequenceLength(); ++i) {
                if(m_accumulated[i] < m_accumulated[0])
                    return i;
            }
            return 0;
        }
        shared_ptr<QImage> m_qimage;
        //maps existing MathTools to ScalarEntries, be synced inside analyze()
        std::map<XNode*, shared_ptr<XScalarEntry>> m_samplePLEntries, m_sampleDPLoPLEntries;
        //to be released from entrylist.
        std::deque<shared_ptr<XScalarEntry>> m_releasedEntries;
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
    const shared_ptr<XUIntNode> m_precedingSkips;
    const shared_ptr<XTouchableNode> m_clearAverage;
    const shared_ptr<XBoolNode> m_autoGainForDisp;
    const shared_ptr<XBoolNode> m_incrementalAverage;
    const shared_ptr<XUIntNode> m_filterIndex;
    const shared_ptr<XDoubleNode> m_gainForDisp;
    const shared_ptr<XDoubleNode> m_gamma;
    const shared_ptr<XDoubleNode> m_minDPLoPLForDisp; //!< [%]
    const shared_ptr<XDoubleNode> m_maxDPLoPLForDisp; //!< [%]
    const shared_ptr<XComboNode> m_dispMethod;
    const shared_ptr<XUIntNode> m_refIntensFrames;
    const shared_ptr<XComboNode> m_sequence;

    const qshared_ptr<FrmODMRImaging> m_form;
    const shared_ptr<X2DImage> m_processedImage;
    std::deque<shared_ptr<XGraph2DMathToolList>> m_sampleToolLists, m_referenceToolLists, m_darkToolLists; //PL for MW off and on.

    shared_ptr<Listener> m_lsnOnClearAverageTouched, m_lsnOnCondChanged;

    weak_ptr<XScalarEntryList> m_entries;

    std::deque<xqcon_ptr> m_conUIs;

    std::deque<shared_ptr<XQGraph2DMathToolConnector>> m_conTools;

    void onClearAverageTouched(const Snapshot &shot, XTouchableNode *);
    void onCondChanged(const Snapshot &shot, XValueNodeBase *);

    constexpr static unsigned int NumSummedCountsPool = 5;
    atomic_shared_ptr<std::vector<uint32_t>> m_summedCountsPool[NumSummedCountsPool];
    local_shared_ptr<std::vector<uint32_t>> summedCountsFromPool(int imagebytes);
};

//---------------------------------------------------------------------------

#endif
