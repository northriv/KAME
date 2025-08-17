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
#ifndef ODMR2DANALYSIS_H
#define ODMR2DANALYSIS_H


#include "graph.h"
#include "graphwidget.h"
#include "xwavengraph.h"
#include "secondarydriver.h"
#include "xnodeconnector.h"
#include "digitalcamera.h"

#include <QPushButton>
#include <QComboBox>
#include <QCheckBox>

class X2DImage;
class XODMRFSpectrum;
class QMainWindow;
class Ui_FrmODMR2DAnalysis;
typedef QForm<QMainWindow, Ui_FrmODMR2DAnalysis> FrmODMR2DAnalysis;


class XWaveNGraph;
class XXYPlot;

class XODMR2DAnalysis : public XSecondaryDriver {
public:
    XODMR2DAnalysis(const char *name, bool runtime,
        Transaction &tr_meas, const shared_ptr<XMeasure> &meas);
    //! ususally nothing to do
    virtual ~XODMR2DAnalysis() {}

    //! Shows all forms belonging to driver
    virtual void showForms() override;
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
public:
    //! driver specific part below
    struct Payload : public XSecondaryDriver::Payload {
        // unsigned int numSamples() const {return m_summedCounts.size();}
        // double minValue() const {return m_minValue;}
        double dfreq() const {return m_dfreq;}
        unsigned int width() const {return m_width;}
        unsigned int height() const {return m_height;}
        bool secondMoment() const {return m_secondMoment;}
    private:
        friend class XODMR2DAnalysis;

        //! 0x8000*PLon/PLoff, freq(unit of df)*0x8000*on/off, (freq(unit of df) - fmid)^2*0x8000*on/off
        local_shared_ptr<std::vector<uint32_t>> m_summedCounts[3];
        //! avg counts, sum freq(unit of df), sum (freq(unit of df) - fmid)^2.
        uint64_t m_accumulated[3];
        double m_coefficients[3], m_offsets[3];
        double m_freq_min; //fmin [MHz]
        bool m_secondMoment; //false during CoG analysis.
        XTime m_timeClearRequested = {};
        unsigned int m_width, m_height;
        shared_ptr<QImage> m_qimage;
        double m_dfreq; //fmin [MHz]
        double m_freq_mid; //for second moment calc. [MHz]
    };


    //! driver specific part below
    const shared_ptr<XComboNode> &regionSelection() const {return m_regionSelection;}
    const shared_ptr<XComboNode> &analysisMethod() const {return m_analysisMethod;}

    const shared_ptr<XUIntNode> &average() const {return m_average;} //
    const shared_ptr<XTouchableNode> &clearAverage() const {return m_clearAverage;}
    const shared_ptr<XBoolNode> &incrementalAverage() const {return m_incrementalAverage;}

    const shared_ptr<XBoolNode> &autoMinMaxForColorMap() const {return m_autoMinMaxForColorMap;}
    const shared_ptr<XDoubleNode> &minForColorMap() const {return m_minForColorMap;}
    const shared_ptr<XDoubleNode> &maxForColorMap() const {return m_maxForColorMap;}
    const shared_ptr<XComboNode> &colorMapMethod() const {return m_colorMapMethod;}

    const shared_ptr<X2DImage> &processedImage() const {return m_processedImage;}

    const shared_ptr<XItemNode<XDriverList, XODMRFSpectrum>> &odmrFSpectrum() const {return m_odmrFSpectrum;}
protected:
    shared_ptr<Listener> m_lsnOnClearAverageTouched, m_lsnOnCondChanged;

    const qshared_ptr<FrmODMR2DAnalysis> m_form;

    const shared_ptr<XItemNode<XDriverList, XODMRFSpectrum> > m_odmrFSpectrum;
    const shared_ptr<XComboNode> m_regionSelection;
    const shared_ptr<XComboNode> m_analysisMethod;
    const shared_ptr<XUIntNode> m_average;
    const shared_ptr<XTouchableNode> m_clearAverage;
    const shared_ptr<XBoolNode> m_incrementalAverage;
    const shared_ptr<XBoolNode> m_autoMinMaxForColorMap;
    const shared_ptr<XDoubleNode> m_minForColorMap;
    const shared_ptr<XDoubleNode> m_maxForColorMap;
    const shared_ptr<XComboNode> m_colorMapMethod;

    const shared_ptr<X2DImage> m_processedImage;

private:
    void onClearAverageTouched(const Snapshot &shot, XTouchableNode *);
    void onCondChanged(const Snapshot &shot, XValueNodeBase *);

    std::deque<xqcon_ptr> m_conUIs;

    ImageSpacePoolAllocator<5> m_pool;
};

#endif
