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
#ifndef ODMRFSPECTRUM_H
#define ODMRFSPECTRUM_H


#include "graph.h"
#include "graphwidget.h"
#include "xwavengraph.h"
#include "secondarydriver.h"
#include "xnodeconnector.h"

#include <QPushButton>
#include <QComboBox>
#include <QCheckBox>

class XODMRImaging;
class XSG;
class QMainWindow;
class Ui_FrmODMRFSpectrum;
typedef QForm<QMainWindow, Ui_FrmODMRFSpectrum> FrmODMRFSpectrum;


class XWaveNGraph;
class XXYPlot;

class XODMRFSpectrum : public XSecondaryDriver {
public:
    XODMRFSpectrum(const char *name, bool runtime,
        Transaction &tr_meas, const shared_ptr<XMeasure> &meas);
    //! ususally nothing to do
    virtual ~XODMRFSpectrum() {}

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
        const std::vector<double> &wave(unsigned int ch) const {return data.at(ch).m_wave;}
        //! Averaged weights.
        const std::vector<double> &weights(unsigned int ch) const {return data.at(ch).m_weights;}
        //! Resolution [Hz].
        unsigned int numChannels() const {return data.size();}
        double res() const {return m_res;}
        //! Value of the first point [Hz].
        double min() const {return m_min;}

        double lastFreqAcquiredInHz() const {return m_lastFreqAcquired * 1e6;} //Hz
    private:
        friend class XODMRFSpectrum;

        double m_res, m_min;

        struct ChData {
            std::vector<double> m_wave;
            std::vector<double> m_weights;
            std::deque<double> m_accum;
            std::deque<double> m_accum_weights;
        };
        std::vector<ChData> data;

        double m_lastFreqAcquired; //!< to avoid inifite averaging after a sweep.
        XTime m_timeClearRequested;
    };


    //! Clears stored points.
    const shared_ptr<XTouchableNode> &clear() const {return m_clear;}

    //! driver specific part below
    const shared_ptr<XItemNode<XDriverList, XSG> > &sg1() const {return m_sg1;}
    const shared_ptr<XItemNode<XDriverList, XODMRImaging> > &odmr() const {return m_odmr;}

    //! [MHz]
    const shared_ptr<XDoubleNode> &centerFreq() const {return m_centerFreq;}
    //! [kHz]
    const shared_ptr<XDoubleNode> &freqSpan() const {return m_freqSpan;}
    //! [kHz]
    const shared_ptr<XDoubleNode> &freqStep() const {return m_freqStep;}
    const shared_ptr<XBoolNode> &active() const {return m_active;}
    const shared_ptr<XBoolNode> &repeatedly() const {return m_repeatedly;}
    const shared_ptr<XBoolNode> &altUpdateSubRegion() const {return m_altUpdateSubRegion;}
    //! [MHz]
    const shared_ptr<XDoubleNode> &subRegionMinFreq() const {return m_subRegionMinFreq;}
    const shared_ptr<XDoubleNode> &subRegionMaxFreq() const {return m_subRegionMaxFreq;}
protected:
    shared_ptr<Listener> m_lsnOnClear, m_lsnOnCondChanged;

    const shared_ptr<XTouchableNode> m_clear;

    std::deque<xqcon_ptr> m_conBaseUIs;

    void onCondChanged(const Snapshot &shot, XValueNodeBase *);

    atomic<int> m_isInstrumControlRequested;
protected:
    const qshared_ptr<FrmODMRFSpectrum> m_form;

    const shared_ptr<XWaveNGraph> m_spectrum;
    void onClear(const Snapshot &shot, XTouchableNode *);

private:
    const shared_ptr<XItemNode<XDriverList, XSG> > m_sg1;
    const shared_ptr<XItemNode<XDriverList, XODMRImaging> > m_odmr;

    const shared_ptr<XDoubleNode> m_centerFreq;
    const shared_ptr<XDoubleNode> m_freqSpan;
    const shared_ptr<XDoubleNode> m_freqStep;
    const shared_ptr<XBoolNode> m_active;
    const shared_ptr<XBoolNode> m_repeatedly;
    const shared_ptr<XBoolNode> m_altUpdateSubRegion;
    const shared_ptr<XDoubleNode> m_subRegionMinFreq, m_subRegionMaxFreq;

    shared_ptr<Listener> m_lsnOnActiveChanged;

    bool setupGraph(Transaction &tr);

    void rearrangeInstrum(const Snapshot &shot_this);

    std::deque<xqcon_ptr> m_conUIs;

    void onActiveChanged(const Snapshot &shot, XValueNodeBase *);
    void onTuningChanged(const Snapshot &shot, XValueNodeBase *); //!< receives signals from AutoLCTuner.

    double m_lastFreqOutsideSubRegion;
};

#endif
