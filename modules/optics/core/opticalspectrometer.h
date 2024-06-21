/***************************************************************************
        Copyright (C) 2002-2022 Kentaro Kitagawa
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

#ifndef opticalspectrumH
#define opticalspectrumH
//---------------------------------------------------------------------------
#include "primarydriverwiththread.h"
#include "xnodeconnector.h"

class XScalarEntry;
class QMainWindow;
class Ui_FrmOpticalSpectrometer;
typedef QForm<QMainWindow, Ui_FrmOpticalSpectrometer> FrmOpticalSpectrometer;

class XGraph;
class XWaveNGraph;
class XXYPlot;

class XQGraph2DMathToolConnector;
class XSpectral1DMathToolList;

//! Base class for optical/multi-channel spectrometer.
class DECLSPEC_SHARED XOpticalSpectrometer : public XPrimaryDriverWithThread {
public:
    XOpticalSpectrometer(const char *name, bool runtime,
		Transaction &tr_meas, const shared_ptr<XMeasure> &meas);
	//! usually nothing to do.
    virtual ~XOpticalSpectrometer() {}
	//! Shows all forms belonging to driver.
    virtual void showForms() override;

    const shared_ptr<XGraph> &graph() const {return m_graph;}

	struct Payload : public XPrimaryDriver::Payload {
        double integrationTime() const {return m_integrationTime;} //! [s]
        const double *counts() const {return &m_counts.at(0);}
        bool isCountsValid() const {return (m_counts.size() == accumLength());}
        const double *darkCounts() const {return &m_darkCounts.at(0);}
        bool isDarkValid() const {return (m_darkCounts.size() == accumLength());}
        const double *accumCounts() const {return &m_accumCounts.at(0);}
        unsigned int accumLength() const {return std::min(m_waveLengths.size(), m_accumCounts.size());}
        const double *waveLengths() const {return &m_waveLengths.at(0);}

        std::vector<double> &waveLengths_() {return m_waveLengths;}
        std::vector<double> &counts_() {return m_counts;}
        std::vector<double> &accumCounts_() {return m_accumCounts;}
        std::vector<double> &darkCounts_() {return m_darkCounts;}
        double m_integrationTime;
        unsigned int m_accumulated;
        double m_electric_dark;
        std::vector<double> m_nonLinCorrCoeffs;
        std::deque<std::pair<double, double>> &markers() {return m_markers;}
    private:
        friend class XOpticalSpectrometer;
        std::vector<double> m_counts, m_accumCounts, m_darkCounts, m_waveLengths;
        std::deque<std::pair<double, double>> m_markers;
    };
protected:

	//! This function is called after committing XPrimaryDriver::analyzeRaw() or XSecondaryDriver::analyze().
	//! This might be called even if the record is invalid (time() == false).
    virtual void visualize(const Snapshot &shot) override;
  
	//! driver specific part below
	const shared_ptr<XScalarEntry> &marker1X() const {return m_marker1X;}
	const shared_ptr<XScalarEntry> &marker1Y() const {return m_marker1Y;}
    const shared_ptr<XDoubleNode> &startWavelen() const {return m_startWavelen;}
    const shared_ptr<XDoubleNode> &stopWavelen() const {return m_stopWavelen;}
    const shared_ptr<XDoubleNode> &integrationTime() const {return m_integrationTime;}
	const shared_ptr<XUIntNode> &average() const {return m_average;}
    const shared_ptr<XTouchableNode> &storeDark() const {return m_storeDark;}
    const shared_ptr<XBoolNode> &subtractDark() const {return m_subtractDark;}
protected:
    virtual void onStartWavelenChanged(const Snapshot &shot, XValueNodeBase *) = 0;
    virtual void onStopWavelenChanged(const Snapshot &shot, XValueNodeBase *) = 0;
    virtual void onAverageChanged(const Snapshot &shot, XValueNodeBase *) = 0;
    virtual void onIntegrationTimeChanged(const Snapshot &shot, XValueNodeBase *) = 0;
    void onStoreDarkTouched(const Snapshot &shot, XTouchableNode *);

    //! This function will be called when raw data are written.
    //! Implement this function to convert the raw data to the record (Payload).
    //! \sa analyze()
    virtual void analyzeRaw(RawDataReader &reader, Transaction &tr) override;
    virtual void convertRawAndAccum(RawDataReader &reader, Transaction &tr) = 0;

    virtual void acquireSpectrum(shared_ptr<RawData> &) = 0;

    const shared_ptr<XScalarEntry> m_marker1X;
    const shared_ptr<XScalarEntry> m_marker1Y;
private:
	const shared_ptr<XWaveNGraph> &waveForm() const {return m_waveForm;}
    const shared_ptr<XDoubleNode> m_startWavelen;
    const shared_ptr<XDoubleNode> m_stopWavelen;
    const shared_ptr<XDoubleNode> m_integrationTime;
	const shared_ptr<XUIntNode> m_average;
    const shared_ptr<XTouchableNode> m_storeDark;
    const shared_ptr<XBoolNode> m_subtractDark;

    const qshared_ptr<FrmOpticalSpectrometer> m_form;
	const shared_ptr<XWaveNGraph> m_waveForm;

    shared_ptr<XSpectral1DMathToolList> m_spectralToolLists;

    shared_ptr<Listener> m_lsnOnStartWavelenChanged;
    shared_ptr<Listener> m_lsnOnStopWavelenChanged;
	shared_ptr<Listener> m_lsnOnAverageChanged;
    shared_ptr<Listener> m_lsnOnIntegrationTimeChanged;
    shared_ptr<Listener> m_lsnOnStoreDarkTouched;

    std::deque<xqcon_ptr> m_conUIs;
    std::deque<shared_ptr<XQGraph2DMathToolConnector>> m_conTools;

	shared_ptr<XGraph> m_graph;
	shared_ptr<XXYPlot> m_markerPlot;
	
    bool m_storeDarkInvoked;

    void *execute(const atomic<bool> &) override;
};

//---------------------------------------------------------------------------

#endif
