/***************************************************************************
		Copyright (C) 2002-2014 Kentaro Kitagawa
		                   kitag@kochi-u.ac.jp
		
		This program is free software; you can redistribute it and/or
		modify it under the terms of the GNU Library General Public
		License as published by the Free Software Foundation; either
		version 2 of the License, or (at your option) any later version.
		
		You should have received a copy of the GNU Library General 
		Public License and a list of authors along with this program; 
		see the files COPYING and AUTHORS.
***************************************************************************/
#ifndef thamwayprotH
#define thamwayprotH

#include "networkanalyzer.h"
#include "signalgenerator.h"
#include "chardevicedriver.h"
#include "xnodeconnector.h"

//! Thamway Impedance Analyzer T300-1049A
class XThamwayT300ImpedanceAnalyzer : public XCharDeviceDriver<XNetworkAnalyzer> {
public:
    XThamwayT300ImpedanceAnalyzer(const char *name, bool runtime,
        Transaction &tr_meas, const shared_ptr<XMeasure> &meas);
    virtual ~XThamwayT300ImpedanceAnalyzer() {}
protected:
    virtual void onStartFreqChanged(const Snapshot &shot, XValueNodeBase *);
    virtual void onStopFreqChanged(const Snapshot &shot, XValueNodeBase *);
    virtual void onAverageChanged(const Snapshot &shot, XValueNodeBase *) {}
    virtual void onPointsChanged(const Snapshot &shot, XValueNodeBase *);

    virtual void onCalOpenTouched(const Snapshot &shot, XTouchableNode *);
    virtual void onCalShortTouched(const Snapshot &shot, XTouchableNode *);
    virtual void onCalTermTouched(const Snapshot &shot, XTouchableNode *);
    virtual void onCalThruTouched(const Snapshot &shot, XTouchableNode *) {}

    virtual void getMarkerPos(unsigned int num, double &x, double &y);
    virtual void oneSweep();
    virtual void startContSweep();
    virtual void acquireTrace(shared_ptr<RawData> &, unsigned int ch);
    //! Converts raw to dispaly-able
    virtual void convertRaw(RawDataReader &reader, Transaction &tr) throw (XRecordError&);

    //! Be called just after opening interface. Call start() inside this routine appropriately.
    virtual void open() throw (XKameError &);
};

class Ui_FrmThamwayPROT;
typedef QForm<QMainWindow, Ui_FrmThamwayPROT> FrmThamwayPROT;

//! Thamway NMR PROT series
template <class tInterface>
class XThamwayPROT : public XCharDeviceDriver<XSG, tInterface> {
public:
    XThamwayPROT(const char *name, bool runtime,
        Transaction &tr_meas, const shared_ptr<XMeasure> &meas);
    virtual ~XThamwayPROT() {}

    //! show all forms belonging to driver
    virtual void showForms(); //!< overrides XSG::showForms()

    const shared_ptr<XDoubleNode> &rxGain() const {return m_rxGain;} //!< Receiver Gain [dB] (0 -- 95)
    const shared_ptr<XDoubleNode> &rxPhase() const {return m_rxPhase;} //!< Receiver phase [deg.]
    const shared_ptr<XDoubleNode> &rxLPFBW() const {return m_rxLPFBW;} //!< Receiver BW of LPF [kHz] (0 -- 200)
protected:
    //! Starts up your threads, connects GUI, and activates signals.
    virtual void start();
    //! Shuts down your threads, unconnects GUI, and deactivates signals
    //! This function may be called even if driver has already stopped.
    virtual void stop();

    virtual void changeFreq(double mhz);
    virtual void onFreqChanged(const Snapshot &shot, XValueNodeBase *node) {XSG::onFreqChanged(shot, node);}
    virtual void onRFONChanged(const Snapshot &shot, XValueNodeBase *);
    virtual void onOLevelChanged(const Snapshot &shot, XValueNodeBase *);
    virtual void onFMONChanged(const Snapshot &shot, XValueNodeBase *) {}
    virtual void onAMONChanged(const Snapshot &shot, XValueNodeBase *) {}
    //! PROT features below
    virtual void onRXGainChanged(const Snapshot &shot, XValueNodeBase *);
    virtual void onRXPhaseChanged(const Snapshot &shot, XValueNodeBase *);
    virtual void onRXLPFBWChanged(const Snapshot &shot, XValueNodeBase *);
private:
    const shared_ptr<XDoubleNode> m_rxGain, m_rxPhase, m_rxLPFBW;

    xqcon_ptr m_conRFON, m_conFreq, m_conOLevel;
    shared_ptr<XListener> m_lsnRFON, m_lsnFreq, m_lsnOLevel;

    xqcon_ptr m_conRXGain, m_conRXPhase, m_conRXLPFBW;
    shared_ptr<XListener> m_lsnRXGain, m_lsnRXPhase, m_lsnRXLPFBW;

    const qshared_ptr<FrmThamwayPROT> m_form;
};

//! Thamway NMR PROT series for GPIB, etc..
class XThamwayCharPROT : public XThamwayPROT<XCharInterface> {
public:
    XThamwayCharPROT(const char *name, bool runtime,
        Transaction &tr_meas, const shared_ptr<XMeasure> &meas);
    virtual ~XThamwayCharPROT() {}
};

#ifdef USE_EZUSB
    #include "ezusbthamway.h"
    class XThamwayMODCUSBInterface : public XWinCUSBInterface {
    public:
        XThamwayMODCUSBInterface(const char *name, bool runtime, const shared_ptr<XDriver> &driver)
            : XWinCUSBInterface(name, runtime, driver, 0x600u, "") {} //DIP-SW address should be 6
        virtual ~XThamwayMODCUSBInterface() {}
    protected:
        //! Starts up your threads, connects GUI, and activates signals.
        virtual void start();
    };

    //! Thamway NMR PROT series for USB
    class XThamwayUSBPROT : public XThamwayPROT<XThamwayMODCUSBInterface> {
    public:
        XThamwayUSBPROT(const char *name, bool runtime,
            Transaction &tr_meas, const shared_ptr<XMeasure> &meas);
        virtual ~XThamwayUSBPROT() {}
    };

#endif

#endif
