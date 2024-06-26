/***************************************************************************
		Copyright (C) 2002-2015 Kentaro Kitagawa
		                   kitag@issp.u-tokyo.ac.jp
		
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
    virtual void onPowerChanged(const Snapshot &shot, XValueNodeBase *) {}

    virtual void onCalOpenTouched(const Snapshot &shot, XTouchableNode *);
    virtual void onCalShortTouched(const Snapshot &shot, XTouchableNode *);
    virtual void onCalTermTouched(const Snapshot &shot, XTouchableNode *);
    virtual void onCalThruTouched(const Snapshot &shot, XTouchableNode *) {}

    virtual void getMarkerPos(unsigned int num, double &x, double &y);
    virtual void oneSweep();
    virtual void startContSweep();
    virtual void acquireTrace(shared_ptr<RawData> &, unsigned int ch);
    //! Converts raw to dispaly-able
    virtual void convertRaw(RawDataReader &reader, Transaction &tr);

    //! Be called just after opening interface. Call start() inside this routine appropriately.
    virtual void open();
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
    virtual void showForms() override; //!< overrides XSG::showForms()

    const shared_ptr<XDoubleNode> &rxGain() const {return m_rxGain;} //!< Receiver Gain [dB] (0 -- 95)
    const shared_ptr<XDoubleNode> &rxPhase() const {return m_rxPhase;} //!< Receiver phase [deg.]
    const shared_ptr<XDoubleNode> &rxLPFBW() const {return m_rxLPFBW;} //!< Receiver BW of LPF [kHz] (0 -- 200)
    const shared_ptr<XDoubleNode> &fwdPWR() const {return m_fwdPWR;} //!< Transmission
    const shared_ptr<XDoubleNode> &bwdPWR() const {return m_bwdPWR;} //!< Reflection
    const shared_ptr<XBoolNode> &ampWarn() const {return m_ampWarn;}
protected:
    //! Starts up your threads, connects GUI, and activates signals.
    virtual void start() override;
    //! Shuts down your threads, unconnects GUI, and deactivates signals
    //! This function may be called even if driver has already stopped.
    virtual void stop() override;

    virtual double getFreq() override; //!< [MHz]
    virtual void changeFreq(double mhz) override;
    virtual void onFreqChanged(const Snapshot &shot, XValueNodeBase *node) override {XSG::onFreqChanged(shot, node);}
    virtual void onRFONChanged(const Snapshot &shot, XValueNodeBase *) override;
    virtual void onOLevelChanged(const Snapshot &shot, XValueNodeBase *) override;
    virtual void onFMONChanged(const Snapshot &shot, XValueNodeBase *) override {}
    virtual void onAMONChanged(const Snapshot &shot, XValueNodeBase *) override {}
    virtual void onAMDepthChanged(const Snapshot &shot, XValueNodeBase *) override {}
    virtual void onFMDevChanged(const Snapshot &shot, XValueNodeBase *) override {}
    virtual void onAMIntSrcFreqChanged(const Snapshot &shot, XValueNodeBase *) override {}
    virtual void onFMIntSrcFreqChanged(const Snapshot &shot, XValueNodeBase *) override {}
    virtual void onSweepCondChanged(const Snapshot &shot, XValueNodeBase *) override {}
    //! PROT features below
    virtual void onRXGainChanged(const Snapshot &shot, XValueNodeBase *);
    virtual void onRXPhaseChanged(const Snapshot &shot, XValueNodeBase *);
    virtual void onRXLPFBWChanged(const Snapshot &shot, XValueNodeBase *);
private:
    double query(const char *cmd);

    const shared_ptr<XDoubleNode> m_rxGain, m_rxPhase, m_rxLPFBW, m_fwdPWR, m_bwdPWR;
    const shared_ptr<XBoolNode> m_ampWarn;

    shared_ptr<Listener> m_lsnRFON, m_lsnFreq, m_lsnOLevel;
    shared_ptr<Listener> m_lsnRXGain, m_lsnRXPhase, m_lsnRXLPFBW;

    std::deque<xqcon_ptr> m_conUIs;

    const qshared_ptr<FrmThamwayPROT> m_form;

    void fetchStatus(const atomic<bool>&, bool single);
    unique_ptr<XThread> m_thread;
};

//! Thamway NMR PROT series for GPIB, etc..
class XThamwayCharPROT : public XThamwayPROT<XCharInterface> {
public:
    XThamwayCharPROT(const char *name, bool runtime,
        Transaction &tr_meas, const shared_ptr<XMeasure> &meas);
    virtual ~XThamwayCharPROT() {}
};

#ifdef USE_THAMWAY_USB
    #include "thamwayusbinterface.h"
    class XThamwayMODCUSBInterface : public XThamwayFX2USBInterface {
    public:
        XThamwayMODCUSBInterface(const char *name, bool runtime, const shared_ptr<XDriver> &driver)
            : XThamwayFX2USBInterface(name, runtime, driver, 0u, "") {} //DIP-SW address should be 6 (=DEV_ADDR_PROT)
        virtual ~XThamwayMODCUSBInterface() {}
    };

    //! Thamway NMR PROT series for USB
    class XThamwayUSBPROT : public XThamwayPROT<XThamwayMODCUSBInterface> {
    public:
        XThamwayUSBPROT(const char *name, bool runtime,
            Transaction &tr_meas, const shared_ptr<XMeasure> &meas);
        virtual ~XThamwayUSBPROT() {}
    protected:
        //! Starts up your threads, connects GUI, and activates signals.
        virtual void start();
    };

#endif

#endif
