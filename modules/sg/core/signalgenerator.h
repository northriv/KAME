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
#ifndef signalgeneratorH
#define signalgeneratorH

#include "primarydriverwiththread.h"
#include "xnodeconnector.h"

class XScalarEntry;

class Ui_FrmSG;
typedef QForm<QMainWindow, Ui_FrmSG> FrmSG;

class DECLSPEC_SHARED XSG : public XPrimaryDriverWithThread {
public:
	XSG(const char *name, bool runtime,
		Transaction &tr_meas, const shared_ptr<XMeasure> &meas);
	//! usually nothing to do
	virtual ~XSG() {}
	//! show all forms belonging to driver
	virtual void showForms();

	struct Payload : public XPrimaryDriver::Payload {
		double freq() const {return m_freq;}
	private:
		friend class XSG;
		double m_freq;
	};
protected:
	//! This function will be called when raw data are written.
	//! Implement this function to convert the raw data to the record (Payload).
	//! \sa analyze()
    virtual void analyzeRaw(RawDataReader &reader, Transaction &tr);
	//! This function is called after committing XPrimaryDriver::analyzeRaw() or XSecondaryDriver::analyze().
	//! This might be called even if the record is invalid (time() == false).
	virtual void visualize(const Snapshot &shot);

    //! driver specific part below
    const shared_ptr<XScalarEntry> &entryFreq() const {return m_entryFreq;}

public:
	//! driver specific part below
	const shared_ptr<XBoolNode> &rfON() const {return m_rfON;} //!< Activate Output
	const shared_ptr<XDoubleNode> &freq() const {return m_freq;} //!< freq [MHz]
	const shared_ptr<XDoubleNode> &oLevel() const {return m_oLevel;} //!< Output Level [dBm]
	const shared_ptr<XBoolNode> &fmON() const {return m_fmON;} //!< Activate FM
	const shared_ptr<XBoolNode> &amON() const {return m_amON;} //!< Activate AM
    const shared_ptr<XDoubleNode> &amDepth() const {return m_amDepth;} //!< [%]
    const shared_ptr<XDoubleNode> &fmDev() const {return m_fmDev;} //!< [MHz]
    const shared_ptr<XDoubleNode> &amIntSrcFreq() const {return m_amIntSrcFreq;} //!< freq [kHz]
    const shared_ptr<XDoubleNode> &fmIntSrcFreq() const {return m_fmIntSrcFreq;} //!< freq [kHz]
    const shared_ptr<XComboNode> &sweepMode() const {return m_sweepMode;}
    const shared_ptr<XDoubleNode> &sweepFreqStart() const {return m_sweepFreqStart;} //!< [MHz]
    const shared_ptr<XDoubleNode> &sweepFreqStop() const {return m_sweepFreqStop;} //!< [MHz]
    const shared_ptr<XDoubleNode> &sweepAmplStart() const {return m_sweepAmplStart;} //!< [dB]
    const shared_ptr<XDoubleNode> &sweepAmplStop() const {return m_sweepAmplStop;} //!< [dB]
    const shared_ptr<XDoubleNode> &sweepDwellTime() const {return m_sweepDwellTime;} //!< [s]
    const shared_ptr<XUIntNode> &sweepPoints() const {return m_sweepPoints;}
protected:
    virtual double getFreq() = 0; //!< [MHz]
	virtual void changeFreq(double mhz) = 0;
	virtual void onRFONChanged(const Snapshot &shot, XValueNodeBase *) = 0;
	virtual void onOLevelChanged(const Snapshot &shot, XValueNodeBase *) = 0;
	virtual void onFMONChanged(const Snapshot &shot, XValueNodeBase *) = 0;
	virtual void onAMONChanged(const Snapshot &shot, XValueNodeBase *) = 0;
    virtual void onFreqChanged(const Snapshot &shot, XValueNodeBase *);
    virtual void onAMDepthChanged(const Snapshot &shot, XValueNodeBase *) = 0;
    virtual void onFMDevChanged(const Snapshot &shot, XValueNodeBase *) = 0;
    virtual void onAMIntSrcFreqChanged(const Snapshot &shot, XValueNodeBase *) = 0;
    virtual void onFMIntSrcFreqChanged(const Snapshot &shot, XValueNodeBase *) = 0;
    virtual void onSweepCondChanged(const Snapshot &shot, XValueNodeBase *) = 0;
private:
    const shared_ptr<XScalarEntry> m_entryFreq;

	const shared_ptr<XBoolNode> m_rfON;
	const shared_ptr<XDoubleNode> m_freq;
	const shared_ptr<XDoubleNode> m_oLevel;
	const shared_ptr<XBoolNode> m_fmON;
	const shared_ptr<XBoolNode> m_amON;
    const shared_ptr<XDoubleNode> m_amDepth, m_fmDev;
    const shared_ptr<XDoubleNode> m_amIntSrcFreq, m_fmIntSrcFreq;
    const shared_ptr<XComboNode> m_sweepMode;
    const shared_ptr<XDoubleNode> m_sweepFreqStart, m_sweepFreqStop;
    const shared_ptr<XDoubleNode> m_sweepAmplStart, m_sweepAmplStop;
    const shared_ptr<XDoubleNode> m_sweepDwellTime;
    const shared_ptr<XUIntNode> m_sweepPoints;

    void *execute(const atomic<bool> &);

    std::deque<xqcon_ptr> m_conUIs;
    shared_ptr<Listener> m_lsnRFON, m_lsnFreq, m_lsnOLevel,
        m_lsnFMON, m_lsnAMON, m_lsnAMDepth, m_lsnFMDev, m_lsnAMIntSrcFreq, m_lsnFMIntSrcFreq,
        m_lsnSweepCond;
  
	const qshared_ptr<FrmSG> m_form;
};//---------------------------------------------------------------------------
#endif
