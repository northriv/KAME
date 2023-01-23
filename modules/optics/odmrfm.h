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
#ifndef odmrfmH
#define odmrfmH
//---------------------------------------------------------------------------
#include "secondarydriver.h"
#include "lockinamp.h"
#include "signalgenerator.h"

#include <complex>

class Ui_FrmODMRFM;
typedef QForm<QMainWindow, Ui_FrmODMRFM> FrmODMRFM;

class XODMRFMControl : public XSecondaryDriver {
public:
    XODMRFMControl(const char *name, bool runtime,
		Transaction &tr_meas, const shared_ptr<XMeasure> &meas);
    virtual ~XODMRFMControl();
  
	//! Shows all forms belonging to driver
    virtual void showForms() override;
protected:

	//! This function is called when a connected driver emit a signal
	virtual void analyze(Transaction &tr, const Snapshot &shot_emitter,
        const Snapshot &shot_others, XDriver *emitter) override;
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
        //! [MHz]
        double freq() const {return m_freq;}
        //! [MHz]
        double freq_err() const {return m_freq_err;}
        //! [T]
        double tesla() const {return m_tesla;}
        //! [T]
        double fmIntens() const {return m_fmIntens;}
        //! [rad]
        double phase_err() const {return m_phase_err;}
    private:
        friend class XODMRFMControl;
        double m_freq, m_freq_err, m_tesla, m_fmIntens, m_phase_err;
        std::complex<double> m_accum = {};
        std::complex<double> m_accum_zsq = {};
        unsigned int m_accumCounts = 0;
    };
  
    const shared_ptr<XScalarEntry> &entryFreq() const {return m_entryFreq;}
    const shared_ptr<XScalarEntry> &entryTesla() const {return m_entryTesla;}
    const shared_ptr<XScalarEntry> &entryTeslaErr() const {return m_entryTeslaErr;}
    const shared_ptr<XScalarEntry> &entryFMIntens() const {return m_entryFMIntens;}

    const shared_ptr<XItemNode<XDriverList, XSG> > &sg() const {return m_sg;}
    const shared_ptr<XItemNode<XDriverList, XLIA> > &lia() const {return m_lia;}

    //! gyromagnetic ratio [MHz/T]
    const shared_ptr<XDoubleNode> &gamma2pi() const {return m_gamma2pi;}
    //! # of LIA readings before changing SG frequency
    const shared_ptr<XUIntNode> &numReadings() const {return m_numReadings;}
    //! required smallness of phase error to change SG frequency [deg.]
    const shared_ptr<XDoubleNode> &PhaseErrWithin() const {return m_PhaseErrWithin;}
    //! LIA phase when peak is at the center of FM [deg]
    const shared_ptr<XDoubleNode> &fmPhaseOrigin() const {return m_fmPhaseOrigin;}
    const shared_ptr<XBoolNode> &ctrlSG() const {return m_ctrlSG;}

private:
    const shared_ptr<XScalarEntry> m_entryFreq;
    const shared_ptr<XScalarEntry> m_entryTesla;
    const shared_ptr<XScalarEntry> m_entryTeslaErr;
    const shared_ptr<XScalarEntry> m_entryFMIntens;

    const shared_ptr<XItemNode<XDriverList, XSG> > m_sg;
    const shared_ptr<XItemNode<XDriverList, XLIA> > m_lia;

    const shared_ptr<XDoubleNode> m_gamma2pi;
    const shared_ptr<XDoubleNode> m_PhaseErrWithin, m_fmPhaseOrigin;
    const shared_ptr<XUIntNode> m_numReadings;
    const shared_ptr<XBoolNode> m_ctrlSG;

    std::deque<xqcon_ptr> m_conUIs;
    
    const qshared_ptr<FrmODMRFM> m_form;
	const shared_ptr<XStatusPrinter> m_statusPrinter;

	void onCondChanged(const Snapshot &shot, XValueNodeBase *);
};

#endif
