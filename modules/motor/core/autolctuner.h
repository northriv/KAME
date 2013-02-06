/***************************************************************************
		Copyright (C) 2002-2013 Kentaro Kitagawa
		                   kitag@kochi-u.ac.jp

		This program is free software; you can redistribute it and/or
		modify it under the terms of the GNU Library General Public
		License as published by the Free Software Foundation; either
		version 2 of the License, or (at your option) any later version.

		You should have received a copy of the GNU Library General
		Public License and a list of authors along with this program;
		see the files COPYING and AUTHORS.
***************************************************************************/
#ifndef AUTOLCTUNER_H_
#define AUTOLCTUNER_H_
//---------------------------------------------------------------------------
#include "secondarydriver.h"
#include "motor.h"
#include "networkanalyzer.h"
//---------------------------------------------------------------------------
class Ui_FrmAutoLCTuner;
typedef QForm<QMainWindow, Ui_FrmAutoLCTuner> FrmAutoLCTuner;

/*
* Tunes the reflection at the target frequency to zero.
*/
class XAutoLCTuner : public XSecondaryDriver {
public:
	XAutoLCTuner(const char *name, bool runtime,
		Transaction &tr_meas, const shared_ptr<XMeasure> &meas);
	virtual ~XAutoLCTuner();

	//! Shows all forms belonging to driver
	virtual void showForms();
protected:

	//! This function is called when a connected driver emit a signal
	virtual void analyze(Transaction &tr, const Snapshot &shot_emitter,
		const Snapshot &shot_others,
		XDriver *emitter) throw (XRecordError&);
	//! This function is called after committing XPrimaryDriver::analyzeRaw() or XSecondaryDriver::analyze().
	//! This might be called even if the record is invalid (time() == false).
	virtual void visualize(const Snapshot &shot);
	//! Checks if the connected drivers have valid time stamps.
	//! \return true if dependency is resolved.
	//! This function must be reentrant unlike analyze().
	virtual bool checkDependency(const Snapshot &shot_this,
		const Snapshot &shot_emitter, const Snapshot &shot_others,
		XDriver *emitter) const;
public:
	const shared_ptr<XItemNode<XDriverList, XMotorDriver> > &stm1() const {return m_stm1;}
	const shared_ptr<XItemNode<XDriverList, XMotorDriver> > &stm2() const {return m_stm2;}
	const shared_ptr<XItemNode<XDriverList, XNetworkAnalyzer> > &netana() const {return m_netana;}

	/// Target frequency [MHz]
	const shared_ptr<XBoolNode> &tuning() const {return m_tuning;}
	const shared_ptr<XBoolNode> &succeeded() const {return m_succeeded;}
	const shared_ptr<XDoubleNode> &target() const {return m_target;}
	const shared_ptr<XDoubleNode> &reflection() const {return m_reflection;}
	const shared_ptr<XBoolNode> &useSTM1() const {return m_useSTM1;}
	const shared_ptr<XBoolNode> &useSTM2() const {return m_useSTM2;}
	const shared_ptr<XTouchableNode> &abortTuning() const {return m_abortTuning;}

	class Payload : public XSecondaryDriver::Payload {
	public:
		enum STAGE {STAGE_FIRST, STAGE_DCA_FIRST, STAGE_DCA_SECOND, STAGE_DCB};
		STAGE stage;

		std::complex<double> ref_f0_first, ref_f0_plus_dCa;
		std::complex<double> ref_fmin_first, ref_fmin_plus_dCa;
		double fmin_first, fmin_plus_dCa;
		double ref_total_first, ref_total_plus_dCa;
		double dCa, dCb;

		std::complex<double> dref_dCa, dref_dCb;
		double dfmin_dCa, dfmin_dCb;

		std::vector<std::complex<double> > trace_prv;
		double ref_sigma, fmin_err;
		double stm1, stm2;
		int sign_of_prev_dfmin;
		int iteration_count;
		double stm1_original, stm2_original;
		std::complex<double> ref_f0_original;
		enum MODE {TUNE_MINIMIZING, TUNE_APPROACHING, TUNE_FINETUNE};
		MODE mode;
		bool isSTMChanged;
	};
private:
	const shared_ptr<XItemNode<XDriverList, XMotorDriver> > m_stm1, m_stm2;
	const shared_ptr<XItemNode<XDriverList, XNetworkAnalyzer> > m_netana;

	const shared_ptr<XBoolNode> m_tuning;
	const shared_ptr<XBoolNode> m_succeeded;
	const shared_ptr<XDoubleNode> m_target;
	const shared_ptr<XDoubleNode> m_reflection;
	const shared_ptr<XBoolNode> m_useSTM1, m_useSTM2;
	const shared_ptr<XTouchableNode> m_abortTuning;

	xqcon_ptr m_conTarget, m_conReflection,
		m_conSTM1, m_conSTM2, m_conNetAna, m_conAbortTuning,
		m_conTuning, m_conSucceeded, m_conUseSTM1, m_conUseSTM2;

	shared_ptr<XListener> m_lsnOnTargetChanged, m_lsnOnAbortTouched;

	const qshared_ptr<FrmAutoLCTuner> m_form;

	void onTargetChanged(const Snapshot &shot, XValueNodeBase *);
	void onAbortTuningTouched(const Snapshot &shot, XTouchableNode *);

	static const double TUNE_DROT_MINIMIZING = 10.0, TUNE_DROT_APPROACH = 5.0,
		TUNE_DROT_FINETUNE = 2.0, TUNE_DROT_ABORT = 360.0; //[deg.]
	static const double TUNE_TRUST_MINIMIZING = 1440.0, TUNE_TRUST_APPROACH = 720.0, TUNE_TRUST_FINETUNE = 180.0; //[deg.]
	static const double TUNE_APPROACH_START = 0.5; //-3dB@minimum
	static const double TUNE_FINETUNE_START = 0.25; //-6dB@f0
	static const double TUNE_DROT_REQUIRED_N_SIGMA = 3.0;

	void determineNextC(double &deltaC1, double &deltaC2, double &err,
		double x, double x_err,
		double y, double y_err,
		double dxdC1, double dxdC2,
		double dydC1, double dydC2,
		const char *msg);
	void abortTuningFromAnalyze(Transaction &tr, std::complex<double> reff0);
};



#endif /* AUTOLCTUNER_H_ */
