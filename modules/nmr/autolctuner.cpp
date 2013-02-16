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
//---------------------------------------------------------------------------
#include "analyzer.h"
#include "autolctuner.h"
#include "ui_autolctunerform.h"

REGISTER_TYPE(XDriverList, AutoLCTuner, "NMR LC autotuner");

static const double TUNE_DROT_MINIMIZING = 10.0, TUNE_DROT_APPROACH = 5.0,
	TUNE_DROT_FINETUNE = 2.0, TUNE_DROT_ABORT = 360.0; //[deg.]
static const double TUNE_TRUST_MINIMIZING = 1440.0, TUNE_TRUST_APPROACH = 720.0, TUNE_TRUST_FINETUNE = 360.0; //[deg.]
static const double TUNE_APPROACH_START = 0.8; //-2dB@minimum
static const double TUNE_FINETUNE_START = 0.5; //-6dB@f0
static const double TUNE_DROT_REQUIRED_N_SIGMA = 2.5;
static const double SOR_FACTOR_MAX = 0.8;
static const double SOR_FACTOR_MIN = 0.3;

//---------------------------------------------------------------------------
XAutoLCTuner::XAutoLCTuner(const char *name, bool runtime,
	Transaction &tr_meas, const shared_ptr<XMeasure> &meas) :
	XSecondaryDriver(name, runtime, ref(tr_meas), meas),
		m_stm1(create<XItemNode<XDriverList, XMotorDriver> >("STM1", false, ref(tr_meas), meas->drivers(), true)),
		m_stm2(create<XItemNode<XDriverList, XMotorDriver> >("STM2", false, ref(tr_meas), meas->drivers(), false)),
		m_netana(create<XItemNode<XDriverList, XNetworkAnalyzer> >("NetworkAnalyzer", false, ref(tr_meas), meas->drivers(), true)),
		m_tuning(create<XBoolNode>("Tuning", true)),
		m_succeeded(create<XBoolNode>("Succeeded", true)),
		m_target(create<XDoubleNode>("Target", true)),
		m_reflection(create<XDoubleNode>("Reflection", false)),
		m_useSTM1(create<XBoolNode>("UseSTM1", false)),
		m_useSTM2(create<XBoolNode>("UseSTM2", false)),
		m_abortTuning(create<XTouchableNode>("AbortTuning", true)),
		m_form(new FrmAutoLCTuner(g_pFrmMain))  {
	connect(stm1());
	connect(stm2());
	connect(netana());

	m_form->setWindowTitle(i18n("NMR LC autotuner - ") + getLabel() );

	m_conSTM1 = xqcon_create<XQComboBoxConnector>(stm1(), m_form->m_cmbSTM1, ref(tr_meas));
	m_conSTM2 = xqcon_create<XQComboBoxConnector>(stm2(), m_form->m_cmbSTM2, ref(tr_meas));
	m_conNetAna = xqcon_create<XQComboBoxConnector>(netana(), m_form->m_cmbNetAna, ref(tr_meas));
	m_conTarget = xqcon_create<XQLineEditConnector>(target(), m_form->m_edTarget);
	m_conReflection = xqcon_create<XQLineEditConnector>(reflection(), m_form->m_edReflection);
	m_conAbortTuning = xqcon_create<XQButtonConnector>(m_abortTuning, m_form->m_btnAbortTuning);
	m_conTuning = xqcon_create<XKLedConnector>(m_tuning, m_form->m_ledTuning);
	m_conSucceeded = xqcon_create<XKLedConnector>(m_succeeded, m_form->m_ledSucceeded);
	m_conUseSTM1 = xqcon_create<XQToggleButtonConnector>(m_useSTM1, m_form->m_ckbUseSTM1);
	m_conUseSTM2 = xqcon_create<XQToggleButtonConnector>(m_useSTM2, m_form->m_ckbUseSTM2);

	for(Transaction tr( *this);; ++tr) {
		tr[ *m_tuning] = false;
		tr[ *m_succeeded] = false;
		tr[ *m_reflection] = -15.0;
		tr[ *m_useSTM1] = true;
		tr[ *m_useSTM2] = true;
		m_lsnOnTargetChanged = tr[ *m_target].onValueChanged().connectWeakly(
			shared_from_this(), &XAutoLCTuner::onTargetChanged);
		m_lsnOnAbortTouched = tr[ *m_abortTuning].onTouch().connectWeakly(
			shared_from_this(), &XAutoLCTuner::onAbortTuningTouched);
		if(tr.commit())
			break;
	}
}
XAutoLCTuner::~XAutoLCTuner() {
}
void XAutoLCTuner::showForms() {
	m_form->show();
	m_form->raise();
}
void XAutoLCTuner::onTargetChanged(const Snapshot &shot, XValueNodeBase *node) {
	Snapshot shot_this( *this);
	shared_ptr<XMotorDriver> stm1__ = shot_this[ *stm1()];
	shared_ptr<XMotorDriver> stm2__ = shot_this[ *stm2()];
	const unsigned int tunebits = 0xffu;
	if(stm1__) {
		for(Transaction tr( *stm1__);; ++tr) {
			tr[ *stm1__->active()] = true; // Activate motor.
			tr[ *stm1__->auxBits()] = tunebits; //For external RF relays.
			if(tr.commit())
				break;
		}
	}
	if(stm2__) {
		for(Transaction tr( *stm2__);; ++tr) {
			tr[ *stm2__->active()] = true; // Activate motor.
			tr[ *stm2__->auxBits()] = tunebits; //For external RF relays.
			if(tr.commit())
				break;
		}
	}
	for(Transaction tr( *this);; ++tr) {
		tr[ *m_tuning] = true;
		tr[ *succeeded()] = false;
		tr[ *this].iteration_count = 0;
		tr[ *this].ref_f0_best = 1e10;
		tr[ *this].isSTMChanged = true;
		tr[ *this].sor_factor = SOR_FACTOR_MAX;
		tr[ *this].stage = Payload::STAGE_FIRST;
		if(tr.commit())
			break;
	}
}
void XAutoLCTuner::onAbortTuningTouched(const Snapshot &shot, XTouchableNode *) {
	for(Transaction tr( *this);; ++tr) {
		if( !tr[ *m_tuning])
			break;
		tr[ *m_tuning] = false;
		if(tr.commit())
			break;
	}
}
bool
XAutoLCTuner::determineNextC(double &deltaC1, double &deltaC2, double &err,
	double x, double x_err,
	double y, double y_err,
	double dxdC1, double dxdC2,
	double dydC1, double dydC2,
	const char *msg) {
	double det = dxdC1 * dydC2 - dxdC2 *dydC1;
	if(det < 1e-40)
		return false;
	double esq = (dydC2 * x_err * dydC2 * x_err + dxdC2 * y_err * dxdC2 * y_err) / det / det
		+ (dydC1 * x_err * dydC1 * x_err + dxdC1 * y_err * dxdC1 * y_err) / det / det;
	if(esq < err * err) {
		err = sqrt(esq);
		fprintf(stderr, "LCTuner: %s: c_err=%.2g\n", msg, err);
		deltaC1 = -(dydC2 * x - dxdC2 * y) / det;
		deltaC2 = -( -dydC1 * x + dxdC1 * y) / det;
		return true;
	}
	return false;
}
bool XAutoLCTuner::checkDependency(const Snapshot &shot_this,
	const Snapshot &shot_emitter, const Snapshot &shot_others,
	XDriver *emitter) const {
	const shared_ptr<XMotorDriver> stm1__ = shot_this[ *stm1()];
	const shared_ptr<XMotorDriver> stm2__ = shot_this[ *stm2()];
	const shared_ptr<XNetworkAnalyzer> na__ = shot_this[ *netana()];
	if( !na__)
		return false;
	if(stm1__ == stm2__)
		return false;
	if(emitter != na__.get())
		return false;

	return true;
}
void
XAutoLCTuner::abortTuningFromAnalyze(Transaction &tr, std::complex<double> reff0) {
	tr[ *m_tuning] = false;
	if(std::abs(reff0) > std::abs(tr[ *this].ref_f0_best)) {
		tr[ *this].isSTMChanged = true;
		tr[ *this].stm1 = tr[ *this].stm1_best;
		tr[ *this].stm2 = tr[ *this].stm2_best;
		throw XRecordError(i18n("Aborting. Out of tune, or capacitors have sticked. Back to better positions."), __FILE__, __LINE__);
	}
	throw XRecordError(i18n("Aborting. Out of tune, or capacitors have sticked."), __FILE__, __LINE__);
}
void
XAutoLCTuner::analyze(Transaction &tr, const Snapshot &shot_emitter,
	const Snapshot &shot_others,
	XDriver *emitter) throw (XRecordError&) {
	const Snapshot &shot_this(tr);
	const Snapshot &shot_na(shot_emitter);

	shared_ptr<XMotorDriver> stm1__ = shot_this[ *stm1()];
	shared_ptr<XMotorDriver> stm2__ = shot_this[ *stm2()];
	//remembers original position.
	if(stm1__)
		tr[ *this].stm1 = shot_others[ *stm1__->position()->value()];
	if(stm2__)
		tr[ *this].stm2 = shot_others[ *stm2__->position()->value()];
	if( !shot_this[ *useSTM1()]) stm1__.reset();
	if( !shot_this[ *useSTM2()]) stm2__.reset();

	if( (stm1__ && !shot_others[ *stm1__->ready()]) ||
			( stm2__  && !shot_others[ *stm2__->ready()]))
		throw XSkippedRecordError(__FILE__, __LINE__); //STM is moving. skip.
	if( shot_this[ *this].isSTMChanged) {
		tr[ *this].isSTMChanged = false;
		throw XSkippedRecordError(__FILE__, __LINE__); //the present data may involve one before STM was moved. reload.
	}
	if( !shot_this[ *tuning()]) {
		throw XSkippedRecordError(__FILE__, __LINE__);
	}

	if( !stm1__ && !stm2__) {
		tr[ *m_tuning] = false;
		throw XSkippedRecordError(__FILE__, __LINE__);
	}

	const shared_ptr<XNetworkAnalyzer> na__ = shot_this[ *netana()];

	const std::complex<double> *trace = shot_na[ *na__].trace();
	int trace_len = shot_na[ *na__].length();
	double trace_dfreq = shot_na[ *na__].freqInterval();
	double trace_start = shot_na[ *na__].startFreq();
	std::complex<double> reffmin(1e10);
	double f0 = shot_this[ *target()];
	//searches for minimum in reflection.
	double fmin = 0;
	double reftotal = 0;
	for(int i = 0; i < trace_len; ++i) {
		double z = std::abs(trace[i]);
		reftotal += z * z;
		if(std::abs(reffmin) > z) {
			reffmin = trace[i];
			fmin = trace_start + i * trace_dfreq;
		}
	}
	reftotal /= trace_len;
	//Reflection at the target frequency.
	std::complex<double> reff0;
	for(int i = 0; i < trace_len; ++i) {
		if(trace_start + i * trace_dfreq >= f0) {
			reff0 = trace[i];
			break;
		}
	}
	fprintf(stderr, "LCtuner: fmin=%.2f, reffmin=%.2f, reftotal=%.2f, reff0=%.2f\n",
			fmin, std::abs(reffmin), reftotal, std::abs(reff0));

	double tune_approach_goal = pow(10.0, 0.05 * shot_this[ *reflection()]);
	if(std::abs(reff0) < tune_approach_goal) {
		tr[ *succeeded()] = true;
		return;
	}

	Payload::STAGE stage = shot_this[ *this].stage;

	tr[ *this].iteration_count++;
	if((shot_this[ *this].iteration_count > 100) || (shot_this[ *this].sor_factor < SOR_FACTOR_MIN * 1.05)) {
		abortTuningFromAnalyze(tr, reff0);//Aborts.
	}

	if(std::abs(shot_this[ *this].ref_f0_best) > std::abs(reff0)) {
		//remembers good positions.
		tr[ *this].stm1_best = tr[ *this].stm1;
		tr[ *this].stm2_best = tr[ *this].stm2;
		tr[ *this].fmin_best = fmin;
		tr[ *this].ref_f0_best = std::abs(reff0);
		tr[ *this].sor_factor = (tr[ *this].sor_factor + SOR_FACTOR_MAX) / 2;
	}
	if((std::abs(shot_this[ *this].ref_f0_best) < std::abs(reff0)) &&
		(fabs(fmin - f0) > fabs(shot_this[ *this].fmin_best - f0)) &&
		(shot_this[ *this].iteration_count > 15)) {
		tr[ *this].iteration_count = 0;
		tr[ *this].sor_factor = (tr[ *this].sor_factor + SOR_FACTOR_MIN) / 2;
		if(stage ==  Payload::STAGE_FIRST) {
			fprintf(stderr, "LCtuner: Rolls back.\n");
			//rolls back to good positions.
			tr[ *this].isSTMChanged = true;
			tr[ *this].stm1 = tr[ *this].stm1_best;
			tr[ *this].stm2 = tr[ *this].stm2_best;
			throw XSkippedRecordError(__FILE__, __LINE__);
		}
	}

	if(stage ==  Payload::STAGE_FIRST) {
		fprintf(stderr, "LCtuner: the first stage\n");
		//Ref(0, 0)
		if(std::abs(reff0) < TUNE_FINETUNE_START) {
			fprintf(stderr, "LCtuner: finetune mode\n");
			tr[ *this].mode = Payload::TUNE_FINETUNE;
		}
		else if(std::abs(reffmin) < TUNE_APPROACH_START) {
			fprintf(stderr, "LCtuner: approach mode\n");
			tr[ *this].mode = Payload::TUNE_APPROACHING;
		}
		else {
			fprintf(stderr, "LCtuner: minimizing mode\n");
			tr[ *this].mode = Payload::TUNE_MINIMIZING;
		}
	}
	//Selects suitable reflection point to be minimized.
	std::complex<double> ref_targeted;
	switch(shot_this[ *this].mode) {
	case Payload::TUNE_MINIMIZING:
		ref_targeted = reftotal;
		break;
	case Payload::TUNE_APPROACHING:
		ref_targeted = reffmin;
		break;
	case Payload::TUNE_FINETUNE:
		ref_targeted = reffmin;
//		ref_targeted = reff0;
		break;
	}
	switch(stage) {
	default:
	case Payload::STAGE_FIRST:
		//Ref(0, 0)
		tr[ *this].fmin_first = fmin;
		double tune_drot;
		switch(shot_this[ *this].mode) {
		case Payload::TUNE_MINIMIZING:
			tune_drot = TUNE_DROT_MINIMIZING;
			break;
		case Payload::TUNE_APPROACHING:
			tune_drot = TUNE_DROT_APPROACH;
			break;
		case Payload::TUNE_FINETUNE:
			tune_drot = TUNE_DROT_FINETUNE;
			break;
		}
		tr[ *this].dCa = tune_drot * ((tr[ *this].dCa > 0) ? 1.0 : -1.0); //same direction.
		tr[ *this].dCb = tune_drot * ((tr[ *this].dCb > 0) ? 1.0 : -1.0);
		tr[ *this].ref_first = ref_targeted;

		tr[ *this].isSTMChanged = true;
		tr[ *this].stage = Payload::STAGE_DCA_FIRST;
		if(stm1__)
			tr[ *this].stm1 += tr[ *this].dCa;
		else
			tr[ *this].stm2 += tr[ *this].dCa;
		throw XSkippedRecordError(__FILE__, __LINE__);  //rotate Ca
		break;
	case Payload::STAGE_DCA_FIRST:
	{
		fprintf(stderr, "LCtuner: +dCa, 1st\n");
		//Ref( +dCa, 0)
		tr[ *this].fmin_plus_dCa = fmin;
		tr[ *this].ref_plus_dCa = ref_targeted;
		tr[ *this].stage = Payload::STAGE_DCA_SECOND;
		tr[ *this].trace_prv.resize(trace_len);
		auto *trace_prv = &tr[ *this].trace_prv[0];
		for(int i = 0; i < trace_len; ++i) {
			trace_prv[i] = trace[i];
		}
		throw XSkippedRecordError(__FILE__, __LINE__); //to next stage.
		break;
	}
	case Payload::STAGE_DCA_SECOND:
	{
		fprintf(stderr, "LCtuner: +dCa, 2nd\n");
		//Ref( +dCa, 0), averaged with the previous.
		tr[ *this].fmin_plus_dCa = (tr[ *this].fmin_plus_dCa + fmin) / 2.0;
		tr[ *this].ref_plus_dCa = (tr[ *this].ref_plus_dCa + ref_targeted) / 2.0;
		//estimates errors.
		if(shot_this[ *this].trace_prv.size() != trace_len) {
			tr[ *m_tuning] = false;
			throw XRecordError(i18n("Record is inconsistent."), __FILE__, __LINE__);
		}
		double ref_sigma = 0.0;
		for(int i = 0; i < trace_len; ++i) {
			ref_sigma += std::norm(trace[i] - shot_this[ *this].trace_prv[i]);
		}
		ref_sigma = sqrt(ref_sigma / trace_len);
		if(ref_sigma > 0.1) {
			fprintf(stderr, "LCtuner: too large errors.\n");
			tr[ *this].stage = Payload::STAGE_FIRST; //to first stage.
			throw XSkippedRecordError(__FILE__, __LINE__);

		}
		tr[ *this].ref_sigma = ref_sigma;

		tr[ *this].trace_prv.clear();
		if(std::abs(reff0) < ref_sigma * 2) {
			tr[ *succeeded()] = true;
			fprintf(stderr, "LCtuner: tuning done within errors.\n");
			return;
		}

		double fmin_err = trace_dfreq;
		for(int i = 0; i < trace_len; ++i) {
			double flen_from_fmin = fabs(trace_start + i * trace_dfreq - fmin);
			if((flen_from_fmin > fmin_err) &&
				(std::abs(reffmin) + ref_sigma * TUNE_DROT_REQUIRED_N_SIGMA > std::abs(trace[i]))) {
					fmin_err = flen_from_fmin;
			}
		}
		tr[ *this].fmin_err = fmin_err;
		fprintf(stderr, "LCtuner: ref_sigma=%f, fmin_err=%f\n", ref_sigma, fmin_err);

		if(( !stm1__ || !stm2__) && (std::abs(fmin - f0) < fmin_err)) {
			tr[ *succeeded()] = true;
			fprintf(stderr, "LCtuner: tuning done within errors.\n");
			return;
		}

		//derivative of freq_min.
		double dfmin = shot_this[ *this].fmin_plus_dCa - shot_this[ *this].fmin_first;
		tr[ *this].dfmin_dCa = dfmin / shot_this[ *this].dCa;

		//derivative of reflection.
		std::complex<double> dref;
		switch(shot_this[ *this].mode) {
		case Payload::TUNE_MINIMIZING:
			tr[ *this].ref_sigma *= 2.0 * sqrt(reftotal) / sqrt(trace_len); //sigma of ref_total.
			break;
		case Payload::TUNE_APPROACHING:
			break;
		case Payload::TUNE_FINETUNE:
			break;
		}
		dref = shot_this[ *this].ref_plus_dCa - shot_this[ *this].ref_first;
		tr[ *this].dref_dCa = dref / shot_this[ *this].dCa;

		if((fabs(dfmin) < fmin_err) && (std::abs(dref) < ref_sigma * TUNE_DROT_REQUIRED_N_SIGMA)) {
			if(fabs(tr[ *this].dCa) < TUNE_DROT_ABORT) {
				if(stm1__)
					tr[ *this].stm1 += 3.0 * tr[ *this].dCa;
				else
					tr[ *this].stm2 += 3.0 * tr[ *this].dCa;
				tr[ *this].dCa *= 4.0; //increases rotation angle to measure derivative.
				tr[ *this].isSTMChanged = true;
				tr[ *this].stage = Payload::STAGE_DCA_FIRST; //rotate C1 more and try again.
				fprintf(stderr, "LCtuner: increasing dCa to %f\n", (double)tr[ *this].dCa);
				throw XSkippedRecordError(__FILE__, __LINE__);
			}
			if( !stm1__ || !stm2__) {
				abortTuningFromAnalyze(tr, reff0);//C1/C2 is useless. Aborts.
			}
			//Ca is useless, try Cb.
			tr[ *this].dref_dCa = 0.0;
			tr[ *this].dfmin_dCa = 0.0;
		}
		if(stm1__ && stm2__) {
			tr[ *this].isSTMChanged = true;
			tr[ *this].stage = Payload::STAGE_DCB; //to next stage.
			tr[ *this].stm1 -= tr[ *this].dCa;
			tr[ *this].stm2 += tr[ *this].dCb;
			throw XSkippedRecordError(__FILE__, __LINE__);
		}
		break; //to final.
	}
	case Payload::STAGE_DCB:
		fprintf(stderr, "LCtuner: +dCb\n");
		//Ref( 0, +dCb)
		double ref_sigma = shot_this[ *this].ref_sigma;
		double fmin_err = shot_this[ *this].fmin_err;

		//derivative of freq_min.
		double dfmin = fmin - shot_this[ *this].fmin_plus_dCa;
		tr[ *this].dfmin_dCb = dfmin / shot_this[ *this].dCb;

		//derivative of reflection.
		std::complex<double> dref;
		dref = ref_targeted - shot_this[ *this].ref_first;
		tr[ *this].dref_dCb = dref / shot_this[ *this].dCb;

		if((fabs(dfmin) < fmin_err) && (std::abs(dref) < ref_sigma * TUNE_DROT_REQUIRED_N_SIGMA)) {
			if(fabs(tr[ *this].dCb) < TUNE_DROT_ABORT) {
				tr[ *this].stm2 += 3.0 * tr[ *this].dCb;
				tr[ *this].dCb *= 4.0; //increases rotation angle to measure derivative.
				tr[ *this].isSTMChanged = true;
				fprintf(stderr, "LCtuner: increasing dCb to %f\n", (double)tr[ *this].dCb);
				//rotate Cb more and try again.
				throw XSkippedRecordError(__FILE__, __LINE__);
			}
			if((shot_this[ *this].dfmin_dCa == 0.0) && (shot_this[ *this].dref_dCa == 0.0))
				abortTuningFromAnalyze(tr, reff0);//C1/C2 is useless. Aborts.
			tr[ *this].dref_dCb = 0.0;
			tr[ *this].dfmin_dCb = 0.0;
		}
		break;
	}
	//Final stage.
	tr[ *this].stage = Payload::STAGE_FIRST;

	std::complex<double> dref_dCa = shot_this[ *this].dref_dCa;
	std::complex<double> dref_dCb = shot_this[ *this].dref_dCb;
	double dabs_ref_dCa = (std::real(ref_targeted) * std::real(dref_dCa) + std::imag(ref_targeted) * std::imag(dref_dCa)) / std::abs(ref_targeted);
	double dabs_ref_dCb = (std::real(ref_targeted) * std::real(dref_dCb) + std::imag(ref_targeted) * std::imag(dref_dCb)) / std::abs(ref_targeted);

	double dfmin_dCa = shot_this[ *this].dfmin_dCa;
	double dfmin_dCb = shot_this[ *this].dfmin_dCb;
	if( !stm1__ || !stm2__) {
		dref_dCb = 0.0;
		dabs_ref_dCb = 0.0;
		dfmin_dCb = 0.0;
	}
	double fmin_err = shot_this[ *this].fmin_err;
	double ref_sigma = shot_this[ *this].ref_sigma;
	double dCa_next = 0;
	double dCb_next = 0;

	fprintf(stderr, "LCtuner: dref_dCa=%.2g, dref_dCb=%.2g, dfmin_dCa=%.2g, dfmin_dCb=%.2g\n",
		dabs_ref_dCa, dabs_ref_dCb, dfmin_dCa, dfmin_dCb);

	double dc_err = 1e10;
	switch(shot_this[ *this].mode) {
	case Payload::TUNE_MINIMIZING:
		if(fabs(dabs_ref_dCa) > fabs(dabs_ref_dCb)) {
			//Decreases reftotal by 2%.
			dCa_next = -0.05 * std::abs(ref_targeted) / dabs_ref_dCa;
		}
		else {
			//Decreases reftotal by 2%.
			dCb_next = -0.05 * std::abs(ref_targeted) / dabs_ref_dCb;
		}
		break;
	case Payload::TUNE_APPROACHING:
		//Solves by |ref| and fmin.
		if( !determineNextC( dCa_next, dCb_next, dc_err,
			std::abs(ref_targeted), ref_sigma,
			fmin - f0, fmin_err,
			dabs_ref_dCa, dabs_ref_dCb,
			dfmin_dCa, dfmin_dCb, "|ref(fmin)| and fmin")) {
			if(dfmin_dCb == 0.0) {
				//Tunes fmin to f0
				dCa_next = -(fmin - f0) / dfmin_dCa;
			}
			else {
				//Tunes fmin to f0
				dCb_next = -(fmin - f0) / dfmin_dCb;
			}
		}
		break;
	case Payload::TUNE_FINETUNE:
		//Solves by |ref| and fmin.
		if( !determineNextC( dCa_next, dCb_next, dc_err,
			std::abs(ref_targeted), ref_sigma,
			fmin - f0, fmin_err,
			dabs_ref_dCa, dabs_ref_dCb,
			dfmin_dCa, dfmin_dCb, "|ref(fmin)| and fmin")) {
			if(dabs_ref_dCb == 0.0) {
				//Decreases ref
				dCa_next = -std::abs(ref_targeted) / dabs_ref_dCa;
			}
			else {
				//Decreases ref
				dCb_next = -std::abs(ref_targeted) / dabs_ref_dCb;
			}
		}
		break;
	}

	fprintf(stderr, "LCtuner: deltaCa=%f, deltaCb=%f\n", dCa_next, dCb_next);

	dCa_next *= shot_this[ *this].sor_factor;
	dCb_next *= shot_this[ *this].sor_factor;
	//restricts changes within the trust region.
	double dc_trust;
	switch(shot_this[ *this].mode) {
	case Payload::TUNE_MINIMIZING:
		dc_trust = TUNE_TRUST_MINIMIZING;
		break;
	case Payload::TUNE_APPROACHING:
		dc_trust = TUNE_TRUST_APPROACH;
		break;
	case Payload::TUNE_FINETUNE:
		dc_trust = TUNE_TRUST_FINETUNE;
		break;
	}
	double dca_trust = fabs(shot_this[ *this].dCa) * 50;
	double dcb_trust = fabs(shot_this[ *this].dCa) * 50;
	double red_fac = std::min(1.0, std::min(std::min(dc_trust, dca_trust) / fabs(dCa_next), std::min(dc_trust, dcb_trust) / fabs(dCb_next)));
	dCa_next *= red_fac;
	dCb_next *= red_fac;
	fprintf(stderr, "LCtuner: deltaCa=%f, deltaCb=%f\n", dCa_next, dCb_next);
	//remembers last direction.
	tr[ *this].dCa = dCa_next;
	tr[ *this].dCb = dCb_next;

	tr[ *this].isSTMChanged = true;
	if(stm1__ && stm2__) {
		tr[ *this].stm1 += dCa_next;
		tr[ *this].stm2 += dCb_next;
	}
	else {
		if(stm1__)
			tr[ *this].stm1 += dCa_next;
		if(stm2__)
			tr[ *this].stm2 += dCa_next;
	}
	throw XSkippedRecordError(__FILE__, __LINE__);
}
void
XAutoLCTuner::visualize(const Snapshot &shot_this) {
	const shared_ptr<XMotorDriver> stm1__ = shot_this[ *stm1()];
	const shared_ptr<XMotorDriver> stm2__ = shot_this[ *stm2()];
	if(shot_this[ *tuning()]) {
		if(shot_this[ *succeeded()]){
			const unsigned int tunebits = 0;
			if(stm1__) {
				for(Transaction tr( *stm1__);; ++tr) {
					tr[ *stm1__->active()] = false; //Deactivates motor.
					tr[ *stm1__->auxBits()] = tunebits; //For external RF relays.
					if(tr.commit())
						break;
				}
			}
			if(stm2__) {
				for(Transaction tr( *stm2__);; ++tr) {
					tr[ *stm2__->active()] = false; //Deactivates motor.
					tr[ *stm2__->auxBits()] = tunebits; //For external RF relays.
					if(tr.commit())
						break;
				}
			}
			msecsleep(50); //waits for relays.
			trans( *tuning()) = false; //finishes tuning successfully.
		}
		if(shot_this[ *this].isSTMChanged) {
			if(stm1__) {
				for(Transaction tr( *stm1__);; ++tr) {
					if(tr[ *stm1__->position()->value()] == shot_this[ *this].stm1)
						break;
					tr[ *stm1__->target()] = shot_this[ *this].stm1;
					if(tr.commit())
						break;
				}
			}
			if(stm2__) {
				for(Transaction tr( *stm2__);; ++tr) {
					if(tr[ *stm2__->position()->value()] == shot_this[ *this].stm2)
						break;
					tr[ *stm2__->target()] = shot_this[ *this].stm2;
					if(tr.commit())
						break;
				}
			}
		}
	}
}

