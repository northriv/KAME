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

static const double TUNE_DROT_APPROACH = 5.0,
	TUNE_DROT_FINETUNE = 2.0, TUNE_DROT_ABORT = 180.0; //[deg.]
static const double TUNE_TRUST_APPROACH = 720.0, TUNE_TRUST_FINETUNE = 360.0; //[deg.]
static const double TUNE_FINETUNE_START = 0.7; //-3dB@f0
static const double TUNE_DROT_REQUIRED_N_SIGMA_FINETUNE = 1.0;
static const double TUNE_DROT_REQUIRED_N_SIGMA_APPROACH = 2.0;
static const double SOR_FACTOR_MAX = 0.9;
static const double SOR_FACTOR_MIN = 0.3;
static const double TUNE_DROT_MUL_FINETUNE = 2.5;
static const double TUNE_DROT_MUL_APPROACH = 3.5;

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
		m_reflectionTargeted(create<XDoubleNode>("ReflectionTargeted", false)),
		m_reflectionRequired(create<XDoubleNode>("ReflectionRequired", false)),
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
	m_conReflectionTargeted = xqcon_create<XQLineEditConnector>(reflectionTargeted(), m_form->m_edReflectionTargeted);
	m_conReflectionRequired = xqcon_create<XQLineEditConnector>(reflectionRequired(), m_form->m_edReflectionRequired);
	m_conAbortTuning = xqcon_create<XQButtonConnector>(m_abortTuning, m_form->m_btnAbortTuning);
	m_conTuning = xqcon_create<XKLedConnector>(m_tuning, m_form->m_ledTuning);
	m_conSucceeded = xqcon_create<XKLedConnector>(m_succeeded, m_form->m_ledSucceeded);
	m_conUseSTM1 = xqcon_create<XQToggleButtonConnector>(m_useSTM1, m_form->m_ckbUseSTM1);
	m_conUseSTM2 = xqcon_create<XQToggleButtonConnector>(m_useSTM2, m_form->m_ckbUseSTM2);

	for(Transaction tr( *this);; ++tr) {
		tr[ *m_tuning] = false;
		tr[ *m_succeeded] = false;
		tr[ *m_reflectionTargeted] = -15.0;
		tr[ *m_reflectionRequired] = -7.0;
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
		tr[ *this].trace.clear();
		if(tr.commit())
			break;
	}
}
void XAutoLCTuner::onAbortTuningTouched(const Snapshot &shot, XTouchableNode *) {
	for(Transaction tr( *this);; ++tr) {
		if( !tr[ *m_tuning])
			break;
		tr[ *m_tuning] = false;
		tr[ *this].isSTMChanged = false;
		if(tr.commit())
			break;
	}
}
void
XAutoLCTuner::determineNextC(double &deltaC1, double &deltaC2,
	double x, double x_err,
	double y, double y_err,
	double dxdC1, double dxdC2,
	double dydC1, double dydC2) {
	double det = dxdC1 * dydC2 - dxdC2 *dydC1;
	double slimit = 1e-60;
	x -= ((x > 0) ? 1 : -1) * std::min(x_err, fabs(x)); //reduces as large as err.
	y -= ((y > 0) ? 1 : -1) * std::min(y_err, fabs(y));
	if(det > slimit) {
		deltaC1 = -(dydC2 * x - dxdC2 * y) / det;
		deltaC2 = -( -dydC1 * x + dxdC1 * y) / det;
	}
	else {
		if(fabs(x) > x_err) {
			//superior to y
			if(fabs(dxdC2) > slimit) {
				deltaC2 = -x / dxdC2;
			}
			if(fabs(dxdC1) > slimit) {
				deltaC1 = -x / dxdC1;
			}
		}
		else {
			if(fabs(dydC2) > slimit) {
				deltaC2 = -y / dydC2;
			}
			if(fabs(dydC1) > slimit) {
				deltaC1 = -y / dydC1;
			}
		}
	}
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

	int trace_len = shot_na[ *na__].length();
	double ref_sigma = 0.0;
	{
		const std::complex<double> *trace = shot_na[ *na__].trace();
		if(shot_this[ *this].trace.size() != trace_len) {
			//copies trace.
			tr[ *this].trace.resize(trace_len);
			for(int i = 0; i < trace_len; ++i) {
				tr[ *this].trace[i] = trace[i];
			}
			//re-acquires the same situation.
			throw XSkippedRecordError(__FILE__, __LINE__);
		}

		//estimates errors.
		for(int i = 0; i < trace_len; ++i) {
			ref_sigma += std::norm(trace[i] - shot_this[ *this].trace[i]);
			tr[ *this].trace[i] = (shot_this[ *this].trace[i] + trace[i]) / 2.0; //takes averages.
		}
		ref_sigma = sqrt(ref_sigma / trace_len);
		if(ref_sigma > 0.1) {
			tr[ *this].trace.clear();
			throw XSkippedRecordError(i18n("Too large errors in the trace."), __FILE__, __LINE__);
		}
	}
	double trace_dfreq = shot_na[ *na__].freqInterval();
	double trace_start = shot_na[ *na__].startFreq();
	double fmin_err;
	double fmin;
	std::complex<double> reffmin(0.0);
	double f0 = shot_this[ *target()];
	std::complex<double> reff0(0.0);
	double reffmin_sigma, reff0_sigma;
	//analyzes trace.
	{
		const std::complex<double> *trace = &shot_this[ *this].trace[0];

		std::complex<double> reffmin_peak(1e10);
		//searches for minimum in reflection.
		double fmin_peak = 0;
		for(int i = 0; i < trace_len; ++i) {
			double z = std::abs(trace[i]);
			if(std::abs(reffmin_peak) > z) {
				reffmin_peak = trace[i];
				fmin_peak = trace_start + i * trace_dfreq;
			}
		}

		//Takes averages around the minimum.
		reffmin = reffmin_peak;
		double ref_sigma_sq = ref_sigma * ref_sigma;
		for(int cnt = 0; cnt < 2; ++cnt) {
			double wsum = 0.0;
			double wsqsum = 0.0;
			std::complex<double> refsum = 0.0;
			double fsum = 0.0;
			double fsqsum = 0.0;
			for(int i = 0; i < trace_len; ++i) {
				double f = trace_start + i * trace_dfreq;
				double zsq = std::norm(trace[i] - reffmin);
				if(zsq < ref_sigma_sq * 10) {
					double w = exp( -zsq / (2.0 * ref_sigma_sq));
					wsum += w;
					wsqsum += w * w;
					refsum += w * trace[i];
					fsum += w * f;
					fsqsum += w * w * (f - fmin) * (f - fmin);
				}
			}
			fmin = fsum / wsum;
			fmin_err = sqrt(fsqsum / wsum / wsum + trace_dfreq * trace_dfreq);
			reffmin = refsum / wsum;
			reffmin_sigma = ref_sigma * sqrt(wsqsum) / wsum;
		}
		//Takes averages around the target frequency.
		double wsum = 0.0;
		double wsqsum = 0.0;
		std::complex<double> refsum = 0.0;
		double f0_err = std::min(trace_dfreq * 5, fmin_err);
		for(int i = 0; i < trace_len; ++i) {
			double f = trace_start + i * trace_dfreq;
			if(fabs(f - f0) < f0_err * 10) {
				double w = exp( -(f - f0) * (f - f0) / (2.0 * f0_err * f0_err));
				wsum += w;
				wsqsum += w * w;
				refsum += w * trace[i];
			}
		}
		reff0 = refsum / wsum;
		reff0_sigma = ref_sigma * sqrt(wsqsum / wsum * wsum);
		fprintf(stderr, "LCtuner: fmin=%.4f+-%.4f, reffmin=%.3f+-%.3f, reff0=%.3f+-%.3f\n",
				fmin, fmin_err, std::abs(reffmin), reffmin_sigma, std::abs(reff0), reff0_sigma);

		tr[ *this].trace.clear();
	}

	if(std::abs(reff0) < reff0_sigma * 2) {
		tr[ *succeeded()] = true;
		fprintf(stderr, "LCtuner: tuning done within errors.\n");
		return;
	}

	if(( !stm1__ || !stm2__) && (std::abs(fmin - f0) < fmin_err)) {
		tr[ *succeeded()] = true;
		fprintf(stderr, "LCtuner: tuning done within errors.\n");
		return;
	}


	double tune_approach_goal = pow(10.0, 0.05 * shot_this[ *reflectionTargeted()]);
	if(std::abs(reff0) < tune_approach_goal) {
		tr[ *succeeded()] = true;
		return;
	}
	double tune_approach_goal2 = pow(10.0, 0.05 * shot_this[ *reflectionRequired()]);
	if(std::abs(reff0) < tune_approach_goal2) {
		if(shot_this[ *this].sor_factor < (SOR_FACTOR_MAX - SOR_FACTOR_MIN) * pow(2.0, -4.0) + SOR_FACTOR_MIN) {
			tr[ *succeeded()] = true;
			return;
		}
	}

	Payload::STAGE stage = shot_this[ *this].stage;

	tr[ *this].iteration_count++;
	if(std::abs(shot_this[ *this].ref_f0_best) > std::abs(reff0)) {
		tr[ *this].iteration_count = 0;
		//remembers good positions.
		tr[ *this].stm1_best = tr[ *this].stm1;
		tr[ *this].stm2_best = tr[ *this].stm2;
		tr[ *this].fmin_best = fmin;
		tr[ *this].ref_f0_best = std::abs(reff0) + reff0_sigma;
		tr[ *this].sor_factor = (tr[ *this].sor_factor + SOR_FACTOR_MAX) / 2;
	}
//	else
//		tr[ *this].sor_factor = std::min(tr[ *this].sor_factor, (SOR_FACTOR_MAX + SOR_FACTOR_MIN) / 2);

	if((std::abs(shot_this[ *this].ref_f0_best) + reff0_sigma < std::abs(reff0)) &&
		((shot_this[ *this].iteration_count > 10) ||
		((shot_this[ *this].iteration_count > 4) && (std::abs(shot_this[ *this].ref_f0_best) * 1.5 <  std::abs(reff0) - reff0_sigma)))) {
		if(stage ==  Payload::STAGE_FIRST) {
			tr[ *this].iteration_count = 0;
			tr[ *this].sor_factor = (tr[ *this].sor_factor + SOR_FACTOR_MIN) / 2;
			if(shot_this[ *this].sor_factor < (SOR_FACTOR_MAX - SOR_FACTOR_MIN) * pow(2.0, -6.0) + SOR_FACTOR_MIN) {
				abortTuningFromAnalyze(tr, reff0);//Aborts.
			}
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
		else {
			fprintf(stderr, "LCtuner: approach mode\n");
			tr[ *this].mode = Payload::TUNE_APPROACHING;
		}
	}
	//Selects suitable reflection point to be minimized.
	std::complex<double> ref_targeted;
	double tune_drot_required_nsigma;
	double tune_drot_mul;
	switch(shot_this[ *this].mode) {
	case Payload::TUNE_FINETUNE:
		ref_targeted = reff0;
		ref_sigma = reff0_sigma;
		tune_drot_required_nsigma = TUNE_DROT_REQUIRED_N_SIGMA_FINETUNE;
		tune_drot_mul = TUNE_DROT_MUL_FINETUNE;
		break;
	case Payload::TUNE_APPROACHING:
		ref_targeted = reffmin;
		ref_sigma = reffmin_sigma;
		tune_drot_required_nsigma = TUNE_DROT_REQUIRED_N_SIGMA_APPROACH;
		tune_drot_mul = TUNE_DROT_MUL_APPROACH;
		break;
	}
	switch(stage) {
	default:
	case Payload::STAGE_FIRST:
		//Ref(0, 0)
		tr[ *this].fmin_first = fmin;
		tr[ *this].ref_first = ref_targeted;
		double tune_drot;
		switch(shot_this[ *this].mode) {
		case Payload::TUNE_APPROACHING:
			tune_drot = TUNE_DROT_APPROACH;
			break;
		case Payload::TUNE_FINETUNE:
			tune_drot = TUNE_DROT_FINETUNE;
			break;
		}
		tr[ *this].dCa = tune_drot * ((shot_this[ *this].dfmin_dCa * (fmin - f0) < 0) ? 1.0 : -1.0); //direction to approach.
		tr[ *this].dCb = tune_drot * ((shot_this[ *this].dfmin_dCb * (fmin - f0) < 0) ? 1.0 : -1.0);

		tr[ *this].isSTMChanged = true;
		tr[ *this].stage = Payload::STAGE_DCA;
		if(stm1__)
			tr[ *this].stm1 += tr[ *this].dCa;
		else
			tr[ *this].stm2 += tr[ *this].dCa;
		throw XSkippedRecordError(__FILE__, __LINE__);  //rotate Ca
		break;
	case Payload::STAGE_DCA:
	{
		fprintf(stderr, "LCtuner: +dCa\n");
		//Ref( +dCa, 0), averaged with the previous.
		tr[ *this].fmin_plus_dCa = fmin;
		tr[ *this].ref_plus_dCa = ref_targeted;

		//derivative of freq_min.
		double dfmin = shot_this[ *this].fmin_plus_dCa - shot_this[ *this].fmin_first;
		tr[ *this].dfmin_dCa = dfmin / shot_this[ *this].dCa;

		//derivative of reflection.
		std::complex<double> dref;
		dref = shot_this[ *this].ref_plus_dCa - shot_this[ *this].ref_first;
		tr[ *this].dref_dCa = dref / shot_this[ *this].dCa;

		if((fabs(dfmin) < fmin_err * tune_drot_required_nsigma) &&
			(std::abs(dref) < ref_sigma * tune_drot_required_nsigma)) {
			if(fabs(tr[ *this].dCa) < TUNE_DROT_ABORT) {
				tr[ *this].dCa *= tune_drot_mul; //increases rotation angle to measure derivative.
				tr[ *this].fmin_first = fmin; //the present data may be influenced by backlashes.
				tr[ *this].ref_first = ref_targeted;
				if(stm1__)
					tr[ *this].stm1 += tr[ *this].dCa;
				else
					tr[ *this].stm2 += tr[ *this].dCa;
				tr[ *this].isSTMChanged = true;
				tr[ *this].stage = Payload::STAGE_DCA; //rotate C1 more and try again.
				fprintf(stderr, "LCtuner: increasing dCa to %f\n", (double)tr[ *this].dCa);
				throw XSkippedRecordError(__FILE__, __LINE__);
			}
			if( !stm1__ || !stm2__) {
				abortTuningFromAnalyze(tr, reff0);//C1/C2 is useless. Aborts.
			}
			//Ca is useless, try Cb.
		}
		if(stm1__ && stm2__) {
			tr[ *this].isSTMChanged = true;
			tr[ *this].stage = Payload::STAGE_DCB; //to next stage.
			tr[ *this].stm2 += tr[ *this].dCb;
			throw XSkippedRecordError(__FILE__, __LINE__);
		}
		break; //to final.
	}
	case Payload::STAGE_DCB:
		fprintf(stderr, "LCtuner: +dCb\n");
		//Ref( 0, +dCb)
		//derivative of freq_min.
		double dfmin = fmin - shot_this[ *this].fmin_plus_dCa;
		tr[ *this].dfmin_dCb = dfmin / shot_this[ *this].dCb;

		//derivative of reflection.
		std::complex<double> dref;
		dref = ref_targeted - shot_this[ *this].ref_plus_dCa;
		tr[ *this].dref_dCb = dref / shot_this[ *this].dCb;

		if((std::min(fabs(shot_this[ *this].dfmin_dCa * shot_this[ *this].dCa), fabs(dfmin)) < fmin_err * tune_drot_required_nsigma) &&
			(std::min(std::abs(shot_this[ *this].dref_dCa * shot_this[ *this].dCa), std::abs(dref)) < ref_sigma * tune_drot_required_nsigma)) {
			if(fabs(tr[ *this].dCb) < TUNE_DROT_ABORT) {
				tr[ *this].dCb *= tune_drot_mul; //increases rotation angle to measure derivative.
				tr[ *this].fmin_plus_dCa = fmin; //the present data may be influenced by backlashes.
				tr[ *this].ref_plus_dCa = ref_targeted;
				tr[ *this].stm2 += tr[ *this].dCb;
				tr[ *this].isSTMChanged = true;
				fprintf(stderr, "LCtuner: increasing dCb to %f\n", (double)tr[ *this].dCb);
				//rotate Cb more and try again.
				throw XSkippedRecordError(__FILE__, __LINE__);
			}
			if(fabs(tr[ *this].dCa) >= TUNE_DROT_ABORT)
				abortTuningFromAnalyze(tr, reff0);//C1/C2 is useless. Aborts.
		}
		break;
	}
	//Final stage.
	tr[ *this].stage = Payload::STAGE_FIRST;

	std::complex<double> dref_dCa = shot_this[ *this].dref_dCa;
	std::complex<double> dref_dCb = shot_this[ *this].dref_dCb;
	double gamma;
	switch(shot_this[ *this].mode) {
	case Payload::TUNE_FINETUNE:
//		gamma = 0.5;
//		break;
	case Payload::TUNE_APPROACHING:
		gamma = 1.0;
		break;
	}
	double a = gamma * 2.0 * pow(std::norm(ref_targeted), gamma - 1.0);
	// d |ref^2|^gamma / dCa,b
	double drefgamma_dCa = a * (std::real(ref_targeted) * std::real(dref_dCa) + std::imag(ref_targeted) * std::imag(dref_dCa));
	double drefgamma_dCb = a * (std::real(ref_targeted) * std::real(dref_dCb) + std::imag(ref_targeted) * std::imag(dref_dCb));

	double dfmin_dCa = shot_this[ *this].dfmin_dCa;
	double dfmin_dCb = shot_this[ *this].dfmin_dCb;
	if( !stm1__ || !stm2__) {
		dref_dCb = 0.0;
		drefgamma_dCb = 0.0;
		dfmin_dCb = 0.0;
	}
	double dCa_next = 0;
	double dCb_next = 0;

	fprintf(stderr, "LCtuner: dref_dCa=%.2g, dref_dCb=%.2g, dfmin_dCa=%.2g, dfmin_dCb=%.2g\n",
		drefgamma_dCa, drefgamma_dCb, dfmin_dCa, dfmin_dCb);

	determineNextC( dCa_next, dCb_next,
		pow(std::norm(ref_targeted), gamma), pow(ref_sigma, gamma * 2.0),
		fmin - f0, fmin_err,
		drefgamma_dCa, drefgamma_dCb,
		dfmin_dCa, dfmin_dCb);

	fprintf(stderr, "LCtuner: deltaCa=%f, deltaCb=%f\n", dCa_next, dCb_next);

	dCa_next *= shot_this[ *this].sor_factor;
	dCb_next *= shot_this[ *this].sor_factor;
	//restricts changes within the trust region.
	double dc_trust;
	switch(shot_this[ *this].mode) {
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
		msecsleep(50); //waits for ready indicators.
		if( !shot_this[ *tuning()]) {
			trans( *this).isSTMChanged = false;
		}
	}
}

