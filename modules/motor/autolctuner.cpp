/***************************************************************************
		Copyright (C) 2002-2012 Kentaro Kitagawa
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
#include "autolctuner.h"
#include "ui_autolctuner.h"

REGISTER_TYPE(XDriverList, AutoLCTuner, "NMR LC autotuner");

//---------------------------------------------------------------------------
XAutoLCTuner::XAutoLCTuner(const char *name, bool runtime,
	Transaction &tr_meas, const shared_ptr<XMeasure> &meas) :
	XSecondaryDriver(name, runtime, ref(tr_meas), meas),
		m_stm1(create<XItemNode<XDriverList, XMotorDriver> >("STM1", false, ref(tr_meas), meas->drivers(), true)),
		m_stm2(create<XItemNode<XDriverList, XMotorDriver> >("STM2", false, ref(tr_meas), meas->drivers(), true)),
		m_netana(create<XItemNode<XDriverList, XNetworkAnalyzer> >("NetworkAnalyzer", false, ref(tr_meas), meas->drivers(), true)),
		m_target(create<XDoubleNode>("Target", false)),
		m_abortTuning(create<XTouchableNode>("AbortTuning", false)),
		m_form(new FrmNMRPulse(g_pFrmMain)) {
	connect(stm1());
	connect(stm2());
	connect(netana());

	m_form->setWindowTitle(i18n("NMR LC autotuner - ") + getLabel() );

	m_conSTM1 = xqcon_create<XQComboBoxConnector>(stm1(), m_form->m_cmbSTM1, ref(tr_meas));
	m_conSTM2 = xqcon_create<XQComboBoxConnector>(stm2(), m_form->m_cmbSTM2, ref(tr_meas));
	m_conNetAna = xqcon_create<XQComboBoxConnector>(netana(), m_form->m_cmbNetAna, ref(tr_meas));
	m_conTarget = xqcon_create<XQLineEditConnector>(target(), m_form->m_edTarget);
	m_conAbortTuning = xqcon_create<XQButtonConnector>(m_abortTuning, m_form->m_btnAbortTuning);

	for(Transaction tr( *this);; ++tr) {
		m_lsnOnTargetChanged = tr[ *m_target].onTouch().connectWeakly(
			shared_from_this(), &XAutoLCTuner::onTargetChanged);
		m_lsnOnAbortTouched = tr[ *m_target].onTouch().connectWeakly(
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
	trans( *this).mode = Payload::TUNE_MINIMIZING;
}
void XAutoLCTuner::onAbortTuningTouched(const Snapshot &shot, XTouchableNode *) {
	trans( *this).mode = Payload::TUNE_INACTIVE;
}
void
XAutoLCTuner::determineNextC(double &deltaC1, double &deltaC2, double &err,
	double x, double x_err,
	double y, double y_err,
	double dxdC1, double dxdC2,
	double dydC1, double dydC2) {
	double det = dxdC1 * dydC2 - dxdC2 *dydC1;
	double esq = (dydC2 * x_err * dydC2 * x_err + dxdC2 * y_err * dxdC2 * y_err) / det / det
		+ (dydC1 * x_err * dydC1 * x_err + dxdC1 * y_err * dxdC1 * y_err) / det / det;
	if(esq < err * err) {
		err = sqrt(esq);
		deltaC1 = -(dydC2 * x - dxdC2 * y) / det;
		deltaC2 = -( -dydC1 * x + dxdC1 * y) / det;
	}
}
bool XAutoLCTuner::checkDependency(const Snapshot &shot_this,
	const Snapshot &shot_emitter, const Snapshot &shot_others,
	XDriver *emitter) const {
	const shared_ptr<XMotorDriver> stm1__ = shot_this[ *stm1()];
	const shared_ptr<XMotorDriver> stm2__ = shot_this[ *stm2()];
	const shared_ptr<XNetworkAnalyzer> na__ = shot_this[ *netana()];
	if (emitter == pulse__.get())
		return false;
	if( !stm1__ || !stm2__ || !na__)
		return false;
	if (emitter != na__.get())
		return false;

	return true;
}
void
XAutoLCTuner::analyze(Transaction &tr, const Snapshot &shot_emitter,
	const Snapshot &shot_others,
	XDriver *emitter) throw (XRecordError&) {
	const Snapshot &shot_this(tr);
	if(shot_this[ *this].mode == Payload::TUNE_INACTIVE)
		throw XSkippedRecordError(__FILE__, __LINE__);

	const shared_ptr<XMotorDriver> stm1__ = shot_this[ *stm1()];
	const shared_ptr<XMotorDriver> stm2__ = shot_this[ *stm2()];
	const shared_ptr<XNetworkAnalyzer> na__ = shot_this[ *netana()];

	if( !shot_others[ *stm1__->ready()] ||  !shot_others[ *stm2__->ready()])
		throw XSkippedRecordError(__FILE__, __LINE__); //STM is moving.
	if( shot_this[ *this].isSTMChanged) {
		tr[ *this].isSTMChanged = false;
		throw XSkippedRecordError(__FILE__, __LINE__); //this data is not reliable.
	}
	bool stm1slip = shot_others[ *stm1__->slipping()];
	bool stm2slip = shot_others[ *stm2__->slipping()];
	double c1 = shot_others[ *stm1__->position()].value();
	double c2 = shot_others[ *stm2__->position()].value();
	tr[ *this].stm1 = c1;
	tr[ *this].stm2 = c2;
	double dc1 = shot_this[ *this].dC1;
	double dc2 = shot_this[ *this].dC2;

	const std::complex<double> *trace = shot_na[ *na].trace();
	int trace_len = shot_na[ *na__].length();
	double trace_dfreq = shot_na[ *na__].freqInterval();
	double trace_start = shot_na[ *na__].startFreq();
	std::complex<double> reffmin(1e10);
	double f0 = shot_this[ *target()];
	//searches for minimum in reflection.
	double fmin;
	for(int i = 0; i < trace_len; ++i) {
		if(std::abs(reffmin) > std::abs(trace[i])) {
			reffmin = trace[i];
			fmin = trace_start + i * trace_dfreq;
		}
	}
	//Reflection at the target frequency.
	std::complex<double> reff0;
	for(int i = 0; i < trace_len; ++i) {
		if(trace_start + i * trace_dfreq >= f0) {
			reff0 = trace[i];
			break;
		}
	}
	fprintf(stderr, "LCtuner: fmin=%f, reffmin=%f, reff0=%f\n", fmin, std::abs(reffmin), std::abs(reff0));
	if(std::abs(reff0) < TUNE_APPROACH_GOAL) {
		tr[ *this].mode = Payload::TUNE_INACTIVE;
		return;
	}

	Payload::STAGE stage = shot_this[ *this].stage;
	switch(stage) {
	default:
	case Payload::STAGE_FIRST:
		fprintf(stderr, "LCtuner: the first stage\n");
		//Ref(0, 0)
		tr[ *this].fmin_first = fmin;
		tr[ *this].ref_fmin_first = reffmin;
		tr[ *this].ref_f0_first = reff0;
		tr[ *this] = Payload::STAGE_DC1_FIRST;
		if(std::abs(reffmin) < TUNE_APPROACH_START) {
			fprintf(stderr, "LCtuner: approach mode\n");
			tr[ *this].mode = TUNE_APPRACHING;
			tr[ *this].dC1 = TUNE_DROT_APPRACH * ((tr[ *this].dC1 > 0) ? 1.0 : -1.0);
			tr[ *this].dC2 = TUNE_DROT_APPRACH * ((tr[ *this].dC2 > 0) ? 1.0 : -1.0);
		}
		else {
			fprintf(stderr, "LCtuner: minimizing mode\n");
			tr[ *this].mode = TUNE_MINIMIZING;
			tr[ *this].dC1 = TUNE_DROT_MINIMIZING * ((tr[ *this].dC1 > 0) ? 1.0 : -1.0);
			tr[ *this].dC2 = TUNE_DROT_MINIMIZING * ((tr[ *this].dC2 > 0) ? 1.0 : -1.0);
		}
		tr[ *this].isSTMChanged = true;
		tr[ *this].stm1 += tr[ *this].dC1;
		throw XSkippedRecordError(__FILE__, __LINE__); //rotate C1
		break;
	case Payload::STAGE_DC1_FIRST:
		fprintf(stderr, "LCtuner: +dC1, 1st\n");
		//Ref( +dC1, 0)
		tr[ *this].fmin_plus_dc1 = fmin;
		tr[ *this].ref_fmin_plus_dc1 = reffmin;
		tr[ *this].ref_f0_plus_dc1 = reff0;
		tr[ *this] = Payload::STAGE_DC1_SECOND;
		tr[ *this].trace_prv.resize(trace_len);
		auto *trace_prv = &tr[ *this].trace_prv[0];
		for(int i = 0; i < trace_len; ++i) {
			trace_prv[i] = trace[i];
		}
		throw XSkippedRecordError(__FILE__, __LINE__); //to next stage.
		break;
	case Payload::STAGE_DC1_SECOND:
	{
		fprintf(stderr, "LCtuner: +dC1, 2nd\n");
		//Ref( +dC1, 0), averaged with the previous.
		tr[ *this].fmin_plus_dc1 = (tr[ *this].fmin_plus_dc1 + fmin) / 2.0;
		tr[ *this].ref_fmin_plus_dc1 = (tr[ *this].ref_fmin_plus_dc1 + reffmin) / 2.0;
		tr[ *this].ref_f0_plus_dc1 = (tr[ *this].ref_f0_plus_dc1 + reff0) / 2.0;

		//estimates errors.
		double ref_sigma = 0.0;
		for(int i = 0; i < trace_len; ++i) {
			ref_sigma += std::norm(trace[i] - shot_this[ *this].trace_prv[i]);
		}
		ref_sigma = sqrt(ref_sigma / trace_len);
		tr[ *this].ref_sigma = ref_sigma;
		tr[ *this].trace_prv.clear();
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

		double dfmin = fmin - shot_this[ *this].fmin_first;
		tr[ *this].dfmin_dC1 = dfmin / shot_this[ *this].dC1;
		std::comlex<double> dref = (shot_this[ *this].mode == TUNE_APPRACHING) ? (ref_f0 - shot_this[ *this].ref_f0_first) : (ref_fmin - shot_this[ *this].ref_fmin_first);
		tr[ *this].dref_dC1 = dref / shot_this[ *this].dC1;
		if((fabs(dfmin) < fmin_err) && (std::abs(dref) < ref_sigma * TUNE_DROT_REQUIRED_N_SIGMA)) {
			if(tr[ *this].dC1 < TUNE_DROT_ABORT) {
				tr[ *this].stm1 += tr[ *this].dC1;
				tr[ *this].dC1 *= 2.0; //increases rotation angle to measure derivative.
				tr[ *this].isSTMChanged = true;
				tr[ *this] = Payload::STAGE_DC1_FIRST; //rotate C1 more and try again.
				fprintf(stderr, "LCtuner: increasing dC1 to %f\n", (double)tr[ *this].dC1);
				throw XSkippedRecordError(__FILE__, __LINE__);

			}
			//C1 is useless, try C2.
		}

		tr[ *this] = Payload::STAGE_DC2; //to next stage.
		throw XSkippedRecordError(__FILE__, __LINE__);
		break;
	}
	case Payload::STAGE_DC2:
		fprintf(stderr, "LCtuner: +dC2\n");
		break;
	}
	//Final stage.
	//Ref( +dC1, +dC2)
	double ref_sigma = shot_this[ *this].ref_sigma;
	double fmin_err = shot_this[ *this].fmin_err;
	double dfmin = fmin - shot_this[ *this].fmin_plus_dc1;
	tr[ *this].dfmin_dC2 = dfmin / shot_this[ *this].dC2;
	std::comlex<double> dref = (shot_this[ *this].mode == TUNE_APPRACHING) ? (ref_f0 - shot_this[ *this].ref_f0_plus_dc1) : (ref_fmin - shot_this[ *this].ref_fmin_plus_dc1);
	tr[ *this].dref_dC2 = dref / shot_this[ *this].dC2;
	if((fabs(dfmin) < fmin_err) && (std::abs(dref) < ref_sigma * TUNE_DROT_REQUIRED_N_SIGMA)) {
		if(tr[ *this].dC2 < TUNE_DROT_ABORT) {
			tr[ *this].stm2 += tr[ *this].dC2;
			tr[ *this].dC2 *= 2.0; //increases rotation angle to measure derivative.
			tr[ *this].isSTMChanged = true;
			fprintf(stderr, "LCtuner: increasing dC2 to %f\n", (double)tr[ *this].dC2);
			//rotate C2 more and try again.
			throw XSkippedRecordError(__FILE__, __LINE__);
		}
		if(tr[ *this].dC1 > TUNE_DROT_ABORT)
			throw XRecordError(i18n("Aborting. the target is out of tune, or capasitors have sticked."), __FILE__, __LINE__); //C1 and C2 are useless. Aborts.
	}

	tr[ *this] = Payload::STAGE_FIRST;

	double dc1 = shot_this[ *this].dC1;
	double dc2 = shot_this[ *this].dC2;

	std::complex<double> dref_dc1 = shot_this[ *this].dref_dC1;
	std::complex<double> dref_dc2 = shot_this[ *this].dref_dC2;
	double dfmin_dc1 = shot_this[ *this].dfmin_dC1;
	double dfmin_dc2 = shot_this[ *this].dfmin_dC2;
	double dc1_next = 0;
	double dc2_next = 0;

	if(shot_this[ *this].mode == TUNE_APPRACHING) {
	//Tunes fmin to f0, and/or ref_f0 to 0
		if(tr[ *this].dC1 > TUNE_DROT_ABORT) {
			dc2_next = -std::abs(ref_f0) / std::abs(dref_dc2);
		}
		else if(tr[ *this].dC2 > TUNE_DROT_ABORT) {
			dc1_next = -std::abs(ref_f0) / std::abs(dref_dc1);
		}
		else {
			double dc_err = 1e10;
			//Solves by real(ref) and imag(ref).
			determineNextC( dc1_next, dc2_next, dc_err,
				std::real(ref), ref_sigma * TUNE_DROT_REQUIRED_N_SIGMA,
				std::imag(ref), ref_sigma * TUNE_DROT_REQUIRED_N_SIGMA,
				std::real(dref_dC1), std::real(dref_dC2),
				std::imag(dref_dC1), std::imag(dref_dC2));
			//Solves by real(ref) and fmin.
			determineNextC( dc1_next, dc2_next, dc_err,
				std::real(ref), ref_sigma * TUNE_DROT_REQUIRED_N_SIGMA,
				fmin, fmin_err,
				std::real(dref_dC1), std::real(dref_dC2),
				dfmin_dC1, dfmin_dC2);
		}
	}
	else {
	//Tunes ref_fmin to 0
		if(std::abs(dref_dc1) > std::abs(dref_dc2)) {
			dc1_next = -std::abs(ref_fmin) / std::abs(dref_dc1);
		}
		else {
			dc2_next = -std::abs(ref_fmin) / std::abs(dref_dc2);
		}
	}
	fprintf(stderr, "LCtuner: deltaC1=%f, deltaC2=%f\n", dc1_next, dc2_next);

	//restricts them within the trust region.
	double dc_max = sqrt(dc1_next * dc1_next + dc2_next * dc2_next);
	double dc_trust = (mode == TUNE_MODE_APPROACHING) ? TUNE_TRUST_APPROACHING :TUNE_TRUST_MINIMIZING;
	if(dc_max > dc_trust) {
		dc1_next *= dc_trust / dc_max;
		dc2_next *= dc_trust / dc_max;
		fprintf(stderr, "LCtuner: deltaC1=%f, deltaC2=%f\n", dc1_next, dc2_next);
	}
	tr[ *this].isSTMChanged = true;
	tr[ *this].stm1 += dc1_next;
	tr[ *this].stm2 += dc2_next;

	throw XSkippedRecordError(__FILE__, __LINE__);
}
void
XAutoLCTuner::visualize() {
	if( shot_this[ *this].isSTMChanged) {
		for(Transaction tr( *stm1);; ++tr) {
			if(tr[ *stm1->position()].value() == shot_this[ *this].stm1)
				break;
			tr[ *stm1].target() = shot_this[ *this].stm1;
			if(tr.commit())
				break;
		}
		for(Transaction tr( *stm2);; ++tr) {
			if(tr[ *stm2->position()].value() == shot_this[ *this].stm2)
				break;
			tr[ *stm2].target() = shot_this[ *this].stm2;
			if(tr.commit())
				break;
		}
	}
}

