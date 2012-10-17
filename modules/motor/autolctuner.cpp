/*
 * autolctuner.cpp
 *
 *  Created on: 2012/07/15
 *      Author: northriv
 */

/*
 * Tunes s11(complex) to zero, and the primary marker to the given frequency.
 * (before locking) firstly decreases |s11| at any freq.
 * (approaching mode) picks up derivatives for two of real(s11), imag(s11), DeltaF so that the det. of Jacobian has larger value.
 */
class XAutoLCTuner : public XSecondaryDriver {
public:
	XAutoLCTuner();
	virtual ~XAutoLCTuner();
	class Payload : public XSconadryDriver::Payload {
	public:
		std::complex<double> lastS11;
		double lastDFreq;
		bool hasSlippedC1, hasSlippedC2;
		std::complex<double> dS11dC1, dS11dC2;
		double dDFreqdC1, dDFreqdC2;
		bool hasMeasDerivC1, hasMeasDerivC2;
		double dC1, dC2;
		enum MODE {TUNE_FIRST_TIME, TUNE_MINIMIZING, TUNE_APPROACHING};
		MODE mode;
	};
private:
	enum {TUNE_DROT_TUNE_MINIMIZING = 0.2, TUNE_DROT_APPROACH = 0.005};
	enum {TUNE_TRUST_TUNE_MINIMIZING = 2.0, TUNE_TRUST_APPROACH = 0.5};
};

void
analyze() {
	if( !shot_others[ *stm1__->ready()] ||  !shot_others[ *stm2__->ready()])
		throw XSkippedRecordError(__FILE__, __LINE__);
//	bool stm1slip = shot_others[ *stm1__->slipping()];
//	bool stm2slip = shot_others[ *stm2__->slipping()];
	double c1 = shot_others[ *stm1__->position()].value();
	double c2 = shot_others[ *stm2__->position()].value();
	double dc1 = shot_this[ *this].dC1;
	double dc2 = shot_this[ *this].dC2;
	const std::complex<double> *trace = shot_na[ *na].trace();
	int trace_len = shot_na[ *na].length();
	double trace_dfreq = shot_na[ *na].freqInterval();
	double trace_start = shot_na[ *na].startFreq();
	std::complex<double> s11min(1e10);
	double f0 = shot_this[ *target()];
	//search for minimum in S11.
	double dfreq = 0.0;
	for(int i = 0; i < trace_len; ++i) {
		if(std::abs(s11min) < std::abs(trace[i])) {
			s11min = trace[i];
			dfreq = trace_start + i * trace_dfreq - f0;
		}
	}
	//S11 at the target frequency.
	std::complex<double> s11atf0;
	for(int i = 0; i < trace_len; ++i) {
		if(trace_start + i * trace_dfreq >= f0) {
			s11atf0 = trace[i];
			break;
		}
	}
	if(std::abs(s11atf0) < TUNE_APPROACH_GOAL) {
		return;
	}
	bool mode = shot_this[ *this].mode;
	std::complex<double> s11; //the value to be tuned to zero.
	switch(mode) {
	case Payload::TUNE_FIRST_TIME:
		mode = tr[ *this].mode = (std::abs(s11min) < TUNE_APPROACH_START) ? TUNE_APPROACH : TUNE_MINIMIZING;
		s11 = (mode == TUNE_APPROACH) ? s11atf0 : s11min;
		tr[ *this].lastS11 = s11;
		tr[ *this].lastDFreq = dfreq;
		tr[ *this].dC1 = 0.0;
		tr[ *this].dC2 = 0.0;
		tr[ *this].hasMeasDerivC1 = false;
		tr[ *this].hasMeasDerivC2 = false;
		break;
	case Payload::TUNE_MINIMIZING:
		s11 = s11min;
		break;
	case Payload::TUNE_APPROACHING:
		s11 = s11atf0;
		break;
	}

	if(hasSlippedC1 && stm1slip && hasSlippedC2 && stm2slip) {
		throw XSkippedRecordError(i18n("Capacitors have sticked"), __FILE__, __LINE__);
	}
	tr[ *this].hasSlippedC1 = shot_this[ *this].hasSlippedC1 || stm1slip;
	tr[ *this].hasSlippedC2 = shot_this[ *this].hasSlippedC2 || stm2slip;
	double drot = (mode == TUNE_APPROACH) ?  TUNE_DROT_APPROACH :  TUNE_DROT_FAST;
	//derivative.
	if((dc1 != 0.0) && (dc2 == 0.0))  {
		tr[ *this].dS11dC1 = stm1slip ? 0.0 : (s11 - shot_this[ *this].lastS11) / dc1;
		tr[ *this].dDFreqdC1 = stm1slip ? 0.0 : (dfreq - shot_this[ *this].lastDFreq) / dc1;
		tr[ *this].hasMeasDerivC1 = true;
	}
	if((dc2 != 0.0) && (dc1 == 0.0))  {
		tr[ *this].dS11dC2 = stm2slip ? 0.0 : (s11 - shot_this[ *this].lastS11) / dc2;
		tr[ *this].dDFreqdC2 = stm2slip ? 0.0 : (dfreq - shot_this[ *this].lastDFreq) / dc2;
		tr[ *this].hasMeasDerivC2 = true;
	}
	if(shot_this[ *this].hasMeasDerivC1 && shot_this[ *this].hasMeasDerivC2) {
		//Derivatives have been obtained.
		double trust = (mode == TUNE_APPROACH) ?  TUNE_TRUST_APPROACH :  TUNE_TRUST_FAST;
		//Solves by real(S11) and imag(S11).
		determineNextC(tr, std::real(s11), std::imag(s11),
			std::real(shot_this[ *this].dS11dC1), std::real(shot_this[ *this].dS11dC2),
			std::imag(shot_this[ *this].dS11dC1), std::imag(shot_this[ *this].dS11dC2),
			trust);
		//Solves by imag(S11) and DFreq.
		determineNextC(tr, std::imag(s11), dfreq,
			std::imag(shot_this[ *this].dS11dC1), std::imag(shot_this[ *this].dS11dC2),
			shot_this[ *this].dFreqdC1, shot_this[ *this].dFreqdC2,
			trust);
		//Solves by real(S11) and DFreq.
		determineNextC(tr, std::real(s11), dfreq,
			std::real(shot_this[ *this].dS11dC1), std::real(shot_this[ *this].dS11dC2),
			shot_this[ *this].dFreqdC1, shot_this[ *this].dFreqdC2,
			trust);
		tr[ *this].mode = TUNE_FIRST_TIME;
		throw XSkippedRecordError(__FILE__, __LINE__);
	}
	//Setting for measuring the derivatives.
	if(shot_this[ *this].hasMeasDerivC1) {
		tr[ *this].dC1 = 0.0;
		tr[ *this].dC2 = drot;
		throw XSkippedRecordError(__FILE__, __LINE__);
	}
	tr[ *this].dC1 = drot;
	tr[ *this].dC2 = 0.0;
	throw XSkippedRecordError(__FILE__, __LINE__);
}
void
visualize() {
	if(shot_this[ *this].dC1 != 0.0) {
		trans( *stm1->target()) += shot_this[ *this].dC1;
	}
	if(shot_this[ *this].dC2 != 0.0) {
		trans( *stm2->target()) += shot_this[ *this].dC2;
	}

}

