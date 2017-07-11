/***************************************************************************
		Copyright (C) 2002-2015 Kentaro Kitagawa
		                   kitagawa@phys.s.u-tokyo.ac.jp

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
#include "graph.h"
#include "autolctuner.h"
#include "ui_autolctunerform.h"

REGISTER_TYPE(XDriverList, AutoLCTuner, "NMR LC autotuner");

#include "nllsfit.h"
#include "rand.h"

//! Series LCR circuit. Additional R in series with a port.
class LCRFit {
public:
    LCRFit(double f0, double rl, bool tight_couple);
    LCRFit(const LCRFit &) = default;
    void fit(const std::vector<double> &s11, double fstart, double fstep, bool randomize = true);
    void computeResidualError(const std::vector<double> &s11, double fstart, double fstep, double omega0, double omega_trust);
    double r1() const {return m_r1;} //!< R of LCR circuit
    double r2() const {return m_r2;} //!< R in series with a port.
    double c1() const {return m_c1;} //!< C of LCR circuit
    double c2() const {return m_c2;} //!< C in parallel to a port.
    void setCaps(double c1, double c2) {
        m_c1 = c1; m_c2 = c2;
    }
    double l1() const {return m_l1;} //!< Fixed value for L.
    double c1err() const {return m_c1_err;}
    double c2err() const {return m_c2_err;}
    bool tightCouple() const {return m_tightCouple;}
    double residualError() const {return m_resErr;}

    //! Resonance freq.
    double f0() const {
        double f = 1.0 / 2 / M_PI / sqrt(l1() * c1());
        for(int it = 0; it < 5; ++it)
            f = 1.0 / 2 / M_PI / sqrt(l1() * c1() -
                c1() * c2() / (1.0/pow(50.0 + r2(), 2.0) + pow(2 * M_PI * f * c2(), 2.0)));
        return f;
    }
    double f0err() const {
        double v = 0.0;
        LCRFit lcr( *this);
        lcr.m_c1 += c1err();
        v += std::norm(lcr.f0() - f0());
        lcr.m_c1 = m_c1;
        lcr.m_c2 += c2err();
        v += std::norm(lcr.f0() - f0());
        return sqrt(v);
    }
    double qValue() const {
        return 2 * M_PI * f0() * l1() / r1();
    }
    //! Reflection
    std::complex<double> rl(double omega, std::complex<double> zlcr1_inv) const {
        auto zL =  1.0 / (std::complex<double>(0.0, omega * c2()) + zlcr1_inv) + r2();
        return (zL - 50.0) / (zL + 50.0);
    }
    std::complex<double> rl(double omega) const {
        return rl(omega, 1.0 / zlcr1(omega));
    }
    double rlerr(double omega) const {
        double v = 0.0;
        LCRFit lcr( *this);
        lcr.m_c1 += c1err();
        v += std::norm(lcr.rl(omega) - rl(omega));
        lcr.m_c1 = m_c1;
        lcr.m_c2 += c2err();
        v += std::norm(lcr.rl(omega) - rl(omega));
        return sqrt(v);
    }
    //! Obtains expected C1 and C2.
    std::pair<double, double> tuneCaps(double target_freq, double target_rl = 0.0) const {
        return tuneCapsInternal(target_freq, target_rl, tightCouple());
    }
    void setTunedCaps(double target_freq, double target_rl = 0.0, bool tight_couple = true) {
        std::tie(m_c1, m_c2) = tuneCapsInternal(target_freq, target_rl, tight_couple);
        m_tightCouple = tight_couple;
    }
private:
    double rlpow(double omega) const {
        return std::abs(rl(omega));
    }
    std::complex<double> zlcr1(double omega) const {
        return std::complex<double>(r1(), omega * l1() - 1.0 / (omega * c1()));
    }
    double isigma(double domega, double omega_trust) const {
//        return 1.0/(pow(rlpow0, 0.5 / POW_ON_FIT) + 0.1) / exp( fabs(domega) / omega_trust / 3);
//        return sqrt(exp( -std::norm(domega / omega_trust) / 2.0) / (sqrt(2 * M_PI)  * omega_trust)); //a weight during the fit.
        return sqrt(1.0 / (M_PI * omega_trust * (1.0 + std::norm(domega / omega_trust)))); //a weight during the fit.
    }
    std::pair<double, double> tuneCapsInternal(double target_freq, double target_rl0, bool tight_couple) const;
    double m_r1, m_r2, m_l1;
    double m_c1, m_c2;
    double m_c1_err, m_c2_err;
    double m_resErr;
    bool m_tightCouple;
};

std::pair<double, double> LCRFit::tuneCapsInternal(double f1, double target_rl, bool tight_couple) const {
    LCRFit nlcr( *this);
    double omega = 2 * M_PI * f1;
    double omegasq = pow(2 * M_PI * f1, 2.0);
    for(int it = 0; it < 100; ++it) {
         //self-consistent eq. to fix resonant freq.
        nlcr.m_c1 = 1.0 / omegasq / (nlcr.l1() - nlcr.c2() /  (1/ pow(50.0 + nlcr.r2(), 2.0) + omegasq * nlcr.c2() * nlcr.c2()));
         //self-consistent eq. to fix rl.
        auto zlcr1_inv = 1.0 / nlcr.zlcr1(omega);
        auto rl1 = nlcr.rl(omega, zlcr1_inv);
        rl1 = target_rl * rl1 / std::abs(rl1); //rearranges |RL|, leaving a phase.
        if((tight_couple ? 1 : -1) * std::real(rl1) < 0)
            rl1 = (tight_couple ? 1 : -1) * target_rl; //forces to select tight or coarse coupling.
        auto target_zlinv = 1.0 / (100.0 / (1.0 - rl1) - 50.0 - r2());
        nlcr.m_c2 = std::imag(target_zlinv - zlcr1_inv) / omega;
    }
    return {nlcr.m_c1, nlcr.m_c2};
}

LCRFit::LCRFit(double init_f0, double init_rl, bool tight_couple) {
    m_l1 = 50.0 / (2.0 * M_PI * init_f0);
    m_c1 = 1.0 / 50.0 / (2.0 * M_PI * init_f0);
    m_c2 = m_c1 * 10;
    double q = 30;
    m_r1 = 2.0 * M_PI * init_f0 * l1() / q;
    m_r2 = 1.0;
    setTunedCaps(init_f0, init_rl, tight_couple);
    double omega = 2 * M_PI * init_f0;
    m_c1_err = m_c1 * 0.1; m_c2_err = m_c2 * 0.1;
//    fprintf(stderr, "Target (%.4g, %.2g) -> (%.4g, %.2g)\n", init_f0, init_rl, f0(), std::abs(rl(omega)));
}

void
LCRFit::computeResidualError(const std::vector<double> &s11, double fstart, double fstep, double omega0, double omega_trust) {
    double x = 0.0;
    double freq = fstart;
    for(size_t i = 0; i < s11.size(); ++i) {
        double omega = 2 * M_PI * freq;
        x += std::norm((s11[i] - rlpow(omega)) * isigma(omega - omega0, omega_trust) * sqrt(2.0 * M_PI * fstep));
        freq += fstep;
    }
    m_resErr = sqrt(x / s11.size());
}

void
LCRFit::fit(const std::vector<double> &s11, double fstart, double fstep, bool randomize) {
    m_resErr = 1.0;
    LCRFit lcr_orig( *this);
    double f0org = lcr_orig.f0();
    double omega0org = 2.0 * M_PI * f0org;
    double rl_orig = std::abs(lcr_orig.rl(omega0org));
    double omega_trust;
    auto eval_omega_trust = [&](double q){
        double omega_avail = 2.0 * M_PI * std::min(f0org - fstart, fstart + fstep * s11.size() - f0org);
        return std::min(omega0org / q * 2, omega_avail / 2);
    };
    double max_q = f0org / fstep;

    auto func = [&s11, this, &fstart, fstep, omega0org, &omega_trust](const double*params, size_t n, size_t p,
            double *f, std::vector<double *> &df) -> bool {
        m_r1 = params[0];
        if(p >= 2) m_c2 = params[1];
        if(p >= 3) m_c1 = fabs(params[2]);
        if(p >= 4) m_r2 = params[3];

        constexpr double DR1 = 1e-2, DR2 = 1e-2, DC1 = 1e-15, DC2 = 1e-15;
        LCRFit plusDR1( *this), plusDR2( *this), plusDC1( *this), plusDC2( *this);
        plusDR1.m_r1 += DR1;
        plusDR2.m_r2 += DR2;
        plusDC1.m_c1 += DC1;
        plusDC2.m_c2 += DC2;
        double freq = fstart;
        for(size_t i = 0; i < n; ++i) {
            double omega = 2 * M_PI * freq;
            double rlpow0 = rlpow(omega);
            double wsqrt = isigma(omega - omega0org, omega_trust) * sqrt(2.0 * M_PI * fstep);
            if(f) {
                f[i] = (rlpow0 - s11[i]) * wsqrt;
            }
            else {
                df[0][i] = (plusDR1.rlpow(omega) - rlpow0) / DR1 * wsqrt;
                if(p >= 2) df[1][i] = (plusDC2.rlpow(omega) - rlpow0) / DC2 * wsqrt;
                if(p >= 3) df[2][i] = (plusDC1.rlpow(omega) - rlpow0) / DC1 * wsqrt;
                if(p >= 4) df[3][i] = (plusDR2.rlpow(omega) - rlpow0) / DR2 * wsqrt;
            }
            freq += fstep;
        }
        return true;
    };

    double omega_trust_scale = 1.5;
    double residualerr = 1.0;
    int it_best;
    NonLinearLeastSquare nlls;
    auto start = XTime::now();
    for(int retry = 0;; retry++) {
        if(retry > 24)
            break;
        if((retry > 8) && (XTime::now() - start > 0.01) && (residualerr < 1e-4))
            break;
//        if(retry > 10)
//            randomize = true;
        if((retry % 2 == 1) && (randomize)) {
            *this = LCRFit(f0org, rl_orig, retry % 4 < 2);
            double q = pow(10.0, (retry % 24) / 24.0 * log10(max_q)) + 2;
            m_r1 = 2.0 * M_PI * f0org * l1() / q;
            omega_trust_scale = (retry % 8) / 8.0 * 2.0 + 0.5;
//            omega_trust = eval_omega_trust(q);
//            computeResidualError(s11, fstart, fstep, omega0org, omega_trust);
//            if(residualError() < residualerr) {
//                residualerr  = residualError();
//                lcr_orig = *this;
//                fprintf(stderr, "Found best tune: rms of residuals = %.3g\n", residualError());
//            }
        }
        if((fabs(r2()) > 10) || (c1() < 0) || (c2() < 0) || (qValue() > max_q) || (qValue() < 2)) {
//            randomize = true; //fitting diverged.
            continue;
        }
        omega_trust = eval_omega_trust(qValue()) * omega_trust_scale;
        if( !(omega_trust > 0))
            continue; //less or nan
        size_t fit_n = s11.size() - 1;
        if(fit_n < 20)
            continue;
        auto nlls1 = NonLinearLeastSquare(func, {m_r1, m_c2, m_c1, m_r2}, fit_n, 200);
        m_r1 = fabs(nlls1.params()[0]);
        m_c2 = nlls1.params()[1];
        m_c1 = nlls1.params()[2];
        m_r2 = nlls1.params()[3];
        m_c2_err = nlls1.errors()[1];
        m_c1_err = nlls1.errors()[2];
        omega_trust = eval_omega_trust(4.0);
        computeResidualError(s11, fstart, fstep, omega0org, omega_trust);
        if(nlls1.isSuccessful() && (residualError() < residualerr) && !std::isnan(f0err())) {
            residualerr  = residualError();
            nlls = std::move(nlls1);
            lcr_orig = *this;
            it_best = retry;
        }
        nlls1 = NonLinearLeastSquare(func, {m_r1}, fit_n);
        m_r1 = fabs(nlls1.params()[0]);
        nlls1 = NonLinearLeastSquare(func, {m_r1, m_c2}, fit_n);
        m_r1 = fabs(nlls1.params()[0]);
        m_c2 = nlls1.params()[1];
    }
    *this = lcr_orig;
    if(nlls.errors().size() == 4) {
        fprintf(stderr, "R1:%.3g+-%.2g, R2:%.3g+-%.2g, L:%.3g, C1:%.3g+-%.2g, C2:%.3g+-%.2g\n",
                r1(), nlls.errors()[0], r2(), nlls.errors()[3], l1(),
                c1(), c1err(), c2(), c2err());
    }
    fprintf(stderr, "rms of residuals = %.3g, elapsed = %f ms & %d iterations.\n",
            residualError(), 1000.0 * (XTime::now() - start), it_best);
}

class XLCRPlot : public XFuncPlot {
public:
    XLCRPlot(const char *name, bool runtime, Transaction &tr, const shared_ptr<XGraph> &graph)
        : XFuncPlot(name, runtime, tr, graph), m_graph(graph)
    {}
    void setLCR(const shared_ptr<LCRFit> &lcr) {
        m_lcr = lcr;
        if(auto graph = m_graph.lock()) {
            Snapshot shot( *graph);
            shot.talk(shot[ *graph].onUpdate(), graph.get());
        }
    }
    virtual double func(double mhz) const {
        if( !m_lcr) return 0.0;
        return 20.0 * log10(std::abs(m_lcr->rl(2.0 * M_PI * mhz * 1e6)));
    }
private:
    shared_ptr<LCRFit> m_lcr;
    weak_ptr<XGraph> m_graph;
};
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
        m_status(create<XStringNode>("FitStatus", true)),
        m_l1(create<XStringNode>("L1", true)),
        m_r1(create<XStringNode>("R1", true)),
        m_r2(create<XStringNode>("R2", true)),
        m_c1(create<XStringNode>("C1", true)),
        m_c2(create<XStringNode>("C2", true)),
        m_form(new FrmAutoLCTuner(g_pFrmMain))  {
	connect(stm1());
	connect(stm2());
	connect(netana());

	m_form->setWindowTitle(i18n("NMR LC autotuner - ") + getLabel() );

    m_form->m_txtStatus->setReadOnly(true);
    m_conUIs = {
        xqcon_create<XQComboBoxConnector>(stm1(), m_form->m_cmbSTM1, ref(tr_meas)),
        xqcon_create<XQComboBoxConnector>(stm2(), m_form->m_cmbSTM2, ref(tr_meas)),
        xqcon_create<XQComboBoxConnector>(netana(), m_form->m_cmbNetAna, ref(tr_meas)),
        xqcon_create<XQLineEditConnector>(target(), m_form->m_edTarget),
        xqcon_create<XQLineEditConnector>(reflectionTargeted(), m_form->m_edReflectionTargeted),
        xqcon_create<XQLineEditConnector>(reflectionRequired(), m_form->m_edReflectionRequired),
        xqcon_create<XQButtonConnector>(m_abortTuning, m_form->m_btnAbortTuning),
        xqcon_create<XQLedConnector>(m_tuning, m_form->m_ledTuning),
        xqcon_create<XQLedConnector>(m_succeeded, m_form->m_ledSucceeded),
        xqcon_create<XQToggleButtonConnector>(m_useSTM1, m_form->m_ckbUseSTM1),
        xqcon_create<XQToggleButtonConnector>(m_useSTM2, m_form->m_ckbUseSTM2),
        xqcon_create<XQLabelConnector>(m_l1, m_form->m_lblL1),
        xqcon_create<XQLabelConnector>(m_r1, m_form->m_lblR1),
        xqcon_create<XQLabelConnector>(m_r2, m_form->m_lblR2),
        xqcon_create<XQLabelConnector>(m_c1, m_form->m_lblC1),
        xqcon_create<XQLabelConnector>(m_c2, m_form->m_lblC2)
    };

	iterate_commit([=](Transaction &tr){
		tr[ *m_tuning] = false;
		tr[ *m_succeeded] = false;
        tr[ *m_reflectionTargeted] = -20.0;
		tr[ *m_reflectionRequired] = -7.0;
		tr[ *m_useSTM1] = true;
		tr[ *m_useSTM2] = true;
        tr[ *abortTuning()].setUIEnabled(false);
        m_lsnOnTargetChanged = tr[ *m_target].onValueChanged().connectWeakly(
			shared_from_this(), &XAutoLCTuner::onTargetChanged);
		m_lsnOnAbortTouched = tr[ *m_abortTuning].onTouch().connectWeakly(
			shared_from_this(), &XAutoLCTuner::onAbortTuningTouched);
        m_lsnOnStatusOut = tr[ *m_status].onValueChanged().connectWeakly(
                    shared_from_this(), &XAutoLCTuner::onStatusChanged, Listener::FLAG_MAIN_THREAD_CALL);
    });
}
void
XAutoLCTuner::onStatusChanged(const Snapshot &shot, XValueNodeBase *) {
    if(shot[ *m_status].to_str().empty())
        m_form->m_txtStatus->clear();
    else
        m_form->m_txtStatus->append(shot[ *m_status].to_str());
}

XAutoLCTuner::~XAutoLCTuner() {
    iterate_commit([=](Transaction &tr){
        clearUIAndPlot(tr);
    });
}
void XAutoLCTuner::showForms() {
    m_form->resize(100,100); //avoids bug on Windows.
    m_form->showNormal();
	m_form->raise();
}
void XAutoLCTuner::onTargetChanged(const Snapshot &shot, XValueNodeBase *node) {
	Snapshot shot_this( *this);
	shared_ptr<XMotorDriver> stm1__ = shot_this[ *stm1()];
	shared_ptr<XMotorDriver> stm2__ = shot_this[ *stm2()];
    const shared_ptr<XMotorDriver> stms[] = {stm1__, stm2__};
    const unsigned int tunebits = 0xffu;
    for(auto &&stm: stms) {
        if(stm) {
            stm->iterate_commit([=](Transaction &tr){
                tr[ *stm->active()] = true; // Activate motor.
                tr[ *stm->auxBits()] = tunebits; //For external RF relays.
            });
        }
    }

    iterate_commit([=](Transaction &tr){
        clearUIAndPlot(tr);

        tr[ *m_tuning] = true;
		tr[ *succeeded()] = false;
        tr[ *m_status] = "";

        tr[ *m_useSTM1].setUIEnabled(false);
        tr[ *m_useSTM2].setUIEnabled(false);
        tr[ *stm1()].setUIEnabled(false);
        tr[ *stm2()].setUIEnabled(false);
        tr[ *abortTuning()].setUIEnabled(true);

        tr[ *this].resetToFirstStage();
        tr[ *this].smallestRLAtF0 = 1.0;
        tr[ *this].iterationCount = 0;
		tr[ *this].isSTMChanged = true;
		tr[ *this].started = XTime::now();
		tr[ *this].isTargetAbondoned = false;
    });

    shared_ptr<XNetworkAnalyzer> na__ = shot_this[ *netana()];
    if(na__)
        na__->graph()->iterate_commit([=](Transaction &tr){
        m_lcrPlot = na__->graph()->plots()->create<XLCRPlot>(
            tr, "FittedCurve", true, tr, na__->graph());
        tr[ *m_lcrPlot->label()] = i18n("Fitted Curve");
        tr[ *m_lcrPlot->axisX()] = tr.list(na__->graph()->axes())->at(0);
        tr[ *m_lcrPlot->axisY()] = tr.list(na__->graph()->axes())->at(1);
        tr[ *m_lcrPlot->lineColor()] = (unsigned int)tr[ *na__->graph()->titleColor()];
        tr[ *m_lcrPlot->intensity()] = 2.0;
    });
}
void XAutoLCTuner::onAbortTuningTouched(const Snapshot &shot, XTouchableNode *) {
    iterate_commit_while([=](Transaction &tr)->bool{
		if( !tr[ *m_tuning])
            return false;
        clearUIAndPlot(tr);
        tr[ *m_tuning] = false;
		tr[ *this].isSTMChanged = false;
        return true;
    });
}

void
XAutoLCTuner::rollBack(Transaction &tr, XString &&message) {
    if(20.0 * log10(tr[ *this].smallestRLAtF0) < -3.0) {
        abortTuningFromAnalyze(tr, 0.0, std::move(message));
    }
    if((tr[ *this].iterationCount == 0) || (tr[ *this].iterationCount > 5)) {
        //The first fitting has failed, or iteration exceeds a limit.
        abortTuningFromAnalyze(tr, 1.0, std::move(message));
    }
    tr[ *m_status] = message + "Rolls back.";
    //rolls back to good positions.
	tr[ *this].isSTMChanged = true;
    tr[ *this].targetSTMValues = tr[ *this].bestSTMValues;
    tr[ *this].smallestRLAtF0 = 1; //resets the memory.
    tr[ *this].resetToFirstStage();
    throw XSkippedRecordError(__FILE__, __LINE__);
}
void
XAutoLCTuner::clearUIAndPlot(Transaction &tr) {
    tr[ *m_useSTM1].setUIEnabled(true);
    tr[ *m_useSTM2].setUIEnabled(true);
    tr[ *stm1()].setUIEnabled(true);
    tr[ *stm2()].setUIEnabled(true);
    tr[ *abortTuning()].setUIEnabled(false);

    const shared_ptr<XNetworkAnalyzer> na__ = tr[ *netana()];
    auto plot = m_lcrPlot;
    if(na__ && plot) {
        try {
            na__->graph()->plots()->release(plot);
        }
        catch (NodeNotFoundError &) {
        }
    }
    m_lcrPlot.reset();
}
void
XAutoLCTuner::abortTuningFromAnalyze(Transaction &tr, double rl_at_f0, XString &&message) {
    message += "\n";
	double tune_approach_goal2 = pow(10.0, 0.05 * tr[ *reflectionRequired()]);
    if((tune_approach_goal2 > tr[ *this].smallestRLAtF0) && !tr[ *this].isTargetAbondoned) {
        message += "Softens target value.\n";
        tr[ *this].resetToFirstStage();
        tr[ *this].smallestRLAtF0 = 1.0;
        tr[ *this].iterationCount = 0;
        tr[ *this].isSTMChanged = true;
        tr[ *this].started = XTime::now();
        tr[ *this].isTargetAbondoned = true;
        rollBack(tr, std::move(message)); //rolls back and skps.
	}
    clearUIAndPlot(tr);
	tr[ *m_tuning] = false;
    tr[ *m_status] = message + "Abort.";
    if(rl_at_f0 > tr[ *this].smallestRLAtF0) {
		tr[ *this].isSTMChanged = true;
        tr[ *this].targetSTMValues = tr[ *this].bestSTMValues;
        throw XRecordError(i18n("Aborting. Out of tune, or capacitors have sticked. Back to better positions."), __FILE__, __LINE__);
	}
	throw XRecordError(i18n("Aborting. Out of tune, or capacitors have sticked."), __FILE__, __LINE__);
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
    if(shot_others[ *stm1__].time() > shot_emitter[ *na__].timeAwared())
        return false; //trace acquired before motor starts.
    if(shot_others[ *stm2__].time() > shot_emitter[ *na__].timeAwared())
        return false; //trace acquired before motor starts.
    return true;
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
        tr[ *this].targetSTMValues[0] = shot_others[ *stm1__->position()->value()];
    if(stm2__)
        tr[ *this].targetSTMValues[1] = shot_others[ *stm2__->position()->value()];

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

    XString message;

	int trace_len = shot_na[ *na__].length();
    double trace_dfreq = shot_na[ *na__].freqInterval();
	double trace_start = shot_na[ *na__].startFreq();
    double fmin = 0.0, fmin_err;
    double rlmin = 1e10;
	double f0 = shot_this[ *target()];
    double rl_at_f0 = 0.0;
    double rl_at_f0_sigma;
    shared_ptr<LCRFit> lcrfit;
	//analyzes trace.
    {
        const std::complex<double> *trace = &shot_na[ *na__].trace()[0];
        double rl_eval_min = 1e10;
        //searches for minimum in reflection around f0.
        for(int i = 0; i < trace_len; ++i) {
            double z = std::abs(trace[i]);
            double f = trace_start + i * trace_dfreq;
            double y = 1.0 - ((1.0 - z) * exp( -std::norm((f0 - f) / f0)) / (2.0 * 1.0 * 1.0));
            if(y < rl_eval_min) {
                rl_eval_min = y;
                rlmin = z;
                fmin = f;
            }
        }
//        message +=
//            formatString("Before Fit: fmin=%.4f MHz, RL=%.2f dB\n",
//                fmin, 20.0 * log10(rlmin));

        //Fits to LCR circuit.
        std::vector<double> rl(trace_len);
        for(int i = 0; i < trace_len; ++i)
            rl[i] = std::abs(trace[i]);

        if(shot_this[ *this].fitOrig)
            lcrfit = std::make_shared<LCRFit>( *shot_this[ *this].fitOrig);
        else
            lcrfit = std::make_shared<LCRFit>(fmin * 1e6, rlmin, false);
        lcrfit->setTunedCaps(fmin * 1e6, rlmin);
        lcrfit->fit(rl, trace_start * 1e6, trace_dfreq * 1e6, !shot_this[ *this].fitOrig);
        double fmin_fit = lcrfit->f0() * 1e-6;
        double fmin_fit_err = lcrfit->f0err() * 1e-6;
        double rlmin_fit = std::abs(lcrfit->rl(2.0 * M_PI * lcrfit->f0()));
        rl_at_f0 = std::abs(lcrfit->rl(2.0 * M_PI * f0 * 1e6));
        rl_at_f0_sigma = lcrfit->rlerr(2.0 * M_PI * f0 * 1e6);
        message +=
            formatString("Fit: fres=%.4f+-%.4f MHz, RL=%.2f dB, Q=%.2g, %s",
                fmin_fit, fmin_fit_err,
                log10(rlmin_fit) * 20.0, lcrfit->qValue(), lcrfit->tightCouple() ? "Tight" : "Loose");
        message +=
            formatString(", C1=%.2f+-%.2f pF, C2=%.2f+-%.2f pF\n",
                         lcrfit->c1() * 1e12, lcrfit->c1err() * 1e12, lcrfit->c2() * 1e12, lcrfit->c2err() * 1e12);
        tr[ *m_l1].str(formatString("L1=%.2f nH", lcrfit->l1() * 1e9));
        tr[ *m_r1].str(formatString("R1=%.2f Ohm", lcrfit->r1()));
        tr[ *m_r2].str(formatString("R2=%.2f Ohm", lcrfit->r2()));
        tr[ *m_c1].str(formatString("C1=%.2f\n  +-%.2f pF", lcrfit->c1() * 1e12, lcrfit->c1err() * 1e12));
        tr[ *m_c2].str(formatString("C2=%.2f\n  +-%.2f pF", lcrfit->c2() * 1e12, lcrfit->c2err() * 1e12));
        auto newcaps = lcrfit->tuneCaps(f0 * 1e6);
        message +=
            formatString("Fit suggests: C1=%.2f pF, C2=%.2f pF\n", newcaps.first * 1e12, newcaps.second * 1e12);
        if(auto plot = m_lcrPlot)
            plot->setLCR(lcrfit);

        if((lcrfit->residualError() > 0.1) || std::isnan(fmin_fit_err)
            || (fabs(fmin - fmin_fit) > (fmin_fit_err + trace_dfreq) * 2) || (fabs(rlmin - rlmin_fit) > 0.2)) {
            message += formatString("Fitting is not reliable, because searched minimum was (%.2f MHz, %.2f dB).\n",
                fmin, 20.0 * log10(rlmin));
            rollBack(tr, std::move(message));
        }
        fmin = fmin_fit;
        fmin_err = fmin_fit_err;
        rlmin = rlmin_fit;
    }

	double tune_approach_goal = pow(10.0, 0.05 * shot_this[ *reflectionTargeted()]);
	if(shot_this[ *this].isTargetAbondoned)
		tune_approach_goal = pow(10.0, 0.05 * shot_this[ *reflectionRequired()]);
    if(rl_at_f0 + rl_at_f0_sigma < tune_approach_goal) {
        tr[ *m_status] = message + "Tuning done satisfactorily.";
		tr[ *succeeded()] = true;
		return;
	}

    if(shot_this[ *this].smallestRLAtF0 > rl_at_f0 + rl_at_f0_sigma) {
        tr[ *this].iterationCount = 0;
		//remembers good positions.
        tr[ *this].bestSTMValues = tr[ *this].targetSTMValues;
        tr[ *this].smallestRLAtF0 = rl_at_f0 + rl_at_f0_sigma;
	}

	bool timeout = (XTime::now() - shot_this[ *this].started > 360); //6min.
	if(timeout) {
        message += "Time out.";
        abortTuningFromAnalyze(tr, rl_at_f0, std::move(message));//Aborts.
		return;
	}

    int target_stm = ((shot_this[ *this].deltaC1perDeltaSTM[0] == 0.0) && stm1__) ? 0 : 1;

    if( !shot_this[ *this].fitOrig) {
    //The stage just before +Delta rotation.
        tr[ *this].iterationCount++;
        message += formatString("Iteration %d.\n", tr[ *this].iterationCount);
        if((shot_this[ *this].iterationCount > 2) && (rl_at_f0 - rl_at_f0_sigma > shot_this[ *this].smallestRLAtF0)) {
            message += "The last 2 iterations made situation worse.";
            rollBack(tr, std::move(message));
        }
        tr[ *this].fitOrig = lcrfit;
        //follows the last rotation direction.
        tr[ *this].stmDelta[target_stm] =
                Payload::TestDeltaFirst * shot_this[ *this].lastDirection(target_stm);
        tr[ *this].targetSTMValues[target_stm] += shot_this[ *this].stmDelta[target_stm];
        tr[ *this].isSTMChanged = true;
        tr[ *m_status] = message + formatString("STM%d: Testing +Delta.", target_stm + 1);;
        throw XSkippedRecordError(__FILE__, __LINE__);
    }
    else {
        if( !shot_this[ *this].fitRotated) {
        //The stage just after +Delta rotation.
            tr[ *this].fitRotated = lcrfit;
            //follows the last rotation direction.
            tr[ *this].stmDelta[target_stm] *= -1.0; //opposite direction.
            tr[ *this].targetSTMValues[target_stm] += shot_this[ *this].stmDelta[target_stm];
            tr[ *this].isSTMChanged = true;
            tr[ *m_status] = message + formatString("STM%d: Testing -Delta.", target_stm + 1);;
            throw XSkippedRecordError(__FILE__, __LINE__);
        }
        else {
        //The stage just after -Delta rotation.
            //calculate capacitance change and backlash.
            double testdelta = shot_this[ *this].stmDelta[target_stm];
            testdelta *= -1; //polarity for +Delta
            double dc1dtest = (shot_this[ *this].fitRotated->c1() - shot_this[ *this].fitOrig->c1()) / testdelta;
            double dc1dtest_err = sqrt(pow(shot_this[ *this].fitRotated->c1err(), 2.0) + pow(lcrfit->c1err(), 2.0)
                    + pow(shot_this[ *this].fitOrig->c1err(), 2.0)) / fabs(testdelta);
            double dc2dtest = (shot_this[ *this].fitRotated->c2() - shot_this[ *this].fitOrig->c2()) / testdelta;
            double dc2dtest_err = sqrt(pow(shot_this[ *this].fitRotated->c2err(), 2.0) + pow(lcrfit->c2err(), 2.0)
                    + pow(shot_this[ *this].fitOrig->c2err(), 2.0)) / fabs(testdelta);
            double dc1dtest_minus_backlash =
                (lcrfit->c1() - shot_this[ *this].fitRotated->c1()) / (-testdelta);
            double dc2dtest_minus_backlash =
                (lcrfit->c2() - shot_this[ *this].fitRotated->c2()) / (-testdelta);
            double backlash1 = testdelta - dc1dtest_minus_backlash / dc1dtest * testdelta;
            double backlash2 = testdelta - dc2dtest_minus_backlash / dc2dtest * testdelta;
            double w1 = dc2dtest_err / fabs(dc2dtest);
            double w2 = dc1dtest_err / fabs(dc1dtest);
            double backlash = w1 * w1 * backlash1 + w2 * w2 * backlash2;
            backlash /= w1 * w1 + w2 * w2;
            message +=
                formatString("STM%d: dC1/dx = %.2g+-%.2g pF/deg., dC2/dx = %.2g+-%.2g pF/deg., backlash = %.1f deg.\n",
                    target_stm + 1, dc1dtest * 1e12, dc1dtest_err * 1e12, dc2dtest * 1e12, dc2dtest_err * 1e12, backlash);
            double sigma_per_change = std::min(w1, w2);
            if((sigma_per_change > 5) ||
                    (backlash / fabs(testdelta) < -0.2) || (fabs(backlash / testdelta) > 0.3)) {
            //Capacitance is sticking, test angle is too small, or poor fitting.
                testdelta *= std::min(4L, 1L + lrint(5.0 / sigma_per_change));
                testdelta *= -1; //polarity for +Delta
               if(fabs(testdelta) > Payload::TestDeltaMax) {
                   abortTuningFromAnalyze(tr, rl_at_f0, std::move(message));//C1/C2 is useless. Aborts.
                   return;
               }
               message +=
                    formatString("Increasing test angle to %.1f, Testing +Delta.", (double)testdelta);
               //rotates C more and try again.
               tr[ *this].fitOrig = lcrfit;
               tr[ *this].fitRotated.reset();
               tr[ *this].stmDelta[target_stm] = fabs(testdelta) * shot_this[ *this].lastDirection(target_stm);
               tr[ *this].targetSTMValues[target_stm] += shot_this[ *this].stmDelta[target_stm];
               tr[ *this].isSTMChanged = true;
               tr[ *m_status] = message;
               throw XSkippedRecordError(__FILE__, __LINE__);
            }
            if(backlash < 0) backlash = 0; //unphysical
            tr[ *this].stmBacklash[target_stm] = backlash;
            tr[ *this].stmTrustArea[target_stm] =
                std::min(fabs(testdelta) * std::min(fabs(testdelta) / backlash, 50.0), Payload::DeltaMax);
            tr[ *this].deltaC1perDeltaSTM[target_stm] = dc1dtest;
            tr[ *this].deltaC2perDeltaSTM[target_stm] = dc2dtest;
            tr[ *this].clearSTMDelta();
            if(stm1__ && stm2__) {
                target_stm = 1 - target_stm; //STM1
                if(tr[ *this].deltaC1perDeltaSTM[target_stm] == 0.0) {
                    //go to next test for another STM.
                    tr[ *this].fitOrig = lcrfit;
                    tr[ *this].fitRotated.reset();
                    //follows the last rotation direction.
                    tr[ *this].stmDelta[target_stm] =
                            Payload::TestDeltaFirst * shot_this[ *this].lastDirection(target_stm);
                    tr[ *this].targetSTMValues[target_stm] += shot_this[ *this].stmDelta[target_stm];
                    tr[ *this].isSTMChanged = true;
                    tr[ *m_status] = message + formatString("STM%d: Testing +Delta", target_stm + 1);
                    throw XSkippedRecordError(__FILE__, __LINE__);
                }
            }
        }
    }
    //Final stage.
    std::array<double, 2> drot{{0.0, 0.0}}; //rotations.
    double c1, c2;
    std::tie(c1, c2) = lcrfit->tuneCaps(f0 * 1e6); //suggests capacitances.
    message +=
        formatString("Fit suggests: C1=%.2f pF, C2=%.2f pF\n", c1 * 1e12, c2 * 1e12);
    double dc1 = c1 - lcrfit->c1();
    double dc2 = c2 - lcrfit->c2();
    if(stm1__ && stm2__) {
        double a,b,c,d;
        a = shot_this[ *this].deltaC1perDeltaSTM[0];
        b = shot_this[ *this].deltaC1perDeltaSTM[1];
        c = shot_this[ *this].deltaC2perDeltaSTM[0];
        d = shot_this[ *this].deltaC2perDeltaSTM[1];
        double det = a*d - b*c;
        drot[0] = (d * dc1 - b * dc2) / det;
        drot[1] = (-c * dc1 + a * dc2) / det;
    }
    else {
        double minrl = 1.0;
        double k1 = dc1 / shot_this[ *this].deltaC1perDeltaSTM[target_stm];
        double k2 = dc2 / shot_this[ *this].deltaC2perDeltaSTM[target_stm];
        LCRFit newlcr( *lcrfit);
        c1 = newlcr.c1();
        c2 = newlcr.c2();
        //searches for minimum.
        for(double coeff = -1; coeff < 2; coeff += 0.1) {
            double k = k1 * coeff + k2 * (1.0 - coeff);
            newlcr.setCaps(c1 + k *  shot_this[ *this].deltaC1perDeltaSTM[target_stm],
                    c2 + k * shot_this[ *this].deltaC2perDeltaSTM[target_stm]);
            double rl = std::abs(newlcr.rl(2.0 * M_PI * f0 * 1e6));
            if(minrl > rl) {
                minrl = rl;
                drot[target_stm] = k;
            }
        }
    }
    message += formatString("Suggested pos: %.1f, %.1f; rot.: %.1f, %.1f\n",
        shot_this[ *this].targetSTMValues[0] + drot[0], shot_this[ *this].targetSTMValues[1]  + drot[1],
        drot[0], drot[1]);
    //Subtracts backlashes
    for(int i: {0, 1}) {
        if(shot_this[ *this].lastDirection(i) * drot[i] < 0)
            drot[i] -= shot_this[ *this].lastDirection(i) * shot_this[ *this].stmBacklash[i];
    }
    //Limits rotations.
    double rotpertrust = fabs(std::max(fabs(drot[0]) / shot_this[*this].stmTrustArea[0],
            fabs(drot[1]) / shot_this[*this].stmTrustArea[1]));
    if(rotpertrust > 1.0)
        for(auto &&dx: drot)
            dx /= rotpertrust;
    for(int i: {0, 1}) {
        tr[ *this].stmDelta[i] = drot[i];
        tr[ *this].targetSTMValues[i] += drot[i];
    }
    tr[ *this].resetToFirstStage();
    tr[ *this].isSTMChanged = true;

    tr[ *m_status] = message;
    throw XSkippedRecordError(__FILE__, __LINE__);
}
void
XAutoLCTuner::visualize(const Snapshot &shot_this) {
	const shared_ptr<XMotorDriver> stm1__ = shot_this[ *stm1()];
	const shared_ptr<XMotorDriver> stm2__ = shot_this[ *stm2()];
    const shared_ptr<XMotorDriver> stms[] = {stm1__, stm2__};
    if(shot_this[ *tuning()]) {
		if(shot_this[ *succeeded()]){
			const unsigned int tunebits = 0;
            for(auto &&stm: stms) {
                if(stm) {
                    stm->iterate_commit([=](Transaction &tr){
                        tr[ *stm->active()] = false; //Deactivates motor.
                        tr[ *stm->auxBits()] = tunebits; //For external RF relays.
                    });
                }
			}
			msecsleep(50); //waits for relays.
            iterate_commit([=](Transaction &tr){
                tr[ *tuning()] = false;//finishes tuning successfully.
                clearUIAndPlot(tr);
            });
        }
	}
	if(shot_this[ *this].isSTMChanged) {
        for(int i: {0, 1}) {
            auto stm = stms[i];
            if(stm) {
                double drot = shot_this[ *this].targetSTMValues[i] - Snapshot( *stm)[ *stm->position()->value()];
                if(fabs(drot) > 0.5) {
                    trans( *m_status) =
                            formatString("STM%d += %.1f deg.", i + 1, drot);
                    stm->iterate_commit([=](Transaction &tr){
                        tr[ *stm->target()] = shot_this[ *this].targetSTMValues[i];
                    });
                }
            }
        }
		msecsleep(50); //waits for ready indicators.
		if( !shot_this[ *tuning()]) {
			trans( *this).isSTMChanged = false;
		}
	}
}

