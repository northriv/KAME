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
#include "pulserdriver.h"
#include "ui_pulserdriverform.h"
#include "ui_pulserdrivermoreform.h"
#include "pulserdriverconnector.h"
#include "interface.h"
#include "analyzer.h"
#include "xnodeconnector.h"
#include <QStatusBar>

#include <gsl/gsl_sf.h>
#define bessel_i0 gsl_sf_bessel_I0

#include <set>

struct __PulseFunc {
	const char *name;
	FFT::twindowfunc fp;
};
const __PulseFunc cg_PulseFuncs[] = {
{"Rect. BW=0.89/T", &FFT::windowFuncRect}, {"Hanning BW=1.44/T", &FFT::windowFuncHanning},
{"Hamming BW=1.30/T", &FFT::windowFuncHamming}, {"Flat-Top BW=3.2/T", &FFT::windowFuncFlatTop},
{"Flat-Top BW=5.3/T", &FFT::windowFuncFlatTopLong}, {"Flat-Top BW=6.8/T", &FFT::windowFuncFlatTopLongLong},
{"Blackman BW=1.7/T", &FFT::windowFuncBlackman}, {"Blackman-Harris BW=1.9/T", &FFT::windowFuncBlackmanHarris},
{"Kaiser(3) BW=1.6/T", &FFT::windowFuncKaiser1}, {"Kaiser(7.2) BW=2.6/T", &FFT::windowFuncKaiser2},
{"Kaiser(15) BW=3.8/T", &FFT::windowFuncKaiser3}, {"Half-sin BW=1.2/T", &FFT::windowFuncHalfSin},
{"", 0}
};
#define PULSE_NO_HAMMING 2
#define PULSE_NO_FLATTOP_LONG_LONG 5

XPulser::tpulsefunc
XPulser::pulseFunc(int no) const {
	int idx = 0;
	for(const __PulseFunc *f = cg_PulseFuncs; f->fp; ++f) {
		if(idx == no)
			return f->fp;
		++idx;
	}
	return &FFT::windowFuncRect;
}
int
XPulser::pulseFuncNo(const XString &str) const {
	int idx = 0;
	for(const __PulseFunc *f = cg_PulseFuncs; f->fp; ++f) {
		if(f->name == str)
			return idx;
		++idx;
	}
	return 0;
}

#define NUM_PHASE_CYCLE_1 "1"
#define NUM_PHASE_CYCLE_2 "2"
#define NUM_PHASE_CYCLE_4 "4"
#define NUM_PHASE_CYCLE_8 "8"
#define NUM_PHASE_CYCLE_16 "16"

#define COMB_MODE_OFF "Comb Off"
#define COMB_MODE_ON "Comb On"
#define COMB_MODE_P1_ALT "P1 ALT"
#define COMB_MODE_COMB_ALT "Comb ALT"

#define RT_MODE_FIXREP "Fix Rep. Time"
#define RT_MODE_FIXREST "Fix Rest Time"

XPulser::XPulser(const char *name, bool runtime, 
	Transaction &tr_meas, const shared_ptr<XMeasure> &meas) :
    XPrimaryDriver(name, runtime, ref(tr_meas), meas),
    m_output(create<XBoolNode>("Output", true)),
    m_combMode(create<XComboNode>("CombMode", false, true)),
    m_rtMode(create<XComboNode>("RTMode", false, true)),
    m_rt(create<XDoubleNode>("RT", false, "%.7g")),
    m_tau(create<XDoubleNode>("Tau", false, "%.7g")),
    m_combPW(create<XDoubleNode>("CombPW", false)),
    m_pw1(create<XDoubleNode>("PW1", false)),
    m_pw2(create<XDoubleNode>("PW2", false)),
    m_combNum(create<XUIntNode>("CombNum", false)),
    m_combPT(create<XDoubleNode>("CombPT", false)),
    m_combP1(create<XDoubleNode>("CombP1", false, "%.7g")),
    m_combP1Alt(create<XDoubleNode>("CombP1Alt", false, "%.7g")),
    m_aswSetup(create<XDoubleNode>("ASWSetup", false)),
    m_aswHold(create<XDoubleNode>("ASWHold", false)),
    m_altSep(create<XDoubleNode>("ALTSep", false)),
    m_g2Setup(create<XDoubleNode>("Gate2Setup", false)),
    m_echoNum(create<XUIntNode>("EchoNum", false)),
    m_combOffRes(create<XDoubleNode>("CombOffRes", false)),
    m_drivenEquilibrium(create<XBoolNode>("DrivenEquilibrium", false)),
    m_numPhaseCycle(create<XComboNode>("NumPhaseCycle", false, true)),
    m_p1Func(create<XComboNode>("P1Func", false, true)),
    m_p2Func(create<XComboNode>("P2Func", false, true)),
    m_combFunc(create<XComboNode>("CombFunc", false, true)),
    m_p1Level(create<XDoubleNode>("P1Level", false)),
    m_p2Level(create<XDoubleNode>("P2Level", false)),
    m_combLevel(create<XDoubleNode>("CombLevel", false)),
    m_masterLevel(create<XDoubleNode>("MasterLevel", false)),
    m_qamOffset1(create<XDoubleNode>("QAMOffset1", false)),
    m_qamOffset2(create<XDoubleNode>("QAMOffset2", false)),
    m_qamLevel1(create<XDoubleNode>("QAMLevel1", false)),
    m_qamLevel2(create<XDoubleNode>("QAMLevel2", false)),
    m_qamDelay1(create<XDoubleNode>("QAMDelay1", false)),
    m_qamDelay2(create<XDoubleNode>("QAMDelay2", false)),
    m_difFreq(create<XDoubleNode>("DIFFreq", false)),
    m_induceEmission(create<XBoolNode>("InduceEmission", false)),
    m_induceEmissionPhase(create<XDoubleNode>("InduceEmissionPhase", false)),
    m_qswDelay(create<XDoubleNode>("QSWDelay", false)),
    m_qswWidth(create<XDoubleNode>("QSWWidth", false)),
    m_qswSoftSWOff(create<XDoubleNode>("QSWSoftSWOff", false)),
    m_invertPhase(create<XBoolNode>("InvertPhase", false)),
    m_conserveStEPhase(create<XBoolNode>("ConserveStEPhase", false)),
    m_qswPiPulseOnly(create<XBoolNode>("QSWPiPulseOnly", false)),
    m_pulseAnalyzerMode(create<XBoolNode>("PulseAnalyzerMode", true)),
    m_paPulseRept(create<XDoubleNode>("PAPulseRept", false)),
    m_paPulseBW(create<XDoubleNode>("PAPulseBW", false)),
    m_firstPhase(create<XUIntNode>("FirstPhase", true)),
    m_moreConfigShow(create<XTouchableNode>("MoreConfigShow", true)),
    m_form(new FrmPulser),
    m_formMore(new FrmPulserMore(m_form.get())) {

	m_form->setWindowTitle(i18n("Pulser Control") + " - " + getLabel() );
	m_formMore->setWindowTitle(i18n("Pulser Control More Config.") + " - " + getLabel() );
	m_form->statusBar()->hide();
	m_formMore->statusBar()->hide();
  
    m_form->m_btnMoreConfig->setIcon(QApplication::style()->standardIcon(QStyle::SP_FileDialogContentsView));

    Snapshot shot = iterate_commit([=](Transaction &tr){
		const Snapshot &shot(tr);
		{
			for(unsigned int i = 0; i < NUM_DO_PORTS; i++) {
				m_portSel[i] = create<XComboNode>(tr, formatString("PortSel%u", i).c_str(), false);
                tr[ *m_portSel[i]].add({
                       "Gate", "PreGate", "Gate3", "Trig1", "Trig2", "ASW",
                       "QSW", "Pulse1", "Pulse2", "Comb", "CombFM",
                       "QPSK-A", "QPSK-B", "QPSK-NonInv", "QPSK-Inv",
                       "QPSK-PS-Gate", "PAnaGate"
                });
	//			m_portSel[i]->setUIEnabled(false);
			}
			tr[ *portSel(0)] = PORTSEL_GATE;
			tr[ *portSel(1)] = PORTSEL_PREGATE;
            tr[ *portSel(3)] = PORTSEL_QPSK_A;
            tr[ *portSel(4)] = PORTSEL_QPSK_B;
            tr[ *portSel(6)] = PORTSEL_TRIG1;
            tr[ *portSel(9)] = PORTSEL_ASW;
        }

	    tr[ *rtime()] = 100.0;
	    tr[ *tau()] = 100.0;
	    tr[ *combPW()] = 5.0;
	    tr[ *pw1()] = 5.0;
	    tr[ *pw2()] = 10.0;
	    tr[ *combNum()] = 1;
	    tr[ *combPT()] = 20.0;
	    tr[ *combP1()] = 0.2;
	    tr[ *combP1Alt()] = 50.0;
	    tr[ *aswSetup()] = 0.02;
	    tr[ *aswHold()] = 0.23;
	    tr[ *altSep()] = 0.25;
	    tr[ *g2Setup()] = 5.0;
	    tr[ *echoNum()] = 1;
	    tr[ *combOffRes()] = 0.0;
	    tr[ *drivenEquilibrium()] = false;
        tr[ *numPhaseCycle()].add({NUM_PHASE_CYCLE_1, NUM_PHASE_CYCLE_2, NUM_PHASE_CYCLE_4, NUM_PHASE_CYCLE_8, NUM_PHASE_CYCLE_16});
	    tr[ *numPhaseCycle()] = NUM_PHASE_CYCLE_16;
	    tr[ *p1Level()] = -5.0;
	    tr[ *p2Level()] = -0.5;
	    tr[ *combLevel()] = -5.0;
	    tr[ *masterLevel()] = -10.0;
	    tr[ *qamLevel1()] = 1.0;
	    tr[ *qamLevel2()] = 1.0;
	    tr[ *qswDelay()] = 5.0;
	    tr[ *qswWidth()] = 10.0;
	    tr[ *qswSoftSWOff()] = 1.0;

        tr[ *combMode()].add({COMB_MODE_OFF ,COMB_MODE_ON, COMB_MODE_P1_ALT, COMB_MODE_COMB_ALT});
	    tr[ *combMode()] = N_COMB_MODE_OFF;
        tr[ *rtMode()].add({RT_MODE_FIXREP, RT_MODE_FIXREST});
	    tr[ *rtMode()] = 1;

		for(const __PulseFunc *f = cg_PulseFuncs; f->fp; ++f) {
		    tr[ *p1Func()].add(f->name);
		    tr[ *p2Func()].add(f->name);
		    tr[ *combFunc()].add(f->name);
		}
	    tr[ *p1Func()] = PULSE_NO_HAMMING; //Hamming
	    tr[ *p2Func()] = PULSE_NO_HAMMING; //Hamming
	    tr[ *combFunc()] = PULSE_NO_HAMMING; //Hamming

	    tr[ *paPulseRept()] = 1; //ms
	    tr[ *paPulseBW()] = 1000.0; //kHz

        tr[ *firstPhase()] = 0;

		m_lsnOnMoreConfigShow = tr[ *m_moreConfigShow].onTouch().connectWeakly(
			shared_from_this(), &XPulser::onMoreConfigShow,
			XListener::FLAG_MAIN_THREAD_CALL | XListener::FLAG_AVOID_DUP);
    });

    m_conUIs = {
        xqcon_create<XQButtonConnector>(m_moreConfigShow, m_form->m_btnMoreConfig),
        xqcon_create<XQToggleButtonConnector>(m_output, m_form->m_ckbOutput),
        xqcon_create<XQComboBoxConnector>(m_combMode, m_form->m_cmbCombMode, Snapshot( *m_combMode)),
        xqcon_create<XQComboBoxConnector>(m_rtMode, m_form->m_cmbRTMode, Snapshot( *m_rtMode)),
        xqcon_create<XQLineEditConnector>(m_rt, m_form->m_edRT),
        xqcon_create<XQLineEditConnector>(m_tau, m_form->m_edTau),
        xqcon_create<XQLineEditConnector>(m_combPW, m_form->m_edCombPW),
        xqcon_create<XQLineEditConnector>(m_pw1, m_form->m_edPW1),
        xqcon_create<XQLineEditConnector>(m_pw2, m_form->m_edPW2),
        xqcon_create<XQSpinBoxUnsignedConnector>(m_combNum, m_form->m_numCombNum),
        xqcon_create<XQLineEditConnector>(m_combPT, m_form->m_edCombPT),
        xqcon_create<XQLineEditConnector>(m_combP1, m_form->m_edCombP1),
        xqcon_create<XQLineEditConnector>(m_combP1Alt, m_form->m_edCombP1Alt),
        xqcon_create<XQLineEditConnector>(m_aswSetup, m_formMore->m_edASWSetup),
        xqcon_create<XQLineEditConnector>(m_aswHold, m_formMore->m_edASWHold),
        xqcon_create<XQLineEditConnector>(m_altSep, m_formMore->m_edALTSep),
        xqcon_create<XQLineEditConnector>(m_g2Setup, m_formMore->m_edG2Setup),
        xqcon_create<XQSpinBoxUnsignedConnector>(m_echoNum, m_formMore->m_numEcho),
        xqcon_create<XQToggleButtonConnector>(m_drivenEquilibrium, m_formMore->m_ckbDrivenEquilibrium),
        xqcon_create<XQComboBoxConnector>(m_numPhaseCycle, m_formMore->m_cmbPhaseCycle, Snapshot( *m_numPhaseCycle)),
        xqcon_create<XQLineEditConnector>(m_combOffRes, m_form->m_edCombOffRes),
        xqcon_create<XQToggleButtonConnector>(m_invertPhase, m_formMore->m_ckbInvertPhase),
        xqcon_create<XQToggleButtonConnector>(m_conserveStEPhase, m_formMore->m_ckbStEPhase),
        xqcon_create<XQComboBoxConnector>(m_p1Func, m_form->m_cmbP1Func, Snapshot( *m_p1Func)),
        xqcon_create<XQComboBoxConnector>(m_p2Func, m_form->m_cmbP2Func, Snapshot( *m_p2Func)),
        xqcon_create<XQComboBoxConnector>(m_combFunc, m_form->m_cmbCombFunc, Snapshot( *m_combFunc)),
        xqcon_create<XQDoubleSpinBoxConnector>(m_p1Level, m_form->m_dblP1Level),
        xqcon_create<XQDoubleSpinBoxConnector>(m_p2Level, m_form->m_dblP2Level),
        xqcon_create<XQDoubleSpinBoxConnector>(m_combLevel, m_form->m_dblCombLevel),
        xqcon_create<XQDoubleSpinBoxConnector>(m_masterLevel, m_form->m_dblMasterLevel, m_form->m_slMasterLevel),
        xqcon_create<XQLineEditConnector>(m_qamOffset1, m_formMore->m_edQAMOffset1),
        xqcon_create<XQLineEditConnector>(m_qamOffset2, m_formMore->m_edQAMOffset2),
        xqcon_create<XQLineEditConnector>(m_qamLevel1, m_formMore->m_edQAMLevel1),
        xqcon_create<XQLineEditConnector>(m_qamLevel2, m_formMore->m_edQAMLevel2),
        xqcon_create<XQLineEditConnector>(m_qamDelay1, m_formMore->m_edQAMDelay1),
        xqcon_create<XQLineEditConnector>(m_qamDelay2, m_formMore->m_edQAMDelay2),
        xqcon_create<XQLineEditConnector>(m_difFreq, m_formMore->m_edDIFFreq),
        xqcon_create<XQToggleButtonConnector>(m_induceEmission, m_formMore->m_ckbInduceEmission),
        xqcon_create<XQDoubleSpinBoxConnector>(m_induceEmissionPhase, m_formMore->m_numInduceEmissionPhase),
        xqcon_create<XQLineEditConnector>(m_qswDelay, m_formMore->m_edQSWDelay),
        xqcon_create<XQLineEditConnector>(m_qswWidth, m_formMore->m_edQSWWidth),
        xqcon_create<XQLineEditConnector>(m_qswSoftSWOff, m_formMore->m_edQSWSoftSWOff),
        xqcon_create<XQToggleButtonConnector>(m_qswPiPulseOnly, m_formMore->m_ckbQSWPiPulseOnly),
        xqcon_create<XQToggleButtonConnector>(m_pulseAnalyzerMode, m_formMore->m_ckbPulseAnalyzerMode),
        xqcon_create<XQLineEditConnector>(m_paPulseRept, m_formMore->m_edPAPulseRept),
        xqcon_create<XQLineEditConnector>(m_paPulseBW, m_formMore->m_edPAPulseBW),
        xqcon_create<XQPulserDriverConnector>(
            dynamic_pointer_cast<XPulser>(shared_from_this()), m_form->m_tblPulse, m_form->m_graph)
    };
    QComboBox*const combo[] = {
        m_formMore->m_cmbPortSel0, m_formMore->m_cmbPortSel1, m_formMore->m_cmbPortSel2, m_formMore->m_cmbPortSel3,
        m_formMore->m_cmbPortSel4, m_formMore->m_cmbPortSel5, m_formMore->m_cmbPortSel6, m_formMore->m_cmbPortSel7,
        m_formMore->m_cmbPortSel8, m_formMore->m_cmbPortSel9, m_formMore->m_cmbPortSel10, m_formMore->m_cmbPortSel11,
        m_formMore->m_cmbPortSel12, m_formMore->m_cmbPortSel13, m_formMore->m_cmbPortSel14, m_formMore->m_cmbPortSel15
    };
    for(unsigned int i = 0; i < NUM_DO_PORTS; i++) {
        m_conUIs.push_back(xqcon_create<XQComboBoxConnector>(m_portSel[i], combo[i], shot));
    }

    m_form->m_dblP1Level->setRange(-20.0, 3.0);
    m_form->m_dblP1Level->setSingleStep(1.0);
    m_form->m_dblP2Level->setRange(-20.0, 3.0);
    m_form->m_dblP2Level->setSingleStep(1.0);
    m_form->m_dblCombLevel->setRange(-20.0, 3.0);
    m_form->m_dblCombLevel->setSingleStep(1.0);
    m_form->m_dblMasterLevel->setRange(-30.0, 0.0);
    m_form->m_dblMasterLevel->setSingleStep(1.0);

	changeUIStatus(true, false);

}

void
XPulser::showForms() {
	// impliment form->show() here
    m_form->showNormal();
    m_form->raise();
}
void
XPulser::onMoreConfigShow(const Snapshot &shot, XTouchableNode *)  {
    m_formMore->showNormal();
	m_formMore->raise();
}

void
XPulser::start() {
	changeUIStatus( !***pulseAnalyzerMode(), true);
	pulseAnalyzerMode()->setUIEnabled(true);

	iterate_commit([=](Transaction &tr){
		m_lsnOnPulseChanged = tr[ *combMode()].onValueChanged().connectWeakly(
			shared_from_this(), &XPulser::onPulseChanged);
		tr[ *rtime()].onValueChanged().connect(m_lsnOnPulseChanged);
		tr[ *tau()].onValueChanged().connect(m_lsnOnPulseChanged);
		tr[ *combPW()].onValueChanged().connect(m_lsnOnPulseChanged);
		tr[ *pw1()].onValueChanged().connect(m_lsnOnPulseChanged);
		tr[ *pw2()].onValueChanged().connect(m_lsnOnPulseChanged);
		tr[ *combNum()].onValueChanged().connect(m_lsnOnPulseChanged);
		tr[ *combPT()].onValueChanged().connect(m_lsnOnPulseChanged);
		tr[ *combP1()].onValueChanged().connect(m_lsnOnPulseChanged);
		tr[ *combP1Alt()].onValueChanged().connect(m_lsnOnPulseChanged);
		tr[ *aswSetup()].onValueChanged().connect(m_lsnOnPulseChanged);
		tr[ *aswHold()].onValueChanged().connect(m_lsnOnPulseChanged);
		tr[ *altSep()].onValueChanged().connect(m_lsnOnPulseChanged);
		tr[ *output()].onValueChanged().connect(m_lsnOnPulseChanged);
		tr[ *g2Setup()].onValueChanged().connect(m_lsnOnPulseChanged);
		tr[ *echoNum()].onValueChanged().connect(m_lsnOnPulseChanged);
		tr[ *drivenEquilibrium()].onValueChanged().connect(m_lsnOnPulseChanged);
		tr[ *numPhaseCycle()].onValueChanged().connect(m_lsnOnPulseChanged);
		tr[ *combOffRes()].onValueChanged().connect(m_lsnOnPulseChanged);
		tr[ *induceEmission()].onValueChanged().connect(m_lsnOnPulseChanged);
		tr[ *induceEmissionPhase()].onValueChanged().connect(m_lsnOnPulseChanged);
		tr[ *qswDelay()].onValueChanged().connect(m_lsnOnPulseChanged);
		tr[ *qswWidth()].onValueChanged().connect(m_lsnOnPulseChanged);
		tr[ *qswSoftSWOff()].onValueChanged().connect(m_lsnOnPulseChanged);
		tr[ *qswPiPulseOnly()].onValueChanged().connect(m_lsnOnPulseChanged);
		tr[ *invertPhase()].onValueChanged().connect(m_lsnOnPulseChanged);
		tr[ *conserveStEPhase()].onValueChanged().connect(m_lsnOnPulseChanged);
		for(unsigned int i = 0; i < NUM_DO_PORTS; i++) {
			tr[ *portSel(i)].onValueChanged().connect(m_lsnOnPulseChanged);
		}

		if(hasQAMPorts()) {
			tr[ *p1Func()].onValueChanged().connect(m_lsnOnPulseChanged);
			tr[ *p2Func()].onValueChanged().connect(m_lsnOnPulseChanged);
			tr[ *combFunc()].onValueChanged().connect(m_lsnOnPulseChanged);
			tr[ *p1Level()].onValueChanged().connect(m_lsnOnPulseChanged);
			tr[ *p2Level()].onValueChanged().connect(m_lsnOnPulseChanged);
			tr[ *combLevel()].onValueChanged().connect(m_lsnOnPulseChanged);
			tr[ *masterLevel()].onValueChanged().connect(m_lsnOnPulseChanged);
			tr[ *qamOffset1()].onValueChanged().connect(m_lsnOnPulseChanged);
			tr[ *qamOffset2()].onValueChanged().connect(m_lsnOnPulseChanged);
			tr[ *qamLevel1()].onValueChanged().connect(m_lsnOnPulseChanged);
			tr[ *qamLevel2()].onValueChanged().connect(m_lsnOnPulseChanged);
			tr[ *qamDelay1()].onValueChanged().connect(m_lsnOnPulseChanged);
			tr[ *qamDelay2()].onValueChanged().connect(m_lsnOnPulseChanged);
			tr[ *difFreq()].onValueChanged().connect(m_lsnOnPulseChanged);
		}

		tr[ *pulseAnalyzerMode()].onValueChanged().connect(m_lsnOnPulseChanged);
		tr[ *paPulseRept()].onValueChanged().connect(m_lsnOnPulseChanged);
		tr[ *paPulseBW()].onValueChanged().connect(m_lsnOnPulseChanged);
    });
}
void
XPulser::stop() {
	m_lsnOnPulseChanged.reset();
  
	changeUIStatus(true, false);
	pulseAnalyzerMode()->setUIEnabled(false);

	for(unsigned int i = 0; i < NUM_DO_PORTS; i++) {
		m_portSel[i]->setUIEnabled(true);
	}
  
	closeInterface();
}

void
XPulser::changeUIStatus(bool nmrmode, bool state) {
    iterate_commit([=](Transaction &tr){
        //Features with QAM in NMR.
        std::vector<shared_ptr<XNode>> runtime_ui{
            p1Func(), p2Func(), combFunc(),
            p1Level(), p2Level(), combLevel(),
            masterLevel(), difFreq()
        };
        bool uienable = state && hasQAMPorts() && nmrmode;
        for(auto &&x: runtime_ui)
            tr[ *x].setUIEnabled(uienable);

        //Features with QAM.
        runtime_ui = {
            qamOffset1(), qamOffset2(),
            qamLevel1(), qamLevel2(),
            qamDelay1(), qamDelay2()
        };
        uienable = state && hasQAMPorts();
        for(auto &&x: runtime_ui)
            tr[ *x].setUIEnabled(uienable);

        runtime_ui = {
            output(), paPulseRept(), paPulseBW()
        };
        for(auto &&x: runtime_ui)
            tr[ *x].setUIEnabled(state);

        //Features in NMR.
        runtime_ui = {
            combMode(), rtMode(),
            rtime(), tau(),
            combPW(), pw1(), pw2(),
            combNum(), combPT(), combP1(), combP1Alt(),
            aswSetup(), aswHold(), altSep(),
            g2Setup(), echoNum(),
            drivenEquilibrium(),
            numPhaseCycle(), combOffRes(),
            induceEmission(), induceEmissionPhase(),
            qswDelay(), qswWidth(), qswSoftSWOff(), qswPiPulseOnly(),
            invertPhase(), conserveStEPhase()
        };
        uienable = state && nmrmode;
        for(auto &&x: runtime_ui)
            tr[ *x].setUIEnabled(uienable);
    });
}

void
XPulser::analyzeRaw(RawDataReader &reader, Transaction &tr) throw (XRecordError&) {
    tr[ *this].m_combMode = reader.pop<int16_t>();
    int16_t pulser_mode = reader.pop<int16_t>();
    tr[ *this].m_pulserMode = pulser_mode;
    switch(pulser_mode) {
    case N_MODE_NMR_PULSER:
    default:
        tr[ *this].m_rtime = reader.pop<double>();
        tr[ *this].m_tau = reader.pop<double>();
        tr[ *this].m_pw1 = reader.pop<double>();
        tr[ *this].m_pw2 = reader.pop<double>();
        tr[ *this].m_combP1 = reader.pop<double>();
        tr[ *this].m_altSep = reader.pop<double>();
        tr[ *this].m_combP1Alt = reader.pop<double>();
        tr[ *this].m_aswSetup = reader.pop<double>();
        tr[ *this].m_aswHold = reader.pop<double>();
        try {
            // ver 2 records
        	tr[ *this].m_difFreq = reader.pop<double>();
        	tr[ *this].m_combPW = reader.pop<double>();
        	tr[ *this].m_combPT = reader.pop<double>();
        	tr[ *this].m_echoNum = reader.pop<uint16_t>();
        	tr[ *this].m_combNum = reader.pop<uint16_t>();
        	tr[ *this].m_rtMode = reader.pop<int16_t>();
        	tr[ *this].m_numPhaseCycle = reader.pop<uint16_t>();
            // ver 3 records
        	tr[ *this].m_invertPhase = reader.pop<uint16_t>();
            // ver 4 records
        	tr[ *this].m_p1Func = reader.pop<int16_t>();
        	tr[ *this].m_p2Func = reader.pop<int16_t>();
        	tr[ *this].m_combFunc = reader.pop<int16_t>();
        	tr[ *this].m_p1Level = reader.pop<double>();
        	tr[ *this].m_p2Level = reader.pop<double>();
        	tr[ *this].m_combLevel = reader.pop<double>();
        	tr[ *this].m_masterLevel = reader.pop<double>();
        	tr[ *this].m_combOffRes= reader.pop<double>();
        	tr[ *this].m_conserveStEPhase = reader.pop<uint16_t>();
        }
        catch (XRecordError &) {
        	const Snapshot &shot(tr);
        	tr[ *this].m_difFreq = shot[ *difFreq()];
        	tr[ *this].m_combPW = shot[ *combPW()];
        	tr[ *this].m_combPT = shot[ *combPT()];
        	tr[ *this].m_echoNum = shot[ *echoNum()];
        	tr[ *this].m_combNum = shot[ *combNum()];
        	tr[ *this].m_rtMode = shot[ *rtMode()];
    		int npat = 16;
    		if(shot[ *numPhaseCycle()].to_str() == NUM_PHASE_CYCLE_1) npat = 1;
    		if(shot[ *numPhaseCycle()].to_str() == NUM_PHASE_CYCLE_2) npat = 2;
    		if(shot[ *numPhaseCycle()].to_str() == NUM_PHASE_CYCLE_4) npat = 4;
    		if(shot[ *numPhaseCycle()].to_str() == NUM_PHASE_CYCLE_8) npat = 8;
    		if(shot[ *numPhaseCycle()].to_str() == NUM_PHASE_CYCLE_16) npat = 16;
    		tr[ *this].m_numPhaseCycle = npat;
    		tr[ *this].m_invertPhase = 0.0;
        	tr[ *this].m_p1Func = PULSE_NO_HAMMING;
        	tr[ *this].m_p2Func = PULSE_NO_HAMMING;
        	tr[ *this].m_combFunc = PULSE_NO_HAMMING;
        	tr[ *this].m_p1Level = 0;
        	tr[ *this].m_p2Level = -5;
        	tr[ *this].m_combLevel = 0;
        	tr[ *this].m_masterLevel = -10;
        	tr[ *this].m_combOffRes= 0;
        	tr[ *this].m_conserveStEPhase = 0;
        }
        tr[ *this].m_paPulseBW = -1;
        tr[ *this].m_paPulseOrigin = 0;

        createRelPatListNMRPulser(tr);
    	break;
    case N_MODE_PULSE_ANALYZER:
        tr[ *this].m_rtime = tr[ *paPulseRept()];;
        tr[ *this].m_tau = 0;
        tr[ *this].m_pw1 = 6800.0 / tr[ *paPulseBW()]; //flattop_longlong
        tr[ *this].m_pw2 = 0;
        tr[ *this].m_combP1 = 0;
        tr[ *this].m_altSep = 0;
        tr[ *this].m_combP1Alt = 0;
        tr[ *this].m_aswSetup = 0.1;
        tr[ *this].m_aswHold = 0.1;
		tr[ *this].m_difFreq = 0;
		tr[ *this].m_combPW = 0;
		tr[ *this].m_combPT = 0;
		tr[ *this].m_echoNum = 1;
		tr[ *this].m_combNum = 1;
		tr[ *this].m_rtMode = N_COMB_MODE_OFF;
		tr[ *this].m_numPhaseCycle = 2;
		tr[ *this].m_invertPhase = 0.0;
    	tr[ *this].m_p1Func = PULSE_NO_FLATTOP_LONG_LONG;
    	tr[ *this].m_p2Func = PULSE_NO_FLATTOP_LONG_LONG;
    	tr[ *this].m_combFunc = PULSE_NO_FLATTOP_LONG_LONG;
    	tr[ *this].m_p1Level = 0;
    	tr[ *this].m_p2Level = -5;
    	tr[ *this].m_combLevel = 0;
    	tr[ *this].m_masterLevel = -10;
    	tr[ *this].m_combOffRes= 0;
    	tr[ *this].m_conserveStEPhase = 0;

        tr[ *this].m_paPulseBW = tr[ *paPulseBW()];
        tr[ *this].m_paPulseOrigin = tr[ *this].m_pw1 / 2;
        createRelPatListPulseAnalyzer(tr);
    	break;
    }
    try {
        createNativePatterns(tr); //calling driver specific virtual funciton.
    }
    catch (XInterface::XInterfaceError &e) {
        e.print();
    }
}

void
XPulser::onPulseChanged(const Snapshot &shot_node, XValueNodeBase *node) {
	XTime time_awared = XTime::now();
	Snapshot shot( *this);

	if(node == pulseAnalyzerMode().get()) {
		changeUIStatus( !shot[ *pulseAnalyzerMode()], true);
	}
	if(shot[ *pulseAnalyzerMode()]) {
		auto writer = std::make_shared<RawData>();

	// ver 1 records below
	    writer->push((int16_t)0);
	    writer->push((int16_t)N_MODE_PULSE_ANALYZER);

		finishWritingRaw(writer, time_awared, XTime::now());
		return;
	}
  
	const double tau__ = rintTermMicroSec(shot[ *tau()]);
	const double asw_setup__ = rintTermMilliSec(shot[ *aswSetup()]);
	const double asw_hold__ = rintTermMilliSec(shot[ *aswHold()]);
	const double alt_sep__ = rintTermMilliSec(shot[ *altSep()]);
	const int echo_num__ = shot[ *echoNum()];
	if(asw_setup__ > 2.0 * tau__)
		trans( *aswSetup()) = 2.0 * tau__;
	if(node != altSep().get()) {
		if(alt_sep__ != asw_setup__ + asw_hold__ + (echo_num__ - 1) * 2 * tau__/1000) {
			trans( *altSep()) = asw_setup__ + asw_hold__ + (echo_num__ - 1) * 2 * tau__/1000;
			return;
		}
	}

	auto writer = std::make_shared<RawData>();

	if( !shot[ *output()]) {
		finishWritingRaw(writer, XTime(), XTime());
		return;
	}

/*  try {
	changeOutput(false, blankpattern);
	}
	catch (XKameError &e) {
	e.print(getLabel() + i18n("Pulser Turn-Off Failed, because"));
	return;
	}
*/    
// ver 1 records below
    writer->push((int16_t)shot[ *combMode()]);
    writer->push((int16_t)N_MODE_NMR_PULSER);
    writer->push((double)rintTermMilliSec(shot[ *rtime()]));
    writer->push((double)tau__);
    writer->push((double)shot[ *pw1()]);
    writer->push((double)shot[ *pw2()]);
    writer->push((double)rintTermMilliSec(shot[ *combP1()]));
    writer->push((double)alt_sep__);
    writer->push((double)rintTermMilliSec(shot[ *combP1Alt()]));
    writer->push((double)asw_setup__);
    writer->push((double)asw_hold__);
// ver 2 records below
    writer->push((double)shot[ *difFreq()]);
    writer->push((double)shot[ *combPW()]);
    writer->push((double)rintTermMicroSec(shot[ *combPT()]));
    writer->push((uint16_t)shot[ *echoNum()]);
    writer->push((uint16_t)shot[ *combNum()]);
    writer->push((int16_t)shot[ *rtMode()]);
	int npat = 16;
	if(shot[ *numPhaseCycle()].to_str() == NUM_PHASE_CYCLE_1) npat = 1;
	if(shot[ *numPhaseCycle()].to_str() == NUM_PHASE_CYCLE_2) npat = 2;
	if(shot[ *numPhaseCycle()].to_str() == NUM_PHASE_CYCLE_4) npat = 4;
	if(shot[ *numPhaseCycle()].to_str() == NUM_PHASE_CYCLE_8) npat = 8;
	if(shot[ *numPhaseCycle()].to_str() == NUM_PHASE_CYCLE_16) npat = 16;
	writer->push((uint16_t)npat);
// ver 3 records below
    writer->push((uint16_t)shot[ *invertPhase()]);
// ver 4 records below
	writer->push((uint16_t)shot[ *p1Func()]);
	writer->push((uint16_t)shot[ *p2Func()]);
	writer->push((uint16_t)shot[ *combFunc()]);
	writer->push((double)shot[ *p1Level()]);
	writer->push((double)shot[ *p2Level()]);
	writer->push((double)shot[ *combLevel()]);
	writer->push((double)shot[ *masterLevel()]);
	writer->push((double)shot[ *combOffRes()]);
	writer->push((uint16_t)shot[ *conserveStEPhase()]);

	finishWritingRaw(writer, time_awared, XTime::now());
}

double
XPulser::Payload::periodicTerm() const {
    assert( !m_relPatList.empty());
    return m_relPatList.back().time;
}

unsigned int 
XPulser::selectedPorts(const Snapshot &shot, int func) const {
	unsigned int mask = 0;
	for(unsigned int i = 0; i < NUM_DO_PORTS; i++) {
		if(shot[ *portSel(i)] == func)
			mask |= 1 << i;
	}
	return mask;
}
void
XPulser::createRelPatListNMRPulser(Transaction &tr) throw (XRecordError&) {
    //A pettern at absolute time
    class tpat {
    public:
        tpat(uint64_t npos, uint32_t newpat, uint32_t nmask) {
            pos = npos; pat = newpat; mask = nmask;
        }
        tpat(const tpat &x) {
            pos = x.pos; pat = x.pat; mask = x.mask;
        }
        uint64_t pos;
        //this pattern bits will be outputted at 'pos'
        uint32_t pat;
        //mask bits
        uint32_t mask;

        bool operator< (const tpat &y) const {
            return pos < y.pos;
        }
    };

    const Snapshot &shot(tr);

	unsigned int g3mask = selectedPorts(shot, PORTSEL_GATE3);
	unsigned int g2mask = selectedPorts(shot, PORTSEL_PREGATE);
	unsigned int pagmask = selectedPorts(shot, PORTSEL_PULSE_ANALYZER_GATE);
	unsigned int g1mask = (selectedPorts(shot, PORTSEL_GATE) | g3mask | pagmask);
	unsigned int trig1mask = selectedPorts(shot, PORTSEL_TRIG1);
	unsigned int trig2mask = selectedPorts(shot, PORTSEL_TRIG2);
	unsigned int aswmask = selectedPorts(shot, PORTSEL_ASW);
	unsigned int qswmask = selectedPorts(shot, PORTSEL_QSW);
	unsigned int pulse1mask = selectedPorts(shot, PORTSEL_PULSE1);
	unsigned int pulse2mask = selectedPorts(shot, PORTSEL_PULSE2);
	unsigned int combmask = selectedPorts(shot, PORTSEL_COMB);
	unsigned int combfmmask = selectedPorts(shot, PORTSEL_COMB_FM);

	bool invert_phase__ = shot[ *this].invertPhase();

	//QPSK patterns correspoinding to 0, pi/2, pi, -pi/2
	unsigned int qpsk[4];
	unsigned int qpskinv[4];
	unsigned int qpskmask;
	qpskmask = bitpatternsOfQPSK(shot, qpsk, qpskinv, invert_phase__);

	uint64_t rtime__ = rintSampsMilliSec(shot[ *this].rtime());
	uint64_t tau__ = rintSampsMicroSec(shot[ *this].tau());
	uint64_t asw_setup__ = rintSampsMilliSec(shot[ *this].aswSetup());
	uint64_t asw_hold__ = rintSampsMilliSec(shot[ *this].aswHold());
	uint64_t alt_sep__ = rintSampsMilliSec(shot[ *this].altSep());
	uint64_t pw1__ = hasQAMPorts() ?
		ceilSampsMicroSec(shot[ *this].pw1()/2)*2 : rintSampsMicroSec(shot[ *this].pw1()/2)*2;
	uint64_t pw2__ = hasQAMPorts() ?
		ceilSampsMicroSec(shot[ *this].pw2()/2)*2 : rintSampsMicroSec(shot[ *this].pw2()/2)*2;
	uint64_t comb_pw__ = hasQAMPorts() ?
		ceilSampsMicroSec(shot[ *this].combPW()/2)*2 : rintSampsMicroSec(shot[ *this].combPW()/2)*2;
	uint64_t comb_pt__ = rintSampsMicroSec(shot[ *this].combPT());
	uint64_t comb_p1__ = rintSampsMilliSec(shot[ *this].combP1());
	uint64_t comb_p1_alt__ = rintSampsMilliSec(shot[ *this].combP1Alt());
	uint64_t g2_setup__ = ceilSampsMicroSec(shot[ *g2Setup()]);
	int echo_num__ = shot[ *this].echoNum();
	int comb_num__ = shot[ *this].combNum();
	int comb_mode__ = shot[ *this].combMode();
	int rt_mode__ = shot[ *this].rtMode();
	int num_phase_cycle__ = shot[ *this].numPhaseCycle();
    int first_phase__ = shot[ *firstPhase()];
  
	bool comb_mode_alt = ((comb_mode__ == N_COMB_MODE_P1_ALT) ||
								(comb_mode__ == N_COMB_MODE_COMB_ALT));
	bool saturation_wo_comb = (comb_num__ == 0);
	bool driven_equilibrium = shot[ *drivenEquilibrium()];
	uint64_t qsw_delay__ = rintSampsMicroSec(shot[ *qswDelay()]);
	uint64_t qsw_width__ = rintSampsMicroSec(shot[ *qswWidth()]);
	uint64_t qsw_softswoff__ = std::min(qsw_width__, rintSampsMicroSec(shot[ *qswSoftSWOff()]));
	bool qsw_pi_only__ = shot[ *qswPiPulseOnly()];
	int comb_rot_num = lrint(shot[ *this].combOffRes() * (shot[ *this].combPW() / 1000.0 * 4));
  
	bool induce_emission__ = shot[ *induceEmission()];
	uint64_t induce_emission___pw = comb_pw__;
	if(comb_mode__ == N_COMB_MODE_OFF)
		num_phase_cycle__ = std::min(num_phase_cycle__, 4);
  
	bool conserve_ste_phase__ = shot[ *this].conserveStEPhase();

	//comb phases
	const uint32_t comb_ste_cancel[MAX_NUM_PHASE_CYCLE] = {
		1, 3, 0, 2, 3, 1, 2, 0, 0, 2, 1, 3, 2, 0, 3, 1
	};
	//induced emission phases
	const uint32_t pindem[MAX_NUM_PHASE_CYCLE] = {
		0, 0, 0, 0, 2, 2, 2, 2, 0, 0, 0, 0, 2, 2, 2, 2
	};

	//pi/2 pulse phases
	const uint32_t p1[MAX_NUM_PHASE_CYCLE] = {
		0, 0, 2, 2, 0, 0, 2, 2, 0, 0, 2, 2, 0, 0, 2, 2
	};
	//pi pulse phases
	const uint32_t p2__[MAX_NUM_PHASE_CYCLE] = {
		0, 2, 1, 3, 0, 2, 1, 3, 0, 2, 1, 3, 0, 2, 1, 3
	};
	const uint32_t *p2 = conserve_ste_phase__ ? p1 : p2__;

	//subsequent pi pulse phases for multiple echoes or for st.e.
	const uint32_t p2multi[MAX_NUM_PHASE_CYCLE] = {
		1, 3, 1, 3, 1, 3, 1, 3, 1, 3, 1, 3, 1, 3, 1, 3
	};
	//stimulated echo pulse phases
	const uint32_t ste_p1[MAX_NUM_PHASE_CYCLE] = {
		0, 2, 0, 2, 0, 2, 0, 2, 0, 2, 0, 2, 0, 2, 0, 2
	};
	const uint32_t ste_p2[MAX_NUM_PHASE_CYCLE] = {
		0, 2, 2, 0, 0, 2, 2, 0, 0, 2, 2, 0, 0, 2, 2, 0
	};
  
	typedef std::multiset<tpat, std::less<tpat> > tpatset;
	tpatset patterns;  // patterns
	tpatset patterns_cheap; //Low priority patterns
	typedef std::multiset<tpat, std::less<tpat> >::iterator tpatset_it;
  
	uint64_t pos = 0;
            
	int echonum = echo_num__;
  
	bool former_of_alt = !invert_phase__;
	for(int i = 0; i < num_phase_cycle__ * (comb_mode_alt ? 2 : 1); i++) {
        int j = (i / (comb_mode_alt ? 2 : 1) + first_phase__) % num_phase_cycle__; //index for phase cycling
		if(invert_phase__)
			j = num_phase_cycle__ - 1 - j;
		former_of_alt = !former_of_alt;
		bool comb_off_res = ((comb_mode__ != N_COMB_MODE_COMB_ALT) || former_of_alt) && (comb_rot_num != 0);
            
		uint64_t p1__ = 0; //0: Comb pulse is OFF.
		if((comb_mode__ != N_COMB_MODE_OFF) &&
		   !((comb_mode__ == N_COMB_MODE_COMB_ALT) && former_of_alt && !(comb_rot_num != 0))) {
			p1__ = ((former_of_alt || (comb_mode__ != N_COMB_MODE_P1_ALT)) ? comb_p1__ : comb_p1_alt__);
		}

		uint64_t rest;
		if(rt_mode__ == N_RT_MODE_FIXREST)
			rest = rtime__;
		else
			rest = rtime__ - p1__;
		if(rest <= 0)
			throw XDriver::XRecordError("Inconsistent pattern of pulser setup.", __FILE__, __LINE__);
    
		if(saturation_wo_comb && (p1__ > 0)) rest = 0;
      
		pos += rest;
      
		//comb pulses
		if((p1__ > 0) && !saturation_wo_comb) {
			uint64_t combpt = std::max(comb_pt__, comb_pw__ + g2_setup__);
			uint64_t cpos = pos - combpt * comb_num__;
     
			patterns_cheap.insert(tpat(cpos - comb_pw__/2 - g2_setup__, comb_off_res ? ~(uint32_t)0 : 0, combfmmask));
			patterns_cheap.insert(tpat(cpos - comb_pw__/2 - g2_setup__, ~(uint32_t)0, combmask));
			bool g2_each__ = (g2_setup__ * 2 + comb_pw__) < combpt;
			for(int k = 0; k < comb_num__; k++) {
				const uint32_t *comb = (conserve_ste_phase__) ?
					 ((k % 2 == 0) ? ste_p1 : ste_p2) : comb_ste_cancel;
				patterns.insert(tpat(cpos + comb_pw__/2, qpsk[comb[j]], qpskmask));
				cpos += combpt;
				cpos -= comb_pw__/2;
				if(g2_each__ || (k == 0))
					patterns_cheap.insert(tpat(cpos - g2_setup__, g2mask, g2mask));
				patterns.insert(tpat(cpos, ~(uint32_t)0, g1mask));
				patterns.insert(tpat(cpos, PAT_QAM_PULSE_IDX_PCOMB, PAT_QAM_PULSE_IDX_MASK));
				cpos += comb_pw__;
				patterns.insert(tpat(cpos, 0 , g1mask));
				patterns.insert(tpat(cpos, 0, PAT_QAM_PULSE_IDX_MASK));
				if(g2_each__ || (k == comb_num__ - 1))
					patterns.insert(tpat(cpos, 0, g2mask));
				if( !qsw_pi_only__) {
					patterns.insert(tpat(cpos + qsw_delay__, ~(uint32_t)0 , qswmask));
					patterns.insert(tpat(cpos + (qsw_delay__ + qsw_width__/2 - qsw_softswoff__/2), 0 , qswmask));
					patterns.insert(tpat(cpos + (qsw_delay__ + qsw_width__/2 + qsw_softswoff__/2), ~(uint32_t)0 , qswmask));
					patterns.insert(tpat(cpos + (qsw_delay__ + qsw_width__), 0 , qswmask));
				}

				cpos -= comb_pw__/2;
			}
			patterns.insert(tpat(cpos + comb_pw__/2, 0, combmask));
			patterns.insert(tpat(cpos + comb_pw__/2, ~(uint32_t)0, combfmmask));
		}   
		pos += p1__;
       
		//pi/2 pulse
		bool g2_kept_p1p2 = false;
		if(pw1__/2) {
			//on
			patterns_cheap.insert(tpat(pos - pw1__/2 - g2_setup__, qpsk[p1[j]], qpskmask));
			patterns_cheap.insert(tpat(pos - pw1__/2 - g2_setup__, ~(uint32_t)0, g2mask));
			patterns.insert(tpat(pos - pw1__/2, ~(uint32_t)0, trig2mask));
			patterns_cheap.insert(tpat(pos - pw1__/2 - g2_setup__, ~(uint32_t)0, pulse1mask));
			patterns.insert(tpat(pos - pw1__/2, PAT_QAM_PULSE_IDX_P1, PAT_QAM_PULSE_IDX_MASK));
			patterns.insert(tpat(pos - pw1__/2, ~(uint32_t)0, g1mask));
			//off
			patterns.insert(tpat(pos + pw1__/2, 0, g1mask));
			patterns.insert(tpat(pos + pw1__/2, 0, PAT_QAM_PULSE_IDX_MASK));
			patterns.insert(tpat(pos + pw1__/2, 0, pulse1mask));
			if( !pw2__/2 || (g2_setup__ * 2 + pw1__/2 + pw2__/2 < tau__)) {
				patterns.insert(tpat(pos + pw1__/2, 0, g2mask));
			}
			else {
				g2_kept_p1p2 = true;
			}
			if( ! qsw_pi_only__) {
				patterns.insert(tpat(pos + pw1__/2 + qsw_delay__, ~(uint32_t)0 , qswmask));
				patterns.insert(tpat(pos + pw1__/2 + (qsw_delay__ + qsw_width__/2 - qsw_softswoff__/2), 0 , qswmask));
				patterns.insert(tpat(pos + pw1__/2 + (qsw_delay__ + qsw_width__/2 + qsw_softswoff__/2), ~(uint32_t)0 , qswmask));
				patterns.insert(tpat(pos + pw1__/2 + (qsw_delay__ + qsw_width__), 0 , qswmask));
			}
		}
		//for pi pulses
		patterns.insert(tpat(pos + pw1__/2, qpsk[p2[j]], qpskmask));
		patterns.insert(tpat(pos + pw1__/2, ~(uint32_t)0, pulse2mask));
     
		//2tau
		pos += 2*tau__;
		//    patterns.insert(tpat(pos - asw_setup__, -1, aswmask | rfswmask, 0));
		patterns.insert(tpat(pos - asw_setup__, ~(uint32_t)0, aswmask));
		patterns.insert(tpat(pos -
			(( !former_of_alt && comb_mode_alt) ? alt_sep__ : 0), ~(uint32_t)0, trig1mask));
                
		//induce emission
		if(induce_emission__) {
			patterns.insert(tpat(pos - induce_emission___pw/2, ~(uint32_t)0, g3mask));
			patterns.insert(tpat(pos - induce_emission___pw/2, PAT_QAM_PULSE_IDX_INDUCE_EMISSION, PAT_QAM_PULSE_IDX_MASK));
			patterns.insert(tpat(pos - induce_emission___pw/2, qpsk[pindem[j]], qpskmask));
			patterns.insert(tpat(pos + induce_emission___pw/2, 0, PAT_QAM_PULSE_IDX_MASK));
			patterns.insert(tpat(pos + induce_emission___pw/2, 0, g3mask));
		}

		//pi pulses 
		pos -= 3*tau__;
		for(int k = 0; k < echonum; k++) {
			pos += 2*tau__;
			if(pw2__/2) {
				patterns.insert(tpat(pos - pw2__/2, 0, trig2mask));
				//on
				if( !g2_kept_p1p2) {
					patterns_cheap.insert(tpat(pos - pw2__/2 - g2_setup__, qpsk[(k == 0) ? p2[j] : p2multi[j]], qpskmask));
					patterns_cheap.insert(tpat(pos - pw2__/2 - g2_setup__, ~(uint32_t)0, g2mask));
				}

				patterns.insert(tpat(pos - pw2__/2, PAT_QAM_PULSE_IDX_P2, PAT_QAM_PULSE_IDX_MASK));
				patterns.insert(tpat(pos - pw2__/2, ~(uint32_t)0, g1mask));
				//off
				patterns.insert(tpat(pos + pw2__/2, 0, PAT_QAM_PULSE_IDX_MASK));
				patterns.insert(tpat(pos + pw2__/2, 0, g1mask));
				patterns.insert(tpat(pos + pw2__/2, 0, pulse2mask));
				patterns.insert(tpat(pos + pw2__/2, 0, g2mask));
				g2_kept_p1p2 = false;
				//QSW
				patterns.insert(tpat(pos + pw2__/2 + qsw_delay__, ~(uint32_t)0 , qswmask));
				patterns.insert(tpat(pos + pw2__/2 + (qsw_delay__ + qsw_width__/2 - qsw_softswoff__/2), 0 , qswmask));
				patterns.insert(tpat(pos + pw2__/2 + (qsw_delay__ + qsw_width__/2 + qsw_softswoff__/2), ~(uint32_t)0 , qswmask));
				patterns.insert(tpat(pos + pw2__/2 + (qsw_delay__ + qsw_width__), 0 , qswmask));
			}
		}
		if(g2_kept_p1p2)
			throw XDriver::XRecordError("Inconsistent pattern of pulser setup.", __FILE__, __LINE__);

		patterns.insert(tpat(pos + tau__ + asw_hold__, 0, aswmask | trig1mask));
		//induce emission
		if(induce_emission__) {
			patterns.insert(tpat(pos + tau__ + asw_hold__ - induce_emission___pw/2, ~(uint32_t)0, g3mask));
			patterns.insert(tpat(pos + tau__ + asw_hold__ - induce_emission___pw/2, PAT_QAM_PULSE_IDX_INDUCE_EMISSION, PAT_QAM_PULSE_IDX_MASK));
			patterns.insert(tpat(pos + tau__ + asw_hold__ - induce_emission___pw/2, qpsk[pindem[j]], qpskmask));
			patterns.insert(tpat(pos + tau__ + asw_hold__ + induce_emission___pw/2, 0, PAT_QAM_PULSE_IDX_MASK));
			patterns.insert(tpat(pos + tau__ + asw_hold__ + induce_emission___pw/2, 0, g3mask));
		}

		if(driven_equilibrium) {
			pos += 2*tau__;
			//pi pulse 
			//on
			patterns_cheap.insert(tpat(pos - pw2__/2 - g2_setup__, qpskinv[p2[j]], qpskmask));
			patterns_cheap.insert(tpat(pos - pw2__/2 - g2_setup__, ~(uint32_t)0, g2mask));
			patterns_cheap.insert(tpat(pos - pw2__/2 - g2_setup__, ~(uint32_t)0, pulse2mask));
			patterns.insert(tpat(pos - pw2__/2, PAT_QAM_PULSE_IDX_P2, PAT_QAM_PULSE_IDX_MASK));
			patterns.insert(tpat(pos - pw2__/2, ~(uint32_t)0, g1mask));
			//off
			patterns.insert(tpat(pos + pw2__/2, 0, pulse2mask));
			patterns.insert(tpat(pos + pw2__/2, 0, PAT_QAM_PULSE_IDX_MASK));
			patterns.insert(tpat(pos + pw2__/2, 0, g1mask | g2mask));
			patterns.insert(tpat(pos + pw2__/2 + qsw_delay__, ~(uint32_t)0 , qswmask));
			if(qsw_softswoff__) {
				patterns.insert(tpat(pos + pw2__/2 + (qsw_delay__ + qsw_width__/2 - qsw_softswoff__/2), 0 , qswmask));
				patterns.insert(tpat(pos + pw2__/2 + (qsw_delay__ + qsw_width__/2 + qsw_softswoff__/2), ~(uint32_t)0 , qswmask));
			}
			patterns.insert(tpat(pos + pw2__/2 + (qsw_delay__ + qsw_width__), 0 , qswmask));
			pos += tau__;
			//pi/2 pulse
			//on
			patterns_cheap.insert(tpat(pos - pw1__/2 - g2_setup__, qpskinv[p1[j]], qpskmask));
			patterns_cheap.insert(tpat(pos - pw1__/2 - g2_setup__, ~(uint32_t)0, pulse1mask));
			patterns_cheap.insert(tpat(pos - pw1__/2 - g2_setup__, ~(uint32_t)0, g2mask));
			patterns.insert(tpat(pos - pw1__/2, PAT_QAM_PULSE_IDX_P1, PAT_QAM_PULSE_IDX_MASK));
			patterns.insert(tpat(pos - pw1__/2, ~(uint32_t)0, g1mask));
			//off
			patterns.insert(tpat(pos + pw1__/2, 0, PAT_QAM_PULSE_IDX_MASK));
			patterns.insert(tpat(pos + pw1__/2, 0, g1mask));
			patterns.insert(tpat(pos + pw1__/2, 0, pulse1mask));
			patterns.insert(tpat(pos + pw1__/2, qpsk[p1[j]], qpskmask));
			patterns.insert(tpat(pos + pw1__/2, 0, g2mask));
			if( !qsw_pi_only__) {
				patterns.insert(tpat(pos + pw1__/2 + qsw_delay__, ~(uint32_t)0 , qswmask));
				patterns.insert(tpat(pos + pw1__/2 + (qsw_delay__ + qsw_width__/2 - qsw_softswoff__/2), 0 , qswmask));
				patterns.insert(tpat(pos + pw1__/2 + (qsw_delay__ + qsw_width__/2 + qsw_softswoff__/2), ~(uint32_t)0 , qswmask));
				patterns.insert(tpat(pos + pw1__/2 + (qsw_delay__ + qsw_width__), 0 , qswmask));
			}
		}
	}
    
    //insert low-priority (cheap) pulses into pattern set
	for(tpatset_it it = patterns_cheap.begin(); it != patterns_cheap.end(); it++) {
		uint64_t npos = it->pos;
		for(tpatset_it kit = patterns.begin(); kit != patterns.end(); kit++) {
            //Avoid overrapping within minPulseWidth(), which is typ. 1 us
            uint64_t diff = llabs((int64_t)kit->pos - (int64_t)npos);
			diff -= pos * (diff / pos);
			if(diff < rintSampsMilliSec(minPulseWidth())) {
				npos = kit->pos;
				break;
			}
		}
		patterns.insert(tpat(npos, it->pat, it->mask));
	}

	//determine the first pattern and the length.
	uint64_t lastpos = 0;
	uint32_t pat = 0;
	for(tpatset_it it = patterns.begin(); it != patterns.end(); it++) {
		lastpos = it->pos - pos;
		pat &= ~it->mask;
		pat |= (it->pat & it->mask);
	}

	tr[ *this].m_relPatList.clear();
	uint64_t patpos = patterns.begin()->pos;
	for(tpatset_it it = patterns.begin(); it != patterns.end();) {
		pat &= ~it->mask;
		pat |= (it->pat & it->mask);
		it++;
		if((it == patterns.end()) || (it->pos != patpos)) {
			//skip duplicated patterns.
			if((it != patterns.end()) &&
				tr[ *this].m_relPatList.size() &&
					(pat == tr[ *this].m_relPatList[tr[ *this].m_relPatList.size() - 1].pattern)) {
				patpos = it->pos;
				continue;
			}
			Payload::RelPat relpat(pat, patpos, patpos - lastpos);
			tr[ *this].m_relPatList.push_back(relpat);
			if(it == patterns.end()) break;
			lastpos = patpos;
			patpos = it->pos;
		}
	}
    
    //Prepares analog pulse waves for QAM
    if(hasQAMPorts()) {
    	for(unsigned int i = 0; i < PAT_QAM_PULSE_IDX_MASK/PAT_QAM_PULSE_IDX; i++)
    		tr[ *this].m_qamWaveForm[i].clear();
    		
		double tau__ = shot[ *this].tau();
		double dif_freq__ = shot[ *this].difFreq();
	
		bool induce_emission__ = shot[ *induceEmission()];
		double induce_emission___phase = shot[ *induceEmissionPhase()] / 180.0 * M_PI;

		makeWaveForm(tr, PAT_QAM_PULSE_IDX_P1/PAT_QAM_PULSE_IDX - 1,
			shot[ *this].pw1()*1e-3,
			pw1__/2, pulseFunc(shot[ *this].p1Func()),
			shot[ *this].p1Level(), dif_freq__ * 1e3, -2 * M_PI * dif_freq__ * 2 * tau__);
		makeWaveForm(tr, PAT_QAM_PULSE_IDX_P2/PAT_QAM_PULSE_IDX - 1,
			shot[ *this].pw2()*1e-3,
			pw2__/2, pulseFunc(shot[ *this].p2Func()),
			shot[ *this].p2Level(), dif_freq__ * 1e3, -2 * M_PI * dif_freq__ * 2 * tau__);
		makeWaveForm(tr, PAT_QAM_PULSE_IDX_PCOMB/PAT_QAM_PULSE_IDX - 1,
			shot[ *this].combPW()*1e-3,
			comb_pw__/2, pulseFunc(shot[ *this].combFunc()),
			shot[ *this].combLevel(), shot[ *this].combOffRes() + dif_freq__ *1000.0);
		if(induce_emission__) {
			makeWaveForm(tr, PAT_QAM_PULSE_IDX_INDUCE_EMISSION/PAT_QAM_PULSE_IDX - 1,
				shot[ *this].combPW()*1e-3,
				 induce_emission___pw/2, pulseFunc(shot[ *this].combFunc()),
				 shot[ *this].combLevel(), shot[ *this].combOffRes() + dif_freq__ *1000.0, induce_emission___phase);
		}
    }
}

unsigned int
XPulser::bitpatternsOfQPSK(const Snapshot &shot, unsigned int qpsk[4], unsigned int qpskinv[4], bool invert) {
	unsigned int qpskamask = selectedPorts(shot, PORTSEL_QPSK_A);
	unsigned int qpskbmask = selectedPorts(shot, PORTSEL_QPSK_B);
	unsigned int qpsknoninvmask = selectedPorts(shot, PORTSEL_QPSK_OLD_NONINV);
	unsigned int qpskinvmask = selectedPorts(shot, PORTSEL_QPSK_OLD_INV);
	unsigned int qpskpsgatemask = selectedPorts(shot, PORTSEL_QPSK_OLD_PSGATE);
	unsigned int qpskmask = qpskamask | qpskbmask |
		qpskinvmask | qpsknoninvmask | qpskpsgatemask | PAT_QAM_PHASE_MASK;

    auto qpsk__ = [=](unsigned int phase) -> unsigned int {
        //unit of phase is pi/2
        auto qpsk_ph__ = [invert](unsigned int phase) -> unsigned int {
            return (phase + (invert ? 2 : 0)) % 4;
        };
        //patterns correspoinding to 0, pi/2, pi, -pi/2
        const unsigned int qpskIQ[4] = {0, 1, 3, 2};
        const unsigned int qpskOLD[4] = {2, 3, 4, 5};
        return
        ((qpskIQ[qpsk_ph__(phase)] & 1) ? qpskamask : 0) |
        ((qpskIQ[qpsk_ph__(phase)] & 2) ? qpskbmask : 0) |
        ((qpskOLD[qpsk_ph__(phase)] & 1) ? qpskpsgatemask : 0) |
        ((qpskOLD[qpsk_ph__(phase)] & 2) ? qpsknoninvmask : 0) |
        ((qpskOLD[qpsk_ph__(phase)] & 4) ? qpskinvmask : 0) |
        (qpsk_ph__(phase) * PAT_QAM_PHASE);
    };

	for(int ph = 0; ph < 4; ++ph) {
		qpsk[ph] = qpsk__(ph);
		qpskinv[ph] = qpsk__((ph + 2) % 4);
	}
	return qpskmask;
}

void
XPulser::createRelPatListPulseAnalyzer(Transaction &tr) throw (XRecordError&) {
	const Snapshot &shot(tr);

	unsigned int trig1mask = selectedPorts(shot, PORTSEL_TRIG1);
	unsigned int trig2mask = selectedPorts(shot, PORTSEL_TRIG2);
	unsigned int aswmask = selectedPorts(shot, PORTSEL_ASW);
	unsigned int pulse1mask = selectedPorts(shot, PORTSEL_PULSE1);
	unsigned int pulse2mask = selectedPorts(shot, PORTSEL_PULSE2);

	unsigned int basebits = aswmask | selectedPorts(shot, PORTSEL_PULSE_ANALYZER_GATE);
	unsigned int trigbits = trig1mask | trig2mask | pulse1mask | pulse2mask;
	unsigned int onbits = 0, onmask = 0;
    if(hasQAMPorts()) {
    	onbits = PAT_QAM_PULSE_IDX_P1;
    	onmask = PAT_QAM_PULSE_IDX_MASK;
    }

	//QPSK patterns correspoinding to 0, pi/2, pi, -pi/2
	unsigned int qpsk[4];
	unsigned int qpskinv[4];
	unsigned int qpskmask;
	qpskmask = bitpatternsOfQPSK(shot, qpsk, qpskinv, false);

	uint64_t rtime__ = rintSampsMilliSec(shot[ *this].rtime());
	uint64_t pw1__ = hasQAMPorts() ?
		ceilSampsMicroSec(shot[ *this].pw1()/2)*2 : rintSampsMicroSec(shot[ *this].pw1()/2)*2;

	if((rtime__ <= pw1__) || (rtime__ <= pw1__) || (rtime__ == 0))
		throw XDriver::XRecordError("Inconsistent pattern of pulser setup.", __FILE__, __LINE__);

	tr[ *this].m_relPatList.clear();
	uint32_t pat = basebits | qpsk[0] | trigbits | onbits;
	tr[ *this].m_relPatList.push_back(Payload::RelPat(pat, rtime__ - pw1__, rtime__ - pw1__));
	pat &= ~onmask;
	pat &= ~trigbits;
	pat = (pat & ~qpskmask) | qpsk[2];
	tr[ *this].m_relPatList.push_back(Payload::RelPat(pat, rtime__, pw1__));
	pat |= onbits;
	tr[ *this].m_relPatList.push_back(Payload::RelPat(pat, 2 * rtime__ - pw1__, rtime__ - pw1__));
	pat &= ~onmask;
	pat &= ~trigbits;
	pat = (pat & ~qpskmask) | qpsk[0];
	tr[ *this].m_relPatList.push_back(Payload::RelPat(pat, 2 * rtime__, pw1__));

    if(hasQAMPorts()) {
    	for(unsigned int i = 0; i < PAT_QAM_PULSE_IDX_MASK/PAT_QAM_PULSE_IDX; i++)
    		tr[ *this].m_qamWaveForm[i].clear();

		makeWaveForm(tr, PAT_QAM_PULSE_IDX_P1/PAT_QAM_PULSE_IDX - 1,
			shot[ *this].pw1()*1e-3,
			pw1__/2, pulseFunc(shot[ *this].p1Func()),
			shot[ *this].p1Level(), 0, 0);
    }
}

void
XPulser::makeWaveForm(Transaction &tr, unsigned int pnum_minus_1,
					  double pw, unsigned int to_center,
					  tpulsefunc func, double dB, double freq, double phase) {
	const Snapshot &shot(tr);
	std::vector<std::complex<double> > &p = tr[ *this].m_qamWaveForm[pnum_minus_1];
	double dma_ao_period = resolutionQAM();
	to_center *= lrint(resolution() / dma_ao_period);
	double delay1 = shot[ *qamDelay1()] * 1e-3 / dma_ao_period;
	double delay2 = shot[ *qamDelay2()] * 1e-3 / dma_ao_period;
	double dx = dma_ao_period / pw;
	double dp = 2*M_PI*freq*dma_ao_period;
	double z = pow(10.0, dB/20.0);
	const int FAC_ANTIALIAS = 3;
	p.resize(to_center * 2);
	std::fill(p.begin(), p.end(), std::complex<double>(0.0));
	std::vector<std::complex<double> > wave(p.size() * FAC_ANTIALIAS, 0.0);
	for(int i = 0; i < (int)wave.size(); ++i) {
		double i1 = (double)(i - (int)wave.size() / 2 - FAC_ANTIALIAS / 2) / FAC_ANTIALIAS - delay1;
		double i2 = i1 + delay1 - delay2;
		double x = z * func(i1 * dx) * cos(i1 * dp + M_PI/4 + phase);
		double y = z * func(i2 * dx) * sin(i2 * dp + M_PI/4 + phase);
		wave[i] = std::complex<double>(x, y) / (double)FAC_ANTIALIAS;
	}
	//Moving average for antialiasing.
	for(int i = 0; i < (int)wave.size(); ++i) {
		int j = i / FAC_ANTIALIAS;
		p[j] += wave[i];
	}
}

void
XPulser::visualize(const Snapshot &shot) {
	const unsigned int blankpattern = selectedPorts(shot, PORTSEL_COMB_FM);
	try {
		changeOutput(shot, shot[ *output()], blankpattern);
	}
	catch (XKameError &e) {
		e.print(getLabel() + i18n("Pulser Turn-On/Off Failed, because"));
	}
}
