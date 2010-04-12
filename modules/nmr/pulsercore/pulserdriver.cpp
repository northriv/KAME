/***************************************************************************
		Copyright (C) 2002-2010 Kentaro Kitagawa
		                   kitag@issp.u-tokyo.ac.jp
		
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
#include <kiconloader.h>

#include <gsl/gsl_sf.h>
#define bessel_i0 gsl_sf_bessel_I0

#define PULSE_FUNC_RECT "Rect. BW=0.89/T"
#define PULSE_FUNC_HANNING "Hanning BW=1.44/T"
#define PULSE_FUNC_HAMMING "Hamming BW=1.30/T"
#define PULSE_FUNC_FLATTOP "Flat-Top BW=3.2/T"
#define PULSE_FUNC_FLATTOP_LONG "Flat-Top BW=5.3/T"
#define PULSE_FUNC_FLATTOP_LONG_LONG "Flat-Top BW=6.8/T"
#define PULSE_FUNC_BLACKMAN "Blackman BW=1.7/T"
#define PULSE_FUNC_BLACKMAN_HARRIS "Blackman-Harris BW=1.9/T"
#define PULSE_FUNC_KAISER_1 "Kaiser(3) BW=1.6/T"
#define PULSE_FUNC_KAISER_2 "Kaiser(7.2) BW=2.6/T"
#define PULSE_FUNC_KAISER_3 "Kaiser(15) BW=3.8/T"
#define PULSE_FUNC_HALF_SIN "Half-sin BW=1.2/T"

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

XPulser::tpulsefunc
XPulser::pulseFunc(const XString &str) const {
	if(str == PULSE_FUNC_HANNING) return &FFT::windowFuncHanning;
	if(str == PULSE_FUNC_HAMMING) return &FFT::windowFuncHamming;
	if(str == PULSE_FUNC_BLACKMAN) return &FFT::windowFuncBlackman;
	if(str == PULSE_FUNC_BLACKMAN_HARRIS) return &FFT::windowFuncBlackmanHarris;
	if(str == PULSE_FUNC_KAISER_1) return &FFT::windowFuncKaiser1;
	if(str == PULSE_FUNC_KAISER_2) return &FFT::windowFuncKaiser2;
	if(str == PULSE_FUNC_KAISER_3) return &FFT::windowFuncKaiser3;
	if(str == PULSE_FUNC_FLATTOP) return &FFT::windowFuncFlatTop;
	if(str == PULSE_FUNC_FLATTOP_LONG) return &FFT::windowFuncFlatTopLong;
	if(str == PULSE_FUNC_FLATTOP_LONG_LONG) return &FFT::windowFuncFlatTopLongLong;
    if(str == PULSE_FUNC_HALF_SIN) return &FFT::windowFuncHalfSin;
	return &FFT::windowFuncRect;
}

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
    m_moreConfigShow(create<XTouchableNode>("MoreConfigShow", true)),
    m_form(new FrmPulser(g_pFrmMain)),
    m_formMore(new FrmPulserMore(g_pFrmMain)) {

	m_form->setWindowTitle(i18n("Pulser Control") + " - " + getLabel() );
	m_formMore->setWindowTitle(i18n("Pulser Control More Config.") + " - " + getLabel() );
	m_form->statusBar()->hide();
	m_formMore->statusBar()->hide();
  
	m_conMoreConfigShow = xqcon_create<XQButtonConnector>(
        m_moreConfigShow, m_form->m_btnMoreConfig);

    m_form->m_btnMoreConfig->setIcon(
		KIconLoader::global()->loadIcon("configure",
		KIconLoader::Toolbar, KIconLoader::SizeSmall, true ) );

	for(Transaction tr( *this);; ++tr) {
		const Snapshot &shot(tr);
		{
	  		const char *desc[] = {
	  			"Gate", "PreGate", "Gate3", "Trig1", "Trig2", "ASW",
	  			"QSW", "Pulse1", "Pulse2", "Comb", "CombFM",
	  			"QPSK-A", "QPSK-B", "QPSK-NonInv", "QPSK-Inv",
	  			"QPSK-PS-Gate", 0L
	  		};
			for(unsigned int i = 0; i < NUM_DO_PORTS; i++) {
				m_portSel[i] = create<XComboNode>(tr, formatString("PortSel%u", i).c_str(), false);
				for(const char **p = &desc[0]; *p; p++)
					tr[ *m_portSel[i]].add(*p);
	//			m_portSel[i]->setUIEnabled(false);
			}
			tr[ *portSel(0)] = PORTSEL_GATE;
			tr[ *portSel(1)] = PORTSEL_PREGATE;
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
	    tr[ *numPhaseCycle()].add(NUM_PHASE_CYCLE_1);
	    tr[ *numPhaseCycle()].add(NUM_PHASE_CYCLE_2);
	    tr[ *numPhaseCycle()].add(NUM_PHASE_CYCLE_4);
	    tr[ *numPhaseCycle()].add(NUM_PHASE_CYCLE_8);
	    tr[ *numPhaseCycle()].add(NUM_PHASE_CYCLE_16);
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

	    tr[ *combMode()].add(COMB_MODE_OFF);
	    tr[ *combMode()].add(COMB_MODE_ON);
	    tr[ *combMode()].add(COMB_MODE_P1_ALT);
	    tr[ *combMode()].add(COMB_MODE_COMB_ALT);
	    tr[ *combMode()] = N_COMB_MODE_OFF;
	    tr[ *rtMode()].add(RT_MODE_FIXREP);
	    tr[ *rtMode()].add(RT_MODE_FIXREST);
	    tr[ *rtMode()] = 1;

	    tr[ *p1Func()].add(PULSE_FUNC_RECT);
	    tr[ *p1Func()].add(PULSE_FUNC_HANNING);
	    tr[ *p1Func()].add(PULSE_FUNC_HAMMING);
	    tr[ *p1Func()].add(PULSE_FUNC_BLACKMAN);
	    tr[ *p1Func()].add(PULSE_FUNC_BLACKMAN_HARRIS);
	    tr[ *p1Func()].add(PULSE_FUNC_FLATTOP);
	    tr[ *p1Func()].add(PULSE_FUNC_FLATTOP_LONG);
	    tr[ *p1Func()].add(PULSE_FUNC_FLATTOP_LONG_LONG);
	    tr[ *p1Func()].add(PULSE_FUNC_KAISER_1);
	    tr[ *p1Func()].add(PULSE_FUNC_KAISER_2);
	    tr[ *p1Func()].add(PULSE_FUNC_KAISER_3);
	    tr[ *p1Func()].add(PULSE_FUNC_HALF_SIN);
	    tr[ *p2Func()].add(PULSE_FUNC_RECT);
	    tr[ *p2Func()].add(PULSE_FUNC_HANNING);
	    tr[ *p2Func()].add(PULSE_FUNC_HAMMING);
	    tr[ *p2Func()].add(PULSE_FUNC_BLACKMAN);
	    tr[ *p2Func()].add(PULSE_FUNC_BLACKMAN_HARRIS);
	    tr[ *p2Func()].add(PULSE_FUNC_FLATTOP);
	    tr[ *p2Func()].add(PULSE_FUNC_FLATTOP_LONG);
	    tr[ *p2Func()].add(PULSE_FUNC_FLATTOP_LONG_LONG);
	    tr[ *p2Func()].add(PULSE_FUNC_KAISER_1);
	    tr[ *p2Func()].add(PULSE_FUNC_KAISER_2);
	    tr[ *p2Func()].add(PULSE_FUNC_KAISER_3);
	    tr[ *p2Func()].add(PULSE_FUNC_HALF_SIN);
	    tr[ *combFunc()].add(PULSE_FUNC_RECT);
	    tr[ *combFunc()].add(PULSE_FUNC_HANNING);
	    tr[ *combFunc()].add(PULSE_FUNC_HAMMING);
	    tr[ *combFunc()].add(PULSE_FUNC_BLACKMAN);
	    tr[ *combFunc()].add(PULSE_FUNC_BLACKMAN_HARRIS);
	    tr[ *combFunc()].add(PULSE_FUNC_FLATTOP);
	    tr[ *combFunc()].add(PULSE_FUNC_FLATTOP_LONG);
	    tr[ *combFunc()].add(PULSE_FUNC_FLATTOP_LONG_LONG);
	    tr[ *combFunc()].add(PULSE_FUNC_KAISER_1);
	    tr[ *combFunc()].add(PULSE_FUNC_KAISER_2);
	    tr[ *combFunc()].add(PULSE_FUNC_KAISER_3);
	    tr[ *combFunc()].add(PULSE_FUNC_HALF_SIN);

	    tr[ *p1Func()] = PULSE_FUNC_KAISER_2;
	    tr[ *p2Func()] = PULSE_FUNC_KAISER_2;
	    tr[ *combFunc()] = PULSE_FUNC_FLATTOP;

		m_lsnOnMoreConfigShow = tr[ *m_moreConfigShow].onTouch().connectWeakly(
			shared_from_this(), &XPulser::onMoreConfigShow,
			XListener::FLAG_MAIN_THREAD_CALL | XListener::FLAG_AVOID_DUP);
		if(tr.commit()) {
			QComboBox*const combo[] = {
				m_formMore->m_cmbPortSel0, m_formMore->m_cmbPortSel1, m_formMore->m_cmbPortSel2, m_formMore->m_cmbPortSel3,
				m_formMore->m_cmbPortSel4, m_formMore->m_cmbPortSel5, m_formMore->m_cmbPortSel6, m_formMore->m_cmbPortSel7,
				m_formMore->m_cmbPortSel8, m_formMore->m_cmbPortSel9, m_formMore->m_cmbPortSel10, m_formMore->m_cmbPortSel11,
				m_formMore->m_cmbPortSel12, m_formMore->m_cmbPortSel13, m_formMore->m_cmbPortSel14, m_formMore->m_cmbPortSel15
			};
			for(unsigned int i = 0; i < NUM_DO_PORTS; i++) {
				m_conPortSel[i] = xqcon_create<XQComboBoxConnector>(m_portSel[i], combo[i], shot);
			}
			break;
		}
	}
  
	m_conOutput = xqcon_create<XQToggleButtonConnector>(m_output, m_form->m_ckbOutput);
	m_conCombMode = xqcon_create<XQComboBoxConnector>(m_combMode, m_form->m_cmbCombMode, Snapshot( *m_combMode));
	m_conRTMode = xqcon_create<XQComboBoxConnector>(m_rtMode, m_form->m_cmbRTMode, Snapshot( *m_rtMode));
	m_conRT = xqcon_create<XQLineEditConnector>(m_rt, m_form->m_edRT);
	m_conTau = xqcon_create<XQLineEditConnector>(m_tau, m_form->m_edTau);
	m_conCombPW = xqcon_create<XQLineEditConnector>(m_combPW, m_form->m_edCombPW);
	m_conPW1 = xqcon_create<XQLineEditConnector>(m_pw1, m_form->m_edPW1);
	m_conPW2 = xqcon_create<XQLineEditConnector>(m_pw2, m_form->m_edPW2);
	m_conCombNum = xqcon_create<XQSpinBoxConnector>(m_combNum, m_form->m_numCombNum);
	m_conCombPT = xqcon_create<XQLineEditConnector>(m_combPT, m_form->m_edCombPT);
	m_conCombP1 = xqcon_create<XQLineEditConnector>(m_combP1, m_form->m_edCombP1);
	m_conCombP1Alt = xqcon_create<XQLineEditConnector>(m_combP1Alt, m_form->m_edCombP1Alt);
	m_conASWSetup = xqcon_create<XQLineEditConnector>(m_aswSetup, m_formMore->m_edASWSetup);
	m_conASWHold = xqcon_create<XQLineEditConnector>(m_aswHold, m_formMore->m_edASWHold);
	m_conALTSep = xqcon_create<XQLineEditConnector>(m_altSep, m_formMore->m_edALTSep);
	m_conG2Setup = xqcon_create<XQLineEditConnector>(m_g2Setup, m_formMore->m_edG2Setup);
	m_conEchoNum = xqcon_create<XQSpinBoxConnector>(m_echoNum, m_formMore->m_numEcho);
	m_conDrivenEquilibrium = xqcon_create<XQToggleButtonConnector>(m_drivenEquilibrium, m_formMore->m_ckbDrivenEquilibrium);
	m_conNumPhaseCycle = xqcon_create<XQComboBoxConnector>(m_numPhaseCycle, m_formMore->m_cmbPhaseCycle, Snapshot( *m_numPhaseCycle));
	m_conCombOffRes = xqcon_create<XQLineEditConnector>(m_combOffRes, m_form->m_edCombOffRes);
	m_conInvertPhase = xqcon_create<XQToggleButtonConnector>(m_invertPhase, m_formMore->m_ckbInvertPhase);
	m_conConserveStEPhase = xqcon_create<XQToggleButtonConnector>(m_conserveStEPhase, m_formMore->m_ckbStEPhase);
	m_conP1Func = xqcon_create<XQComboBoxConnector>(m_p1Func, m_form->m_cmbP1Func, Snapshot( *m_p1Func));
	m_conP2Func = xqcon_create<XQComboBoxConnector>(m_p2Func, m_form->m_cmbP2Func, Snapshot( *m_p2Func));
	m_conCombFunc = xqcon_create<XQComboBoxConnector>(m_combFunc, m_form->m_cmbCombFunc, Snapshot( *m_combFunc));
	m_form->m_dblP1Level->setRange(-20.0, 3.0, 1.0, false);
	m_conP1Level = xqcon_create<XKDoubleNumInputConnector>(m_p1Level, m_form->m_dblP1Level);
	m_form->m_dblP2Level->setRange(-20.0, 3.0, 1.0, false);
	m_conP2Level = xqcon_create<XKDoubleNumInputConnector>(m_p2Level, m_form->m_dblP2Level);
	m_form->m_dblCombLevel->setRange(-20.0, 3.0, 1.0, false);
	m_conCombLevel = xqcon_create<XKDoubleNumInputConnector>(m_combLevel, m_form->m_dblCombLevel);
	m_form->m_dblMasterLevel->setRange(-30.0, 0.0, 1.0, true);
	m_conMasterLevel = xqcon_create<XKDoubleNumInputConnector>(m_masterLevel, m_form->m_dblMasterLevel);
	m_conQAMOffset1 = xqcon_create<XQLineEditConnector>(m_qamOffset1, m_formMore->m_edQAMOffset1);  
	m_conQAMOffset2 = xqcon_create<XQLineEditConnector>(m_qamOffset2, m_formMore->m_edQAMOffset2);
	m_conQAMLevel1 = xqcon_create<XQLineEditConnector>(m_qamLevel1, m_formMore->m_edQAMLevel1);  
	m_conQAMLevel2 = xqcon_create<XQLineEditConnector>(m_qamLevel2, m_formMore->m_edQAMLevel2);
	m_conQAMDelay1 = xqcon_create<XQLineEditConnector>(m_qamDelay1, m_formMore->m_edQAMDelay1);  
	m_conQAMDelay2 = xqcon_create<XQLineEditConnector>(m_qamDelay2, m_formMore->m_edQAMDelay2);
	m_conDIFFreq = xqcon_create<XQLineEditConnector>(m_difFreq, m_formMore->m_edDIFFreq);  
	m_conInduceEmission = xqcon_create<XQToggleButtonConnector>(m_induceEmission, m_formMore->m_ckbInduceEmission);
	m_conInduceEmissionPhase = xqcon_create<XKDoubleNumInputConnector>(m_induceEmissionPhase, m_formMore->m_numInduceEmissionPhase);
	m_conQSWDelay = xqcon_create<XQLineEditConnector>(m_qswDelay, m_formMore->m_edQSWDelay);  
	m_conQSWWidth = xqcon_create<XQLineEditConnector>(m_qswWidth, m_formMore->m_edQSWWidth);  
	m_conQSWSoftSWOff = xqcon_create<XQLineEditConnector>(m_qswSoftSWOff, m_formMore->m_edQSWSoftSWOff);  
	m_conQSWPiPulseOnly = xqcon_create<XQToggleButtonConnector>(m_qswPiPulseOnly, m_formMore->m_ckbQSWPiPulseOnly);  
 
	output()->setUIEnabled(false);
	combMode()->setUIEnabled(false);
	rtMode()->setUIEnabled(false);
	rtime()->setUIEnabled(false);
	tau()->setUIEnabled(false);
	combPW()->setUIEnabled(false);
	pw1()->setUIEnabled(false);
	pw2()->setUIEnabled(false);
	combNum()->setUIEnabled(false);
	combPT()->setUIEnabled(false);
	combP1()->setUIEnabled(false);
	combP1Alt()->setUIEnabled(false);
	aswSetup()->setUIEnabled(false);
	aswHold()->setUIEnabled(false);
	altSep()->setUIEnabled(false);
	g2Setup()->setUIEnabled(false);
	echoNum()->setUIEnabled(false);
	drivenEquilibrium()->setUIEnabled(false);
	numPhaseCycle()->setUIEnabled(false);
	combOffRes()->setUIEnabled(false);
	p1Func()->setUIEnabled(false);
	p2Func()->setUIEnabled(false);
	combFunc()->setUIEnabled(false);
	p1Level()->setUIEnabled(false);
	p2Level()->setUIEnabled(false);
	combLevel()->setUIEnabled(false);
	masterLevel()->setUIEnabled(false);
	qamOffset1()->setUIEnabled(false);
	qamOffset2()->setUIEnabled(false);
	qamLevel1()->setUIEnabled(false);
	qamLevel2()->setUIEnabled(false);
	qamDelay1()->setUIEnabled(false);
	qamDelay2()->setUIEnabled(false);
	difFreq()->setUIEnabled(false);
	induceEmission()->setUIEnabled(false);
	induceEmissionPhase()->setUIEnabled(false);
	qswDelay()->setUIEnabled(false);
	qswWidth()->setUIEnabled(false);
	qswSoftSWOff()->setUIEnabled(false);
	qswPiPulseOnly()->setUIEnabled(false);
	invertPhase()->setUIEnabled(false);
	conserveStEPhase()->setUIEnabled(false);

	m_conPulserDriver = xqcon_create<XQPulserDriverConnector>(
		dynamic_pointer_cast<XPulser>(shared_from_this()), m_form->m_tblPulse, m_form->m_graph);
}

void
XPulser::showForms() {
	//! impliment form->show() here
    m_form->show();
    m_form->raise();
}
void
XPulser::onMoreConfigShow(const Snapshot &shot, XTouchableNode *)  {
	m_formMore->show();
	m_formMore->raise();
}

void
XPulser::start() {
	output()->setUIEnabled(true);
	combMode()->setUIEnabled(true);
	rtMode()->setUIEnabled(true);
	rtime()->setUIEnabled(true);
	tau()->setUIEnabled(true);
	combPW()->setUIEnabled(true);
	pw1()->setUIEnabled(true);
	pw2()->setUIEnabled(true);
	combNum()->setUIEnabled(true);
	combPT()->setUIEnabled(true);
	combP1()->setUIEnabled(true);
	combP1Alt()->setUIEnabled(true);
	aswSetup()->setUIEnabled(true);
	aswHold()->setUIEnabled(true);
	altSep()->setUIEnabled(true);
	g2Setup()->setUIEnabled(true);
	echoNum()->setUIEnabled(true);
	drivenEquilibrium()->setUIEnabled(true);
	numPhaseCycle()->setUIEnabled(true);
	combOffRes()->setUIEnabled(true);
	induceEmission()->setUIEnabled(true);
	induceEmissionPhase()->setUIEnabled(true);
	qswDelay()->setUIEnabled(true);
	qswWidth()->setUIEnabled(true);
	qswSoftSWOff()->setUIEnabled(true);
	qswPiPulseOnly()->setUIEnabled(true);
	invertPhase()->setUIEnabled(true);
	conserveStEPhase()->setUIEnabled(true);
	//Port0 is locked.
	for(unsigned int i = 1; i < NUM_DO_PORTS; i++) {
//	m_portSel[i]->setUIEnabled(true);
	}
  
	if(haveQAMPorts()) {
		p1Func()->setUIEnabled(true);
		p2Func()->setUIEnabled(true);
		combFunc()->setUIEnabled(true);
		p1Level()->setUIEnabled(true);
		p2Level()->setUIEnabled(true);
		combLevel()->setUIEnabled(true);
		masterLevel()->setUIEnabled(true);
		qamOffset1()->setUIEnabled(true);
		qamOffset2()->setUIEnabled(true);
		qamLevel1()->setUIEnabled(true);
		qamLevel2()->setUIEnabled(true);
		qamDelay1()->setUIEnabled(true);
		qamDelay2()->setUIEnabled(true);
		difFreq()->setUIEnabled(true);  
	}

	for(Transaction tr( *this);; ++tr) {
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

		if(haveQAMPorts()) {
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
		if(tr.commit())
			break;
	}
}
void
XPulser::stop() {
	m_lsnOnPulseChanged.reset();
  
	output()->setUIEnabled(false);
	combMode()->setUIEnabled(false);
	rtMode()->setUIEnabled(false);
	rtime()->setUIEnabled(false);
	tau()->setUIEnabled(false);
	combPW()->setUIEnabled(false);
	pw1()->setUIEnabled(false);
	pw2()->setUIEnabled(false);
	combNum()->setUIEnabled(false);
	combPT()->setUIEnabled(false);
	combP1()->setUIEnabled(false);
	combP1Alt()->setUIEnabled(false);
	aswSetup()->setUIEnabled(false);
	aswHold()->setUIEnabled(false);
	altSep()->setUIEnabled(false);
	g2Setup()->setUIEnabled(false);
	echoNum()->setUIEnabled(false);
	drivenEquilibrium()->setUIEnabled(false);
	numPhaseCycle()->setUIEnabled(false);
	combOffRes()->setUIEnabled(false);
	p1Func()->setUIEnabled(false);
	p2Func()->setUIEnabled(false);
	combFunc()->setUIEnabled(false);
	p1Level()->setUIEnabled(false);
	p2Level()->setUIEnabled(false);
	combLevel()->setUIEnabled(false);
	masterLevel()->setUIEnabled(false);
	qamOffset1()->setUIEnabled(false);
	qamOffset2()->setUIEnabled(false);
	qamLevel1()->setUIEnabled(false);
	qamLevel2()->setUIEnabled(false);
	qamDelay1()->setUIEnabled(false);
	qamDelay2()->setUIEnabled(false);
	difFreq()->setUIEnabled(false);
	induceEmission()->setUIEnabled(false);
	induceEmissionPhase()->setUIEnabled(false);
	qswDelay()->setUIEnabled(false);
	qswWidth()->setUIEnabled(false);
	qswSoftSWOff()->setUIEnabled(false);
	qswPiPulseOnly()->setUIEnabled(false);
	invertPhase()->setUIEnabled(false);
	conserveStEPhase()->setUIEnabled(false);
	for(unsigned int i = 0; i < NUM_DO_PORTS; i++) {
//	m_portSel[i]->setUIEnabled(false);
		m_portSel[i]->setUIEnabled(true);
	}
  
	afterStop();
//    m_thread->waitFor();
//  thread must do interface()->close() at the end
}

void
XPulser::analyzeRaw(RawDataReader &reader, Transaction &tr) throw (XRecordError&) {
    tr[ *this].m_combMode = reader.pop<int16_t>();
    reader.pop<int16_t>(); //reserve
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
        //! ver 2 records
    	tr[ *this].m_difFreq = reader.pop<double>();
    	tr[ *this].m_combPW = reader.pop<double>();
    	tr[ *this].m_combPT = reader.pop<double>();
    	tr[ *this].m_echoNum = reader.pop<uint16_t>();
    	tr[ *this].m_combNum = reader.pop<uint16_t>();
    	tr[ *this].m_rtMode = reader.pop<int16_t>();
    	tr[ *this].m_numPhaseCycle = reader.pop<uint16_t>();
        //! ver 3 records
    	tr[ *this].m_invertPhase = reader.pop<uint16_t>();
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
    }
    rawToRelPat(tr);
}

void
XPulser::onPulseChanged(const Snapshot &shot_node, XValueNodeBase *node) {
	XTime time_awared = XTime::now();
	Snapshot shot( *this);
  
	const double _tau = rintTermMicroSec(shot[ *tau()]);
	const double _asw_setup = rintTermMilliSec(shot[ *aswSetup()]);
	const double _asw_hold = rintTermMilliSec(shot[ *aswHold()]);
	const double _alt_sep = rintTermMilliSec(shot[ *altSep()]);
	const int _echo_num = shot[ *echoNum()];
	if(_asw_setup > 2.0 * _tau)
		trans( *aswSetup()) = 2.0 * _tau;
	if(node != altSep().get()) {
		if(_alt_sep != _asw_setup + _asw_hold + (_echo_num - 1) * 2 * _tau/1000) {
			trans( *altSep()) = _asw_setup + _asw_hold + (_echo_num - 1) * 2 * _tau/1000;
			return;
		}
	}

	shared_ptr<RawData> writer(new RawData);

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
//! ver 1 records below
    writer->push((int16_t)shot[ *combMode()]);
    writer->push((int16_t)0); //reserve
    writer->push((double)rintTermMilliSec(shot[ *rtime()]));
    writer->push((double)_tau);
    writer->push((double)shot[ *pw1()]);
    writer->push((double)shot[ *pw2()]);
    writer->push((double)rintTermMilliSec(shot[ *combP1()]));
    writer->push((double)_alt_sep);
    writer->push((double)rintTermMilliSec(shot[ *combP1Alt()]));
    writer->push((double)_asw_setup);
    writer->push((double)_asw_hold);
//! ver 2 records below
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
//! ver 3 records below
    writer->push((uint16_t)shot[ *invertPhase()]);

	finishWritingRaw(writer, time_awared, XTime::now());
}
double
XPulser::Payload::periodicTerm() const {
    ASSERT( !m_relPatList.empty());
    return m_relPatList.back().time;
}

#include <set>
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
XPulser::rawToRelPat(Transaction &tr) throw (XRecordError&) {
	const Snapshot &shot(tr);

	unsigned int g3mask = selectedPorts(shot, PORTSEL_GATE3);
	unsigned int g2mask = selectedPorts(shot, PORTSEL_PREGATE);
	unsigned int g1mask = (selectedPorts(shot, PORTSEL_GATE) | g3mask);
	unsigned int trig1mask = selectedPorts(shot, PORTSEL_TRIG1);
	unsigned int trig2mask = selectedPorts(shot, PORTSEL_TRIG2);
	unsigned int aswmask = selectedPorts(shot, PORTSEL_ASW);
	unsigned int qswmask = selectedPorts(shot, PORTSEL_QSW);
	unsigned int pulse1mask = selectedPorts(shot, PORTSEL_PULSE1);
	unsigned int pulse2mask = selectedPorts(shot, PORTSEL_PULSE2);
	unsigned int combmask = selectedPorts(shot, PORTSEL_COMB);
	unsigned int combfmmask = selectedPorts(shot, PORTSEL_COMB_FM);
	unsigned int qpskamask = selectedPorts(shot, PORTSEL_QPSK_A);
	unsigned int qpskbmask = selectedPorts(shot, PORTSEL_QPSK_B);
	unsigned int qpsknoninvmask = selectedPorts(shot, PORTSEL_QPSK_OLD_NONINV);
	unsigned int qpskinvmask = selectedPorts(shot, PORTSEL_QPSK_OLD_INV);
	unsigned int qpskpsgatemask = selectedPorts(shot, PORTSEL_QPSK_OLD_PSGATE);
	unsigned int qpskmask = qpskamask | qpskbmask |
		qpskinvmask | qpsknoninvmask | qpskpsgatemask | PAT_QAM_PHASE_MASK;
	
	uint64_t _rtime = rintSampsMilliSec(shot[ *this].rtime());
	uint64_t _tau = rintSampsMicroSec(shot[ *this].tau());
	uint64_t _asw_setup = rintSampsMilliSec(shot[ *this].aswSetup());
	uint64_t _asw_hold = rintSampsMilliSec(shot[ *this].aswHold());
	uint64_t _alt_sep = rintSampsMilliSec(shot[ *this].altSep());
	uint64_t _pw1 = haveQAMPorts() ? 
		ceilSampsMicroSec(shot[ *this].pw1()/2)*2 : rintSampsMicroSec(shot[ *this].pw1()/2)*2;
	uint64_t _pw2 = haveQAMPorts() ?
		ceilSampsMicroSec(shot[ *this].pw2()/2)*2 : rintSampsMicroSec(shot[ *this].pw2()/2)*2;
	uint64_t _comb_pw = haveQAMPorts() ?
		ceilSampsMicroSec(shot[ *this].combPW()/2)*2 : rintSampsMicroSec(shot[ *this].combPW()/2)*2;
	uint64_t _comb_pt = rintSampsMicroSec(shot[ *this].combPT());
	uint64_t _comb_p1 = rintSampsMilliSec(shot[ *this].combP1());
	uint64_t _comb_p1_alt = rintSampsMilliSec(shot[ *this].combP1Alt());
	uint64_t _g2_setup = ceilSampsMicroSec(shot[ *g2Setup()]);
	int _echo_num = shot[ *this].echoNum();
	int _comb_num = shot[ *this].combNum();
	int _comb_mode = shot[ *this].combMode();
	int _rt_mode = shot[ *this].rtMode();
	int _num_phase_cycle = shot[ *this].numPhaseCycle();
  
	bool comb_mode_alt = ((_comb_mode == N_COMB_MODE_P1_ALT) ||
								(_comb_mode == N_COMB_MODE_COMB_ALT));
	bool saturation_wo_comb = (_comb_num == 0);
	bool driven_equilibrium = shot[ *drivenEquilibrium()];
	uint64_t _qsw_delay = rintSampsMicroSec(shot[ *qswDelay()]);
	uint64_t _qsw_width = rintSampsMicroSec(shot[ *qswWidth()]);
	uint64_t _qsw_softswoff = std::min(_qsw_width, rintSampsMicroSec(shot[ *qswSoftSWOff()]));
	bool _qsw_pi_only = shot[ *qswPiPulseOnly()];
	int comb_rot_num = lrint(shot[ *combOffRes()] * (shot[ *this].combPW() / 1000.0 * 4));
  
	bool _induce_emission = shot[ *induceEmission()];
	uint64_t _induce_emission_pw = _comb_pw;
	if((_comb_mode == N_COMB_MODE_OFF))
		_num_phase_cycle = std::min(_num_phase_cycle, 4);
  
	bool _invert_phase = shot[ *this].invertPhase();
	bool _conserve_ste_phase = shot[ *conserveStEPhase()];

	//patterns correspoinding to 0, pi/2, pi, -pi/2
	const unsigned int qpskIQ[4] = {0, 1, 3, 2};
	const unsigned int qpskOLD[4] = {2, 3, 4, 5};

	//unit of phase is pi/2
#define __qpsk(phase) ((((phase) + (_invert_phase ? 2 : 0)) % 4))
#define _qpsk(phase)  ( \
  	((qpskIQ[__qpsk(phase)] & 1) ? qpskamask : 0) | \
  	((qpskIQ[__qpsk(phase)] & 2) ? qpskbmask : 0) | \
  	((qpskOLD[__qpsk(phase)] & 1) ? qpskpsgatemask : 0) | \
  	((qpskOLD[__qpsk(phase)] & 2) ? qpsknoninvmask : 0) | \
  	((qpskOLD[__qpsk(phase)] & 4) ? qpskinvmask : 0) | \
  	(__qpsk(phase) * PAT_QAM_PHASE))
	const unsigned int qpsk[4] = {_qpsk(0), _qpsk(1), _qpsk(2), _qpsk(3)};
	const unsigned int qpskinv[4] = {_qpsk(2), _qpsk(3), _qpsk(0), _qpsk(1)};

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
	const uint32_t _p2[MAX_NUM_PHASE_CYCLE] = {
		0, 2, 1, 3, 0, 2, 1, 3, 0, 2, 1, 3, 0, 2, 1, 3
	};
	const uint32_t *p2 = _conserve_ste_phase ? p1 : _p2;

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

	tr[ *this].m_relPatList.clear();
  
	uint64_t pos = 0;
            
	int echonum = _echo_num;
  
	bool former_of_alt = !_invert_phase;
	for(int i = 0; i < _num_phase_cycle * (comb_mode_alt ? 2 : 1); i++) {
		int j = (i / (comb_mode_alt ? 2 : 1)) % _num_phase_cycle; //index for phase cycling
		if(_invert_phase)
			j = _num_phase_cycle - 1 - j;
		former_of_alt = !former_of_alt;
		bool comb_off_res = ((_comb_mode != N_COMB_MODE_COMB_ALT) || former_of_alt) && (comb_rot_num != 0);
            
		uint64_t _p1 = 0;
		if((_comb_mode != N_COMB_MODE_OFF) &&
		   !((_comb_mode == N_COMB_MODE_COMB_ALT) && former_of_alt && !(comb_rot_num != 0))) {
			_p1 = ((former_of_alt || (_comb_mode != N_COMB_MODE_P1_ALT)) ? _comb_p1 : _comb_p1_alt);
		}

		uint64_t rest;
		if(_rt_mode == N_RT_MODE_FIXREST)
			rest = _rtime;
		else
			rest = _rtime - _p1;
    
		if(saturation_wo_comb && (_p1 > 0)) rest = 0;
      
		if(rest > 0) pos += rest;
      
		//comb pulses
		if((_p1 > 0) && !saturation_wo_comb) {
			uint64_t combpt = std::max(_comb_pt, _comb_pw + _g2_setup);
			uint64_t cpos = pos - combpt * _comb_num;
     
			patterns_cheap.insert(tpat(cpos - _comb_pw/2 - _g2_setup, comb_off_res ? ~(uint32_t)0 : 0, combfmmask));
			patterns_cheap.insert(tpat(cpos - _comb_pw/2 - _g2_setup, ~(uint32_t)0, combmask));
			bool _g2_each = (_g2_setup * 2 + _comb_pw) < combpt;
			for(int k = 0; k < _comb_num; k++) {
				const uint32_t *comb = (_conserve_ste_phase) ?
					 ((k % 2 == 0) ? ste_p1 : ste_p2) : comb_ste_cancel;
				patterns.insert(tpat(cpos + _comb_pw/2, qpsk[comb[j]], qpskmask));
				cpos += combpt;
				cpos -= _comb_pw/2;
				if(_g2_each || (k == 0))
					patterns_cheap.insert(tpat(cpos - _g2_setup, g2mask, g2mask));
				patterns.insert(tpat(cpos, ~(uint32_t)0, g1mask));
				patterns.insert(tpat(cpos, PAT_QAM_PULSE_IDX_PCOMB, PAT_QAM_PULSE_IDX_MASK));
				cpos += _comb_pw;      
				patterns.insert(tpat(cpos, 0 , g1mask));
				patterns.insert(tpat(cpos, 0, PAT_QAM_PULSE_IDX_MASK));
				if(_g2_each || (k == _comb_num - 1))
					patterns.insert(tpat(cpos, 0, g2mask));
				if( ! _qsw_pi_only) {
					patterns.insert(tpat(cpos + _qsw_delay, ~(uint32_t)0 , qswmask));
					patterns.insert(tpat(cpos + (_qsw_delay + _qsw_width/2 - _qsw_softswoff/2), 0 , qswmask));
					patterns.insert(tpat(cpos + (_qsw_delay + _qsw_width/2 + _qsw_softswoff/2), ~(uint32_t)0 , qswmask));
					patterns.insert(tpat(cpos + (_qsw_delay + _qsw_width), 0 , qswmask));
				}

				cpos -= _comb_pw/2;
			}
			patterns.insert(tpat(cpos + _comb_pw/2, 0, combmask));
			patterns.insert(tpat(cpos + _comb_pw/2, ~(uint32_t)0, combfmmask));
		}   
		pos += _p1;
       
		//pi/2 pulse
		bool _g2_kept_p1p2 = false;
		if(_pw1/2) {
			//on
			patterns_cheap.insert(tpat(pos - _pw1/2 - _g2_setup, qpsk[p1[j]], qpskmask));
			patterns_cheap.insert(tpat(pos - _pw1/2 - _g2_setup, ~(uint32_t)0, g2mask));
			patterns.insert(tpat(pos - _pw1/2, ~(uint32_t)0, trig2mask));
			patterns_cheap.insert(tpat(pos - _pw1/2 - _g2_setup, ~(uint32_t)0, pulse1mask));
			patterns.insert(tpat(pos - _pw1/2, PAT_QAM_PULSE_IDX_P1, PAT_QAM_PULSE_IDX_MASK));
			patterns.insert(tpat(pos - _pw1/2, ~(uint32_t)0, g1mask));
			//off
			patterns.insert(tpat(pos + _pw1/2, 0, g1mask));
			patterns.insert(tpat(pos + _pw1/2, 0, PAT_QAM_PULSE_IDX_MASK));
			patterns.insert(tpat(pos + _pw1/2, 0, pulse1mask));
			if( !_pw2/2 || (_g2_setup * 2 + _pw1/2 + _pw2/2 < _tau)) {
				patterns.insert(tpat(pos + _pw1/2, 0, g2mask));
			}
			else {
				_g2_kept_p1p2 = true;
			}
			if( ! _qsw_pi_only) {
				patterns.insert(tpat(pos + _pw1/2 + _qsw_delay, ~(uint32_t)0 , qswmask));
				patterns.insert(tpat(pos + _pw1/2 + (_qsw_delay + _qsw_width/2 - _qsw_softswoff/2), 0 , qswmask));
				patterns.insert(tpat(pos + _pw1/2 + (_qsw_delay + _qsw_width/2 + _qsw_softswoff/2), ~(uint32_t)0 , qswmask));
				patterns.insert(tpat(pos + _pw1/2 + (_qsw_delay + _qsw_width), 0 , qswmask));
			}
		}
		//for pi pulses
		patterns.insert(tpat(pos + _pw1/2, qpsk[p2[j]], qpskmask));
		patterns.insert(tpat(pos + _pw1/2, ~(uint32_t)0, pulse2mask));
     
		//2tau
		pos += 2*_tau;
		//    patterns.insert(tpat(pos - _asw_setup, -1, aswmask | rfswmask, 0));
		patterns.insert(tpat(pos - _asw_setup, ~(uint32_t)0, aswmask));
		patterns.insert(tpat(pos -
			(( !former_of_alt && comb_mode_alt) ? _alt_sep : 0), ~(uint32_t)0, trig1mask));
                
		//induce emission
		if(_induce_emission) {
			patterns.insert(tpat(pos - _induce_emission_pw/2, ~(uint32_t)0, g3mask));
			patterns.insert(tpat(pos - _induce_emission_pw/2, PAT_QAM_PULSE_IDX_INDUCE_EMISSION, PAT_QAM_PULSE_IDX_MASK));
			patterns.insert(tpat(pos - _induce_emission_pw/2, qpsk[pindem[j]], qpskmask));
			patterns.insert(tpat(pos + _induce_emission_pw/2, 0, PAT_QAM_PULSE_IDX_MASK));
			patterns.insert(tpat(pos + _induce_emission_pw/2, 0, g3mask));
		}

		//pi pulses 
		pos -= 3*_tau;
		for(int k = 0; k < echonum; k++) {
			pos += 2*_tau;
			if(_pw2/2) {
				patterns.insert(tpat(pos - _pw2/2, 0, trig2mask));
				//on
				if( !_g2_kept_p1p2) {
					patterns_cheap.insert(tpat(pos - _pw2/2 - _g2_setup, qpsk[(k == 0) ? p2[j] : p2multi[j]], qpskmask));
					patterns_cheap.insert(tpat(pos - _pw2/2 - _g2_setup, ~(uint32_t)0, g2mask));
				}

				patterns.insert(tpat(pos - _pw2/2, PAT_QAM_PULSE_IDX_P2, PAT_QAM_PULSE_IDX_MASK));
				patterns.insert(tpat(pos - _pw2/2, ~(uint32_t)0, g1mask));
				//off
				patterns.insert(tpat(pos + _pw2/2, 0, PAT_QAM_PULSE_IDX_MASK));
				patterns.insert(tpat(pos + _pw2/2, 0, g1mask));
				patterns.insert(tpat(pos + _pw2/2, 0, pulse2mask));
				patterns.insert(tpat(pos + _pw2/2, 0, g2mask));
				_g2_kept_p1p2 = false;
				//QSW
				patterns.insert(tpat(pos + _pw2/2 + _qsw_delay, ~(uint32_t)0 , qswmask));
				patterns.insert(tpat(pos + _pw2/2 + (_qsw_delay + _qsw_width/2 - _qsw_softswoff/2), 0 , qswmask));
				patterns.insert(tpat(pos + _pw2/2 + (_qsw_delay + _qsw_width/2 + _qsw_softswoff/2), ~(uint32_t)0 , qswmask));
				patterns.insert(tpat(pos + _pw2/2 + (_qsw_delay + _qsw_width), 0 , qswmask));
			}
		}
		if(_g2_kept_p1p2)
			throw XDriver::XRecordError("Inconsistent pattern of pulser setup.", __FILE__, __LINE__);

		patterns.insert(tpat(pos + _tau + _asw_hold, 0, aswmask | trig1mask));
		//induce emission
		if(_induce_emission) {
			patterns.insert(tpat(pos + _tau + _asw_hold - _induce_emission_pw/2, ~(uint32_t)0, g3mask));
			patterns.insert(tpat(pos + _tau + _asw_hold - _induce_emission_pw/2, PAT_QAM_PULSE_IDX_INDUCE_EMISSION, PAT_QAM_PULSE_IDX_MASK));
			patterns.insert(tpat(pos + _tau + _asw_hold - _induce_emission_pw/2, qpsk[pindem[j]], qpskmask));
			patterns.insert(tpat(pos + _tau + _asw_hold + _induce_emission_pw/2, 0, PAT_QAM_PULSE_IDX_MASK));
			patterns.insert(tpat(pos + _tau + _asw_hold + _induce_emission_pw/2, 0, g3mask));
		}

		if(driven_equilibrium) {
			pos += 2*_tau;
			//pi pulse 
			//on
			patterns_cheap.insert(tpat(pos - _pw2/2 - _g2_setup, qpskinv[p2[j]], qpskmask));
			patterns_cheap.insert(tpat(pos - _pw2/2 - _g2_setup, ~(uint32_t)0, g2mask));
			patterns_cheap.insert(tpat(pos - _pw2/2 - _g2_setup, ~(uint32_t)0, pulse2mask));
			patterns.insert(tpat(pos - _pw2/2, PAT_QAM_PULSE_IDX_P2, PAT_QAM_PULSE_IDX_MASK));
			patterns.insert(tpat(pos - _pw2/2, ~(uint32_t)0, g1mask));
			//off
			patterns.insert(tpat(pos + _pw2/2, 0, pulse2mask));
			patterns.insert(tpat(pos + _pw2/2, 0, PAT_QAM_PULSE_IDX_MASK));
			patterns.insert(tpat(pos + _pw2/2, 0, g1mask | g2mask));
			patterns.insert(tpat(pos + _pw2/2 + _qsw_delay, ~(uint32_t)0 , qswmask));
			if(_qsw_softswoff) {
				patterns.insert(tpat(pos + _pw2/2 + (_qsw_delay + _qsw_width/2 - _qsw_softswoff/2), 0 , qswmask));
				patterns.insert(tpat(pos + _pw2/2 + (_qsw_delay + _qsw_width/2 + _qsw_softswoff/2), ~(uint32_t)0 , qswmask));
			}
			patterns.insert(tpat(pos + _pw2/2 + (_qsw_delay + _qsw_width), 0 , qswmask));
			pos += _tau;
			//pi/2 pulse
			//on
			patterns_cheap.insert(tpat(pos - _pw1/2 - _g2_setup, qpskinv[p1[j]], qpskmask));
			patterns_cheap.insert(tpat(pos - _pw1/2 - _g2_setup, ~(uint32_t)0, pulse1mask));
			patterns_cheap.insert(tpat(pos - _pw1/2 - _g2_setup, ~(uint32_t)0, g2mask));
			patterns.insert(tpat(pos - _pw1/2, PAT_QAM_PULSE_IDX_P1, PAT_QAM_PULSE_IDX_MASK));
			patterns.insert(tpat(pos - _pw1/2, ~(uint32_t)0, g1mask));
			//off
			patterns.insert(tpat(pos + _pw1/2, 0, PAT_QAM_PULSE_IDX_MASK));
			patterns.insert(tpat(pos + _pw1/2, 0, g1mask));
			patterns.insert(tpat(pos + _pw1/2, 0, pulse1mask));
			patterns.insert(tpat(pos + _pw1/2, qpsk[p1[j]], qpskmask));
			patterns.insert(tpat(pos + _pw1/2, 0, g2mask));
			if( ! _qsw_pi_only) {
				patterns.insert(tpat(pos + _pw1/2 + _qsw_delay, ~(uint32_t)0 , qswmask));
				patterns.insert(tpat(pos + _pw1/2 + (_qsw_delay + _qsw_width/2 - _qsw_softswoff/2), 0 , qswmask));
				patterns.insert(tpat(pos + _pw1/2 + (_qsw_delay + _qsw_width/2 + _qsw_softswoff/2), ~(uint32_t)0 , qswmask));
				patterns.insert(tpat(pos + _pw1/2 + (_qsw_delay + _qsw_width), 0 , qswmask));
			}
		}
	}
    
	//insert low priority (cheap) pulses into pattern set
	for(tpatset_it it = patterns_cheap.begin(); it != patterns_cheap.end(); it++) {
		uint64_t npos = it->pos;
		for(tpatset_it kit = patterns.begin(); kit != patterns.end(); kit++) {
			//Avoid overrapping within 1 us
			uint64_t diff = llabs(kit->pos - npos);
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
    
    if(haveQAMPorts()) {
    	for(unsigned int i = 0; i < PAT_QAM_PULSE_IDX_MASK/PAT_QAM_PULSE_IDX; i++)
    		tr[ *this].m_qamWaveForm[i].clear();
    		
		double _tau = shot[ *this].tau();
		double _dif_freq = shot[ *this].difFreq();
	
		bool _induce_emission = shot[ *induceEmission()];
		double _induce_emission_phase = shot[ *induceEmissionPhase()] / 180.0 * M_PI;

		makeWaveForm(tr, PAT_QAM_PULSE_IDX_P1/PAT_QAM_PULSE_IDX - 1,
			shot[ *this].pw1()*1e-3,
			_pw1/2, pulseFunc(shot[ *p1Func()].to_str() ),
			shot[ *p1Level()], _dif_freq * 1e3, -2 * M_PI * _dif_freq * 2 * _tau);
		makeWaveForm(tr, PAT_QAM_PULSE_IDX_P2/PAT_QAM_PULSE_IDX - 1,
			shot[ *this].pw2()*1e-3,
			_pw2/2, pulseFunc(shot[ *p2Func()].to_str() ),
			shot[ *p2Level()], _dif_freq * 1e3, -2 * M_PI * _dif_freq * 2 * _tau);
		makeWaveForm(tr, PAT_QAM_PULSE_IDX_PCOMB/PAT_QAM_PULSE_IDX - 1,
			shot[ *this].combPW()*1e-3,
			_comb_pw/2, pulseFunc(shot[ *combFunc()].to_str() ),
			shot[ *combLevel()], shot[ *combOffRes()] + _dif_freq *1000.0);
		if(_induce_emission) {
			makeWaveForm(tr, PAT_QAM_PULSE_IDX_INDUCE_EMISSION/PAT_QAM_PULSE_IDX - 1,
				shot[ *this].combPW()*1e-3,
				 _induce_emission_pw/2, pulseFunc(shot[ *combFunc()].to_str() ),
				 shot[ *combLevel()], shot[ *combOffRes()] + _dif_freq *1000.0, _induce_emission_phase);
		}
    }
	createNativePatterns(tr);
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
	for(int i = 0; i < (int)to_center*2; i++) {
		double i1 = i - (int)to_center + 0.5 - delay1;
		double i2 = i - (int)to_center + 0.5 - delay2;
		double x = z * func(i1 * dx) * cos(i1 * dp + M_PI/4 + phase);
		double y = z * func(i2 * dx) * sin(i2 * dp + M_PI/4 + phase);
		p.push_back(std::complex<double>(x, y));
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
