/***************************************************************************
		Copyright (C) 2002-2007 Kentaro Kitagawa
		                   kitagawa@scphys.kyoto-u.ac.jp
		
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
#include "forms/pulserdriverform.h"
#include "forms/pulserdrivermoreform.h"
#include "pulserdriverconnector.h"
#include "interface.h"
#include "analyzer.h"
#include "xnodeconnector.h"
#include <knuminput.h>
#include <qpushbutton.h>
#include <qcheckbox.h>
#include <qstatusbar.h>
#include <kapplication.h>
#include <kiconloader.h>
#include <klocale.h>

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
#define PULSE_FUNC_HALF_COS "Half-cos BW=1.2/T"
#define PULSE_FUNC_CHOPPED_HALF_COS "Chopped-Half-cos BW=1.0/T"

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

#include <fftw.h>

double bessel_i0(double x) {
float y;
float pow = 1.0;
int s = 1;
float z = (x/2)*(x/2);
    y = 1.0;
    for(int k = 1; k < 5; k++) {
        s *= k;
        pow *= z;
        y += pow / (s*s);
    }
    return y;
}

double XPulser::pulseFuncRect(double ) {
//	return (fabs(x) <= 0.5) ? 1 : 0;
	return 1.0;
}
double XPulser::pulseFuncHanning(double x) {
	return 0.5 + 0.5*cos(2*PI*x);
}
double XPulser::pulseFuncHamming(double x) {
	return 0.54 + 0.46*cos(2*PI*x);
}
double XPulser::pulseFuncBlackman(double x) {
	return 0.42323+0.49755*cos(2*PI*x)+0.07922*cos(4*PI*x);
}
double XPulser::pulseFuncBlackmanHarris(double x) {
	return 0.35875+0.48829*cos(2*PI*x)+0.14128*cos(4*PI*x)+0.01168*cos(6*PI*x);
}
double XPulser::pulseFuncFlatTop(double x) {
	return pulseFuncHamming(x)*((fabs(x) < 1e-4) ? 1 : sin(4*PI*x)/(4*PI*x));
}
double XPulser::pulseFuncFlatTopLong(double x) {
	return pulseFuncHamming(x)*((fabs(x) < 1e-4) ? 1 : sin(6*PI*x)/(6*PI*x));
}
double XPulser::pulseFuncFlatTopLongLong(double x) {
	return pulseFuncHamming(x)*((fabs(x) < 1e-4) ? 1 : sin(8*PI*x)/(8*PI*x));
}
double XPulser::pulseFuncKaiser(double x, double alpha) {
	x = 2*x;
	x = sqrt(std::max(1 - x*x, 0.0));	
	return bessel_i0(PI*alpha*x) / bessel_i0(PI*alpha);
}
double XPulser::pulseFuncKaiser1(double x) {
	return pulseFuncKaiser(x, 3.0);
}
double XPulser::pulseFuncKaiser2(double x) {
	return pulseFuncKaiser(x, 7.2);
}
double XPulser::pulseFuncKaiser3(double x) {
	return pulseFuncKaiser(x, 15.0);
}
double XPulser::pulseFuncHalfCos(double x) {
    return cos(PI*x);
}
double XPulser::pulseFuncChoppedHalfCos(double x) {
    return 2.0*std::min(0.5, cos(PI*x));
}

XPulser::tpulsefunc
XPulser::pulseFunc(const std::string &str) const {
	if(str == PULSE_FUNC_HANNING) return &pulseFuncHanning;
	if(str == PULSE_FUNC_HAMMING) return &pulseFuncHamming;
	if(str == PULSE_FUNC_BLACKMAN) return &pulseFuncBlackman;
	if(str == PULSE_FUNC_BLACKMAN_HARRIS) return &pulseFuncBlackmanHarris;
	if(str == PULSE_FUNC_KAISER_1) return &pulseFuncKaiser1;
	if(str == PULSE_FUNC_KAISER_2) return &pulseFuncKaiser2;
	if(str == PULSE_FUNC_KAISER_3) return &pulseFuncKaiser3;
	if(str == PULSE_FUNC_FLATTOP) return &pulseFuncFlatTop;
	if(str == PULSE_FUNC_FLATTOP_LONG) return &pulseFuncFlatTopLong;
	if(str == PULSE_FUNC_FLATTOP_LONG_LONG) return &pulseFuncFlatTopLongLong;
    if(str == PULSE_FUNC_HALF_COS) return &pulseFuncHalfCos;
    if(str == PULSE_FUNC_CHOPPED_HALF_COS) return &pulseFuncChoppedHalfCos;
	return &pulseFuncRect;
}

XPulser::XPulser(const char *name, bool runtime, 
   const shared_ptr<XScalarEntryList> &scalarentries,
   const shared_ptr<XInterfaceList> &interfaces,
   const shared_ptr<XThermometerList> &thermometers,
   const shared_ptr<XDriverList> &drivers) :
    XPrimaryDriver(name, runtime, scalarentries, interfaces, thermometers, drivers),
    m_output(create<XBoolNode>("Output", true)),
    m_combMode(create<XComboNode>("CombMode", false, true)),
    m_rtMode(create<XComboNode>("RTMode", false, true)),
    m_rt(create<XDoubleNode>("RT", false)),
    m_tau(create<XDoubleNode>("Tau", false)),
    m_combPW(create<XDoubleNode>("CombPW", false)),
    m_pw1(create<XDoubleNode>("PW1", false)),
    m_pw2(create<XDoubleNode>("PW2", false)),
    m_combNum(create<XUIntNode>("CombNum", false)),
    m_combPT(create<XDoubleNode>("CombPT", false)),
    m_combP1(create<XDoubleNode>("CombP1", false)),
    m_combP1Alt(create<XDoubleNode>("CombP1Alt", false)),
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
    m_invertPhase(create<XBoolNode>("InvertPhase", false)),
    m_qswPiPulseOnly(create<XBoolNode>("QSWPiPulseOnly", false)),
    m_moreConfigShow(create<XNode>("MoreConfigShow", true)),
    m_form(new FrmPulser(g_pFrmMain)),
    m_formMore(new FrmPulserMore(g_pFrmMain))
{
	{
		QComboBox*const combo[] = {
			m_formMore->m_cmbPortSel0, m_formMore->m_cmbPortSel1, m_formMore->m_cmbPortSel2, m_formMore->m_cmbPortSel3,
			m_formMore->m_cmbPortSel4, m_formMore->m_cmbPortSel5, m_formMore->m_cmbPortSel6, m_formMore->m_cmbPortSel7, 
			m_formMore->m_cmbPortSel8, m_formMore->m_cmbPortSel9, m_formMore->m_cmbPortSel10, m_formMore->m_cmbPortSel11, 
			m_formMore->m_cmbPortSel12, m_formMore->m_cmbPortSel13, m_formMore->m_cmbPortSel14, m_formMore->m_cmbPortSel15
		};
  		const char *desc[] = {
  			"Gate", "PreGate", "Gate3", "Trig1", "Trig2", "ASW",
  			"QSW", "Pulse1", "Pulse2", "Comb", "CombFM",
  			"QPSK-A", "QPSK-B", "QPSK-NonInv", "QPSK-Inv",
  			"QPSK-PS-Gate", 0L
  		};
		for(unsigned int i = 0; i < NUM_DO_PORTS; i++) {
			m_portSel[i] = create<XComboNode>(formatString("PortSel%u", i).c_str(), false);
			m_conPortSel[i] = xqcon_create<XQComboBoxConnector>(m_portSel[i], combo[i]);
			for(const char **p = &desc[0]; *p; p++)
				m_portSel[i]->add(*p);
//			m_portSel[i]->setUIEnabled(false);
		}
		portSel(0)->value(PORTSEL_GATE);
		portSel(1)->value(PORTSEL_PREGATE);
	}
	
    m_form->m_btnMoreConfig->setIconSet(
             KApplication::kApplication()->iconLoader()->loadIconSet("configure", 
            KIcon::Toolbar, KIcon::SizeSmall, true ) );     
    
  rtime()->value(100.0);
  tau()->value(100.0);
  combPW()->value(5.0);
  pw1()->value(5.0);
  pw2()->value(10.0);
  combNum()->value(1);
  combPT()->value(20.0);
  combP1()->value(0.2);
  combP1Alt()->value(50.0);
  aswSetup()->value(0.02);
  aswHold()->value(0.23);
  altSep()->value(0.25);
  g2Setup()->value(5.0);
  echoNum()->value(1);
  combOffRes()->value(0.0);
  drivenEquilibrium()->value(false);
  numPhaseCycle()->add(NUM_PHASE_CYCLE_1);
  numPhaseCycle()->add(NUM_PHASE_CYCLE_2);
  numPhaseCycle()->add(NUM_PHASE_CYCLE_4);
  numPhaseCycle()->add(NUM_PHASE_CYCLE_8);
  numPhaseCycle()->add(NUM_PHASE_CYCLE_16);
  numPhaseCycle()->value(NUM_PHASE_CYCLE_16);
  p1Level()->value(-5.0);
  p2Level()->value(-0.5);
  combLevel()->value(-5.0);
  masterLevel()->value(-10.0);
  qamLevel1()->value(1.0);
  qamLevel2()->value(1.0);
  qswDelay()->value(5.0);
  qswWidth()->value(10.0);
 
  combMode()->add(COMB_MODE_OFF);
  combMode()->add(COMB_MODE_ON);
  combMode()->add(COMB_MODE_P1_ALT);
  combMode()->add(COMB_MODE_COMB_ALT);
  combMode()->value(N_COMB_MODE_OFF);
  rtMode()->add(RT_MODE_FIXREP);
  rtMode()->add(RT_MODE_FIXREST);
  rtMode()->value(1);

  p1Func()->add(PULSE_FUNC_RECT);
  p1Func()->add(PULSE_FUNC_HANNING);
  p1Func()->add(PULSE_FUNC_HAMMING);
  p1Func()->add(PULSE_FUNC_BLACKMAN);
  p1Func()->add(PULSE_FUNC_BLACKMAN_HARRIS);
  p1Func()->add(PULSE_FUNC_FLATTOP);
  p1Func()->add(PULSE_FUNC_FLATTOP_LONG);
  p1Func()->add(PULSE_FUNC_FLATTOP_LONG_LONG);
  p1Func()->add(PULSE_FUNC_KAISER_1);
  p1Func()->add(PULSE_FUNC_KAISER_2);
  p1Func()->add(PULSE_FUNC_KAISER_3);
  p1Func()->add(PULSE_FUNC_HALF_COS);
  p1Func()->add(PULSE_FUNC_CHOPPED_HALF_COS);
    p2Func()->add(PULSE_FUNC_RECT);
    p2Func()->add(PULSE_FUNC_HANNING);
    p2Func()->add(PULSE_FUNC_HAMMING);
    p2Func()->add(PULSE_FUNC_BLACKMAN);
    p2Func()->add(PULSE_FUNC_BLACKMAN_HARRIS);
    p2Func()->add(PULSE_FUNC_FLATTOP);
    p2Func()->add(PULSE_FUNC_FLATTOP_LONG);
    p2Func()->add(PULSE_FUNC_FLATTOP_LONG_LONG);
    p2Func()->add(PULSE_FUNC_KAISER_1);
    p2Func()->add(PULSE_FUNC_KAISER_2);
    p2Func()->add(PULSE_FUNC_KAISER_3);
    p2Func()->add(PULSE_FUNC_HALF_COS);
    p2Func()->add(PULSE_FUNC_CHOPPED_HALF_COS);
  combFunc()->add(PULSE_FUNC_RECT);
  combFunc()->add(PULSE_FUNC_HANNING);
  combFunc()->add(PULSE_FUNC_HAMMING);
  combFunc()->add(PULSE_FUNC_BLACKMAN);
  combFunc()->add(PULSE_FUNC_BLACKMAN_HARRIS);
  combFunc()->add(PULSE_FUNC_FLATTOP);
  combFunc()->add(PULSE_FUNC_FLATTOP_LONG);
  combFunc()->add(PULSE_FUNC_FLATTOP_LONG_LONG);
  combFunc()->add(PULSE_FUNC_KAISER_1);
  combFunc()->add(PULSE_FUNC_KAISER_2);
  combFunc()->add(PULSE_FUNC_KAISER_3);
  combFunc()->add(PULSE_FUNC_HALF_COS);
  combFunc()->add(PULSE_FUNC_CHOPPED_HALF_COS);
    
  p1Func()->value(PULSE_FUNC_KAISER_2);
  p2Func()->value(PULSE_FUNC_KAISER_2);
  combFunc()->value(PULSE_FUNC_FLATTOP);
  
  m_form->setCaption(KAME::i18n("Pulser Control") + " - " + getLabel() );
  m_formMore->setCaption(KAME::i18n("Pulser Control More Config.") + " - " + getLabel() );
  m_form->statusBar()->hide();
  m_formMore->statusBar()->hide();
  
  m_conMoreConfigShow = xqcon_create<XQButtonConnector>(
        m_moreConfigShow, m_form->m_btnMoreConfig);
  m_lsnOnMoreConfigShow = m_moreConfigShow->onTouch().connectWeak(
                      true, shared_from_this(), &XPulser::onMoreConfigShow, true);
  
  m_conOutput = xqcon_create<XQToggleButtonConnector>(m_output, m_form->m_ckbOutput);
  m_conCombMode = xqcon_create<XQComboBoxConnector>(m_combMode, m_form->m_cmbCombMode);
  m_conRTMode = xqcon_create<XQComboBoxConnector>(m_rtMode, m_form->m_cmbRTMode);
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
  m_conNumPhaseCycle = xqcon_create<XQComboBoxConnector>(m_numPhaseCycle, m_formMore->m_cmbPhaseCycle);
  m_conCombOffRes = xqcon_create<XQLineEditConnector>(m_combOffRes, m_form->m_edCombOffRes);
  m_conInvertPhase = xqcon_create<XQToggleButtonConnector>(m_invertPhase, m_formMore->m_ckbInvertPhase);
  m_conP1Func = xqcon_create<XQComboBoxConnector>(m_p1Func, m_form->m_cmbP1Func);
  m_conP2Func = xqcon_create<XQComboBoxConnector>(m_p2Func, m_form->m_cmbP2Func);
  m_conCombFunc = xqcon_create<XQComboBoxConnector>(m_combFunc, m_form->m_cmbCombFunc);
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
  qswPiPulseOnly()->setUIEnabled(false);
  invertPhase()->setUIEnabled(false);
  
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
XPulser::onMoreConfigShow(const shared_ptr<XNode> &) 
{
   m_formMore->show();
   m_formMore->raise();
}

void
XPulser::start()
{
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
  qswPiPulseOnly()->setUIEnabled(true);
  invertPhase()->setUIEnabled(true);
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

  m_lsnOnPulseChanged = combMode()->onValueChanged().connectWeak(
                         false, shared_from_this(), &XPulser::onPulseChanged);
  rtime()->onValueChanged().connect(m_lsnOnPulseChanged);
  tau()->onValueChanged().connect(m_lsnOnPulseChanged);
  combPW()->onValueChanged().connect(m_lsnOnPulseChanged);
  pw1()->onValueChanged().connect(m_lsnOnPulseChanged);
  pw2()->onValueChanged().connect(m_lsnOnPulseChanged);
  combNum()->onValueChanged().connect(m_lsnOnPulseChanged);
  combPT()->onValueChanged().connect(m_lsnOnPulseChanged);
  combP1()->onValueChanged().connect(m_lsnOnPulseChanged);
  combP1Alt()->onValueChanged().connect(m_lsnOnPulseChanged);
  aswSetup()->onValueChanged().connect(m_lsnOnPulseChanged);
  aswHold()->onValueChanged().connect(m_lsnOnPulseChanged);
  altSep()->onValueChanged().connect(m_lsnOnPulseChanged);
  output()->onValueChanged().connect(m_lsnOnPulseChanged);
  g2Setup()->onValueChanged().connect(m_lsnOnPulseChanged);
  echoNum()->onValueChanged().connect(m_lsnOnPulseChanged);
  drivenEquilibrium()->onValueChanged().connect(m_lsnOnPulseChanged);
  numPhaseCycle()->onValueChanged().connect(m_lsnOnPulseChanged);
  combOffRes()->onValueChanged().connect(m_lsnOnPulseChanged);
  induceEmission()->onValueChanged().connect(m_lsnOnPulseChanged);    
  induceEmissionPhase()->onValueChanged().connect(m_lsnOnPulseChanged);    
  qswDelay()->onValueChanged().connect(m_lsnOnPulseChanged);
  qswWidth()->onValueChanged().connect(m_lsnOnPulseChanged);
  qswPiPulseOnly()->onValueChanged().connect(m_lsnOnPulseChanged);
  invertPhase()->onValueChanged().connect(m_lsnOnPulseChanged);
  for(unsigned int i = 0; i < NUM_DO_PORTS; i++) {
  	portSel(i)->onValueChanged().connect(m_lsnOnPulseChanged);
  }
  
  if(haveQAMPorts()) {
	  p1Func()->onValueChanged().connect(m_lsnOnPulseChanged);
	  p2Func()->onValueChanged().connect(m_lsnOnPulseChanged);
	  combFunc()->onValueChanged().connect(m_lsnOnPulseChanged);
	  p1Level()->onValueChanged().connect(m_lsnOnPulseChanged);
	  p2Level()->onValueChanged().connect(m_lsnOnPulseChanged);
	  combLevel()->onValueChanged().connect(m_lsnOnPulseChanged);
	  masterLevel()->onValueChanged().connect(m_lsnOnPulseChanged);
	  qamOffset1()->onValueChanged().connect(m_lsnOnPulseChanged);
	  qamOffset2()->onValueChanged().connect(m_lsnOnPulseChanged);
	  qamLevel1()->onValueChanged().connect(m_lsnOnPulseChanged);
	  qamLevel2()->onValueChanged().connect(m_lsnOnPulseChanged);
	  qamDelay1()->onValueChanged().connect(m_lsnOnPulseChanged);
	  qamDelay2()->onValueChanged().connect(m_lsnOnPulseChanged);
	  difFreq()->onValueChanged().connect(m_lsnOnPulseChanged);    
  }
}
void
XPulser::stop()
{
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
  qswPiPulseOnly()->setUIEnabled(false);
  invertPhase()->setUIEnabled(false);
  for(unsigned int i = 0; i < NUM_DO_PORTS; i++) {
//	m_portSel[i]->setUIEnabled(false);
	m_portSel[i]->setUIEnabled(true);
  }
  
  afterStop();
//    m_thread->waitFor();
//  thread must do interface()->close() at the end
}

inline double
XPulser::rintTermMilliSec(double msec) const {
	const double res = resolution();
	return rint(msec / res) * res;
}
inline double
XPulser::rintTermMicroSec(double usec) const {
	const double res = resolution() * 1e3;
	return rint(usec / res) * res;
}
inline uint64_t
XPulser::ceilSampsMicroSec(double usec) const {
	const double res = resolution() * 1e3;
	return llrint(usec / res + 0.499);
}
inline uint64_t
XPulser::rintSampsMicroSec(double usec) const {
	const double res = resolution() * 1e3;
	return llrint(usec / res);
}
inline uint64_t
XPulser::rintSampsMilliSec(double msec) const {
	const double res = resolution();
	return llrint(msec / res);
}
void
XPulser::analyzeRaw() throw (XRecordError&)
{
    m_combModeRecorded = pop<short>();
    pop<short>(); //reserve
    m_rtimeRecorded = pop<double>();
    m_tauRecorded = pop<double>();
    m_pw1Recorded = pop<double>();
    m_pw2Recorded = pop<double>();
    m_combP1Recorded = pop<double>();
    m_altSepRecorded = pop<double>();
    m_combP1AltRecorded = pop<double>();
    m_aswSetupRecorded = pop<double>();
    m_aswHoldRecorded = pop<double>();
    try {
        //! ver 2 records
        m_difFreqRecorded = pop<double>();
        m_combPWRecorded = pop<double>();
        m_combPTRecorded = pop<double>();
        m_echoNumRecorded = pop<unsigned short>();
        m_combNumRecorded = pop<unsigned short>();
        m_rtModeRecorded = pop<short>();
        m_numPhaseCycleRecorded = pop<unsigned short>();
        //! ver 3 records
        m_invertPhaseRecorded = pop<unsigned short>();
    }
    catch (XRecordError &) {
        m_difFreqRecorded = *difFreq();
        m_combPWRecorded = *combPW();
        m_combPTRecorded = *combPT();
        m_echoNumRecorded = *echoNum();
        m_combNumRecorded = *combNum();
        m_rtModeRecorded = *rtMode();
      int npat = 16;
      if(numPhaseCycle()->to_str() == NUM_PHASE_CYCLE_1) npat = 1;
      if(numPhaseCycle()->to_str() == NUM_PHASE_CYCLE_2) npat = 2;
      if(numPhaseCycle()->to_str() == NUM_PHASE_CYCLE_4) npat = 4;
      if(numPhaseCycle()->to_str() == NUM_PHASE_CYCLE_8) npat = 8;
      if(numPhaseCycle()->to_str() == NUM_PHASE_CYCLE_16) npat = 16;
        m_numPhaseCycleRecorded = npat;
        m_invertPhaseRecorded = 0.0;
    }    
    rawToRelPat();
}

void
XPulser::visualize()
{
 //! impliment extra codes which do not need write-lock of record
 //! record is read-locked
}

void
XPulser::onPulseChanged(const shared_ptr<XValueNodeBase> &node)
{
  XTime time_awared = XTime::now();
  
  const double _tau = rintTermMicroSec(*tau());
  const double _asw_setup = rintTermMilliSec(*aswSetup());
  const double _asw_hold = rintTermMilliSec(*aswHold());
  const double _alt_sep = rintTermMilliSec(*altSep());
  const int _echo_num = *echoNum();
  if(_asw_setup > 2.0 * _tau)
    aswSetup()->value(2.0 * _tau);
  if(node != altSep())
    if(_alt_sep != _asw_setup + _asw_hold + (_echo_num - 1) * 2 * _tau/1000)
      {
            altSep()->value(_asw_setup + _asw_hold + (_echo_num - 1) * 2 * _tau/1000);
        return;
      }

  clearRaw();

  const unsigned int blankpattern = selectedPorts(PORTSEL_COMB_FM);
  
  if(!*output())
    {
      finishWritingRaw(XTime(), XTime());
	  try {
	      changeOutput(false, blankpattern);
	  }
	  catch (XKameError &e) {
	      e.print(getLabel() + KAME::i18n("Pulser Turn-Off Failed, because"));
	  }
      return;
    }

/*  try {
      changeOutput(false, blankpattern);
  }
  catch (XKameError &e) {
      e.print(getLabel() + KAME::i18n("Pulser Turn-Off Failed, because"));
      return;
  }
*/    
//! ver 1 records below
    push((short)*combMode());
    push((short)0); //reserve
    push((double)rintTermMilliSec(*rtime()));
    push((double)_tau);
    push((double)*pw1());
    push((double)*pw2());
    push((double)rintTermMilliSec(*combP1()));
    push((double)_alt_sep);
    push((double)rintTermMilliSec(*combP1Alt()));
    push((double)_asw_setup);
    push((double)_asw_hold);
//! ver 2 records below
    push((double)*difFreq());
    push((double)*combPW());
    push((double)rintTermMicroSec(*combPT()));
    push((unsigned short)*echoNum());
    push((unsigned short)*combNum());
    push((short)*rtMode());
  int npat = 16;
  if(numPhaseCycle()->to_str() == NUM_PHASE_CYCLE_1) npat = 1;
  if(numPhaseCycle()->to_str() == NUM_PHASE_CYCLE_2) npat = 2;
  if(numPhaseCycle()->to_str() == NUM_PHASE_CYCLE_4) npat = 4;
  if(numPhaseCycle()->to_str() == NUM_PHASE_CYCLE_8) npat = 8;
  if(numPhaseCycle()->to_str() == NUM_PHASE_CYCLE_16) npat = 16;
   push((unsigned short)npat);
//! ver 3 records below
    push((unsigned short)*invertPhase());

  finishWritingRaw(time_awared, XTime::now());
  
  try {
	  createNativePatterns();
      changeOutput(true, blankpattern);
  }
  catch (XKameError &e) {
      e.print(getLabel() + KAME::i18n("Pulser Turn-On Failed, because"));
  } 
}
double
XPulser::periodicTermRecorded() const {
    ASSERT(!m_relPatList.empty());
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
XPulser::selectedPorts(int func) const
{
	unsigned int mask = 0;
	for(unsigned int i = 0; i < NUM_DO_PORTS; i++) {
		if(*portSel(i) == func)
			mask |= 1 << i;
	}
	return mask;
}
void
XPulser::rawToRelPat() throw (XRecordError&)
{
  const unsigned int g3mask = selectedPorts(PORTSEL_GATE3);
  const unsigned int g2mask = selectedPorts(PORTSEL_PREGATE);
  const unsigned int g1mask = (selectedPorts(PORTSEL_GATE) | g3mask);
  const unsigned int trig1mask = selectedPorts(PORTSEL_TRIG1);
  const unsigned int trig2mask = selectedPorts(PORTSEL_TRIG2);
  const unsigned int aswmask = selectedPorts(PORTSEL_ASW);
  const unsigned int qswmask = selectedPorts(PORTSEL_QSW);
  const unsigned int pulse1mask = selectedPorts(PORTSEL_PULSE1);
  const unsigned int pulse2mask = selectedPorts(PORTSEL_PULSE2);
  const unsigned int combmask = selectedPorts(PORTSEL_COMB);
  const unsigned int combfmmask = selectedPorts(PORTSEL_COMB_FM);
  const unsigned int qpskamask = selectedPorts(PORTSEL_QPSK_A);
  const unsigned int qpskbmask = selectedPorts(PORTSEL_QPSK_B);
  const unsigned int qpsknoninvmask = selectedPorts(PORTSEL_QPSK_OLD_NONINV);
  const unsigned int qpskinvmask = selectedPorts(PORTSEL_QPSK_OLD_INV);
  const unsigned int qpskpsgatemask = selectedPorts(PORTSEL_QPSK_OLD_PSGATE);
  const unsigned int qpskmask = qpskamask | qpskbmask |
  	 qpskinvmask | qpsknoninvmask | qpskpsgatemask | PAT_QAM_PHASE_MASK;
	
  const uint64_t _rtime = rintSampsMilliSec(m_rtimeRecorded);
  const uint64_t _tau = rintSampsMicroSec(m_tauRecorded);
  const uint64_t _asw_setup = rintSampsMilliSec(m_aswSetupRecorded);
  const uint64_t _asw_hold = rintSampsMilliSec(m_aswHoldRecorded);
  const uint64_t _alt_sep = rintSampsMilliSec(m_altSepRecorded);
  const uint64_t _pw1 = haveQAMPorts() ? 
  	ceilSampsMicroSec(m_pw1Recorded/2)*2 : rintSampsMicroSec(m_pw1Recorded/2)*2;
  const uint64_t _pw2 = haveQAMPorts() ?
  	ceilSampsMicroSec(m_pw2Recorded/2)*2 : rintSampsMicroSec(m_pw2Recorded/2)*2;
  const uint64_t _comb_pw = haveQAMPorts() ?
  	ceilSampsMicroSec(m_combPWRecorded/2)*2 : rintSampsMicroSec(m_combPWRecorded/2)*2;
  const uint64_t _comb_pt = rintSampsMicroSec(m_combPTRecorded);
  const uint64_t _comb_p1 = rintSampsMilliSec(m_combP1Recorded);
  const uint64_t _comb_p1_alt = rintSampsMilliSec(m_combP1AltRecorded);
  const uint64_t _g2_setup = ceilSampsMicroSec(*g2Setup());
  const int _echo_num = m_echoNumRecorded;
  const int _comb_num = m_combNumRecorded;
  const int _comb_mode = m_combModeRecorded;
  const int _rt_mode = m_rtModeRecorded;
  int _num_phase_cycle = m_numPhaseCycleRecorded;
  
  const bool comb_mode_alt = ((_comb_mode == N_COMB_MODE_P1_ALT) ||
            (_comb_mode == N_COMB_MODE_COMB_ALT));
  const bool saturation_wo_comb = (_comb_num == 0);
  const bool driven_equilibrium = *drivenEquilibrium();
  const uint64_t _qsw_delay = rintSampsMicroSec(*qswDelay());
  const uint64_t _qsw_width = rintSampsMicroSec(*qswWidth());
  const bool _qsw_pi_only = *qswPiPulseOnly();
  const int comb_rot_num = lrint(*combOffRes() * (m_combPWRecorded / 1000.0 * 4));
  
  const bool _induce_emission = *induceEmission();
  const uint64_t _induce_emission_pw = _comb_pw;
  if((_comb_mode == N_COMB_MODE_OFF))
  	 _num_phase_cycle = std::min(_num_phase_cycle, 4);
  
  const bool _invert_phase = m_invertPhaseRecorded;

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
  const uint32_t comb[MAX_NUM_PHASE_CYCLE] = {
    1, 3, 0, 2, 3, 1, 2, 0, 0, 2, 1, 3, 2, 0, 3, 1
  };
  //induced emission phases
  const uint32_t pindem[MAX_NUM_PHASE_CYCLE] = {
    0, 0, 0, 0, 2, 2, 2, 2, 0, 0, 0, 0, 2, 2, 2, 2
  };

  //pi/2 pulse phases
  const uint32_t p1single[MAX_NUM_PHASE_CYCLE] = {
    0, 0, 2, 2, 0, 0, 2, 2, 0, 0, 2, 2, 0, 0, 2, 2
  };
  //pi pulse phases
  const uint32_t p2single[MAX_NUM_PHASE_CYCLE] = {
    0, 2, 1, 3, 0, 2, 1, 3, 0, 2, 1, 3, 0, 2, 1, 3
  };
  //pi/2 pulse phases for multiple echoes
  const uint32_t p1multi[MAX_NUM_PHASE_CYCLE] = {
    2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2
  };
  //pi pulse phases for multiple echoes
  const uint32_t p2multi[MAX_NUM_PHASE_CYCLE] = {
    1, 3, 1, 3, 1, 3, 1, 3, 1, 3, 1, 3, 1, 3, 1, 3
  };
  
  const uint32_t _qpsk_driven_equilibrium[4] = {2, 1, 0, 3};
  #define qpsk_driven_equilibrium(phase) qpsk[_qpsk_driven_equilibrium[(phase) % 4]]
  #define qpsk_driven_equilibrium_inv(phase) (qpsk_driven_equilibrium(((phase) + 2) % 4))

  typedef std::multiset<tpat, std::less<tpat> > tpatset;
  tpatset patterns;  // patterns
  tpatset patterns_cheap; //Low priority patterns
  typedef std::multiset<tpat, std::less<tpat> >::iterator tpatset_it;

  m_relPatList.clear();
  
  uint64_t pos = 0;
            
  int echonum = _echo_num;
  const uint32_t *const p1 = (echonum > 1) ? p1multi : p1single;
  const uint32_t *const p2 = (echonum > 1) ? p2multi : p2single;
  
  bool former_of_alt = !_invert_phase;
  for(int i = 0; i < _num_phase_cycle * (comb_mode_alt ? 2 : 1); i++)
    {
      int j = (i / (comb_mode_alt ? 2 : 1)) % _num_phase_cycle; //index for phase cycling
      if(_invert_phase)
      	j = _num_phase_cycle - 1 - j;
      former_of_alt = !former_of_alt;
      bool comb_off_res = ((_comb_mode != N_COMB_MODE_COMB_ALT) || former_of_alt) && (comb_rot_num != 0);
            
      uint64_t _p1 = 0;
      if((_comb_mode != N_COMB_MODE_OFF) &&
     !((_comb_mode == N_COMB_MODE_COMB_ALT) && former_of_alt && !(comb_rot_num != 0)))
      {
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
     if((_p1 > 0) && !saturation_wo_comb)
     {
     const uint64_t combpt = std::max(_comb_pt, _comb_pw);
     uint64_t cpos = pos - combpt*_comb_num;
     
      patterns_cheap.insert(tpat(cpos - _comb_pw/2 - _g2_setup,
                     g2mask, g2mask));
      patterns_cheap.insert(tpat(cpos - _comb_pw/2 - _g2_setup, comb_off_res ? ~(uint32_t)0 : 0, combfmmask));
      patterns_cheap.insert(tpat(cpos - _comb_pw/2 - _g2_setup, ~(uint32_t)0, combmask));
        for(int k = 0; k < _comb_num; k++)
        {
          patterns.insert(tpat(cpos + _comb_pw/2 , qpsk[comb[j]], qpskmask));
          cpos += combpt;
          cpos -= _comb_pw/2;
          patterns.insert(tpat(cpos, ~(uint32_t)0, g1mask));
          patterns.insert(tpat(cpos, PAT_QAM_PULSE_IDX_PCOMB, PAT_QAM_PULSE_IDX_MASK));
          cpos += _comb_pw;      
          patterns.insert(tpat(cpos, 0 , g1mask));
          patterns.insert(tpat(cpos, 0, PAT_QAM_PULSE_IDX_MASK));

          cpos -= _comb_pw/2;
        }
      patterns.insert(tpat(cpos + _comb_pw/2, 0, g2mask));
      patterns.insert(tpat(cpos + _comb_pw/2, 0, combmask));
      patterns.insert(tpat(cpos + _comb_pw/2, ~(uint32_t)0, combfmmask));
      if(! _qsw_pi_only) {
          patterns.insert(tpat(cpos + _comb_pw/2 + _qsw_delay, ~(uint32_t)0 , qswmask));
          patterns.insert(tpat(cpos + _comb_pw/2 + (_qsw_delay + _qsw_width), 0 , qswmask));
      }
     }   
      pos += _p1;
       
       //pi/2 pulse
      //on
      patterns_cheap.insert(tpat(pos - _pw1/2 - _g2_setup, qpsk[p1[j]], qpskmask));
      patterns_cheap.insert(tpat(pos - _pw1/2 - _g2_setup, ~(uint32_t)0, pulse1mask));
      patterns_cheap.insert(tpat(pos - _pw1/2 - _g2_setup, ~(uint32_t)0, g2mask));
      patterns.insert(tpat(pos - _pw1/2, PAT_QAM_PULSE_IDX_P1, PAT_QAM_PULSE_IDX_MASK));
      patterns.insert(tpat(pos - _pw1/2, ~(uint32_t)0, g1mask | trig2mask));
      //off
      patterns.insert(tpat(pos + _pw1/2, 0, g1mask));
      patterns.insert(tpat(pos + _pw1/2, 0, PAT_QAM_PULSE_IDX_MASK));
      patterns.insert(tpat(pos + _pw1/2, 0, pulse1mask));
      patterns.insert(tpat(pos + _pw1/2, qpsk[p2[j]], qpskmask));
      patterns.insert(tpat(pos + _pw1/2, ~(uint32_t)0, pulse2mask));
      if(! _qsw_pi_only) {
          patterns.insert(tpat(pos + _pw1/2 + _qsw_delay, ~(uint32_t)0 , qswmask));
          patterns.insert(tpat(pos + _pw1/2 + (_qsw_delay + _qsw_width), 0 , qswmask));
      }
     
      //2tau
      pos += 2*_tau;
      //    patterns.insert(tpat(pos - _asw_setup, -1, aswmask | rfswmask, 0));
      patterns.insert(tpat(pos - _asw_setup, ~(uint32_t)0, aswmask));
      patterns.insert(tpat(pos -
               ((!former_of_alt && comb_mode_alt) ? _alt_sep : 0), ~(uint32_t)0, trig1mask));
                
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
      for(int k = 0;k < echonum; k++)
      {
          pos += 2*_tau;
          //on
      	  if(k >= 1) {
              patterns_cheap.insert(tpat(pos - _pw2/2 - _g2_setup, ~(uint32_t)0, g2mask));
          }
          patterns.insert(tpat(pos - _pw2/2, 0, trig2mask));
          patterns.insert(tpat(pos - _pw2/2, PAT_QAM_PULSE_IDX_P2, PAT_QAM_PULSE_IDX_MASK));
          patterns.insert(tpat(pos - _pw2/2, ~(uint32_t)0, g1mask));
          //off
          patterns.insert(tpat(pos + _pw2/2, 0, pulse2mask));
          patterns.insert(tpat(pos + _pw2/2, 0, PAT_QAM_PULSE_IDX_MASK));
          patterns.insert(tpat(pos + _pw2/2, 0, g1mask));
          patterns.insert(tpat(pos + _pw2/2, 0, g2mask));
          patterns.insert(tpat(pos + _pw2/2 + _qsw_delay, ~(uint32_t)0 , qswmask));
          patterns.insert(tpat(pos + _pw2/2 + (_qsw_delay + _qsw_width), 0 , qswmask));
      }

       patterns.insert(tpat(pos + _tau + _asw_hold, 0, aswmask | trig1mask));
      //induce emission
      if(_induce_emission) {
          patterns.insert(tpat(pos + _tau + _asw_hold - _induce_emission_pw/2, ~(uint32_t)0, g3mask));
          patterns.insert(tpat(pos + _tau + _asw_hold - _induce_emission_pw/2, PAT_QAM_PULSE_IDX_INDUCE_EMISSION, PAT_QAM_PULSE_IDX_MASK));
          patterns.insert(tpat(pos + _tau + _asw_hold - _induce_emission_pw/2, qpsk[pindem[j]], qpskmask));
          patterns.insert(tpat(pos + _tau + _asw_hold + _induce_emission_pw/2, 0, PAT_QAM_PULSE_IDX_MASK));
          patterns.insert(tpat(pos + _tau + _asw_hold + _induce_emission_pw/2, 0, g3mask));
      }

      if(driven_equilibrium)
      {
        pos += 2*_tau;
        //pi pulse 
        //on
        patterns_cheap.insert(tpat(pos - _pw2/2 - _g2_setup, qpsk_driven_equilibrium(p2[j]), qpskmask));
        patterns_cheap.insert(tpat(pos - _pw2/2 - _g2_setup, ~(uint32_t)0, g2mask));
        patterns_cheap.insert(tpat(pos - _pw2/2 - _g2_setup, ~(uint32_t)0, pulse2mask));
        patterns.insert(tpat(pos - _pw2/2, PAT_QAM_PULSE_IDX_P2, PAT_QAM_PULSE_IDX_MASK));
        patterns.insert(tpat(pos - _pw2/2, ~(uint32_t)0, g1mask));
        //off
        patterns.insert(tpat(pos + _pw2/2, 0, pulse2mask));
        patterns.insert(tpat(pos + _pw2/2, 0, PAT_QAM_PULSE_IDX_MASK));
        patterns.insert(tpat(pos + _pw2/2, 0, g1mask | g2mask));
        patterns.insert(tpat(pos + _pw2/2 + _qsw_delay, ~(uint32_t)0 , qswmask));
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
        if(! _qsw_pi_only) {
            patterns.insert(tpat(pos + _pw1/2 + _qsw_delay, ~(uint32_t)0 , qswmask));
            patterns.insert(tpat(pos + _pw1/2 + (_qsw_delay + _qsw_width), 0 , qswmask));
        }
      }
    }
    
  //insert low priority (cheap) pulses into pattern set
  for(tpatset_it it = patterns_cheap.begin(); it != patterns_cheap.end(); it++)
    {
      uint64_t npos = it->pos;
      for(tpatset_it kit = patterns.begin(); kit != patterns.end(); kit++)
	    {
	      //Avoid overrapping within 1 us
	      uint64_t diff = llabs(kit->pos - npos);
	      diff -= pos * (diff / pos);
	      if(diff < rintSampsMilliSec(minPulseWidth()))
	        {
	          npos = kit->pos;
	          break;
	        }
	    }
      patterns.insert(tpat(npos, it->pat, it->mask));
    }

  uint64_t curpos = patterns.begin()->pos;
  uint64_t lastpos = 0;
  uint32_t pat = 0;
  for(tpatset_it it = patterns.begin(); it != patterns.end(); it++)
    {
      lastpos = it->pos - pos;
      pat &= ~it->mask;
      pat |= (it->pat & it->mask);
    }
    
  for(tpatset_it it = patterns.begin(); it != patterns.end();)
    {
      pat &= ~it->mask;
      pat |= (it->pat & it->mask);
      it++;
      if((it == patterns.end()) || (it->pos != curpos))
        {
        RelPat relpat(pat, curpos, curpos - lastpos);
        
            m_relPatList.push_back(relpat);
                
            if(it == patterns.end()) break;
            lastpos = curpos;
            curpos = it->pos;
        }
    }
    
    if(haveQAMPorts()) {
    	for(unsigned int i = 0; i < PAT_QAM_PULSE_IDX_MASK/PAT_QAM_PULSE_IDX; i++)
    		m_qamWaveForm[i].clear();
    		
	  const double _tau = m_tauRecorded;
	  const double _dif_freq = m_difFreqRecorded;
	
	  const bool _induce_emission = *induceEmission();
	  const double _induce_emission_phase = *induceEmissionPhase() / 180.0 * PI;

		  makeWaveForm(PAT_QAM_PULSE_IDX_P1/PAT_QAM_PULSE_IDX - 1, m_pw1Recorded*1e-3,
			  	_pw1/2, pulseFunc(p1Func()->to_str() ),
			  *p1Level(), _dif_freq * 1e3, -2 * PI * _dif_freq * 2 * _tau);
		  makeWaveForm(PAT_QAM_PULSE_IDX_P2/PAT_QAM_PULSE_IDX - 1, m_pw2Recorded*1e-3, 
			   _pw2/2, pulseFunc(p2Func()->to_str() ),
			  *p2Level(), _dif_freq * 1e3, -2 * PI * _dif_freq * 2 * _tau);
		  makeWaveForm(PAT_QAM_PULSE_IDX_PCOMB/PAT_QAM_PULSE_IDX - 1, m_combPWRecorded*1e-3,
			   _comb_pw/2, pulseFunc(combFunc()->to_str() ),
			  *combLevel(), *combOffRes() + _dif_freq *1000.0);
		  if(_induce_emission) {
		      makeWaveForm(PAT_QAM_PULSE_IDX_INDUCE_EMISSION/PAT_QAM_PULSE_IDX - 1, m_combPWRecorded*1e-3,
		      _induce_emission_pw/2, pulseFunc(combFunc()->to_str() ),
		      	*combLevel(), *combOffRes() + _dif_freq *1000.0, _induce_emission_phase);
		  }
    }
}

void
XPulser::makeWaveForm(unsigned int pnum_minus_1, 
		 double pw, unsigned int to_center,
	  	 tpulsefunc func, double dB, double freq, double phase)
{
	std::vector<std::complex<double> > &p = m_qamWaveForm[pnum_minus_1];
	const double dma_ao_period = resolutionQAM();
	to_center *= lrint(resolution() / dma_ao_period);
	const double delay1 = *qamDelay1() * 1e-3 / dma_ao_period;
	const double delay2 = *qamDelay2() * 1e-3 / dma_ao_period;
	double dx = dma_ao_period / pw;
	double dp = 2*PI*freq*dma_ao_period;
	double z = pow(10.0, dB/20.0);
	for(int i = 0; i < (int)to_center*2; i++) {
		double i1 = i - (int)to_center + 0.5 - delay1;
		double i2 = i - (int)to_center + 0.5 - delay2;
		double x = z * func(i1 * dx) * cos(i1 * dp + PI/4 + phase);
		double y = z * func(i2 * dx) * sin(i2 * dp + PI/4 + phase);
		p.push_back(std::complex<double>(x, y));
	}
}
