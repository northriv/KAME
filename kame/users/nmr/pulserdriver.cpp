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
XPulser::pulseFunc(const std::string &str) {
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
    m_combMode(create<XComboNode>("CombMode", false)),
    m_rtMode(create<XComboNode>("RTMode", false)),
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
    m_numPhaseCycle(create<XComboNode>("NumPhaseCycle", false)),
    m_p1Func(create<XComboNode>("P1Func", false)),
    m_p2Func(create<XComboNode>("P2Func", false)),
    m_combFunc(create<XComboNode>("CombFunc", false)),
    m_p1Level(create<XDoubleNode>("P1Level", false)),
    m_p2Level(create<XDoubleNode>("P2Level", false)),
    m_combLevel(create<XDoubleNode>("CombLevel", false)),
    m_masterLevel(create<XDoubleNode>("MasterLevel", false)),
    m_aswFilter(create<XComboNode>("ASWFilter", false)), 
    m_portLevel8(create<XDoubleNode>("PortLevel8", false)),
    m_portLevel9(create<XDoubleNode>("PortLevel9", false)),
    m_portLevel10(create<XDoubleNode>("PortLevel10", false)),
    m_portLevel11(create<XDoubleNode>("PortLevel11", false)),
    m_portLevel12(create<XDoubleNode>("PortLevel12", false)),
    m_portLevel13(create<XDoubleNode>("PortLevel13", false)),  
    m_portLevel14(create<XDoubleNode>("PortLevel14", false)),
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
  m_conASWFilter = xqcon_create<XQComboBoxConnector>(m_aswFilter, m_formMore->m_cmbASWFilter);
  m_conPortLevel8 = xqcon_create<XQLineEditConnector>(m_portLevel8, m_formMore->m_edPortLevel8);  
  m_conPortLevel9 = xqcon_create<XQLineEditConnector>(m_portLevel9, m_formMore->m_edPortLevel9);  
  m_conPortLevel10 = xqcon_create<XQLineEditConnector>(m_portLevel10, m_formMore->m_edPortLevel10);  
  m_conPortLevel11 = xqcon_create<XQLineEditConnector>(m_portLevel11, m_formMore->m_edPortLevel11);  
  m_conPortLevel12 = xqcon_create<XQLineEditConnector>(m_portLevel12, m_formMore->m_edPortLevel12);  
  m_conPortLevel13 = xqcon_create<XQLineEditConnector>(m_portLevel13, m_formMore->m_edPortLevel13);  
  m_conPortLevel14 = xqcon_create<XQLineEditConnector>(m_portLevel14, m_formMore->m_edPortLevel14);  
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
  aswFilter()->setUIEnabled(false);
  portLevel8()->setUIEnabled(false);
  portLevel9()->setUIEnabled(false);
  portLevel10()->setUIEnabled(false);
  portLevel11()->setUIEnabled(false);
  portLevel12()->setUIEnabled(false);
  portLevel13()->setUIEnabled(false);
  portLevel14()->setUIEnabled(false);
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
  openInterfaces();

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
  p1Func()->setUIEnabled(true);
  p2Func()->setUIEnabled(true);
  combFunc()->setUIEnabled(true);
  p1Level()->setUIEnabled(true);
  p2Level()->setUIEnabled(true);
  combLevel()->setUIEnabled(true);
  masterLevel()->setUIEnabled(true);
  aswFilter()->setUIEnabled(true);
  portLevel8()->setUIEnabled(true);
  portLevel9()->setUIEnabled(true);
  portLevel10()->setUIEnabled(true);
  portLevel11()->setUIEnabled(true);
  portLevel12()->setUIEnabled(true);
  portLevel13()->setUIEnabled(true);
  portLevel14()->setUIEnabled(true);
  qamOffset1()->setUIEnabled(true);
  qamOffset2()->setUIEnabled(true);
  qamLevel1()->setUIEnabled(true);
  qamLevel2()->setUIEnabled(true);
  qamDelay1()->setUIEnabled(true);
  qamDelay2()->setUIEnabled(true);
  difFreq()->setUIEnabled(true);  
  induceEmission()->setUIEnabled(true);
  induceEmissionPhase()->setUIEnabled(true);
  qswDelay()->setUIEnabled(true);
  qswWidth()->setUIEnabled(true);
  qswPiPulseOnly()->setUIEnabled(true);
  invertPhase()->setUIEnabled(true);

  afterStart();
      
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
  p1Func()->onValueChanged().connect(m_lsnOnPulseChanged);
  p2Func()->onValueChanged().connect(m_lsnOnPulseChanged);
  combFunc()->onValueChanged().connect(m_lsnOnPulseChanged);
  p1Level()->onValueChanged().connect(m_lsnOnPulseChanged);
  p2Level()->onValueChanged().connect(m_lsnOnPulseChanged);
  combLevel()->onValueChanged().connect(m_lsnOnPulseChanged);
  masterLevel()->onValueChanged().connect(m_lsnOnPulseChanged);
  aswFilter()->onValueChanged().connect(m_lsnOnPulseChanged);
  portLevel8()->onValueChanged().connect(m_lsnOnPulseChanged);
  portLevel9()->onValueChanged().connect(m_lsnOnPulseChanged);
  portLevel10()->onValueChanged().connect(m_lsnOnPulseChanged);
  portLevel11()->onValueChanged().connect(m_lsnOnPulseChanged);
  portLevel12()->onValueChanged().connect(m_lsnOnPulseChanged);
  portLevel13()->onValueChanged().connect(m_lsnOnPulseChanged);
  portLevel14()->onValueChanged().connect(m_lsnOnPulseChanged);
  qamOffset1()->onValueChanged().connect(m_lsnOnPulseChanged);
  qamOffset2()->onValueChanged().connect(m_lsnOnPulseChanged);
  qamLevel1()->onValueChanged().connect(m_lsnOnPulseChanged);
  qamLevel2()->onValueChanged().connect(m_lsnOnPulseChanged);
  qamDelay1()->onValueChanged().connect(m_lsnOnPulseChanged);
  qamDelay2()->onValueChanged().connect(m_lsnOnPulseChanged);
  difFreq()->onValueChanged().connect(m_lsnOnPulseChanged);    
  induceEmission()->onValueChanged().connect(m_lsnOnPulseChanged);    
  induceEmissionPhase()->onValueChanged().connect(m_lsnOnPulseChanged);    
  qswDelay()->onValueChanged().connect(m_lsnOnPulseChanged);
  qswWidth()->onValueChanged().connect(m_lsnOnPulseChanged);
  qswPiPulseOnly()->onValueChanged().connect(m_lsnOnPulseChanged);
  invertPhase()->onValueChanged().connect(m_lsnOnPulseChanged);
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
  aswFilter()->setUIEnabled(false);
  portLevel8()->setUIEnabled(false);
  portLevel9()->setUIEnabled(false);
  portLevel10()->setUIEnabled(false);
  portLevel11()->setUIEnabled(false);
  portLevel12()->setUIEnabled(false);
  portLevel13()->setUIEnabled(false);
  portLevel14()->setUIEnabled(false);
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
  
  closeInterfaces();
//    m_thread->waitFor();
//  thread must do interface()->close() at the end
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
  
  double _tau = *tau();
  double _asw_setup = *aswSetup();
  double _asw_hold = *aswHold();
  double _alt_sep = *altSep();
  int _echo_num = *echoNum();
  if(_asw_setup > 2.0 * _tau)
    aswSetup()->value(2.0 * _tau);
  if(node != altSep())
    if(_alt_sep != _asw_setup + _asw_hold + (_echo_num - 1) * 2 * _tau/1000)
      {
            altSep()->value(_asw_setup + _asw_hold + (_echo_num - 1) * 2 * _tau/1000);
        return;
      }

  clearRaw();

  if(!*output())
    {
      finishWritingRaw(XTime(), XTime());
      return;
    }
    
//! ver 1 records below
    push((short)*combMode());
    push((short)0); //reserve
    push((double)*rtime());
    push((double)_tau);
    push((double)*pw1());
    push((double)*pw2());
    push((double)*combP1());
    push((double)_alt_sep);
    push((double)*combP1Alt());
    push((double)_asw_setup);
    push((double)_asw_hold);
//! ver 2 records below
    push((double)*difFreq());
    push((double)*combPW());
    push((double)*combPT());
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
  
  	if(time()) {
      try {
		  createNativePatterns();
          changeOutput(true);
      }
      catch (XKameError &e) {
          e.print(getLabel() + KAME::i18n("Pulser Turn-On Failed, because"));
      } 
  	}
  	else {
	  try {
	      changeOutput(false);
	  }
	  catch (XKameError &e) {
	      e.print(getLabel() + KAME::i18n("Pulser Turn-Off Failed, because"));
	      return;
	  }
  	}
}
double
XPulser::periodicTermRecorded() const {
    ASSERT(!m_relPatList.empty());
    return m_relPatList.back().time;
}
