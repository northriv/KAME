//---------------------------------------------------------------------------
#include "nmrpulse.h"
#include "forms/nmrpulseform.h"

#include "graph.h"
#include "graphwidget.h"
#include "graphnurlform.h"
#include "xwavengraph.h"
#include "analyzer.h"
#include "xnodeconnector.h"

#include <knuminput.h>
#include <klocale.h>
#include <qpushbutton.h>
#include <qcheckbox.h>
#include <kapplication.h>
#include <kiconloader.h>

#define WINDOW_FUNC_RECT "Rect."
#define WINDOW_FUNC_HANNING "Hanning"
#define WINDOW_FUNC_HAMMING "Hamming"
#define WINDOW_FUNC_FLATTOP "Flat-Top"
#define WINDOW_FUNC_BLACKMAN "Blackman"
#define WINDOW_FUNC_BLACKMAN_HARRIS "Blackman-Harris"
#define WINDOW_FUNC_KAISER_1 "Kaiser a=3"
#define WINDOW_FUNC_KAISER_2 "Kaiser a=7.2"
#define WINDOW_FUNC_KAISER_3 "Kaiser a=15"


double XNMRPulseAnalyzer::windowFuncRect(double) {
//	return (fabs(x) <= 0.5) ? 1 : 0;
	return 1.0;
}
double XNMRPulseAnalyzer::windowFuncHanning(double x) {
	if(fabs(x) >= 0.5) return 0.0;
	return 0.5 + 0.5*cos(2*PI*x);
}
double XNMRPulseAnalyzer::windowFuncHamming(double x) {
	if(fabs(x) >= 0.5) return 0.0;
	return 0.54 + 0.46*cos(2*PI*x);
}
double XNMRPulseAnalyzer::windowFuncBlackman(double x) {
	if(fabs(x) >= 0.5) return 0.0;
	return 0.42323+0.49755*cos(2*PI*x)+0.07922*cos(4*PI*x);
}
double XNMRPulseAnalyzer::windowFuncBlackmanHarris(double x) {
	if(fabs(x) >= 0.5) return 0.0;
	return 0.35875+0.48829*cos(2*PI*x)+0.14128*cos(4*PI*x)+0.01168*cos(6*PI*x);
}
double XNMRPulseAnalyzer::windowFuncFlatTop(double x) {
	return windowFuncHamming(x)*((fabs(x) < 1e-4) ? 1 : sin(4*PI*x)/(4*PI*x));
}
double XNMRPulseAnalyzer::windowFuncKaiser(double x, double alpha) {
	if(fabs(x) >= 0.5) return 0.0;
	x *= 2;
	x = sqrt(std::max(1 - x*x, 0.0));
	return bessel_i0(PI*alpha*x) / bessel_i0(PI*alpha);
}
double XNMRPulseAnalyzer::windowFuncKaiser1(double x) {
	return windowFuncKaiser(x, 3.0);
}
double XNMRPulseAnalyzer::windowFuncKaiser2(double x) {
	return windowFuncKaiser(x, 7.2);
}
double XNMRPulseAnalyzer::windowFuncKaiser3(double x) {
	return windowFuncKaiser(x, 15.0);
}

//---------------------------------------------------------------------------
XNMRPulseAnalyzer::XNMRPulseAnalyzer(const char *name, bool runtime,
   const shared_ptr<XScalarEntryList> &scalarentries,
   const shared_ptr<XInterfaceList> &interfaces,
   const shared_ptr<XThermometerList> &thermometers,
   const shared_ptr<XDriverList> &drivers)
  : XSecondaryDriver(name, runtime, scalarentries, interfaces, thermometers, drivers),
  m_entryCosAv(create<XScalarEntry>("CosAv", false,
    dynamic_pointer_cast<XDriver>(shared_from_this()))),
  m_entrySinAv(create<XScalarEntry>("SinAv", false,
    dynamic_pointer_cast<XDriver>(shared_from_this()))),
  m_dso(create<XItemNode<XDriverList, XDSO> >("DSO", false, drivers)),
  m_fromTrig(create<XDoubleNode>("FromTrig", false)),
  m_width(create<XDoubleNode>("Width", false)),
  m_phaseAdv(create<XDoubleNode>("PhaseAdv", false)),
  m_useDNR(create<XBoolNode>("UseDNR", false)),
  m_bgPos(create<XDoubleNode>("BGPos", false)),
  m_bgWidth(create<XDoubleNode>("BGWidth", false)),
  m_fftPos(create<XDoubleNode>("FFTPos", false)),
  m_fftLen(create<XUIntNode>("FFTLen", false)),
  m_windowFunc(create<XComboNode>("WindowFunc", false)),
  m_difFreq(create<XDoubleNode>("DIFFreq", false)),
  m_exAvgIncr(create<XBoolNode>("ExAvgIncr", false)),
  m_extraAvg(create<XUIntNode>("ExtraAvg", false)),
  m_numEcho(create<XUIntNode>("NumEcho", false)),
  m_echoPeriod(create<XDoubleNode>("EchoPeriod", false)),
  m_fftShow(create<XNode>("FFTShow", true)),
  m_avgClear(create<XNode>("AvgClear", true)),
  m_epcEnabled(create<XBoolNode>("FoolAvgEnabled", false)),
  m_epc4x(create<XBoolNode>("FoolAvg4x", false)),
  m_pulser(create<XItemNode<XDriverList, XPulser> >("Pulser", false, drivers)),
  m_epccnt(0),
  m_form(new FrmNMRPulse(g_pFrmMain)),
  m_statusPrinter(XStatusPrinter::create(m_form.get())),
  m_fftForm(new FrmGraphNURL(g_pFrmMain)),
  m_waveGraph(create<XWaveNGraph>("Wave", true,
         m_form->m_graph, m_form->m_urlDump, m_form->m_btnDump)),
  m_ftWaveGraph(create<XWaveNGraph>("Spectrum", true, m_fftForm.get())),
  m_fftlen(-1),
  m_dnrsubfftlen(-1),
  m_dnrpulsefftlen(-1)
{
    m_form->m_btnAvgClear->setIconSet(
             KApplication::kApplication()->iconLoader()->loadIconSet("editdelete", 
            KIcon::Toolbar, KIcon::SizeSmall, true ) );      
    m_form->m_btnFFT->setIconSet(
             KApplication::kApplication()->iconLoader()->loadIconSet("graph", 
            KIcon::Toolbar, KIcon::SizeSmall, true ) );      
    
  connect(dso());
  connect(pulser());


  scalarentries->insert(entryCosAv());
  scalarentries->insert(entrySinAv());
  
  fromTrig()->value(-0.005);
  width()->value(0.02);
  bgPos()->value(0.03);
  bgWidth()->value(0.03);
  fftPos()->value(0.004);
  fftLen()->value(16384);
  numEcho()->value(1);
  windowFunc()->add(WINDOW_FUNC_RECT);
  windowFunc()->add(WINDOW_FUNC_HANNING);
  windowFunc()->add(WINDOW_FUNC_HAMMING);
  windowFunc()->add(WINDOW_FUNC_BLACKMAN);
  windowFunc()->add(WINDOW_FUNC_BLACKMAN_HARRIS);
  windowFunc()->add(WINDOW_FUNC_FLATTOP);
  windowFunc()->add(WINDOW_FUNC_KAISER_1);
  windowFunc()->add(WINDOW_FUNC_KAISER_2);
  windowFunc()->add(WINDOW_FUNC_KAISER_3);
  windowFunc()->value(WINDOW_FUNC_RECT);

  m_form->setCaption(KAME::i18n("NMR Pulse - ") + getLabel() );

  m_fftForm->setCaption(KAME::i18n("NMR-FFT - ") + getLabel() );

  m_conAvgClear = xqcon_create<XQButtonConnector>(m_avgClear, m_form->m_btnAvgClear);
  m_conFFTShow = xqcon_create<XQButtonConnector>(m_fftShow, m_form->m_btnFFT);
    
  m_conFromTrig = xqcon_create<XQLineEditConnector>(fromTrig(), m_form->m_edPos);
  m_conWidth = xqcon_create<XQLineEditConnector>(width(), m_form->m_edWidth);
  m_form->m_numPhaseAdv->setRange(-180.0, 180.0, 1.0, true);
  m_conPhaseAdv = xqcon_create<XKDoubleNumInputConnector>(phaseAdv(), m_form->m_numPhaseAdv);
  m_conUseDNR = xqcon_create<XQToggleButtonConnector>(useDNR(), m_form->m_ckbDNR);
  m_conBGPos = xqcon_create<XQLineEditConnector>(bgPos(), m_form->m_edBGPos);
  m_conBGWidth = xqcon_create<XQLineEditConnector>(bgWidth(), m_form->m_edBGWidth);
  m_conFFTPos = xqcon_create<XQLineEditConnector>(fftPos(), m_form->m_edFFTPos);
  m_conFFTLen = xqcon_create<XQLineEditConnector>(fftLen(), m_form->m_edFFTLen);
  m_conExtraAv = xqcon_create<XQSpinBoxConnector>(extraAvg(), m_form->m_numExtraAvg);
  m_conExAvgIncr = xqcon_create<XQToggleButtonConnector>(exAvgIncr(), m_form->m_ckbIncrAvg);
  m_conNumEcho = xqcon_create<XQSpinBoxConnector>(numEcho(), m_form->m_numEcho);
  m_conEchoPeriod = xqcon_create<XQLineEditConnector>(echoPeriod(), m_form->m_edEchoPeriod);
  m_conWindowFunc = xqcon_create<XQComboBoxConnector>(windowFunc(), m_form->m_cmbWindowFunc);
  m_conDIFFreq = xqcon_create<XQLineEditConnector>(difFreq(), m_form->m_edDIFFreq);

  m_conEPCEnabled = xqcon_create<XQToggleButtonConnector>(m_epcEnabled, m_form->m_ckbEPCEnabled);
  m_conEPC4x = xqcon_create<XQToggleButtonConnector>(m_epc4x, m_form->m_ckbEPC4x);
  
  m_conPulser = xqcon_create<XQComboBoxConnector>(m_pulser, m_form->m_cmbPulser);
  m_conDSO = xqcon_create<XQComboBoxConnector>(dso(), m_form->m_cmbDSO);
      
  {
      const char *labels[] = {"Time [ms]", "Cos [V]", "Sin [V]"};
      waveGraph()->setColCount(3, labels);
      waveGraph()->selectAxes(0, 1, 2);
      waveGraph()->plot1()->label()->value(KAME::i18n("real part"));
      waveGraph()->plot1()->drawPoints()->value(false);
      waveGraph()->plot2()->label()->value(KAME::i18n("imag. part"));
      waveGraph()->plot2()->drawPoints()->value(false);
      waveGraph()->clear();
  }
  {
      const char *labels[] = {"Freq. [kHz]", "Re. [V]", "Im. [V]", "Abs. [V]", "Phase [deg]"};
      ftWaveGraph()->setColCount(5, labels);
      ftWaveGraph()->selectAxes(0, 3, 4);
      ftWaveGraph()->plot1()->label()->value(KAME::i18n("abs."));
      ftWaveGraph()->plot1()->drawBars()->value(true);
      ftWaveGraph()->plot1()->drawLines()->value(false);
      ftWaveGraph()->plot1()->drawPoints()->value(false);
      ftWaveGraph()->plot2()->label()->value(KAME::i18n("phase"));
      ftWaveGraph()->plot2()->drawPoints()->value(false); 
      ftWaveGraph()->clear();
  }
  
  m_lsnOnAvgClear = m_avgClear->onTouch().connectWeak(
                   false, shared_from_this(), &XNMRPulseAnalyzer::onAvgClear);
  m_lsnOnFFTShow = m_fftShow->onTouch().connectWeak(
                  true, shared_from_this(), &XNMRPulseAnalyzer::onFFTShow, true);
  
  m_lsnOnCondChanged = fromTrig()->onValueChanged().connectWeak(
                     true, shared_from_this(), &XNMRPulseAnalyzer::onCondChanged, true);
  width()->onValueChanged().connect(m_lsnOnCondChanged);
  phaseAdv()->onValueChanged().connect(m_lsnOnCondChanged);
  useDNR()->onValueChanged().connect(m_lsnOnCondChanged);
  bgPos()->onValueChanged().connect(m_lsnOnCondChanged);
  bgWidth()->onValueChanged().connect(m_lsnOnCondChanged);
  fftPos()->onValueChanged().connect(m_lsnOnCondChanged);
  fftLen()->onValueChanged().connect(m_lsnOnCondChanged);
  extraAvg()->onValueChanged().connect(m_lsnOnCondChanged);
  exAvgIncr()->onValueChanged().connect(m_lsnOnCondChanged);
  numEcho()->onValueChanged().connect(m_lsnOnCondChanged);
  echoPeriod()->onValueChanged().connect(m_lsnOnCondChanged);
  windowFunc()->onValueChanged().connect(m_lsnOnCondChanged);
  difFreq()->onValueChanged().connect(m_lsnOnCondChanged); 
}
XNMRPulseAnalyzer::~XNMRPulseAnalyzer() {
      if(m_fftlen >= 0)  fftw_destroy_plan(m_fftplan);
      if(m_dnrpulsefftlen >= 0)  fftw_destroy_plan(m_dnrpulsefftplan);
      if(m_dnrsubfftlen >= 0)  fftw_destroy_plan(m_dnrsubfftplan);
}
void
XNMRPulseAnalyzer::onFFTShow(const shared_ptr<XNode> &)
{
   m_fftForm->show();
   m_fftForm->raise();
}
void
XNMRPulseAnalyzer::showForms()
{
   m_form->show();
   m_form->raise();
}

void
XNMRPulseAnalyzer::backgroundSub(const std::deque<std::complex<double> > &wave,
     int length, int bgpos, int bglength, twindowfunc windowfunc, double phase_shift)
{
	std::complex<double> rot_cmpx(exp(std::complex<double>(0,phase_shift)));

  if(*useDNR() && (bglength > 0))
    {
      int dnrlen = bglength;
      int dnrpos = bgpos;

      //FFT plan for subtraction
      if(m_dnrsubfftlen != dnrlen)
    {
      if(m_dnrsubfftlen >= 0) fftw_destroy_plan(m_dnrsubfftplan);
      m_dnrsubfftlen = dnrlen;
      m_dnrsubfftin.resize(m_dnrsubfftlen);
      m_dnrsubfftout.resize(m_dnrsubfftlen);
      m_dnrsubfftplan = fftw_create_plan(m_dnrsubfftlen, FFTW_FORWARD, FFTW_ESTIMATE);
    }
      //IFFT plan for the subtracted
      if(m_dnrpulsefftlen != std::max(dnrlen, length))
    {
      if(m_dnrpulsefftlen >= 0)  fftw_destroy_plan(m_dnrpulsefftplan);
      m_dnrpulsefftlen = std::max(dnrlen, length);
      m_dnrpulsefftin.resize(m_dnrpulsefftlen);
      m_dnrpulsefftout.resize(m_dnrpulsefftlen);
      m_dnrpulsefftplan = fftw_create_plan(m_dnrpulsefftlen, FFTW_BACKWARD, FFTW_ESTIMATE);
    }

      for(int i = 0; i < m_dnrsubfftlen; i++)
    {
      m_dnrsubfftin[i].re = std::real(wave[i + dnrpos]);
      m_dnrsubfftin[i].im = std::imag(wave[i + dnrpos]);
    }
      fftw_one(m_dnrsubfftplan, &m_dnrsubfftin[0], &m_dnrsubfftout[0]);

      //calcurate noise level
      double nlevel = 0.0;
      int nlevel_cnt = 0;
#define NOISE_BW 20
      for(int i = 0; i < m_dnrsubfftlen; i++)
    {
      int k = (i < m_dnrsubfftlen/2) ? i : (i - m_dnrsubfftlen);
          if((abs(k) < NOISE_BW) && (k != 0))
        { 
          nlevel += sqrt(m_dnrsubfftout[i].re * m_dnrsubfftout[i].re
                 + m_dnrsubfftout[i].im * m_dnrsubfftout[i].im);
          nlevel_cnt++;
        }
    }
      nlevel /= nlevel_cnt;
      nlevel *= 2.0; //Threshold value

      for(int i = 0; i < m_dnrpulsefftlen; i++)
    {
      m_dnrpulsefftin[i].re = 0.0;
      m_dnrpulsefftin[i].im = 0.0;
    }        

    m_noisePower = 0.0;
      for(int i = 0; i < m_dnrsubfftlen; i++)
    {
      int k = (i < m_dnrsubfftlen/2) ? i : (i - m_dnrsubfftlen);
          if((k != 0)
         && (sqrt(m_dnrsubfftout[i].re * m_dnrsubfftout[i].re + m_dnrsubfftout[i].im * m_dnrsubfftout[i].im) < nlevel))
        {
              m_noisePower += m_dnrsubfftout[i].re * m_dnrsubfftout[i].re 
                    + m_dnrsubfftout[i].im * m_dnrsubfftout[i].im;
          continue;            
        }
          double dphase = - 2 * PI * dnrpos * i / m_dnrsubfftlen;

          double cx = cos(dphase) / m_dnrsubfftlen;
          double sx = sin(dphase) / m_dnrsubfftlen;
      int j = i * (m_dnrpulsefftlen - 1) / (m_dnrsubfftlen - 1);
      ASSERT((j >= 0) || (j < m_dnrpulsefftlen));
          m_dnrpulsefftin[j].re += cx * m_dnrsubfftout[i].re - sx * m_dnrsubfftout[i].im;
          m_dnrpulsefftin[j].im += cx * m_dnrsubfftout[i].im + sx * m_dnrsubfftout[i].re;
    }
        m_noisePower /= bglength*bglength;   

      fftw_one(m_dnrpulsefftplan, &m_dnrpulsefftin[0], &m_dnrpulsefftout[0]);
      for(int i = 0; i < length; i++)
    {
      int j = i % m_dnrpulsefftlen;
      std::complex<double> c(m_dnrpulsefftout[j].re, m_dnrpulsefftout[j].im);
      m_wave[i] += (wave[i] - c) * rot_cmpx;
//    WaveRe[i] = dnrpulsefftout[j].re;
//    WaveIm[i] = dnrpulsefftout[j].im;
    }
  }
  else
  {
      std::complex<double> bg = 0;
      m_noisePower = 1.0;
      if(bglength)
    {
      double normalize = 0.0;
          for(int i = 0; i < bglength; i++)
        {
        double z = windowfunc( (double)i / bglength - 0.5);
          bg += z * wave[i + bgpos];
          normalize += z;
        }
          bg /= normalize;

          m_noisePower = 0.0;
          for(int i = bgpos; i < bgpos + bglength; i++)
        {
              m_noisePower += std::norm(wave[i] - bg);
        }
          m_noisePower /= bglength;
    }

      for(int i = 0; i < length; i++)
    {
      m_wave[i] += (wave[i] - bg) * rot_cmpx;
    }
  }            
}
void
XNMRPulseAnalyzer::rotNFFT(int ftpos, 
    double ph, std::deque<std::complex<double> > &wave, 
    std::deque<std::complex<double> > &ftwave, twindowfunc windowfunc, int diffreq)
{
int length = wave.size();
  //phase advance
  std::complex<double> cph(cos(ph), sin(ph));
  for(int i = 0; i < length; i++)
    {
      wave[i] *= cph;
    }

  //fft
  for(int i = 0; i < m_fftlen; i++)
    {
      int j = (ftpos + i >= m_fftlen) ? (ftpos + i - m_fftlen) : (ftpos + i);
      double z = windowfunc((j - ftpos) / (double)length);
      if((j >= length) || (j < 0)) {
            m_fftin[i].re = 0;
            m_fftin[i].im = 0;
      }
      else {
         wave[j] *= z;
         m_fftin[i].re = std::real(wave[j]);
         m_fftin[i].im = std::imag(wave[j]);
      }
    }
    
  fftw_one(m_fftplan, &m_fftin[0], &m_fftout[0]);

  for(int i = 0; i < m_fftlen; i++)
    {
      int k = i + diffreq;
      if((k >= 0) && (k < m_fftlen)) {
          int j = (k < m_fftlen / 2) ? (m_fftlen / 2 + k) : (k - m_fftlen / 2);
          double normalize = 1.0 / length;
          ftwave[i] = std::complex<double>(m_fftout[j].re * normalize, m_fftout[j].im * normalize);
      }
      else {
         ftwave[i] = 0;
      }
    }
}
void
XNMRPulseAnalyzer::onAvgClear(const shared_ptr<XNode> &)
{
    m_timeClearRequested = XTime::now();
    requestAnalysis();
}
void
XNMRPulseAnalyzer::onCondChanged(const shared_ptr<XValueNodeBase> &node)
{
    if((node == fromTrig()) || (node == numEcho()) || (node == difFreq()))
        m_timeClearRequested = XTime::now();
    requestAnalysis();
}
bool
XNMRPulseAnalyzer::checkDependency(const shared_ptr<XDriver> &emitter) const {
	shared_ptr<XPulser> _pulser = *pulser();
    if(emitter == _pulser) return false;
    shared_ptr<XDSO> _dso = *dso();
    return _dso;
}
void
XNMRPulseAnalyzer::analyze(const shared_ptr<XDriver> &) throw (XRecordError&)
{
  shared_ptr<XDSO> _dso = *dso();
  ASSERT( _dso );
  ASSERT( _dso->time() );
  
  if(_dso->numChannelsRecorded() < 1) {
    throw XRecordError(KAME::i18n("No record in DSO"), __FILE__, __LINE__);
  }
  if(_dso->numChannelsRecorded() < 2) {
    throw XRecordError(KAME::i18n("Two channels needed in DSO"), __FILE__, __LINE__);
  }

  double interval = _dso->timeIntervalRecorded();
  
  if(interval <= 0) {
    throw XRecordError(KAME::i18n("Invalid time interval in waveforms."), __FILE__, __LINE__);
  }
  //[sec]
  m_interval = interval;
  //[Hz]
  m_dFreq = 1.0 / m_fftlen / interval;
  int pos = lrint(*fromTrig() *1e-3 / interval + _dso->trigPosRecorded());
  //[sec]
  m_startTime = (pos - _dso->trigPosRecorded()) * interval;
  
  if(pos >= (int)_dso->lengthRecorded()) {
    throw XRecordError(KAME::i18n("Position beyond waveforms."), __FILE__, __LINE__);
  }
  if(pos < 0) {
    throw XRecordError(KAME::i18n("Position beyond waveforms."), __FILE__, __LINE__);
  }
  int length = lrint(*width() / 1000 /  interval);
  if(pos + length >= (int)_dso->lengthRecorded()) {
    throw XRecordError(KAME::i18n("Invalid length."), __FILE__, __LINE__);
  }  
  if(length <= 0) {
    throw XRecordError(KAME::i18n("Invalid length."), __FILE__, __LINE__);
  }

  bool skip = (m_timeClearRequested > _dso->timeAwared());
  bool avgclear = skip;

  if((int)*fftLen() != m_fftlen)
    {
      if(m_fftlen >= 0)  fftw_destroy_plan(m_fftplan);
      m_fftlen = *fftLen();
      if(lrint(pow(2.0f, rintf(logf((float)m_fftlen) / logf(2.0f)))) != m_fftlen )
            m_statusPrinter->printWarning(KAME::i18n("FFT length is not a power of 2."), true);
      m_fftin.resize(m_fftlen);
      m_fftout.resize(m_fftlen);
      m_fftplan = fftw_create_plan(m_fftlen, FFTW_FORWARD, FFTW_ESTIMATE);
      m_ftWave.resize(m_fftlen);
      m_ftWaveSum.resize(m_fftlen);
    }
  if(length > (int)m_wave.size()) {
      avgclear = true;
    }
  m_wave.resize(length);
  m_waveSum.resize(length);
  m_rawWaveSum.resize(length);
  if(!*exAvgIncr() && (*extraAvg() < m_waveAv.size())) {
      avgclear = true;
    }
  if(avgclear || *exAvgIncr()) {
      m_waveAv.clear();
    }
  if(avgclear) {
      std::fill(m_rawWaveSum.begin(), m_rawWaveSum.end(), 0.0 );
      m_avcount = 0;
    }

  int bgpos = lrint((*bgPos() - *fromTrig()) / 1000 / interval);
  if(pos + bgpos >= (int)_dso->lengthRecorded()) {
    throw XRecordError(KAME::i18n("Position for BG. sub. beyond waveforms."), __FILE__, __LINE__);
  }  
  if(bgpos < 0) {
    throw XRecordError(KAME::i18n("Position for BG. sub. beyond waveforms."), __FILE__, __LINE__);
  }  
  int bglength = lrint(*bgWidth() / 1000 /  interval);
  if(pos + bgpos + bglength >= (int)_dso->lengthRecorded()) {
    throw XRecordError(KAME::i18n("Invalid Length for BG. sub."), __FILE__, __LINE__);
  }  
  if(bglength < 0) {
    throw XRecordError(KAME::i18n("Invalid Length for BG. sub."), __FILE__, __LINE__);
  }  
  
  if(bglength < length*1.5)
     m_statusPrinter->printWarning(KAME::i18n("Maybe, length for BG. sub. is too short."));
  if((bgpos < length) && (bgpos > 0))
     m_statusPrinter->printWarning(KAME::i18n("Maybe, position for BG. sub. is overrapped against echoes"), true);
  
  int echoperiod = lrint(*echoPeriod() / 1000 /interval);
  int numechoes = *numEcho();
  bool bg_after_last_echo = (echoperiod < bgpos + bglength);
  if(numechoes > 1)
  {
    if(pos + echoperiod * (numechoes - 1) + length >= (int)_dso->lengthRecorded()) {
        throw XRecordError(KAME::i18n("Invalid Multiecho settings."), __FILE__, __LINE__);
    }
    if(echoperiod < length) {
        throw XRecordError(KAME::i18n("Invalid Multiecho settings."), __FILE__, __LINE__);
    }
    if(!bg_after_last_echo)
    {
        if(bgpos + bglength > echoperiod) {
            throw XRecordError(KAME::i18n("Invalid Multiecho settings."), __FILE__, __LINE__);
        }
        if(pos + echoperiod * (numechoes - 1) + bgpos + bglength >= (int)_dso->lengthRecorded()) {
            throw XRecordError(KAME::i18n("Invalid Multiecho settings."), __FILE__, __LINE__);
        }       
    }
  }
  
//  if(skip && (*exAvgIncr() || (*extraAvg > 1))) {
  if(skip) {
    throw XSkippedRecordError(__FILE__, __LINE__);
  }
  
  std::deque<std::complex<double> > wave(_dso->lengthRecorded(), 0.0);
  
  double *rawwavecos, *rawwavesin = NULL;
  ASSERT( _dso->numChannelsRecorded() );
  rawwavecos = _dso->waveRecorded(0);
  if(_dso->numChannelsRecorded() > 1)
      rawwavesin = _dso->waveRecorded(1);

  for(int i = 0; i < numechoes; i++)
  {
      int rpos = pos + i * echoperiod;
      for(int j = 0;
         j < (!bg_after_last_echo ? std::max(bgpos + bglength, length) : length) ; j++)
      {
          int k = rpos + j;
          ASSERT(k < (int)_dso->lengthRecorded());
          wave[j] += std::complex<double>(rawwavecos[k] / numechoes,
            rawwavesin ? (rawwavesin[k] / numechoes) : 0.0);
      }
  }
  if(bg_after_last_echo)
  {
     for(int j = bgpos; j < bgpos + bglength ; j++)
      {
         int k = pos + j;
         ASSERT(k < (int)_dso->lengthRecorded());
         wave[j] += std::complex<double>(rawwavecos[k],
            rawwavesin ? rawwavesin[k] : 0.0);
      }
  }

  //Windowing
  twindowfunc windowfunc = &windowFuncRect;
  if(windowFunc()->to_str() == WINDOW_FUNC_HANNING) windowfunc = &windowFuncHanning;
  if(windowFunc()->to_str() == WINDOW_FUNC_HAMMING) windowfunc = &windowFuncHamming;
  if(windowFunc()->to_str() == WINDOW_FUNC_FLATTOP) windowfunc = &windowFuncFlatTop;
  if(windowFunc()->to_str() == WINDOW_FUNC_BLACKMAN) windowfunc = &windowFuncBlackman;
  if(windowFunc()->to_str() == WINDOW_FUNC_BLACKMAN_HARRIS) windowfunc = &windowFuncBlackmanHarris;
  if(windowFunc()->to_str() == WINDOW_FUNC_KAISER_1) windowfunc = &windowFuncKaiser1;
  if(windowFunc()->to_str() == WINDOW_FUNC_KAISER_2) windowfunc = &windowFuncKaiser2;
  if(windowFunc()->to_str() == WINDOW_FUNC_KAISER_3) windowfunc = &windowFuncKaiser3;
  

	// Echo Phase Cycling
  shared_ptr<XPulser> _pulser(*pulser());
  bool epcenabled = *m_epcEnabled;
  double phase_origin = 0.0;
  if(epcenabled && !_pulser) {
    epcenabled = false;
    gErrPrint(getLabel() + ": " + KAME::i18n("No Pulser!"));
  }
  unsigned int epcnum = (epcenabled) ? ((*m_epc4x) ? 4 : 2) : 1;

  if(epcenabled) {
  		ASSERT( _pulser->time() );
        phase_origin = _pulser->phaseOriginRecorded();
        double new_ph = phase_origin + 360.0 / epcnum;
        new_ph -= floor(new_ph / 360.0) * 360.0;
        _pulser->phaseOrigin()->value(new_ph);
  }

  if(!epcenabled || (m_epccnt == 0)) {
	  std::fill(m_wave.begin(), m_wave.end(), 0.0);
  }
  //background subtraction or dynamic noise reduction
  //accumlate echo into m_wave
  backgroundSub(wave, length, bgpos, bglength, windowfunc, -phase_origin / 180.0 * PI);
  
  if(epcenabled) {
 	  m_epccnt++;
	  if(m_epccnt < epcnum)
	    throw XSkippedRecordError(__FILE__, __LINE__);
	  for(int i = 0; i < length; i++) {
	  	m_wave[i] /= epcnum;
	  }
  }
  m_epccnt = 0;
  
    if(!*exAvgIncr() && (*extraAvg() == m_waveAv.size()) && !m_waveAv.empty())  {
      for(int i = 0; i < length; i++) {
        m_rawWaveSum[i] -= m_waveAv.front()[i];
        }
      m_waveAv.pop_front();
      m_avcount--;
    }
    
    for(int i = 0; i < length; i++) {
      m_rawWaveSum[i] += m_wave[i];
    }
    m_avcount++;
    if(!*exAvgIncr()) {
      m_waveAv.push_back(m_wave);
    }
   
  int ftpos = lrint(*fftPos() * 1e-3 / interval + _dso->trigPosRecorded() - pos);  
  if((windowfunc != &windowFuncRect) && (abs(ftpos - length/2) > length*0.1))
     m_statusPrinter->printWarning(KAME::i18n("FFTPos is off-centered for window func."));  
  double ph = *phaseAdv() * PI / 180;

  rotNFFT(ftpos, ph, m_wave, m_ftWave, windowfunc, lrint(*difFreq() * 1000.0 / dFreq()) ); 
  
    entryCosAv()->value(std::real(m_ftWave[m_fftlen / 2]));
    entrySinAv()->value(std::imag(m_ftWave[m_fftlen / 2])); 
  
  std::copy(m_rawWaveSum.begin(), m_rawWaveSum.end(), m_waveSum.begin() );
  rotNFFT(ftpos, ph, m_waveSum, m_ftWaveSum, windowfunc, lrint(*difFreq() * 1000.0 / dFreq())); 
}
void
XNMRPulseAnalyzer::visualize()
{
  if(!time()) {
      ftWaveGraph()->clear();
      waveGraph()->clear();
      return;
  }

  unsigned int length = m_wave.size();
  {   XScopedWriteLock<XWaveNGraph> lock(*ftWaveGraph());
      ftWaveGraph()->setRowCount(m_fftlen);
      for(int i = 0; i < m_fftlen; i++)
        {
          double normalize = 1.0 / m_avcount;
          ftWaveGraph()->cols(0)[i]
                = 0.001 * (double)(i - m_fftlen / 2) / m_fftlen / m_interval;
          ftWaveGraph()->cols(1)[i] = std::real(m_ftWaveSum[i]) * normalize;
          ftWaveGraph()->cols(2)[i] = std::imag(m_ftWaveSum[i]) * normalize;
          ftWaveGraph()->cols(3)[i] = std::abs(m_ftWaveSum[i]) * normalize;
          ftWaveGraph()->cols(4)[i] = std::arg(m_ftWaveSum[i]) / PI * 180;
        }
  }

  {   XScopedWriteLock<XWaveNGraph> lock(*waveGraph());
      waveGraph()->setRowCount(length);
      for(unsigned int i = 0; i < length; i++)
        {
          waveGraph()->cols(0)[i] = (startTime() + i * m_interval) * 1e3;
          waveGraph()->cols(1)[i] = (std::real(m_waveSum[i]) / m_avcount);
          waveGraph()->cols(2)[i] = (std::imag(m_waveSum[i]) / m_avcount);
        }
  }
}

