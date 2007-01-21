//---------------------------------------------------------------------------
#include "forms/nmrfspectrumform.h"
#include "nmrfspectrum.h"
#include "nmrpulse.h"

#include "graph.h"
#include "graphwidget.h"
#include "xwavengraph.h"
#include "signalgenerator.h"

#include <klocale.h>
#include <qpushbutton.h>
#include <qcheckbox.h>
#include <qcombobox.h>
#include <kapplication.h>
#include <kiconloader.h>

//---------------------------------------------------------------------------
XNMRFSpectrum::XNMRFSpectrum(const char *name, bool runtime,
   const shared_ptr<XScalarEntryList> &scalarentries,
   const shared_ptr<XInterfaceList> &interfaces,
   const shared_ptr<XThermometerList> &thermometers,
   const shared_ptr<XDriverList> &drivers)
  : XSecondaryDriver(name, runtime, scalarentries, interfaces, thermometers, drivers),
  m_pulse(create<XItemNode<XDriverList, XNMRPulseAnalyzer> >("PulseAnalyzer", false, drivers)),
  m_sg1(create<XItemNode<XDriverList, XSG> >("SG1", false, drivers)),
  m_sg2(create<XItemNode<XDriverList, XSG> >("SG2", false, drivers)),
  m_sg1FreqOffset(create<XDoubleNode>("SG1FreqOffset", false)),
  m_sg2FreqOffset(create<XDoubleNode>("SG2FreqOffset", false)),
  m_centerFreq(create<XDoubleNode>("CenterFreq", false)),
  m_bandWidth(create<XDoubleNode>("BandWidth", false)),
  m_freqSpan(create<XDoubleNode>("FreqSpan", false)),
  m_freqStep(create<XDoubleNode>("FreqStep", false)),
  m_active(create<XBoolNode>("Active", true)),
  m_clear(create<XNode>("Clear", true)),
  m_form(new FrmNMRFSpectrum(g_pFrmMain)),
  m_spectrum(create<XWaveNGraph>("Spectrum", true,
     m_form->m_graph, m_form->m_urlDump, m_form->m_btnDump))
{
    m_form->m_btnClear->setIconSet(
             KApplication::kApplication()->iconLoader()->loadIconSet("editdelete", 
            KIcon::Toolbar, KIcon::SizeSmall, true ) );      
    
  connect(sg1());
  connect(sg2());
  connect(pulse());

  m_form->setCaption(KAME::i18n("NMR Spectrum (Freq. Sweep) - ") + getLabel() );

  {
      const char *labels[] = {"Freq [MHz]", "Re [V]", "Im [V]", "Counts"};
      m_spectrum->setColCount(4, labels);
      m_spectrum->selectAxes(0, 1, 2, 3);
      m_spectrum->plot1()->label()->value(KAME::i18n("real part"));
      m_spectrum->plot1()->drawPoints()->value(false);
      m_spectrum->plot2()->label()->value(KAME::i18n("imag. part"));
      m_spectrum->plot2()->drawPoints()->value(false);
      m_spectrum->clear();
  }
  
  centerFreq()->value(20);
  bandWidth()->value(50);
  sg1FreqOffset()->value(700);
  sg2FreqOffset()->value(700);
  freqSpan()->value(200);
  freqStep()->value(1);

  m_lsnOnClear = m_clear->onTouch().connectWeak(
                   false, shared_from_this(), &XNMRFSpectrum::onClear);
  m_lsnOnActiveChanged = active()->onValueChanged().connectWeak(  
                            false, shared_from_this(), &XNMRFSpectrum::onActiveChanged);
  m_lsnOnCondChanged = centerFreq()->onValueChanged().connectWeak(
                     true, shared_from_this(), &XNMRFSpectrum::onCondChanged, true);
  m_bandWidth->onValueChanged().connect(m_lsnOnCondChanged);
  m_freqSpan->onValueChanged().connect(m_lsnOnCondChanged);
  m_freqStep->onValueChanged().connect(m_lsnOnCondChanged);

  m_conSG1FreqOffset = xqcon_create<XQLineEditConnector>(m_sg1FreqOffset, m_form->m_edSG1FreqOffset);
  m_conSG2FreqOffset = xqcon_create<XQLineEditConnector>(m_sg2FreqOffset, m_form->m_edSG2FreqOffset);
  m_conCenterFreq = xqcon_create<XQLineEditConnector>(m_centerFreq, m_form->m_edCenterFreq);
  m_conBandWidth = xqcon_create<XQLineEditConnector>(m_bandWidth, m_form->m_edBW);
  m_conFreqSpan = xqcon_create<XQLineEditConnector>(m_freqSpan, m_form->m_edFreqSpan);
  m_conFreqStep = xqcon_create<XQLineEditConnector>(m_freqStep, m_form->m_edFreqStep);
  m_conSG1 = xqcon_create<XQComboBoxConnector>(m_sg1, m_form->m_cmbSG1);
  m_conSG2 = xqcon_create<XQComboBoxConnector>(m_sg2, m_form->m_cmbSG2);
  m_conPulse = xqcon_create<XQComboBoxConnector>(m_pulse, m_form->m_cmbPulse);
  m_conActive = xqcon_create<XQToggleButtonConnector>(m_active, m_form->m_ckbActive);
  m_conClear = xqcon_create<XQButtonConnector>(m_clear, m_form->m_btnClear);
}
void
XNMRFSpectrum::showForms()
{
   m_form->show();
   m_form->raise();
}
void 
XNMRFSpectrum::onCondChanged(const shared_ptr<XValueNodeBase> &)
{
    requestAnalysis();
}
void
XNMRFSpectrum::onClear(const shared_ptr<XNode> &)
{
    m_timeClearRequested = XTime::now();
    requestAnalysis();
}
void
XNMRFSpectrum::onActiveChanged(const shared_ptr<XValueNodeBase> &)
{
    if(*active())    
    {
        shared_ptr<XSG> _sg1 = *sg1();
        if(_sg1) _sg1->freq()->value(*centerFreq() - *freqSpan()/2e3 + *sg1FreqOffset());
        shared_ptr<XSG> _sg2 = *sg2();
        if(_sg2) _sg2->freq()->value(*centerFreq() - *freqSpan()/2e3 + *sg2FreqOffset());
        m_timeClearRequested = XTime::now();
    }
}
bool
XNMRFSpectrum::checkDependency(const shared_ptr<XDriver> &emitter) const {
    shared_ptr<XSG> _sg1 = *sg1();
    shared_ptr<XSG> _sg2 = *sg2();
    shared_ptr<XNMRPulseAnalyzer> _pulse = *pulse();
    if(!_sg1 || !_pulse) return false;
    if(emitter == shared_from_this()) return true;
    if(emitter != _pulse) return false;
    if(_pulse->timeAwared() < _sg1->time()) return false;
    if(_sg2) {
        if(_pulse->timeAwared() < _sg2->time()) return false;
    }
    return true;
}
void
XNMRFSpectrum::analyze(const shared_ptr<XDriver> &emitter) throw (XRecordError&)
{
    shared_ptr<XSG> _sg1 = *sg1();
  ASSERT( _sg1 );
  ASSERT( _sg1->time() );
    shared_ptr<XSG> _sg2 = *sg2();
    double freq = _sg1->freqRecorded() - *sg1FreqOffset(); //MHz
    if(_sg2) {
        if(fabs(freq - (_sg2->freqRecorded() - *sg2FreqOffset())) > freq * 1e-6)
            throw XRecordError(KAME::i18n("Conflicting freqs of 2 SGs."), __FILE__, __LINE__); 
    }
    
  shared_ptr<XNMRPulseAnalyzer> _pulse = *pulse();
  ASSERT( _pulse );
  ASSERT( _pulse->time() );
  ASSERT( emitter != _sg1 );
  ASSERT( emitter != _sg2 );
   
  double cfreq = *centerFreq(); //MHz
  double freq_span = *freqSpan() * 1e-3; //MHz
  double freq_step = *freqStep() * 1e-3; //MHz
  if(cfreq <= freq_span/2) {
    throw XRecordError(KAME::i18n("Invalid center freq."), __FILE__, __LINE__);
  }
  if(freq_span <= freq_step*2) {
    throw XRecordError(KAME::i18n("Too large freq. step."), __FILE__, __LINE__);
  }

  bool clear = (m_timeClearRequested > _pulse->timeAwared());
  
  int len = _pulse->ftWave().size();
  double _df = _pulse->dFreq() * 1e-6; //MHz
  if((len == 0) || (_df == 0)) {
    throw XRecordError(KAME::i18n("Invalid waveform."), __FILE__, __LINE__);  
  }
  
  double freq_min = cfreq - freq_span/2;
  double freq_max = cfreq + freq_span/2;

  if((fabs(df() - _df) > 1e-6) || clear)
    {
      m_df = _df;
      m_wave.clear();
      m_counts.clear();
    }
  else {
    for(int i = 0; i < rint(m_fMin / df()) - rint(freq_min / df()); i++) {
         m_wave.push_front(0.0);
         m_counts.push_front(0);
    }
    for(int i = 0; i < rint(freq_min / df()) - rint(m_fMin / df()); i++) {
         if(!m_wave.empty()) {
             m_wave.pop_front();
             m_counts.pop_front();
         }
    }
  }
  m_fMin = freq_min;
  int length = lrint(freq_span / df());
  m_wave.resize(length, 0.0);
  m_counts.resize(length, 0);

  if(clear) {
    m_spectrum->clear();
	throw XSkippedRecordError(__FILE__, __LINE__);
  }
  if(emitter != _pulse) throw XSkippedRecordError(__FILE__, __LINE__);
  
  int bw = lrint(*bandWidth() * 1e-3 / df());
  for(int i = std::max(0, (len - bw) / 2); i < std::min(len, (len + bw) / 2); i++)
    {
      double f = (i - len/2) * df(); //MHz
        add(freq + f, _pulse->ftWave()[i]);
    }
    
     //set new freq
   if(*active())
   {
      double newf = freq + freq_step; //MHz
    
      if(_sg1) _sg1->freq()->value(newf + *sg1FreqOffset());
      if(_sg2) _sg2->freq()->value(newf + *sg2FreqOffset());
      if(newf >= freq_max)
            active()->value(false);
   }
}
void
XNMRFSpectrum::visualize()
{
  if(!time()) {
      m_spectrum->clear();
      return;
  }

  int length = wave().size();
  {   XScopedWriteLock<XWaveNGraph> lock(*m_spectrum);
      m_spectrum->setRowCount(length);
      for(int i = 0; i < length; i++)
        {
          m_spectrum->cols(0)[i] = fMin() + i * df();
          m_spectrum->cols(1)[i] = (counts()[i] > 0) ? std::real(wave()[i]) / counts()[i] : 0;
          m_spectrum->cols(2)[i] = (counts()[i] > 0) ? std::imag(wave()[i]) / counts()[i] : 0;
          m_spectrum->cols(3)[i] = counts()[i];
        }
  }
}

void
XNMRFSpectrum::add(double freq, std::complex<double> c)
{
  int idx = lrint((freq - fMin()) / df());
  if((idx >= (int)wave().size()) || (idx < 0)) return;
  m_wave[idx] += c;
  m_counts[idx]++;
  return;
}
