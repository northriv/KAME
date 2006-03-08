//---------------------------------------------------------------------------
#include "forms/lockinampform.h"
#include "lockinamp.h"
#include "interface.h"
#include "analyzer.h"
#include "xnodeconnector.h"
#include <qstatusbar.h>
#include <qpushbutton.h>
#include <qcheckbox.h>
#include <klocale.h>

XLIA::XLIA(const char *name, bool runtime, 
   const shared_ptr<XScalarEntryList> &scalarentries,
   const shared_ptr<XInterfaceList> &interfaces,
   const shared_ptr<XThermometerList> &thermometers,
   const shared_ptr<XDriverList> &drivers) : 
    XPrimaryDriver(name, runtime, scalarentries, interfaces, thermometers, drivers),
    m_valueX(create<XScalarEntry>("ValueX", false, 
        dynamic_pointer_cast<XDriver>(shared_from_this()))),
    m_valueY(create<XScalarEntry>("ValueY", false, 
        dynamic_pointer_cast<XDriver>(shared_from_this()))),
    m_output(create<XDoubleNode>("Output", false)),
    m_frequency(create<XDoubleNode>("Frequency", false)),
    m_sensitivity(create<XComboNode>("Sensitivity", false)),
    m_timeConst(create<XComboNode>("TimeConst", false)),
    m_autoScaleX(create<XBoolNode>("AutoScaleX", false)),
    m_autoScaleY(create<XBoolNode>("AutoScaleY", false)),
    m_fetchFreq(create<XDoubleNode>("FetchFreq", false)),
    m_form(new FrmLIA(g_pFrmMain))
{
  fetchFreq()->value(1);
  
  scalarentries->insert(m_valueX);
  scalarentries->insert(m_valueY);

  m_form->statusBar()->hide();
  m_form->setCaption(i18n("Lock-in-Amp - ") + getName() );

  m_output->setUIEnabled(false);
  m_frequency->setUIEnabled(false);
  m_sensitivity->setUIEnabled(false);
  m_timeConst->setUIEnabled(false);
  m_autoScaleX->setUIEnabled(false);
  m_autoScaleY->setUIEnabled(false);
  m_fetchFreq->setUIEnabled(false);

  m_conSens = xqcon_create<XQComboBoxConnector>(m_sensitivity, m_form->m_cmbSens);
  m_conTimeConst = xqcon_create<XQComboBoxConnector>(m_timeConst, m_form->m_cmbTimeConst);
  m_conFreq = xqcon_create<XQLineEditConnector>(m_frequency, m_form->m_edFreq);
  m_conOutput = xqcon_create<XQLineEditConnector>(m_output, m_form->m_edOutput);
  m_conAutoScaleX = xqcon_create<XQToggleButtonConnector>(m_autoScaleX, m_form->m_ckbAutoScaleX);
  m_conAutoScaleY = xqcon_create<XQToggleButtonConnector>(m_autoScaleY, m_form->m_ckbAutoScaleY);
  m_conFetchFreq = xqcon_create<XQLineEditConnector>(m_fetchFreq, m_form->m_edFetchFreq);
}

void
XLIA::showForms() {
//! impliment form->show() here
    m_form->show();
    m_form->raise();
}

void
XLIA::start()
{
    interface()->open();
    
    m_thread.reset(new XThread<XLIA>(shared_from_this(), &XLIA::execute));
    m_thread->resume();
}
void
XLIA::stop()
{  
    if(m_thread) m_thread->terminate();
//    m_thread->waitFor();
//  thread must do interface()->close() at the end
}

void
XLIA::analyzeRaw() throw (XRecordError&)
{
    double x, y;
    x = pop<double>();
    y = pop<double>();
    m_valueX->value(x);
    m_valueY->value(y);
}
void
XLIA::visualize()
{
 //! impliment extra codes which do not need write-lock of record
 //! record is read-locked
}

void 
XLIA::onOutputChanged(const shared_ptr<XValueNodeBase> &)
{
    try {
        changeOutput(*output());
    }
    catch (XKameError& e) {
        e.print(getName() + " " + i18n("Error while changing output, "));
        return;
    }
}
void 
XLIA::onFreqChanged(const shared_ptr<XValueNodeBase> &)
{
    try {
        changeFreq(*frequency());
    }
    catch (XKameError& e) {
        e.print(getName() + " " + i18n("Error while changing frequency, "));
        return;
    }
}
void 
XLIA::onSensitivityChanged(const shared_ptr<XValueNodeBase> &)
{
    try {
        changeSensitivity(*sensitivity());
    }
    catch (XKameError& e) {
        e.print(getName() + " " + i18n("Error while changing sensitivity, "));
        return;
    }
}
void 
XLIA::onTimeConstChanged(const shared_ptr<XValueNodeBase> &)
{
    try {
        changeTimeConst(*timeConst());
    }
    catch (XKameError& e) {
        e.print(getName() + " " + i18n("Error while changing time const., "));
        return;
    }
}


void *
XLIA::execute(const atomic<bool> &terminated)
{
  try {
      afterStart();
  }
  catch (XKameError &e) {
      e.print(getName() + " " +  i18n("Error while starting, "));
      interface()->close();
      return NULL;
  }

    
  m_output->setUIEnabled(true);
  m_frequency->setUIEnabled(true);
  m_sensitivity->setUIEnabled(true);
  m_timeConst->setUIEnabled(true);
  m_autoScaleX->setUIEnabled(true);
  m_autoScaleY->setUIEnabled(true);
  m_fetchFreq->setUIEnabled(true);
        
  m_lsnOutput = output()->onValueChanged().connectWeak(
                          false, shared_from_this(), &XLIA::onOutputChanged);
  m_lsnFreq = frequency()->onValueChanged().connectWeak(
                          false, shared_from_this(), &XLIA::onFreqChanged);
  m_lsnSens = sensitivity()->onValueChanged().connectWeak(
                        false, shared_from_this(), &XLIA::onSensitivityChanged);
  m_lsnTimeConst = timeConst()->onValueChanged().connectWeak(
                         false, shared_from_this(), &XLIA::onTimeConstChanged);

  while(!terminated)
    {
      double fetch_freq = *fetchFreq();
      double wait = 0;
      if(fetch_freq > 0) {
         sscanf(timeConst()->to_str().c_str(), "%lf", &wait);
         wait *= 1000.0 / fetch_freq;
      }
      if(wait > 0) msecsleep(lrint(wait));
      
      double x, y;
      XTime time_awared = XTime::now();
      // try/catch exception of communication errors
      try {
          get(&x, &y);
      }
      catch (XKameError &e) {
          e.print(getName() + " " + i18n("Read Error, "));
          continue;
      }
      startWritingRaw();
      push(x);
      push(y);
      finishWritingRaw(time_awared, XTime::now(), true);
    }
  
  m_lsnOutput.reset();
  m_lsnFreq.reset();
  m_lsnSens.reset();
  m_lsnTimeConst.reset();
  
  m_output->setUIEnabled(false);
  m_frequency->setUIEnabled(false);
  m_sensitivity->setUIEnabled(false);
  m_timeConst->setUIEnabled(false);
  m_autoScaleX->setUIEnabled(false);
  m_autoScaleY->setUIEnabled(false);
  m_fetchFreq->setUIEnabled(false);
  
  try {
      beforeStop();
  }
  catch (XKameError &e) {
      e.print(getName() + " " + i18n("Error while closing, "));
  }
    
  interface()->close();
  return NULL;
}
