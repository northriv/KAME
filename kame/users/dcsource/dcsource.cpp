#include "dcsourceform.h"
#include "dcsource.h"
#include "charinterface.h"
#include "xnodeconnector.h"
#include <qstatusbar.h>
#include <qpushbutton.h>
#include <qcheckbox.h>
#include <klocale.h>

XDCSource::XDCSource(const char *name, bool runtime, 
   const shared_ptr<XScalarEntryList> &scalarentries,
   const shared_ptr<XInterfaceList> &interfaces,
   const shared_ptr<XThermometerList> &thermometers,
   const shared_ptr<XDriverList> &drivers) : 
    XPrimaryDriver(name, runtime, scalarentries, interfaces, thermometers, drivers),
    m_function(create<XComboNode>("Function", false)),
    m_output(create<XBoolNode>("Output", true)),
    m_value(create<XDoubleNode>("Value", false)),
    m_form(new FrmDCSource(g_pFrmMain))
{
  m_form->statusBar()->hide();
  m_form->setCaption(KAME::i18n("DC Source - ") + getLabel() );

  m_output->setUIEnabled(false);
  m_function->setUIEnabled(false);
  m_value->setUIEnabled(false);

  m_conFunction = xqcon_create<XQComboBoxConnector>(m_function, m_form->m_cmbFunction);
  m_conOutput = xqcon_create<XQToggleButtonConnector>(m_output, m_form->m_ckbOutput);
  m_conValue = xqcon_create<XQLineEditConnector>(m_value, m_form->m_edValue);
}

void
XDCSource::showForms() {
//! impliment form->show() here
    m_form->show();
    m_form->raise();
}

void
XDCSource::start()
{
  m_output->setUIEnabled(true);
  m_function->setUIEnabled(true);
  m_value->setUIEnabled(true);
        
  m_lsnOutput = output()->onValueChanged().connectWeak(
                          false, shared_from_this(), &XDCSource::onOutputChanged);
  m_lsnFunction = function()->onValueChanged().connectWeak(
                          false, shared_from_this(), &XDCSource::onFunctionChanged);
  m_lsnValue = value()->onValueChanged().connectWeak(
                        false, shared_from_this(), &XDCSource::onValueChanged);
}
void
XDCSource::stop()
{
  m_lsnOutput.reset();
  m_lsnFunction.reset();
  m_lsnValue.reset();
  
  m_output->setUIEnabled(false);
  m_function->setUIEnabled(false);
  m_value->setUIEnabled(false);
  
  afterStop();
}

void
XDCSource::analyzeRaw() throw (XRecordError&)
{
}
void
XDCSource::visualize()
{
 //! impliment extra codes which do not need write-lock of record
 //! record is read-locked
}

void 
XDCSource::onOutputChanged(const shared_ptr<XValueNodeBase> &)
{
    try {
        changeOutput(*output());
    }
    catch (XKameError& e) {
        e.print(getLabel() + KAME::i18n(": Error while changing output, "));
        return;
    }
}
void 
XDCSource::onFunctionChanged(const shared_ptr<XValueNodeBase> &)
{
    try {
        changeFunction(*function());
    }
    catch (XKameError& e) {
        e.print(getLabel() + KAME::i18n(": Error while changing function, "));
        return;
    }
}
void 
XDCSource::onValueChanged(const shared_ptr<XValueNodeBase> &)
{
    try {
        changeValue(*value());
    }
    catch (XKameError& e) {
        e.print(getLabel() + KAME::i18n(": Error while changing value, "));
        return;
    }
}

XYK7651::XYK7651(const char *name, bool runtime, 
   const shared_ptr<XScalarEntryList> &scalarentries,
   const shared_ptr<XInterfaceList> &interfaces,
   const shared_ptr<XThermometerList> &thermometers,
   const shared_ptr<XDriverList> &drivers) 
   : XCharDeviceDriver<XDCSource>(name, runtime, scalarentries, interfaces, thermometers, drivers)
{
  function()->add("F1");
  function()->add("F5");
}
void
XYK7651::changeFunction(int )
{
  interface()->send(function()->to_str() + "E");
}
void
XYK7651::changeOutput(bool x)
{
  interface()->sendf("O%uE", x ? 1 : 0);
}
void
XYK7651::changeValue(double x)
{
  interface()->sendf("SA%.10fE", x);
}
/*
  if(node == &Inverse)
  {
  Lock();
  // Inverse Polarity
  Send("SG2");
  Send("E"); //Send Trigger
  Unlock();
  }
*/
