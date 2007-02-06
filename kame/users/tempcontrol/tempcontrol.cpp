#include "tempcontrol.h"
#include "forms/tempcontrolform.h"
#include "interface.h"
#include "analyzer.h"
#include "xnodeconnector.h"
#include <qstatusbar.h>
#include <klocale.h>

XTempControl::XChannel::XChannel(const char *name, bool runtime,
    const shared_ptr<XThermometerList> &list)
 : XNode(name, runtime), 
   m_thermometer(create<XItemNode<XThermometerList, XThermometer> >(
        "Thermometer", false, list)),
   m_excitation(create<XComboNode>("Excitation", false))
{
    try {
        m_thermometer->str(std::string("Raw"));
    }
    catch (XKameError &e) {
        e.print();
    }
}

XTempControl::XTempControl(const char *name, bool runtime, 
   const shared_ptr<XScalarEntryList> &scalarentries,
   const shared_ptr<XInterfaceList> &interfaces,
   const shared_ptr<XThermometerList> &thermometers,
   const shared_ptr<XDriverList> &drivers) :
    XPrimaryDriver(name, runtime, scalarentries, interfaces, thermometers, drivers),
    m_channels(create<XChannelList>("Channels", false)),
    m_currentChannel
        (create<XItemNode<XChannelList, XChannel> >("CurrentChannel", true, m_channels)),
    m_setupChannel
        (create<XItemNode<XChannelList, XChannel> >("SetupChannel", true, m_channels)),
    m_targetTemp(create<XDoubleNode>("TargetTemp", true, "%.5g")),
    m_manualPower(create<XDoubleNode>("ManualPower", true, "%.4g")),
    m_prop(create<XDoubleNode>("P", false, "%.4g")),
    m_int(create<XDoubleNode>("I", false, "%.4g")),
    m_deriv(create<XDoubleNode>("D", false, "%.4g")),
    m_heaterMode(create<XComboNode>("HeaterMode", false)),
    m_powerRange(create<XComboNode>("PowerRange", false)),
    m_heaterPower(create<XDoubleNode>("HeaterPower", false, "%.4g")),
    m_sourceTemp(create<XDoubleNode>("SourceTemp", false, "%.5g")),
    m_stabilized(create<XDoubleNode>("Stabilized", true, "%g")),
    m_form(new FrmTempControl(g_pFrmMain))
{
  std::deque<shared_ptr<XScalarEntry> > m_entry_temps;
  std::deque<shared_ptr<XScalarEntry> > m_entry_raws;
 
  m_conSetupChannel = xqcon_create<XQComboBoxConnector>(
                        m_setupChannel, m_form->m_cmbSetupChannel);
  m_conCurrentChannel = xqcon_create<XQComboBoxConnector>(
                        m_currentChannel, m_form->m_cmbSourceChannel);
  m_conPowerRange = xqcon_create<XQComboBoxConnector>(
                        m_powerRange, m_form->m_cmbPowerRange);
  m_conHeaterMode = xqcon_create<XQComboBoxConnector>(
                        m_heaterMode, m_form->m_cmbHeaterMode);
  m_conP = xqcon_create<XQLineEditConnector>(m_prop, m_form->m_edP);
  m_conI = xqcon_create<XQLineEditConnector>(m_int, m_form->m_edI);
  m_conD = xqcon_create<XQLineEditConnector>(m_deriv, m_form->m_edD);
  m_conManualPower = xqcon_create<XQLineEditConnector>(m_manualPower, m_form->m_edManHeater);
  m_conTargetTemp = xqcon_create<XQLineEditConnector>(m_targetTemp, m_form->m_edTargetTemp);
  m_conHeater = xqcon_create<XQLCDNumberConnector>(
                     m_heaterPower, m_form->m_lcdHeater);
  m_conTemp = xqcon_create<XQLCDNumberConnector>(
                     m_sourceTemp, m_form->m_lcdSourceTemp); 
 
  m_currentChannel->setUIEnabled(false);
  m_powerRange->setUIEnabled(false);
  m_heaterMode->setUIEnabled(false);
  m_prop->setUIEnabled(false);
  m_int->setUIEnabled(false);
  m_deriv->setUIEnabled(false);
  m_manualPower->setUIEnabled(false);
  m_targetTemp->setUIEnabled(false);

  m_lsnOnSetupChannelChanged = m_setupChannel->onValueChanged().connectWeak(
                  false, shared_from_this(), &XTempControl::onSetupChannelChanged);

  m_form->statusBar()->hide();
  m_form->setCaption(KAME::i18n("TempControl - ") + getLabel() );
}

void
XTempControl::showForms() {
//! impliment form->show() here
    m_form->show();
    m_form->raise();
}

void
XTempControl::start()
{
  m_thread.reset(new XThread<XTempControl>(shared_from_this(), &XTempControl::execute));
  m_thread->resume();

  m_currentChannel->setUIEnabled(true);
  m_powerRange->setUIEnabled(true);
  m_heaterMode->setUIEnabled(true);
  m_prop->setUIEnabled(true);
  m_int->setUIEnabled(true);
  m_deriv->setUIEnabled(true);
  m_manualPower->setUIEnabled(true);
  m_targetTemp->setUIEnabled(true);
}
void
XTempControl::stop()
{
  m_currentChannel->setUIEnabled(false);
  m_powerRange->setUIEnabled(false);
  m_heaterMode->setUIEnabled(false);
  m_prop->setUIEnabled(false);
  m_int->setUIEnabled(false);
  m_deriv->setUIEnabled(false);
  m_manualPower->setUIEnabled(false);
  m_targetTemp->setUIEnabled(false);
  	
    if(m_thread) m_thread->terminate();
//    m_thread->waitFor();
//  thread must do interface()->close() at the end
}

void
XTempControl::analyzeRaw() throw (XRecordError&)
{
    try {
        for(;;) {
            //! Since raw buffer is Fast-in Fast-out, use the same sequence of push()es for pop()s
            unsigned short chno = pop<unsigned short>();
            pop<unsigned short>(); //reserve
            float raw = pop<float>();
            float temp = pop<float>();
            if(!m_multiread) chno = 0;
            if(chno >= m_entry_temps.size()) throw XBufferUnderflowRecordError(__FILE__, __LINE__);
            m_entry_temps[chno]->value(temp);
            m_entry_raws[chno]->value(raw);
        }
    }
    catch (XRecordError&) {
    }
}
void
XTempControl::visualize()
{
 //! impliment extra codes which do not need write-lock of record
 //! record is read-locked
}
void
XTempControl::onSetupChannelChanged(const shared_ptr<XValueNodeBase> &)
{
  m_conThermometer.reset();
  m_conExcitation.reset();
  m_lsnOnExcitationChanged.reset();
  shared_ptr<XChannel> channel = *m_setupChannel;
  if(!channel) return;
  m_conThermometer = xqcon_create<XQComboBoxConnector>(
        channel->thermometer(), m_form->m_cmbThermometer);
  m_conExcitation = xqcon_create<XQComboBoxConnector>(
	    channel->excitation(), m_form->m_cmbExcitation);
  m_lsnOnExcitationChanged = channel->excitation()->onValueChanged().connectWeak(
			    false, shared_from_this(), &XTempControl::onExcitationChanged);
}

void
XTempControl::createChannels(const shared_ptr<XScalarEntryList> &scalarentries,
    const shared_ptr<XThermometerList> &thermometers, 
    bool multiread, const char **channel_names, const char **excitations)
{
  shared_ptr<XScalarEntryList> entries(scalarentries);
  m_multiread = multiread;
  
  for(int i = 0; channel_names[i]; i++) {
      shared_ptr<XChannel> channel = 
        m_channels->create<XChannel>(channel_names[i], true, thermometers);  
      for(int j = 0; excitations[j]; j++) {
        channel->excitation()->add(excitations[j]);
      }
  }
  if(multiread)
    {
      atomic_shared_ptr<const XNode::NodeList> list(m_channels->children());
      if(list) { 
          for(XNode::NodeList::const_iterator it = list->begin(); it != list->end(); it++) {
            shared_ptr<XChannel> channel = dynamic_pointer_cast<XChannel>(*it);
            shared_ptr<XScalarEntry> entry_temp(create<XScalarEntry>(
    		  QString("Ch.%1").arg(channel->getName()).latin1()
    		  , false,
               dynamic_pointer_cast<XDriver>(shared_from_this())
              , "%.5g"));
            shared_ptr<XScalarEntry> entry_raw(create<XScalarEntry>(
              QString("Ch.%1.raw").arg(channel->getName()).latin1()
              , false,
               dynamic_pointer_cast<XDriver>(shared_from_this())
              , "%.5g"));
             m_entry_temps.push_back(entry_temp);
             m_entry_raws.push_back(entry_raw);
             entries->insert(entry_temp);
             entries->insert(entry_raw);
          }
      }
    }
  else
    {
        shared_ptr<XScalarEntry> entry_temp(create<XScalarEntry>(
          "Temp"
          , false,
           dynamic_pointer_cast<XDriver>(shared_from_this())
          , "%.5g"));
        shared_ptr<XScalarEntry> entry_raw(create<XScalarEntry>(
          "Raw"
          , false,
           dynamic_pointer_cast<XDriver>(shared_from_this())
          , "%.5g"));
         m_entry_temps.push_back(entry_temp);
         m_entry_raws.push_back(entry_raw);
         entries->insert(entry_temp);
         entries->insert(entry_raw);
    }
}

void *
XTempControl::execute(const atomic<bool> &terminated)
{
  double tempAvg = 0.0;
  double tempErrAvg = 0.0;
  XTime lasttime = XTime::now();
  
  m_lsnOnPChanged = prop()->onValueChanged().connectWeak(
                false, shared_from_this(), &XTempControl::onPChanged);
  m_lsnOnIChanged = interval()->onValueChanged().connectWeak(
                false, shared_from_this(), &XTempControl::onIChanged);
  m_lsnOnDChanged = deriv()->onValueChanged().connectWeak(
                false, shared_from_this(), &XTempControl::onDChanged);
  m_lsnOnTargetTempChanged = targetTemp()->onValueChanged().connectWeak(
                false, shared_from_this(), &XTempControl::onTargetTempChanged);
  m_lsnOnManualPowerChanged = manualPower()->onValueChanged().connectWeak(
                false, shared_from_this(), &XTempControl::onManualPowerChanged);
  m_lsnOnHeaterModeChanged = heaterMode()->onValueChanged().connectWeak(
                false, shared_from_this(), &XTempControl::onHeaterModeChanged);
  m_lsnOnPowerRangeChanged = powerRange()->onValueChanged().connectWeak(
                false, shared_from_this(), &XTempControl::onPowerRangeChanged);
  m_lsnOnCurrentChannelChanged = m_currentChannel->onValueChanged().connectWeak(
                false, shared_from_this(), &XTempControl::onCurrentChannelChanged);
    
  while(!terminated)
    {
      msecsleep(10);

      double raw, src_raw = 0, src_temp = 0, temp;
      clearRaw();
      XTime time_awared = XTime::now();
      // try/catch exception of communication errors
      try {
          shared_ptr<XChannel> src_ch = *m_currentChannel;
          if(src_ch)
            {
              shared_ptr<XThermometer> thermo = *src_ch->thermometer();
              src_raw = getRaw(src_ch);
              src_temp = (!thermo) ? src_raw : thermo->getTemp(src_raw);
              m_sourceTemp->value(src_temp);
            }
          atomic_shared_ptr<const XNode::NodeList> list(m_channels->children());
          if(list) { 
              unsigned int idx = 0;
              for(XNode::NodeList::const_iterator it = list->begin(); it != list->end(); it++) {
                shared_ptr<XChannel> ch = dynamic_pointer_cast<XChannel>(*it);
                  if(src_ch == ch)
                    {
                      temp = src_temp;
                      raw = src_raw;
                    }
                  else
                    {
                      if(!m_multiread) continue;
                      shared_ptr<XThermometer> thermo = *ch->thermometer();
                      raw = getRaw(ch);
                      temp = (!thermo) ? raw : thermo->getTemp(raw);
                    }
                  push((unsigned short)idx);
                  push((unsigned short)0); // reserve
                  push(float(raw));
                  push(float(temp));
                  idx++;
                }
          }
            
          //calicurate std. deviations in some periods
          double tau = *interval() * 4.0;
          if(tau <= 1) tau = 4.0;
          XTime newtime = XTime::now();
          double dt = newtime - lasttime;
          lasttime = newtime;
          double terr = src_temp - *targetTemp();
          terr = terr * terr;
          tempAvg = (tempAvg - temp) * exp(-dt / tau) + temp;
          tempErrAvg = (tempErrAvg - terr) * exp(-dt / tau) + terr;
          stabilized()->value(sqrt(tempErrAvg)); //stderr
          heaterPower()->value(getHeater());
      }
      catch (XKameError &e) {
          e.print(getLabel() + "; ");
          continue;
      }
      finishWritingRaw(time_awared, XTime::now());
    }

  m_setupChannel->value(shared_ptr<XThermometer>());
    
  m_lsnOnPChanged.reset();
  m_lsnOnIChanged.reset();
  m_lsnOnDChanged.reset();
  m_lsnOnTargetTempChanged.reset();
  m_lsnOnManualPowerChanged.reset();
  m_lsnOnHeaterModeChanged.reset();
  m_lsnOnPowerRangeChanged.reset();
  m_lsnOnCurrentChannelChanged.reset();

  
  afterStop(); 
  return NULL;
}

