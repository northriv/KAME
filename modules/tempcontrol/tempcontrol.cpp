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
#include <tempcontrol.h>
#include <forms/tempcontrolform.h>
#include <interface.h>
#include <analyzer.h>
#include <xnodeconnector.h>
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
    m_heaterMode(create<XComboNode>("HeaterMode", false, true)),
    m_powerRange(create<XComboNode>("PowerRange", false, true)),
    m_heaterPower(create<XDoubleNode>("HeaterPower", false, "%.4g")),
    m_sourceTemp(create<XDoubleNode>("SourceTemp", false, "%.5g")),
    m_extDCSource
        (create<XItemNode<XDriverList, XDCSource> >("ExtDCSource", false, drivers)),
    m_extDCSourceChannel(create<XComboNode>("ExtDCSourceChannel", false, true)),
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
  m_conExtDCSource = xqcon_create<XQComboBoxConnector>(
                        m_extDCSource, m_form->m_cmbExtDCSrc);
  m_conExtDCSourceChannel = xqcon_create<XQComboBoxConnector>(
                        m_extDCSourceChannel, m_form->m_cmbExtDCSrcCh);
 
  m_currentChannel->setUIEnabled(false);
  m_powerRange->setUIEnabled(false);
  m_heaterMode->setUIEnabled(false);
  m_prop->setUIEnabled(false);
  m_int->setUIEnabled(false);
  m_deriv->setUIEnabled(false);
  m_manualPower->setUIEnabled(false);
  m_targetTemp->setUIEnabled(false);
  
  m_extDCSource->setUIEnabled(true);
  m_extDCSourceChannel->setUIEnabled(true);
  m_lsnOnExtDCSourceChanged = m_extDCSource->onValueChanged().connectWeak(
                  shared_from_this(), &XTempControl::onExtDCSourceChanged);

  m_lsnOnSetupChannelChanged = m_setupChannel->onValueChanged().connectWeak(
                  shared_from_this(), &XTempControl::onSetupChannelChanged);

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
  if(shared_ptr<XDCSource>(*extDCSource())) {
	heaterMode()->clear();
	heaterMode()->add("Off");
	heaterMode()->add("PID");
	heaterMode()->add("Man");
	powerRange()->clear();  
	powerRange()->add("0");
	powerRange()->add("1u");
	powerRange()->add("10u");
	powerRange()->add("100u");
	powerRange()->add("1m");
	powerRange()->add("10m");
	powerRange()->add("100m");
	powerRange()->add("1");
	powerRange()->add("10");
  }

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
  
  m_extDCSource->setUIEnabled(false);
  m_extDCSourceChannel->setUIEnabled(false);  
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
  	
  m_extDCSource->setUIEnabled(true);
  m_extDCSourceChannel->setUIEnabled(true);  

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
			    shared_from_this(), &XTempControl::onExcitationChanged);
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

double
XTempControl::pid(XTime time, double temp)
{
	double interv = *interval();
	double derivertive = *deriv();
	m_pidIntegralLastValues.push_back(std::pair<XTime, double>(time, temp));
	while(m_pidIntegralLastValues.size() && (
		time - m_pidIntegralLastValues.front().first > PID_FIN_RESPONSE * interv)) {
		m_pidIntegralLastValues.pop_front();
	}
	double target = *targetTemp();
	double acc = 0.0;
	double dxdt = 0.0;
	if((interv > 0) && (m_pidIntegralLastValues.size() >= 2)) {
		XTime lasttime = m_pidIntegralLastValues.front().first;
		for(std::deque<std::pair<XTime, double> >::iterator it = ++(m_pidIntegralLastValues.begin());
			it != m_pidIntegralLastValues.end(); it++) {
			acc += (it->second - target) * (it->first - lasttime) * exp(-(time - it->first) / interv / sqrt(PID_FIN_RESPONSE));
			if(time - it->first > derivertive / 2)
				dxdt = (temp - it->second) / (time - it->first);
			lasttime = it->first;
		}
		acc /= interv;
	}
	return -(temp - target + acc + dxdt * derivertive) * *prop();
}
void *
XTempControl::execute(const atomic<bool> &terminated)
{
  double tempAvg = 0.0;
  double tempErrAvg = 0.0;
  XTime lasttime = XTime::now();
  
  m_lsnOnPChanged = prop()->onValueChanged().connectWeak(
                shared_from_this(), &XTempControl::onPChanged);
  m_lsnOnIChanged = interval()->onValueChanged().connectWeak(
                shared_from_this(), &XTempControl::onIChanged);
  m_lsnOnDChanged = deriv()->onValueChanged().connectWeak(
                shared_from_this(), &XTempControl::onDChanged);
  m_lsnOnTargetTempChanged = targetTemp()->onValueChanged().connectWeak(
                shared_from_this(), &XTempControl::onTargetTempChanged);
  m_lsnOnManualPowerChanged = manualPower()->onValueChanged().connectWeak(
                shared_from_this(), &XTempControl::onManualPowerChanged);
  m_lsnOnHeaterModeChanged = heaterMode()->onValueChanged().connectWeak(
                shared_from_this(), &XTempControl::onHeaterModeChanged);
  m_lsnOnPowerRangeChanged = powerRange()->onValueChanged().connectWeak(
                shared_from_this(), &XTempControl::onPowerRangeChanged);
  m_lsnOnCurrentChannelChanged = m_currentChannel->onValueChanged().connectWeak(
                shared_from_this(), &XTempControl::onCurrentChannelChanged);
    
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
          tempAvg = (tempAvg - temp) * exp(-dt / tau) + temp;
          tempErrAvg = (tempErrAvg - terr * terr) * exp(-dt / tau) + terr * terr;
          stabilized()->value(sqrt(tempErrAvg)); //stderr
          
          double power = 0.0;
			if(shared_ptr<XDCSource> dcsrc = *extDCSource()) {
				int ch = *extDCSourceChannel();
				if(ch >= 0) {
					if(src_ch) {
						if(heaterMode()->to_str() == "PID") {
							power = pid(newtime, src_temp);
						}
						if(heaterMode()->to_str() == "Man") {
							power = *manualPower();
						}
					}
					power = std::max(std::min(power, 100.0), 0.0);
					double limit = 0.0;
					if(*powerRange() > 0)
						limit = 1e-6 * pow(10.0, (double)(*powerRange() - 1));
					dcsrc->changeValue(ch, limit * power / 100.0);
				}
			}
			else
				power = getHeater();
          
          heaterPower()->value(power);
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
void
XTempControl::onExtDCSourceChanged(const shared_ptr<XValueNodeBase> &)
{
	extDCSourceChannel()->clear();
	if(shared_ptr<XDCSource> dcsrc = *extDCSource()) {
		shared_ptr<const std::deque<XItemNodeBase::Item> > strings(dcsrc->channel()->itemStrings());
	    for(std::deque<XItemNodeBase::Item>::const_iterator it = strings->begin(); it != strings->end(); it++) {
			extDCSourceChannel()->add(it->label);
	    }
	}
}
void
XTempControl::onPChanged(const shared_ptr<XValueNodeBase> &)
{
	try {
		if(!shared_ptr<XDCSource>(*extDCSource()))
			onPChanged(*prop());
	}
	catch (XInterface::XInterfaceError& e) {
		e.print();
	}
}
void
XTempControl::onIChanged(const shared_ptr<XValueNodeBase> &)
{
	try {
		if(!shared_ptr<XDCSource>(*extDCSource()))
			onIChanged(*interval());
	}
	catch (XInterface::XInterfaceError& e) {
		e.print();
	}
}
void
XTempControl::onDChanged(const shared_ptr<XValueNodeBase> &)
{
	try {
		if(!shared_ptr<XDCSource>(*extDCSource()))
			onDChanged(*deriv());
	}
	catch (XInterface::XInterfaceError& e) {
		e.print();
	}
}
void
XTempControl::onTargetTempChanged(const shared_ptr<XValueNodeBase> &)
{
	try {
		if(!shared_ptr<XDCSource>(*extDCSource()))
			onTargetTempChanged(*targetTemp());
	}
	catch (XInterface::XInterfaceError& e) {
		e.print();
	}
}
void
XTempControl::onManualPowerChanged(const shared_ptr<XValueNodeBase> &)
{
	try {
		if(!shared_ptr<XDCSource>(*extDCSource()))
			onManualPowerChanged(*manualPower());
	}
	catch (XInterface::XInterfaceError& e) {
		e.print();
	}
}
void
XTempControl::onHeaterModeChanged(const shared_ptr<XValueNodeBase> &)
{
	try {
		if(!shared_ptr<XDCSource>(*extDCSource()))
			onHeaterModeChanged(*heaterMode());
	}
	catch (XInterface::XInterfaceError& e) {
		e.print();
	}
}
void
XTempControl::onPowerRangeChanged(const shared_ptr<XValueNodeBase> &)
{
	try {
		if(!shared_ptr<XDCSource>(*extDCSource()))
			onPowerRangeChanged(*powerRange());
	}
	catch (XInterface::XInterfaceError& e) {
		e.print();
	}
}
void
XTempControl::onCurrentChannelChanged(const shared_ptr<XValueNodeBase> &)
{
	try {
		shared_ptr<XChannel> ch(*currentChannel());
		if(!ch) return;
		onCurrentChannelChanged(ch);
	}
	catch (XInterface::XInterfaceError& e) {
		e.print();
	}
}
void
XTempControl::onExcitationChanged(const shared_ptr<XValueNodeBase> &node)
{
	try {
		shared_ptr<XChannel> ch;
		atomic_shared_ptr<const XNode::NodeList> list(channels()->children());
		if(list) { 
			for(XNode::NodeList::const_iterator it = list->begin(); it != list->end(); it++) {
				shared_ptr<XChannel> _ch = dynamic_pointer_cast<XChannel>(*it);
				if(_ch->excitation() == node)
	                ch = _ch;
			}
		}
		if(!ch) return;
		int exc = *ch->excitation();
		if(exc < 0) return;
		onExcitationChanged(ch, exc);
	}
	catch (XInterface::XInterfaceError& e) {
		e.print();
	}
}
