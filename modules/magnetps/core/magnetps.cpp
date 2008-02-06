/***************************************************************************
		Copyright (C) 2002-2007 Kentaro Kitagawa
		                   kitag@issp.u-tokyo.ac.jp
		
		This program is free software; you can redistribute it and/or
		modify it under the terms of the GNU Library General Public
		License as published by the Free Software Foundation; either
		version 2 of the License, or (at your option) any later version.
		
		You should have received a copy of the GNU Library General 
		Public License and a list of authors along with this program; 
		see the files COPYING and AUTHORS.
***************************************************************************/
#include "forms/magnetpsform.h"
#include "magnetps.h"
#include "interface.h"
#include "analyzer.h"
#include "xnodeconnector.h"
#include <qstatusbar.h>
#include <qcheckbox.h>
#include <qpushbutton.h>

XMagnetPS::XMagnetPS(const char *name, bool runtime, 
					 const shared_ptr<XScalarEntryList> &scalarentries,
					 const shared_ptr<XInterfaceList> &interfaces,
					 const shared_ptr<XThermometerList> &thermometers,
					 const shared_ptr<XDriverList> &drivers) :
    XPrimaryDriver(name, runtime, scalarentries, interfaces, thermometers, drivers),
    m_field(create<XScalarEntry>("Field", false,
								 dynamic_pointer_cast<XDriver>(shared_from_this()))),
    m_current(create<XScalarEntry>("Current", false, 
								   dynamic_pointer_cast<XDriver>(shared_from_this()))),
    m_targetField(create<XDoubleNode>("TargetField", true)),
    m_sweepRate(create<XDoubleNode>("SweepRate", true)),
    m_allowPersistent(create<XBoolNode>("AllowPersistent", true)),
    m_stabilized(create<XDoubleNode>("Stabilized", true)),
    m_magnetField(create<XDoubleNode>("MagnetField", true)),
    m_outputField(create<XDoubleNode>("OutpuField", true)),
    m_outputCurrent(create<XDoubleNode>("OutputCurrent", true)),
    m_outputVolt(create<XDoubleNode>("OutputVolt", true)),
    m_pcsHeater(create<XBoolNode>("PCSHeater", true)),
    m_persistent(create<XBoolNode>("Persistent", true)),
    m_form(new FrmMagnetPS(g_pFrmMain)),
    m_statusPrinter(XStatusPrinter::create(m_form.get()))
{
	scalarentries->insert(m_field);
	scalarentries->insert(m_current);
	m_form->statusBar()->hide();
	m_form->setCaption("Magnet Power Supply - " + getLabel() );
  
	m_conAllowPersistent = xqcon_create<XQToggleButtonConnector>(
		allowPersistent(), m_form->m_ckbAllowPersistent);
	m_conTargetField = xqcon_create<XQLineEditConnector>(
		targetField(), m_form->m_edTargetField);
	m_conSweepRate = xqcon_create<XQLineEditConnector>(
		sweepRate(), m_form->m_edSweepRate);
	m_conMagnetField = xqcon_create<XQLCDNumberConnector>(
		magnetField(), m_form->m_lcdMagnetField);
	m_conOutputField = xqcon_create<XQLCDNumberConnector>(
		outputField(), m_form->m_lcdOutputField);
	m_conOutputCurrent= xqcon_create<XQLCDNumberConnector>(
		outputCurrent(), m_form->m_lcdCurrent);
	m_conOutputVolt = xqcon_create<XQLCDNumberConnector>(
		outputVolt(), m_form->m_lcdVoltage);
	m_conPCSH = xqcon_create<XKLedConnector>(
		pcsHeater(), m_form->m_ledSwitchHeater);
	m_conPersist = xqcon_create<XKLedConnector>(
		persistent(), m_form->m_ledPersistent);
	allowPersistent()->value(false);
	targetField()->setUIEnabled(false);
	sweepRate()->setUIEnabled(false);
	targetField()->setUIEnabled(false);
	sweepRate()->setUIEnabled(false);
	allowPersistent()->setUIEnabled(false);
	//    Field->Value.Precision = 0.001;
	//    Current->Value.Precision = 0.0001;
}
void
XMagnetPS::showForms() {
//! impliment form->show() here
    m_form->show();
    m_form->raise();
}

void
XMagnetPS::start()
{
	m_thread.reset(new XThread<XMagnetPS>(shared_from_this(), &XMagnetPS::execute));
	m_thread->resume();
  
	targetField()->setUIEnabled(true);
	sweepRate()->setUIEnabled(true);
}
void
XMagnetPS::stop()
{
	targetField()->setUIEnabled(false);
	sweepRate()->setUIEnabled(false);
	allowPersistent()->setUIEnabled(false);
	
    if(m_thread) m_thread->terminate();
//    m_thread->waitFor();
//  thread must do interface()->close() at the end
}

void
XMagnetPS::analyzeRaw() throw (XRecordError&)
{
    m_magnetFieldRecorded = pop<float>();
    m_outputCurrentRecorded = pop<float>();
    m_field->value(m_magnetFieldRecorded);
    m_current->value(m_outputCurrentRecorded);
}
void
XMagnetPS::visualize()
{
	//! impliment extra codes which do not need write-lock of record
	//! record is read-locked
}

void
XMagnetPS::onRateChanged(const shared_ptr<XValueNodeBase> &)
{
    try {
        setRate(*sweepRate());
    }
    catch (XKameError &e) {
		e.print(getLabel() + "; ");
    }
}

void *
XMagnetPS::execute(const atomic<bool> &terminated)
{   
	double havg = 0.0;
	double dhavg = 0.0;
	double lasth = 0.0;
	XTime lasttime = XTime::now();
	const double pcsh_stab_period = 30.0;
	XTime pcsh_time = XTime::now();
	pcsh_time -= pcsh_stab_period;
	double field_resolution;
	bool is_pcs_fitted;
	bool last_pcsh;
  
	try {
		field_resolution = fieldResolution();
		is_pcs_fitted = isPCSFitted();
		sweepRate()->value(getSweepRate());
		targetField()->value(getTargetField());
		last_pcsh = isPCSHeaterOn();
	}
	catch (XKameError&e) {
		e.print(getLabel());
		afterStop();
		return NULL;
	}

	if(is_pcs_fitted ) allowPersistent()->setUIEnabled(true);
	m_lsnRate = sweepRate()->onValueChanged().connectWeak(
		shared_from_this(), &XMagnetPS::onRateChanged);

  
    while(!terminated)
	{
		msecsleep(10);
		double magnet_field;
		double output_field;
		double output_current;
		double output_volt;
		double target_field;
		bool pcs_heater = true;
		bool pcsh_stable = false;
      
		// try/catch exception of communication errors
		try {
			magnet_field = getMagnetField();
			output_field = getOutputField();
			output_current = getOutputCurrent();
			output_volt = getOutputVolt();
			target_field = getTargetField();
			if(is_pcs_fitted) {
				pcs_heater = isPCSHeaterOn();
				if(pcs_heater != last_pcsh)
					pcsh_time = XTime::now();
				last_pcsh = pcs_heater;
				pcsh_stable = (XTime::now() - pcsh_time > pcsh_stab_period);
			}
		}
		catch (XKameError &e) {
			e.print(getLabel() + "; ");
			continue;
		}
		clearRaw();
		push((float)magnet_field);
		push((float)output_current);
 
		finishWritingRaw(XTime::now(), XTime::now());
      
		magnetField()->value(magnet_field);
		outputField()->value(output_field);
		outputCurrent()->value(output_current);
		outputVolt()->value(output_volt);
		pcsHeater()->value(pcs_heater && is_pcs_fitted);

		persistent()->value(!pcs_heater && pcsh_stable && is_pcs_fitted);

		//calicurate std. deviations in some periods
		XTime newtime = XTime::now();
		double dt = fabs(newtime - lasttime);
		lasttime = newtime;
		havg = (havg - magnet_field) * exp(-dt / 3.0) + magnet_field;
		stabilized()->value(fabs(havg - *targetField())); //stderr
      
		double dhdt = (magnet_field - lasth) / dt;
		lasth = magnet_field;
		dhavg = (dhavg - dhdt) * exp(-dt / 3.0) + dhdt;
      
		try {
			if(is_pcs_fitted)
			{
				if(pcs_heater) {
					//pcs heater is on
					if(fabs(target_field - *targetField()) >= field_resolution) {
						if(pcsh_stable) {
							setPoint(*targetField());
							toSetPoint();    
						}
					}
					else {
						if((fabs(dhavg) < field_resolution / 10) && 
						   (fabs(magnet_field - target_field) < field_resolution) &&
						   *allowPersistent() )
						{
							//field is not sweeping, and persistent is allowed
							m_statusPrinter->printMessage(getLabel() + " " + 
														  KAME::i18n("Turning on Perisistent mode."));
							pcsh_time = XTime::now();
							toPersistent();
						}
					}
				}
				else {
					//pcs heater if off
					if(fabs(magnet_field - *targetField()) >= field_resolution) {
						//start sweeping.
						if(fabs(magnet_field - output_field) < field_resolution) {
							if(fabs(target_field  - magnet_field) < field_resolution) {
								//ready to go non-persistent.
								m_statusPrinter->printMessage(getLabel() + " " + 
															  KAME::i18n("Non-Perisistent mode."));
								double h = getPersistentField();
								if(fabs(h - output_field) > field_resolution)
									throw XInterface::XInterfaceError(getLabel() + 
																	  KAME::i18n("Huh? Magnet field confusing."), __FILE__, __LINE__);
								pcsh_time = XTime::now();
								toNonPersistent();
							}
						}
						else {
							//set output to persistent field.
							if(pcsh_stable) {
								if(target_field != magnet_field)
									setPoint(magnet_field);
								toSetPoint();  
							}
						}
					}
					else {
						if(pcsh_stable) {
							toZero();
						}
					}
				}
			}
			else {
				// pcsh is not fitted.
				if(fabs(target_field - *targetField()) >= field_resolution) {                
					setPoint(*targetField());
					toSetPoint();
				}
			}
		}
		catch (XKameError &e) {
			e.print(getLabel());
		}
	}
 
	m_lsnRate.reset();
      
	afterStop();
	return NULL;
}
