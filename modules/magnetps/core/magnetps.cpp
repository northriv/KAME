/***************************************************************************
		Copyright (C) 2002-2012 Kentaro Kitagawa
		                   kitag@kochi-u.ac.jp
		
		This program is free software; you can redistribute it and/or
		modify it under the terms of the GNU Library General Public
		License as published by the Free Software Foundation; either
		version 2 of the License, or (at your option) any later version.
		
		You should have received a copy of the GNU Library General 
		Public License and a list of authors along with this program; 
		see the files COPYING and AUTHORS.
***************************************************************************/
#include "ui_magnetpsform.h"
#include "magnetps.h"
#include "interface.h"
#include "analyzer.h"
#include "xnodeconnector.h"
#include <QStatusBar>

XMagnetPS::XMagnetPS(const char *name, bool runtime, 
	Transaction &tr_meas, const shared_ptr<XMeasure> &meas) :
    XPrimaryDriver(name, runtime, ref(tr_meas), meas),
    m_field(create<XScalarEntry>("Field", false,
								 dynamic_pointer_cast<XDriver>(shared_from_this()), "%.8g")),
    m_current(create<XScalarEntry>("Current", false, 
								   dynamic_pointer_cast<XDriver>(shared_from_this()), "%.8g")),
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
    m_statusPrinter(XStatusPrinter::create(m_form.get())) {
	meas->scalarEntries()->insert(tr_meas, m_field);
	meas->scalarEntries()->insert(tr_meas, m_current);
	m_form->statusBar()->hide();
	m_form->setWindowTitle(XString("Magnet Power Supply - " + getLabel() ));
  
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
	for(Transaction tr( *this);; ++tr) {
		tr[ *allowPersistent()] = false;
		tr[ *targetField()].setUIEnabled(false);
		tr[ *sweepRate()].setUIEnabled(false);
		tr[ *targetField()].setUIEnabled(false);
		tr[ *sweepRate()].setUIEnabled(false);
		tr[ *allowPersistent()].setUIEnabled(false);
		if(tr.commit())
			break;
	}
}
void
XMagnetPS::showForms() {
    m_form->show();
    m_form->raise();
}

void
XMagnetPS::start() {
	m_thread.reset(new XThread<XMagnetPS>(shared_from_this(), &XMagnetPS::execute));
	m_thread->resume();
  
	targetField()->setUIEnabled(true);
	sweepRate()->setUIEnabled(true);
}
void
XMagnetPS::stop() {
	targetField()->setUIEnabled(false);
	sweepRate()->setUIEnabled(false);
	allowPersistent()->setUIEnabled(false);
	
    if(m_thread) m_thread->terminate();
}

void
XMagnetPS::analyzeRaw(RawDataReader &reader, Transaction &tr) throw (XRecordError&) {
	tr[ *this].m_magnetField = reader.pop<float>();
	tr[ *this].m_outputCurrent = reader.pop<float>();
	m_field->value(tr, tr[ *this].m_magnetField);
	m_current->value(tr, tr[*this].m_outputCurrent);
}
void
XMagnetPS::visualize(const Snapshot &shot) {
}

void
XMagnetPS::onRateChanged(const Snapshot &shot, XValueNodeBase *) {
    try {
        setRate(shot[ *sweepRate()]);
    }
    catch (XKameError &e) {
		e.print(getLabel() + "; ");
    }
}

void *
XMagnetPS::execute(const atomic<bool> &terminated) {
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
		trans( *sweepRate()) = getSweepRate();
		trans( *targetField()) = getTargetField();
		last_pcsh = isPCSHeaterOn();
	}
	catch (XKameError&e) {
		e.print(getLabel());
		afterStop();
		return NULL;
	}

	if(is_pcs_fitted) allowPersistent()->setUIEnabled(true);
	for(Transaction tr( *this);; ++tr) {
		m_lsnRate = tr[ *sweepRate()].onValueChanged().connectWeakly(
			shared_from_this(), &XMagnetPS::onRateChanged);
		if(tr.commit())
			break;
	}

  
    while( !terminated) {
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
		shared_ptr<RawData> writer(new RawData);
		writer->push((float)magnet_field);
		writer->push((float)output_current);
 
		finishWritingRaw(writer, XTime::now(), XTime::now());
      
		for(Transaction tr( *this);; ++tr) {
			tr[ *magnetField()] = magnet_field;
			tr[ *outputField()] = output_field;
			tr[ *outputCurrent()] = output_current;
			tr[ *outputVolt()] = output_volt;
			tr[ *pcsHeater()] = pcs_heater && is_pcs_fitted;

			tr[ *persistent()] =  !pcs_heater && pcsh_stable && is_pcs_fitted;
			if(tr.commit())
				break;
		}

		//calicurate std. deviations in some periods
		XTime newtime = XTime::now();
		double dt = fabs(newtime - lasttime);
		lasttime = newtime;
		havg = (havg - magnet_field) * exp(-dt / 3.0) + magnet_field;
		trans( *stabilized()) = fabs(havg - (double)***targetField()); //stderr
      
		double dhdt = (magnet_field - lasth) / dt;
		lasth = magnet_field;
		dhavg = (dhavg - dhdt) * exp(-dt / 3.0) + dhdt;
      
		try {
			Snapshot shot( *this);
			if(is_pcs_fitted) {
				if(pcs_heater) {
					//pcs heater is on
					if(fabs(target_field - shot[ *targetField()]) >= field_resolution) {
						if(pcsh_stable) {
							setPoint(shot[ *targetField()]);
							toSetPoint();    
						}
					}
					else {
						if((fabs(dhavg) < field_resolution / 10) && 
						   (fabs(magnet_field - target_field) < field_resolution) &&
							   shot[ *allowPersistent()]) {
							//field is not sweeping, and persistent is allowed
							m_statusPrinter->printMessage(getLabel() + " " + 
														  i18n("Turning on Perisistent mode."));
							pcsh_time = XTime::now();
							toPersistent();
						}
					}
				}
				else {
					//pcs heater if off
					if(fabs(magnet_field - shot[ *targetField()]) >= field_resolution) {
						//start sweeping.
						if(fabs(magnet_field - output_field) < field_resolution) {
							if(fabs(target_field  - magnet_field) < field_resolution) {
								//ready to go non-persistent.
								m_statusPrinter->printMessage(getLabel() + " " + 
															  i18n("Non-Perisistent mode."));
								double h = getPersistentField();
								if(fabs(h - output_field) > field_resolution)
									throw XInterface::XInterfaceError(getLabel() + 
																	  i18n("Huh? Magnet field confusing."), __FILE__, __LINE__);
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
				if(fabs(target_field - shot[ *targetField()]) >= field_resolution) {
					setPoint(shot[ *targetField()]);
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
