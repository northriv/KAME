/***************************************************************************
		Copyright (C) 2002-2013 Kentaro Kitagawa
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
#include "ui_magnetpsconfigform.h"
#include "magnetps.h"
#include "interface.h"
#include "analyzer.h"
#include "xnodeconnector.h"
#include <QStatusBar>

XMagnetPS::XMagnetPS(const char *name, bool runtime, 
	Transaction &tr_meas, const shared_ptr<XMeasure> &meas) :
    XPrimaryDriverWithThread(name, runtime, ref(tr_meas), meas),
    m_field(create<XScalarEntry>("Field", false,
								 dynamic_pointer_cast<XDriver>(shared_from_this()), "%.8g")),
    m_current(create<XScalarEntry>("Current", false, 
								   dynamic_pointer_cast<XDriver>(shared_from_this()), "%.8g")),
    m_entries(meas->scalarEntries()),
    m_targetField(create<XDoubleNode>("TargetField", true)),
    m_sweepRate(create<XDoubleNode>("SweepRate", true)),
    m_allowPersistent(create<XBoolNode>("AllowPersistent", true)),
    m_approach(create<XComboNode>("Approach", false, true)),
    m_stabilized(create<XDoubleNode>("Stabilized", true)),
    m_magnetField(create<XDoubleNode>("MagnetField", true)),
    m_outputField(create<XDoubleNode>("OutpuField", true)),
    m_outputCurrent(create<XDoubleNode>("OutputCurrent", true)),
    m_outputVolt(create<XDoubleNode>("OutputVolt", true)),
    m_pcsHeater(create<XBoolNode>("PCSHeater", true)),
    m_persistent(create<XBoolNode>("Persistent", true)),
    m_aborting(create<XBoolNode>("Aborting", true)),
    m_configShow(create<XTouchableNode>("ConfigShow", true)),
    m_rateLimit1(create<XDoubleNode>("RateLimit1", false)),
    m_rateLimit1UBound(create<XDoubleNode>("RateLimit1UBound", false)),
    m_rateLimit2(create<XDoubleNode>("RateLimit2", false)),
    m_rateLimit2UBound(create<XDoubleNode>("RateLimit2UBound", false)),
    m_rateLimit3(create<XDoubleNode>("RateLimit3", false)),
    m_rateLimit3UBound(create<XDoubleNode>("RateLimit3UBound", false)),
    m_rateLimit4(create<XDoubleNode>("RateLimit4", false)),
    m_rateLimit4UBound(create<XDoubleNode>("RateLimit4UBound", false)),
    m_rateLimit5(create<XDoubleNode>("RateLimit5", false)),
    m_rateLimit5UBound(create<XDoubleNode>("RateLimit5UBound", false)),
    m_secondaryPSMultiplier(create<XDoubleNode>("SecondaryPSMultiplier", false)),
    m_secondaryPS(create<XItemNode<XDriverList, XMagnetPS> >("SecondaryPS", false, ref(tr_meas), meas->drivers())),
    m_safeCond1Entry(create<XItemNode<XScalarEntryList, XScalarEntry> >("SafeCond1Entry", false, ref(tr_meas), meas->scalarEntries())),
    m_safeCond1Min(create<XDoubleNode>("SafeCond1Min", false)),
    m_safeCond1Max(create<XDoubleNode>("SafeCond1Max", false)),
    m_safeCond2Entry(create<XItemNode<XScalarEntryList, XScalarEntry> >("SafeCond2Entry", false, ref(tr_meas), meas->scalarEntries())),
    m_safeCond2Min(create<XDoubleNode>("SafeCond2Min", false)),
    m_safeCond2Max(create<XDoubleNode>("SafeCond2Max", false)),
    m_safeCond3Entry(create<XItemNode<XScalarEntryList, XScalarEntry> >("SafeCond3Entry", false, ref(tr_meas), meas->scalarEntries())),
    m_safeCond3Min(create<XDoubleNode>("SafeCond3Min", false)),
    m_safeCond3Max(create<XDoubleNode>("SafeCond3Max", false)),
    m_persistentCondEntry(create<XItemNode<XScalarEntryList, XScalarEntry> >("PersistentCondEntry", false, ref(tr_meas), meas->scalarEntries())),
    m_persistentCondMax(create<XDoubleNode>("PersistentCondMax", false)),
    m_nonPersistentCondEntry(create<XItemNode<XScalarEntryList, XScalarEntry> >("NonPersistentCondEntry", false, ref(tr_meas), meas->scalarEntries())),
    m_nonPersistentCondMin(create<XDoubleNode>("NonPersistentCondMin", false)),
    m_pcshWait(create<XDoubleNode>("PCSHWait", false)),
    m_form(new FrmMagnetPS(g_pFrmMain)),
    m_formConfig(new FrmMagnetPSConfig(g_pFrmMain)),
    m_statusPrinter(XStatusPrinter::create(m_form.get())) {
	meas->scalarEntries()->insert(tr_meas, m_field);
	meas->scalarEntries()->insert(tr_meas, m_current);
	m_form->statusBar()->hide();
	m_form->setWindowTitle(XString("Magnet Power Supply - " + getLabel() ));
	m_formConfig->statusBar()->hide();
    m_formConfig->setWindowTitle(XString("Magnet PS Detail Configuration - " + getLabel() ));

	m_conConfigShow = xqcon_create<XQButtonConnector>(
        m_configShow, m_form->m_btnConfig);

    m_form->m_btnConfig->setIcon(QApplication::style()->standardIcon(QStyle::SP_DialogOpenButton));

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
	m_conPCSH = xqcon_create<XQLedConnector>(
		pcsHeater(), m_form->m_ledSwitchHeater);
	m_conPersist = xqcon_create<XQLedConnector>(
		persistent(), m_form->m_ledPersistent);
	m_conAborting = xqcon_create<XQLedConnector>(
		m_aborting, m_form->m_ledAborting);
	m_conApproach = xqcon_create<XQComboBoxConnector>(
		m_approach, m_form->m_cmbApproach, Snapshot( *m_approach));
	m_conRateLimit1 = xqcon_create<XQLineEditConnector>(
		m_rateLimit1, m_formConfig->m_edRateLimit1);
	m_conRateLimit2 = xqcon_create<XQLineEditConnector>(
		m_rateLimit2, m_formConfig->m_edRateLimit2);
	m_conRateLimit3 = xqcon_create<XQLineEditConnector>(
		m_rateLimit3, m_formConfig->m_edRateLimit3);
	m_conRateLimit4 = xqcon_create<XQLineEditConnector>(
		m_rateLimit4, m_formConfig->m_edRateLimit4);
	m_conRateLimit5 = xqcon_create<XQLineEditConnector>(
		m_rateLimit5, m_formConfig->m_edRateLimit5);
	m_conRateLimit1UBound = xqcon_create<XQLineEditConnector>(
		m_rateLimit1UBound, m_formConfig->m_edRateLimit1Bound);
	m_conRateLimit2UBound = xqcon_create<XQLineEditConnector>(
		m_rateLimit2UBound, m_formConfig->m_edRateLimit2Bound);
	m_conRateLimit3UBound = xqcon_create<XQLineEditConnector>(
		m_rateLimit3UBound, m_formConfig->m_edRateLimit3Bound);
	m_conRateLimit4UBound = xqcon_create<XQLineEditConnector>(
		m_rateLimit4UBound, m_formConfig->m_edRateLimit4Bound);
	m_conRateLimit5UBound = xqcon_create<XQLineEditConnector>(
		m_rateLimit5UBound, m_formConfig->m_edRateLimit5Bound);
	m_conSecondaryPS = xqcon_create<XQComboBoxConnector>(
		m_secondaryPS, m_formConfig->m_cmbSecondaryPS, ref(tr_meas));
	m_conSecondaryPSMultiplier = xqcon_create<XQLineEditConnector>(
		m_secondaryPSMultiplier, m_formConfig->m_edSecondaryPSMultiplier);
	m_conSafeCond1Entry = xqcon_create<XQComboBoxConnector>(
		m_safeCond1Entry, m_formConfig->m_cmbSafeCond1Entry, ref(tr_meas));
	m_conSafeCond1Min = xqcon_create<XQLineEditConnector>(
		m_safeCond1Min, m_formConfig->m_edSafeCond1Min);
	m_conSafeCond1Max = xqcon_create<XQLineEditConnector>(
		m_safeCond1Max, m_formConfig->m_edSafeCond1Max);
	m_conSafeCond2Entry = xqcon_create<XQComboBoxConnector>(
		m_safeCond2Entry, m_formConfig->m_cmbSafeCond2Entry, ref(tr_meas));
	m_conSafeCond2Min = xqcon_create<XQLineEditConnector>(
		m_safeCond2Min, m_formConfig->m_edSafeCond2Min);
	m_conSafeCond2Max = xqcon_create<XQLineEditConnector>(
		m_safeCond2Max, m_formConfig->m_edSafeCond2Max);
	m_conSafeCond3Entry = xqcon_create<XQComboBoxConnector>(
		m_safeCond3Entry, m_formConfig->m_cmbSafeCond3Entry, ref(tr_meas));
	m_conSafeCond3Min = xqcon_create<XQLineEditConnector>(
		m_safeCond3Min, m_formConfig->m_edSafeCond3Min);
	m_conSafeCond3Max = xqcon_create<XQLineEditConnector>(
		m_safeCond3Max, m_formConfig->m_edSafeCond3Max);
	m_conPersistentCondEntry = xqcon_create<XQComboBoxConnector>(
		m_persistentCondEntry, m_formConfig->m_cmbPersistentCondEntry, ref(tr_meas));
	m_conPersistentCondMax = xqcon_create<XQLineEditConnector>(
		m_persistentCondMax, m_formConfig->m_edPersistentCondMax);
	m_conNonPersistentCondEntry = xqcon_create<XQComboBoxConnector>(
		m_nonPersistentCondEntry, m_formConfig->m_cmbNonPersistentCondEntry, ref(tr_meas));
	m_conNonPersistentCondMin = xqcon_create<XQLineEditConnector>(
		m_nonPersistentCondMin, m_formConfig->m_edNonPersistentCondMin);
	m_conPCSHWait = xqcon_create<XQLineEditConnector>(
		m_pcshWait, m_formConfig->m_edPCSHWait);

	for(Transaction tr( *this);; ++tr) {
		tr[ *allowPersistent()] = false;
		tr[ *approach()].add("Linear");
		tr[ *approach()].add("Oscillating");
		tr[ *m_pcshWait] = 40.0; //sec
		tr[ *m_safeCond1Max] = 100.0;
		tr[ *m_safeCond2Max] = 100.0;
		tr[ *m_safeCond3Max] = 100.0;
		tr[ *targetField()].setUIEnabled(false);
		tr[ *sweepRate()].setUIEnabled(false);
		tr[ *targetField()].setUIEnabled(false);
		tr[ *sweepRate()].setUIEnabled(false);
		tr[ *allowPersistent()].setUIEnabled(false);
		m_lsnConfigShow = tr[ *m_configShow].onTouch().connectWeakly(
			shared_from_this(), &XMagnetPS::onConfigShow,
			XListener::FLAG_MAIN_THREAD_CALL | XListener::FLAG_AVOID_DUP);
		if(tr.commit())
			break;
	}
}
void
XMagnetPS::showForms() {
    m_form->showNormal();
    m_form->raise();
}
void
XMagnetPS::onConfigShow(const Snapshot &shot, XTouchableNode *) {
    m_formConfig->showNormal();
    m_formConfig->raise();
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
bool
XMagnetPS::isSafeConditionSatisfied(const Snapshot &shot, const Snapshot &shot_entries) {
	if(shared_ptr<XScalarEntry> entry = shot[ *m_safeCond1Entry]) {
		try {
			double x = shot_entries[ *entry->value()];
			if((x >= shot[ *m_safeCond1Max]) || (x <= shot[ *m_safeCond1Min]))
				return false;
		}
		catch(NodeNotFoundError &) {
		}
	}
	if(shared_ptr<XScalarEntry> entry = shot[ *m_safeCond2Entry]) {
		try {
			double x = shot_entries[ *entry->value()];
			if((x >= shot[ *m_safeCond2Max]) || (x <= shot[ *m_safeCond2Min]))
				return false;
		}
		catch(NodeNotFoundError &) {
		}
	}
	if(shared_ptr<XScalarEntry> entry = shot[ *m_safeCond3Entry]) {
		try {
			double x = shot_entries[ *entry->value()];
			if((x >= shot[ *m_safeCond3Max]) || (x <= shot[ *m_safeCond3Min]))
				return false;
		}
		catch(NodeNotFoundError &) {
		}
	}
	return true;
}
bool
XMagnetPS::isPersistentStabilized(const Snapshot &shot, const Snapshot &shot_entries, const XTime &pcsh_off_time) {
	if(shared_ptr<XScalarEntry> entry = shot[ *m_persistentCondEntry]) {
		try {
			double x = shot_entries[ *entry->value()];
			if(x >= shot[ *m_persistentCondMax])
				return false;
		}
		catch(NodeNotFoundError &) {
		}
	}
	if(XTime::now() - pcsh_off_time < std::max(10.0, (double)shot[ *m_pcshWait]))
		return false;
	return true;
}
bool
XMagnetPS::isNonPersistentStabilized(const Snapshot &shot, const Snapshot &shot_entries, const XTime &pcsh_on_time) {
	if(shared_ptr<XScalarEntry> entry = shot[ *m_nonPersistentCondEntry]) {
		try {
			double x = shot_entries[ *entry->value()];
			if(x <= shot[ *m_nonPersistentCondMin])
				return false;
		}
		catch(NodeNotFoundError &) {
		}
	}
	if(XTime::now() - pcsh_on_time < std::max(10.0, (double)shot[ *m_pcshWait]))
		return false;
	return true;
}
double
XMagnetPS::limitSweepRate(double field, double rate, const Snapshot &shot) {
	if((shot[ *m_rateLimit1UBound] > 0.0) && (fabs(field) < shot[ *m_rateLimit1UBound])) {
		return std::min( rate, (double)shot[ *m_rateLimit1]);
	}
	if((shot[ *m_rateLimit2UBound] > 0.0) && (fabs(field) < shot[ *m_rateLimit2UBound])) {
		return std::min( rate, (double)shot[ *m_rateLimit2]);
	}
	if((shot[ *m_rateLimit3UBound] > 0.0) && (fabs(field) < shot[ *m_rateLimit3UBound])) {
		return std::min( rate, (double)shot[ *m_rateLimit3]);
	}
	if((shot[ *m_rateLimit4UBound] > 0.0) && (fabs(field) < shot[ *m_rateLimit4UBound])) {
		return std::min( rate, (double)shot[ *m_rateLimit4]);
	}
	if((shot[ *m_rateLimit5UBound] > 0.0) && (fabs(field) < shot[ *m_rateLimit5UBound])) {
		return std::min( rate, (double)shot[ *m_rateLimit5]);
	}
	return rate;
}
double
XMagnetPS::limitTargetField(double field, const Snapshot &shot) {
	double max_h = std::max((double)shot[ *m_rateLimit1UBound], (double)shot[ *m_rateLimit2UBound]);
	max_h = std::max(max_h, (double)shot[ *m_rateLimit3UBound]);
	max_h = std::max(max_h, (double)shot[ *m_rateLimit4UBound]);
	max_h = std::max(max_h, (double)shot[ *m_rateLimit5UBound]);
	if((max_h > 0.0) && (fabs(field) > max_h)) return max_h * field / fabs(field);
	return field;
}

void *
XMagnetPS::execute(const atomic<bool> &terminated) {
	double havg = 0.0;
	XTime lasttime = XTime::now();
	XTime last_unstab_time = XTime::now();
	XTime pcsh_time = XTime::now();
	pcsh_time -= Snapshot( *this)[ *m_pcshWait];
	double field_resolution;
	bool is_pcs_fitted;
	bool last_pcsh;
	trans( *m_aborting) = false;

	field_resolution = fieldResolution();
	is_pcs_fitted = isPCSFitted();
	trans( *sweepRate()) = getSweepRate();
	trans( *targetField()) = getTargetField();
	last_pcsh = isPCSHeaterOn();

	targetField()->setUIEnabled(true);
	sweepRate()->setUIEnabled(true);

	double target_field_old = Snapshot( *this)[ *targetField()];
	double target_corr = 0.0;

	if(is_pcs_fitted) allowPersistent()->setUIEnabled(true);
	for(Transaction tr( *this);; ++tr) {
		m_lsnRate = tr[ *sweepRate()].onValueChanged().connectWeakly(
			shared_from_this(), &XMagnetPS::onRateChanged);
		if(tr.commit())
			break;
	}

    while( !terminated) {
		msecsleep(100);
		double magnet_field;
		double output_field;
		double output_current;
		double output_volt;
		double target_field_ps;
		bool pcs_heater = true;

		Snapshot shot_entries( *m_entries);

		try {
			// Reading magnet status.
			output_field = getOutputField();
			output_current = getOutputCurrent();
			output_volt = getOutputVolt();
			target_field_ps = getTargetField();
			if(is_pcs_fitted) {
				pcs_heater = isPCSHeaterOn();
				if(pcs_heater != last_pcsh)
					pcsh_time = XTime::now();
				last_pcsh = pcs_heater;
			}
			if( !is_pcs_fitted || pcs_heater) {
				magnet_field = output_field;
			}
			else {
				magnet_field = getPersistentField();
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
			Snapshot &shot(tr);
			tr[ *magnetField()] = magnet_field;
			tr[ *outputField()] = output_field;
			tr[ *outputCurrent()] = output_current;
			tr[ *outputVolt()] = output_volt;
			tr[ *pcsHeater()] = pcs_heater && is_pcs_fitted;

			tr[ *persistent()] = !pcs_heater && is_pcs_fitted && isPersistentStabilized(shot, shot_entries, pcsh_time);

			if(shot[ *m_aborting]) {
				//Aborting.
				tr[ *targetField()].setUIEnabled(false);
				tr[ *targetField()] = 0;
				tr[ *sweepRate()] = limitSweepRate(magnet_field, 1.0, shot) / 2.0; //-0.5T/min.
			}
			//Limits sweep rate and field by software.
			double sweep_rate =  limitSweepRate(magnet_field, shot[ *sweepRate()], shot);
			if(sweep_rate != shot[ *sweepRate()]) {
				m_statusPrinter->printMessage(getLabel() + " " +
											  i18n("Limits sweep rate."));
				tr[ *sweepRate()] =  sweep_rate;
			}

			XTime newtime = XTime::now();
			double dt = fabs(newtime - lasttime);
			//Estimates field deviation.
			havg = (havg - magnet_field) * exp( -0.1 * dt) + magnet_field; //LPF by 10sec.
			tr[ *stabilized()] = std::max(fabs(magnet_field - target_field_ps), fabs(havg - target_field_ps));
			double field_resolution_for_stab = field_resolution * 1.3;
			if(tr[ *stabilized()] > field_resolution_for_stab)
				last_unstab_time = XTime::now();
			if(tr.commit()) {
				lasttime = newtime;

				//Checks abort condition.
				if( !shot[ *m_aborting] && !isSafeConditionSatisfied(shot, shot_entries)) {
					m_statusPrinter->printMessage(getLabel() + " " +
												  i18n("Aborting."));
					trans( *m_aborting) = true;
					break;
				}
				if( shot[ *m_aborting] && isSafeConditionSatisfied(shot, shot_entries)) {
					m_statusPrinter->printMessage(getLabel() + " " +
												  i18n("Safe conditions are satisfied."));
					trans( *m_aborting) = false;
					trans( *targetField()).setUIEnabled(true);
					break;
				}

				try {
					shared_ptr<XMagnetPS> secondaryps = shot[ *m_secondaryPS];
					if(secondaryps.get() == this)
						secondaryps.reset();

					if(pcs_heater || !is_pcs_fitted) {
						//pcs heater is on or not fitted.
						if((target_field_old != shot[ *targetField()]) ||
							((fabs(target_field_ps - target_field_old - target_corr) > field_resolution) &&
								(fabs(target_field_ps - magnet_field) < field_resolution))) {
							//Target has changed, or field has reached the temporary target.
							if( !is_pcs_fitted || isNonPersistentStabilized(shot, shot_entries, pcsh_time)) {
								//Sweeping starts.
								double next_target_ps = shot[ *targetField()];
								if(target_field_old != shot[ *targetField()])
									target_corr = 0.0;
								target_field_old = shot[ *targetField()];
								if(shot[ *approach()] == APPROACH_OSC) {
									next_target_ps += 0.2 * (next_target_ps - magnet_field) + target_corr;
								}
								target_corr = 0.0;
								if((next_target_ps * magnet_field < 0) && (fabs(magnet_field) > field_resolution) &&
									!canChangePolarityDuringSweep()) {
									target_corr = next_target_ps - shot[ *targetField()];
									next_target_ps = 0.0; //First go to zero before setting target with different polarity.
								}
								//Limits target.
								double x =  limitTargetField(next_target_ps, shot);
								if(x != next_target_ps) {
									m_statusPrinter->printMessage(getLabel() + " " +
																  i18n("Limits field."));
									next_target_ps = x;
								}
								setPoint(next_target_ps);
								toSetPoint();
								if(secondaryps) {
									double mul = shot[ *m_secondaryPSMultiplier];
									if(fabs(mul) > 0.4) {
										m_statusPrinter->printMessage(getLabel() + " " +
																	  i18n("Multiplier too large."));
									}
									else {
										double sweep_rate = getSweepRate();
										for(Transaction tr( *secondaryps);; ++tr) {
											tr[ *secondaryps->sweepRate()] = sweep_rate * mul;
											tr[ *secondaryps->targetField()] = next_target_ps * mul;
											tr[ *secondaryps->approach()] = (int)shot[ *approach()];
											if(tr.commit())
												break;
										}
									}
								}
							}
						}
						else {
							if(is_pcs_fitted &&
							   (fabs(magnet_field - shot[ *targetField()]) < field_resolution) && shot[ *allowPersistent()]) {
								if(XTime::now() - last_unstab_time >
										std::max(30.0, (double)shot[ *m_pcshWait]) + field_resolution_for_stab * 1.05 / sweep_rate * 60.0) {
									//field is not sweeping, and persistent mode is allowed
									m_statusPrinter->printMessage(getLabel() + " " +
																  i18n("Turning on Perisistent mode."));
									pcsh_time = XTime::now();
									toPersistent();
								}
							}
						}
					}
					else {
						//pcs heater is off
						if(fabs(magnet_field - shot[ *targetField()]) >= field_resolution) {
							if((fabs(magnet_field - output_field) < field_resolution) &&
								(fabs(target_field_ps  - magnet_field) < field_resolution)) {
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
							else  {
								//set output to persistent field.
								if(shot[ *m_persistent]) {
									setPoint(magnet_field);
									toSetPoint();
								}
							}
						}
						else {
							if(shot[ *m_persistent] && (fabs(output_field) > field_resolution)) {
								toZero();
							}
						}
					}
				}
				catch (XKameError &e) {
					e.print(getLabel());
				}

				break;
			}
		}
	}

	targetField()->setUIEnabled(false);
	sweepRate()->setUIEnabled(false);
	allowPersistent()->setUIEnabled(false);

	m_lsnRate.reset();
	return NULL;
}
