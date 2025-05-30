/***************************************************************************
        Copyright (C) 2002-2025 Kentaro Kitagawa
		                   kitag@issp.u-tokyo.ac.jp
		
		This program is free software; you can redistribute it and/or
		modify it under the terms of the GNU Library General Public
		License as published by the Free Software Foundation; either
		version 2 of the License, or (at your option) any later version.
		
		You should have received a copy of the GNU Library General 
		Public License and a list of authors along with this program; 
		see the files COPYING and AUTHORS.
 ***************************************************************************/
#include "tempcontrol.h"
#include "ui_tempcontrolform.h"
#include "interface.h"
#include "analyzer.h"
#include "xnodeconnector.h"
#include <QStatusBar>
#include <QToolBox>

XTempControl::XChannel::XChannel(const char *name, bool runtime,
	Transaction &tr_list, const shared_ptr<XThermometerList> &list) :
	XNode(name, runtime),
	m_thermometer(create<XItemNode<XThermometerList,
		XThermometer> > ("Thermometer", false, ref(tr_list), list)),
    m_excitation(create<XComboNode> ("Excitation", true)),
    m_enabled(create<XBoolNode> ("Enabled", true)),
    m_scanDwellSeconds(create<XDoubleNode> ("ScanDwellSeconds", true)),
    // m_instrumentThermoeterTable(create<XComboNode> ("InstrumentThermometerTable", true)),
    m_thermometers(list),
    m_info(create<XStringNode> ("Info", true)) {
    iterate_commit([=](Transaction &tr){
        tr[ enabled()] = true;
    });
}

XTempControl::Loop::Loop(const char *name, bool runtime, shared_ptr<XTempControl> tempctrl, Transaction &tr,
		unsigned int idx, Transaction &tr_meas, const shared_ptr<XMeasure> &meas) :
		XNode(name, runtime),
		m_tempctrl(tempctrl),
		m_idx(idx),
		m_targetTemp(create<XDoubleNode> ("TargetTemp" , true, "%.5g")),
		m_manualPower(create<XDoubleNode> ("ManualPower", true, "%.4g")),
		m_prop(create<XDoubleNode> ("P", false, "%.4g")),
		m_int(create<XDoubleNode> ("I", false, "%.4g")),
		m_deriv(create<XDoubleNode> ("D", false, "%.4g")),
		m_heaterMode(create<XComboNode> ("HeaterMode", false, true)),
		m_powerRange(create<XComboNode> ("PowerRange", false, true)),
		m_powerMax(create<XDoubleNode> ("PowerMax", false, "%.4g")),
		m_powerMin(create<XDoubleNode> ("PowerMin", false, "%.4g")),
		m_heaterPower(create<XDoubleNode> ("HeaterPower", false, "%.4g")),
		m_sourceTemp(create<XDoubleNode> ("SourceTemp", false, "%.5g")),
		m_stabilized(create<XDoubleNode> ("Stabilized", true, "%g")),
		m_extDevice(create<XItemNode<XDriverList, XDCSource, XFlowControllerDriver> > ("ExtDevice", false, ref(tr_meas), meas->drivers())),
		m_extDCSourceChannel(create<XComboNode> ("ExtDCSourceChannel", false, true)),
		m_extIsPositive(create<XBoolNode> ("ExtIsPositive", false)) {
	m_currentChannel =
		create<XItemNode<XChannelList, XChannel> >("CurrentChannel", true, tr,
		tempctrl->m_channels);

    iterate_commit([=](Transaction &tr){
		m_lsnOnExtDeviceChanged = tr[ *m_extDevice].onValueChanged().connectWeakly(
			shared_from_this(), &XTempControl::Loop::onExtDeviceChanged);
    });

	m_currentChannel->setUIEnabled(false);
	m_powerRange->setUIEnabled(false);
	m_heaterMode->setUIEnabled(false);
	m_prop->setUIEnabled(false);
	m_int->setUIEnabled(false);
	m_deriv->setUIEnabled(false);
	m_manualPower->setUIEnabled(false);
	m_powerMax->setUIEnabled(true);
	m_powerMin->setUIEnabled(true);
	m_targetTemp->setUIEnabled(false);

	m_extDevice->setUIEnabled(true);
	m_extDCSourceChannel->setUIEnabled(true);
	m_extIsPositive->setUIEnabled(true);

	tempctrl->m_form->m_toolBox->setItemText(m_idx, getLabel());
}
void
XTempControl::Loop::start() {
	auto tempctrl = m_tempctrl.lock();
	if( !tempctrl) return;
    iterate_commit([=](Transaction &tr){
		const Snapshot &shot(tr);
		if(hasExtDevice(shot)) {
			tr[ m_heaterMode].clear();
			tr[ m_heaterMode].add("Off");
			tr[ m_heaterMode].add("PID");
			tr[ m_heaterMode].add("Man");
		}
		else
			tr[ *m_powerRange].setUIEnabled(true);
    });

	m_currentChannel->setUIEnabled(true);
	m_heaterMode->setUIEnabled(true);
	m_prop->setUIEnabled(true);
	m_int->setUIEnabled(true);
	m_deriv->setUIEnabled(true);
	m_manualPower->setUIEnabled(true);
	m_targetTemp->setUIEnabled(true);

	m_extDevice->setUIEnabled(false);
	m_extDCSourceChannel->setUIEnabled(false);
	m_extIsPositive->setUIEnabled(false);

	m_tempAvg = 0.0;
	m_tempErrAvg = 0.0;
	m_lasttime = XTime::now();

    iterate_commit([=](Transaction &tr){
        m_lsnOnPChanged = tr[ *m_prop].onValueChanged().connectWeakly(shared_from_this(), &XTempControl::Loop::onPChanged);
        m_lsnOnIChanged = tr[ *m_int].onValueChanged().connectWeakly(shared_from_this(), &XTempControl::Loop::onIChanged);
        m_lsnOnDChanged = tr[ *m_deriv].onValueChanged().connectWeakly(shared_from_this(), &XTempControl::Loop::onDChanged);
        m_lsnOnTargetTempChanged = tr[ *m_targetTemp].onValueChanged().connectWeakly(shared_from_this(), &XTempControl::Loop::onTargetTempChanged);
        m_lsnOnManualPowerChanged = tr[ *m_manualPower].onValueChanged().connectWeakly(shared_from_this(), &XTempControl::Loop::onManualPowerChanged);
        m_lsnOnHeaterModeChanged = tr[ *m_heaterMode].onValueChanged().connectWeakly(shared_from_this(), &XTempControl::Loop::onHeaterModeChanged);
        m_lsnOnPowerRangeChanged = tr[ *m_powerRange].onValueChanged().connectWeakly(shared_from_this(), &XTempControl::Loop::onPowerRangeChanged);
        m_lsnOnCurrentChannelChanged
            = tr[ *m_currentChannel].onValueChanged().connectWeakly(shared_from_this(), &XTempControl::Loop::onCurrentChannelChanged);
    });
}
void
XTempControl::Loop::stop() {
	m_currentChannel->setUIEnabled(false);
	m_powerRange->setUIEnabled(false);
	m_heaterMode->setUIEnabled(false);
	m_prop->setUIEnabled(false);
	m_int->setUIEnabled(false);
	m_deriv->setUIEnabled(false);
	m_manualPower->setUIEnabled(false);
	m_targetTemp->setUIEnabled(false);

	m_extDevice->setUIEnabled(true);
	m_extDCSourceChannel->setUIEnabled(true);
	m_extIsPositive->setUIEnabled(true);

	m_lsnOnPChanged.reset();
	m_lsnOnIChanged.reset();
	m_lsnOnDChanged.reset();
	m_lsnOnTargetTempChanged.reset();
	m_lsnOnManualPowerChanged.reset();
	m_lsnOnPowerMaxChanged.reset();
	m_lsnOnPowerMinChanged.reset();
	m_lsnOnHeaterModeChanged.reset();
	m_lsnOnPowerRangeChanged.reset();
	m_lsnOnCurrentChannelChanged.reset();
}
void
XTempControl::Loop::update(double temp) {
	auto tempctrl = m_tempctrl.lock();
	if( !tempctrl) return;
	Snapshot shot( *tempctrl);

    //calculates std. deviations in some periods
    double tau = tempctrl->currentIntervalSettingInSec(shot, m_idx) * 4.0;
    tau = std::max(1.0, tau);
    tau = std::min(300.0, tau);
	XTime newtime = XTime::now();
	double dt = newtime - m_lasttime;
	m_lasttime = newtime;
	double terr = temp - shot[ *m_targetTemp];
	m_tempAvg = (m_tempAvg - temp) * exp( -dt / tau) + temp;
	m_tempErrAvg = (m_tempErrAvg - terr * terr) * exp( -dt / tau) + terr * terr;
    m_tempErrAvg = std::min(m_tempErrAvg, temp * temp * 0.04);

	double power = 0.0;
	shared_ptr<XDCSource> dcsrc = shot[ *m_extDevice];
	shared_ptr<XFlowControllerDriver> flowctrl = shot[ *m_extDevice];
	if(dcsrc || flowctrl) {
		double limit_min = shot[ *m_powerMin];
		double limit_max = shot[ *m_powerMax];
		if(shot[ *m_heaterMode].to_str() == "PID") {
			power = pid(shot, newtime, temp);
		}
		if(shot[ *m_heaterMode].to_str() == "Man") {
			power = shot[ *m_manualPower];
		}
		if(dcsrc) {
			int ch = shot[ *m_extDCSourceChannel];
			if(ch >= 0) {
				limit_max = std::min(limit_max, dcsrc->max(ch, false));
				power = (limit_max - limit_min) * sqrt(power) / 10.0 + limit_min;
				dcsrc->changeValue(ch, power, false);
			}
		}
		if(flowctrl) {
			limit_max = std::min(limit_max, Snapshot( *flowctrl)[ *flowctrl].fullScale());
			power = (limit_max - limit_min) * power / 100.0 + limit_min;
			trans( *flowctrl->target()) = power;
		}
	}
	else
		power = tempctrl->getHeater(m_idx);

    iterate_commit([=](Transaction &tr){
		tr[ *m_sourceTemp] = temp;
		tr[ *m_stabilized] = sqrt(m_tempErrAvg); //stderr
		tr[ *m_heaterPower] = power;
    });
    //UIs should be handled in the main thread.
    tempctrl->m_tlkOnLoopUpdated.talk(m_idx, XString(getLabel() + formatString(": %.5g K, %.3g%s", temp, power, tempctrl->m_heaterPowerUnit(m_idx))));
}
void XTempControl::onLoopUpdated(int index, const XString &str) {
    m_form->m_toolBox->setItemText(index, str);
}

double XTempControl::Loop::pid(const Snapshot &shot, XTime time, double temp) {
	double p = shot[ *m_prop];
	double i = shot[ *m_int];
	double d = shot[ *m_deriv];

	double dt = temp - shot[ *m_targetTemp];
	if(shot[ *m_extIsPositive])
		dt *= -1.0;
	double dxdt = 0.0;
	double acc = 0.0;
	if((i > 0) && (time - m_pidLastTime < i)) {
		m_pidAccum += (time - m_pidLastTime) * dt;
		dxdt = (temp - m_pidLastTemp) / (time - m_pidLastTime);
		acc = m_pidAccum / i;
		acc = -std::min(std::max( -acc * p, -2.0), 100.0) / p;
		m_pidAccum = acc * i;
	}
	else
		m_pidAccum = 0;

	m_pidLastTime = time;
	m_pidLastTemp = temp;

	return -(dt + acc + dxdt * d) * p;
}
void XTempControl::Loop::onExtDeviceChanged(const Snapshot &shot, XValueNodeBase *) {
    iterate_commit([=](Transaction &tr){
		const Snapshot &shot(tr);
		tr[ *m_extDCSourceChannel].clear();
		shared_ptr<XDCSource> dcsrc = shot[ *m_extDevice];
		if(dcsrc) {
			//registers channel names.
            auto strings(dcsrc->channel()->itemStrings(Snapshot( *dcsrc)));
            for(auto it = strings.begin(); it != strings.end(); it++) {
				tr[ *m_extDCSourceChannel].add(it->label);
			}
		}
    });
}
void XTempControl::Loop::onPChanged(const Snapshot &shot, XValueNodeBase *) {
	auto tempctrl = m_tempctrl.lock();
	if( !tempctrl) return;
	try {
		Snapshot shot( *this);
		if( !hasExtDevice(shot))
			tempctrl->onPChanged(m_idx, shot[ *m_prop]);
	}
	catch(XInterface::XInterfaceError& e) {
		e.print();
	}
}
void XTempControl::Loop::onIChanged(const Snapshot &shot, XValueNodeBase *) {
	auto tempctrl = m_tempctrl.lock();
	if( !tempctrl) return;
	try {
		Snapshot shot( *tempctrl);
		if( !hasExtDevice(shot))
			tempctrl->onIChanged(m_idx, shot[ *m_int]);
	}
	catch(XInterface::XInterfaceError& e) {
		e.print();
	}
}
void XTempControl::Loop::onDChanged(const Snapshot &shot, XValueNodeBase *) {
	auto tempctrl = m_tempctrl.lock();
	if( !tempctrl) return;
	try {
		Snapshot shot( *tempctrl);
		if( !hasExtDevice(shot))
			tempctrl->onDChanged(m_idx, shot[ *m_deriv]);
	}
	catch(XInterface::XInterfaceError& e) {
		e.print();
	}
}
void XTempControl::Loop::onTargetTempChanged(const Snapshot &shot, XValueNodeBase *) {
	auto tempctrl = m_tempctrl.lock();
	if( !tempctrl) return;
	try {
		Snapshot shot( *tempctrl);
		if( !hasExtDevice(shot))
			tempctrl->onTargetTempChanged(m_idx, shot[ *m_targetTemp]);
	}
	catch(XInterface::XInterfaceError& e) {
		e.print();
	}
}
void XTempControl::Loop::onManualPowerChanged(const Snapshot &shot, XValueNodeBase *) {
	auto tempctrl = m_tempctrl.lock();
	if( !tempctrl) return;
	try {
		Snapshot shot( *tempctrl);
		if( !hasExtDevice(shot))
			tempctrl->onManualPowerChanged(m_idx, shot[ *m_manualPower]);
	}
	catch(XInterface::XInterfaceError& e) {
		e.print();
	}
}
void XTempControl::Loop::onHeaterModeChanged(const Snapshot &shot, XValueNodeBase *) {
	auto tempctrl = m_tempctrl.lock();
	if( !tempctrl) return;
	m_pidAccum = 0;
	try {
		Snapshot shot( *tempctrl);
		if( !hasExtDevice(shot))
			tempctrl->onHeaterModeChanged(m_idx, shot[ *m_heaterMode]);
	}
	catch(XInterface::XInterfaceError& e) {
		e.print();
	}
}
void XTempControl::Loop::onPowerMaxChanged(const Snapshot &shot, XValueNodeBase *) {
	auto tempctrl = m_tempctrl.lock();
	if( !tempctrl) return;
	try {
		Snapshot shot( *tempctrl);
		if( !hasExtDevice(shot))
			tempctrl->onPowerRangeChanged(m_idx, shot[ *m_powerMax]);
	}
	catch(XInterface::XInterfaceError& e) {
		e.print();
	}
}
void XTempControl::Loop::onPowerMinChanged(const Snapshot &shot, XValueNodeBase *) {
	auto tempctrl = m_tempctrl.lock();
	if( !tempctrl) return;
	try {
		Snapshot shot( *tempctrl);
		if( !hasExtDevice(shot))
			tempctrl->onPowerRangeChanged(m_idx, shot[ *m_powerMin]);
	}
	catch(XInterface::XInterfaceError& e) {
		e.print();
	}
}
void XTempControl::Loop::onPowerRangeChanged(const Snapshot &shot, XValueNodeBase *) {
	auto tempctrl = m_tempctrl.lock();
	if( !tempctrl) return;
	try {
		Snapshot shot( *tempctrl);
		if( !hasExtDevice(shot))
			tempctrl->onPowerRangeChanged(m_idx, shot[ *m_powerRange]);
	}
	catch(XInterface::XInterfaceError& e) {
		e.print();
	}
}
void XTempControl::Loop::onCurrentChannelChanged(const Snapshot &shot, XValueNodeBase *) {
	auto tempctrl = m_tempctrl.lock();
	if( !tempctrl) return;
	m_pidAccum = 0;
	try {
		shared_ptr<XChannel> ch(shot[ *m_currentChannel]);
		if( !ch)
			return;
		tempctrl->onCurrentChannelChanged(m_idx, ch);
	}
	catch(XInterface::XInterfaceError& e) {
		e.print();
	}
}

XTempControl::XTempControl(const char *name, bool runtime,
	Transaction &tr_meas, const shared_ptr<XMeasure> &meas) :
	XPrimaryDriverWithThread(name, runtime, ref(tr_meas), meas),
	m_channels(create<XChannelList> ("Channels", false)),
    m_form(new FrmTempControl) {

    iterate_commit([=](Transaction &tr){
		m_setupChannel =
			create<XItemNode<XChannelList, XChannel> >(tr, "SetupChannel", true, tr, m_channels);
    });
	m_conSetupChannel = xqcon_create<XQComboBoxConnector> (m_setupChannel,
		m_form->m_cmbSetupChannel, Snapshot( *m_channels));

    iterate_commit([=](Transaction &tr){
		m_lsnOnSetupChannelChanged = tr[ *m_setupChannel].onValueChanged().connectWeakly(
            shared_from_this(), &XTempControl::onSetupChannelChangedInternal);
        m_lsnOnLoopUpdated = m_tlkOnLoopUpdated.connectWeakly(shared_from_this(), &XTempControl::onLoopUpdated,
            Listener::FLAG_MAIN_THREAD_CALL | Listener::FLAG_AVOID_DUP);
    });

	m_form->statusBar()->hide();
	m_form->setWindowTitle(i18n("TempControl - ") + getLabel());
}

void XTempControl::showForms() {
	//! impliment form->show() here
    m_form->showNormal();
	m_form->raise();
}

void XTempControl::analyzeRaw(RawDataReader &reader, Transaction &tr) {
	try {
		for(;;) {
			//! Since raw buffer is Fast-in Fast-out, use the same sequence of push()es for pop()s
			uint16_t chno = reader.pop<uint16_t> ();
			reader.pop<uint16_t> (); //reserve
			float raw = reader.pop<float> ();
			float temp = reader.pop<float> ();
			if( !m_multiread)
				chno = 0;
			if(chno >= m_entry_temps.size())
				throw XBufferUnderflowRecordError(__FILE__, __LINE__);
			m_entry_temps[chno]->value(tr, temp);
			m_entry_raws[chno]->value(tr, raw);
		}
	}
	catch(XRecordError&) {
	}
}
void XTempControl::visualize(const Snapshot &shot) {
}

void XTempControl::onSetupChannelChangedInternal(const Snapshot &shot, XValueNodeBase *) {
    m_conChannelUIs.clear();
    m_lsnOnExcitationChanged.reset();
    m_lsnOnScanDwellSecChanged.reset();
    m_lsnOnChannelEnableChanged.reset();
	shared_ptr<XChannel> channel = shot[ *m_setupChannel];
	if( !channel)
		return;
    try {
        onSetupChannelChanged(channel);
    }
    catch(XInterface::XInterfaceError& e) {
        e.print();
        return;
    }
    m_conChannelUIs = {
        xqcon_create<XQComboBoxConnector> (
            channel->thermometer(), m_form->m_cmbThermometerSoft, Snapshot( *channel->thermometers())),
        xqcon_create<XQComboBoxConnector> (channel->excitation(),
                                          m_form->m_cmbExcitation, Snapshot( *channel->excitation())),
        xqcon_create<XQToggleButtonConnector> (channel->enabled(),
                                              m_form->m_ckbChannelEnabled),
        xqcon_create<XQDoubleSpinBoxConnector> (channel->scanDwellSeconds(),
                                               m_form->m_dsbChannelDwellSec),
        xqcon_create<XQTextBrowserConnector> (channel->info(),
                                             m_form->m_txtChannelInfo),
    };
    iterate_commit([=](Transaction &tr){
		m_lsnOnExcitationChanged
            = tr[ *channel->excitation()].onValueChanged().connectWeakly(
                shared_from_this(),
                &XTempControl::onExcitationChangedInternal);
        m_lsnOnScanDwellSecChanged
            = tr[ *channel->scanDwellSeconds()].onValueChanged().connectWeakly(
                shared_from_this(),
                &XTempControl::onScanDwellSecondChangedInternal);
        m_lsnOnChannelEnableChanged
            = tr[ *channel->enabled()].onValueChanged().connectWeakly(
                shared_from_this(),
                &XTempControl::onChannelEnableChangedInternal);
    });
}

void XTempControl::createChannels(
    Transaction &tr_meas, const shared_ptr<XMeasure> &meas,
    bool multiread, std::initializer_list<XString> channel_names,
    std::initializer_list<XString> loop_names,
    bool autoscanning, bool disabling) {
	shared_ptr<XScalarEntryList> entries(meas->scalarEntries());
	m_multiread = multiread;

    iterate_commit([=, &tr_meas](Transaction &tr){
        for(auto &&ch: channel_names) {
			shared_ptr<XChannel> channel = m_channels->create<XChannel> (
                tr, ch.c_str(), false, ref(tr_meas), meas->thermometers());
            if( !autoscanning)
                tr[ *channel->scanDwellSeconds()].disable();
            if( !disabling)
                tr[ *channel->enabled()].disable();
        }
    });
	if(multiread) {
		Snapshot shot( *m_channels);
		if(shot.size()) {
			const XNode::NodeList &list( *shot.list());
			for(XNode::const_iterator it = list.begin(); it != list.end(); it++) {
				shared_ptr<XChannel> channel =
					dynamic_pointer_cast<XChannel> (*it);
				shared_ptr<XScalarEntry> entry_temp(
					create<XScalarEntry>(
						QString("Ch.%1").arg(channel->getName()).toLocal8Bit().data(),
						false, dynamic_pointer_cast<XDriver> (shared_from_this()), "%.5g"));
				shared_ptr<XScalarEntry> entry_raw(
					create<XScalarEntry> (
						QString("Ch.%1.raw").arg(
						channel->getName()).toLocal8Bit().data(), false,
						dynamic_pointer_cast<XDriver> (shared_from_this()), "%.5g"));
				m_entry_temps.push_back(entry_temp);
				m_entry_raws.push_back(entry_raw);
				entries->insert(tr_meas, entry_temp);
				entries->insert(tr_meas, entry_raw);
			}
		}
	}
	else {
		shared_ptr<XScalarEntry> entry_temp(create<XScalarEntry> (
			"Temp", false,
			dynamic_pointer_cast<XDriver> (shared_from_this()), "%.5g"));
		shared_ptr<XScalarEntry> entry_raw(create<XScalarEntry> (
			"Raw", false,
			dynamic_pointer_cast<XDriver> (shared_from_this()), "%.5g"));
		m_entry_temps.push_back(entry_temp);
		m_entry_raws.push_back(entry_raw);
		entries->insert(tr_meas, entry_temp);
		entries->insert(tr_meas, entry_raw);
	}
	//creates loops.
    unsigned int num_of_loops = 0;
    for(auto &&lp: loop_names) {
		shared_ptr<Loop> p;
        iterate_commit([=, &p, &tr_meas](Transaction &tr){
			p = create<Loop>(tr,
                lp.c_str(), false,
                dynamic_pointer_cast<XTempControl>(shared_from_this()), tr, num_of_loops,
				ref(tr_meas), meas);
        });
		m_loops.push_back(p);
        num_of_loops++;
    }

    for(int idx = 0; idx < num_of_loops; ++idx) {
        auto lp = loop(idx);
        switch(idx) {
        case 0:
            lp->m_conUIs = {
                xqcon_create<XQComboBoxConnector> (lp->m_currentChannel,
                    m_form->m_cmbSourceChannel, Snapshot( *m_channels)),
                xqcon_create<XQComboBoxConnector> (lp->m_powerRange,
                    m_form->m_cmbPowerRange, Snapshot( *lp->m_powerRange)),
                xqcon_create<XQComboBoxConnector> (lp->m_heaterMode,
                    m_form->m_cmbHeaterMode, Snapshot( *lp->m_heaterMode)),
                xqcon_create<XQLineEditConnector> (lp->m_prop, m_form->m_edP),
                xqcon_create<XQLineEditConnector> (lp->m_int, m_form->m_edI),
                xqcon_create<XQLineEditConnector> (lp->m_deriv, m_form->m_edD),
                xqcon_create<XQLineEditConnector> (lp->m_manualPower, m_form->m_edManHeater),
                xqcon_create<XQLineEditConnector> (lp->m_powerMax, m_form->m_edPowerMax),
                xqcon_create<XQLineEditConnector> (lp->m_powerMin, m_form->m_edPowerMin),
                xqcon_create<XQLineEditConnector> (lp->m_targetTemp, m_form->m_edTargetTemp),
                xqcon_create<XQLCDNumberConnector> (lp->m_heaterPower, m_form->m_lcdHeater),
                xqcon_create<XQLCDNumberConnector> (lp->m_sourceTemp, m_form->m_lcdSourceTemp),
                xqcon_create<XQComboBoxConnector> (lp->m_extDevice, m_form->m_cmbExtDevice, ref(tr_meas)),
                xqcon_create<XQComboBoxConnector> (
                    lp->m_extDCSourceChannel, m_form->m_cmbExtDCSrcCh, Snapshot( *lp->m_extDCSourceChannel)),
                xqcon_create<XQToggleButtonConnector>( lp->m_extIsPositive, m_form->m_ckbExtIsPositive)
            };
            break;
        case 1:
            lp->m_conUIs = {
                xqcon_create<XQComboBoxConnector> (lp->m_currentChannel,
                    m_form->m_cmbSourceChannel2, Snapshot( *m_channels)),
                xqcon_create<XQComboBoxConnector> (lp->m_powerRange,
                    m_form->m_cmbPowerRange2, Snapshot( *lp->m_powerRange)),
                xqcon_create<XQComboBoxConnector> (lp->m_heaterMode,
                    m_form->m_cmbHeaterMode2, Snapshot( *lp->m_heaterMode)),
                xqcon_create<XQLineEditConnector> (lp->m_prop, m_form->m_edP2),
                xqcon_create<XQLineEditConnector> (lp->m_int, m_form->m_edI2),
                xqcon_create<XQLineEditConnector> (lp->m_deriv, m_form->m_edD2),
                xqcon_create<XQLineEditConnector> (lp->m_manualPower, m_form->m_edManHeater2),
                xqcon_create<XQLineEditConnector> (lp->m_powerMax, m_form->m_edPowerMax2),
                xqcon_create<XQLineEditConnector> (lp->m_powerMin, m_form->m_edPowerMin2),
                xqcon_create<XQLineEditConnector> (lp->m_targetTemp, m_form->m_edTargetTemp2),
                xqcon_create<XQLCDNumberConnector> (lp->m_heaterPower, m_form->m_lcdHeater2),
                xqcon_create<XQLCDNumberConnector> (lp->m_sourceTemp, m_form->m_lcdSourceTemp2),
                xqcon_create<XQComboBoxConnector> (lp->m_extDevice, m_form->m_cmbExtDevice2, ref(tr_meas)),
                xqcon_create<XQComboBoxConnector> (
                    lp->m_extDCSourceChannel, m_form->m_cmbExtDCSrcCh2, Snapshot( *lp->m_extDCSourceChannel)),
                xqcon_create<XQToggleButtonConnector>( lp->m_extIsPositive, m_form->m_ckbExtIsPositive2)
            };
            break;
        case 2:
            lp->m_conUIs = {
                xqcon_create<XQComboBoxConnector> (lp->m_currentChannel,
                    m_form->m_cmbSourceChannel3, Snapshot( *m_channels)),
                xqcon_create<XQComboBoxConnector> (lp->m_powerRange,
                    m_form->m_cmbPowerRange3, Snapshot( *lp->m_powerRange)),
                xqcon_create<XQComboBoxConnector> (lp->m_heaterMode,
                    m_form->m_cmbHeaterMode3, Snapshot( *lp->m_heaterMode)),
                xqcon_create<XQLineEditConnector> (lp->m_prop, m_form->m_edP3),
                xqcon_create<XQLineEditConnector> (lp->m_int, m_form->m_edI3),
                xqcon_create<XQLineEditConnector> (lp->m_deriv, m_form->m_edD3),
                xqcon_create<XQLineEditConnector> (lp->m_manualPower, m_form->m_edManHeater3),
                xqcon_create<XQLineEditConnector> (lp->m_powerMax, m_form->m_edPowerMax3),
                xqcon_create<XQLineEditConnector> (lp->m_powerMin, m_form->m_edPowerMin3),
                xqcon_create<XQLineEditConnector> (lp->m_targetTemp, m_form->m_edTargetTemp3),
                xqcon_create<XQLCDNumberConnector> (lp->m_heaterPower, m_form->m_lcdHeater3),
                xqcon_create<XQLCDNumberConnector> (lp->m_sourceTemp, m_form->m_lcdSourceTemp3),
                xqcon_create<XQComboBoxConnector> (lp->m_extDevice, m_form->m_cmbExtDevice3, ref(tr_meas)),
                xqcon_create<XQComboBoxConnector> (
                    lp->m_extDCSourceChannel, m_form->m_cmbExtDCSrcCh3, Snapshot( *lp->m_extDCSourceChannel)),
                xqcon_create<XQToggleButtonConnector>( lp->m_extIsPositive, m_form->m_ckbExtIsPositive3)
            };
            break;
        case 3:
            lp->m_conUIs = {
                xqcon_create<XQComboBoxConnector> (lp->m_currentChannel,
                    m_form->m_cmbSourceChannel4, Snapshot( *m_channels)),
                xqcon_create<XQComboBoxConnector> (lp->m_powerRange,
                    m_form->m_cmbPowerRange4, Snapshot( *lp->m_powerRange)),
                xqcon_create<XQComboBoxConnector> (lp->m_heaterMode,
                    m_form->m_cmbHeaterMode4, Snapshot( *lp->m_heaterMode)),
                xqcon_create<XQLineEditConnector> (lp->m_prop, m_form->m_edP4),
                xqcon_create<XQLineEditConnector> (lp->m_int, m_form->m_edI4),
                xqcon_create<XQLineEditConnector> (lp->m_deriv, m_form->m_edD4),
                xqcon_create<XQLineEditConnector> (lp->m_manualPower, m_form->m_edManHeater4),
                xqcon_create<XQLineEditConnector> (lp->m_powerMax, m_form->m_edPowerMax4),
                xqcon_create<XQLineEditConnector> (lp->m_powerMin, m_form->m_edPowerMin4),
                xqcon_create<XQLineEditConnector> (lp->m_targetTemp, m_form->m_edTargetTemp4),
                xqcon_create<XQLCDNumberConnector> (lp->m_heaterPower, m_form->m_lcdHeater4),
                xqcon_create<XQLCDNumberConnector> (lp->m_sourceTemp, m_form->m_lcdSourceTemp4),
                xqcon_create<XQComboBoxConnector> (lp->m_extDevice, m_form->m_cmbExtDevice4, ref(tr_meas)),
                xqcon_create<XQComboBoxConnector> (
                    lp->m_extDCSourceChannel, m_form->m_cmbExtDCSrcCh4, Snapshot( *lp->m_extDCSourceChannel)),
                xqcon_create<XQToggleButtonConnector>( lp->m_extIsPositive, m_form->m_ckbExtIsPositive4)
            };
            break;
        default:
            break;
        };
    }
    if(num_of_loops < 4) {
        m_form->m_toolBox->removeItem(3);
        m_form->m_pageLoop4->hide();
    }
    if(num_of_loops < 3) {
        m_form->m_toolBox->removeItem(2);
        m_form->m_pageLoop3->hide();
    }
    if(num_of_loops < 2) {
        m_form->m_toolBox->removeItem(1);
        m_form->m_pageLoop2->hide();
    }
    if(num_of_loops < 1) {
        m_form->m_toolBox->removeItem(0);
        m_form->m_pageLoop1->hide();
    }
}

void *
XTempControl::execute(const atomic<bool> &terminated) {
	for(auto it = m_loops.begin(); it != m_loops.end(); ++it) {
		( *it)->start();
	}

	while( !terminated) {
		msecsleep(10);

		auto writer = std::make_shared<RawData>();
		Snapshot shot( *this);
		double raw, temp;
		XTime time_awared = XTime::now();
		// try/catch exception of communication errors
		try {
			if(shot.size(m_channels)) {
				const XNode::NodeList &list( *shot.list(m_channels));
				unsigned int idx = 0;
				for(XNode::const_iterator it = list.begin(); it != list.end(); it++) {
					shared_ptr<XChannel> ch = static_pointer_cast<XChannel>( *it);
					bool src_found = false;
					for(auto lit = m_loops.begin(); lit != m_loops.end(); ++lit) {
						shared_ptr<XChannel> curch = shot[ ( *lit)->m_currentChannel];
						if(curch == ch)
                            src_found = true;
					}
					if(m_multiread || src_found) {
						shared_ptr<XThermometer> thermo = shot[ *ch->thermometer()];
						raw = getRaw(ch);
						temp = ( !thermo) ? getTemp(ch) : thermo->getTemp(raw);

						for(auto lit = m_loops.begin(); lit != m_loops.end(); ++lit) {
							shared_ptr<XChannel> curch = shot[ ( *lit)->m_currentChannel];
							if(curch == ch)
								( *lit)->update(temp);
						}
						writer->push((uint16_t) idx);
						writer->push((uint16_t) 0); // reserve
						writer->push(float(raw));
						writer->push(float(temp));
					}
					idx++;
				}
			}

		}
		catch(XKameError &e) {
			e.print(getLabel() + "; ");
			continue;
		}
		finishWritingRaw(writer, time_awared, XTime::now());
	}

	trans( *m_setupChannel) = shared_ptr<XThermometer>();

	for(auto it = m_loops.begin(); it != m_loops.end(); ++it) {
		( *it)->stop();
	}

	return NULL;
}

void XTempControl::onScanDwellSecondChangedInternal(const Snapshot &, XValueNodeBase *node) {
    try {
        shared_ptr<XChannel> ch;
        Snapshot shot( *channels());
        if(shot.size()) {
            const XNode::NodeList &list( *shot.list());
            for(XNode::const_iterator it = list.begin(); it != list.end(); it++) {
                shared_ptr<XChannel> ch__ =
                    dynamic_pointer_cast<XChannel> ( *it);
                if(ch__->scanDwellSeconds().get() == node)
                    ch = ch__;
            }
        }
        if( !ch)
            return;
        double sec = shot[ *ch->scanDwellSeconds()];
        if(sec <= 0)
            return;
        onScanDwellSecChanged(ch, sec);
    }
    catch(XInterface::XInterfaceError& e) {
        e.print();
    }
}
void XTempControl::onChannelEnableChangedInternal(const Snapshot &, XValueNodeBase *node) {
    try {
        shared_ptr<XChannel> ch;
        Snapshot shot( *channels());
        if(shot.size()) {
            const XNode::NodeList &list( *shot.list());
            for(XNode::const_iterator it = list.begin(); it != list.end(); it++) {
                shared_ptr<XChannel> ch__ =
                    dynamic_pointer_cast<XChannel> ( *it);
                if(ch__->enabled().get() == node)
                    ch = ch__;
            }
        }
        if( !ch)
            return;
        bool enabled = shot[ *ch->enabled()];
        onChannelEnableChanged(ch, enabled);
    }
    catch(XInterface::XInterfaceError& e) {
        e.print();
    }
}
void XTempControl::onExcitationChangedInternal(const Snapshot &, XValueNodeBase *node) {
    try {
        shared_ptr<XChannel> ch;
        Snapshot shot( *channels());
        if(shot.size()) {
            const XNode::NodeList &list( *shot.list());
            for(XNode::const_iterator it = list.begin(); it != list.end(); it++) {
                shared_ptr<XChannel> ch__ =
                    dynamic_pointer_cast<XChannel> ( *it);
                if(ch__->excitation().get() == node)
                    ch = ch__;
            }
        }
        if( !ch)
            return;
        int exc = shot[ *ch->excitation()];
        if(exc < 0)
            return;
        onExcitationChanged(ch, exc);
    }
    catch(XInterface::XInterfaceError& e) {
        e.print();
    }
}
