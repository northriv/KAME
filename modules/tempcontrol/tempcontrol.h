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
#ifndef tempcontrolH
#define tempcontrolH
//---------------------------------------------------------------------------
#include "thermometer.h"
#include "dcsource.h"
#include "primarydriverwiththread.h"
#include "xnodeconnector.h"

class XScalarEntry;
class Ui_FrmTempControl;
typedef QForm<QMainWindow, Ui_FrmTempControl> FrmTempControl;

class XTempControl : public XPrimaryDriverWithThread {
public:
	XTempControl(const char *name, bool runtime, Transaction &tr_meas, const shared_ptr<XMeasure> &meas);
	//! usually nothing to do
	virtual ~XTempControl() {}
	//! show all forms belonging to driver
	virtual void showForms();
  
	class XChannel : public XNode {
	public:
		XChannel(const char *name, bool runtime,
			Transaction &tr_list, const shared_ptr<XThermometerList> &list);
		const shared_ptr<XItemNode<XThermometerList, XThermometer> > &thermometer() const {return m_thermometer;}
		const shared_ptr<XThermometerList> &thermometers() const {return m_thermometers;}
		const   shared_ptr<XComboNode> &excitation() const {return m_excitation;}
	private:
		const shared_ptr<XItemNode<XThermometerList, XThermometer> > m_thermometer;
		const shared_ptr<XComboNode> m_excitation;
		const shared_ptr<XThermometerList> m_thermometers;
	};
  
	typedef  XAliasListNode<XChannel> XChannelList;
  
	const shared_ptr<XChannelList> &channels() const {return m_channels;}
	//! heater-control channel
	const shared_ptr<XItemNode<XChannelList, XChannel> > &currentChannel() const {return m_currentChannel;}
	const shared_ptr<XDoubleNode> &targetTemp() const {return m_targetTemp;}
	const shared_ptr<XDoubleNode> &manualPower() const {return m_manualPower;}
	const shared_ptr<XDoubleNode> &prop() const {return m_prop;}
	const shared_ptr<XDoubleNode> &interval() const {return m_int;}
	const shared_ptr<XDoubleNode> &deriv() const {return m_deriv;}
	const shared_ptr<XComboNode> &heaterMode() const {return m_heaterMode;}
	const shared_ptr<XComboNode> &powerRange() const {return m_powerRange;}
	const shared_ptr<XDoubleNode> &heaterPower() const {return m_heaterPower;}
	const shared_ptr<XDoubleNode> &sourceTemp() const {return m_sourceTemp;}
	const shared_ptr<XItemNode<XDriverList, XDCSource> > &extDCSource() const {return m_extDCSource;}
	const shared_ptr<XComboNode> &extDCSourceChannel() const {return m_extDCSourceChannel;}
	//! holds an averaged error between target temp and actual one
	const shared_ptr<XDoubleNode> &stabilized() const {return m_stabilized;}
protected:
	//! This function will be called when raw data are written.
	//! Implement this function to convert the raw data to the record (Payload).
	//! \sa analyze()
	virtual void analyzeRaw(RawDataReader &reader, Transaction &tr) throw (XRecordError&);
	//! This function is called after committing XPrimaryDriver::analyzeRaw() or XSecondaryDriver::analyze().
	//! This might be called even if the record is invalid (time() == false).
	virtual void visualize(const Snapshot &shot);
  
	//! register channel names in your constructor
	//! \param multiread if true, simultaneous reading of multi channels.
	//! \param channel_names array of pointers to channel name. ends with null pointer.
	void createChannels(Transaction &tr, const shared_ptr<XMeasure> &meas,
						bool multiread, const char **channel_names, const char **excitations);
  
	//! reads sensor value from the instrument
	virtual double getRaw(shared_ptr<XChannel> &channel) = 0;
	//! reads a value in Kelvin from the instrument
	virtual double getTemp(shared_ptr<XChannel> &channel) = 0;
	//! obtains current heater power
	//! \sa m_heaterPowerUnit()
	virtual double getHeater() = 0;
	//! ex. "W", "dB", or so
	virtual const char *m_heaterPowerUnit() = 0;
  
	virtual void onPChanged(double p) = 0;
	virtual void onIChanged(double i) = 0;
	virtual void onDChanged(double d) = 0;
	virtual void onTargetTempChanged(double temp) = 0;
	virtual void onManualPowerChanged(double pow) = 0;
	virtual void onHeaterModeChanged(int mode) = 0;
	virtual void onPowerRangeChanged(int range) = 0;
	virtual void onCurrentChannelChanged(const shared_ptr<XChannel> &ch) = 0;
	virtual void onExcitationChanged(const shared_ptr<XChannel> &ch, int exc) = 0;
private:
	void onPChanged(const Snapshot &shot, XValueNodeBase *);
	void onIChanged(const Snapshot &shot, XValueNodeBase *);
	void onDChanged(const Snapshot &shot, XValueNodeBase *);
	void onTargetTempChanged(const Snapshot &shot, XValueNodeBase *);
	void onManualPowerChanged(const Snapshot &shot, XValueNodeBase *);
	void onHeaterModeChanged(const Snapshot &shot, XValueNodeBase *);
	void onPowerRangeChanged(const Snapshot &shot, XValueNodeBase *);
	void onCurrentChannelChanged(const Snapshot &shot, XValueNodeBase *);
	void onExcitationChanged(const Snapshot &shot, XValueNodeBase *);
	void onExtDCSourceChanged(const Snapshot &shot, XValueNodeBase *);

	const shared_ptr<XChannelList> m_channels;
	shared_ptr<XItemNode<XChannelList, XChannel> > m_currentChannel;
	shared_ptr<XItemNode<XChannelList, XChannel> > m_setupChannel;
	const shared_ptr<XDoubleNode> m_targetTemp;
	const shared_ptr<XDoubleNode> m_manualPower;
	const shared_ptr<XDoubleNode> m_prop, m_int, m_deriv;
	const shared_ptr<XComboNode> m_heaterMode;
	const shared_ptr<XComboNode> m_powerRange;
	const shared_ptr<XDoubleNode> m_heaterPower, m_sourceTemp;
	const shared_ptr<XItemNode<XDriverList, XDCSource> > m_extDCSource;
	const shared_ptr<XComboNode> m_extDCSourceChannel;
	//! holds an averaged error between target temp and actual one
	const shared_ptr<XDoubleNode> m_stabilized;
  
	shared_ptr<XListener> m_lsnOnPChanged, m_lsnOnIChanged, m_lsnOnDChanged,
		m_lsnOnTargetTempChanged, m_lsnOnManualPowerChanged, m_lsnOnHeaterModeChanged,
		m_lsnOnPowerRangeChanged, m_lsnOnCurrentChannelChanged,
		m_lsnOnSetupChannelChanged, m_lsnOnExcitationChanged, m_lsnOnExtDCSourceChanged;

	void onSetupChannelChanged(const Snapshot &shot, XValueNodeBase *);

	std::deque<shared_ptr<XScalarEntry> > m_entry_temps;
	std::deque<shared_ptr<XScalarEntry> > m_entry_raws;
 
	const qshared_ptr<FrmTempControl> m_form;
	bool m_multiread;

	xqcon_ptr m_conCurrentChannel, m_conSetupChannel,
		m_conHeaterMode, m_conPowerRange,
		m_conExcitation, m_conThermometer;
	xqcon_ptr m_conTargetTemp, m_conManualPower, m_conP, m_conI, m_conD;
	xqcon_ptr m_conHeater;
	xqcon_ptr m_conTemp;
	xqcon_ptr m_conExtDCSource, m_conExtDCSourceChannel;
	
	double pid(XTime time, double temp);
	double m_pidAccum;
	double m_pidLastTemp;
	XTime m_pidLastTime;
	
	void *execute(const atomic<bool> &);
  
};

#endif
