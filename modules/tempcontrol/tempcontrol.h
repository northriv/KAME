/***************************************************************************
		Copyright (C) 2002-2015 Kentaro Kitagawa
		                   kitagawa@phys.s.u-tokyo.ac.jp
		
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
#include "flowcontroller.h"
#include "primarydriverwiththread.h"
#include "xnodeconnector.h"

class XScalarEntry;
class Ui_FrmTempControl;
typedef QForm<QMainWindow, Ui_FrmTempControl> FrmTempControl;

class XTempControl : public XPrimaryDriverWithThread {
public:
	XTempControl(const char *name, bool runtime, Transaction &tr_meas, const shared_ptr<XMeasure> &meas);
	//! usually nothing to do
    virtual ~XTempControl() = default;
	//! show all forms belonging to driver
    virtual void showForms() override;
  
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

    const shared_ptr<XScalarEntry> &entryTemp(unsigned int ch) const {return m_entry_temps[ch];}
    const shared_ptr<XScalarEntry> &entryRaw(unsigned int ch) const {return m_entry_raws[ch];}

	//! LOOPs
	unsigned int numOfLoops() const {return m_loops.size();}
    XString loopLabel(unsigned int lp) const {return loop(lp)->getLabel();}
	const shared_ptr<XItemNode<XChannelList, XChannel> > &currentChannel(unsigned int lp) const {return loop(lp)->m_currentChannel;}
	const shared_ptr<XDoubleNode> &targetTemp(unsigned int lp) const {return loop(lp)->m_targetTemp;}
	const shared_ptr<XDoubleNode> &manualPower(unsigned int lp) const {return loop(lp)->m_manualPower;}
	const shared_ptr<XDoubleNode> &prop(unsigned int lp) const {return loop(lp)->m_prop;}
	const shared_ptr<XDoubleNode> &interval(unsigned int lp) const {return loop(lp)->m_int;}
	const shared_ptr<XDoubleNode> &deriv(unsigned int lp) const {return loop(lp)->m_deriv;}
	const shared_ptr<XComboNode> &heaterMode(unsigned int lp) const {return loop(lp)->m_heaterMode;}
	const shared_ptr<XComboNode> &powerRange(unsigned int lp) const {return loop(lp)->m_powerRange;}
	const shared_ptr<XDoubleNode> &heaterPower(unsigned int lp) const {return loop(lp)->m_heaterPower;}
	const shared_ptr<XDoubleNode> &sourceTemp(unsigned int lp) const {return loop(lp)->m_sourceTemp;}
	const shared_ptr<XDoubleNode> &powerMax(unsigned int lp) const {return loop(lp)->m_powerMax;}
	const shared_ptr<XDoubleNode> &powerMin(unsigned int lp) const {return loop(lp)->m_powerMin;}
	//! holds an averaged error between target temp and actual one
	const shared_ptr<XDoubleNode> &stabilized(unsigned int lp) const {return loop(lp)->m_stabilized;}
	//! PID control of an external device.
	const shared_ptr<XItemNode<XDriverList, XDCSource, XFlowControllerDriver> > &extDevice(unsigned int lp) const {return loop(lp)->m_extDevice;}
	const shared_ptr<XComboNode> &extDCSourceChannel(unsigned int lp) const {return loop(lp)->m_extDCSourceChannel;}
	const shared_ptr<XBoolNode> &extIsPositive(unsigned int lp) const {return loop(lp)->m_extIsPositive;}

protected:
	//! This function will be called when raw data are written.
	//! Implement this function to convert the raw data to the record (Payload).
	//! \sa analyze()
    virtual void analyzeRaw(RawDataReader &reader, Transaction &tr) override;
    //! This function is called after committing XPrimaryDriver::analyzeRaw() or XSecondaryDriver::analyze().
    //! This might be called even if the record is invalid (time() == false).
    virtual void visualize(const Snapshot &shot) override;
  
	//! Prepares channel names in your constructor.
	//! \param multiread if true, simultaneous reading of multi channels.
	//! \param channel_names array of pointers to channel name. ends with null pointer.
	void createChannels(Transaction &tr, const shared_ptr<XMeasure> &meas,
                        bool multiread, std::initializer_list<XString> channel_names,
                        std::initializer_list<XString> excitations,
                        std::initializer_list<XString> loop_names);
  
	//! reads sensor value from the instrument
	virtual double getRaw(shared_ptr<XChannel> &channel) = 0;
	//! reads a value in Kelvin from the instrument
	virtual double getTemp(shared_ptr<XChannel> &channel) = 0;
	//! obtains current heater power
	//! \sa m_heaterPowerUnit()
	virtual double getHeater(unsigned int loop) = 0;
	//! ex. "W", "dB", or so
	virtual const char *m_heaterPowerUnit(unsigned int loop) = 0;
    //! converts displayed interval setting to unit in second to calculate \a stabilized().
    virtual double currentIntervalSettingInSec(const Snapshot &shot, unsigned int lp) {return shot[ *interval(lp)];}
  
	bool hasExtDevice(const Snapshot &shot, unsigned int lp) const {return loop(lp)->hasExtDevice(shot);}

	virtual void onPChanged(unsigned int loop, double p) = 0;
	virtual void onIChanged(unsigned int loop, double i) = 0;
	virtual void onDChanged(unsigned int loop, double d) = 0;
	virtual void onTargetTempChanged(unsigned int loop, double temp) = 0;
	virtual void onManualPowerChanged(unsigned int loop, double pow) = 0;
	virtual void onHeaterModeChanged(unsigned int loop, int mode) = 0;
	virtual void onPowerRangeChanged(unsigned int loop, int range) = 0;
	virtual void onPowerMaxChanged(unsigned int loop, double v) = 0;
	virtual void onPowerMinChanged(unsigned int loop, double v) = 0;
	virtual void onCurrentChannelChanged(unsigned int loop, const shared_ptr<XChannel> &ch) = 0;

	virtual void onExcitationChanged(const shared_ptr<XChannel> &ch, int exc) = 0;
private:
	shared_ptr<XChannelList> m_channels;
	//! LOOPs
	class Loop : public XNode {
	public:
		Loop(const char *name, bool runtime, shared_ptr<XTempControl>, Transaction &tr,
			unsigned int idx, Transaction &tr_meas, const shared_ptr<XMeasure> &meas);

		weak_ptr<XTempControl> m_tempctrl;
		const unsigned int m_idx;
		shared_ptr<XItemNode<XChannelList, XChannel> >  m_currentChannel;
		const shared_ptr<XDoubleNode> m_targetTemp;
		const shared_ptr<XDoubleNode> m_manualPower;
		const shared_ptr<XDoubleNode> m_prop, m_int, m_deriv;
		const shared_ptr<XComboNode> m_heaterMode;
		const shared_ptr<XComboNode> m_powerRange;
		const shared_ptr<XDoubleNode> m_powerMax, m_powerMin;
		const shared_ptr<XDoubleNode> m_heaterPower, m_sourceTemp;
		//! holds an averaged error between target temp and actual one
		const shared_ptr<XDoubleNode> m_stabilized;

		const shared_ptr<XItemNode<XDriverList, XDCSource, XFlowControllerDriver> > m_extDevice;
		const shared_ptr<XComboNode> m_extDCSourceChannel;
		const shared_ptr<XBoolNode> m_extIsPositive;

		void start();
		void stop();
		void update(double temp);
		double pid(const Snapshot &shot, XTime time, double temp);

		bool hasExtDevice(const Snapshot &shot) const {
			return shared_ptr<XDCSource>(shot[ *m_extDevice]) ||
				shared_ptr<XFlowControllerDriver>(shot[ *m_extDevice]);
		}

		void onPChanged(const Snapshot &shot, XValueNodeBase *);
		void onIChanged(const Snapshot &shot, XValueNodeBase *);
		void onDChanged(const Snapshot &shot, XValueNodeBase *);
		void onTargetTempChanged(const Snapshot &shot, XValueNodeBase *);
		void onManualPowerChanged(const Snapshot &shot, XValueNodeBase *);
		void onHeaterModeChanged(const Snapshot &shot, XValueNodeBase *);
		void onPowerRangeChanged(const Snapshot &shot, XValueNodeBase *);
		void onPowerMaxChanged(const Snapshot &shot, XValueNodeBase *);
		void onPowerMinChanged(const Snapshot &shot, XValueNodeBase *);
		void onCurrentChannelChanged(const Snapshot &shot, XValueNodeBase *);
		void onExtDeviceChanged(const Snapshot &shot, XValueNodeBase *);

        std::deque<xqcon_ptr> m_conUIs;

        shared_ptr<Listener> m_lsnOnPChanged, m_lsnOnIChanged, m_lsnOnDChanged,
			m_lsnOnTargetTempChanged, m_lsnOnManualPowerChanged, m_lsnOnHeaterModeChanged,
			m_lsnOnPowerMaxChanged, m_lsnOnPowerMinChanged,
			m_lsnOnPowerRangeChanged, m_lsnOnCurrentChannelChanged,
            m_lsnOnSetupChannelChanged, m_lsnOnExtDeviceChanged;

		double m_pidAccum;
		double m_pidLastTemp;
		XTime m_pidLastTime;

		double m_tempAvg;
		double m_tempErrAvg;
		XTime m_lasttime;
	};
	std::deque<shared_ptr<Loop> > m_loops;
	shared_ptr<Loop> loop(unsigned int lp) {return m_loops.at(lp);}
	const shared_ptr<Loop> loop(unsigned int lp) const {return m_loops.at(lp);}

	shared_ptr<XItemNode<XChannelList, XChannel> > m_setupChannel;

    shared_ptr<Listener> m_lsnOnSetupChannelChanged, m_lsnOnExcitationChanged,
        m_lsnOnLoopUpdated;
    Transactional::Talker<int, XString> m_tlkOnLoopUpdated;

	void onSetupChannelChanged(const Snapshot &shot, XValueNodeBase *);
    void onExcitationChangedInternal(const Snapshot &shot, XValueNodeBase *);
    void onLoopUpdated(int index, const XString &);

    std::deque<shared_ptr<XScalarEntry> > m_entry_temps, m_entry_raws;
 
	const qshared_ptr<FrmTempControl> m_form;
	bool m_multiread;

	xqcon_ptr m_conSetupChannel,
		m_conExcitation, m_conThermometer;
	
    virtual void *execute(const atomic<bool> &) override;
  
};

#endif
