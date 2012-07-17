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
#ifndef dcsourceH
#define dcsourceH

#include "primarydriver.h"
#include "xnodeconnector.h"

class QMainWindow;
class Ui_FrmDCSource;
typedef QForm<QMainWindow, Ui_FrmDCSource> FrmDCSource;

class XDCSource: public XPrimaryDriver {
public:
	XDCSource(const char *name, bool runtime,
		Transaction &tr_meas, const shared_ptr<XMeasure> &meas);
	//! usually nothing to do
	virtual ~XDCSource() {
	}
	//! show all forms belonging to driver
	virtual void showForms();

	//! driver specific part below
	virtual void changeFunction(int ch, int x) = 0;
	virtual void changeOutput(int ch, bool x) = 0;
	virtual void changeValue(int ch, double x, bool autorange) = 0;
	virtual void changeRange(int ch, int x) = 0;
	virtual void queryStatus(Transaction &tr, int ch) = 0;
	virtual double max(int ch, bool autorange) const = 0;

	const shared_ptr<XComboNode> &function() const {
		return m_function;
	}
	const shared_ptr<XBoolNode> &output() const {
		return m_output;
	}
	const shared_ptr<XDoubleNode> &value() const {
		return m_value;
	}
	const shared_ptr<XComboNode> &channel() const {
		return m_channel;
	}
	const shared_ptr<XComboNode> &range() const {
		return m_range;
	}
protected:
	//! Starts up your threads, connects GUI, and activates signals.
	virtual void start();
	//! Shuts down your threads, unconnects GUI, and deactivates signals
	//! This function may be called even if driver has already stopped.
	virtual void stop();

	//! This function will be called when raw data are written.
	//! Implement this function to convert the raw data to the record (Payload).
	//! \sa analyze()
	virtual void analyzeRaw(RawDataReader &reader, Transaction &tr) throw (XRecordError&);
	//! This function is called after committing XPrimaryDriver::analyzeRaw() or XSecondaryDriver::analyze().
	//! This might be called even if the record is invalid (time() == false).
	virtual void visualize(const Snapshot &shot);

	void updateStatus() {
		Snapshot shot( *channel());
		onChannelChanged(shot, channel().get());
	}

private:
	xqcon_ptr m_conFunction, m_conOutput, m_conValue, m_conChannel, m_conRange;
	const shared_ptr<XComboNode> m_function;
	const shared_ptr<XBoolNode> m_output;
	const shared_ptr<XDoubleNode> m_value;
	const shared_ptr<XComboNode> m_channel;
	const shared_ptr<XComboNode> m_range;
	shared_ptr<XListener> m_lsnFunction, m_lsnOutput, m_lsnValue, m_lsnChannel,
		m_lsnRange;

	virtual void onFunctionChanged(const Snapshot &shot, XValueNodeBase *);
	virtual void onOutputChanged(const Snapshot &shot, XValueNodeBase *);
	virtual void onValueChanged(const Snapshot &shot, XValueNodeBase *);
	virtual void onChannelChanged(const Snapshot &shot, XValueNodeBase *);
	virtual void onRangeChanged(const Snapshot &shot, XValueNodeBase *);

	const qshared_ptr<FrmDCSource> m_form;

	void finish(const XTime &time_awared);
};

#endif

