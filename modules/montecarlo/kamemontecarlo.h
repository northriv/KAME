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
#ifndef KAMEMONTECARLO_H_
#define KAMEMONTECARLO_H_

#include "primarydriver.h"
#include "dummydriver.h"
#include "xwavengraph.h"
#include <fftw3.h>

class XScalarEntry;
class MonteCarlo;
class Ui_FrmMonteCarlo;
typedef QForm<QMainWindow, Ui_FrmMonteCarlo> FrmMonteCarlo;

class XMonteCarloDriver : public XDummyDriver<XPrimaryDriver> {
public:
	XMonteCarloDriver(const char *name, bool runtime,
		Transaction &tr_meas, const shared_ptr<XMeasure> &meas);
	//! usually nothing to do
	virtual ~XMonteCarloDriver();
	//! show all forms belonging to driver
	virtual void showForms();

	struct Payload : public XPrimaryDriver::Payload {
		Payload() : m_fftlen(-1) {}
	private:
		friend class XMonteCarloDriver;
		shared_ptr<MonteCarlo> m_loop, m_store;
		int m_fftlen;
		fftw_complex *m_pFFTin[3];
		fftw_complex *m_pFFTout[3];
		fftw_plan m_fftplan[3];

		long double m_sumDU, m_sumDS, m_sumDUav;
		long double m_testsTotal;
		double m_flippedTotal;
		double m_dU;
		double m_DUav, m_Mav;
		double m_lastTemp;
		//! along field direction.
		double m_lastField, m_lastMagnetization;
	};
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
private:
	shared_ptr<XDoubleNode> m_targetTemp;
	shared_ptr<XDoubleNode> m_targetField;
	shared_ptr<XDoubleNode> m_hdirx;
	shared_ptr<XDoubleNode> m_hdiry;
	shared_ptr<XDoubleNode> m_hdirz;
	shared_ptr<XUIntNode> m_L;
	shared_ptr<XDoubleNode> m_cutoffReal;
	shared_ptr<XDoubleNode> m_cutoffRec;
	shared_ptr<XDoubleNode> m_alpha;
	shared_ptr<XDoubleNode> m_minTests;
	shared_ptr<XDoubleNode> m_minFlips;
	shared_ptr<XTouchableNode> m_step;
	shared_ptr<XComboNode> m_graph3D;
	shared_ptr<XScalarEntry> m_entryT, m_entryH,
		m_entryU, m_entryC, m_entryCoT,
		m_entryS, m_entryM, m_entry2in2, m_entry1in3;
  
	xqcon_ptr m_conLength, m_conCutoffReal, m_conCutoffRec, m_conAlpha,
		m_conTargetTemp, m_conTargetField,
		m_conHDirX, m_conHDirY, m_conHDirZ, m_conMinTests, m_conMinFlips, m_conStep,
		m_conGraph3D;
	qshared_ptr<FrmMonteCarlo> m_form;
	shared_ptr<XWaveNGraph> m_wave3D;
	void execute(int flips, long double tests);
	void onTargetChanged(const Snapshot &shot, XValueNodeBase *);
	void onGraphChanged(const Snapshot &shot, XValueNodeBase *);
	void onStepTouched(const Snapshot &shot, XTouchableNode *);
	shared_ptr<XListener> m_lsnTargetChanged, m_lsnStepTouched, m_lsnGraphChanged;
	shared_ptr<XStatusPrinter> m_statusPrinter;
};

#endif /*KAMEMONTECARLO_H_*/
