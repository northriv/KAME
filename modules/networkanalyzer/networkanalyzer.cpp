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
#include "networkanalyzer.h"
#include "forms/networkanalyzerform.h"
#include "graph.h"
#include "graphwidget.h"
#include "xwavengraph.h"

#include "interface.h"
#include "analyzer.h"
#include "xnodeconnector.h"
#include <klocale.h>
#include <kapplication.h>

XNetworkAnalyzer::XNetworkAnalyzer(const char *name, bool runtime,
		   const shared_ptr<XScalarEntryList> &scalarentries,
		   const shared_ptr<XInterfaceList> &interfaces,
		   const shared_ptr<XThermometerList> &thermometers,
		   const shared_ptr<XDriverList> &drivers) :
	XPrimaryDriver(name, runtime, scalarentries, interfaces, thermometers, drivers),
    m_marker1X(create<XScalarEntry>("Marker1X", false, 
								  dynamic_pointer_cast<XDriver>(shared_from_this()))),
    m_marker1Y(create<XScalarEntry>("Marker1Y", false, 
								  dynamic_pointer_cast<XDriver>(shared_from_this()))),
    m_marker2X(create<XScalarEntry>("Marker2X", false, 
								  dynamic_pointer_cast<XDriver>(shared_from_this()))),
    m_marker2Y(create<XScalarEntry>("Marker2Y", false, 
								  dynamic_pointer_cast<XDriver>(shared_from_this()))),
	m_startFreq(create<XDoubleNode>("StartFreq", false)),
	m_stopFreq(create<XDoubleNode>("StopFreq", false)),
	m_form(new FrmNetworkAnalyzer(g_pFrmMain)),
	m_waveForm(create<XWaveNGraph>("WaveForm", false, 
								   m_form->m_graphwidget, m_form->m_urlDump, m_form->m_btnDump))
{
	scalarentries->insert(m_marker1X);
	scalarentries->insert(m_marker1Y);
	scalarentries->insert(m_marker2X);
	scalarentries->insert(m_marker2Y);
	
	startFreq()->setUIEnabled(false);
	stopFreq()->setUIEnabled(false);
	
	const char *labels[] = {"Freq [MHz]", "Level [dBm]"};
	m_waveForm->setColCount(2, labels); 
}
void
XNetworkAnalyzer::showForms() {
//! impliment form->show() here
    m_form->show();
    m_form->raise();
}

void
XNetworkAnalyzer::start()
{
	m_thread.reset(new XThread<XNetworkAnalyzer>(shared_from_this(), &XNetworkAnalyzer::execute));
	m_thread->resume();
  
	startFreq()->setUIEnabled(true);
	stopFreq()->setUIEnabled(true);
}
void
XNetworkAnalyzer::stop()
{   
	startFreq()->setUIEnabled(false);
	stopFreq()->setUIEnabled(false);
  	
	if(m_thread) m_thread->terminate();
//    m_thread->waitFor();
//  thread must do interface()->close() at the end
}

void
XNetworkAnalyzer::analyzeRaw() throw (XRecordError&)
{
	unsigned int numtr = pop<unsigned int>();
	if(numtr != 1)
		return; 
	unsigned int ch = pop<unsigned int>();
	if(ch != 2)
		return; 
    m_marker1X->value(pop<double>());
    m_marker1Y->value(pop<double>());
    m_marker2X->value(pop<double>());
    m_marker2Y->value(pop<double>());

    convertRaw();
}
void
XNetworkAnalyzer::visualize()
{
//  if(!time()) {
//  	m_waveForm->clear();
//  	return;
//  }
	const unsigned int length = lengthRecorded();
	{ XScopedWriteLock<XWaveNGraph> lock(*m_waveForm);
	m_waveForm->setRowCount(length);
    
	double *freqs = m_waveForm->cols(0);
	double fint = freqIntervalRecorded();
	double f = startFreqRecorded() * fint;
	for(unsigned int i = 0; i < length; i++) {
		*freqs++ = f;
		f += fint;
	}
        
	memcpy(m_waveForm->cols(1), traceRecorded(), length * sizeof(double));
	}
}

void *
XNetworkAnalyzer::execute(const atomic<bool> &terminated)
{
	m_lsnOnStartFreqChanged = startFreq()->onValueChanged().connectWeak(
		shared_from_this(), &XNetworkAnalyzer::onStartFreqChanged);
	m_lsnOnStopFreqChanged = stopFreq()->onValueChanged().connectWeak(
		shared_from_this(), &XNetworkAnalyzer::onStopFreqChanged);

	while(!terminated) {
		double mx[2], my[2];
		mx[0] = 0;
		my[0] = 0;
		mx[1] = 0;
		my[1] = 0;
		XTime time_awared = XTime::now();
		clearRaw();
		// try/catch exception of communication errors
		try {
			oneSweep();
			push((unsigned int)1); //# of traces.
			getMarkerPos(0, mx[0], my[0]);
			getMarkerPos(1, mx[1], my[1]);
			push((unsigned int)2); //# of markers.
			push(mx[0]);
			push(my[0]);
			push(mx[1]);
			push(my[1]);
			acquireTrace(0);
		}
		catch (XDriver::XSkippedRecordError&) {
			continue;
		}
		catch (XKameError &e) {
			e.print(getLabel());
			continue;
		}
		finishWritingRaw(time_awared, XTime::now());
    }
	m_lsnOnStartFreqChanged.reset();
	m_lsnOnStopFreqChanged.reset();
	return NULL;
}
