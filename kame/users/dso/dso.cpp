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
#include <qpushbutton.h>
#include <qcheckbox.h>
#include <knuminput.h>
#include "forms/dsoform.h"
#include "dso.h"
#include "graph.h"
#include "graphwidget.h"
#include "xwavengraph.h"
#include "fir.h"

#include "interface.h"
#include "analyzer.h"
#include "xnodeconnector.h"
#include <qstatusbar.h>
#include <klocale.h>
#include <kiconloader.h>
#include <kapplication.h>

const char *XDSO::s_trace_names[] = {
	"Time [sec]", "Trace1 [V]", "Trace2 [V]"
};
    
XDSO::XDSO(const char *name, bool runtime,
		   const shared_ptr<XScalarEntryList> &scalarentries,
		   const shared_ptr<XInterfaceList> &interfaces,
		   const shared_ptr<XThermometerList> &thermometers,
		   const shared_ptr<XDriverList> &drivers) :
	XPrimaryDriver(name, runtime, scalarentries, interfaces, thermometers, drivers),
	m_average(create<XUIntNode>("Average", false)),
	m_singleSequence(create<XBoolNode>("SingleSequence", false)),
	m_trigSource(create<XComboNode>("TrigSource", false)),
	m_trigFalling(create<XBoolNode>("TrigFalling", false)),
	m_trigPos(create<XDoubleNode>("TrigPos", false)),
	m_trigLevel(create<XDoubleNode>("TrigLevel", false)),
	m_timeWidth(create<XDoubleNode>("TimeWidth", false)),
	m_vFullScale1(create<XComboNode>("VFullScale1", false, true)),
	m_vFullScale2(create<XComboNode>("VFullScale2", false, true)),
	m_vOffset1(create<XDoubleNode>("VOffset1", false)),
	m_vOffset2(create<XDoubleNode>("VOffset2", false)),
	m_recordLength(create<XUIntNode>("RecordLength", false)),
	m_forceTrigger(create<XNode>("ForceTrigger", true)),  
	m_trace1(create<XComboNode>("Trace1", false)),
	m_trace2(create<XComboNode>("Trace2", false)),
	m_fetchMode(create<XComboNode>("FetchMode", false, true)),
	m_firEnabled(create<XBoolNode>("FIREnabled", false)),
	m_firBandWidth(create<XDoubleNode>("FIRBandWidth", false)),
	m_firCenterFreq(create<XDoubleNode>("FIRCenterFreq", false)),
	m_firSharpness(create<XDoubleNode>("FIRSharpness", false)),
	m_form(new FrmDSO(g_pFrmMain)),
	m_waveForm(create<XWaveNGraph>("WaveForm", false, 
								   m_form->m_graphwidget, m_form->m_urlDump, m_form->m_btnDump)),
	m_numChannelsDisp(0),
	m_rawDisplayOnly(false),
	m_conAverage(xqcon_create<XQLineEditConnector>(m_average, m_form->m_edAverage)),
	m_conSingle(xqcon_create<XQToggleButtonConnector>(m_singleSequence, m_form->m_ckbSingleSeq)),
	m_conTrace1(xqcon_create<XQComboBoxConnector>(m_trace1, m_form->m_cmbTrace1)),
	m_conTrace2(xqcon_create<XQComboBoxConnector>(m_trace2, m_form->m_cmbTrace2)),
	m_conTrigSource(xqcon_create<XQComboBoxConnector>(m_trigSource, m_form->m_cmbTrigSource)),
	m_conTrigPos(xqcon_create<XKDoubleNumInputConnector>(m_trigPos, m_form->m_numTrigPos)),
	m_conTrigLevel(xqcon_create<XQLineEditConnector>(m_trigLevel, m_form->m_edTrigLevel)),
	m_conTrigFalling(xqcon_create<XQToggleButtonConnector>(m_trigFalling, m_form->m_ckbTrigFalling)),
	m_conFetchMode(xqcon_create<XQComboBoxConnector>(m_fetchMode, m_form->m_cmbFetchMode)),
	m_conTimeWidth(xqcon_create<XQLineEditConnector>(m_timeWidth, m_form->m_edTimeWidth)),
	m_conVFullScale1(xqcon_create<XQComboBoxConnector>(m_vFullScale1, m_form->m_cmbVFS1)),
	m_conVFullScale2(xqcon_create<XQComboBoxConnector>(m_vFullScale2, m_form->m_cmbVFS2)),
	m_conVOffset1(xqcon_create<XQLineEditConnector>(m_vOffset1, m_form->m_edVOffset1)),
	m_conVOffset2(xqcon_create<XQLineEditConnector>(m_vOffset2, m_form->m_edVOffset2)),
	m_conForceTrigger(xqcon_create<XQButtonConnector>(m_forceTrigger, m_form->m_btnForceTrigger)),
	m_conRecordLength(xqcon_create<XQLineEditConnector>(m_recordLength, m_form->m_edRecordLength)),
	m_conFIREnabled(xqcon_create<XQToggleButtonConnector>(m_firEnabled, m_form->m_ckbFIREnabled)),
	m_conFIRBandWidth(xqcon_create<XQLineEditConnector>(m_firBandWidth, m_form->m_edFIRBandWidth)),
	m_conFIRSharpness(xqcon_create<XQLineEditConnector>(m_firSharpness, m_form->m_edFIRSharpness)),
	m_conFIRCenterFreq(xqcon_create<XQLineEditConnector>(m_firCenterFreq, m_form->m_edFIRCenterFreq)),
	m_statusPrinter(XStatusPrinter::create(m_form.get()))
{
	m_form->m_btnForceTrigger->setIconSet(
		KApplication::kApplication()->iconLoader()->loadIconSet("apply", 
																KIcon::Toolbar, KIcon::SizeSmall, true ) );  
	m_form->m_numTrigPos->setRange(0.0, 100.0, 1.0, true);
    
	singleSequence()->value(true);
	firBandWidth()->value(1000.0);
	firCenterFreq()->value(.0);
	firSharpness()->value(4.5);
  
	m_lsnOnCondChanged = firEnabled()->onValueChanged().connectWeak(
		shared_from_this(), &XDSO::onCondChanged);
	firBandWidth()->onValueChanged().connect(m_lsnOnCondChanged);
	firCenterFreq()->onValueChanged().connect(m_lsnOnCondChanged);
	firSharpness()->onValueChanged().connect(m_lsnOnCondChanged);
  
	{
		const char *modes[] = {"Never", "Averaging", "Sequence", 0L};
		for(const char **mode = &modes[0]; *mode; mode++)
			fetchMode()->add(*mode);
	}
	fetchMode()->value(FETCHMODE_SEQ);
  
	average()->setUIEnabled(false);
	singleSequence()->setUIEnabled(false);
//  fetchMode()->setUIEnabled(false);
	timeWidth()->setUIEnabled(false);
	trigSource()->setUIEnabled(false);
	trigPos()->setUIEnabled(false);
	trigLevel()->setUIEnabled(false);
	trigFalling()->setUIEnabled(false);
	vFullScale1()->setUIEnabled(false);
	vFullScale2()->setUIEnabled(false);
	vOffset1()->setUIEnabled(false);
	vOffset2()->setUIEnabled(false);
	forceTrigger()->setUIEnabled(false);
	recordLength()->setUIEnabled(false);
  
	m_waveForm->setColCount(2, s_trace_names); 
	m_waveForm->selectAxes(0, 1, -1);
	m_waveForm->graph()->persistence()->value(0);
	m_waveForm->clear();
}
void
XDSO::showForms() {
//! impliment form->show() here
    m_form->show();
    m_form->raise();
}

void
XDSO::start()
{
	m_thread.reset(new XThread<XDSO>(shared_from_this(), &XDSO::execute));
	m_thread->resume();
  
//  trace1()->setUIEnabled(false);
//  trace2()->setUIEnabled(false);
  
	average()->setUIEnabled(true);
	singleSequence()->setUIEnabled(true);
	timeWidth()->setUIEnabled(true);
	trigSource()->setUIEnabled(true);
	trigPos()->setUIEnabled(true);
	trigLevel()->setUIEnabled(true);
	trigFalling()->setUIEnabled(true);
	vFullScale1()->setUIEnabled(true);
	vFullScale2()->setUIEnabled(true);
	vOffset1()->setUIEnabled(true);
	vOffset2()->setUIEnabled(true);
	forceTrigger()->setUIEnabled(true);
	recordLength()->setUIEnabled(true);
}
void
XDSO::stop()
{   
//  trace1()->setUIEnabled(true);
//  trace2()->setUIEnabled(true);
  
	average()->setUIEnabled(false);
	singleSequence()->setUIEnabled(false);
	timeWidth()->setUIEnabled(false);
	trigSource()->setUIEnabled(false);
	trigPos()->setUIEnabled(false);
	trigLevel()->setUIEnabled(false);
	trigFalling()->setUIEnabled(false);
	vFullScale1()->setUIEnabled(false);
	vFullScale2()->setUIEnabled(false);
	vOffset1()->setUIEnabled(false);
	vOffset2()->setUIEnabled(false);
	forceTrigger()->setUIEnabled(false);
	recordLength()->setUIEnabled(false);  
  	
	if(m_thread) m_thread->terminate();
//    m_thread->waitFor();
//  thread must do interface()->close() at the end

	m_numChannelsDisp = 0;
}
unsigned int
XDSO::lengthRecorded() const
{
    return m_wavesRecorded.size() / numChannelsRecorded();
}
const double *
XDSO::waveRecorded(unsigned int ch) const
{
    return &m_wavesRecorded[lengthRecorded() * ch];
}
unsigned int
XDSO::lengthDisp() const
{
    return m_wavesDisp.size() / numChannelsDisp();
}
double *
XDSO::waveDisp(unsigned int ch)
{
    return &m_wavesDisp[lengthDisp() * ch];
}

void
XDSO::visualize()
{
	XScopedLock<XRecursiveMutex> lock(m_dispMutex);
  
	m_statusPrinter->clear();
  
//  if(!time()) {
//  	m_waveForm->clear();
//  	return;
//  }
	const unsigned int num_channels = numChannelsDisp();
	if(!num_channels) {
		m_waveForm->clear();
		return;
	}
	const unsigned int length = lengthDisp();
	{ XScopedWriteLock<XWaveNGraph> lock(*m_waveForm);
	m_waveForm->setColCount(num_channels + 1, s_trace_names);
	if((m_waveForm->colX() != 0) || (m_waveForm->colY1() != 1) ||
	   (m_waveForm->colY2() != ((num_channels > 1) ? 2 : -1))) {
		m_waveForm->selectAxes(0, 1, (num_channels > 1) ? 2 : -1);
	}
	m_waveForm->plot1()->drawPoints()->value(false);
	if(num_channels > 1)
		m_waveForm->plot2()->drawPoints()->value(false);
          
	m_waveForm->setRowCount(length);
    
	double *times = m_waveForm->cols(0);
	double tint = timeIntervalDisp();
	double t = -trigPosDisp() * tint;
	for(unsigned int i = 0; i < length; i++) {
		*times++ = t;
		t += tint;
	}
        
	for(unsigned int i = 0; i < num_channels; i++) {
		memcpy(m_waveForm->cols(i + 1), waveDisp(i), length * sizeof(double));
	}
	}
}

void *
XDSO::execute(const atomic<bool> &terminated)
{
	XTime time_awared = XTime::now();
	int last_count = 0;
  
	m_lsnOnAverageChanged = average()->onValueChanged().connectWeak(
		shared_from_this(), &XDSO::onAverageChanged);
	m_lsnOnSingleChanged = singleSequence()->onValueChanged().connectWeak(
		shared_from_this(), &XDSO::onSingleChanged);
	m_lsnOnTimeWidthChanged = timeWidth()->onValueChanged().connectWeak(
		shared_from_this(), &XDSO::onTimeWidthChanged);
	m_lsnOnTrigSourceChanged = trigSource()->onValueChanged().connectWeak(
		shared_from_this(), &XDSO::onTrigSourceChanged);
	m_lsnOnTrigPosChanged = trigPos()->onValueChanged().connectWeak(
		shared_from_this(), &XDSO::onTrigPosChanged);
	m_lsnOnTrigLevelChanged = trigLevel()->onValueChanged().connectWeak(
		shared_from_this(), &XDSO::onTrigLevelChanged);
	m_lsnOnTrigFallingChanged = trigFalling()->onValueChanged().connectWeak(
		shared_from_this(), &XDSO::onTrigFallingChanged);
	m_lsnOnTrace1Changed = trace1()->onValueChanged().connectWeak(
		shared_from_this(), &XDSO::onTrace1Changed);
	m_lsnOnTrace2Changed = trace2()->onValueChanged().connectWeak(
		shared_from_this(), &XDSO::onTrace2Changed);
	m_lsnOnVFullScale1Changed = vFullScale1()->onValueChanged().connectWeak(
		shared_from_this(), &XDSO::onVFullScale1Changed);
	m_lsnOnVFullScale2Changed = vFullScale2()->onValueChanged().connectWeak(
		shared_from_this(), &XDSO::onVFullScale2Changed);
	m_lsnOnVOffset1Changed = vOffset1()->onValueChanged().connectWeak(
		shared_from_this(), &XDSO::onVOffset1Changed);
	m_lsnOnVOffset2Changed = vOffset2()->onValueChanged().connectWeak(
		shared_from_this(), &XDSO::onVOffset2Changed);
	m_lsnOnForceTriggerTouched = forceTrigger()->onTouch().connectWeak(
		shared_from_this(), &XDSO::onForceTriggerTouched);
	m_lsnOnRecordLengthChanged = recordLength()->onValueChanged().connectWeak(
		shared_from_this(), &XDSO::onRecordLengthChanged);

	while(!terminated) {
		const int fetch_mode = *fetchMode();
		if(!fetch_mode || (fetch_mode == FETCHMODE_NEVER)) {
			msecsleep(50);
			continue;
		}
		bool seq_busy = false;
		try {
			int count = acqCount(&seq_busy);
			if(!count) {
				time_awared = XTime::now();
				last_count = 0;
				msecsleep(10);
				continue;
			}
			if(count == last_count) {
				msecsleep(10);
				continue;
			}
			if(fetch_mode == FETCHMODE_SEQ) {
				if(*singleSequence() && seq_busy) {
					msecsleep(10);
					continue;
				}
			}
			last_count =  count;
		}
		catch (XKameError& e) {
			e.print(getLabel());
			continue;
		}

		std::deque<std::string> channels;
		channels.push_back(trace1()->to_str());
		if(channels.front().empty()) {
            statusPrinter()->printMessage(getLabel() + " " + KAME::i18n("Select traces!."));
            continue;
		}
		channels.push_back(trace2()->to_str());
		if(channels.back().empty()) {
            channels.pop_back();
		}
      
		clearRaw();
		// try/catch exception of communication errors
		try {
			getWave(channels);
		}
		catch (XDriver::XSkippedRecordError&) {
			continue;
		}
		catch (XKameError &e) {
			e.print(getLabel());
			continue;
		}
      
		m_rawDisplayOnly = (*singleSequence() && seq_busy);

		finishWritingRaw(time_awared, XTime::now());
	      
		if(*singleSequence() && !seq_busy) {
			// try/catch exception of communication errors
			try {
				startSequence();
				time_awared = XTime::now();
			}
			catch (XKameError &e) {
				e.print(getLabel());
				continue;
			}
		}
    }
    m_rawDisplayOnly = false;

	m_lsnOnAverageChanged.reset();
	m_lsnOnSingleChanged.reset();
	m_lsnOnTimeWidthChanged.reset();
	m_lsnOnTrigSourceChanged.reset();
	m_lsnOnTrigPosChanged.reset();
	m_lsnOnTrigLevelChanged.reset();
	m_lsnOnTrigFallingChanged.reset();
	m_lsnOnVFullScale1Changed.reset();
	m_lsnOnVFullScale2Changed.reset();
	m_lsnOnTrace1Changed.reset();
	m_lsnOnTrace2Changed.reset();
	m_lsnOnVOffset1Changed.reset();
	m_lsnOnVOffset2Changed.reset();
	m_lsnOnForceTriggerTouched.reset();
	m_lsnOnRecordLengthChanged.reset();
                            
	afterStop();

	return NULL;
}

void
XDSO::onCondChanged(const shared_ptr<XValueNodeBase> &)
{
//  readLockRecord();
	visualize();
//  readUnlockRecord();
}
void
XDSO::setParameters(unsigned int channels, double startpos, double interval, unsigned int length)
{
	m_numChannelsDisp = channels;
	m_wavesDisp.resize(channels * length, 0.0);
	m_trigPosDisp = -startpos / interval;
	m_timeIntervalDisp = interval;
}
void
XDSO::convertRawToDisp() throw (XRecordError&) {
    convertRaw();
    
	if(*firEnabled()) {
		double  bandwidth = *firBandWidth()*1000.0*timeIntervalDisp();
		double fir_sharpness = *firSharpness();
		if(fir_sharpness < 4.0)
			m_statusPrinter->printWarning(KAME::i18n("Too small number of taps for FIR filter."));
		int taps = std::min((int)lrint(2 * fir_sharpness / bandwidth), 5000);
		m_fir.setupBPF(taps, bandwidth, *firCenterFreq() * 1000.0 * timeIntervalDisp());
		unsigned int num_channels = numChannelsDisp();
		unsigned int length = lengthDisp();
		std::vector<double> buf(length);
		for(unsigned int i = 0; i < num_channels; i++) {
			m_fir.doFIR(waveDisp(i), 
						&buf[0], length);
			memcpy(waveDisp(i), &buf[0], length * sizeof(double));
		}
	}
}
void
XDSO::analyzeRaw() throw (XRecordError&) {
	XScopedLock<XRecursiveMutex> lock(m_dispMutex);

    convertRawToDisp();

	if(m_rawDisplayOnly) {
		visualize();
		throw XSkippedRecordError(__FILE__, __LINE__);
	}
//    std::fill(m_wavesRecorded.begin(), m_wavesRecorded.end(), 0.0);
	m_numChannelsRecorded = m_numChannelsDisp;
	m_wavesRecorded.resize(m_wavesDisp.size());
	m_trigPosRecorded = m_trigPosDisp;
	m_timeIntervalRecorded = m_timeIntervalDisp;
	memcpy(&m_wavesRecorded[0], &m_wavesDisp[0], m_wavesDisp.size() * sizeof(double));
}
