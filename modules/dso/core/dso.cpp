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
#include <kiconloader.h>
#include <knuminput.h>
#include "dso.h"
#include "graph.h"
#include "graphwidget.h"
#include "xwavengraph.h"
#include "fir.h"

#include "interface.h"
#include "analyzer.h"
#include "xnodeconnector.h"
#include "signalgenerator.h"

#include "ui_dsoform.h"

const char *XDSO::s_trace_names[] = {
	"Time [sec]", "Trace1 [V]", "Trace2 [V]", "Trace3 [V]", "Trace4 [V]"
};
const unsigned int XDSO::s_trace_colors[] = {
	clRed, clGreen, clLime, clAqua
};
    
XDSO::XDSO(const char *name, bool runtime,
	Transaction &tr_meas, const shared_ptr<XMeasure> &meas) :
	XPrimaryDriverWithThread(name, runtime, ref(tr_meas), meas),
	m_average(create<XUIntNode>("Average", false)),
	m_singleSequence(create<XBoolNode>("SingleSequence", false)),
	m_trigSource(create<XComboNode>("TrigSource", false)),
	m_trigFalling(create<XBoolNode>("TrigFalling", false)),
	m_trigPos(create<XDoubleNode>("TrigPos", false)),
	m_trigLevel(create<XDoubleNode>("TrigLevel", false)),
	m_timeWidth(create<XDoubleNode>("TimeWidth", false)),
	m_vFullScale1(create<XComboNode>("VFullScale1", false, true)),
	m_vFullScale2(create<XComboNode>("VFullScale2", false, true)),
	m_vFullScale3(create<XComboNode>("VFullScale3", false, true)),
	m_vFullScale4(create<XComboNode>("VFullScale4", false, true)),
	m_vOffset1(create<XDoubleNode>("VOffset1", false)),
	m_vOffset2(create<XDoubleNode>("VOffset2", false)),
	m_vOffset3(create<XDoubleNode>("VOffset3", false)),
	m_vOffset4(create<XDoubleNode>("VOffset4", false)),
	m_recordLength(create<XUIntNode>("RecordLength", false)),
	m_forceTrigger(create<XTouchableNode>("ForceTrigger", true)),
	m_restart(create<XTouchableNode>("Restart", true)),
	m_trace1(create<XComboNode>("Trace1", false)),
	m_trace2(create<XComboNode>("Trace2", false)),
	m_trace3(create<XComboNode>("Trace3", false)),
	m_trace4(create<XComboNode>("Trace4", false)),
	m_fetchMode(create<XComboNode>("FetchMode", false, true)),
	m_firEnabled(create<XBoolNode>("FIREnabled", false)),
	m_firBandWidth(create<XDoubleNode>("FIRBandWidth", false)),
	m_firCenterFreq(create<XDoubleNode>("FIRCenterFreq", false)),
	m_firSharpness(create<XDoubleNode>("FIRSharpness", false)),
	m_dRFMode(create<XComboNode>("RFMode", false)),
	m_dRFSG(create<XItemNode<XDriverList, XSG> >("RFSG", false, ref(tr_meas), meas->drivers(), true)),
	m_dRFFreq(create<XDoubleNode>("RFFreq", false)),
	m_form(new FrmDSO(g_pFrmMain)),
	m_waveForm(create<XWaveNGraph>("WaveForm", false, 
								   m_form->m_graphwidget, m_form->m_urlDump, m_form->m_btnDump)),
	m_conAverage(xqcon_create<XQLineEditConnector>(m_average, m_form->m_edAverage)),
	m_conSingle(xqcon_create<XQToggleButtonConnector>(m_singleSequence, m_form->m_ckbSingleSeq)),
	m_conTrace1(xqcon_create<XQComboBoxConnector>(m_trace1, m_form->m_cmbTrace1, Snapshot( *m_trace1))),
	m_conTrace2(xqcon_create<XQComboBoxConnector>(m_trace2, m_form->m_cmbTrace2, Snapshot( *m_trace2))),
	m_conTrace3(xqcon_create<XQComboBoxConnector>(m_trace3, m_form->m_cmbTrace3, Snapshot( *m_trace3))),
	m_conTrace4(xqcon_create<XQComboBoxConnector>(m_trace4, m_form->m_cmbTrace4, Snapshot( *m_trace4))),
	m_conFetchMode(xqcon_create<XQComboBoxConnector>(m_fetchMode, m_form->m_cmbFetchMode, Snapshot( *m_fetchMode))),
	m_conTimeWidth(xqcon_create<XQLineEditConnector>(m_timeWidth, m_form->m_edTimeWidth)),
	m_conVFullScale1(xqcon_create<XQComboBoxConnector>(m_vFullScale1, m_form->m_cmbVFS1, Snapshot( *m_vFullScale1))),
	m_conVFullScale2(xqcon_create<XQComboBoxConnector>(m_vFullScale2, m_form->m_cmbVFS2, Snapshot( *m_vFullScale2))),
	m_conVFullScale3(xqcon_create<XQComboBoxConnector>(m_vFullScale3, m_form->m_cmbVFS3, Snapshot( *m_vFullScale3))),
	m_conVFullScale4(xqcon_create<XQComboBoxConnector>(m_vFullScale4, m_form->m_cmbVFS4, Snapshot( *m_vFullScale4))),
	m_conTrigSource(xqcon_create<XQComboBoxConnector>(m_trigSource, m_form->m_cmbTrigSource, Snapshot( *m_trigSource))),
	m_conTrigPos(xqcon_create<XQDoubleSpinBoxConnector>(m_trigPos, m_form->m_dblTrigPos, m_form->m_slTrigPos)),
	m_conTrigLevel(xqcon_create<XQLineEditConnector>(m_trigLevel, m_form->m_edTrigLevel)),
	m_conTrigFalling(xqcon_create<XQToggleButtonConnector>(m_trigFalling, m_form->m_ckbTrigFalling)),
	m_conVOffset1(xqcon_create<XQLineEditConnector>(m_vOffset1, m_form->m_edVOffset1)),
	m_conVOffset2(xqcon_create<XQLineEditConnector>(m_vOffset2, m_form->m_edVOffset2)),
	m_conVOffset3(xqcon_create<XQLineEditConnector>(m_vOffset3, m_form->m_edVOffset3)),
	m_conVOffset4(xqcon_create<XQLineEditConnector>(m_vOffset4, m_form->m_edVOffset4)),
	m_conForceTrigger(xqcon_create<XQButtonConnector>(m_forceTrigger, m_form->m_btnForceTrigger)),
	m_conRecordLength(xqcon_create<XQLineEditConnector>(m_recordLength, m_form->m_edRecordLength)),
	m_conFIREnabled(xqcon_create<XQToggleButtonConnector>(m_firEnabled, m_form->m_ckbFIREnabled)),
	m_conFIRBandWidth(xqcon_create<XQLineEditConnector>(m_firBandWidth, m_form->m_edFIRBandWidth)),
	m_conFIRSharpness(xqcon_create<XQLineEditConnector>(m_firSharpness, m_form->m_edFIRSharpness)),
	m_conFIRCenterFreq(xqcon_create<XQLineEditConnector>(m_firCenterFreq, m_form->m_edFIRCenterFreq)),
	m_conDRFMode(xqcon_create<XQComboBoxConnector>(m_dRFMode, m_form->m_cmbRFMode, Snapshot( *m_dRFMode))),
	m_conDRFSG(xqcon_create<XQComboBoxConnector>(m_dRFSG, m_form->m_cmbRFSG, ref(tr_meas))),
	m_conDRFFreq(xqcon_create<XQLineEditConnector>(m_dRFFreq, m_form->m_edRFFreq)),
	m_statusPrinter(XStatusPrinter::create(m_form.get())) {
	m_form->m_btnForceTrigger->setIcon(
		KIconLoader::global()->loadIcon("quickopen",
																KIconLoader::Toolbar, KIconLoader::SizeSmall, true ) );
	m_form->m_dblTrigPos->setRange(0.0, 100.0);
	m_form->m_dblTrigPos->setSingleStep(1.0);
	m_form->tabifyDockWidget(m_form->m_dockTrace1, m_form->m_dockTrace2);
	m_form->tabifyDockWidget(m_form->m_dockTrace2, m_form->m_dockTrace3);
	m_form->tabifyDockWidget(m_form->m_dockTrace3, m_form->m_dockTrace4);
	m_form->m_dockTrace1->show();
	m_form->m_dockTrace1->raise();
	m_form->tabifyDockWidget(m_form->m_dockTrigger, m_form->m_dockRF);
	m_form->m_dockTrigger->show();
	m_form->m_dockTrigger->raise();
	m_form->resize( QSize(m_form->width(), 400) );

	for(Transaction tr( *this);; ++tr) {
		tr[ *singleSequence()] = true;
		tr[ *firBandWidth()] = 1000.0;
		tr[ *firCenterFreq()] = .0;
		tr[ *firSharpness()] = 4.5;

		m_lsnOnCondChanged = tr[ *firEnabled()].onValueChanged().connectWeakly(
			shared_from_this(), &XDSO::onCondChanged);
		tr[ *firBandWidth()].onValueChanged().connect(m_lsnOnCondChanged);
		tr[ *firCenterFreq()].onValueChanged().connect(m_lsnOnCondChanged);
		tr[ *firSharpness()].onValueChanged().connect(m_lsnOnCondChanged);
		{
			const char *modes[] = {"Never", "Averaging", "Sequence", 0L};
			for(const char **mode = &modes[0]; *mode; mode++)
				tr[ *fetchMode()].add(*mode);
		}
		tr[ *fetchMode()] = FETCHMODE_SEQ;

		 tr[ *dRFMode()].add("OFF");
		 tr[ *dRFMode()].add("By Given Freq.");
		 tr[ *dRFMode()].add("By SG Freq.");
		 tr[ *dRFMode()].add("With Coherent SG");

		m_lsnOnDRFCondChanged = tr[ *dRFMode()].onValueChanged().connectWeakly(
			shared_from_this(), &XDSO::onDRFCondChanged);
		 tr[ *dRFSG()].onValueChanged().connect(m_lsnOnDRFCondChanged);
		 tr[ *dRFFreq()].onValueChanged().connect(m_lsnOnDRFCondChanged);

		if(tr.commit())
			break;
	}
  
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
	vFullScale3()->setUIEnabled(false);
	vFullScale4()->setUIEnabled(false);
	vOffset1()->setUIEnabled(false);
	vOffset2()->setUIEnabled(false);
	vOffset3()->setUIEnabled(false);
	vOffset4()->setUIEnabled(false);
	forceTrigger()->setUIEnabled(false);
	recordLength()->setUIEnabled(false);
//	dRFMode()->setUIEnabled(false);
//	dRFFreq()->setUIEnabled(false);
//	dRFSG()->setUIEnabled(false);

	for(Transaction tr( *m_waveForm);; ++tr) {
		tr[ *m_waveForm].setColCount(4, s_trace_names);
		tr[ *m_waveForm->graph()->persistence()] = 0;
		tr[ *m_waveForm].clearPoints();
		if(tr.commit())
			break;
	}
}
void
XDSO::showForms() {
//! impliment form->show() here
    m_form->show();
    m_form->raise();
}

unsigned int
XDSO::Payload::length() const {
    return m_waves.size() / numChannels();
}
const double *
XDSO::Payload::wave(unsigned int ch) const {
    return &m_waves[length() * ch];
}
unsigned int
XDSO::Payload::lengthDisp() const {
    return m_wavesDisp.size() / numChannelsDisp();
}
double *
XDSO::Payload::waveDisp(unsigned int ch) {
    return &m_wavesDisp[lengthDisp() * ch];
}
const double *
XDSO::Payload::waveDisp(unsigned int ch) const {
    return &m_wavesDisp[lengthDisp() * ch];
}
void
XDSO::visualize(const Snapshot &shot) {
	m_statusPrinter->clear();
  
//  if(!time()) {
//  	m_waveForm->clear();
//  	return;
//  }
	const unsigned int num_channels = shot[ *this].numChannelsDisp();
	if( !num_channels) {
		for(Transaction tr( *this);; ++tr) {
			tr[ *m_waveForm].clearPoints();
			if(tr.commit())
				break;
		}
		return;
	}
	const unsigned int length = shot[ *this].lengthDisp();
	for(Transaction tr( *m_waveForm);; ++tr) {
		tr[ *m_waveForm].setColCount(num_channels + 1, s_trace_names);
		if(tr[ *m_waveForm].numPlots() != num_channels) {
			tr[ *m_waveForm].clearPlots();
			for(unsigned int i = 0; i < num_channels; i++) {
				tr[ *m_waveForm].insertPlot(s_trace_names[i + 1], 0, i + 1);
			}
			tr[ *tr[ *m_waveForm].axisy()->label()] = i18n("Traces [V]");
		}
		for(unsigned int i = 0; i < num_channels; i++) {
			tr[ *tr[ *m_waveForm].plot(i)->drawPoints()] = false;
			tr[ *tr[ *m_waveForm].plot(i)->lineColor()] = s_trace_colors[i];
			tr[ *tr[ *m_waveForm].plot(i)->pointColor()] = s_trace_colors[i];
			tr[ *tr[ *m_waveForm].plot(i)->barColor()] = s_trace_colors[i];
		}

		tr[ *m_waveForm].setRowCount(length);

		double *times = tr[ *m_waveForm].cols(0);
		double tint = shot[ *this].timeIntervalDisp();
		double t = -shot[ *this].trigPosDisp() * tint;
		for(unsigned int i = 0; i < length; i++) {
			*times++ = t;
			t += tint;
		}

		for(unsigned int i = 0; i < num_channels; i++) {
			memcpy(tr[ *m_waveForm].cols(i + 1), shot[ *this].waveDisp(i), length * sizeof(double));
		}
		m_waveForm->drawGraph(tr);
		if(tr.commit())
			break;
	}
}
void
XDSO::onRestartTouched(const Snapshot &shot, XTouchableNode *) {
	m_timeSequenceStarted = XTime::now();
	startSequence();
}
void *
XDSO::execute(const atomic<bool> &terminated) {
	m_timeSequenceStarted = XTime::now();
	int last_count = 0;
  
	//  trace1()->setUIEnabled(false);
	//  trace2()->setUIEnabled(false);
	//  trace3()->setUIEnabled(false);
	//  trace4()->setUIEnabled(false);

	average()->setUIEnabled(true);
	singleSequence()->setUIEnabled(true);
	timeWidth()->setUIEnabled(true);
	trigSource()->setUIEnabled(true);
	trigPos()->setUIEnabled(true);
	trigLevel()->setUIEnabled(true);
	trigFalling()->setUIEnabled(true);
	vFullScale1()->setUIEnabled(true);
	vFullScale2()->setUIEnabled(true);
	vFullScale3()->setUIEnabled(true);
	vFullScale4()->setUIEnabled(true);
	vOffset1()->setUIEnabled(true);
	vOffset2()->setUIEnabled(true);
	vOffset3()->setUIEnabled(true);
	vOffset4()->setUIEnabled(true);
	forceTrigger()->setUIEnabled(true);
	recordLength()->setUIEnabled(true);
	dRFMode()->setUIEnabled(true);
	dRFFreq()->setUIEnabled(true);
	dRFSG()->setUIEnabled(true);

	for(Transaction tr( *this);; ++tr) {
		m_lsnOnAverageChanged = tr[ *average()].onValueChanged().connectWeakly(
			shared_from_this(), &XDSO::onAverageChanged);
		m_lsnOnSingleChanged = tr[ *singleSequence()].onValueChanged().connectWeakly(
			shared_from_this(), &XDSO::onSingleChanged);
		m_lsnOnTimeWidthChanged = tr[ *timeWidth()].onValueChanged().connectWeakly(
			shared_from_this(), &XDSO::onTimeWidthChanged);
		m_lsnOnTrigSourceChanged = tr[ *trigSource()].onValueChanged().connectWeakly(
			shared_from_this(), &XDSO::onTrigSourceChanged);
		m_lsnOnTrigPosChanged = tr[ *trigPos()].onValueChanged().connectWeakly(
			shared_from_this(), &XDSO::onTrigPosChanged);
		m_lsnOnTrigLevelChanged = tr[ *trigLevel()].onValueChanged().connectWeakly(
			shared_from_this(), &XDSO::onTrigLevelChanged);
		m_lsnOnTrigFallingChanged = tr[ *trigFalling()].onValueChanged().connectWeakly(
			shared_from_this(), &XDSO::onTrigFallingChanged);
		m_lsnOnTrace1Changed = tr[ *trace1()].onValueChanged().connectWeakly(
			shared_from_this(), &XDSO::onTrace1Changed);
		m_lsnOnTrace2Changed = tr[ *trace2()].onValueChanged().connectWeakly(
			shared_from_this(), &XDSO::onTrace2Changed);
		m_lsnOnTrace3Changed = tr[ *trace3()].onValueChanged().connectWeakly(
			shared_from_this(), &XDSO::onTrace3Changed);
		m_lsnOnTrace4Changed = tr[ *trace4()].onValueChanged().connectWeakly(
			shared_from_this(), &XDSO::onTrace4Changed);
		m_lsnOnVFullScale1Changed = tr[ *vFullScale1()].onValueChanged().connectWeakly(
			shared_from_this(), &XDSO::onVFullScale1Changed);
		m_lsnOnVFullScale2Changed = tr[ *vFullScale2()].onValueChanged().connectWeakly(
			shared_from_this(), &XDSO::onVFullScale2Changed);
		m_lsnOnVFullScale3Changed = tr[ *vFullScale3()].onValueChanged().connectWeakly(
			shared_from_this(), &XDSO::onVFullScale3Changed);
		m_lsnOnVFullScale4Changed = tr[ *vFullScale4()].onValueChanged().connectWeakly(
			shared_from_this(), &XDSO::onVFullScale4Changed);
		m_lsnOnVOffset1Changed = tr[ *vOffset1()].onValueChanged().connectWeakly(
			shared_from_this(), &XDSO::onVOffset1Changed);
		m_lsnOnVOffset2Changed = tr[ *vOffset2()].onValueChanged().connectWeakly(
			shared_from_this(), &XDSO::onVOffset2Changed);
		m_lsnOnVOffset3Changed = tr[ *vOffset3()].onValueChanged().connectWeakly(
			shared_from_this(), &XDSO::onVOffset3Changed);
		m_lsnOnVOffset4Changed = tr[ *vOffset4()].onValueChanged().connectWeakly(
			shared_from_this(), &XDSO::onVOffset4Changed);
		m_lsnOnRecordLengthChanged = tr[ *recordLength()].onValueChanged().connectWeakly(
			shared_from_this(), &XDSO::onRecordLengthChanged);
		m_lsnOnForceTriggerTouched = tr[ *forceTrigger()].onTouch().connectWeakly(
			shared_from_this(), &XDSO::onForceTriggerTouched);
		m_lsnOnRestartTouched = tr[ *restart()].onTouch().connectWeakly(
			shared_from_this(), &XDSO::onRestartTouched);
		 if(tr.commit())
			break;
	}

	while( !terminated) {
		Snapshot shot( *this);
		const int fetch_mode = shot[ *fetchMode()];
		if( !fetch_mode || (fetch_mode == FETCHMODE_NEVER)) {
			msecsleep(100);
			continue;
		}
		std::deque<XString> channels;
		{
			XString chstr = shot[ *trace1()].to_str();
			if( !chstr.empty())
				channels.push_back(chstr);
			chstr = shot[ *trace2()].to_str();
			if( !chstr.empty())
				channels.push_back(chstr);
			chstr = shot[ *trace3()].to_str();
			if( !chstr.empty())
				channels.push_back(chstr);
			chstr = shot[ *trace4()].to_str();
			if( !chstr.empty())
				channels.push_back(chstr);
		}
		if( !channels.size()) {
            statusPrinter()->printMessage(getLabel() + " " + i18n("Select traces!."));
            msecsleep(500);
            continue;
		}
		
		bool seq_busy = false;
		int count;
		try {
			count = acqCount( &seq_busy);
			if( !count) {
				last_count = 0;
				msecsleep(10);
				continue;
			}
			if(count == last_count) {
			//Nothing happened after the last reading.
				msecsleep(10);
				continue;
			}
			if(fetch_mode == FETCHMODE_SEQ) {
				if(shot[ *singleSequence()] && seq_busy) {
					msecsleep(10);
					continue;
				}
			}
		}
		catch (XKameError& e) {
			e.print(getLabel());
			continue;
		}
      
		shared_ptr<RawData> writer(new RawData);
		// try/catch exception of communication errors
		try {
			getWave(writer, channels);
		}
		catch (XDriver::XSkippedRecordError&) {
			continue;
		}
		catch (XKameError &e) {
			e.print(getLabel());
			continue;
		}
      
		trans( *this).m_rawDisplayOnly = (shot[ *singleSequence()] && seq_busy);

		finishWritingRaw(writer, m_timeSequenceStarted, XTime::now());
	      
		last_count =  count;
		
		if(shot[ *singleSequence()] && !seq_busy) {
			last_count = 0;
			m_timeSequenceStarted = XTime::now();
			// try/catch exception of communication errors
			try {
				startSequence();
			}
			catch (XKameError &e) {
				e.print(getLabel());
				continue;
			}
		}
    }
    trans( *this).m_rawDisplayOnly = false;

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
	vFullScale3()->setUIEnabled(false);
	vFullScale4()->setUIEnabled(false);
	vOffset1()->setUIEnabled(false);
	vOffset2()->setUIEnabled(false);
	vOffset3()->setUIEnabled(false);
	vOffset4()->setUIEnabled(false);
	forceTrigger()->setUIEnabled(false);
	recordLength()->setUIEnabled(false);
//	dRFMode()->setUIEnabled(false);
//	dRFFreq()->setUIEnabled(false);
//	dRFSG()->setUIEnabled(false);

	m_lsnOnAverageChanged.reset();
	m_lsnOnSingleChanged.reset();
	m_lsnOnTimeWidthChanged.reset();
	m_lsnOnTrigSourceChanged.reset();
	m_lsnOnTrigPosChanged.reset();
	m_lsnOnTrigLevelChanged.reset();
	m_lsnOnTrigFallingChanged.reset();
	m_lsnOnVFullScale1Changed.reset();
	m_lsnOnVFullScale2Changed.reset();
	m_lsnOnVFullScale3Changed.reset();
	m_lsnOnVFullScale4Changed.reset();
	m_lsnOnTrace1Changed.reset();
	m_lsnOnTrace2Changed.reset();
	m_lsnOnTrace3Changed.reset();
	m_lsnOnTrace4Changed.reset();
	m_lsnOnVOffset1Changed.reset();
	m_lsnOnVOffset2Changed.reset();
	m_lsnOnVOffset3Changed.reset();
	m_lsnOnVOffset4Changed.reset();
	m_lsnOnForceTriggerTouched.reset();
	m_lsnOnRestartTouched.reset();
	m_lsnOnRecordLengthChanged.reset();
//	m_lsnOnDRFCondChanged.reset();

	return NULL;
}

void
XDSO::onCondChanged(const Snapshot &shot, XValueNodeBase *) {
	Snapshot shot_this( *this);
	visualize(shot_this);
}
void
XDSO::onDRFCondChanged(const Snapshot &shot, XValueNodeBase *) {
	for(Transaction tr( *this);; ++tr) {
		tr[ *this].m_dRFRefWave.reset();
		tr[ *restart()].touch();
		if(tr.commit())
			break;
	}
}
double
XDSO::phaseOfRF(const Snapshot &shot_of_this, uint64_t count, double interval) {
	double freq;
	switch (shot_of_this[ *dRFMode()]) {
	default:
		return 0.0;
	case DRFMODE_GIVEN_FREQ:
		freq = shot_of_this[ *dRFFreq()] * 1e6;
		break;
	case DRFMODE_COHERENT_SG:
	case DRFMODE_FREQ_BY_SG:
		shared_ptr<XSG> sg = shot_of_this[ *dRFSG()];
		if( !sg)
			return 0.0;
		freq = ***sg->freq() * 1e6;
		break;
	}
	const uint64_t tens = 10000000000uLL; //# of tens for SG PLL.
	uint64_t a = llrint(freq * interval * tens);
	a = a % tens;
	count = count % tens;
	double x = ((long double)a * count) / tens;
	return 2.0 * M_PI * x; //2pi f/T
}
void
XDSO::Payload::setParameters(unsigned int channels, double startpos, double interval, unsigned int length) {
	m_numChannelsDisp = channels;
	m_wavesDisp.resize(channels * length, 0.0);
	m_trigPosDisp = -startpos / interval;
	m_timeIntervalDisp = interval;
}
void
XDSO::demodulateDisp(Transaction &tr) throw (XRecordError&) {
	Snapshot &shot(tr);
	unsigned int num_channels = shot[ *this].numChannelsDisp();
	unsigned int length = shot[ *this].lengthDisp();
	if( !shot[ *this].m_dRFRefWave) {
		tr[ *this].m_dRFRefWave.reset(new std::vector<std::complex<double> >(length));
		auto *vec = &shot[ *this].m_dRFRefWave->at(0);
		double omega = phaseOfRF(shot, 1, shot[ *this].timeIntervalDisp());
		double trigpos = shot[ *this].trigPosDisp();
		for(int i = 0; i < length; ++i) {
			vec[i] = std::polar(1.0, - omega * (i - trigpos)); // exp( -i omega t)
		}
	}

	auto *wave_ref = &shot[ *this].m_dRFRefWave->at(0);
	switch(shot[ *dRFMode()]) {
	case DRFMODE_COHERENT_SG:
		{
			 if( !isDRFCoherentSGSupported())
					throw XSkippedRecordError(i18n("RF with coherent SG is not supported."), __FILE__, __LINE__);
			if(num_channels % 2 == 1)
				throw XSkippedRecordError(i18n("Inconsistent number of channels."), __FILE__, __LINE__);
			for(unsigned int i = 0; i < num_channels; i += 2) {
				double *wave_re = tr[ *this].waveDisp(i);
				double *wave_im = tr[ *this].waveDisp(i + 1);
				for(int i = 0; i < length; ++i) {
					auto z = wave_ref[i] * std::complex<double>(wave_re[i], wave_im[i]);
					wave_re[i] = std::real(z);
					wave_im[i] = std::imag(z);
				}
			}
		}
		break;
	default:
		{
			for(unsigned int i = 0; i < num_channels; ++i) {
				double *wave_re = tr[ *this].waveDisp(i);
				for(int i = 0; i < length; ++i) {
					wave_re[i] = std::real(wave_ref[i] * wave_re[i]);
				}
			}
		}
		break;
	}
}
void
XDSO::convertRawToDisp(RawDataReader &reader, Transaction &tr) throw (XRecordError&) {
    convertRaw(reader, tr);
    
	Snapshot &shot(tr);
	unsigned int num_channels = shot[ *this].numChannelsDisp();
	if( !num_channels) {
		throw XSkippedRecordError(__FILE__, __LINE__);
	}
	if(shot[ *dRFMode()] > DRFMODE_OFF) {
		demodulateDisp(tr); //digital RF demodulation.
	}
	if(shot[ *firEnabled()]) {
		double  bandwidth = shot[ *firBandWidth()] * 1000.0 * shot[ *this].timeIntervalDisp();
		double fir_sharpness = shot[ *firSharpness()];
		if(fir_sharpness < 4.0)
			m_statusPrinter->printWarning(i18n("Too small number of taps for FIR filter."));
		int taps = std::min((int)lrint(2 * fir_sharpness / bandwidth), 5000);
		double center = shot[ *firCenterFreq()] * 1000.0 * shot[ *this].timeIntervalDisp();
		if( !shot[ *this].m_fir || (taps != shot[ *this].m_fir->taps()) ||
			(bandwidth != shot[ *this].m_fir->bandWidth()) || (center != shot[ *this].m_fir->centerFreq()))
			tr[ *this].m_fir.reset(new FIR(taps, bandwidth, center));
		unsigned int length = shot[ *this].lengthDisp();
		std::vector<double> buf(length);
		for(unsigned int i = 0; i < num_channels; i++) {
			shot[ *this].m_fir->exec(tr[ *this].waveDisp(i), &buf[0], length);
			memcpy(tr[ *this].waveDisp(i), &buf[0], length * sizeof(double));
		}
	}
}
void
XDSO::analyzeRaw(RawDataReader &reader, Transaction &tr) throw (XRecordError&) {
    convertRawToDisp(reader, tr);

	if(tr[ *this].m_rawDisplayOnly) {
		throw XSkippedRecordError(__FILE__, __LINE__); //visualize() will be called.
	}
	//    std::fill(m_wavesRecorded.begin(), m_wavesRecorded.end(), 0.0);
	tr[ *this].m_numChannels = tr[ *this].m_numChannelsDisp;
	tr[ *this].m_waves.resize(tr[ *this].m_wavesDisp.size());
	tr[ *this].m_trigPos = tr[ *this].m_trigPosDisp;
	tr[ *this].m_timeInterval = tr[ *this].m_timeIntervalDisp;
	memcpy( &tr[ *this].m_waves[0], &tr[ *this].m_wavesDisp[0], tr[ *this].m_wavesDisp.size() * sizeof(double));
}
