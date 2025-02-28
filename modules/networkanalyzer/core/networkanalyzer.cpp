/***************************************************************************
		Copyright (C) 2002-2015 Kentaro Kitagawa
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
#include "ui_networkanalyzerform.h"
#include "xwavengraph.h"
#include "graph.h"
#include "graphwidget.h"

#include "interface.h"
#include "analyzer.h"
#include "xnodeconnector.h"

XNetworkAnalyzer::XNetworkAnalyzer(const char *name, bool runtime,
	Transaction &tr_meas, const shared_ptr<XMeasure> &meas) :
	XPrimaryDriverWithThread(name, runtime, ref(tr_meas), meas),
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
	m_points(create<XComboNode>("Points", false, true)),
	m_average(create<XUIntNode>("Average", false)),
    m_power(create<XDoubleNode>("Power", false)),
    m_calOpen(create<XTouchableNode>("CalOpen", true)),
	m_calShort(create<XTouchableNode>("CalShort", true)),
	m_calTerm(create<XTouchableNode>("CalTerm", true)),
	m_calThru(create<XTouchableNode>("CalThru", true)),
    m_form(new FrmNetworkAnalyzer),
	m_waveForm(create<XWaveNGraph>("WaveForm", false, 
        m_form->m_graphwidget, m_form->m_edDump, m_form->m_tbDump, m_form->m_btnDump,
        m_form->m_tbMath, meas, static_pointer_cast<XDriver>(shared_from_this()))) {

	meas->scalarEntries()->insert(tr_meas, m_marker1X);
	meas->scalarEntries()->insert(tr_meas, m_marker1Y);
	meas->scalarEntries()->insert(tr_meas, m_marker2X);
	meas->scalarEntries()->insert(tr_meas, m_marker2Y);
	
	startFreq()->setUIEnabled(false);
	stopFreq()->setUIEnabled(false);
	points()->setUIEnabled(false);
	average()->setUIEnabled(false);
    power()->setUIEnabled(false);
    calOpen()->setUIEnabled(false);
	calShort()->setUIEnabled(false);
	calTerm()->setUIEnabled(false);
	calThru()->setUIEnabled(false);

    m_conUIs = {
        xqcon_create<XQLineEditConnector>(startFreq(), m_form->m_edStart),
        xqcon_create<XQLineEditConnector>(stopFreq(), m_form->m_edStop),
        xqcon_create<XQComboBoxConnector>(points(), m_form->m_cmbPoints, Snapshot( *points())),
        xqcon_create<XQLineEditConnector>(average(), m_form->m_edAverage),
        xqcon_create<XQLineEditConnector>(power(), m_form->m_edPower),
        xqcon_create<XQButtonConnector>(m_calOpen, m_form->m_btnCalOpen),
        xqcon_create<XQButtonConnector>(m_calShort, m_form->m_btnCalShort),
        xqcon_create<XQButtonConnector>(m_calTerm, m_form->m_btnCalTerm),
        xqcon_create<XQButtonConnector>(m_calThru, m_form->m_btnCalThru)
    };

    m_waveForm->iterate_commit([=](Transaction &tr){
		const char *labels[] = {"Freq [MHz]", "Level [dB]", "Phase [deg.]"};
		tr[ *m_waveForm].setColCount(3, labels);
        if( !tr[ *m_waveForm].insertPlot(tr, labels[1], 0, 1))
            return;
        if( !tr[ *m_waveForm].insertPlot(tr, labels[2], 0, -1, 2))
            return;

		m_graph = m_waveForm->graph();
//		tr[ *m_graph->backGround()] = QColor(0x0A, 0x05, 0x34).rgb();
//		tr[ *m_graph->titleColor()] = clWhite;
		shared_ptr<XAxis> axisx = tr[ *m_waveForm].axisx();
		shared_ptr<XAxis> axisy = tr[ *m_waveForm].axisy();
		shared_ptr<XAxis> axisy2 = tr[ *m_waveForm].axisy2();
//		tr[ *axisx->ticColor()] = clWhite;
//		tr[ *axisx->labelColor()] = clWhite;
//		tr[ *axisx->ticLabelColor()] = clWhite;
//		tr[ *axisy->ticColor()] = clWhite;
//		tr[ *axisy->labelColor()] = clWhite;
//		tr[ *axisy->ticLabelColor()] = clWhite;
		tr[ *axisy->autoScale()] = false;
		tr[ *axisy->maxValue()] = 13.0;
		tr[ *axisy->minValue()] = -70.0;
//		tr[ *axisy2->ticColor()] = clWhite;
//		tr[ *axisy2->labelColor()] = clWhite;
//		tr[ *axisy2->ticLabelColor()] = clWhite;
		tr[ *tr[ *m_waveForm].plot(0)->drawPoints()] = false;
//		tr[ *tr[ *m_waveForm].plot(0)->lineColor()] = clGreen;
//		tr[ *tr[ *m_waveForm].plot(0)->pointColor()] = clGreen;
		tr[ *tr[ *m_waveForm].plot(0)->intensity()] = 2.0;
		tr[ *tr[ *m_waveForm].plot(1)->drawPoints()] = false;
//		tr[ *tr[ *m_waveForm].plot(1)->lineColor()] = clBlue;
//		tr[ *tr[ *m_waveForm].plot(1)->pointColor()] = clBlue;
        tr[ *tr[ *m_waveForm].plot(1)->intensity()] = 0.5;
		shared_ptr<XXYPlot> plot = m_graph->plots()->create<XXYPlot>(
			tr, "Markers", true, tr, m_graph);
        if( !plot) return;
		m_markerPlot = plot;
		tr[ *plot->label()] = i18n("Markers");
		tr[ *plot->axisX()] = axisx;
		tr[ *plot->axisY()] = axisy;
		tr[ *plot->drawLines()] = false;
		tr[ *plot->drawBars()] = true;
		tr[ *plot->pointColor()] = clRed;
		tr[ *plot->barColor()] = clRed;
		tr[ *plot->intensity()] = 2.0;
		tr[ *plot->clearPoints()].setUIEnabled(false);
		tr[ *plot->maxCount()].setUIEnabled(false);
		tr[ *m_waveForm].clearPoints();
    });
}
void
XNetworkAnalyzer::showForms() {
// impliment form->show() here
    m_form->showNormal();
    m_form->raise();
}

void
XNetworkAnalyzer::analyzeRaw(RawDataReader &reader, Transaction &tr) {
	const Snapshot &shot(tr);
	unsigned int numtr = reader.pop<unsigned int>();
	if(numtr != 1)
		return; 
	unsigned int nummk = reader.pop<unsigned int>();
	tr[ *this].m_markers.resize(nummk);
	for(unsigned int i = 0; i < nummk; i++) {
		tr[ *this].m_markers[i].first = reader.pop<double>();
		tr[ *this].m_markers[i].second = reader.pop<double>();
	}

    convertRaw(reader, tr);

    nummk = shot[*this].m_markers.size(); //# of markers may be changed by convertRaw()
    if(nummk == 0) {
        //searches for minimum in reflection around f0.
        int trace_len = shot[ *this].length();
        double trace_dfreq = shot[ *this].freqInterval();
        double trace_start = shot[ *this].startFreq();
        const std::complex<double> *trace = &shot[ *this].trace()[0];
        double zmin = 1e10;
        double fmin = 0.0;
        for(int i = 0; i < trace_len; ++i) {
            double z = std::abs(trace[i]);
            double f = trace_start + i * trace_dfreq;
            if(z < zmin) {
                zmin = z;
                fmin = f;
            }
        }
        m_marker1X->value(tr, fmin);
        m_marker1Y->value(tr, zmin);
    }
    if(nummk >= 1) {
        m_marker1X->value(tr, shot[ *this].m_markers[0].first);
        m_marker1Y->value(tr, shot[ *this].m_markers[0].second);
    }
    if(nummk >= 2) {
        m_marker2X->value(tr, shot[ *this].m_markers[1].first);
        m_marker2Y->value(tr, shot[ *this].m_markers[1].second);
    }
}
void
XNetworkAnalyzer::visualize(const Snapshot &shot) {
	  if( !shot[ *this].time()) {
		return;
	  }
	const unsigned int length = shot[ *this].length();
    m_waveForm->iterate_commit([=](Transaction &tr){
		tr[ *m_markerPlot->maxCount()] = shot[ *this].m_markers.size();
        auto &points(tr[ *m_markerPlot].points());
		points.clear();
		for(std::deque<std::pair<double, double> >::const_iterator it = shot[ *this].m_markers.begin();
			it != shot[ *this].m_markers.end(); it++) {
			points.push_back(XGraph::ValPoint(it->first, it->second));
		}

		tr[ *m_waveForm].setRowCount(length);
        std::vector<float> freqs(length), mags(length), phases(length);

		const std::complex<double> *z = shot[ *this].trace();
		double fint = shot[ *this].freqInterval();
		double f = shot[ *this].startFreq();
		for(unsigned int i = 0; i < length; i++) {
            freqs[i] = f;
            mags[i] = 20.0 * log10(std::abs( *z));
            phases[i] = std::arg( *z) * 180.0 /  M_PI;
			z++;
			f += fint;
		}
        tr[ *m_waveForm].setColumn(0, std::move(freqs), 7);
        tr[ *m_waveForm].setColumn(1, std::move(mags), 5);
        tr[ *m_waveForm].setColumn(2, std::move(phases), 5);

		m_waveForm->drawGraph(tr);
    });
}

void *
XNetworkAnalyzer::execute(const atomic<bool> &terminated) {

	startFreq()->setUIEnabled(true);
	stopFreq()->setUIEnabled(true);
	points()->setUIEnabled(true);
	average()->setUIEnabled(true);
    power()->setUIEnabled(true);
    calOpen()->setUIEnabled(true);
	calShort()->setUIEnabled(true);
	calTerm()->setUIEnabled(true);
	calThru()->setUIEnabled(true);

	iterate_commit([=](Transaction &tr){
		m_lsnOnStartFreqChanged = tr[ *startFreq()].onValueChanged().connectWeakly(
			shared_from_this(), &XNetworkAnalyzer::onStartFreqChanged);
		m_lsnOnStopFreqChanged = tr[ *stopFreq()].onValueChanged().connectWeakly(
			shared_from_this(), &XNetworkAnalyzer::onStopFreqChanged);
		m_lsnOnPointsChanged = tr[ *points()].onValueChanged().connectWeakly(
			shared_from_this(), &XNetworkAnalyzer::onPointsChanged);
		m_lsnOnAverageChanged = tr[ *average()].onValueChanged().connectWeakly(
			shared_from_this(), &XNetworkAnalyzer::onAverageChanged);
        m_lsnOnPowerChanged = tr[ *power()].onValueChanged().connectWeakly(
            shared_from_this(), &XNetworkAnalyzer::onPowerChanged);
        m_lsnCalOpen = tr[ *m_calOpen].onTouch().connectWeakly(shared_from_this(), &XNetworkAnalyzer::onCalOpenTouched);
		m_lsnCalShort = tr[ *m_calShort].onTouch().connectWeakly(shared_from_this(), &XNetworkAnalyzer::onCalShortTouched);
		m_lsnCalTerm = tr[ *m_calTerm].onTouch().connectWeakly(shared_from_this(), &XNetworkAnalyzer::onCalTermTouched);
		m_lsnCalThru = tr[ *m_calThru].onTouch().connectWeakly(shared_from_this(), &XNetworkAnalyzer::onCalThruTouched);
    });

	while( !terminated) {
		XTime time_awared = XTime::now();
		auto writer = std::make_shared<RawData>();
		// try/catch exception of communication errors
		try {
			oneSweep();
		}
		catch (XDriver::XSkippedRecordError&) {
			msecsleep(100);
			continue;
		}
		catch (XKameError &e) {
			e.print(getLabel());
			continue;
		}
		writer->push((unsigned int)1); //# of traces.
		double mx[8], my[8];
		unsigned int nummk = 0;
		try {
			for(;nummk < 8;nummk++) {
				mx[nummk] = 0.0;
				my[nummk] = 0.0;
				getMarkerPos(nummk, mx[nummk], my[nummk]);
			}
		}
		catch (XDriver::XSkippedRecordError&) {
        //Markers are missing.
			msecsleep(10);
		}
		catch (XKameError &e) {
			e.print(getLabel());
			continue;
		}
		writer->push((unsigned int)nummk); //# of markers.
		for(unsigned int i = 0; i < nummk; i++) {
			writer->push(mx[i]);
			writer->push(my[i]);
		}
		try {
			acquireTrace(writer, 0);
		}
		catch (XDriver::XSkippedRecordError&) {
			msecsleep(10);
			continue;
		}
		catch (XKameError &e) {
			e.print(getLabel());
			continue;
		}
		finishWritingRaw(writer, time_awared, XTime::now());
    }
	try {
		startContSweep();
	}
	catch (XKameError &e) {
		e.print(getLabel());
	}
	startFreq()->setUIEnabled(false);
	stopFreq()->setUIEnabled(false);
	points()->setUIEnabled(false);
	average()->setUIEnabled(false);
    power()->setUIEnabled(false);
    calOpen()->setUIEnabled(false);
	calShort()->setUIEnabled(false);
	calTerm()->setUIEnabled(false);
	calThru()->setUIEnabled(false);

	m_lsnOnStartFreqChanged.reset();
	m_lsnOnStopFreqChanged.reset();
	m_lsnOnPointsChanged.reset();
	m_lsnOnAverageChanged.reset();
    m_lsnOnPowerChanged.reset();
    m_lsnCalOpen.reset();
	m_lsnCalShort.reset();
	m_lsnCalTerm.reset();
	m_lsnCalThru.reset();
	return NULL;
}
