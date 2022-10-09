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
#include "opticalspectrometer.h"
#include "ui_opticalspectrometerform.h"
#include "xwavengraph.h"
#include "graph.h"
#include "graphwidget.h"

#include "interface.h"
#include "analyzer.h"
#include "xnodeconnector.h"

XOpticalSpectrometer::XOpticalSpectrometer(const char *name, bool runtime,
	Transaction &tr_meas, const shared_ptr<XMeasure> &meas) :
	XPrimaryDriverWithThread(name, runtime, ref(tr_meas), meas),
    m_marker1X(create<XScalarEntry>("Marker1X", false, 
								  dynamic_pointer_cast<XDriver>(shared_from_this()))),
    m_marker1Y(create<XScalarEntry>("Marker1Y", false, 
								  dynamic_pointer_cast<XDriver>(shared_from_this()))),
    m_startWavelen(create<XDoubleNode>("StartWavelen", true)),
    m_stopWavelen(create<XDoubleNode>("StopWavelen", true)),
    m_integrationTime(create<XDoubleNode>("IntegrationTime", true)),
    m_average(create<XUIntNode>("Average", false)),
    m_storeDark(create<XTouchableNode>("StoreDark", true)),
    m_subtractDark(create<XBoolNode>("SubtractDark", false)),
    m_form(new FrmOpticalSpectrometer),
	m_waveForm(create<XWaveNGraph>("WaveForm", false, 
                                   m_form->m_graphwidget, m_form->m_edDump, m_form->m_tbDump, m_form->m_btnDump)) {

	meas->scalarEntries()->insert(tr_meas, m_marker1X);
	meas->scalarEntries()->insert(tr_meas, m_marker1Y);

    m_conUIs = {
        xqcon_create<XQLineEditConnector>(startWavelen(), m_form->m_edStart),
        xqcon_create<XQLineEditConnector>(stopWavelen(), m_form->m_edStop),
        xqcon_create<XQLineEditConnector>(average(), m_form->m_edAverage),
        xqcon_create<XQLineEditConnector>(integrationTime(), m_form->m_edIntegrationTime),
    };

    m_waveForm->iterate_commit([=](Transaction &tr){
        const char *labels[] = {"Wavelen [nm]", "Count"};
        tr[ *m_waveForm].setColCount(2, labels);
		tr[ *m_waveForm].insertPlot(labels[1], 0, 1);

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
		shared_ptr<XXYPlot> plot = m_graph->plots()->create<XXYPlot>(
			tr, "Markers", true, tr, m_graph);
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
    std::vector<shared_ptr<XNode>> runtime_ui{
        startWavelen(),
        stopWavelen(),
        average(),
        storeDark(),
        subtractDark()
    };
    iterate_commit([=](Transaction &tr){
        for(auto &&x: runtime_ui)
            tr[ *x].setUIEnabled(false);
    });
}
void
XOpticalSpectrometer::showForms() {
// impliment form->show() here
    m_form->showNormal();
    m_form->raise();
}

void
XOpticalSpectrometer::onStoreDarkTouched(const Snapshot &shot, XTouchableNode *) {
    m_storeDarkInvoked = true;
}

void
XOpticalSpectrometer::analyzeRaw(RawDataReader &reader, Transaction &tr)  {
    if(tr[ *this].m_accumulated >= tr[ *average()]) {
        tr[ *this].m_accumulated = 0;
        std::fill(tr[ *this].accumCounts_().begin(), tr[ *this].accumCounts_().end(), 0.0);
    }
    convertRawAndAccum(reader, tr);
    unsigned int accumulated = tr[ *this].m_accumulated;
    if(m_storeDarkInvoked) {
        m_storeDarkInvoked = false;
        tr[ *this].darkCounts_() = tr[ *this].accumCounts_();
        for(auto& x: tr[ *this].darkCounts_())
            x /= accumulated;
        gMessagePrint(i18n("Dark spectrum has been stored."));
    }

    const unsigned int length = tr[ *this].length();
    const double *av = &tr[ *this].accumCounts_()[0];
    double *v = &tr[ *this].counts_()[0];
    const double *vd = nullptr;
    if(tr[ *subtractDark()] && (tr[ *this].darkCounts_().size() == length))
        vd = tr[ *this].darkCounts();
    for(unsigned int i = 0; i < length; i++) {
        double y = *av++ / accumulated;
        if(vd)
            y -= *vd++;
        *v++ = y;
    }
    if(accumulated < tr[ *average()]) {
        throw XSkippedRecordError(__FILE__, __LINE__); //visualize() will be called.
    }

    //markers
    auto it = std::max_element(tr[ *this].counts_().begin(), tr[ *this].counts_().end());
    int idx = std::distance(it, tr[ *this].counts_().begin());
    tr[ *this].markers().clear();
    tr[ *this].markers().emplace_back(tr[ *this].waveLengths()[idx], *it);
    m_marker1X->value(tr, tr[ *this].waveLengths()[idx]);
    m_marker1Y->value(tr, *it);
    //ugly hack
    double sum = 0.0;
    int cnt = 0;
    for(unsigned int i = 0; i < length; ++i) {
        double lambda = tr[ *this].waveLengths()[i];
        if((lambda >= tr[ *startWavelen()]) && (lambda <= tr[ *stopWavelen()])) {
            sum += tr[ *this].counts()[i];
            cnt++;
        }
    }
    if(cnt)
        m_marker1Y->value(tr, sum);
}
void
XOpticalSpectrometer::visualize(const Snapshot &shot) {
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
        std::vector<float> wavelens(length), mags(length);

        const double *v = shot[ *this].counts();
        const double *wl = shot[ *this].waveLengths();
		for(unsigned int i = 0; i < length; i++) {
            wavelens[i] = *wl++;
            double y = *v++;
            mags[i] = y;
		}
        tr[ *m_waveForm].setColumn(0, std::move(wavelens), 7);
        tr[ *m_waveForm].setColumn(1, std::move(mags), 5);

		m_waveForm->drawGraph(tr);
    });
}

void *
XOpticalSpectrometer::execute(const atomic<bool> &terminated) {

    std::vector<shared_ptr<XNode>> runtime_ui{
        startWavelen(),
        stopWavelen(),
        average(),
        storeDark(),
        subtractDark()
        };

    m_storeDarkInvoked = false;

	iterate_commit([=](Transaction &tr){
        m_lsnOnStartWavelenChanged = tr[ *startWavelen()].onValueChanged().connectWeakly(
            shared_from_this(), &XOpticalSpectrometer::onStartWavelenChanged);
        m_lsnOnStopWavelenChanged = tr[ *stopWavelen()].onValueChanged().connectWeakly(
            shared_from_this(), &XOpticalSpectrometer::onStopWavelenChanged);
		m_lsnOnAverageChanged = tr[ *average()].onValueChanged().connectWeakly(
            shared_from_this(), &XOpticalSpectrometer::onAverageChanged);
        m_lsnOnStoreDarkTouched = tr[ *storeDark()].onTouch().connectWeakly(
            shared_from_this(), &XOpticalSpectrometer::onStoreDarkTouched, Listener::FLAG_MAIN_THREAD_CALL);
        for(auto &&x: runtime_ui)
            tr[ *x].setUIEnabled(true);
    });

	while( !terminated) {
		XTime time_awared = XTime::now();
		auto writer = std::make_shared<RawData>();
		// try/catch exception of communication errors
		try {
            acquireSpectrum(writer);
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

    iterate_commit([=](Transaction &tr){
        for(auto &&x: runtime_ui)
            tr[ *x].setUIEnabled(false);
    });

    m_lsnOnStartWavelenChanged.reset();
    m_lsnOnStopWavelenChanged.reset();
	m_lsnOnAverageChanged.reset();
    m_lsnOnStoreDarkTouched.reset();
	return NULL;
}
