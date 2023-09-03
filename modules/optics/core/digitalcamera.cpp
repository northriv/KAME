/***************************************************************************
        Copyright (C) 2002-2023 Kentaro Kitagawa
		                   kitagawa@phys.s.u-tokyo.ac.jp
		
		This program is free software; you can redistribute it and/or
		modify it under the terms of the GNU Library General Public
		License as published by the Free Software Foundation; either
		version 2 of the License, or (at your option) any later version.
		
		You should have received a copy of the GNU Library General 
		Public License and a list of authors along with this program; 
		see the files COPYING and AUTHORS.
***************************************************************************/
#include "digitalcamera.h"
#include "ui_digitalcameraform.h"
#include "xwavengraph.h"
#include "graph.h"
#include "graphwidget.h"

#include "interface.h"
#include "analyzer.h"
#include "xnodeconnector.h"

XDigitalCamera::XDigitalCamera(const char *name, bool runtime,
	Transaction &tr_meas, const shared_ptr<XMeasure> &meas) :
	XPrimaryDriverWithThread(name, runtime, ref(tr_meas), meas),
    m_exposure(create<XDoubleNode>("Exposure", true)),
    m_average(create<XUIntNode>("Average", false)),
    m_storeDark(create<XTouchableNode>("StoreDark", true)),
    m_subtractDark(create<XBoolNode>("SubtractDark", false)),
    m_form(new FrmDigitalCamera),
	m_waveForm(create<XWaveNGraph>("WaveForm", false, 
                                   m_form->m_graphwidget, m_form->m_edDump, m_form->m_tbDump, m_form->m_btnDump)) {


    m_conUIs = {
        xqcon_create<XQSpinBoxUnsignedConnector>(average(), m_form->m_spbAverage),
        xqcon_create<XQLineEditConnector>(exposure(), m_form->m_edIntegrationTime),
        xqcon_create<XQButtonConnector>(storeDark(), m_form->m_btnStoreDark),
        xqcon_create<XQToggleButtonConnector>(subtractDark(), m_form->m_ckbSubtractDark)
    };

    m_waveForm->iterate_commit([=](Transaction &tr){
        const char *labels[] = {"Wavelength [nm]", "Count", "Averaging Count", "Dark Count"};
        tr[ *m_waveForm].setColCount(4, labels);
		tr[ *m_waveForm].insertPlot(labels[1], 0, 1);
        tr[ *m_waveForm].insertPlot(labels[2], 0, 2);

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
        tr[ *axisy->autoScale()] = true;
//		tr[ *axisy2->ticColor()] = clWhite;
//		tr[ *axisy2->labelColor()] = clWhite;
//		tr[ *axisy2->ticLabelColor()] = clWhite;
		tr[ *tr[ *m_waveForm].plot(0)->drawPoints()] = false;
//		tr[ *tr[ *m_waveForm].plot(0)->lineColor()] = clGreen;
//		tr[ *tr[ *m_waveForm].plot(0)->pointColor()] = clGreen;
		tr[ *tr[ *m_waveForm].plot(0)->intensity()] = 2.0;
        tr[ *tr[ *m_waveForm].plot(1)->drawPoints()] = false;
        tr[ *tr[ *m_waveForm].plot(1)->lineColor()] = clGreen;
        shared_ptr<XXYPlot> plot = m_graph->plots()->create<XXYPlot>(
			tr, "Markers", true, tr, m_graph);
        tr[ *plot->label()] = i18n("Marker");
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
//        startWavelen(),
//        stopWavelen(),
//        average(),
//        storeDark(),
//        subtractDark(),
        exposure()
    };
    iterate_commit([=](Transaction &tr){
        tr[ *average()] = 1;
        for(auto &&x: runtime_ui)
            tr[ *x].setUIEnabled(false);
    });
}
void
XDigitalCamera::showForms() {
// impliment form->show() here
    m_form->showNormal();
    m_form->raise();
}

void
XDigitalCamera::onStoreDarkTouched(const Snapshot &shot, XTouchableNode *) {
    m_storeDarkInvoked = true;
}

void
XDigitalCamera::analyzeRaw(RawDataReader &reader, Transaction &tr)  {
}
void
XDigitalCamera::visualize(const Snapshot &shot) {
	  if( !shot[ *this].time()) {
		return;
	  }
}

void *
XDigitalCamera::execute(const atomic<bool> &terminated) {

    std::vector<shared_ptr<XNode>> runtime_ui{
//        startWavelen(),
//        stopWavelen(),
//        average(),
//        storeDark(),
//        subtractDark()
        exposure()
        };

    m_storeDarkInvoked = false;

	iterate_commit([=](Transaction &tr){
		m_lsnOnAverageChanged = tr[ *average()].onValueChanged().connectWeakly(
            shared_from_this(), &XDigitalCamera::onAverageChanged);
        m_lsnOnExposureChanged = tr[ *exposure()].onValueChanged().connectWeakly(
                    shared_from_this(), &XDigitalCamera::onExposureChanged);
        m_lsnOnStoreDarkTouched = tr[ *storeDark()].onTouch().connectWeakly(
            shared_from_this(), &XDigitalCamera::onStoreDarkTouched, Listener::FLAG_MAIN_THREAD_CALL);
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

    m_lsnOnExposureChanged.reset();
	m_lsnOnAverageChanged.reset();
    m_lsnOnStoreDarkTouched.reset();
	return NULL;
}
