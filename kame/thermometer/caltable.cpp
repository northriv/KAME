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
//---------------------------------------------------------------------
#include "caltable.h"
#include "measure.h"
#include <QPushButton>
#include <fstream>
#include "graph.h"
#include "graphwidget.h"
#include "xwavengraph.h"
#include "ui_caltableform.h"
#include "ui_graphnurlform.h"

//---------------------------------------------------------------------

XConCalTable::XConCalTable
(const shared_ptr<XThermometerList> &list, FrmCalTable *form)
	:  XQConnector(list, form), 
	   m_list(list),
	   m_display(XNode::createOrphan<XTouchableNode>("display") ),
	   m_temp(XNode::createOrphan<XDoubleNode>("temp") ),
	   m_value(XNode::createOrphan<XDoubleNode>("value") ),
	   m_pForm(form),
	   m_waveform(new FrmGraphNURL(g_pFrmMain, Qt::Window)),
	   m_wave(XNode::createOrphan<XWaveNGraph>("Waveform", true, m_waveform.get())) {

    list->iterate_commit([=](Transaction &tr){
		m_thermometer = XNode::createOrphan<XItemNode<XThermometerList, XThermometer> >(
						 "thermometer", false, ref(tr), list, true);
    });

    m_conThermo = xqcon_create<XQComboBoxConnector> (m_thermometer,
		(QComboBox *) form->cmbThermometer, Snapshot( *list));
	m_conTemp = xqcon_create<XQLineEditConnector> (m_temp, form->edTemp, false);
	m_conValue = xqcon_create<XQLineEditConnector> (m_value, form->edValue, false);
	m_conDisplay = xqcon_create<XQButtonConnector> (m_display, form->btnDisplay);

    temp()->iterate_commit([=](Transaction &tr){
		m_lsnTemp = tr[ *temp()].onValueChanged().connectWeakly(
			shared_from_this(),
			&XConCalTable::onTempChanged);
    });
    value()->iterate_commit([=](Transaction &tr){
		m_lsnValue = tr[ *value()].onValueChanged().connectWeakly(
			shared_from_this(),
			&XConCalTable::onValueChanged);
    });
    display()->iterate_commit([=](Transaction &tr){
		m_lsnDisplay = tr[ *display()].onTouch().connectWeakly(
			shared_from_this(),
			&XConCalTable::onDisplayTouched, Listener::FLAG_MAIN_THREAD_CALL);
    });

	m_waveform->setWindowTitle(i18n("Thermometer Calibration"));
    m_wave->iterate_commit([=](Transaction &tr){
		const char *labels[] = {"Temp. [K]", "Value", "T(v(T))-T [K]"};
		tr[ *m_wave].setColCount(3, labels);
		tr[ *m_wave].insertPlot(labels[1], 0, 1);
		tr[ *m_wave].insertPlot(labels[2], 0, -1, 2);
		tr[ *tr[ *m_wave].plot(0)->label()] = i18n("Curve");
		tr[ *tr[ *m_wave].plot(0)->drawPoints()] = false;
		tr[ *tr[ *m_wave].plot(1)->label()] = i18n("Error");
		tr[ *tr[ *m_wave].plot(1)->drawPoints()] = false;
		shared_ptr<XAxis> axisx = tr[ *m_wave].axisx();
		tr[ *axisx->logScale()] = true;
		shared_ptr<XAxis> axisy = tr[ *m_wave].axisy();
		tr[ *axisy->logScale()] = true;
		m_wave->drawGraph(tr);
		tr[ *m_wave].clearPoints();
    });
}

void
XConCalTable::onTempChanged(const Snapshot &shot, XValueNodeBase *) {
	shared_ptr<XThermometer> thermo = ***thermometer();
	if( !thermo) return;
	double ret = thermo->getRawValue(shot[ *temp()]);
    value()->iterate_commit([=](Transaction &tr){
		tr[ *value()] = ret;
		tr.unmark(m_lsnValue);
    });
}
void
XConCalTable::onValueChanged(const Snapshot &shot, XValueNodeBase *) {
	shared_ptr<XThermometer> thermo = ***thermometer();
	if( !thermo) return;
	double ret = thermo->getTemp(shot[ *value()]);
    temp()->iterate_commit([=](Transaction &tr){
		tr[ *temp()] = ret;
		tr.unmark(m_lsnTemp);
    });
}
void
XConCalTable::onDisplayTouched(const Snapshot &shot, XTouchableNode *) {
	shared_ptr<XThermometer> thermo = ***thermometer();
	if( !thermo) {
        m_wave->iterate_commit([=](Transaction &tr){
			tr[ *m_wave].clearPoints();
        });
		return;
	}
	const int length = 1000;
	Snapshot shot_th( *thermo);
	double step = (log(shot_th[ *thermo->tempMax()]) - log(shot_th[ *thermo->tempMin()])) / length;
    m_wave->iterate_commit([=](Transaction &tr){
		tr[ *m_wave].setRowCount(length);
		double lt = log(shot_th[ *thermo->tempMin()]);
		for(int i = 0; i < length; ++i) {
			double t = exp(lt);
			double r = thermo->getRawValue(t);
			tr[ *m_wave].cols(0)[i] = t;
			tr[ *m_wave].cols(1)[i] = r;
			tr[ *m_wave].cols(2)[i] = thermo->getTemp(r) - t;
			lt += step;
		}
		m_wave->drawGraph(tr);
    });
    m_waveform->showNormal();
	m_waveform->raise();  
}
