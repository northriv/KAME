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
	   m_waveform(new FrmGraphNURL(g_pFrmMain)),
	   m_wave(XNode::createOrphan<XWaveNGraph>("Waveform", true, m_waveform.get())) {

	for(Transaction tr( *list);; ++tr) {
		m_thermometer = XNode::createOrphan<XItemNode<XThermometerList, XThermometer> >(
						 "thermometer", false, ref(tr), list, true);
		if(tr.commit())
			break;
	}

    m_conThermo = xqcon_create<XQComboBoxConnector> (m_thermometer,
		(QComboBox *) form->cmbThermometer, Snapshot( *list));
	m_conTemp = xqcon_create<XQLineEditConnector> (m_temp, form->edTemp, false);
	m_conValue = xqcon_create<XQLineEditConnector> (m_value, form->edValue, false);
	m_conDisplay = xqcon_create<XQButtonConnector> (m_display, form->btnDisplay);

	for(Transaction tr( *temp());; ++tr) {
		m_lsnTemp = tr[ *temp()].onValueChanged().connectWeakly(
			shared_from_this(),
			&XConCalTable::onTempChanged);
		if(tr.commit())
			break;
	}
	for(Transaction tr( *value());; ++tr) {
		m_lsnValue = tr[ *value()].onValueChanged().connectWeakly(
			shared_from_this(),
			&XConCalTable::onValueChanged);
		if(tr.commit())
			break;
	}
	for(Transaction tr( *display());; ++tr) {
		m_lsnDisplay = tr[ *display()].onTouch().connectWeakly(
			shared_from_this(),
			&XConCalTable::onDisplayTouched, XListener::FLAG_MAIN_THREAD_CALL);
		if(tr.commit())
			break;
	}

	m_waveform->setWindowTitle(i18n("Thermometer Calibration"));
	for(Transaction tr( *m_wave);; ++tr) {
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
		if(tr.commit())
			break;
	}
}

void
XConCalTable::onTempChanged(const Snapshot &shot, XValueNodeBase *) {
	shared_ptr<XThermometer> thermo = ***thermometer();
	if( !thermo) return;
	double ret = thermo->getRawValue(shot[ *temp()]);
	for(Transaction tr( *value());; ++tr) {
		tr[ *value()] = ret;
		tr.unmark(m_lsnValue);
		if(tr.commit())
			break;
	}
}
void
XConCalTable::onValueChanged(const Snapshot &shot, XValueNodeBase *) {
	shared_ptr<XThermometer> thermo = ***thermometer();
	if( !thermo) return;
	double ret = thermo->getTemp(shot[ *value()]);
	for(Transaction tr( *temp());; ++tr) {
		tr[ *temp()] = ret;
		tr.unmark(m_lsnTemp);
		if(tr.commit())
			break;
	}
}
void
XConCalTable::onDisplayTouched(const Snapshot &shot, XTouchableNode *) {
	shared_ptr<XThermometer> thermo = ***thermometer();
	if( !thermo) {
		for(Transaction tr( *m_wave);; ++tr) {
			tr[ *m_wave].clearPoints();
			if(tr.commit())
				break;
		}
		return;
	}
	const int length = 1000;
	Snapshot shot_th( *thermo);
	double step = (log(shot_th[ *thermo->tempMax()]) - log(shot_th[ *thermo->tempMin()])) / length;
	for(Transaction tr( *m_wave);; ++tr) {
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
		if(tr.commit())
			break;
	}
    m_waveform->showNormal();
	m_waveform->raise();  
}
