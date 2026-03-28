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
//---------------------------------------------------------------------
#include "caltable.h"
#include "measure.h"
#include <QPushButton>
#include <QTableWidget>
#include <QHeaderView>
#include <QInputDialog>
#include <algorithm>
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
						 "thermometer", false, tr, list, true);
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
    m_thermometer->iterate_commit([=](Transaction &tr){
        m_lsnThermometerChanged = tr[ *m_thermometer].onValueChanged().connectWeakly(
            shared_from_this(),
            &XConCalTable::onThermometerChanged, Listener::FLAG_MAIN_THREAD_CALL);
    });

    form->tblPoints->horizontalHeader()->setSectionResizeMode(0, QHeaderView::Stretch);
    form->tblPoints->horizontalHeader()->setSectionResizeMode(1, QHeaderView::Stretch);
    connect(form->tblPoints, &QTableWidget::cellChanged,
            this, &XConCalTable::onTableCellChanged);
    connect(form->btnAddRow, &QPushButton::clicked,
            this, &XConCalTable::onAddRowClicked);
    connect(form->btnRemoveRow, &QPushButton::clicked,
            this, &XConCalTable::onRemoveRowClicked);
    connect(form->btnNew, &QPushButton::clicked,
            this, &XConCalTable::onNewClicked);
    connect(form->btnDelete, &QPushButton::clicked,
            this, &XConCalTable::onDeleteClicked);

    populateTable();

	m_waveform->m_btnMathTool->hide();
	m_waveform->setWindowTitle(i18n("Thermometer Calibration"));
    m_wave->iterate_commit([=](Transaction &tr){
		const char *labels[] = {"Temp. [K]", "Value", "T(v(T))-T [K]"};
		tr[ *m_wave].setColCount(3, labels);
        if( !tr[ *m_wave].insertPlot(tr, labels[1], 0, 1)) return;
        if( !tr[ *m_wave].insertPlot(tr, labels[2], 0, -1, 2)) return;
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
XConCalTable::onThermometerChanged(const Snapshot &, XValueNodeBase *) {
    populateTable();
    refreshGraph();
}
void
XConCalTable::populateTable() {
    shared_ptr<XThermometer> thermo = ***m_thermometer;
    XString graphtitle = thermo ? thermo->getLabel() : XString(i18n("Thermometer Calibration"));
    m_waveform->setWindowTitle(thermo ?
        QString("%1 - %2").arg(i18n("Thermometer Calibration"), thermo->getLabel().c_str()) :
        i18n("Thermometer Calibration"));
    trans( *m_wave->graph()->label()) = graphtitle;
    auto approx = dynamic_pointer_cast<XApproxThermometer>(thermo);
    m_pForm->tblPoints->setEnabled(approx != nullptr);
    m_pForm->btnAddRow->setEnabled(approx != nullptr);
    m_pForm->btnRemoveRow->setEnabled(approx != nullptr);
    m_pForm->edTMin->setEnabled(approx != nullptr);
    m_pForm->edTMax->setEnabled(approx != nullptr);
    if( !approx) {
        if( !m_connectedApprox.expired()) {
            m_lsnTMin.reset();
            m_lsnTMax.reset();
            m_conTMin.reset();
            m_conTMax.reset();
            m_connectedApprox.reset();
        }
        m_updatingTable = true;
        m_pForm->tblPoints->setRowCount(0);
        m_updatingTable = false;
        return;
    }
    if(m_connectedApprox.lock() != approx) {
        m_lsnTMin.reset();
        m_lsnTMax.reset();
        m_conTMin.reset();
        m_conTMax.reset();
        m_conTMin = xqcon_create<XQLineEditConnector>(approx->tempMin(), m_pForm->edTMin, false);
        m_conTMax = xqcon_create<XQLineEditConnector>(approx->tempMax(), m_pForm->edTMax, false);
        approx->tempMin()->iterate_commit([=](Transaction &tr){
            m_lsnTMin = tr[ *approx->tempMin()].onValueChanged().connectWeakly(
                shared_from_this(), &XConCalTable::onTMinMaxChanged);
        });
        approx->tempMax()->iterate_commit([=](Transaction &tr){
            m_lsnTMax = tr[ *approx->tempMax()].onValueChanged().connectWeakly(
                shared_from_this(), &XConCalTable::onTMinMaxChanged);
        });
        m_connectedApprox = approx;
    }

    m_updatingTable = true;
    Snapshot shot( *approx);
    int n = 0;
    if(shot.size(approx->resList()) && shot.size(approx->tempList())) {
        const auto &res_list( *shot.list(approx->resList()));
        const auto &tmp_list( *shot.list(approx->tempList()));
        n = (int)std::min(res_list.size(), tmp_list.size());
        m_pForm->tblPoints->setRowCount(n);
        for(int i = 0; i < n; ++i) {
            double r = shot[ *static_pointer_cast<XDoubleNode>(res_list.at(i))];
            double t = shot[ *static_pointer_cast<XDoubleNode>(tmp_list.at(i))];
            m_pForm->tblPoints->setItem(i, 0, new QTableWidgetItem(QString::number(r, 'g', 10)));
            m_pForm->tblPoints->setItem(i, 1, new QTableWidgetItem(QString::number(t, 'g', 10)));
        }
    } else {
        m_pForm->tblPoints->setRowCount(0);
    }
    m_updatingTable = false;
}
void
XConCalTable::onTableCellChanged(int row, int col) {
    if(m_updatingTable) return;
    shared_ptr<XThermometer> thermo = ***m_thermometer;
    auto approx = dynamic_pointer_cast<XApproxThermometer>(thermo);
    if( !approx) return;
    QTableWidgetItem *item = m_pForm->tblPoints->item(row, col);
    if( !item) return;
    bool ok;
    double val = item->text().toDouble(&ok);
    if( !ok) return;
    Snapshot shot( *approx);
    const shared_ptr<XApproxThermometer::XDoubleListNode> &list =
        (col == 0) ? approx->resList() : approx->tempList();
    if( !shot.size(list)) return;
    const auto &node_list( *shot.list(list));
    if(row >= (int)node_list.size()) return;
    trans( *static_pointer_cast<XDoubleNode>(node_list.at(row))) = val;
    approx->invalidateCache();
    if(col == 0)
        sortByResistance(approx); // reorders rows by resistance, calls populateTable
    else
        refreshGraph();
}
void
XConCalTable::onAddRowClicked() {
    shared_ptr<XThermometer> thermo = ***m_thermometer;
    auto approx = dynamic_pointer_cast<XApproxThermometer>(thermo);
    if( !approx) return;
    approx->resList()->createByTypename("", "");
    approx->tempList()->createByTypename("", "");
    approx->invalidateCache();
    populateTable();
    refreshGraph();
}
void
XConCalTable::onRemoveRowClicked() {
    shared_ptr<XThermometer> thermo = ***m_thermometer;
    auto approx = dynamic_pointer_cast<XApproxThermometer>(thermo);
    if( !approx) return;
    int row = m_pForm->tblPoints->currentRow();
    if(row < 0) return;
    Snapshot shot( *approx);
    if( !shot.size(approx->resList()) || !shot.size(approx->tempList())) return;
    const auto &res_list( *shot.list(approx->resList()));
    const auto &tmp_list( *shot.list(approx->tempList()));
    if(row >= (int)res_list.size() || row >= (int)tmp_list.size()) return;
    auto rnode = static_pointer_cast<XDoubleNode>(res_list.at(row));
    auto tnode = static_pointer_cast<XDoubleNode>(tmp_list.at(row));
    approx->resList()->release(rnode);
    approx->tempList()->release(tnode);
    approx->invalidateCache();
    populateTable();
    refreshGraph();
}
void
XConCalTable::onTMinMaxChanged(const Snapshot &, XValueNodeBase *) {
    refreshGraph();
}
void
XConCalTable::drawGraph(const shared_ptr<XThermometer> &thermo) {
    if( !thermo) {
        m_wave->iterate_commit([=](Transaction &tr){ tr[ *m_wave].clearPoints(); });
        return;
    }
    const int length = 1000;
    Snapshot shot_th( *thermo);
    try {
        double step = (log(shot_th[ *thermo->tempMax()]) - log(shot_th[ *thermo->tempMin()])) / length;
        m_wave->iterate_commit([=](Transaction &tr){
            tr[ *m_wave].setRowCount(length);
            std::vector<double> colt(length), colr(length), coldt(length);
            double lt = log(shot_th[ *thermo->tempMin()]);
            for(int i = 0; i < length; ++i) {
                double t = exp(lt);
                double r = thermo->getRawValue(t);
                colt[i] = t;
                colr[i] = r;
                coldt[i] = thermo->getTemp(r) - t;
                lt += step;
            }
            tr[ *m_wave].setColumn(0, std::move(colt), 5);
            tr[ *m_wave].setColumn(1, std::move(colr), 5);
            tr[ *m_wave].setColumn(2, std::move(coldt), 5);
            m_wave->drawGraph(tr);
        });
    }
    catch(XKameError &) {}
}
void
XConCalTable::refreshGraph() {
    if( !m_waveform->isVisible()) return;
    drawGraph( ***thermometer());
}
void
XConCalTable::sortByResistance(const shared_ptr<XApproxThermometer> &approx) {
    Snapshot shot( *approx);
    if( !shot.size(approx->resList()) || !shot.size(approx->tempList())) return;
    const auto &res_list( *shot.list(approx->resList()));
    const auto &tmp_list( *shot.list(approx->tempList()));
    int n = (int)std::min(res_list.size(), tmp_list.size());
    std::vector<std::pair<double,double>> pts(n);
    for(int i = 0; i < n; ++i) {
        pts[i].first  = shot[ *static_pointer_cast<XDoubleNode>(res_list.at(i))];
        pts[i].second = shot[ *static_pointer_cast<XDoubleNode>(tmp_list.at(i))];
    }
    std::sort(pts.begin(), pts.end());
    for(int i = 0; i < n; ++i) {
        trans( *static_pointer_cast<XDoubleNode>(res_list.at(i))) = pts[i].first;
        trans( *static_pointer_cast<XDoubleNode>(tmp_list.at(i))) = pts[i].second;
    }
    approx->invalidateCache();
    populateTable();
    refreshGraph();
}
void
XConCalTable::onNewClicked() {
    bool ok;
    QString name = QInputDialog::getText(m_pForm, i18n("New Thermometer"),
        i18n("Name:"), QLineEdit::Normal, QString(), &ok);
    if( !ok || name.isEmpty()) return;
    m_list->createThermometer("ApproxThermometer", name.toUtf8().data());
}
void
XConCalTable::onDeleteClicked() {
    shared_ptr<XThermometer> thermo = ***m_thermometer;
    if( !thermo) return;
    m_list->release(thermo);
}
void
XConCalTable::onDisplayTouched(const Snapshot &, XTouchableNode *) {
    drawGraph( ***thermometer());
    m_waveform->showNormal();
    m_waveform->raise();
}
