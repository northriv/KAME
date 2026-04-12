/***************************************************************************
		Copyright (C) 2002-2025 Kentaro Kitagawa
		                   kitag@issp.u-tokyo.ac.jp

		This program is free software; you can redistribute it and/or
		modify it under the terms of the GNU General Public
		License as published by the Free Software Foundation; either
		version 2 of the License, or (at your option) any later version.

		You should have received a copy of the GNU General
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
#include "ui_drivercreate.h"
typedef QForm<QDialog, Ui_DlgCreateDriver> DlgCreateCalibration;

//---------------------------------------------------------------------

XConCalTable::XConCalTable
(const shared_ptr<XCalibrationCurveList> &list, FrmCalTable *form)
	:  XQConnector(list, form),
	   m_list(list),
	   m_display(XNode::createOrphan<XTouchableNode>("display") ),
	   m_temp(XNode::createOrphan<XDoubleNode>("temp") ),
	   m_value(XNode::createOrphan<XDoubleNode>("value") ),
	   m_pForm(form),
	   m_waveform(new FrmGraphNURL(g_pFrmMain, Qt::Window)),
	   m_wave(XNode::createOrphan<XWaveNGraph>("Waveform", true, m_waveform.get())) {

    list->iterate_commit([=](Transaction &tr){
        m_caltable = XNode::createOrphan<XItemNode<XCalibrationCurveList, XCalibrationCurve> >(
                         "CalTable", false, tr, list, true);
    });

    m_conCalTable = xqcon_create<XQComboBoxConnector> (m_caltable,
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
    m_caltable->iterate_commit([=](Transaction &tr){
        m_lsnThermometerChanged = tr[ *m_caltable].onValueChanged().connectWeakly(
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
	m_waveform->setWindowTitle(i18n("Calibration Table"));
    m_wave->iterate_commit([=](Transaction &tr){
		const char *labels[] = {"Output", "Raw", "Error"};
		tr[ *m_wave].setColCount(3, labels);
        if( !tr[ *m_wave].insertPlot(tr, labels[1], 0, 1)) return;
        if( !tr[ *m_wave].insertPlot(tr, labels[2], 0, -1, 2)) return;
        tr[ *m_wave->graph()->label()] = i18n("Calibration Table");
		tr[ *tr[ *m_wave].plot(0)->label()] = i18n("Curve");
		tr[ *tr[ *m_wave].plot(0)->drawPoints()] = false;
		tr[ *tr[ *m_wave].plot(1)->label()] = i18n("Error");
		tr[ *tr[ *m_wave].plot(1)->drawPoints()] = false;
		m_wave->drawGraph(tr);
		tr[ *m_wave].clearPoints();
    });
}

void
XConCalTable::onTempChanged(const Snapshot &shot, XValueNodeBase *) {
    shared_ptr<XCalibrationCurve> curve = ***calibrationTable();
	if( !curve) return;
	double ret = curve->getRaw(shot[ *temp()]);
    value()->iterate_commit([=](Transaction &tr){
		tr[ *value()] = ret;
		tr.unmark(m_lsnValue);
    });
}
void
XConCalTable::onValueChanged(const Snapshot &shot, XValueNodeBase *) {
    shared_ptr<XCalibrationCurve> curve = ***calibrationTable();
	if( !curve) return;
	double ret = curve->getOutput(shot[ *value()]);
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
    shared_ptr<XCalibrationCurve> curve = ***m_caltable;
    // Update window title and internal graph title.
    XString graphtitle = curve ? curve->getLabel() : XString(i18n("Calibration Table"));
    m_waveform->setWindowTitle(curve ?
        QString("%1 - %2").arg(i18n("Calibration Table"), curve->getLabel().c_str()) :
        i18n("Calibration Table"));
    trans( *m_wave->graph()->label()) = graphtitle;

    // Update UI labels from virtual label functions.
    XString outlabel = curve ? curve->outputLabel() : "Output";
    XString outunit  = curve ? curve->outputUnit()  : "";
    XString rawlabel = curve ? curve->rawLabel()    : "Raw";
    XString rawunit  = curve ? curve->rawUnit()     : "";
    m_pForm->lblOutputName->setText(outlabel.c_str());
    m_pForm->lblOutputUnit->setText(outunit.c_str());
    m_pForm->lblRawName->setText(rawlabel.c_str());
    m_pForm->lblRawUnit->setText(rawunit.c_str());
    QString outFull = outunit.empty() ? outlabel.c_str()
        : QString("%1 [%2]").arg(outlabel.c_str(), outunit.c_str());
    QString rawFull = rawunit.empty() ? rawlabel.c_str()
        : QString("%1 [%2]").arg(rawlabel.c_str(), rawunit.c_str());
    m_pForm->lblOutMin->setText(outFull + " Min");
    m_pForm->lblOutMax->setText(outFull + " Max");
    // Update graph axis labels (with units) and log scale. Axes may not exist yet on first call.
    // Graph X axis = output, graph Y axis = raw.
    bool log_x = curve && curve->useLogScaleOutput();
    bool log_y = curve && curve->useLogScaleRaw();
    m_wave->iterate_commit([=](Transaction &tr){
        auto fmtLabel = [](const XString &lbl, const XString &unit) -> XString {
            return unit.empty() ? lbl : XString(lbl + " [" + unit + "]");
        };
        if(auto ax = tr[ *m_wave].axisx()) {
            tr[ *ax->label()] = fmtLabel(outlabel, outunit);
            tr[ *ax->logScale()] = log_x;
        }
        if(auto ax = tr[ *m_wave].axisy()) {
            tr[ *ax->label()] = fmtLabel(rawlabel, rawunit);
            tr[ *ax->logScale()] = log_y;
        }
        if(auto ax = tr[ *m_wave].axisy2()) tr[ *ax->label()] = outunit.empty()
            ? XString("Error") : XString("Error [") + outunit + "]";
    });

    // Update table column headers.
    m_pForm->tblPoints->setHorizontalHeaderItem(0, new QTableWidgetItem(rawFull));
    m_pForm->tblPoints->setHorizontalHeaderItem(1, new QTableWidgetItem(outFull));

    auto spline = dynamic_pointer_cast<XCSplineCalibrationIF>(curve);
    m_pForm->grpPoints->setEnabled(spline != nullptr);
    m_pForm->edTMin->setEnabled(curve != nullptr);
    m_pForm->edTMax->setEnabled(curve != nullptr);

    m_pForm->edOutputLabel->setEnabled(false);
    m_pForm->edOutputUnit->setEnabled(false);
    m_pForm->edRawLabel->setEnabled(false);
    m_pForm->edRawUnit->setEnabled(false);
    if( !curve) {
        if( !m_connectedCurve.expired()) {
            m_lsnTMin.reset(); m_lsnTMax.reset();
            m_conTMin.reset(); m_conTMax.reset();
            m_lsnOutputLabel.reset(); m_lsnOutputUnit.reset();
            m_lsnRawLabel.reset();   m_lsnRawUnit.reset();
            m_conOutputLabel.reset(); m_conOutputUnit.reset();
            m_conRawLabel.reset();    m_conRawUnit.reset();
            m_connectedCurve.reset();
        }
        m_updatingTable = true;
        m_pForm->tblPoints->setRowCount(0);
        m_updatingTable = false;
        return;
    }
    if(m_connectedCurve.lock() != curve) {
        m_lsnTMin.reset(); m_lsnTMax.reset();
        m_conTMin.reset(); m_conTMax.reset();
        m_lsnOutputLabel.reset(); m_lsnOutputUnit.reset();
        m_lsnRawLabel.reset();   m_lsnRawUnit.reset();
        m_conOutputLabel.reset(); m_conOutputUnit.reset();
        m_conRawLabel.reset();    m_conRawUnit.reset();
        m_conTMin = xqcon_create<XQLineEditConnector>(curve->outMin(), m_pForm->edTMin, false);
        m_conTMax = xqcon_create<XQLineEditConnector>(curve->outMax(), m_pForm->edTMax, false);
        curve->outMin()->iterate_commit([=](Transaction &tr){
            m_lsnTMin = tr[ *curve->outMin()].onValueChanged().connectWeakly(
                shared_from_this(), &XConCalTable::onTMinMaxChanged);
        });
        curve->outMax()->iterate_commit([=](Transaction &tr){
            m_lsnTMax = tr[ *curve->outMax()].onValueChanged().connectWeakly(
                shared_from_this(), &XConCalTable::onTMinMaxChanged);
        });
        if(auto generic = dynamic_pointer_cast<XGenericCalibration>(curve)) {
            m_conOutputLabel = xqcon_create<XQLineEditConnector>(generic->outputLabelNode(), m_pForm->edOutputLabel, false);
            m_conOutputUnit  = xqcon_create<XQLineEditConnector>(generic->outputUnitNode(),  m_pForm->edOutputUnit,  false);
            m_conRawLabel    = xqcon_create<XQLineEditConnector>(generic->rawLabelNode(),    m_pForm->edRawLabel,    false);
            m_conRawUnit     = xqcon_create<XQLineEditConnector>(generic->rawUnitNode(),     m_pForm->edRawUnit,     false);
            auto connectLabel = [=](const shared_ptr<XStringNode> &node, shared_ptr<Listener> &lsn) {
                node->iterate_commit([&](Transaction &tr){
                    lsn = tr[ *node].onValueChanged().connectWeakly(
                        shared_from_this(), &XConCalTable::onLabelUnitChanged,
                        Listener::FLAG_MAIN_THREAD_CALL);
                });
            };
            connectLabel(generic->outputLabelNode(), m_lsnOutputLabel);
            connectLabel(generic->outputUnitNode(),  m_lsnOutputUnit);
            connectLabel(generic->rawLabelNode(),    m_lsnRawLabel);
            connectLabel(generic->rawUnitNode(),     m_lsnRawUnit);
        }
        m_connectedCurve = curve;
    }
    auto generic = dynamic_pointer_cast<XGenericCalibration>(curve);
    m_pForm->edOutputLabel->setEnabled(generic != nullptr);
    m_pForm->edOutputUnit->setEnabled(generic != nullptr);
    m_pForm->edRawLabel->setEnabled(generic != nullptr);
    m_pForm->edRawUnit->setEnabled(generic != nullptr);

    if( !spline) {
        m_updatingTable = true;
        m_pForm->tblPoints->setRowCount(0);
        m_updatingTable = false;
        return;
    }

    m_updatingTable = true;
    Snapshot shot( *curve);
    int n = 0;
    if(shot.size(spline->rawList()) && shot.size(spline->outputList())) {
        const auto &raw_list( *shot.list(spline->rawList()));
        const auto &out_list( *shot.list(spline->outputList()));
        n = (int)std::min(raw_list.size(), out_list.size());
        m_pForm->tblPoints->setRowCount(n);
        for(int i = 0; i < n; ++i) {
            double r = shot[ *static_pointer_cast<XDoubleNode>(raw_list.at(i))];
            double o = shot[ *static_pointer_cast<XDoubleNode>(out_list.at(i))];
            m_pForm->tblPoints->setItem(i, 0, new QTableWidgetItem(QString::number(r, 'g', 10)));
            m_pForm->tblPoints->setItem(i, 1, new QTableWidgetItem(QString::number(o, 'g', 10)));
        }
    } else {
        m_pForm->tblPoints->setRowCount(0);
    }
    m_updatingTable = false;
}
void
XConCalTable::onTableCellChanged(int row, int col) {
    if(m_updatingTable) return;
    shared_ptr<XCalibrationCurve> curve = ***m_caltable;
    auto spline = dynamic_pointer_cast<XCSplineCalibrationIF>(curve);
    if( !spline) return;
    QTableWidgetItem *item = m_pForm->tblPoints->item(row, col);
    if( !item) return;
    bool ok;
    double val = item->text().toDouble(&ok);
    if( !ok) return;
    const shared_ptr<XCSplineCalibrationIF::XDoubleListNode> &list =
        (col == 0) ? spline->rawList() : spline->outputList();
    Snapshot shot( *curve);
    if( !shot.size(list)) return;
    const auto &node_list( *shot.list(list));
    if(row >= (int)node_list.size()) return;
    trans( *static_pointer_cast<XDoubleNode>(node_list.at(row))) = val;
    spline->invalidateCache();
    if(col == 0)
        sortByRaw(spline);
    else
        refreshGraph();
}
void
XConCalTable::onAddRowClicked() {
    shared_ptr<XCalibrationCurve> curve = ***m_caltable;
    auto spline = dynamic_pointer_cast<XCSplineCalibrationIF>(curve);
    if( !spline) return;
    spline->rawList()->createByTypename("", "");
    spline->outputList()->createByTypename("", "");
    spline->invalidateCache();
    populateTable();
    refreshGraph();
}
void
XConCalTable::onRemoveRowClicked() {
    shared_ptr<XCalibrationCurve> curve = ***m_caltable;
    auto spline = dynamic_pointer_cast<XCSplineCalibrationIF>(curve);
    if( !spline) return;
    int row = m_pForm->tblPoints->currentRow();
    if(row < 0) return;
    Snapshot shot( *curve);
    if( !shot.size(spline->rawList()) || !shot.size(spline->outputList())) return;
    const auto &raw_list( *shot.list(spline->rawList()));
    const auto &out_list( *shot.list(spline->outputList()));
    if(row >= (int)raw_list.size() || row >= (int)out_list.size()) return;
    auto rnode = static_pointer_cast<XDoubleNode>(raw_list.at(row));
    auto onode = static_pointer_cast<XDoubleNode>(out_list.at(row));
    spline->rawList()->release(rnode);
    spline->outputList()->release(onode);
    spline->invalidateCache();
    populateTable();
    refreshGraph();
}
void
XConCalTable::onLabelUnitChanged(const Snapshot &, XValueNodeBase *) {
    populateTable();  // refreshes all UI labels, axis labels, column headers
}
void
XConCalTable::onTMinMaxChanged(const Snapshot &, XValueNodeBase *) {
    refreshGraph();
}
void
XConCalTable::drawGraph(const shared_ptr<XCalibrationCurve> &curve) {
    if( !curve) {
        m_wave->iterate_commit([=](Transaction &tr){ tr[ *m_wave].clearPoints(); });
        return;
    }
    const int length = 1000;
    Snapshot shot_th( *curve);
    try {
        double outmin = shot_th[ *curve->outMin()];
        double outmax = shot_th[ *curve->outMax()];
        bool lsy = curve->useLogScaleOutput(); // output axis
        if(outmax <= outmin || (lsy && outmin <= 0)) {
            m_wave->iterate_commit([=](Transaction &tr){ tr[ *m_wave].clearPoints(); });
            return;
        }
        m_wave->iterate_commit([=](Transaction &tr){
            tr[ *m_wave].setRowCount(length);
            std::vector<double> colout(length), colraw(length), colerr(length);
            for(int i = 0; i < length; ++i) {
                double t = (double)i / (length - 1);
                double o = lsy ? exp(log(outmin) + t * (log(outmax) - log(outmin)))
                               : outmin + t * (outmax - outmin);
                double r = curve->getRaw(o);
                colout[i] = o;
                colraw[i] = r;
                colerr[i] = curve->getOutput(r) - o;
            }
            tr[ *m_wave].setColumn(0, std::move(colout), 5);
            tr[ *m_wave].setColumn(1, std::move(colraw), 5);
            tr[ *m_wave].setColumn(2, std::move(colerr), 5);
            m_wave->drawGraph(tr);
        });
    }
    catch(XKameError &) {}
}
void
XConCalTable::refreshGraph() {
    if( !m_waveform->isVisible()) return;
    drawGraph( ***calibrationTable());
}
void
XConCalTable::sortByRaw(const shared_ptr<XCSplineCalibrationIF> &spline) {
    auto node = dynamic_pointer_cast<XNode>(spline);
    if( !node) return;
    Snapshot shot( *node);
    if( !shot.size(spline->rawList()) || !shot.size(spline->outputList())) return;
    const auto &raw_list( *shot.list(spline->rawList()));
    const auto &out_list( *shot.list(spline->outputList()));
    int n = (int)std::min(raw_list.size(), out_list.size());
    std::vector<std::pair<double,double>> pts(n);
    for(int i = 0; i < n; ++i) {
        pts[i].first  = shot[ *static_pointer_cast<XDoubleNode>(raw_list.at(i))];
        pts[i].second = shot[ *static_pointer_cast<XDoubleNode>(out_list.at(i))];
    }
    std::sort(pts.begin(), pts.end());
    for(int i = 0; i < n; ++i) {
        trans( *static_pointer_cast<XDoubleNode>(raw_list.at(i))) = pts[i].first;
        trans( *static_pointer_cast<XDoubleNode>(out_list.at(i))) = pts[i].second;
    }
    spline->invalidateCache();
    populateTable();
    refreshGraph();
}
void
XConCalTable::onNewClicked() {
    qshared_ptr<DlgCreateCalibration> dlg(new DlgCreateCalibration(m_pForm));
    dlg->setWindowTitle(i18n("New Calibration"));
    dlg->setModal(true);
    static int num = 0;
    dlg->m_edName->setText(QString("NewCalibration%1").arg(++num));
    dlg->m_lstType->clear();
    auto labels = m_list->typelabels();
    auto typenames = m_list->typenames();
    std::map<std::string, std::string> map; // sorts by label
    for(unsigned int i = 0; i < std::min(typenames.size(), labels.size()); ++i)
        map.insert({labels[i], typenames[i]});
    for(auto &&x: map)
        new QListWidgetItem(x.first.c_str(), dlg->m_lstType);
    dlg->m_lstType->setCurrentRow(0);
    if(dlg->exec() == QDialog::Rejected) return;
    int idx = dlg->m_lstType->currentRow();
    if(idx < 0 || idx >= (int)map.size()) return;
    if(m_list->getChild(dlg->m_edName->text().toUtf8().data())) {
        gErrPrint(i18n("Duplicated name."));
        return;
    }
    auto it = map.begin();
    std::advance(it, idx);
    m_list->createCalibration(it->second, dlg->m_edName->text().toUtf8().data());
}
void
XConCalTable::onDeleteClicked() {
    shared_ptr<XCalibrationCurve> curve = ***m_caltable;
    if( !curve) return;
    m_list->release(curve);
}
void
XConCalTable::onDisplayTouched(const Snapshot &, XTouchableNode *) {
    drawGraph( ***calibrationTable());
    m_waveform->showNormal();
    m_waveform->raise();
}
