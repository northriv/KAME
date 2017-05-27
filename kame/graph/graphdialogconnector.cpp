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
#include "graphdialogconnector.h"
#include <qpixmap.h>
#include <QPushButton>
#include <qpainter.h>
#include <qtooltip.h>
#include <qtabwidget.h>
#include <qimage.h>
#include <qlayout.h>
#include <QCheckBox>
#include "ui_graphdialog.h"

XQGraphDialogConnector::XQGraphDialogConnector
(const shared_ptr<XGraph> &graph, DlgGraphSetup* item) :
    XQConnector(graph, item),
    m_pItem(item) {

    graph->iterate_commit([=](Transaction &tr){
		m_selPlot = XNode::createOrphan<XItemNode<XPlotList, XPlot> >("", true, tr, graph->plots(), true);
		m_selAxis = XNode::createOrphan<XItemNode<XAxisList, XAxis> >("", true, tr, graph->axes(), true);
    });
    //Ranges should be preset in prior to connectors.
    m_pItem->m_dblIntensity->setRange(0.0, 2.0);
    m_pItem->m_dblIntensity->setSingleStep(0.1);
    m_pItem->m_dblPersistence->setRange(0.0, 1.0);
    m_pItem->m_dblPersistence->setSingleStep(0.1);

    m_conBackGround = xqcon_create<XColorConnector>
					(graph->backGround(), m_pItem->m_clrBackGroundColor);
    m_conDrawLegends = xqcon_create<XQToggleButtonConnector>
					 (graph->drawLegends(), m_pItem->m_ckbDrawLegends);
    m_conPersistence = xqcon_create<XQDoubleSpinBoxConnector>
                     (graph->persistence(), m_pItem->m_dblPersistence, m_pItem->m_slPersistence);
    m_conPlots = xqcon_create<XQListWidgetConnector>(m_selPlot, m_pItem->lbPlots, Snapshot( *graph));
    m_conAxes = xqcon_create<XQListWidgetConnector>(m_selAxis, m_pItem->lbAxes, Snapshot( *graph));

    m_selAxis->iterate_commit([=](Transaction &tr){
	    m_lsnAxisChanged = tr[ *m_selAxis].onValueChanged().connectWeakly
	        (shared_from_this(), &XQGraphDialogConnector::onSelAxisChanged, Listener::FLAG_MAIN_THREAD_CALL);
    });
    m_selPlot->iterate_commit([=](Transaction &tr){
	    m_lsnPlotChanged = tr[ *m_selPlot].onValueChanged().connectWeakly
	        (shared_from_this(), &XQGraphDialogConnector::onSelPlotChanged, Listener::FLAG_MAIN_THREAD_CALL);
    });

    m_pItem->showNormal();
}   
XQGraphDialogConnector::~XQGraphDialogConnector() {
    if(isItemAlive()) m_pItem->close();
}
 
void
XQGraphDialogConnector::onSelAxisChanged(const Snapshot &shot, XValueNodeBase *) {
    m_conAutoScale.reset();
    m_conLogScale.reset();
    m_conDisplayTicLabels.reset();
    m_conDisplayMajorTics.reset();
    m_conDisplayMinorTics.reset();
    m_conAxisMin.reset();
    m_conAxisMax.reset();
    m_conTicLabelFormat.reset();
	shared_ptr<XAxis> axis = shot[ *m_selAxis];
	if( !axis) {
		return;
	}
	m_conAutoScale = xqcon_create<XQToggleButtonConnector>(
		axis->autoScale(), m_pItem->ckbAutoScale);
	m_conLogScale = xqcon_create<XQToggleButtonConnector>(
		axis->logScale(), m_pItem->ckbLogScale);
	m_conDisplayTicLabels = xqcon_create<XQToggleButtonConnector>
		(axis->displayTicLabels(), m_pItem->ckbDisplayTicLabels);
	m_conDisplayMajorTics = xqcon_create<XQToggleButtonConnector>
		(axis->displayMajorTics(), m_pItem->ckbDisplayMajorTics);
	m_conDisplayMinorTics = xqcon_create<XQToggleButtonConnector>
		(axis->displayMinorTics(), m_pItem->ckbDisplayMinorTics);
	m_conAxisMin = xqcon_create<XQLineEditConnector>(axis->minValue(), m_pItem->edAxisMin);
	m_conAxisMax = xqcon_create<XQLineEditConnector>(axis->maxValue(), m_pItem->edAxisMax);
	m_conTicLabelFormat = xqcon_create<XQLineEditConnector>
		(axis->ticLabelFormat(), m_pItem->edTicLabelFormat);
}
void
XQGraphDialogConnector::onSelPlotChanged(const Snapshot &shot, XValueNodeBase *) {
    m_conDrawPoints.reset();
    m_conDrawLines.reset();
    m_conDrawBars.reset();
    m_conDisplayMajorGrids.reset();
    m_conDisplayMinorGrids.reset();
    m_conMajorGridColor.reset();
    m_conMinorGridColor.reset();
    m_conPointColor.reset();
    m_conLineColor.reset();
    m_conBarColor.reset();
    m_conMaxCount.reset();
    m_conClearPoints.reset();
    m_conIntensity.reset();
    m_conColorPlot.reset();
    m_conColorPlotColorLow.reset();
    m_conColorPlotColorHigh.reset();

	shared_ptr<XPlot> plot = shot[ *m_selPlot];
	if( !plot) {
		return;
	}
	m_conDrawPoints = xqcon_create<XQToggleButtonConnector>
		(plot->drawPoints(), m_pItem->ckbDrawPoints);
	m_conDrawLines = xqcon_create<XQToggleButtonConnector>
		(plot->drawLines(), m_pItem->ckbDrawLines);
	m_conDrawBars = xqcon_create<XQToggleButtonConnector>
		(plot->drawBars(), m_pItem->ckbDrawBars);
	m_conDisplayMajorGrids = xqcon_create<XQToggleButtonConnector>
		(plot->displayMajorGrid(), m_pItem->ckbDisplayMajorGrids);
	m_conDisplayMinorGrids = xqcon_create<XQToggleButtonConnector>
		(plot->displayMinorGrid(), m_pItem->ckbDisplayMinorGrids);
    m_conMajorGridColor = xqcon_create<XColorConnector>
		(plot->majorGridColor(), m_pItem->clrMajorGridColor);
    m_conMinorGridColor = xqcon_create<XColorConnector>
		(plot->minorGridColor(), m_pItem->clrMinorGridColor);
    m_conPointColor = xqcon_create<XColorConnector>
		(plot->pointColor(), m_pItem->clrPointColor);
    m_conLineColor = xqcon_create<XColorConnector>
		(plot->lineColor(), m_pItem->clrLineColor);
    m_conBarColor = xqcon_create<XColorConnector>
		(plot->barColor(), m_pItem->clrBarColor);
	m_conMaxCount = xqcon_create<XQLineEditConnector>
		(plot->maxCount(), m_pItem->edMaxCount);
	m_conClearPoints = xqcon_create<XQButtonConnector>
		(plot->clearPoints(), m_pItem->btnClearPoints);
    m_conIntensity = xqcon_create<XQDoubleSpinBoxConnector>
        (plot->intensity(), m_pItem->m_dblIntensity, m_pItem->m_slIntensity);
	m_conColorPlot = xqcon_create<XQToggleButtonConnector>
		(plot->colorPlot(), m_pItem->ckbColorPlot);
    m_conColorPlotColorLow = xqcon_create<XColorConnector>
		(plot->colorPlotColorLow(), m_pItem->clrColorPlotLow);
    m_conColorPlotColorHigh = xqcon_create<XColorConnector>
		(plot->colorPlotColorHigh(), m_pItem->clrColorPlotHigh);
}


