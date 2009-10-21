/***************************************************************************
		Copyright (C) 2002-2009 Kentaro Kitagawa
		                   kitag@issp.u-tokyo.ac.jp
		
		This program is free software; you can redistribute it and/or
		modify it under the terms of the GNU Library General Public
		License as published by the Free Software Foundation; either
		version 2 of the License, or (at your option) any later version.
		
		You should have received a copy of the GNU Library General 
		Public License and a list of authors along with this program; 
		see the files COPYING and AUTHORS.
***************************************************************************/
#ifndef GRAPHDIALOGCONNECTOR_H_
#define GRAPHDIALOGCONNECTOR_H_

#include "xnodeconnector.h"
#include "graph.h"

class QDialog;
class Ui_DlgGraphSetup;
typedef QForm<QDialog, Ui_DlgGraphSetup> DlgGraphSetup;

class XQGraphDialogConnector : public XQConnector
{
    Q_OBJECT
    XQCON_OBJECT
protected:
	XQGraphDialogConnector(const shared_ptr<XGraph> &graph, DlgGraphSetup* item);
public:
	virtual ~XQGraphDialogConnector();
private:
	DlgGraphSetup *const m_pItem;
  
	shared_ptr<XItemNode<XPlotList, XPlot> > m_selPlot;
	shared_ptr<XItemNode<XAxisList, XAxis> > m_selAxis;
	shared_ptr<XListener> m_lsnAxisChanged;
	shared_ptr<XListener> m_lsnPlotChanged;

	xqcon_ptr m_conDrawLines, m_conDisplayMajorGrids,
		m_conDisplayMinorGrids, m_conDrawPoints, m_conDrawBars,
		m_conAutoScale, m_conLogScale,
		m_conDisplayMajorTics, m_conDisplayMinorTics, m_conDisplayTicLabels,
		m_conTicLabelFormat, m_conAxisMin, m_conAxisMax, m_conMaxCount,
		m_conBackGround, m_conMajorGridColor,
		m_conMinorGridColor, m_conPointColor, m_conLineColor, m_conBarColor, m_conClearPoints,
		m_conColorPlot, m_conColorPlotColorHigh, m_conColorPlotColorLow,
		m_conPlots, m_conAxes, m_conIntensity, m_conDrawLegends, m_conPersistence;  
 
	void onSelAxisChanged(const shared_ptr<XValueNodeBase> &node);
	void onSelPlotChanged(const shared_ptr<XValueNodeBase> &node);

};
#endif /*GRAPHDIALOGCONNECTOR_H_*/
