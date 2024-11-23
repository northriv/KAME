/***************************************************************************
        Copyright (C) 2002-2023 Kentaro Kitagawa
		                   kitag@issp.u-tokyo.ac.jp
		
		This program is free software; you can redistribute it and/or
		modify it under the terms of the GNU Library General Public
		License as published by the Free Software Foundation; either
		version 2 of the License, or (at your option) any later version.
		
		You should have received a copy of the GNU Library General 
		Public License and a list of authors along with this program; 
		see the files COPYING and AUTHORS.
***************************************************************************/
#include "graph.h"
#include "support.h"

#include <cmath>

// \todo Use Payload for scales.

#include "graphpainter.h"

using std::min;
using std::max;

#define GRAPH_UI_DELAY 10

#include <float.h>

#define AxisToLabel 0.09
#define AxisToTicLabel 0.015
#define PLOT_POINT_SIZE 5.0

#ifdef USE_QGLWIDGET
    #define PLOT_POINT_INTENS 0.5
    #define PLOT_LINE_INTENS 0.7
    #define PLOT_BAR_INTENS 0.4
#else
    #define PLOT_POINT_INTENS 0.7
    #define PLOT_LINE_INTENS 1.0
    #define PLOT_BAR_INTENS 0.8
#endif

XGraph::Theme XGraph::s_theme = XGraph::Theme::Night;

XGraph::XGraph(const char *name, bool runtime) : 
    XNode(name, runtime),
    m_label(create<XStringNode>("Label", true)),
    m_axes(create<XAxisList>("Axes", true)),
    m_plots(create<XPlotList>("Plots", true)),
    m_backGround(create<XHexNode>("BackGround", true)),
    m_titleColor(create<XHexNode>("TitleColor", true)),
    m_drawLegends(create<XBoolNode>("DrawLegends", true)),
    m_persistence(create<XDoubleNode>("Persistence", true)),
    m_onScreenStrings(create<XStringNode>("OnScreenStrings", true)) {

    iterate_commit([=](Transaction &tr){
		m_lsnPropertyChanged = tr[ *label()].onValueChanged().connect(*this,
																   &XGraph::onPropertyChanged);
		tr[ *backGround()].onValueChanged().connect(lsnPropertyChanged());
		tr[ *titleColor()].onValueChanged().connect(lsnPropertyChanged());
		tr[ *drawLegends()].onValueChanged().connect(lsnPropertyChanged());
		tr[ *persistence()].onValueChanged().connect(lsnPropertyChanged());
        tr[ *onScreenStrings()].onValueChanged().connect(lsnPropertyChanged());
        tr[ *drawLegends()] = true;
        tr[ *persistence()] = 0.3;

		tr[ *label()] = name;

        auto xaxis = axes()->create<XAxis>(tr, "XAxis", true, XAxis::AxisDirection::X
                               , false, tr, static_pointer_cast<XGraph>(shared_from_this()));
        if( !xaxis) return; //transaction has failed.
	    tr[ *xaxis->label()] = i18n("X Axis");
        auto yaxis = axes()->create<XAxis>(tr, "YAxis", true, XAxis::AxisDirection::Y
                               , false, tr, static_pointer_cast<XGraph>(shared_from_this()));
        if( !yaxis) return; //transaction has failed.
        tr[ *yaxis->label()] = i18n("Y Axis");

        applyTheme(tr, true);
    });
}

void
XGraph::applyTheme(Transaction &tr, bool reset_to_default, Theme theme) {
    if(theme == Theme::Current)
        theme = currentTheme();

    auto reset_or_complement = [&](const shared_ptr<XHexNode> &node, unsigned int default_color) {
        if(reset_to_default)
            tr[ *node] = default_color;
        else if(currentTheme() != theme)
            tr[ *node] = 0xffffffu - tr[ *node];
    };

    tr[ *backGround()] = (theme == Theme::Night) ?
        QColor(0x0A, 0x05, 0x45).rgb() : clWhite;
    auto textcolor = (theme == Theme::Night) ? clWhite : clBlack;
    tr[ *titleColor()] = textcolor;

    unsigned int night_colors[] = {QColor(0xff, 0xff, 0x12).rgb(), clAqua, clRed, clGreen};
    unsigned int night_point_colors[] = {QColor(0xff, 0xf1, 0x2c).rgb(), clAqua, clRed, clGreen};
    unsigned int daylight_colors[] = {clRed, clGreen, clLime, clAqua};
    unsigned int daylight_pointcolors[] = {clRed, clGreen, clLime, clAqua};
    auto barline_colors = (theme == Theme::Night) ? night_colors : daylight_colors;
    auto point_colors = (theme == Theme::Night) ? night_point_colors : daylight_pointcolors;
    unsigned int major_grid_color = (theme == Theme::Night) ?
        QColor(0x00, 0x60, 0xa0).rgb() : clAqua;
    unsigned int minor_grid_color = (theme == Theme::Night) ?
        QColor(0x60, 0x00, 0x60).rgb() : clLime;
    const Snapshot &shot(tr);
    if(shot.size(plots())) {
        const XNode::NodeList &axes_list( *shot.list(axes()));
        for(unsigned int i = 0; i < axes_list.size(); ++i) {
            auto axis = static_pointer_cast<XAxis>(axes_list[i]);
            reset_or_complement(axis->labelColor(), textcolor);
            reset_or_complement(axis->ticColor(), textcolor);
            reset_or_complement(axis->ticLabelColor(), textcolor);
        }
    }
    if(shot.size(plots())) {
        const XNode::NodeList &plots_list( *shot.list(plots()));
        for(unsigned int i = 0; i < plots_list.size(); ++i) {
            auto plot = static_pointer_cast<XPlot>(plots_list[i]);
            auto k = i % 4;
            tr[ *plot->majorGridColor()] = major_grid_color;
            tr[ *plot->minorGridColor()] = minor_grid_color;
            reset_or_complement(plot->lineColor(), barline_colors[k]);
            reset_or_complement(plot->pointColor(), point_colors[k]);
            reset_or_complement(plot->barColor(), barline_colors[k]);
            if((i > 0) && reset_to_default)
                tr[ *plot->displayMajorGrid()] = false;
        }
    }
}

void
XGraph::onPropertyChanged(const Snapshot &shot, XValueNodeBase *) {
	Snapshot shot_this( *this);
	shot_this.talk(shot_this[ *this].onUpdate(), this);
}

void
XGraph::setupRedraw(Transaction &tr, float resolution, float screenaspectratio) {
	const Snapshot &shot(tr);
	if(shot.size(axes())) {
		const XNode::NodeList &axes_list( *shot.list(axes()));
		for(auto it = axes_list.begin(); it != axes_list.end(); ++it) {
			auto axis = static_pointer_cast<XAxis>( *it);
			axis->startAutoscale(shot, resolution, shot[ *axis->autoScale()]);
		}
	}
	if(shot.size(plots())) {
		const XNode::NodeList &plots_list( *shot.list(plots()));
		for(auto it = plots_list.begin(); it != plots_list.end(); ++it) {
			auto plot = static_pointer_cast<XPlot>( *it);
			if(plot->fixScales(tr)) {
				plot->snapshot(shot);
				plot->validateAutoScale(shot);
			}
		}
	}
	if(shot.size(axes())) {
		const XNode::NodeList &axes_list( *shot.list(axes()));
		for(auto it = axes_list.begin(); it != axes_list.end(); ++it) {
			auto axis = static_pointer_cast<XAxis>( *it);
			if(shot[ *axis->autoScale()])
                axis->zoom(true, true, 1 - shot[ *axis->marginDuringAutoScale()]);
			axis->fixScale(tr, resolution, true);
		}
	}
    if(shot.size(plots())) {
        const XNode::NodeList &plots_list( *shot.list(plots()));
        for(auto it = plots_list.begin(); it != plots_list.end(); ++it) {
            auto plot = static_pointer_cast<XPlot>( *it);
            if(shot[ *plot->keepXYAspectRatioToOne()]) {
                shared_ptr<XAxis> axisx = shot[ *plot->axisX()];
                shared_ptr<XAxis> axisy = shot[ *plot->axisY()];
                double pxaspectratio_org = shot[ *axisx->length()] / shot[ *axisy->length()];
                pxaspectratio_org /= axisx->fixedMax() - axisx->fixedMin(); //todo log scale
                pxaspectratio_org *= axisy->fixedMax() - axisy->fixedMin();
                pxaspectratio_org *= screenaspectratio;
                if(pxaspectratio_org > 1.0) {
                    axisx->zoom(true, true, 1 / pxaspectratio_org);
                    axisx->fixScale(tr, resolution, true);
                }
                else {
                    axisy->zoom(true, true, pxaspectratio_org);
                    axisy->fixScale(tr, resolution, true);
                }
            }
        }
    }
}

void
XGraph::zoomAxes(Transaction &tr, float resolution,
				 XGraph::GFloat scale, const XGraph::ScrPoint &center) {
	const Snapshot &shot(tr);
	if(shot.size(axes())) {
		const XNode::NodeList &axes_list( *shot.list(axes()));
		for(auto it = axes_list.begin(); it != axes_list.end(); ++it) {
			auto axis = static_pointer_cast<XAxis>( *it);
			axis->startAutoscale(shot, resolution);
		}
	}
//   validateAutoScale();
	if(shot.size(axes())) {
		const XNode::NodeList &axes_list( *shot.list(axes()));
		for(auto it = axes_list.begin(); it != axes_list.end(); ++it) {
			auto axis = static_pointer_cast<XAxis>( *it);
			axis->zoom(true, true, scale, axis->screenToAxis(shot, center));
			axis->fixScale(tr, resolution);
		}
	}
}

XPlot::XPlot(const char *name, bool runtime, Transaction &tr_graph, const shared_ptr<XGraph> &graph)
	: XNode(name, runtime),
	  m_graph(graph),
	  m_label(create<XStringNode>("Label", true)),
	  m_maxCount(create<XUIntNode>("MaxCount", true)),
	  m_displayMajorGrid(create<XBoolNode>("DisplayMajorGrid", true)),
	  m_displayMinorGrid(create<XBoolNode>("DisplayMinorGrid", true)),
	  m_drawLines(create<XBoolNode>("DrawLines", true)),
	  m_drawBars(create<XBoolNode>("DrawBars", true)),
	  m_drawPoints(create<XBoolNode>("DrawPoints", true)),
	  m_colorPlot(create<XBoolNode>("ColorPlot", true)),
	  m_majorGridColor(create<XHexNode>("MajorGridColor", true)),
	  m_minorGridColor(create<XHexNode>("MinorGridColor", true)),
	  m_pointColor(create<XHexNode>("PointColor", true)),
	  m_lineColor(create<XHexNode>("LineColor", true)),
	  m_barColor(create<XHexNode>("BarColor", true)), //, BarInnerColor;
	  m_colorPlotColorHigh(create<XHexNode>("ColorPlotColorHigh", true)),
	  m_colorPlotColorLow(create<XHexNode>("ColorPlotColorLow", true)),
	  m_clearPoints(create<XTouchableNode>("ClearPoints", true)),
	  m_axisX(create<XItemNode<XAxisList, XAxis> >("AxisX", true, ref(tr_graph), graph->axes())),
	  m_axisY(create<XItemNode<XAxisList, XAxis> >("AxisY", true, ref(tr_graph), graph->axes())),
	  m_axisZ(create<XItemNode<XAxisList, XAxis> >("AxisZ", true, ref(tr_graph), graph->axes())),
	  m_axisW(create<XItemNode<XAxisList, XAxis> >("AxisW", true, ref(tr_graph), graph->axes())),
	  //! z value without AxisZ
	  m_zwoAxisZ(create<XDoubleNode>("ZwoAxisZ", true)),
      m_intensity(create<XDoubleNode>("Intensity", true)),
      m_keepXYAspectRatioToOne(create<XBoolNode>("KeepXYAspectRatioToOne", true)) {

    iterate_commit([=](Transaction &tr){
	//  MaxCount.value(0);
		tr[ *drawLines()] = true;
		tr[ *drawBars()] = false;
		tr[ *drawPoints()] = true;
		tr[ *displayMajorGrid()] = true;
		tr[ *displayMinorGrid()] = false;
		intensity()->setFormat("%.2f");
        tr[ *intensity()] = 0.6;
		tr[ *colorPlot()] = false;
        tr[ *majorGridColor()] = QColor(0x00, 0x60, 0xa0).rgb(); //oldstyle: clAqua
        tr[ *minorGridColor()] = QColor(0x60, 0x00, 0x60).rgb(); //oldstyle: clLime;
        tr[ *lineColor()] = QColor(0xff, 0xff, 0x12).rgb(); //oldstyle: clRed;
        tr[ *pointColor()] = QColor(0xff, 0xf1, 0x2c).rgb(); //oldstyle: clRed;
        tr[ *barColor()] = QColor(0xff, 0xff, 0x12).rgb(); //oldstyle: clRed;
        tr[ *colorPlotColorHigh()] = clRed;
		tr[ *colorPlotColorLow()] = clBlue;

		m_lsnClearPoints = tr[ *clearPoints()].onTouch().connect(
			*this, &XPlot::onClearPoints);

		tr[ *drawLines()].onValueChanged().connect(graph->lsnPropertyChanged());
		tr[ *drawBars()].onValueChanged().connect(graph->lsnPropertyChanged());
		tr[ *drawPoints()].onValueChanged().connect(graph->lsnPropertyChanged());
		tr[ *majorGridColor()].onValueChanged().connect(graph->lsnPropertyChanged());
		tr[ *minorGridColor()].onValueChanged().connect(graph->lsnPropertyChanged());
		tr[ *pointColor()].onValueChanged().connect(graph->lsnPropertyChanged());
		tr[ *lineColor()].onValueChanged().connect(graph->lsnPropertyChanged());
		tr[ *barColor()].onValueChanged().connect(graph->lsnPropertyChanged());
		tr[ *displayMajorGrid()].onValueChanged().connect(graph->lsnPropertyChanged());
		tr[ *displayMinorGrid()].onValueChanged().connect(graph->lsnPropertyChanged());
		tr[ *intensity()].onValueChanged().connect(graph->lsnPropertyChanged());
		tr[ *colorPlot()].onValueChanged().connect(graph->lsnPropertyChanged());
		tr[ *colorPlotColorHigh()].onValueChanged().connect(graph->lsnPropertyChanged());
		tr[ *colorPlotColorLow()].onValueChanged().connect(graph->lsnPropertyChanged());

		tr[ *zwoAxisZ()] = 0.15;
    });
}

bool
XPlot::fixScales(const Snapshot &shot) {
	shared_ptr<XAxis> axisx = shot[ *axisX()];
	shared_ptr<XAxis> axisy = shot[ *axisY()];
	shared_ptr<XAxis> axisz = shot[ *axisZ()];
	shared_ptr<XAxis> axisw = shot[ *axisW()];
	if( !axisx || !axisy) {
		return false;
	}
	m_curAxisX = axisx;
	m_curAxisY = axisy;
	m_curAxisZ = axisz;
	m_curAxisW = axisw;
	m_scr0.x = shot[ *m_curAxisX->x()];
	m_scr0.y = shot[ *m_curAxisY->y()];
	m_len.x = shot[ *m_curAxisX->length()];
	m_len.y = shot[ *m_curAxisY->length()];
	if( !m_curAxisZ) {
		m_scr0.z = shot[ *zwoAxisZ()];
		m_len.z = (XGraph::SFloat)0.0;
	}
	else {
		m_scr0.z = shot[ *m_curAxisZ->z()];
		m_len.z = shot[ *m_curAxisZ->length()];
	}
	return true;
}

void
XPlot::screenToGraph(const Snapshot &shot, const XGraph::ScrPoint &pt, XGraph::GPoint *g) const {
	shared_ptr<XAxis> axisx = shot[ *axisX()];
	shared_ptr<XAxis> axisy = shot[ *axisY()];
	shared_ptr<XAxis> axisz = shot[ *axisZ()];
    if( !axisz)
        g->z = (XGraph::GFloat)0.0;
    else
        g->z = (pt.z - shot[ *axisz->z()]) / shot[ *axisz->length()];
    g->x = (pt.x - shot[ *axisx->x()]) / shot[ *axisx->length()];
    g->y = (pt.y - shot[ *axisy->y()]) / shot[ *axisy->length()];
}
void
XPlot::graphToVal(const Snapshot &shot, const XGraph::GPoint &pt, XGraph::ValPoint *val) {
	if(fixScales(shot)) {
		val->x = m_curAxisX->axisToVal(pt.x);
		val->y = m_curAxisY->axisToVal(pt.y);
		if(m_curAxisZ)
			val->z = m_curAxisZ->axisToVal(pt.z);
		else
			val->z = (XGraph::VFloat)0.0;
    }
}
int
XPlot::screenToVal(const Snapshot &shot, const XGraph::ScrPoint &scr, XGraph::ValPoint *val, XGraph::SFloat scr_prec) {
	if(fixScales(shot)) {
		val->x = m_curAxisX->axisToVal(m_curAxisX->screenToAxis(shot, scr), scr_prec / m_len.x);
		val->y = m_curAxisY->axisToVal(m_curAxisY->screenToAxis(shot, scr), scr_prec / m_len.y);
		val->z = (m_curAxisZ) ? m_curAxisZ->axisToVal(
			m_curAxisZ->screenToAxis(shot, scr), scr_prec / m_len.z) : (XGraph::GFloat)0.0;
		return 0;
    }
    return -1;
}
void
XPlot::graphToScreen(const Snapshot &shot, const XGraph::GPoint &pt, XGraph::ScrPoint *scr) {
	if(fixScales(shot)) {
		graphToScreenFast(pt, scr);
    }
}
void
XPlot::graphToScreenFast(const XGraph::GPoint &pt, XGraph::ScrPoint *scr) const {
	scr->x = m_scr0.x + m_len.x * pt.x;
	scr->y = m_scr0.y + m_len.y * pt.y;
	scr->z = m_scr0.z + m_len.z * pt.z;
	scr->w = std::min(std::max(pt.w, (XGraph::GFloat)0.0), (XGraph::GFloat)1.0);
}
void
XPlot::valToGraphFast(const XGraph::ValPoint &pt, XGraph::GPoint *gr) const {
	gr->x = m_curAxisX->valToAxis(pt.x);
	gr->y = m_curAxisY->valToAxis(pt.y);
	if(m_curAxisZ)
		gr->z = m_curAxisZ->valToAxis(pt.z);
	else
		gr->z = 0.0;
	if(m_curAxisW)
		gr->w = m_curAxisW->valToAxis(pt.w);
	else
		gr->w = pt.w;
}
int
XPlot::findPoint(const Snapshot &shot, int start, const XGraph::GPoint &gmin, const XGraph::GPoint &gmax,
				 XGraph::GFloat width, XGraph::ValPoint *val, XGraph::GPoint *g1) {
	if(fixScales(shot)) {
		for(unsigned int i = start; i < m_ptsSnapped.size(); i++) {
            XGraph::ValPoint v;
            XGraph::GPoint g2;
			v = m_ptsSnapped[i];
			valToGraphFast(v, &g2);
			if(g2.distance2(gmin, gmax) < width*width) {
				*val = v;
				*g1 = g2;
				return i;
			}
		}
    }
    return -1;
}

void
XPlot::onClearPoints(const Snapshot &, XTouchableNode *) {
    m_graph.lock()->iterate_commit([=](Transaction &tr){
		clearAllPoints(tr);
    });
}

void
XPlot::drawGrid(const Snapshot &shot, XQGraphPainter *painter, shared_ptr<XAxis> &axis1, shared_ptr<XAxis> &axis2) {
    int len = std::lrint(1.0/painter->resScreen());
	painter->beginLine(1.0);
	bool disp_major = shot[ *displayMajorGrid()];
	bool disp_minor = shot[ *displayMinorGrid()];
	if(disp_major || disp_minor) {
		XGraph::ScrPoint s1, s2;
		unsigned int major_color = shot[ *majorGridColor()];
		unsigned int minor_color = shot[ *minorGridColor()];
		double intens = shot[ *intensity()];
		for(int i = 0; i < len; i++) {
			XGraph::GFloat x = (XGraph::GFloat)i/len;
			graphToScreenFast(XGraph::GPoint((axis1 == m_curAxisX) ? x : 0.0,
											 (axis1 == m_curAxisY) ? x : 0.0, (axis1 == m_curAxisZ) ? x : 0.0), &s1);
			graphToScreenFast(XGraph::GPoint(
								  (axis1 == m_curAxisX) ? x : ((axis2 == m_curAxisX) ? 1.0 : 0.0),
								  (axis1 == m_curAxisY) ? x : ((axis2 == m_curAxisY) ? 1.0 : 0.0),
								  (axis1 == m_curAxisZ) ? x : ((axis2 == m_curAxisZ) ? 1.0 : 0.0)),
							  &s2);
			XGraph::VFloat tempx;
			switch(axis1->queryTic(len, i, &tempx)) {
            case XAxis::Tic::Major:
				if(disp_major) {
					painter->setColor(major_color,
									  max(0.0, min(intens * 0.7, 0.5)) );
					painter->setVertex(s1);
					painter->setVertex(s2);
				}
				break;
            case XAxis::Tic::Minor:
				if(disp_minor) {
					painter->setColor(minor_color,
									  max(0.0, min(intens * 0.5, 0.5)) );
					painter->setVertex(s1);
					painter->setVertex(s2);
				}
				break;
			default:
				break;
			}
		}
	}
    painter->endLine();
}
void
XPlot::drawGrid(const Snapshot &shot, XQGraphPainter *painter, bool drawzaxis) {
    if(fixScales(shot)) {
		drawGrid(shot, painter, m_curAxisX, m_curAxisY);
		drawGrid(shot, painter, m_curAxisY, m_curAxisX);
		if(m_curAxisZ && drawzaxis) {
            drawGrid(shot, painter, m_curAxisX, m_curAxisZ);
			drawGrid(shot, painter, m_curAxisZ, m_curAxisX);
            drawGrid(shot, painter, m_curAxisY, m_curAxisZ);
			drawGrid(shot, painter, m_curAxisZ, m_curAxisY);
		}
    }
}
int
XPlot::drawLegend(const Snapshot &shot, XQGraphPainter *painter, const XGraph::ScrPoint &spt, float dx, float dy) {
    if(fixScales(shot)) {
		bool colorplot = shot[ *colorPlot()];
		bool hasweight = !!m_curAxisW;
		unsigned int colorhigh = shot[ *colorPlotColorHigh()];
		unsigned int colorlow = shot[ *colorPlotColorLow()];
		float alpha1 = hasweight ? 0.2 : 1.0;
		float alpha2 = 1.0;
		if(shot[ *drawBars()]) {
			float alpha = max(0.0f, min((float)(shot[ *intensity()] * PLOT_BAR_INTENS), 1.0f));
			painter->beginLine(1.0);
			XGraph::ScrPoint s1, s2;
			s1 = spt;
            s1 += XGraph::ScrPoint(0, dy/2, 0);
			s2 = spt;
            s2 -= XGraph::ScrPoint(0, dy/2, 0);
			painter->setColor(shot[ *barColor()], alpha1*alpha);
			if(colorplot) painter->setColor(colorhigh, alpha1*alpha); 
			painter->setVertex(s1);
			painter->setColor(shot[ *barColor()], alpha2*alpha);
			if(colorplot) painter->setColor(colorlow, alpha2*alpha);
			painter->setVertex(s2);
			painter->endLine();
		}
		if(shot[ *drawLines()]) {
			float alpha = max(0.0f, min((float)(shot[ *intensity()] * PLOT_LINE_INTENS), 1.0f));
			painter->beginLine(1.0);
			XGraph::ScrPoint s1, s2;
			s1 = spt;
            s1 += XGraph::ScrPoint(dx/2, 0, 0);
			s2 = spt;
            s2 -= XGraph::ScrPoint(dx/2, 0, 0);
			painter->setColor(shot[ *lineColor()], alpha1*alpha);
			if(colorplot) painter->setColor(colorhigh, alpha1*alpha); 
			painter->setVertex(s1);
			painter->setColor(shot[ *lineColor()], alpha2*alpha);
			if(colorplot) painter->setColor(colorlow, alpha2*alpha);
			painter->setVertex(s2);
			painter->endLine();
		}
		if(shot[ *drawPoints()]) {
			float alpha = max(0.0f, min((float)(shot[ *intensity()] * PLOT_POINT_INTENS), 1.0f));
			painter->setColor(shot[ *pointColor()], alpha );
			painter->beginPoint(PLOT_POINT_SIZE);
			if(colorplot)
				painter->setColor(colorhigh);
			painter->setVertex(spt);
			if(colorplot) {
				painter->setColor(colorhigh);
				painter->setVertex(spt);
			}
			painter->endPoint();
		}
    	return 0;
    }
    return -1;
}

int
XPlot::drawPlot(const Snapshot &shot, XQGraphPainter *painter) {
    if(fixScales(shot)) {
		bool colorplot = shot[ *colorPlot()];
		bool hasweight = !!m_curAxisW;
		int cnt = m_ptsSnapped.size();
		m_canvasPtsSnapped.resize(cnt);
		tCanvasPoint *cpt;
		{
			XGraph::ScrPoint s1;
			XGraph::GPoint g1;
			unsigned int colorhigh = shot[ *colorPlotColorHigh()];
			unsigned int colorlow = shot[ *colorPlotColorLow()];
			unsigned int linecolor = shot[ *lineColor()];
			cpt = &m_canvasPtsSnapped[0];
			for(int i = 0; i < cnt; ++i) {
				XGraph::ValPoint pt = m_ptsSnapped[i];
				valToGraphFast(pt, &g1);
				graphToScreenFast(g1, &s1);
				cpt->scr = s1;
				cpt->graph = g1;
				cpt->insidecube = isPtIncluded(g1);
				if(colorplot) {
					cpt->color = blendColor(colorlow, colorhigh, 
											hasweight ? cpt->graph.w : cpt->graph.z);
				}
				else {
					cpt->color = linecolor;
				}
				cpt++;
			}
		}
		if(shot[ *drawBars()]) {
			XGraph::ScrPoint s1, s2;
			tCanvasPoint pt2;
			double g0y = m_curAxisY->valToAxis(0.0);
			m_curAxisY->axisToScreen(shot, g0y, &s2);
			double s0y = s2.y;
			float alpha = max(0.0f, min((float)(shot[ *intensity()] * PLOT_BAR_INTENS), 1.0f));
			painter->setColor(shot[ *barColor()], alpha );
			painter->beginLine(1.0);
			cpt = &m_canvasPtsSnapped[0];
			for(int i = 0; i < cnt; ++i) {
				pt2 = *cpt;
				pt2.graph.y = g0y;
				pt2.scr.y = s0y;
				pt2.insidecube = isPtIncluded(pt2.graph);
				if(clipLine( *cpt, pt2, &s1, &s2, false, NULL, NULL, NULL, NULL)) {
					if(colorplot || hasweight) painter->setColor(cpt->color, alpha * cpt->scr.w); 
					painter->setVertex(s1);
					painter->setVertex(s2);
				}
				cpt++;
			}
			painter->endLine();
		}
		if(shot[ *drawLines()]) {
			float alpha = max(0.0f, min((float)(shot[ *intensity()] * PLOT_LINE_INTENS), 1.0f));
			painter->setColor(shot[ *lineColor()], alpha );
			painter->beginLine(1.0);
			XGraph::ScrPoint s1, s2;
			cpt = &m_canvasPtsSnapped[0];
			unsigned int color1, color2;
			float alpha1, alpha2;
			for(int i = 1; i < cnt; ++i) {
				if(clipLine( *cpt, *(cpt + 1), &s1, &s2, colorplot || hasweight,
					&color1, &color2, &alpha1, &alpha2)) {
					if(colorplot || hasweight) painter->setColor(color1, alpha*alpha1);
					painter->setVertex(s1);
					if(colorplot || hasweight) painter->setColor(color2, alpha*alpha2);
					painter->setVertex(s2);
				}
				cpt++;
			}
			painter->endLine();
		}
		if(shot[ *drawPoints()]) {
			float alpha = max(0.0f, min((float)(shot[ *intensity()] * PLOT_POINT_INTENS), 1.0f));
			painter->setColor(shot[ *pointColor()], alpha );
			painter->beginPoint(PLOT_POINT_SIZE);
			unsigned int pointcolor = shot[ *pointColor()];
			cpt = &m_canvasPtsSnapped[0];
			for(int i = 0; i < cnt; ++i) {
				if(cpt->insidecube) {
					if(colorplot)
						painter->setColor(cpt->color, alpha * cpt->scr.w);
					else
						if(hasweight)
							painter->setColor(pointcolor, alpha * cpt->scr.w);
					painter->setVertex(cpt->scr);
				}
				cpt++;
			}
			painter->endPoint();
		}
		return 0;
	}
	return -1;
}

inline unsigned int
XPlot::blendColor(unsigned int c1, unsigned int c2, float t) const {
    unsigned char c1red = qRed((QRgb)c1);
    unsigned char c1green = qGreen((QRgb)c1);
    unsigned char c1blue = qBlue((QRgb)c1);
    unsigned char c2red = qRed((QRgb)c2);
    unsigned char c2green = qGreen((QRgb)c2);
    unsigned char c2blue = qBlue((QRgb)c2);
    return (unsigned int)qRgb(
        lrintf(c2red * t + c1red * (1.0f - t)),
        lrintf(c2green * t + c1green * (1.0f - t)),
        lrintf(c2blue * t + c1blue * (1.0f - t)));
}

inline bool
XPlot::isPtIncluded(const XGraph::GPoint &pt) const {
	return (pt.x >= 0) && (pt.x <= 1) &&
		(pt.y >= 0) && (pt.y <= 1) &&
		(pt.z >= 0) && (pt.z <= 1);
}

inline bool
XPlot::clipLine(const tCanvasPoint &c1, const tCanvasPoint &c2,
				XGraph::ScrPoint *s1, XGraph::ScrPoint *s2, bool blendcolor,
                unsigned int *color1, unsigned int *color2, float *alpha1, float *alpha2) const {
	if(c1.insidecube && c2.insidecube) {
		*s1 = c1.scr; *s2 = c2.scr;
		if(blendcolor) {
			*color1 = c1.color;
			*color2 = c2.color;
			*alpha1 = c1.scr.w;
			*alpha2 = c1.scr.w;
		}
		return true;
	}
	XGraph::GPoint g1 = c1.graph;
	XGraph::GPoint g2 = c2.graph;
  
	XGraph::GFloat idx = 1.0 / (g1.x - g2.x);
	XGraph::GFloat tx0 = -g2.x * idx;
	XGraph::GFloat tx1 = (1.0 - g2.x) * idx;
	XGraph::GFloat txmin = min(tx0, tx1);
	XGraph::GFloat txmax = max(tx0, tx1);
  
	XGraph::GFloat idy = 1.0 / (g1.y - g2.y);
	XGraph::GFloat ty0 = -g2.y * idy;
	XGraph::GFloat ty1 = (1.0 - g2.y) * idy;
	XGraph::GFloat tymin = min(ty0, ty1);
	XGraph::GFloat tymax = max(ty0, ty1);
  
	XGraph::GFloat tmin = max(txmin, tymin);
	XGraph::GFloat tmax = min(txmax, tymax);
  
	if(m_curAxisZ) {
		if(tmin >= tmax) return false;
		XGraph::GFloat idz = 1.0 / (g1.z - g2.z);
		XGraph::GFloat tz0 = -g2.z * idz;
		XGraph::GFloat tz1 = (1.0 - g2.z) * idz;
		XGraph::GFloat tzmin = min(tz0, tz1);
		XGraph::GFloat tzmax = max(tz0, tz1);
		tmin = max(tmin, tzmin);
		tmax = min(tmax, tzmax);
	}  

	if(tmin >= tmax) return false;
	if(tmin >= 1.0) return false;
	if(tmax <= 0.0) return false;
   
	if(tmin > 0.0) {
		graphToScreenFast(XGraph::GPoint(
							  tmin * g1.x + (1.0 - tmin) * g2.x,
							  tmin * g1.y + (1.0 - tmin) * g2.y,
							  tmin * g1.z + (1.0 - tmin) * g2.z)
						  , s2);
		if(blendcolor) {
			*color2 = blendColor(c2.color, c1.color, tmin);
			*alpha2 = tmin * c1.scr.w + (1.0 - tmin) * c2.scr.w;
		}
	}
	else {
		*s2 = c2.scr;
		if(blendcolor) {
			*color2 = c2.color;
			*alpha2 = c2.scr.w;
		}
	}
	if(tmax < 1.0) {
		graphToScreenFast(XGraph::GPoint(
							  tmax * g1.x + (1.0 - tmax) * g2.x,
							  tmax * g1.y + (1.0 - tmax) * g2.y,
							  tmax * g1.z + (1.0 - tmax) * g2.z)
						  , s1);
		if(blendcolor) {
			*color2 = blendColor(c2.color, c1.color, tmax);
			*alpha2 = tmax * c1.scr.w + (1.0 - tmax) * c2.scr.w;
		}
	}
	else {
		*s1 = c1.scr;
		if(blendcolor) {
			*color1 = c1.color;
			*alpha1 = c1.scr.w;
		}
	}
	return true;
}

int
XPlot::validateAutoScale(const Snapshot &shot) {
	bool autoscale_x = shot[ *m_curAxisX->autoScale()];
	bool autoscale_y = shot[ *m_curAxisY->autoScale()];
	bool autoscale_z = m_curAxisZ ? shot[ *m_curAxisZ->autoScale()] : false;
	bool autoscale_w = m_curAxisW ? shot[ *m_curAxisW->autoScale()] : false;
	for(unsigned int i = 0; i < m_ptsSnapped.size(); ++i) {
		XGraph::ValPoint &pt = m_ptsSnapped[i];
		bool included = true;
		included = included && (autoscale_x || m_curAxisX->isIncluded(pt.x));
	    included = included && (autoscale_y || m_curAxisY->isIncluded(pt.y));
	    if(m_curAxisZ)
	        included = included && (autoscale_z || m_curAxisZ->isIncluded(pt.z));
	    if(m_curAxisW)
	        included = included && (autoscale_w || m_curAxisW->isIncluded(pt.w));
	    else
	        included = included && (pt.w > (XGraph::VFloat)1e-20);
	    if(included) {
	        m_curAxisX->tryInclude(pt.x);
	        m_curAxisY->tryInclude(pt.y);
	        if(m_curAxisZ)
	            m_curAxisZ->tryInclude(pt.z);
	        if(m_curAxisW)
	            m_curAxisW->tryInclude(pt.w);
	    }
	}
	return 0;
}

void
XXYPlot::clearAllPoints(Transaction &tr) {
    const Snapshot &shot(tr);
    tr[ *this].points().clear();
//    tr[ *this].points().shrink_to_fit();
//    tr[ *this].points().reserve(shot[ *maxCount()]);
    tr[ *this].m_startPos = 0;
    shared_ptr<XGraph> graph(m_graph.lock());
	tr.mark(shot[ *graph].onUpdate(), graph.get());
}

void
XXYPlot::snapshot(const Snapshot &shot) {
    const auto &points(shot[ *this].points());
    int offset = shot[ *this].m_startPos;
    int cnt = (int)points.size();
	m_ptsSnapped.resize(cnt);
    int j = offset;
    for(int i = 0; i < cnt; ++i) {
        m_ptsSnapped[i] = points[j % cnt];
        ++j;
	}
}
void
XXYPlot::addPoint(Transaction &tr,
	XGraph::VFloat x, XGraph::VFloat y, XGraph::VFloat z, XGraph::VFloat w) {
	XGraph::ValPoint npt(x, y, z, w);

	shared_ptr<XGraph> graph(m_graph.lock());

    const Snapshot &shot(tr);
    auto &points(tr[ *this].points());
    unsigned int offset = shot[ *this].m_startPos;
    unsigned int maxcount = shot[ *maxCount()];
    if((offset && (points.size() < maxcount)) || (points.size() > maxcount)) {
    //maxcount has changed. reorders.
        auto buf = points;
        int j = offset;
        if(points.size() > maxcount) {
        //maxcount has decreased.
            j += points.size() - maxcount;
            j = j % points.size();
            points.resize(maxcount);
        }
        for(int i = 0; i < points.size(); ++i) {
            points[i] = buf[j % buf.size()];
            ++j;
        }
        offset = 0;
        tr[ *this].m_startPos = 0;
    }


    if(points.size() == maxcount) {
        unsigned int startpos = offset + 1;
        if(startpos >= points.size())
            startpos = 0;
        tr[ *this].m_startPos = startpos;
        if(points.size())
            points[offset] = npt;
    }
    else {
        points.push_back(npt);
    }
	tr.mark(shot[ *graph].onUpdate(), graph.get());
}

XAxis::XAxis(const char *name, bool runtime,
			 AxisDirection dir, bool rightOrTop, Transaction &tr_graph, const shared_ptr<XGraph> &graph) :
	XNode(name, runtime),
	m_direction(dir),
    m_dirVector( (dir == AxisDirection::X) ? 1.0 : 0.0,
                 (dir == AxisDirection::Y) ? 1.0 : 0.0,
                 (dir == AxisDirection::Z) ? 1.0 : 0.0),
	m_graph(graph),
	m_label(create<XStringNode>("Label", true)),
	m_x(create<XDoubleNode>("X", true)),
	m_y(create<XDoubleNode>("Y", true)),
	m_z(create<XDoubleNode>("Z", true)),
	m_length(create<XDoubleNode>("Length", true)), // in screen coordinate
	m_majorTicScale(create<XDoubleNode>("MajorTicScale", true)),
	m_minorTicScale(create<XDoubleNode>("MinorTicScale", true)),
	m_displayMajorTics(create<XBoolNode>("DisplayMajorTics", true)),
	m_displayMinorTics(create<XBoolNode>("DisplayMinorTics", true)),
	m_max(create<XDoubleNode>("Max", true)),
	m_min(create<XDoubleNode>("Min", true)),
	m_rightOrTopSided(create<XBoolNode>("RightOrTopSided", true)), //sit on right, top
    m_invertAxis(create<XBoolNode>("InvertAxis", true)),
    m_invisible(create<XBoolNode>("Invisible", true)),
    m_ticLabelFormat(create<XStringNode>("TicLabelFormat", true)),
	m_displayLabel(create<XBoolNode>("DisplayLabel", true)),
	m_displayTicLabels(create<XBoolNode>("DisplayTicLabels", true)),
	m_ticColor(create<XHexNode>("TicColor", true)),
	m_labelColor(create<XHexNode>("LabelColor", true)),
	m_ticLabelColor(create<XHexNode>("TicLabelColor", true)),
	m_autoFreq(create<XBoolNode>("AutoFreq", true)),
	m_autoScale(create<XBoolNode>("AutoScale", true)),
    m_logScale(create<XBoolNode>("LogScale", true)),
    m_marginDuringAutoScale(create<XDoubleNode>("MarginDuringAutoScale", true)) {
    m_ticLabelFormat->setValidator(&formatDoubleValidator);
  
    iterate_commit([=](Transaction &tr){
		tr[ *x()] = 0.15;
		tr[ *y()] = 0.15;
		tr[ *z()] = 0.15;
		tr[ *length()] = 0.7;
		tr[ *maxValue()] = 0;
		tr[ *minValue()] = 0;
		tr[ *ticLabelFormat()] = "";
		tr[ *logScale()] = false;
        tr[ *marginDuringAutoScale()] = 0.04;
		tr[ *displayLabel()] = true;
		tr[ *displayTicLabels()] = true;
		tr[ *displayMajorTics()] = true;
		tr[ *displayMinorTics()] = true;
		tr[ *autoFreq()] = true;
		tr[ *autoScale()] = true;
		tr[ *rightOrTopSided()] = rightOrTop;
		tr[ *majorTicScale()] = 10;
		tr[ *minorTicScale()] = 1;
        tr[ *labelColor()] = clWhite; //oldstyle: clBlack;
        tr[ *ticColor()] = clWhite; //oldstyle: clBlack;
        tr[ *ticLabelColor()] = clWhite; //oldstyle: clBlack;

		if(rightOrTop) {
            if(dir == AxisDirection::Y) tr[ *x()] = 1.0- tr[ *x()];
            if(dir == AxisDirection::X) tr[ *y()] = 1.0- tr[ *y()];
		}

		startAutoscale_(tr, true);

		tr[ *maxValue()].onValueChanged().connect(graph->lsnPropertyChanged());
		tr[ *minValue()].onValueChanged().connect(graph->lsnPropertyChanged());
		tr[ *label()].onValueChanged().connect(graph->lsnPropertyChanged());
		tr[ *logScale()].onValueChanged().connect(graph->lsnPropertyChanged());
		tr[ *autoScale()].onValueChanged().connect(graph->lsnPropertyChanged());
		tr[ *x()].onValueChanged().connect(graph->lsnPropertyChanged());
		tr[ *y()].onValueChanged().connect(graph->lsnPropertyChanged());
		tr[ *z()].onValueChanged().connect(graph->lsnPropertyChanged());
		tr[ *length()].onValueChanged().connect(graph->lsnPropertyChanged());
		tr[ *ticLabelFormat()].onValueChanged().connect(graph->lsnPropertyChanged());
		tr[ *displayLabel()].onValueChanged().connect(graph->lsnPropertyChanged());
		tr[ *displayTicLabels()].onValueChanged().connect(graph->lsnPropertyChanged());
		tr[ *displayMajorTics()].onValueChanged().connect(graph->lsnPropertyChanged());
		tr[ *displayMinorTics()].onValueChanged().connect(graph->lsnPropertyChanged());
		tr[ *autoFreq()].onValueChanged().connect(graph->lsnPropertyChanged());
		tr[ *rightOrTopSided()].onValueChanged().connect(graph->lsnPropertyChanged());
		tr[ *majorTicScale()].onValueChanged().connect(graph->lsnPropertyChanged());
		tr[ *minorTicScale()].onValueChanged().connect(graph->lsnPropertyChanged());
		tr[ *labelColor()].onValueChanged().connect(graph->lsnPropertyChanged());
		tr[ *ticColor()].onValueChanged().connect(graph->lsnPropertyChanged());
		tr[ *ticLabelColor()].onValueChanged().connect(graph->lsnPropertyChanged());
    });
}


void
XAxis::startAutoscale_(const Snapshot &shot, bool clearscale) {
	m_bLogscaleFixed = shot[ *logScale()];
	m_bAutoscaleFixed = shot[ *autoScale()];
	if(clearscale) {
        m_minFixed = std::numeric_limits<XGraph::VFloat>::max();
        m_maxFixed = m_bLogscaleFixed ? 0 : std::numeric_limits<XGraph::VFloat>::lowest();
	}
	else {
		m_minFixed = m_bLogscaleFixed ?
			max((XGraph::VFloat)shot[ *minValue()], (XGraph::VFloat)0.0) :
			(XGraph::VFloat)shot[ *minValue()];
		m_maxFixed = shot[ *maxValue()];
	}
	m_invMaxMinusMinFixed = -1; //undef
	m_invLogMaxOverMinFixed = -1; //undef
}
void
XAxis::startAutoscale(const Snapshot &shot, float, bool clearscale) {
    startAutoscale_(shot, clearscale);
}
void
XAxis::fixScale(Transaction &tr, float resolution, bool suppressupdate) {
	const Snapshot &shot(tr);
	shared_ptr<XGraph> graph(m_graph.lock());
    if(m_minFixed == m_maxFixed) {
		XGraph::VFloat x = m_minFixed;
        m_maxFixed = x ? std::max(x * 1.01, x * 0.99) : 0.01;
        m_minFixed = x ? std::min(x * 1.01, x * 0.99) : -0.01;
    }
    XGraph::VFloat min_tmp = m_bLogscaleFixed ? 
        max((XGraph::VFloat)shot[ *minValue()], (XGraph::VFloat)0.0) :
        (XGraph::VFloat)shot[ *minValue()];
    if(m_minFixed != min_tmp) {
        minValue()->setFormat(shot[ *ticLabelFormat()].to_str().c_str());
        tr[ *minValue()] = m_minFixed;
    }
    if(m_maxFixed != shot[ *maxValue()]) {
        maxValue()->setFormat(shot[ *ticLabelFormat()].to_str().c_str());
        tr[ *maxValue()] = m_maxFixed;
    }
    if(suppressupdate) {
        tr.unmark(graph->lsnPropertyChanged());
    }
    performAutoFreq(shot, resolution);
    m_bInverted = shot[ *m_invertAxis];
}
void
XAxis::performAutoFreq(const Snapshot &shot, float resolution) {
	if(shot[ *autoFreq()] &&
	   ( !m_bLogscaleFixed || (m_minFixed >= 0)) &&
	   (m_minFixed < m_maxFixed)) {
        float fac = max(0.8f, std::log10(2e-3F / resolution) );
        m_majorFixed = (std::pow((XGraph::VFloat)10.0,
            std::rint(std::log10(m_maxFixed - m_minFixed) - fac)));
		m_minorFixed = m_majorFixed / (XGraph::VFloat)2.0;
	}
	else {
		m_majorFixed = shot[ *majorTicScale()];
		m_minorFixed = shot[ *minorTicScale()];
	}
}

inline bool
XAxis::isIncluded(XGraph::VFloat x) const {
	return (x >= m_minFixed)
        && ((m_direction == AxisDirection::Weight) || (x <= m_maxFixed));
}
inline void
XAxis::tryInclude(XGraph::VFloat x) {
	//omits negative values in log scaling
	if( !(m_bLogscaleFixed && (x <= 0))) {
		if(m_bAutoscaleFixed) {
			if(x > m_maxFixed) {
				m_maxFixed = x;
				m_invMaxMinusMinFixed = -1; //undef
				m_invLogMaxOverMinFixed = -1; //undef
			}         
			if(x < m_minFixed) {
				m_minFixed = x;
				m_invMaxMinusMinFixed = -1; //undef
				m_invLogMaxOverMinFixed = -1; //undef
			}
		}
	}
}

void
XAxis::zoom(bool minchange, bool maxchange, XGraph::GFloat prop, XGraph::GFloat center) {
    if(direction() == AxisDirection::Weight) return;
	
	if(maxchange) {
        m_maxFixed = axisToVal(center + (XGraph::GFloat)0.5 / prop * (m_bInverted ? -1 : 1));
	}
	if(minchange) {
        m_minFixed = axisToVal(center - (XGraph::GFloat)0.5 / prop * (m_bInverted ? -1 : 1));
	}
	m_invMaxMinusMinFixed = -1; //undef
	m_invLogMaxOverMinFixed = -1; //undef
}

XGraph::GFloat
XAxis::valToAxis(XGraph::VFloat x) {
	XGraph::GFloat pos;
	if(m_bLogscaleFixed) {
		if ((x <= 0) || (m_minFixed <= 0) || (m_maxFixed <= m_minFixed))
            return std::numeric_limits<XGraph::GFloat>::lowest();
		if(m_invLogMaxOverMinFixed < 0)
            m_invLogMaxOverMinFixed = 1 / std::log(m_maxFixed / m_minFixed);
        pos = std::log(x / m_minFixed) * m_invLogMaxOverMinFixed;
	}
	else {
		if(m_maxFixed <= m_minFixed) return -1;
		if(m_invMaxMinusMinFixed < 0)
			m_invMaxMinusMinFixed = 1 / (m_maxFixed - m_minFixed);
		pos = (x - m_minFixed) * m_invMaxMinusMinFixed;
	}
    if(m_bInverted) pos = 1.0f - pos;
	return pos;
}

XGraph::VFloat
XAxis::axisToVal(XGraph::GFloat pos, XGraph::GFloat axis_prec) const {
	XGraph::VFloat x = 0;
    if(m_bInverted) pos = 1.0f - pos;
    if(axis_prec <= 0) {
		if(m_bLogscaleFixed) {
			if((m_minFixed <= 0) || (m_maxFixed < m_minFixed)) return 0;
            x = m_minFixed * std::exp(std::log(m_maxFixed / m_minFixed) * pos);
		}
		else {
			if(m_maxFixed < m_minFixed) return 0;
			x = m_minFixed + pos *(m_maxFixed - m_minFixed);
		}
		return x;
	}
	else {
		x = axisToVal(pos);
		XGraph::VFloat dx = axisToVal(pos + axis_prec) - x;
		return setprec(x, dx);
    }
}

void
XAxis::axisToScreen(const Snapshot &shot, XGraph::GFloat pos, XGraph::ScrPoint *scr) const {
	XGraph::SFloat len = shot[ *length()];
	pos *= len;
    scr->x = shot[ *x()] + ((m_direction == AxisDirection::X) ? pos: (XGraph::SFloat) 0.0);
    scr->y = shot[ *y()] + ((m_direction == AxisDirection::Y) ? pos: (XGraph::SFloat) 0.0);
    scr->z = shot[ *z()] + ((m_direction == AxisDirection::Z) ? pos: (XGraph::SFloat) 0.0);
}
XGraph::GFloat
XAxis::screenToAxis(const Snapshot &shot, const XGraph::ScrPoint &scr) const {
	XGraph::SFloat _x = scr.x - shot[ *x()];
	XGraph::SFloat _y = scr.y - shot[ *y()];
	XGraph::SFloat _z = scr.z - shot[ *z()];
    switch(m_direction) {
    case AxisDirection::X:
        return _x / (XGraph::SFloat)shot[ *length()];
    case AxisDirection::Y:
        return _y / (XGraph::SFloat)shot[ *length()];
    case AxisDirection::Z:
        return _z / (XGraph::SFloat)shot[ *length()];
    case AxisDirection::Weight:
        return 0;
    }
}
XGraph::VFloat
XAxis::screenToVal(const Snapshot &shot, const XGraph::ScrPoint &scr) const {
    return axisToVal(screenToAxis(shot, scr));
}
void
XAxis::valToScreen(const Snapshot &shot, XGraph::VFloat val, XGraph::ScrPoint *scr) {
    axisToScreen(shot, valToAxis(val), scr);
}
XString
XAxis::valToString(XGraph::VFloat val) const {
    return formatDouble(( **ticLabelFormat())->to_str().c_str(), val);
}

XAxis::Tic
XAxis::queryTic(int len, int pos, XGraph::VFloat *ticnum) {
	XGraph::VFloat x, t;
	if(m_bLogscaleFixed) {
		x = axisToVal((XGraph::GFloat)pos / len);
        if(x <= 0) return Tic::None;
        t = std::pow((XGraph::VFloat)10.0, std::rint(std::log10(x)));
        if(std::lrint(valToAxis(t) * len) == pos) {
			*ticnum = t;
            return Tic::Major;
		}
		x = x / t;
		if(x < 1)
            t = std::rint(x / (XGraph::VFloat)0.1) * (XGraph::VFloat)0.1 * t;
		else
            t = std::rint(x) * t;
        if(std::lrint(valToAxis(t) * len) == pos) {
			*ticnum = t;
            return Tic::Minor;
		}
        return Tic::None;
	}
	else {
		x = axisToVal((XGraph::GFloat)pos / len);
        t = std::rint(x / m_majorFixed) * m_majorFixed;
        if(std::lrint(valToAxis(t) * len) == pos) {
			*ticnum = t;
            return Tic::Major;
		}
        t = std::rint(x / m_minorFixed) * m_minorFixed;
        if(std::lrint(valToAxis(t) * len) == pos) {
			*ticnum = t;
            return Tic::Minor;
		}
        return Tic::None;
	}
}


void
XAxis::drawLabel(const Snapshot &shot, XQGraphPainter *painter) {
    if(m_direction == AxisDirection::Weight) return;
    auto oso = painter->createOneTimeOnScreenObject<OnScreenTextObject>();
    oso->setBaseColor(shot[ *labelColor()]);

	const int sizehint = 2;
	XGraph::ScrPoint s1, s2, s3;
    axisToScreen(shot, 0.5, &s1);
    s2 = s1;
    axisToScreen(shot, 1.5, &s3);
    s3 -= s2;
    painter->posOffAxis(m_dirVector, &s1, AxisToLabel);
    s2 -= s1;
    s2 *= -1;
    if( !oso->selectFont(shot[ *label()], s1, s2, s3, sizehint)) {
        oso->drawText(s1, shot[ *label()]);
        return;
    }
    
    axisToScreen(shot, 1.02, &s1);
    axisToScreen(shot, 1.05, &s2);
    s2 -= s1;
    s3 = s1;
    painter->posOffAxis(m_dirVector, &s3, 0.7);
    s3 -= s1;
    if( !oso->selectFont(shot[ *label()], s1, s2, s3, sizehint)) {
        oso->drawText(s1, shot[ *label()]);
        return;
    }
    
    axisToScreen(shot, 0.5, &s1);
    s2 = s1;
    axisToScreen(shot, 1.5, &s3);
    s3 -= s2;
    painter->posOffAxis(m_dirVector, &s1, -AxisToLabel);
    s2 -= s1;
    s2 *= -1;
    if( !oso->selectFont(shot[ *label()], s1, s2, s3, sizehint)) {
        oso->drawText(s1, shot[ *label()]);
        return;
    }
}


int
XAxis::drawAxis(const Snapshot &shot, XQGraphPainter *painter) {
    if(m_direction == AxisDirection::Weight) return -1;
    if(shot[ *invisible()]) return -1;
  
	XGraph::SFloat LenMajorTicL = 0.01;
	XGraph::SFloat LenMinorTicL = 0.005;

	painter->setColor(shot[ *ticColor()]);

	XGraph::ScrPoint s1, s2;
	axisToScreen(shot, 0.0, &s1);
	axisToScreen(shot, 1.0, &s2);

	painter->beginLine();
	painter->setVertex(s1);
	painter->setVertex(s2);
	painter->endLine();
  
	if(shot[ *displayLabel()]) {
		drawLabel(shot, painter);
	}
	if(m_bLogscaleFixed && (m_minFixed < 0)) return -1;
	if(m_maxFixed <= m_minFixed) return -1;

    auto oso = painter->createOneTimeOnScreenObject<OnScreenTextObject>();
    oso->setBaseColor(shot[ *ticColor()]);

    int len = std::lrint(shot[ *length()] / painter->resScreen());
	XGraph::GFloat mindx = 2, lastg = -1;
	//dry-running to determine a font
	for(int i = 0; i < len; ++i) {
		XGraph::VFloat z;
		XGraph::GFloat x = (XGraph::GFloat)i / len;
        if(queryTic(len, i, &z) == Tic::Major) {
			if(mindx > x - lastg) {
				axisToScreen(shot, x, &s1);
				s2 = s1;
				XGraph::ScrPoint s3;
				axisToScreen(shot, lastg, &s3);
				s3 -= s2;
				s3 *= 0.7;
				painter->posOffAxis(m_dirVector, &s1, AxisToTicLabel);
				s2 -= s1;
				s2 *= -1;
            
				double var = setprec(z, m_bLogscaleFixed ? (XGraph::VFloat)z :  m_minorFixed);
                oso->selectFont(valToString(var), s1, s2, s3, 0);
            
				mindx = x - lastg;
			}
			lastg = x;
		}
	}

	for(int i = 0; i < len; ++i) {
		XGraph::VFloat z;
		XGraph::GFloat x = (XGraph::GFloat)i / len;
		switch(queryTic(len, i, &z)) {
        case Tic::Major:
			if(shot[ *displayMajorTics()]) {
				axisToScreen(shot, x, &s1);
				painter->posOffAxis(m_dirVector, &s1, LenMajorTicL);
				axisToScreen(shot, x, &s2);
				painter->posOffAxis(m_dirVector, &s2, -LenMajorTicL);
				painter->setColor(shot[ *ticColor()]);
				painter->beginLine(1.0);
				painter->setVertex(s1);
				painter->setVertex(s2);
				painter->endLine();
			}
			if(shot[ *displayTicLabels()]) {
				axisToScreen(shot, x, &s1);
				painter->posOffAxis(m_dirVector, &s1, AxisToTicLabel);
				double var = setprec(z, m_bLogscaleFixed ? (XGraph::VFloat)z : m_minorFixed);
                oso->drawText(s1,
                    formatDouble(shot[ *ticLabelFormat()].to_str().c_str(), var)); //valToString(var)
			}
			break;
        case Tic::Minor:
			if(shot[ *displayMinorTics()]) {
				axisToScreen(shot, x, &s1);
				painter->posOffAxis(m_dirVector, &s1, LenMinorTicL);
				axisToScreen(shot, x, &s2);
				painter->posOffAxis(m_dirVector, &s2, -LenMinorTicL);
				painter->setColor(shot[ *ticColor()]);
				painter->beginLine(1.0);
				painter->setVertex(s1);
				painter->setVertex(s2);
				painter->endLine();
			}
			break;
        case Tic::None:
			break;
		}
    }
	return 0;
}

XFuncPlot::XFuncPlot(const char *name, bool runtime, Transaction &tr_graph, const shared_ptr<XGraph> &graph)
	: XPlot(name, runtime, tr_graph, graph) {
	trans( *maxCount()) = 300;
    iterate_commit([=](Transaction &tr){
        tr[ *drawPoints()] = false;
        tr[ *drawBars()].setUIEnabled(false);
        tr[ *barColor()].setUIEnabled(false);
        tr[ *clearPoints()].setUIEnabled(false);
    });
}

void
XFuncPlot::snapshot(const Snapshot &shot) {
	unsigned int cnt = (unsigned int)shot[ *maxCount()];
	m_ptsSnapped.resize(cnt);
	for(unsigned int i = 0; i < cnt; ++i) {
		XGraph::ValPoint &pt(m_ptsSnapped[i]);
		pt.x = m_curAxisX->axisToVal((XGraph::GFloat)i / cnt);
		pt.y = func(pt.x);
		pt.z = 0.0;
	}
}

X2DImagePlot::X2DImagePlot(const char *name, bool runtime, Transaction &tr_graph, const shared_ptr<XGraph> &graph)
    : XPlot(name, runtime, tr_graph, graph) {
    iterate_commit([=](Transaction &tr){
        tr[ *displayMajorGrid()] = false;
        tr[ *displayMinorGrid()] = false;
        tr[ *intensity()] = 1.0;
    });
}
X2DImagePlot::~X2DImagePlot() {
}
void
X2DImagePlot::setImage(Transaction &tr, const shared_ptr<QImage>& image) {
    tr[ *this].m_image = image;
    shared_ptr<XGraph> graph(m_graph.lock());
    tr.mark(tr[ *graph].onUpdate(), graph.get());
}
void
X2DImagePlot::snapshot(const Snapshot &shot) {
    m_image = shot[ *this].m_image;
}
int
X2DImagePlot::drawPlot(const Snapshot &shot, XQGraphPainter *painter) {
    if(m_image) {
//        auto texture = m_texture;
//        if(texture && (painter != texture->painter()))
//            texture.reset(); //not valid anymore.
//        if(!texture || (m_image != m_image_textured)) {
//            if(texture && m_image_textured && (m_image_textured->width() == m_image->width())
//                    && (m_image_textured->height() == m_image->height())
//                    && (m_image_textured->format() == m_image->format()))
//                texture->repaint(m_image);
//            else {
//                m_texture = painter->createTextureWeakly(m_image);
//            }
//            m_image_textured = m_image;
//        }
        auto texture = painter->createTextureDuringListing(m_image).lock();
        if(texture && fixScales(shot)) {
            XGraph::ScrPoint spt[4];
            XGraph::ValPoint v1(0, 0);
            XGraph::GPoint g;
            valToGraphFast(v1, &g);
            graphToScreenFast(g, &spt[0]);
            XGraph::ValPoint v2(m_image->width(), 0);
            valToGraphFast(v2, &g);
            graphToScreenFast(g, &spt[1]);
            XGraph::ValPoint v3(m_image->width(), m_image->height());
            valToGraphFast(v3, &g);
            graphToScreenFast(g, &spt[2]);
            XGraph::ValPoint v4(0, m_image->height());
            valToGraphFast(v4, &g);
            graphToScreenFast(g, &spt[3]);
            texture->placeObject(spt[0], spt[1], spt[2], spt[3], OnScreenTexture::HowToEvade::Never, {});
        }
    }
    return XPlot::drawPlot(shot, painter);
}
int
X2DImagePlot::validateAutoScale(const Snapshot &shot) {
    m_curAxisX->tryInclude(0);
    if(shot[ *this].image())
        m_curAxisX->tryInclude(shot[ *this].image()->width());
    m_curAxisY->tryInclude(0);
    if(shot[ *this].image())
        m_curAxisY->tryInclude(shot[ *this].image()->height());
    return 0;
}
