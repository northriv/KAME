/***************************************************************************
		Copyright (C) 2002-2014 Kentaro Kitagawa
		                   kitagawa@phys.s.u-tokyo.ac.jp
		
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

// \todo Use Payload for scales.

#include "graphpainter.h"
#include <qgl.h>

using std::min;
using std::max;

#define GRAPH_UI_DELAY 10

#include <float.h>
const XGraph::SFloat XGraph::SFLOAT_MAX = FLT_MAX;
const XGraph::GFloat XGraph::GFLOAT_MAX = FLT_MAX;
const XGraph::VFloat XGraph::VFLOAT_MAX = DBL_MAX;

#define SFLOAT_EXP expf
#define GFLOAT_EXP expf
#define VFLOAT_EXP exp

#define SFLOAT_POW powf
#define GFLOAT_POW powf
#define VFLOAT_POW pow

#define SFLOAT_LOG logf
#define GFLOAT_LOG logf
#define VFLOAT_LOG log

#define SFLOAT_LOG10 log10f
#define GFLOAT_LOG10 log10f
#define VFLOAT_LOG10 log10

#define SFLOAT_LRINT lrintf
#define GFLOAT_LRINT lrintf
#define VFLOAT_LRINT lrint

#define VFLOAT_RINT lrint

#define AxisToLabel 0.09
#define AxisToTicLabel 0.015
#define UNZOOM_ABIT 0.95

#define PLOT_POINT_SIZE 5.0

#define PLOT_POINT_INTENS 0.5
#define PLOT_LINE_INTENS 0.7
#define PLOT_BAR_INTENS 0.4

XGraph::XGraph(const char *name, bool runtime) : 
    XNode(name, runtime),
    m_label(create<XStringNode>("Label", true)),
    m_axes(create<XAxisList>("Axes", true)),
    m_plots(create<XPlotList>("Plots", true)),
    m_backGround(create<XHexNode>("BackGround", true)),
    m_titleColor(create<XHexNode>("TitleColor", true)),
    m_drawLegends(create<XBoolNode>("DrawLegends", true)),
    m_persistence(create<XDoubleNode>("Persistence", true)) {

	for(Transaction tr(*this);; ++tr) {
		m_lsnPropertyChanged = tr[ *label()].onValueChanged().connect(*this,
																   &XGraph::onPropertyChanged);
		tr[ *backGround()].onValueChanged().connect(lsnPropertyChanged());
		tr[ *titleColor()].onValueChanged().connect(lsnPropertyChanged());
		tr[ *drawLegends()].onValueChanged().connect(lsnPropertyChanged());
		tr[ *persistence()].onValueChanged().connect(lsnPropertyChanged());
		tr[ *backGround()] = clWhite;
		tr[ *titleColor()] = clBlack;
		tr[ *drawLegends()] = true;

		tr[ *label()] = name;

	    auto xaxis = axes()->create<XAxis>(tr, "XAxis", true, XAxis::DirAxisX
							   , false, ref(tr), static_pointer_cast<XGraph>(shared_from_this()));
	    tr[ *xaxis->label()] = i18n("X Axis");
	    auto yaxis = axes()->create<XAxis>(tr, "YAxis", true, XAxis::DirAxisY
							   , false, ref(tr), static_pointer_cast<XGraph>(shared_from_this()));
	    tr[ *yaxis->label()] = i18n("Y Axis");

	    if(tr.commit())
			break;
	}
}

void
XGraph::onPropertyChanged(const Snapshot &shot, XValueNodeBase *) {
	Snapshot shot_this( *this);
	shot_this.talk(shot_this[ *this].onUpdate(), this);
}

void
XGraph::setupRedraw(Transaction &tr, float resolution) {
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
				axis->zoom(true, true, UNZOOM_ABIT);
			axis->fixScale(tr, resolution, true);
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
	  m_intensity(create<XDoubleNode>("Intensity", true)) {

	for(Transaction tr( *this);; ++tr) {
	//  MaxCount.value(0);
		tr[ *drawLines()] = true;
		tr[ *drawBars()] = false;
		tr[ *drawPoints()] = true;
		tr[ *majorGridColor()] = clAqua;
		tr[ *minorGridColor()] = clLime;
		tr[ *lineColor()] = clRed;
		tr[ *pointColor()] = clRed;
		tr[ *barColor()] = clRed;
		tr[ *displayMajorGrid()] = true;
		tr[ *displayMinorGrid()] = false;
		intensity()->setFormat("%.2f");
		tr[ *intensity()] = 1.0;
		tr[ *colorPlot()] = false;
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

		if(tr.commit())
			break;
	}
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
XPlot::screenToGraph(const Snapshot &shot, const XGraph::ScrPoint &pt, XGraph::GPoint *g) {
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
inline void
XPlot::graphToScreenFast(const XGraph::GPoint &pt, XGraph::ScrPoint *scr) {
	scr->x = m_scr0.x + m_len.x * pt.x;
	scr->y = m_scr0.y + m_len.y * pt.y;
	scr->z = m_scr0.z + m_len.z * pt.z;
	scr->w = std::min(std::max(pt.w, (XGraph::GFloat)0.0), (XGraph::GFloat)1.0);
}
inline void
XPlot::valToGraphFast(const XGraph::ValPoint &pt, XGraph::GPoint *gr) {
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
	for(Transaction tr( *m_graph.lock());; ++tr) {
		clearAllPoints(tr);
		if(tr.commit())
			break;
	}
}

void
XPlot::drawGrid(const Snapshot &shot, XQGraphPainter *painter, shared_ptr<XAxis> &axis1, shared_ptr<XAxis> &axis2) {
	int len = SFLOAT_LRINT(1.0/painter->resScreen());
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
			case XAxis::MajorTic:
				if(disp_major) {
					painter->setColor(major_color,
									  max(0.0, min(intens * 0.7, 0.5)) );
					painter->setVertex(s1);
					painter->setVertex(s2);
				}
				break;
			case XAxis::MinorTic:
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
XPlot::blendColor(unsigned int c1, unsigned int c2, float t) {
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
XPlot::isPtIncluded(const XGraph::GPoint &pt) {
	return (pt.x >= 0) && (pt.x <= 1) &&
		(pt.y >= 0) && (pt.y <= 1) &&
		(pt.z >= 0) && (pt.z <= 1);
}

inline bool
XPlot::clipLine(const tCanvasPoint &c1, const tCanvasPoint &c2,
				XGraph::ScrPoint *s1, XGraph::ScrPoint *s2, bool blendcolor,
				unsigned int *color1, unsigned int *color2, float *alpha1, float *alpha2) {
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
	tr[ *this].points().clear();
	shared_ptr<XGraph> graph(m_graph.lock());
	const Snapshot &shot(tr);
	tr.mark(shot[ *graph].onUpdate(), graph.get());
}

void
XXYPlot::snapshot(const Snapshot &shot) {
    const auto &points(shot[ *this].points());
	unsigned int cnt = std::min((unsigned int)shot[ *maxCount()], (unsigned int)points.size());
	m_ptsSnapped.resize(cnt);
	for(unsigned int i = 0; i < cnt; ++i) {
		m_ptsSnapped[i] = points[i];
	}
}
void
XXYPlot::addPoint(Transaction &tr,
	XGraph::VFloat x, XGraph::VFloat y, XGraph::VFloat z, XGraph::VFloat w) {
	XGraph::ValPoint npt(x, y, z, w);

	shared_ptr<XGraph> graph(m_graph.lock());

    auto &points(tr[ *this].points());
	const Snapshot &shot(tr);
	while((points.size() >= shot[ *maxCount()]) && points.size()) {
		points.pop_front();
	}
	points.push_back(npt);
	tr.mark(shot[ *graph].onUpdate(), graph.get());
}

XAxis::XAxis(const char *name, bool runtime,
			 AxisDirection dir, bool rightOrTop, Transaction &tr_graph, const shared_ptr<XGraph> &graph) :
	XNode(name, runtime),
	m_direction(dir),
	m_dirVector( (dir == DirAxisX) ? 1.0 : 0.0, 
				 (dir == DirAxisY) ? 1.0 : 0.0, (dir == DirAxisZ) ? 1.0 : 0.0),
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
	m_ticLabelFormat(create<XStringNode>("TicLabelFormat", true)),
	m_displayLabel(create<XBoolNode>("DisplayLabel", true)),
	m_displayTicLabels(create<XBoolNode>("DisplayTicLabels", true)),
	m_ticColor(create<XHexNode>("TicColor", true)),
	m_labelColor(create<XHexNode>("LabelColor", true)),
	m_ticLabelColor(create<XHexNode>("TicLabelColor", true)),
	m_autoFreq(create<XBoolNode>("AutoFreq", true)),
	m_autoScale(create<XBoolNode>("AutoScale", true)),
	m_logScale(create<XBoolNode>("LogScale", true)) {
	m_ticLabelFormat->setValidator(&formatDoubleValidator);
  
	for(Transaction tr(*this);; ++tr) {
		tr[ *x()] = 0.15;
		tr[ *y()] = 0.15;
		tr[ *z()] = 0.15;
		tr[ *length()] = 0.7;
		tr[ *maxValue()] = 0;
		tr[ *minValue()] = 0;
		tr[ *ticLabelFormat()] = "";
		tr[ *logScale()] = false;
		tr[ *displayLabel()] = true;
		tr[ *displayTicLabels()] = true;
		tr[ *displayMajorTics()] = true;
		tr[ *displayMinorTics()] = true;
		tr[ *autoFreq()] = true;
		tr[ *autoScale()] = true;
		tr[ *rightOrTopSided()] = rightOrTop;
		tr[ *majorTicScale()] = 10;
		tr[ *minorTicScale()] = 1;
		tr[ *labelColor()] = clBlack;
		tr[ *ticColor()] = clBlack;
		tr[ *ticLabelColor()] = clBlack;

		if(rightOrTop) {
			if(dir == DirAxisY) tr[ *x()] = 1.0- tr[ *x()];
			if(dir == DirAxisX) tr[ *y()] = 1.0- tr[ *y()];
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

		if(tr.commit())
			break;
	}
}


void
XAxis::startAutoscale_(const Snapshot &shot, bool clearscale) {
	m_bLogscaleFixed = shot[ *logScale()];
	m_bAutoscaleFixed = shot[ *autoScale()];
	if(clearscale) {
		m_minFixed = XGraph::VFLOAT_MAX;
		m_maxFixed = m_bLogscaleFixed ? 0 : - XGraph::VFLOAT_MAX;
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
}
void
XAxis::performAutoFreq(const Snapshot &shot, float resolution) {
	if(shot[ *autoFreq()] &&
	   ( !m_bLogscaleFixed || (m_minFixed >= 0)) &&
	   (m_minFixed < m_maxFixed)) {
		float fac = max(0.8f, log10f(2e-3 / resolution) );
		m_majorFixed = (VFLOAT_POW((XGraph::VFloat)10.0,
								   VFLOAT_RINT(VFLOAT_LOG10(m_maxFixed - m_minFixed) - fac)));
		m_minorFixed = m_majorFixed / (XGraph::VFloat)2.0;
	}
	else {
		m_majorFixed = shot[ *majorTicScale()];
		m_minorFixed = shot[ *minorTicScale()];
	}
}

inline bool
XAxis::isIncluded(XGraph::VFloat x) {
	return (x >= m_minFixed)
		&& ((m_direction == AxisWeight) || (x <= m_maxFixed));
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
	if(direction() == AxisWeight) return;
	
	if(maxchange) {
		m_maxFixed = axisToVal(center + (XGraph::GFloat)0.5 / prop);
	}
	if(minchange) {
		m_minFixed = axisToVal(center - (XGraph::GFloat)0.5 / prop);
	}
	m_invMaxMinusMinFixed = -1; //undef
	m_invLogMaxOverMinFixed = -1; //undef
}

XGraph::GFloat
XAxis::valToAxis(XGraph::VFloat x) {
	XGraph::GFloat pos;
	if(m_bLogscaleFixed) {
		if ((x <= 0) || (m_minFixed <= 0) || (m_maxFixed <= m_minFixed))
			return - XGraph::GFLOAT_MAX;
		if(m_invLogMaxOverMinFixed < 0)
			m_invLogMaxOverMinFixed = 1 / VFLOAT_LOG(m_maxFixed / m_minFixed);  
		pos = VFLOAT_LOG(x / m_minFixed) * m_invLogMaxOverMinFixed;
	}
	else {
		if(m_maxFixed <= m_minFixed) return -1;
		if(m_invMaxMinusMinFixed < 0)
			m_invMaxMinusMinFixed = 1 / (m_maxFixed - m_minFixed);
		pos = (x - m_minFixed) * m_invMaxMinusMinFixed;
	}
	return pos;
}

XGraph::VFloat
XAxis::axisToVal(XGraph::GFloat pos, XGraph::GFloat axis_prec) {
	XGraph::VFloat x = 0;
	if(axis_prec <= 0) {
		if(m_bLogscaleFixed) {
			if((m_minFixed <= 0) || (m_maxFixed < m_minFixed)) return 0;
			x = m_minFixed * VFLOAT_EXP(VFLOAT_LOG(m_maxFixed / m_minFixed) * pos);
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
XAxis::axisToScreen(const Snapshot &shot, XGraph::GFloat pos, XGraph::ScrPoint *scr) {
	XGraph::SFloat len = shot[ *length()];
	pos *= len;
	scr->x = shot[ *x()] + ((m_direction == DirAxisX) ? pos: (XGraph::SFloat) 0.0);
	scr->y = shot[ *y()] + ((m_direction == DirAxisY) ? pos: (XGraph::SFloat) 0.0);
	scr->z = shot[ *z()] + ((m_direction == DirAxisZ) ? pos: (XGraph::SFloat) 0.0);
}
XGraph::GFloat
XAxis::screenToAxis(const Snapshot &shot, const XGraph::ScrPoint &scr) {
	XGraph::SFloat _x = scr.x - shot[ *x()];
	XGraph::SFloat _y = scr.y - shot[ *y()];
	XGraph::SFloat _z = scr.z - shot[ *z()];
	XGraph::GFloat pos = ((m_direction == DirAxisX) ? _x :
						  ((m_direction == DirAxisY) ? _y : _z)) / (XGraph::SFloat)shot[ *length()];
	return pos;
}
XGraph::VFloat
XAxis::screenToVal(const Snapshot &shot, const XGraph::ScrPoint &scr) {
    return axisToVal(screenToAxis(shot, scr));
}
void
XAxis::valToScreen(const Snapshot &shot, XGraph::VFloat val, XGraph::ScrPoint *scr) {
    axisToScreen(shot, valToAxis(val), scr);
}
XString
XAxis::valToString(XGraph::VFloat val) {
    return formatDouble(( **ticLabelFormat())->to_str().c_str(), val);
}

XAxis::Tic
XAxis::queryTic(int len, int pos, XGraph::VFloat *ticnum) {
	XGraph::VFloat x, t;
	if(m_bLogscaleFixed) {
		x = axisToVal((XGraph::GFloat)pos / len);
		if(x <= 0) return NoTics;
		t = VFLOAT_POW((XGraph::VFloat)10.0, VFLOAT_RINT(VFLOAT_LOG10(x)));
		if(GFLOAT_LRINT(valToAxis(t) * len) == pos) {
			*ticnum = t;
			return MajorTic;
		}
		x = x / t;
		if(x < 1)
			t = VFLOAT_RINT(x / (XGraph::VFloat)0.1) * (XGraph::VFloat)0.1 * t;
		else
			t = VFLOAT_RINT(x) * t;
		if(GFLOAT_LRINT(valToAxis(t) * len) == pos) {
			*ticnum = t;
			return MinorTic;
		}
		return NoTics;
	}
	else {
		x = axisToVal((XGraph::GFloat)pos / len);
		t = VFLOAT_RINT(x / m_majorFixed) * m_majorFixed;
		if(GFLOAT_LRINT(valToAxis(t) * len) == pos) {
			*ticnum = t;
			return MajorTic;
		}
		t = VFLOAT_RINT(x / m_minorFixed) * m_minorFixed;
		if(GFLOAT_LRINT(valToAxis(t) * len) == pos) {
			*ticnum = t;
			return MinorTic;
		}
		return NoTics;
	}
}


void
XAxis::drawLabel(const Snapshot &shot, XQGraphPainter *painter) {
    if(m_direction == AxisWeight) return;
  
	const int sizehint = 2;
	painter->setColor(shot[ *labelColor()]);
	XGraph::ScrPoint s1, s2, s3;
    axisToScreen(shot, 0.5, &s1);
    s2 = s1;
    axisToScreen(shot, 1.5, &s3);
    s3 -= s2;
    painter->posOffAxis(m_dirVector, &s1, AxisToLabel);
    s2 -= s1;
    s2 *= -1;
    if( !painter->selectFont(shot[ *label()], s1, s2, s3, sizehint)) {
        painter->drawText(s1, shot[ *label()]);
        return;
    }
    
    axisToScreen(shot, 1.02, &s1);
    axisToScreen(shot, 1.05, &s2);
    s2 -= s1;
    s3 = s1;
    painter->posOffAxis(m_dirVector, &s3, 0.7);
    s3 -= s1;
    if( !painter->selectFont(shot[ *label()], s1, s2, s3, sizehint)) {
        painter->drawText(s1, shot[ *label()]);
        return;
    }
    
    axisToScreen(shot, 0.5, &s1);
    s2 = s1;
    axisToScreen(shot, 1.5, &s3);
    s3 -= s2;
    painter->posOffAxis(m_dirVector, &s1, -AxisToLabel);
    s2 -= s1;
    s2 *= -1;
    if( !painter->selectFont(shot[ *label()], s1, s2, s3, sizehint)) {
        painter->drawText(s1, shot[ *label()]);
        return;
    }
}


int
XAxis::drawAxis(const Snapshot &shot, XQGraphPainter *painter) {
	if(m_direction == AxisWeight) return -1;
  
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
  
	int len = SFLOAT_LRINT(shot[ *length()] / painter->resScreen());
	painter->defaultFont();
	XGraph::GFloat mindx = 2, lastg = -1;
	//dry-running to determine a font
	for(int i = 0; i < len; ++i) {
		XGraph::VFloat z;
		XGraph::GFloat x = (XGraph::GFloat)i / len;
		if(queryTic(len, i, &z) == MajorTic) {
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
				painter->selectFont(valToString(var), s1, s2, s3, 0);
            
				mindx = x - lastg;
			}
			lastg = x;
		}
	}
  
	for(int i = 0; i < len; ++i) {
		XGraph::VFloat z;
		XGraph::GFloat x = (XGraph::GFloat)i / len;
		switch(queryTic(len, i, &z)) {
		case MajorTic:
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
				painter->drawText(s1, valToString(var));
			}
			break;
		case MinorTic:
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
		case NoTics:
			break;
		}
    }
	return 0;
}

XFuncPlot::XFuncPlot(const char *name, bool runtime, Transaction &tr_graph, const shared_ptr<XGraph> &graph)
	: XPlot(name, runtime, tr_graph, graph) {
	trans( *maxCount()) = 300;
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

