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
#include "graphpainter.h"
#include "measure.h"
#include "graphwidget.h"


#define SELECT_WIDTH 0.08
#define SELECT_DEPTH 0.05

using std::min;
using std::max;

XQGraphPainter::XQGraphPainter(const shared_ptr<XGraph> &graph, XQGraph* item) :
	m_graph(graph),
	m_pItem(item),
	m_selectionStateNow(Selecting),
	m_selectionModeNow(SelNone),
	m_listpoints(0),
	m_listaxes(0),
	m_listgrids(0),
	m_listplanes(0),
	m_bIsRedrawNeeded(true),
	m_bIsAxisRedrawNeeded(false),
	m_bTilted(false),
	m_bReqHelp(false) {

	openFont();
	item->m_painter.reset(this);
	for(Transaction tr( *graph);; ++tr) {
		m_lsnRedraw = tr[ *graph].onUpdate().connectWeakly(
	        shared_from_this(), &XQGraphPainter::onRedraw,
	        XListener::FLAG_MAIN_THREAD_CALL | XListener::FLAG_AVOID_DUP | XListener::FLAG_DELAY_ADAPTIVE);
		if(tr.commit())
			break;
	}
}
float
XQGraphPainter::resScreen() {
    return 1.0f / max(m_pItem->width(), m_pItem->height());
} 

void
XQGraphPainter::repaintGraph(int x1, int y1, int x2, int y2) {
	repaintBuffer(x1, y1, x2, y2);
}

void
XQGraphPainter::posOffAxis(const XGraph::ScrPoint &dir, 
						   XGraph::ScrPoint *src, XGraph::SFloat offset) {
	XGraph::ScrPoint scr, dir_proj;
	windowToScreen(m_pItem->width() / 2, m_pItem->height() / 2, 0.0, &scr);
	windowToScreen(m_pItem->width() / 2, m_pItem->height() / 2, 1.0, &dir_proj);
	dir_proj -= scr;
	dir_proj.normalize();
	double x, y, z;
	if(screenToWindow(*src, &x, &y, &z)) return;
	x = x / m_pItem->width() - 0.5;
	y = y / m_pItem->height() - 0.5;
	double r1 = x*x + y*y;
	XGraph::ScrPoint s1 = dir_proj;
	s1.vectorProduct(dir);
	s1.normalize();
	s1 *= offset;
	*src += s1;
	int ret = screenToWindow(*src, &x, &y, &z);
	x = x / m_pItem->width() - 0.5;
	y = y / m_pItem->height() - 0.5;
	double r2 = x*x + y*y;
	if(ret || ((r2 - r1) * offset < 0) ) {
		*src -= s1;
		*src -= s1;
	}
}

shared_ptr<XAxis> 
XQGraphPainter::findAxis(const Snapshot &shot, const XGraph::ScrPoint &s1) {
    shared_ptr<XAxis> found_axis;
    double zmin = 0.1;
	if(shot.size(m_graph->axes())) {
		const XNode::NodeList &axes_list( *shot.list(m_graph->axes()));
		for(auto it = axes_list.begin(); it != axes_list.end(); it++) {
			auto axis = dynamic_pointer_cast<XAxis>(*it);
			XGraph::SFloat z1;
			XGraph::ScrPoint s2;
			axis->axisToScreen(shot, axis->screenToAxis(shot, s1), &s2);
			z1 = sqrtf(s1.distance2(s2));
			if(zmin > z1) {
				zmin = z1;
				found_axis = axis;
			}
		}
    }
    return found_axis;
}
shared_ptr<XPlot> 
XQGraphPainter::findPlane(const Snapshot &shot, const XGraph::ScrPoint &s1,
						  shared_ptr<XAxis> *axis1, shared_ptr<XAxis> *axis2) {
	double zmin = 0.1;
	shared_ptr<XPlot> plot_found;
	if(shot.size(m_graph->plots())) {
		const auto &plots_list( *shot.list(m_graph->plots()));
		for(auto it = plots_list.begin(); it != plots_list.end(); it++) {
			shared_ptr<XPlot> plot = dynamic_pointer_cast<XPlot>( *it);
			XGraph::GPoint g1;
			shared_ptr<XAxis> axisx = shot[ *plot->axisX()];
			shared_ptr<XAxis> axisy = shot[ *plot->axisY()];
			shared_ptr<XAxis> axisz = shot[ *plot->axisZ()];
			if( !axisx || !axisy)
				continue;
			plot->screenToGraph(shot, s1, &g1);
			if((fabs(g1.x) < zmin) && axisz) {
				plot_found = plot;
				zmin = fabs(g1.x);
				*axis1 = axisy;
				*axis2 = axisz;
			}
			if((fabs(g1.y) < zmin) && axisz) {
				plot_found = plot;
				zmin = fabs(g1.y);
				*axis1 = axisx;
				*axis2 = axisz;
			}
			if(fabs(g1.z) < zmin) {
				plot_found = plot;
				zmin = fabs(g1.z);
				*axis1 = axisx;
				*axis2 = axisy;
			}
		}       
    }
    return plot_found;
}

void
XQGraphPainter::selectObjs(int x, int y, SelectionState state, SelectionMode mode) {
	m_pointerLastPos[0] = x;
	m_pointerLastPos[1] = y;

	if(m_bReqHelp) {
		if(state != SelStart) return;
		m_bReqHelp = false;
		return;
	}
	double z;
    
	m_selectionStateNow = state;
	switch(state) {
	case SelStart:
		m_selectionModeNow = mode;
		m_selStartPos[0] = x;
		m_selStartPos[1] = y;
		m_tiltLastPos[0] = x;
		m_tiltLastPos[1] = y;
		switch(mode) {
		case SelPlane:
			m_foundPlane.reset();
			z = selectPlane(x, y, 
							(int)(SELECT_WIDTH * m_pItem->width()), 
							(int)(SELECT_WIDTH * m_pItem->height()),
							&m_startScrPos, &m_startScrDX, &m_startScrDY);
			if(z < 1.0)
				m_foundPlane = findPlane(Snapshot( *m_graph), m_startScrPos, &m_foundPlaneAxis1, &m_foundPlaneAxis2);
			m_finishScrPos = m_startScrPos;
			break;
		case SelAxis:
			m_foundAxis.reset();
			z = selectAxis(x, y, 
						   (int)(SELECT_WIDTH * m_pItem->width()),
						   (int)(SELECT_WIDTH * m_pItem->height()),
						   &m_startScrPos, &m_startScrDX, &m_startScrDY);
			if(z < 1.0) m_foundAxis = findAxis(Snapshot( *m_graph), m_startScrPos);
			m_finishScrPos = m_startScrPos;
			break;
		default:
			break;
		}
		break;
	case Selecting:
		//restore mode
		mode = m_selectionModeNow;
		break;
	case SelFinish:
		//restore mode
		mode = m_selectionModeNow;
		m_selectionModeNow = SelNone;
		break;
	}
	switch(mode) {
	case SelNone:
		m_foundPlane.reset();
		z = selectPlane(x, y, 
						(int)(SELECT_WIDTH * m_pItem->width()),
						(int)(SELECT_WIDTH * m_pItem->height()),
						&m_finishScrPos, &m_finishScrDX, &m_finishScrDY);
		if(z < 1.0)
            m_foundPlane = findPlane(Snapshot( *m_graph), m_finishScrPos, &m_foundPlaneAxis1, &m_foundPlaneAxis2);
		break;
	case SelPlane:
		selectPlane(x, y, 
					(int)(SELECT_WIDTH * m_pItem->width()),
					(int)(SELECT_WIDTH * m_pItem->height()),
					&m_finishScrPos, &m_finishScrDX, &m_finishScrDY);
		break;
	case SelAxis:
		selectAxis(x, y, 
				   (int)(SELECT_WIDTH * m_pItem->width()),
				   (int)(SELECT_WIDTH * m_pItem->height()),
				   &m_finishScrPos, &m_finishScrDX, &m_finishScrDY);
		break;
	case TiltTracking:
	{
		float x0 = (float)m_tiltLastPos[0] / m_pItem->width() - 0.5;
		float y0 = (1.0f - (float)m_tiltLastPos[1] / m_pItem->height()) - 0.5;
		float z0 = 0.5;
		float x1 = (float)x / m_pItem->width() - 0.5;
		float y1 = (1.0f - (float)y / m_pItem->height()) - 0.5;
		float z1 = z0;
		float x2, y2, z2;
		x2 = y0 * z1 - z0 * y1;
		y2 = z0 * x1 - x0 * z1;
		z2 = x0 * y1 - y0 * x1;
		float k = sqrt(x2*x2 + y2*y2 + z2*z2);
		x2 /= k; y2 /= k; z2 /= k;
		viewRotate( -pow(k/(z0/4), 1.4f)*90, x2, y2, z2);
		m_tiltLastPos[0] = x;
		m_tiltLastPos[1] = y;
	}
	break;
	default:
		break;
	}
	if(state == SelFinish) {
	    if((abs(x - m_selStartPos[0]) < 3) && (abs(y - m_selStartPos[1]) < 3)) {
			switch(mode) {
			case SelPlane:
				break;
			case SelAxis:
				for(Transaction tr( *m_graph);; ++tr) {
					if( !m_foundAxis) {
						//Autoscales all axes
						if(tr.size(m_graph->axes())) {
							const auto &axes_list( *tr.list(m_graph->axes()));
							for(auto it = axes_list.begin(); it != axes_list.end(); it++) {
								shared_ptr<XAxis> axis = static_pointer_cast<XAxis>(*it);
								if(tr[ *axis->autoScale()].isUIEnabled())
									tr[ *axis->autoScale()] = true;
							}
						}
					}
					else {
						if(tr[ *m_foundAxis->autoScale()].isUIEnabled())
							tr[ *m_foundAxis->autoScale()] = true;
					}
					if(tr.commit())
						break;
				}
				break;
			case TiltTracking:
				viewRotate(0.0, 0.0, 0.0, 0.0, true);
				break;
			default:
				break;
			}
	    }
	    else {
			for(Transaction tr( *m_graph);; ++tr) {
				switch(mode) {
				case SelPlane:
					if(m_foundPlane && !(m_startScrPos == m_finishScrPos) ) {
						XGraph::VFloat src1 = m_foundPlaneAxis1->screenToVal(tr, m_startScrPos);
						XGraph::VFloat src2 = m_foundPlaneAxis2->screenToVal(tr, m_startScrPos);
						XGraph::VFloat dst1 = m_foundPlaneAxis1->screenToVal(tr, m_finishScrPos);
						XGraph::VFloat dst2 = m_foundPlaneAxis2->screenToVal(tr, m_finishScrPos);

						if(tr[ *m_foundPlaneAxis1->minValue()].isUIEnabled())
							tr[ *m_foundPlaneAxis1->minValue()] = double(min(src1, dst1));
						if(tr[ *m_foundPlaneAxis1->maxValue()].isUIEnabled())
							tr[ *m_foundPlaneAxis1->maxValue()] = double(max(src1, dst1));
						if(tr[ *m_foundPlaneAxis1->autoScale()].isUIEnabled())
							tr[ *m_foundPlaneAxis1->autoScale()] = false;
						if(tr[ *m_foundPlaneAxis2->minValue()].isUIEnabled())
							tr[ *m_foundPlaneAxis2->minValue()] = double(min(src2, dst2));
						if(tr[ *m_foundPlaneAxis2->maxValue()].isUIEnabled())
							tr[ *m_foundPlaneAxis2->maxValue()] = double(max(src2, dst2));
						if(tr[ *m_foundPlaneAxis2->autoScale()].isUIEnabled())
							tr[ *m_foundPlaneAxis2->autoScale()] = false;

					}
					break;
				case SelAxis:
					if(m_foundAxis && !(m_startScrPos == m_finishScrPos) ) {
						XGraph::VFloat src = m_foundAxis->screenToVal(tr, m_startScrPos);
						XGraph::VFloat dst = m_foundAxis->screenToVal(tr, m_finishScrPos);
						double min__ = std::min(src, dst);
						double max__ = std::max(src, dst);
						if(tr[ *m_foundAxis->minValue()].isUIEnabled())
							tr[ *m_foundAxis->minValue()] = min__;
						if(tr[ *m_foundAxis->maxValue()].isUIEnabled())
							tr[ *m_foundAxis->maxValue()] = max__;
						if(tr[ *m_foundAxis->autoScale()].isUIEnabled())
							tr[ *m_foundAxis->autoScale()] = false;
					}
					break;
				default:
					break;
				}
				if(tr.commit())
					break;
			}
	    }
	}
	
	repaintGraph(0, 0, m_pItem->width(), m_pItem->height() );
}

void
XQGraphPainter::wheel(int x, int y, double deg)
{
	double a = ((double)x / m_pItem->width() - 0.5);
	double b = ((double)y / m_pItem->height() - 0.5);
	if( max(fabs(a), fabs(b)) < 0.35) {
		zoom(min(1.15, max(0.85, exp(deg * 0.04))), x, y);
	}
	else {
		if( (a - b) * (a + b) > 0 ) {
			viewRotate(30.0 * deg / fabs(deg), -1.0, 0.0, 0.0, false);
		}
		else {
			viewRotate(30.0 * deg / fabs(deg), 0.0, 1.0, 0.0, false);
		}
		repaintGraph(0, 0, m_pItem->width(), m_pItem->height() );
	}
}
void
XQGraphPainter::zoom(double zoomscale, int , int ) {
	XGraph::ScrPoint s1(0.5, 0.5, 0.5);
  
	for(Transaction tr( *m_graph);; ++tr) {
		if(tr.size(m_graph->axes())) {
			const auto &axes_list( *tr.list(m_graph->axes()));
			for(auto it = axes_list.begin(); it != axes_list.end(); ++it) {
				shared_ptr<XAxis> axis = static_pointer_cast<XAxis>( *it);
				if(tr[ *axis->autoScale()].isUIEnabled())
					tr[ *axis->autoScale()] = false;
			}
		}
		m_graph->zoomAxes(tr, resScreen(), zoomscale, s1);
		if(tr.commit())
			break;
	}
}
void
XQGraphPainter::onRedraw(const Snapshot &, XGraph *graph) {
	redrawOffScreen();
	repaintGraph(0, 0, m_pItem->width(), m_pItem->height() );  
}
void
XQGraphPainter::drawOnScreenObj(const Snapshot &shot) {
	QString msg = "";
//   if(SelectionStateNow != Selecting) return;
	switch ( m_selectionModeNow ) {
	case SelNone:
		if(m_foundPlane) {
			XGraph::VFloat dst1 = m_foundPlaneAxis1->screenToVal(shot, m_finishScrPos);
			XGraph::VFloat dst1dx = m_foundPlaneAxis1->screenToVal(shot, m_finishScrDX) - dst1;
			XGraph::VFloat dst1dy = m_foundPlaneAxis1->screenToVal(shot, m_finishScrDY) - dst1;
			XGraph::VFloat dst2 = m_foundPlaneAxis2->screenToVal(shot, m_finishScrPos);
			XGraph::VFloat dst2dx = m_foundPlaneAxis2->screenToVal(shot, m_finishScrDX) - dst2;
			XGraph::VFloat dst2dy = m_foundPlaneAxis2->screenToVal(shot, m_finishScrDY) - dst2;

			dst1 = setprec(dst1, sqrt(dst1dx*dst1dx + dst1dy*dst1dy));
			dst2 = setprec(dst2, sqrt(dst2dx*dst2dx + dst2dy*dst2dy));
			msg += QString("(%1, %2)")
				.arg(m_foundPlaneAxis1->valToString(dst1).c_str())
				.arg(m_foundPlaneAxis2->valToString(dst2).c_str());
		}
		else {
			msg = i18n("R-DBL-CLICK TO SHOW HELP").toUtf8().data();
		}
		break;
	case SelPlane:
		if(m_foundPlane && !(m_startScrPos == m_finishScrPos) ) {
			XGraph::VFloat src1 = m_foundPlaneAxis1->screenToVal(shot, m_startScrPos);
			XGraph::VFloat src1dx = m_foundPlaneAxis1->screenToVal(shot, m_startScrDX) - src1;
			XGraph::VFloat src1dy = m_foundPlaneAxis1->screenToVal(shot, m_startScrDY) - src1;
			XGraph::VFloat src2 = m_foundPlaneAxis2->screenToVal(shot, m_startScrPos);
			XGraph::VFloat src2dx = m_foundPlaneAxis2->screenToVal(shot, m_startScrDX) - src2;
			XGraph::VFloat src2dy = m_foundPlaneAxis2->screenToVal(shot, m_startScrDY) - src2;
			XGraph::VFloat dst1 = m_foundPlaneAxis1->screenToVal(shot, m_finishScrPos);
			XGraph::VFloat dst1dx = m_foundPlaneAxis1->screenToVal(shot, m_finishScrDX) - dst1;
			XGraph::VFloat dst1dy = m_foundPlaneAxis1->screenToVal(shot, m_finishScrDY) - dst1;
			XGraph::VFloat dst2 = m_foundPlaneAxis2->screenToVal(shot, m_finishScrPos);
			XGraph::VFloat dst2dx = m_foundPlaneAxis2->screenToVal(shot, m_finishScrDX) - dst2;
			XGraph::VFloat dst2dy = m_foundPlaneAxis2->screenToVal(shot, m_finishScrDY) - dst2;

			src1 = setprec(src1, sqrt(src1dx*src1dx + src1dy*src1dy));
			src2 = setprec(src2, sqrt(src2dx*src2dx + src2dy*src2dy));
			dst1 = setprec(dst1, sqrt(dst1dx*dst1dx + dst1dy*dst1dy));
			dst2 = setprec(dst2, sqrt(dst2dx*dst2dx + dst2dy*dst2dy));
			msg += QString("(%1, %2) - (%3, %4)")
				.arg(m_foundPlaneAxis1->valToString(src1).c_str())
				.arg(m_foundPlaneAxis2->valToString(src2).c_str())
				.arg(m_foundPlaneAxis1->valToString(dst1).c_str())
				.arg(m_foundPlaneAxis2->valToString(dst2).c_str());
		
			XGraph::ScrPoint sd1, sd2;
			m_foundPlaneAxis1->valToScreen(shot, dst1, &sd1);
			m_foundPlaneAxis1->valToScreen(shot, src1, &sd2);
			sd1 -= sd2;
			sd1 += m_startScrPos;
			XGraph::ScrPoint ss1, ss2;
			m_foundPlaneAxis2->valToScreen(shot, dst2, &ss1);
			m_foundPlaneAxis2->valToScreen(shot, src2, &ss2);
			ss1 -= ss2;
			ss1 += m_startScrPos;
		
			beginQuad(true);
			setColor(clBlue, 0.2);
			setVertex(m_startScrPos);
			setVertex(sd1);
			setVertex(m_finishScrPos);
			setVertex(ss1);
			endQuad();
		}
		break;
	case SelAxis:
		if(m_foundAxis && !(m_startScrPos == m_finishScrPos) ) {
			XGraph::VFloat src = m_foundAxis->screenToVal(shot, m_startScrPos);
			XGraph::VFloat srcdx = m_foundAxis->screenToVal(shot, m_startScrDX) - src;
			XGraph::VFloat srcdy = m_foundAxis->screenToVal(shot, m_startScrDY) - src;
			XGraph::VFloat dst = m_foundAxis->screenToVal(shot, m_finishScrPos);
			XGraph::VFloat dstdx = m_foundAxis->screenToVal(shot, m_finishScrDX) - dst;
			XGraph::VFloat dstdy = m_foundAxis->screenToVal(shot, m_finishScrDY) - dst;
				
			src = setprec(src, sqrt(srcdx*srcdx + srcdy*srcdy));
			dst = setprec(dst, sqrt(dstdx*dstdx + dstdy*dstdy));
		
			msg += QString("%1 - %2")
				.arg(m_foundAxis->valToString(src).c_str())
				.arg(m_foundAxis->valToString(dst).c_str());
		
			XGraph::GFloat src1 = m_foundAxis->valToAxis(src);
			XGraph::GFloat dst1 = m_foundAxis->valToAxis(dst);		
			beginQuad(true);
			setColor( clRed, 0.4 );
			XGraph::ScrPoint s1, s2, s3, s4;
			m_foundAxis->axisToScreen(shot, src1, &s1);
			posOffAxis(m_foundAxis->dirVector(), &s1, -0.02);
			m_foundAxis->axisToScreen(shot, src1, &s2);
			posOffAxis(m_foundAxis->dirVector(), &s2, +0.02);
			m_foundAxis->axisToScreen(shot, dst1, &s3);
			posOffAxis(m_foundAxis->dirVector(), &s3, +0.02);
			m_foundAxis->axisToScreen(shot, dst1, &s4);
			posOffAxis(m_foundAxis->dirVector(), &s4, -0.02);
			setVertex(s1);
			setVertex(s2);
			setVertex(s3);
			setVertex(s4);
			endQuad();
			beginLine();
			setColor( clBlue, 1.0 );
			m_foundAxis->axisToScreen(shot, src1, &s1);
			posOffAxis(m_foundAxis->dirVector(), &s1, -0.1);
			m_foundAxis->axisToScreen(shot, src1, &s2);
			posOffAxis(m_foundAxis->dirVector(), &s2, 0.05);
			setVertex(s1);
			setVertex(s2);
			m_foundAxis->axisToScreen(shot, dst1, &s1);
			posOffAxis(m_foundAxis->dirVector(), &s1, -0.1);
			m_foundAxis->axisToScreen(shot, dst1, &s2);
			posOffAxis(m_foundAxis->dirVector(), &s2, 0.05);
			setVertex(s1);
			setVertex(s2);
			endLine();
		}
		break;
	case TiltTracking:
		break;
	default:
		break;
	}
	m_onScreenMsg = msg.toUtf8().data();
}
void
XQGraphPainter::showHelp() {
	m_bReqHelp = true;
	repaintGraph(0, 0, m_pItem->width(), m_pItem->height());
}
void
XQGraphPainter::drawOnScreenViewObj(const Snapshot &shot) {
	//Draw Title
	setColor(shot[ *m_graph->titleColor()]);
	defaultFont();
	m_curAlign = Qt::AlignTop | Qt::AlignHCenter;
	drawText(XGraph::ScrPoint(0.5, 0.99, 0.01), shot[ *m_graph->label()]);
  
	if(m_onScreenMsg.length() ) {
		selectFont(m_onScreenMsg, XGraph::ScrPoint(0.6, 0.05, 0.01), XGraph::ScrPoint(1, 0, 0), XGraph::ScrPoint(0, 0.05, 0), 0);
	 	setColor(shot[ *m_graph->titleColor()]);
		m_curAlign = Qt::AlignBottom | Qt::AlignLeft;
  		drawText(XGraph::ScrPoint(0.01, 0.01, 0.01), m_onScreenMsg);
	}
	//Legends
	if(shot[ *m_graph->drawLegends()] &&
			(m_selectionModeNow == SelNone)) {
		if(shot.size(m_graph->plots())) {
			const XNode::NodeList &plots_list( *shot.list(m_graph->plots()));
			float z = 0.98;
			float dy = 0.04;
			float x1 = 0.75;
			float y1 = 0.81;
			if(m_pointerLastPos[0] > m_pItem->width() / 2)
				x1 = 1.06f - x1;
			if(m_pointerLastPos[1] < m_pItem->height() / 2)
				y1 = 1.0f - y1 + plots_list.size() * dy;
			float x2 = x1 - 0.01;
			float x3 = x1 + 0.08;
			defaultFont();
			m_curAlign = Qt::AlignVCenter | Qt::AlignRight;
			float y2 = y1;
			for(auto it = plots_list.begin(); it != plots_list.end(); it++) {
				auto plot = static_pointer_cast<XPlot>( *it);
				selectFont(shot[ *plot->label()], XGraph::ScrPoint(x2,y2,z), XGraph::ScrPoint(1, 0, 0), XGraph::ScrPoint(0, dy, 0), 0);
				y2 -= dy;
			}
			setColor(shot[ *m_graph->backGround()], 0.7);
			beginQuad(true);
			setVertex(XGraph::ScrPoint(x1, y1 + dy/2, z));
			setVertex(XGraph::ScrPoint(x1, y2 + dy/2, z));
			setVertex(XGraph::ScrPoint(x3, y2 + dy/2, z));
			setVertex(XGraph::ScrPoint(x3, y1 + dy/2, z));
			endQuad();
			setColor(shot[ *m_graph->titleColor()], 0.05);
			beginQuad(true);
			setVertex(XGraph::ScrPoint(x1, y1 + dy/2, z));
			setVertex(XGraph::ScrPoint(x1, y2 + dy/2, z));
			setVertex(XGraph::ScrPoint(x3, y2 + dy/2, z));
			setVertex(XGraph::ScrPoint(x3, y1 + dy/2, z));
			endQuad();
			m_curAlign = Qt::AlignVCenter | Qt::AlignRight;
			float y = y1;
			for(auto it = plots_list.begin(); it != plots_list.end(); it++) {
				setColor(shot[ *m_graph->titleColor()], 1.0);
				auto plot = static_pointer_cast<XPlot>( *it);
				drawText(XGraph::ScrPoint(x2,y,z), shot[ *plot->label()]);
				plot->drawLegend(shot, this, XGraph::ScrPoint((x3 + x1)/2, y, z), (x3 - x1)/1.5f, dy/1.2f);
				y -= dy;
			}
		}
	}
	
	if(m_bReqHelp) drawOnScreenHelp(shot);
}
void
XQGraphPainter::drawOnScreenHelp(const Snapshot &shot) {
	float z = 0.99;
	setColor(shot[ *m_graph->backGround()], 0.3);
	beginQuad(true);
	setVertex(XGraph::ScrPoint(0.0, 0.0, z));
	setVertex(XGraph::ScrPoint(1.0, 0.0, z));
	setVertex(XGraph::ScrPoint(1.0, 1.0, z));
	setVertex(XGraph::ScrPoint(0.0, 1.0, z));
	endQuad();
	setColor(shot[ *m_graph->titleColor()], 0.55);
	double y = 1.0;
	beginQuad(true);
	setVertex(XGraph::ScrPoint(1.0 - y, 1.0 - y, z));
	setVertex(XGraph::ScrPoint(1.0 - y, y, z));
	setVertex(XGraph::ScrPoint(y, y, z));
	setVertex(XGraph::ScrPoint(y, 1.0 - y, z));
	endQuad();
	y -= 0.02;
	setColor(shot[ *m_graph->backGround()], 1.0);
	defaultFont();
	m_curAlign = Qt::AlignTop | Qt::AlignHCenter;
	drawText(XGraph::ScrPoint(0.5, y, z), i18n("QUICK HELP!").toUtf8().data());
	m_curAlign = Qt::AlignVCenter | Qt::AlignLeft;
	y -= 0.1;
	double x = 0.1;
	double dy = -y/10;
	selectFont(i18n("Single Click Right Button on Axis : Auto-scale").toUtf8().data(), XGraph::ScrPoint(x,y,z), XGraph::ScrPoint(1, 0, 0), XGraph::ScrPoint(0, dy, 0), 0);
	
	drawText(XGraph::ScrPoint(x, y, z), i18n("Press Left Button on Plot : Manual Scale").toUtf8().data());
	y += dy;
	drawText(XGraph::ScrPoint(x, y, z), i18n("Press Right Button along Axis: Manual Scale").toUtf8().data());
	y += dy;
	drawText(XGraph::ScrPoint(x, y, z), i18n("Single Click Right Button on Axis : Auto-scale").toUtf8().data());
	y += dy;
	drawText(XGraph::ScrPoint(x, y, z), i18n("Single Click Right Button elsewhere : Auto-scale all").toUtf8().data());
	y += dy;
	drawText(XGraph::ScrPoint(x, y, z), i18n("Press Middle Button : Tilt plots").toUtf8().data());
	y += dy;
	drawText(XGraph::ScrPoint(x, y, z), i18n("Single Click Middle Button : Reset tilting").toUtf8().data());
	y += dy;
	drawText(XGraph::ScrPoint(x, y, z), i18n("Wheel around Center : (Un)Zoom all Plots").toUtf8().data());
	y += dy;
	drawText(XGraph::ScrPoint(x, y, z), i18n("Wheel at Side : Tilt by 30deg.").toUtf8().data());
	y += dy;
	drawText(XGraph::ScrPoint(x, y, z), i18n("Double Click Left Button : Show Dialog").toUtf8().data());
	y += dy;
	drawText(XGraph::ScrPoint(x, y, z), i18n("Double Click Right Button : This Help").toUtf8().data());
}

Snapshot
XQGraphPainter::startDrawing() {
	for(Transaction tr( *m_graph);; ++tr) {
		m_graph->setupRedraw(tr, resScreen());
		if(tr.commit())
			return tr;
	}
}
void
XQGraphPainter::drawOffScreenPlanes(const Snapshot &shot) {
	setColor((QRgb)shot[ *m_graph->backGround()], 0.3);
	if(shot.size(m_graph->plots())) {
		const auto &plots_list( *shot.list(m_graph->plots()));
		for(auto it = plots_list.begin(); it != plots_list.end(); it++) {
			auto plot = static_pointer_cast<XPlot>( *it);
			XGraph::GPoint g1(0.0, 0.0, 0.0),
				g2(1.0, 0.0, 0.0),
				g3(0.0, 1.0, 0.0),
				g4(1.0, 1.0, 0.0),
				g5(0.0, 0.0, 1.0),
				g6(0.0, 1.0, 1.0),
				g7(1.0, 0.0, 1.0),
				g8(1.0, 1.0, 1.0);
			XGraph::ScrPoint s1, s2, s3, s4, s5, s6, s7, s8;
			plot->graphToScreen(shot, g1, &s1);
			plot->graphToScreen(shot, g2, &s2);
			plot->graphToScreen(shot, g3, &s3);
			plot->graphToScreen(shot, g4, &s4);
			plot->graphToScreen(shot, g5, &s5);
			plot->graphToScreen(shot, g6, &s6);
			plot->graphToScreen(shot, g7, &s7);
			plot->graphToScreen(shot, g8, &s8);
			beginQuad(true);
			setColor( shot[ *m_graph->backGround()], 0.2);
			setVertex(s1);
			setVertex(s2);
			setVertex(s4);
			setVertex(s3);
			shared_ptr<XAxis> axisz = shot[ *plot->axisZ()];
			if(axisz) {
				setVertex(s1);
				setVertex(s2);
				setVertex(s7);
				setVertex(s5);
				setVertex(s1);
				setVertex(s3);
				setVertex(s6);
				setVertex(s5);
			}
			endQuad();
		}
	}
}
void
XQGraphPainter::drawOffScreenGrids(const Snapshot &shot) {
	if(shot.size(m_graph->plots())) {
		const auto &plots_list( *shot.list(m_graph->plots()));
		for(auto it = plots_list.begin(); it != plots_list.end(); it++) {
			auto plot = dynamic_pointer_cast<XPlot>( *it);
			plot->drawGrid(shot, this, m_bTilted);
		}
	}
}
void
XQGraphPainter::drawOffScreenPoints(const Snapshot &shot) {
	if(shot.size(m_graph->plots())) {
		const auto &plots_list( *shot.list(m_graph->plots()));
		for(auto it = plots_list.begin(); it != plots_list.end(); it++) {
			auto plot = static_pointer_cast<XPlot>( *it);
			plot->drawPlot(shot, this);
		}
	}
}
void
XQGraphPainter::drawOffScreenAxes(const Snapshot &shot) {
	if(shot.size(m_graph->axes())) {
		const auto &axes_list( *shot.list(m_graph->axes()));
		for(auto it = axes_list.begin(); it != axes_list.end(); it++) {
			auto axis = static_pointer_cast<XAxis>( *it);
			if((axis->direction() != XAxis::DirAxisZ) || m_bTilted)
				axis->drawAxis(shot, this);
		}
	}
}
