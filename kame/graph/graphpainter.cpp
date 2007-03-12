/***************************************************************************
		Copyright (C) 2002-2007 Kentaro Kitagawa
		                   kitagawa@scphys.kyoto-u.ac.jp
		
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
#include <klocale.h>

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
	m_bReqHelp(false)
{
	openFont();
	item->m_painter.reset(this);
	m_lsnRedraw = graph->onUpdate().connectWeak(
        shared_from_this(), &XQGraphPainter::onRedraw,
        XListener::FLAG_MAIN_THREAD_CALL | XListener::FLAG_AVOID_DUP | XListener::FLAG_DELAY_ADAPTIVE);
}

float
XQGraphPainter::resScreen() {
    return 1.0f/max(m_pItem->width(), m_pItem->height());
} 

void
XQGraphPainter::repaintGraph(int x1, int y1, int x2, int y2)
{
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
XQGraphPainter::findAxis(const XGraph::ScrPoint &s1)
{
    shared_ptr<XAxis> found_axis;
    double zmin = 0.1;
    atomic_shared_ptr<const XNode::NodeList> axes_list(m_graph->axes()->children());
    if(axes_list) { 
        for(XNode::NodeList::const_iterator it = axes_list->begin(); it != axes_list->end(); it++)
		{
			shared_ptr<XAxis> axis = dynamic_pointer_cast<XAxis>(*it);
			XGraph::SFloat z1;
			XGraph::ScrPoint s2;
			axis->axisToScreen(axis->screenToAxis(s1), &s2);
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
XQGraphPainter::findPlane(const XGraph::ScrPoint &s1,
						  shared_ptr<XAxis> *axis1, shared_ptr<XAxis> *axis2)
{
	double zmin = 0.1;
	shared_ptr<XPlot> plot_found;
	atomic_shared_ptr<const XNode::NodeList> plots_list(m_graph->plots()->children());
	if(plots_list) { 
		for(XNode::NodeList::const_iterator it = plots_list->begin(); it != plots_list->end(); it++)
		{
			shared_ptr<XPlot> plot = dynamic_pointer_cast<XPlot>(*it);
			XGraph::GPoint g1;
			shared_ptr<XAxis> axisx = *plot->axisX();
			shared_ptr<XAxis> axisy = *plot->axisY();
			shared_ptr<XAxis> axisz = *plot->axisZ();
			plot->screenToGraph(s1, &g1);
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
XQGraphPainter::selectObjs(int x, int y, SelectionState state, SelectionMode mode)
{
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
				m_foundPlane = findPlane(m_startScrPos, &m_foundPlaneAxis1, &m_foundPlaneAxis2);
			m_finishScrPos = m_startScrPos;
			break;
		case SelAxis:
			m_foundAxis.reset();
			z = selectAxis(x, y, 
						   (int)(SELECT_WIDTH * m_pItem->width()),
						   (int)(SELECT_WIDTH * m_pItem->height()),
						   &m_startScrPos, &m_startScrDX, &m_startScrDY);
			if(z < 1.0) m_foundAxis = findAxis(m_startScrPos);
			m_finishScrPos = m_startScrPos;
			break;
		default:
			break;
		}
		break;
	case Selecting:
		//resotre mode
		mode = m_selectionModeNow;
		break;
	case SelFinish:
		//resoter mode
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
            m_foundPlane = findPlane(m_finishScrPos, &m_foundPlaneAxis1, &m_foundPlaneAxis2);
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
	    if((x == m_selStartPos[0]) && (y == m_selStartPos[1])) {
			switch(mode) {
			case SelPlane:
				break;
			case SelAxis:
            {
                XScopedLock<XGraph> lock(*m_graph);
				if(!m_foundAxis) {
					//if no axis, autoscale all axes
                    atomic_shared_ptr<const XNode::NodeList> axes_list(m_graph->axes()->children());
                    if(axes_list) { 
                        for(XNode::NodeList::const_iterator it = axes_list->begin(); it != axes_list->end(); it++)
						{
							shared_ptr<XAxis> axis = dynamic_pointer_cast<XAxis>(*it);
							if(axis->autoScale()->isUIEnabled())
								axis->autoScale()->value(true);
						}
                    }
				}
				else {
					if(m_foundAxis->autoScale()->isUIEnabled())
						m_foundAxis->autoScale()->value(true);
				}
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
			XScopedLock<XGraph> lock(*m_graph);
			switch(mode) {
			case SelPlane:
				if(m_foundPlane && !(m_startScrPos == m_finishScrPos) ) {
					XGraph::VFloat src1 = m_foundPlaneAxis1->screenToVal(m_startScrPos);
					XGraph::VFloat src2 = m_foundPlaneAxis2->screenToVal(m_startScrPos);
					XGraph::VFloat dst1 = m_foundPlaneAxis1->screenToVal(m_finishScrPos);
					XGraph::VFloat dst2 = m_foundPlaneAxis2->screenToVal(m_finishScrPos);
				
					if(m_foundPlaneAxis1->minValue()->isUIEnabled())
						m_foundPlaneAxis1->minValue()->value(double(min(src1, dst1)));
					if(m_foundPlaneAxis1->maxValue()->isUIEnabled())
						m_foundPlaneAxis1->maxValue()->value(double(max(src1, dst1)));
					if(m_foundPlaneAxis1->autoScale()->isUIEnabled())
						m_foundPlaneAxis1->autoScale()->value(false);
					if(m_foundPlaneAxis2->minValue()->isUIEnabled())
						m_foundPlaneAxis2->minValue()->value(double(min(src2, dst2)));
					if(m_foundPlaneAxis2->maxValue()->isUIEnabled())
						m_foundPlaneAxis2->maxValue()->value(double(max(src2, dst2)));
					if(m_foundPlaneAxis2->autoScale()->isUIEnabled())
						m_foundPlaneAxis2->autoScale()->value(false);
				
				}
				break;
			case SelAxis:
				if(m_foundAxis && !(m_startScrPos == m_finishScrPos) ) {
					XGraph::VFloat src = m_foundAxis->screenToVal(m_startScrPos);
					XGraph::VFloat dst = m_foundAxis->screenToVal(m_finishScrPos);
					double _min = std::min(src, dst);
					double _max = std::max(src, dst);
					if(m_foundAxis->minValue()->isUIEnabled())
						m_foundAxis->minValue()->value(_min);
					if(m_foundAxis->maxValue()->isUIEnabled())
						m_foundAxis->maxValue()->value(_max);
					if(m_foundAxis->autoScale()->isUIEnabled())
						m_foundAxis->autoScale()->value(false);
				}
				break;
			default:
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
XQGraphPainter::zoom(double zoomscale, int , int )
{
	XGraph::ScrPoint s1(0.5, 0.5, 0.5);
  
	XScopedLock<XGraph> lock(*m_graph);
	atomic_shared_ptr<const XNode::NodeList> axes_list(m_graph->axes()->children());
	if(axes_list) { 
		for(XNode::NodeList::const_iterator it = axes_list->begin(); it != axes_list->end(); it++)
		{
			shared_ptr<XAxis> axis = dynamic_pointer_cast<XAxis>(*it);
			if(axis->autoScale()->isUIEnabled())
				axis->autoScale()->value(false);
		}
	}
	m_graph->zoomAxes(resScreen(), zoomscale, s1);
}
void
XQGraphPainter::onRedraw(const shared_ptr<XGraph> &)
{
	redrawOffScreen();
	repaintGraph(0, 0, m_pItem->width(), m_pItem->height() );  
}
void
XQGraphPainter::drawOnScreenObj()
{
	QString msg = "";
//   if(SelectionStateNow != Selecting) return;
	switch ( m_selectionModeNow )
	{
	case SelNone:
		if(m_foundPlane) {
			XGraph::VFloat dst1 = m_foundPlaneAxis1->screenToVal(m_finishScrPos);
			XGraph::VFloat dst1dx = m_foundPlaneAxis1->screenToVal(m_finishScrDX) - dst1;
			XGraph::VFloat dst1dy = m_foundPlaneAxis1->screenToVal(m_finishScrDY) - dst1;
			XGraph::VFloat dst2 = m_foundPlaneAxis2->screenToVal(m_finishScrPos);
			XGraph::VFloat dst2dx = m_foundPlaneAxis2->screenToVal(m_finishScrDX) - dst2;
			XGraph::VFloat dst2dy = m_foundPlaneAxis2->screenToVal(m_finishScrDY) - dst2;

			dst1 = setprec(dst1, sqrt(dst1dx*dst1dx + dst1dy*dst1dy));
			dst2 = setprec(dst2, sqrt(dst2dx*dst2dx + dst2dy*dst2dy));
			msg += QString("(%1, %2)")
				.arg(m_foundPlaneAxis1->valToString(dst1))
				.arg(m_foundPlaneAxis2->valToString(dst2));
		}
		else {
			msg = KAME::i18n("R-DBL-CLICK TO SHOW HELP");
		}
		break;
	case SelPlane:
		if(m_foundPlane && !(m_startScrPos == m_finishScrPos) ) {
			XGraph::VFloat src1 = m_foundPlaneAxis1->screenToVal(m_startScrPos);
			XGraph::VFloat src1dx = m_foundPlaneAxis1->screenToVal(m_startScrDX) - src1;
			XGraph::VFloat src1dy = m_foundPlaneAxis1->screenToVal(m_startScrDY) - src1;
			XGraph::VFloat src2 = m_foundPlaneAxis2->screenToVal(m_startScrPos);
			XGraph::VFloat src2dx = m_foundPlaneAxis2->screenToVal(m_startScrDX) - src2;
			XGraph::VFloat src2dy = m_foundPlaneAxis2->screenToVal(m_startScrDY) - src2;
			XGraph::VFloat dst1 = m_foundPlaneAxis1->screenToVal(m_finishScrPos);
			XGraph::VFloat dst1dx = m_foundPlaneAxis1->screenToVal(m_finishScrDX) - dst1;
			XGraph::VFloat dst1dy = m_foundPlaneAxis1->screenToVal(m_finishScrDY) - dst1;
			XGraph::VFloat dst2 = m_foundPlaneAxis2->screenToVal(m_finishScrPos);
			XGraph::VFloat dst2dx = m_foundPlaneAxis2->screenToVal(m_finishScrDX) - dst2;
			XGraph::VFloat dst2dy = m_foundPlaneAxis2->screenToVal(m_finishScrDY) - dst2;

			src1 = setprec(src1, sqrt(src1dx*src1dx + src1dy*src1dy));
			src2 = setprec(src2, sqrt(src2dx*src2dx + src2dy*src2dy));
			dst1 = setprec(dst1, sqrt(dst1dx*dst1dx + dst1dy*dst1dy));
			dst2 = setprec(dst2, sqrt(dst2dx*dst2dx + dst2dy*dst2dy));
			msg += QString("(%1, %2) - (%3, %4)")
				.arg(m_foundPlaneAxis1->valToString(src1))
				.arg(m_foundPlaneAxis2->valToString(src2))
				.arg(m_foundPlaneAxis1->valToString(dst1))
				.arg(m_foundPlaneAxis2->valToString(dst2));
		
			XGraph::ScrPoint sd1, sd2;
			m_foundPlaneAxis1->valToScreen(dst1, &sd1);
			m_foundPlaneAxis1->valToScreen(src1, &sd2);
			sd1 -= sd2;
			sd1 += m_startScrPos;
			XGraph::ScrPoint ss1, ss2;
			m_foundPlaneAxis2->valToScreen(dst2, &ss1);
			m_foundPlaneAxis2->valToScreen(src2, &ss2);
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
			XGraph::VFloat src = m_foundAxis->screenToVal(m_startScrPos);
			XGraph::VFloat srcdx = m_foundAxis->screenToVal(m_startScrDX) - src;
			XGraph::VFloat srcdy = m_foundAxis->screenToVal(m_startScrDY) - src;
			XGraph::VFloat dst = m_foundAxis->screenToVal(m_finishScrPos);
			XGraph::VFloat dstdx = m_foundAxis->screenToVal(m_finishScrDX) - dst;
			XGraph::VFloat dstdy = m_foundAxis->screenToVal(m_finishScrDY) - dst;
				
			src = setprec(src, sqrt(srcdx*srcdx + srcdy*srcdy));
			dst = setprec(dst, sqrt(dstdx*dstdx + dstdy*dstdy));
		
			msg += QString("%1 - %2")
				.arg(m_foundAxis->valToString(src))
				.arg(m_foundAxis->valToString(dst));
		
			XGraph::GFloat src1 = m_foundAxis->valToAxis(src);
			XGraph::GFloat dst1 = m_foundAxis->valToAxis(dst);		
			beginQuad(true);
			setColor( clRed, 0.4 );
			XGraph::ScrPoint s1, s2, s3, s4;
			m_foundAxis->axisToScreen(src1, &s1);
			posOffAxis(m_foundAxis->dirVector(), &s1, -0.02);
			m_foundAxis->axisToScreen(src1, &s2);
			posOffAxis(m_foundAxis->dirVector(), &s2, +0.02);
			m_foundAxis->axisToScreen(dst1, &s3);
			posOffAxis(m_foundAxis->dirVector(), &s3, +0.02);
			m_foundAxis->axisToScreen(dst1, &s4);
			posOffAxis(m_foundAxis->dirVector(), &s4, -0.02);
			setVertex(s1);
			setVertex(s2);
			setVertex(s3);
			setVertex(s4);
			endQuad();
			beginLine();
			setColor( clBlue, 1.0 );
			m_foundAxis->axisToScreen(src1, &s1);
			posOffAxis(m_foundAxis->dirVector(), &s1, -0.1);
			m_foundAxis->axisToScreen(src1, &s2);
			posOffAxis(m_foundAxis->dirVector(), &s2, 0.05);
			setVertex(s1);
			setVertex(s2);
			m_foundAxis->axisToScreen(dst1, &s1);
			posOffAxis(m_foundAxis->dirVector(), &s1, -0.1);
			m_foundAxis->axisToScreen(dst1, &s2);
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
	m_onScreenMsg = msg.utf8();
}
void
XQGraphPainter::showHelp()
{
	m_bReqHelp = true;
	repaintGraph(0, 0, m_pItem->width(), m_pItem->height());
}
void
XQGraphPainter::drawOnScreenViewObj()
{
	//Draw Title
	setColor(*m_graph->titleColor());
	defaultFont();
	m_curAlign = AlignTop | AlignHCenter;
	drawText(XGraph::ScrPoint(0.5, 0.99, 0.01), *m_graph->label());
  
	if(m_onScreenMsg.length() ) {
		selectFont(m_onScreenMsg, XGraph::ScrPoint(0.6, 0.05, 0.01), XGraph::ScrPoint(1, 0, 0), XGraph::ScrPoint(0, 0.05, 0), 0);
	 	setColor(*m_graph->titleColor());
		m_curAlign = AlignBottom | AlignLeft;
  		drawText(XGraph::ScrPoint(0.01, 0.01, 0.01), m_onScreenMsg);
	}
	
	if(m_bReqHelp) drawOnScreenHelp();
}
void
XQGraphPainter::drawOnScreenHelp()
{
	float z = 0.99;
	setColor(*m_graph->backGround(), 0.3);
	beginQuad(true);
	setVertex(XGraph::ScrPoint(0.0, 0.0, z));
	setVertex(XGraph::ScrPoint(1.0, 0.0, z));
	setVertex(XGraph::ScrPoint(1.0, 1.0, z));
	setVertex(XGraph::ScrPoint(0.0, 1.0, z));
	endQuad();
	setColor(*m_graph->titleColor(), 0.55);
	double y = 1.0;
	beginQuad(true);
	setVertex(XGraph::ScrPoint(1.0 - y, 1.0 - y, z));
	setVertex(XGraph::ScrPoint(1.0 - y, y, z));
	setVertex(XGraph::ScrPoint(y, y, z));
	setVertex(XGraph::ScrPoint(y, 1.0 - y, z));
	endQuad();
	y -= 0.02;
	setColor(*m_graph->backGround(), 1.0);
	defaultFont();
	m_curAlign = AlignTop | AlignHCenter;
	drawText(XGraph::ScrPoint(0.5, y, z), KAME::i18n("QUICK HELP!"));
	m_curAlign = AlignVCenter | AlignLeft;
	y -= 0.1;
	double x = 0.1;
	double dy = -y/10;
	selectFont(KAME::i18n("Single Click Right Button on Axis : Auto-scale"), XGraph::ScrPoint(x,y,z), XGraph::ScrPoint(1, 0, 0), XGraph::ScrPoint(0, dy, 0), 0);
	
	drawText(XGraph::ScrPoint(x, y, z), KAME::i18n("Press Left Button on Plot : Manual Scale"));
	y += dy;
	drawText(XGraph::ScrPoint(x, y, z), KAME::i18n("Press Right Button along Axis: Manual Scale"));
	y += dy;
	drawText(XGraph::ScrPoint(x, y, z), KAME::i18n("Single Click Right Button on Axis : Auto-scale"));
	y += dy;
	drawText(XGraph::ScrPoint(x, y, z), KAME::i18n("Single Click Right Button elsewhere : Auto-scale all"));
	y += dy;
	drawText(XGraph::ScrPoint(x, y, z), KAME::i18n("Press Middle Button : Tilt plots"));
	y += dy;
	drawText(XGraph::ScrPoint(x, y, z), KAME::i18n("Single Click Middle Button : Reset tilting"));
	y += dy;
	drawText(XGraph::ScrPoint(x, y, z), KAME::i18n("Wheel around Center : (Un)Zoom all Plots"));
	y += dy;
	drawText(XGraph::ScrPoint(x, y, z), KAME::i18n("Wheel at Side : Tilt by 30deg."));
	y += dy;
	drawText(XGraph::ScrPoint(x, y, z), KAME::i18n("Double Click Left Button : Show Dialog"));
	y += dy;
	drawText(XGraph::ScrPoint(x, y, z), KAME::i18n("Double Click Right Button : This Help"));
}

void
XQGraphPainter::drawOffScreenStart()
{
	m_graph->setupRedraw(resScreen());
}
void
XQGraphPainter::drawOffScreenPlanes()
{
	setColor((QRgb)*m_graph->backGround(), 0.3);
	atomic_shared_ptr<const XNode::NodeList> plots_list(m_graph->plots()->children());
	if(plots_list) { 
		for(XNode::NodeList::const_iterator it = plots_list->begin(); it != plots_list->end(); it++)
		{
			shared_ptr<XPlot> plot = dynamic_pointer_cast<XPlot>(*it);
			XGraph::GPoint g1(0.0, 0.0, 0.0),
				g2(1.0, 0.0, 0.0),
				g3(0.0, 1.0, 0.0),
				g4(1.0, 1.0, 0.0),
				g5(0.0, 0.0, 1.0),
				g6(0.0, 1.0, 1.0),
				g7(1.0, 0.0, 1.0),
				g8(1.0, 1.0, 1.0);
			XGraph::ScrPoint s1, s2, s3, s4, s5, s6, s7, s8;
			plot->graphToScreen(g1, &s1);
			plot->graphToScreen(g2, &s2);
			plot->graphToScreen(g3, &s3);
			plot->graphToScreen(g4, &s4);
			plot->graphToScreen(g5, &s5);
			plot->graphToScreen(g6, &s6);
			plot->graphToScreen(g7, &s7);
			plot->graphToScreen(g8, &s8);
			beginQuad(true);
			setColor( *m_graph->backGround(), 0.2);
			setVertex(s1);
			setVertex(s2);
			setVertex(s4);
			setVertex(s3);
			shared_ptr<XAxis> axisz = *plot->axisZ();
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
XQGraphPainter::drawOffScreenGrids()
{
	atomic_shared_ptr<const XNode::NodeList> plots_list(m_graph->plots()->children());
	if(plots_list) { 
		for(XNode::NodeList::const_iterator it = plots_list->begin(); it != plots_list->end(); it++)
		{
			shared_ptr<XPlot> plot = dynamic_pointer_cast<XPlot>(*it);
			plot->drawGrid(this, m_bTilted);
		}
	}
}
void
XQGraphPainter::drawOffScreenPoints()
{
	atomic_shared_ptr<const XNode::NodeList> plots_list(m_graph->plots()->children());
	if(plots_list) { 
		for(XNode::NodeList::const_iterator it = plots_list->begin(); it != plots_list->end(); it++)
		{
			shared_ptr<XPlot> plot = dynamic_pointer_cast<XPlot>(*it);
			plot->drawPlot(this);
		}
	}
}
void
XQGraphPainter::drawOffScreenAxes()
{
	atomic_shared_ptr<const XNode::NodeList> axes_list(m_graph->axes()->children());
	if(axes_list) { 
		for(XNode::NodeList::const_iterator it = axes_list->begin(); it != axes_list->end(); it++)
		{
			shared_ptr<XAxis> axis = dynamic_pointer_cast<XAxis>(*it);
			if((axis->direction() != XAxis::DirAxisZ) || m_bTilted)
				axis->drawAxis(this);
		}
	}
}
