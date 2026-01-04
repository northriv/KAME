/***************************************************************************
        Copyright (C) 2002-2026 Kentaro Kitagawa
		                   kitag@issp.u-tokyo.ac.jp
		
		This program is free software; you can redistribute it and/or
		modify it under the terms of the GNU Library General Public
		License as published by the Free Software Foundation; either
		version 2 of the License, or (at your option) any later version.
		
		You should have received a copy of the GNU Library General 
		Public License and a list of authors along with this program; 
		see the files COPYING and AUTHORS.
***************************************************************************/
#include "graphpainter.h"
#include "graphwidget.h"
#include <QFont>
#include <QFontMetrics>
#include <QPainter>
#include "graphmathtool.h"

#define SELECT_WIDTH 0.02
#define SELECT_DEPTH 0.1

using std::min;
using std::max;

float
XQGraphPainter::resScreen() {
    return 1.0f / max(m_pItem->width(), m_pItem->height());
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

std::pair<XQGraphPainter::SelectedResult, XQGraphPainter::SelectedResult> XQGraphPainter::selectObjs(int x, int y, SelectionState state, SelectionMode mode,
    const XString& tool_desc) {
    m_foundPickableNode.reset();

    m_toolDescForSelection = tool_desc;
    std::pair<XQGraphPainter::SelectedResult, XQGraphPainter::SelectedResult> ret{};

	m_pointerLastPos[0] = x;
	m_pointerLastPos[1] = y;

	if(m_bReqHelp) {
        if(state != SelectionState::SelStart) return ret;
		m_bReqHelp = false;
        return ret;
	}

	m_selectionStateNow = state;
	switch(state) {
    case SelectionState::SelStart:
		m_selectionModeNow = mode;
		m_selStartPos[0] = x;
		m_selStartPos[1] = y;
		m_tiltLastPos[0] = x;
		m_tiltLastPos[1] = y;
		switch(mode) {
        case SelectionMode::SelPlane: {
                m_foundPlane.reset();
                auto [zmin, objid, scr, sdx, sdy] = selectPlane(x, y,
                                (int)(SELECT_WIDTH * m_pItem->width()),
                                (int)(SELECT_WIDTH * m_pItem->height()));
                if(zmin < 1.0)
                    m_foundPlane = findPlane(Snapshot( *m_graph), scr, &m_foundPlaneAxis1, &m_foundPlaneAxis2);
                m_finishScrPos = m_startScrPos = scr;
                m_startScrDX = sdx; m_startScrDY = sdy;
            }
			break;
        case SelectionMode::SelAxis: {
                m_foundAxis.reset();
                auto [zmin, objid, scr, sdx, sdy] = selectAxis(x, y,
                               (int)(SELECT_WIDTH * m_pItem->width()),
                               (int)(SELECT_WIDTH * m_pItem->height()));
                if(zmin < 1.0) m_foundAxis = findAxis(Snapshot( *m_graph), scr);
                m_finishScrPos = m_startScrPos = scr;
                m_startScrDX = sdx; m_startScrDY = sdy;
            }
			break;
		default:
			break;
		}
		break;
    case SelectionState::Selecting:
		//restore mode
		mode = m_selectionModeNow;
		break;
    case SelectionState::SelFinish:
        //restore mode
		mode = m_selectionModeNow;
        m_selectionModeNow = SelectionMode::SelNone;
		break;
    }
	switch(mode) {
    case SelectionMode::SelNone: {
            m_foundPlane.reset();
            auto [zmin, objid, scr, sdx, sdy] = selectPlane(x, y,
                            (int)(SELECT_WIDTH * m_pItem->width()),
                            (int)(SELECT_WIDTH * m_pItem->height()));
            if(zmin < 1.0)
                m_foundPlane = findPlane(Snapshot( *m_graph), scr, &m_foundPlaneAxis1, &m_foundPlaneAxis2);
            m_finishScrPos = scr;
            m_finishScrDX = sdx; m_finishScrDY = sdy;
        }
		break;
    case SelectionMode::SelPlane: {
            auto [zmin, objid, scr, sdx, sdy] = selectPlane(x, y,
                        (int)(SELECT_WIDTH * m_pItem->width()),
                        (int)(SELECT_WIDTH * m_pItem->height()));
            m_finishScrPos = scr;
            m_finishScrDX = sdx; m_finishScrDY = sdy;
        }
		break;
    case SelectionMode::SelAxis: {
            auto [zmin, objid, scr, sdx, sdy] = selectAxis(x, y,
                       (int)(SELECT_WIDTH * m_pItem->width()),
                       (int)(SELECT_WIDTH * m_pItem->height()));
            m_finishScrPos = scr;
            m_finishScrDX = sdx; m_finishScrDY = sdy;
        }
		break;
    case SelectionMode::TiltTracking:
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

    auto [zmin0, oso, scr0, sdx0, sdy0] = selectOSO(x, y,
                    (int)(SELECT_WIDTH * m_pItem->width()),
                    (int)(SELECT_WIDTH * m_pItem->height()));
    if(auto o = dynamic_pointer_cast<OnScreenPickableObject>(oso.lock())) {
        m_foundPickableNode = o->pickableNode();
//                fprintf(stderr, "OSO:%p from node:%p\n", o.get(), o->pickableNode().get());
    }

    if(state == SelectionState::SelFinish) {
        if(tool_desc.empty() && (abs(x - m_selStartPos[0]) < 3) && (abs(y - m_selStartPos[1]) < 3)) {
            //mouse movement was nearly zero.
			switch(mode) {
            case SelectionMode::SelPlane:
				break;
            case SelectionMode::SelAxis:
				m_graph->iterate_commit([=](Transaction &tr){
					if( !m_foundAxis) {
                        if(auto node = dynamic_pointer_cast<XGraphMathTool>(m_nodeHighlighten.lock())) {
                            auto p = node->parentList();
                            Snapshot shot_list( *p);
                            if(auto list = dynamic_pointer_cast<XGraph1DMathToolList>(p)) {
                                trans( *list).popupMenu().talk(shot_list, x, y, node.get());
                            }
                            if(auto list = dynamic_pointer_cast<XGraph2DMathToolList>(p)) {
                                trans( *list).popupMenu().talk(shot_list, x, y, node.get());
                            }
                        }
                        else {
                            bool all_autoscaled = true;
                            //Autoscales all axes
                            if(tr.size(m_graph->axes())) {
                                const auto &axes_list( *tr.list(m_graph->axes()));
                                for(auto it = axes_list.begin(); it != axes_list.end(); it++) {
                                    shared_ptr<XAxis> axis = static_pointer_cast<XAxis>(*it);
                                    if(tr[ *axis->autoScale()].isUIEnabled())
                                        if( !tr[ *axis->autoScale()]) {
                                            all_autoscaled = false; //at least 1 axis is manually scaled.
                                            tr[ *axis->autoScale()] = true;
                                        }
                                }
                            }
                            if(all_autoscaled) {
                                //alt way for middle click.
                                viewRotate(0.0, 0.0, 0.0, 0.0, true);
                            }
                        }
					}
					else {
						if(tr[ *m_foundAxis->autoScale()].isUIEnabled())
							tr[ *m_foundAxis->autoScale()] = true;
					}
                });
				break;
            case SelectionMode::TiltTracking:
				viewRotate(0.0, 0.0, 0.0, 0.0, true);
                break;
			default:
				break;
			}
	    }
	    else {
            m_graph->iterate_commit([&](Transaction &tr){
				switch(mode) {
                case SelectionMode::SelPlane:
					if(m_foundPlane && !(m_startScrPos == m_finishScrPos) ) {
                        XGraph::VFloat src1 = m_foundPlaneAxis1->screenToVal(tr, m_startScrPos);
                        XGraph::VFloat src2 = m_foundPlaneAxis2->screenToVal(tr, m_startScrPos);
                        XGraph::VFloat dst1 = m_foundPlaneAxis1->screenToVal(tr, m_finishScrPos);
                        XGraph::VFloat dst2 = m_foundPlaneAxis2->screenToVal(tr, m_finishScrPos);
                        if(tool_desc.length()) {
                            ret = std::pair<SelectedResult, SelectedResult>{{m_foundPlaneAxis1, src1, dst1},
                                {m_foundPlaneAxis2, src2, dst2}};
                            return;
                        }

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
                case SelectionMode::SelAxis:
					if(m_foundAxis && !(m_startScrPos == m_finishScrPos) ) {
                        XGraph::VFloat src = m_foundAxis->screenToVal(tr, m_startScrPos);
                        XGraph::VFloat dst = m_foundAxis->screenToVal(tr, m_finishScrPos);
                        if(tool_desc.length()) {
                            ret = std::pair<SelectedResult, SelectedResult>{{m_foundAxis, src, dst}, {}};
                            return;
                        }
                        double _min = std::min(src, dst);
						double _max = std::max(src, dst);
                        if(tr[ *m_foundAxis->minValue()].isUIEnabled())
							tr[ *m_foundAxis->minValue()] = _min;
						if(tr[ *m_foundAxis->maxValue()].isUIEnabled())
							tr[ *m_foundAxis->maxValue()] = _max;
						if(tr[ *m_foundAxis->autoScale()].isUIEnabled())
							tr[ *m_foundAxis->autoScale()] = false;
					}
					break;
				default:
					break;
                }
            });
        }
	}
    if((state !=SelectionState::Selecting) && (state != SelectionState::SelStart))
        requestRepaint(); //supresses frequent update.
    else
        m_pItem->update();
    return ret;
}

void
XQGraphPainter::wheel(int x, int y, double deg)
{
    if(fabs(deg) < 1.0) return;
	double a = ((double)x / m_pItem->width() - 0.5);
	double b = ((double)y / m_pItem->height() - 0.5);
    if( max(fabs(a), fabs(b)) < 0.45) { //not near the edges.
        zoom(min(1.15, max(0.85, exp(deg * 0.04))), x, y);
	}
	else {
        glGetError(); //reset error

        if( (a - b) * (a + b) > 0 ) {
			viewRotate(30.0 * deg / fabs(deg), -1.0, 0.0, 0.0, false);
		}
		else {
			viewRotate(30.0 * deg / fabs(deg), 0.0, 1.0, 0.0, false);
		}

        m_pItem->update();
    }
}
void
XQGraphPainter::zoom(double zoomscale, int , int ) {
    XGraph::ScrPoint s1(0.5, 0.5, 0.5);
    if( m_selectionModeNow == SelectionMode::SelNone) {
        if(m_foundPlane) {
            s1 = m_finishScrPos; //zoom center.
        }
        m_foundPlane.reset();
    }
	m_graph->iterate_commit([=](Transaction &tr){
		if(tr.size(m_graph->axes())) {
			const auto &axes_list( *tr.list(m_graph->axes()));
			for(auto it = axes_list.begin(); it != axes_list.end(); ++it) {
				shared_ptr<XAxis> axis = static_pointer_cast<XAxis>( *it);
				if(tr[ *axis->autoScale()].isUIEnabled())
					tr[ *axis->autoScale()] = false;
			}
		}
		m_graph->zoomAxes(tr, resScreen(), zoomscale, s1);
    });
}
void
XQGraphPainter::onRedraw(const Snapshot &, XGraph *graph) {
    m_bIsRedrawNeeded = true;
    requestRepaint(); //may defer update.
}
void
XQGraphPainter::drawOnScreenObj(const Snapshot &shot) {
    QString msg = "";
    if(m_toolDescForSelection.length())
        msg = m_toolDescForSelection + i18n(" (Selecting) ");
//   if(SelectionStateNow != Selecting) return;
	switch ( m_selectionModeNow ) {
    case SelectionMode::SelNone:
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
            if(m_toolDescForSelection.length())
                msg = m_toolDescForSelection;
            else
                msg = i18n("R-DBL-CLICK TO SHOW HELP");
		}
		break;
    case SelectionMode::SelPlane:
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

            auto oso = createOneTimeOnScreenObject<OnScreenRectObject>(OnScreenRectObject::Type::Selection, graph());
            oso->setBaseColor(clBlue);
            oso->placeObject(m_startScrPos, sd1, m_finishScrPos, ss1);
        }
		break;
    case SelectionMode::SelAxis:
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
			XGraph::ScrPoint s1, s2, s3, s4;
			m_foundAxis->axisToScreen(shot, src1, &s1);
			posOffAxis(m_foundAxis->dirVector(), &s1, -0.02);
			m_foundAxis->axisToScreen(shot, src1, &s2);
			posOffAxis(m_foundAxis->dirVector(), &s2, +0.02);
			m_foundAxis->axisToScreen(shot, dst1, &s3);
			posOffAxis(m_foundAxis->dirVector(), &s3, +0.02);
			m_foundAxis->axisToScreen(shot, dst1, &s4);
			posOffAxis(m_foundAxis->dirVector(), &s4, -0.02);
            auto oso = createOneTimeOnScreenObject<OnScreenRectObject>(OnScreenRectObject::Type::Selection, graph());
            oso->setBaseColor(clRed);
            oso->placeObject(s1, s2, s3, s4);
            m_foundAxis->axisToScreen(shot, src1, &s1);
			posOffAxis(m_foundAxis->dirVector(), &s1, -0.1);
            m_foundAxis->axisToScreen(shot, src1, &s4);
            posOffAxis(m_foundAxis->dirVector(), &s4, 0.05);
            m_foundAxis->axisToScreen(shot, dst1, &s2);
            posOffAxis(m_foundAxis->dirVector(), &s2, -0.1);
            m_foundAxis->axisToScreen(shot, dst1, &s3);
            posOffAxis(m_foundAxis->dirVector(), &s3, 0.05);
            oso = createOneTimeOnScreenObject<OnScreenRectObject>(OnScreenRectObject::Type::BorderLines, graph());
            oso->setBaseColor(clBlue);
            oso->placeObject(s1, s2, s3, s4);
        }
		break;
    case SelectionMode::TiltTracking:
		break;
	default:
		break;
	}
	m_onScreenMsg = msg;
}
void
XQGraphPainter::showHelp() {
	m_bReqHelp = true;
    m_pItem->update();
}
void
XQGraphPainter::drawOnScreenViewObj(const Snapshot &shot) {
    //Draws Title
    auto oso = createOneTimeOnScreenObject<OnScreenTextObject>(graph());
    oso->setAlignment(Qt::AlignTop | Qt::AlignHCenter);
    oso->setBaseColor(shot[ *m_graph->titleColor()]);
    if(shot[ *m_graph->onScreenStrings()].to_str().length()) {
        oso->drawText(XGraph::ScrPoint(0.5, 0.99, 0.01), shot[ *m_graph->label()].to_str() + " " + shot[ *m_graph->onScreenStrings()].to_str());
    }
    else {
        oso->drawText(XGraph::ScrPoint(0.5, 0.99, 0.01), shot[ *m_graph->label()]);
    }

    if(auto node = m_nodeHighlighten.lock()) {
        if(m_foundPickableNode.lock() != node) {
        //Undo highlighting the OSO object.
            m_nodeHighlighten.reset();
            if(auto o = dynamic_pointer_cast<XGraphMathTool>(node)) {
                o->highlight(false, shared_from_this());
            }
        }
    }
    if(auto node = m_foundPickableNode.lock()) {
        //highlighting the OSO object.
        if(auto o = dynamic_pointer_cast<XGraphMathTool>(node)) {
            if(node != m_nodeHighlighten.lock()) {
                o->highlight(true, shared_from_this());
            }
            m_onScreenMsg += " " + o->getMenuLabel();
        }
        m_nodeHighlighten = node; //avoids highlighting again, and to recover the state when unhovered.
    }
    if(m_onScreenMsg.length() ) {
        //Draws message at the bottom left corner
        auto oso = createOneTimeOnScreenObject<OnScreenTextObject>(graph());
        oso->selectFont(m_onScreenMsg, XGraph::ScrPoint(0.6, 0.05, 0.01), XGraph::ScrPoint(1, 0, 0), XGraph::ScrPoint(0, 0.05, 0), 0);
        oso->setAlignment(Qt::AlignBottom | Qt::AlignLeft);
        oso->setBaseColor(shot[ *m_graph->titleColor()]);
        oso->drawText(XGraph::ScrPoint(0.01, 0.01, 0.01), m_onScreenMsg);
    }
    //Legends
	if(shot[ *m_graph->drawLegends()] &&
            (m_selectionModeNow == SelectionMode::SelNone)) {
		if(shot.size(m_graph->plots())) {
            const XNode::NodeList &plots_list( *shot.list(m_graph->plots()));
            XString longest_label;
            for(auto &&x: plots_list) {
                auto plot = static_pointer_cast<XPlot>(x);
                if(XString(shot[ *plot->label()]).length() > longest_label.length())
                    longest_label = shot[ *plot->label()];
            }
            float text_width = 0.13;
            float dy = 0.05; //height of each legend.

            float x1 = 0.77;
			float y1 = 0.81;
			if(m_pointerLastPos[0] > m_pItem->width() / 2)
                x1 = 1.08f - x1;
			if(m_pointerLastPos[1] < m_pItem->height() / 2)
				y1 = 1.0f - y1 + plots_list.size() * dy;
            float x2 = x1 - 0.01; //right edge of text field.

            auto oso = createOneTimeOnScreenObject<OnScreenTextObject>(graph());
            oso->setBaseColor(shot[ *m_graph->titleColor()]);
            oso->setAlignment(Qt::AlignVCenter | Qt::AlignRight);
            float z = 0.97;
            oso->selectFont(longest_label, XGraph::ScrPoint(text_width * 0.9, y1, z), XGraph::ScrPoint(-1, 0, 0), XGraph::ScrPoint(0, dy * 0.85, 0), 0);
			float x3 = x1 + 0.08;
            float y2 = y1 - plots_list.size() * dy;
			float y = y1;
            z = 0.99;
            for(auto it = plots_list.begin(); it != plots_list.end(); it++) {
				setColor(shot[ *m_graph->titleColor()], 1.0);
                auto plot = static_pointer_cast<XPlot>( *it);
                oso->drawText(XGraph::ScrPoint(x2,y,z), shot[ *plot->label()]);
                plot->drawLegend(shot, this, XGraph::ScrPoint((x3 + x1)/2, y, z),
                                 (x3 - x1)/1.5f, dy/1.2f);
				y -= dy;
			}
            z = 0.97;
            float minx = x1 - text_width;
            auto oso_rect = createOneTimeOnScreenObject<OnScreenRectObject>(OnScreenRectObject::Type::Legends, graph());
            oso_rect->setBaseColor(shot[ *m_graph->titleColor()]);
            oso_rect->placeObject(
                {minx, y1 + dy/2, z}, {minx, y2 + dy/2, z},
                {x3, y2 + dy/2, z}, {x3, y1 + dy/2, z});
        }
	}
}
void
XQGraphPainter::drawOnScreenHelp(const Snapshot &shot, QPainter *qpainter) {
    double y = 1.0;
    float z = 0.99;

    QColor cl(QRgb((unsigned int)shot[ *m_graph->backGround()]));
    cl.setAlpha(lrint(0.3 * 255));
    qpainter->fillRect(QRect(0, 0, m_pItem->width(), m_pItem->height()), cl);
    cl = QColor(QRgb((unsigned int)shot[ *m_graph->titleColor()]));
    cl.setAlpha(lrint(0.55 * 255));
    qpainter->fillRect(QRect(0, 0, m_pItem->width(), m_pItem->height()), cl);

    auto oso = createOneTimeOnScreenObject<OnScreenTextObject>(graph());
    oso->setBaseColor(shot[ *m_graph->backGround()]);
    oso->setAlignment(Qt::AlignVCenter | Qt::AlignLeft);
    y -= 0.1;
	double x = 0.1;
	double dy = -y/10;
    oso->selectFont(i18n("Single Click Right Button on Axis : Auto-scale"), XGraph::ScrPoint(x,y,z), XGraph::ScrPoint(1, 0, 0), XGraph::ScrPoint(0, dy, 0), 0);
    oso->setAlignment(Qt::AlignTop | Qt::AlignHCenter);
    oso->drawText(XGraph::ScrPoint(0.5, 1.0, z), i18n("QUICK HELP!"));
    oso->setAlignment(Qt::AlignVCenter | Qt::AlignLeft);

    oso->drawText(XGraph::ScrPoint(x, y, z), i18n("Press Left Button on Plot : Manual Scale"));
	y += dy;
    oso->drawText(XGraph::ScrPoint(x, y, z), i18n("Press Right Button along Axis: Manual Scale"));
	y += dy;
    oso->drawText(XGraph::ScrPoint(x, y, z), i18n("Single Click Right Button on Axis : Auto-scale"));
	y += dy;
    oso->drawText(XGraph::ScrPoint(x, y, z), i18n("Single Click Right Button elsewhere : Auto-scale all"));
	y += dy;
    oso->drawText(XGraph::ScrPoint(x, y, z), i18n("Press Middle Button : Tilt plots"));
	y += dy;
    oso->drawText(XGraph::ScrPoint(x, y, z), i18n("Single Click Middle Button : Reset tilting"));
	y += dy;
    oso->drawText(XGraph::ScrPoint(x, y, z), i18n("Wheel around Center : (Un)Zoom all Plots"));
	y += dy;
    oso->drawText(XGraph::ScrPoint(x, y, z), i18n("Wheel at Side : Tilt by 30deg."));
	y += dy;
    oso->drawText(XGraph::ScrPoint(x, y, z), i18n("Double Click Left Button : Show Dialog"));
	y += dy;
    oso->drawText(XGraph::ScrPoint(x, y, z), i18n("Double Click Right Button : This Help"));
}

Snapshot
XQGraphPainter::startDrawing() {
    return m_graph->iterate_commit([=](Transaction &tr){
        m_graph->setupRedraw(tr, resScreen(), (double)m_pItem->widthMM() / m_pItem->heightMM());
    });
}
void
XQGraphPainter::drawOffScreenPlanes(const Snapshot &shot, ObjClassColorR red_color_picking) {
    if(shot.size(m_graph->plots())) {
		const auto &plots_list( *shot.list(m_graph->plots()));
        unsigned int plot_num = 0;
        for(auto &&x: plots_list) {
            glColor4f((int)red_color_picking/256.0f, (plot_num/256u)/256.0f, (plot_num % 256u)/256.0f, 1.0f);
            plot_num++;

            auto plot = static_pointer_cast<XPlot>(x);
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
XQGraphPainter::drawOffScreenAxes(const Snapshot &shot, ObjClassColorR red_color_picking) {
    if(shot.size(m_graph->axes())) {
        const auto &axes_list( *shot.list(m_graph->axes()));
        unsigned int axis_num = 0;
        for(auto &&x: axes_list) {
            auto axis = static_pointer_cast<XAxis>(x);
            if((axis->direction() != XAxis::AxisDirection::Z) || m_bTilted) {
                if(red_color_picking == ObjClassColorR::None)
                    axis->drawAxis(shot, this);
                else {
                    glColor4f((int)red_color_picking/256.0f, (axis_num/256u)/256.0f, (axis_num%256u)/256.0f, 1.0f);
                    axis_num++;
                    constexpr double axistomarker = 0.05;
                    XGraph::ScrPoint s10,s11,s20,s21,vdir;
                    axis->axisToScreen(shot, 0.0, &s10);
                    axis->axisToScreen(shot, 1.0, &s20);
                    s11 = s10;
                    s21 = s20;
                    vdir = s20;
                    vdir -= s10;
                    posOffAxis(vdir, &s10, axistomarker);
                    posOffAxis(vdir, &s11, -axistomarker);
                    posOffAxis(vdir, &s20, axistomarker);
                    posOffAxis(vdir, &s21, -axistomarker);
                    beginQuad(true);
                    setVertex(s10);
                    setVertex(s11);
                    setVertex(s21);
                    setVertex(s20);
                    endQuad();
                }
            }
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
XQGraphPainter::drawOffScreenPoints(const Snapshot &shot, ObjClassColorR red_color_picking) {
	if(shot.size(m_graph->plots())) {
		const auto &plots_list( *shot.list(m_graph->plots()));
		for(auto it = plots_list.begin(); it != plots_list.end(); it++) {
			auto plot = static_pointer_cast<XPlot>( *it);
			plot->drawPlot(shot, this);
		}
	}
}

