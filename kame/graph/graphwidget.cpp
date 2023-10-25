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
#include "graphwidget.h"
#include "graph.h"
#include <QDialog>
#include <QPixmap>
#include <QPainter>
#include <QImage>
#include <QLayout>
#include <QMouseEvent>
#include "graphpainter.h"
#include "measure.h"
#include "graphdialogconnector.h"
#include "ui_graphdialog.h"

typedef QForm<QDialog, Ui_DlgGraphSetup> DlgGraphSetup;

XQGraph::XQGraph( QWidget* parent, Qt::WindowFlags fl ) :
#ifdef USE_QGLWIDGET
    QGLWidget( QGLFormat(QGL::AlphaChannel | QGL::DoubleBuffer | QGL::Rgba |
                         QGL::DepthBuffer | QGL::AccumBuffer |
                         QGL::SampleBuffers)
               , parent, 0, Qt::WindowFlags(fl | Qt::WA_PaintOnScreen)) {
    setAutoFillBackground(false);
#else
    QOpenGLWidget(parent) {

    QSurfaceFormat format;
    format.setAlphaBufferSize(8);
    format.setDepthBufferSize(24);
//#ifndef __APPLE__
////    format.setSamples(4); //osx retina cannot handle this properly.
//#endif
#ifdef USE_PBO
    format.setVersion(3, 3);
#else
    format.setVersion(2, 0);
#endif
//    format.setSwapBehavior(QSurfaceFormat::DoubleBuffer);
//    format.setProfile(QSurfaceFormat::CoreProfile);
//   format.setProfile(QSurfaceFormat::OpenGLContextProfile::CompatibilityProfile);
   setFormat(format);
#endif
//    if( !parent->layout() ) {
//        parent->setLayout(new QHBoxLayout(this));
//        parent->layout()->addWidget(this);
//    }
    setMouseTracking(true);
}
XQGraph::~XQGraph() {
    m_painter.reset();
}
void
XQGraph::setGraph(const shared_ptr<XGraph> &graph) {
    m_conDialog.reset();
    m_painter.reset();
    m_graph = graph;
//    if(graph && !isHidden()) {
//		showEvent(NULL);
//    }

//    graph->iterate_commit([=](Transaction &tr){
//        m_planeSelectionTool = XNode::createOrphan<XBoolNode>("", true, tr);
//        m_axisSelectionTool = XNode::createOrphan<XBoolNode>("", true, tr);
//    });

//    m_activate1DAreaTool->iterate_commit([=](Transaction &tr){
//        m_lsn1DAreaTouched = tr[ *m_activate2DAreaTool].onValueChanged().connectWeakly
//            (shared_from_this(), &XQGraph::onSelAxisChanged, Listener::FLAG_MAIN_THREAD_CALL);
//    });

}
void
XQGraph::activateAxisSelectionTool(XAxis::AxisDirection dir, const XString &tool_desc) {
    m_toolDesc = tool_desc;
    m_toolDirX = dir;
    m_isAxisSelectionByTool = true;
    m_isPlaneSelectionByTool = false;
}
void
XQGraph::activatePlaneSelectionTool(XAxis::AxisDirection dirx, XAxis::AxisDirection diry, const XString &tool_desc) {
    m_toolDesc = tool_desc;
    m_toolDirX = dirx;
    m_toolDirY = diry;
    m_isPlaneSelectionByTool = true;
    m_isAxisSelectionByTool = false;
}
void
XQGraph::mousePressEvent ( QMouseEvent* e) {
	if( !m_painter ) return;
	XQGraphPainter::SelectionMode mode;
	switch (e->button()) {
	case Qt::RightButton:
        if(m_isPlaneSelectionByTool) {
            m_toolDesc = {};
            m_isPlaneSelectionByTool = false;
        }
        mode = XQGraphPainter::SelectionMode::SelAxis;
		break;
	case Qt::LeftButton:
        if(m_isAxisSelectionByTool) {
            m_toolDesc = {};
            m_isAxisSelectionByTool = false;
        }
        if(QApplication::queryKeyboardModifiers() & Qt::ShiftModifier)
            mode = XQGraphPainter::SelectionMode::TiltTracking;
        else
            mode = XQGraphPainter::SelectionMode::SelPlane;
		break;
    case Qt::MiddleButton:
        m_isPlaneSelectionByTool = false;
        m_isAxisSelectionByTool = false;
        m_toolDesc = {};
        mode = XQGraphPainter::SelectionMode::TiltTracking;
		break;
	default:
        mode = XQGraphPainter::SelectionMode::SelNone;
		break;
	}
    makeCurrent(); //needed to select objects.
    m_painter->selectObjs(e->pos().x(), e->pos().y(), XQGraphPainter::SelectionState::SelStart, mode, m_toolDesc);
    doneCurrent();
}
void
XQGraph::mouseMoveEvent ( QMouseEvent* e) {
    static XTime lasttime = XTime::now();
	if(XTime::now() - lasttime < 0.033) return;
	if( !m_painter ) return;
//    makeCurrent(); //this makes collapse of texture during mouse over.
    m_painter->selectObjs(e->pos().x(), e->pos().y(), XQGraphPainter::SelectionState::Selecting, XQGraphPainter::SelectionMode::SelNone, m_toolDesc);
//    doneCurrent();
}
void
XQGraph::mouseReleaseEvent ( QMouseEvent* e) {
	if( !m_painter ) return;
//    makeCurrent();
    auto [r1, r2] = m_painter->selectObjs(e->pos().x(), e->pos().y(), XQGraphPainter::SelectionState::SelFinish, XQGraphPainter::SelectionMode::SelNone, m_toolDesc);
    auto [axis1, src1, dst1] = r1;
    auto [axis2, src2, dst2] = r2;
    Snapshot shot( *m_graph);
    if(m_isAxisSelectionByTool
        && axis1 && (axis1->direction() == m_toolDirX)) {
        XGraph::VFloat vsrc1 = axis1->screenToVal(shot, src1);
        XGraph::VFloat vdst1 = axis1->screenToVal(shot, dst1);
        onAxisSelectedByTool().talk(Snapshot( *m_graph), std::tuple<XString, XGraph::VFloat, XGraph::VFloat>{m_toolDesc, vsrc1, vdst1});
    }
    if(m_isPlaneSelectionByTool && axis1 && axis2) {
        XGraph::VFloat vsrc1 = axis1->screenToVal(shot, src1);
        XGraph::VFloat vsrc2 = axis2->screenToVal(shot, src1);
        XGraph::VFloat vdst1 = axis1->screenToVal(shot, dst1);
        XGraph::VFloat vdst2 = axis2->screenToVal(shot, dst1);
        if((axis1->direction() == m_toolDirY) && (axis2->direction() == m_toolDirX)) {
             //swaps 1 and 2.
            onPlaneSelectedByTool().talk(Snapshot( *m_graph),
                std::tuple<XString, XGraph::ValPoint,XGraph::ValPoint, XQGraph*>
                                         {m_toolDesc, {vsrc2, vsrc1}, {vdst2, vdst1}, this});
        }
        else if((axis1->direction() == m_toolDirX) && (axis2->direction() == m_toolDirY)) {
            onPlaneSelectedByTool().talk(Snapshot( *m_graph),
                std::tuple<XString, XGraph::ValPoint,XGraph::ValPoint, XQGraph*>
                                         {m_toolDesc, {vsrc1, vsrc2}, {vdst1, vdst2}, this});
        }
    }
    m_isPlaneSelectionByTool = false;
    m_isAxisSelectionByTool = false;
    m_toolDesc = {};
//    doneCurrent();
}
void
XQGraph::mouseDoubleClickEvent ( QMouseEvent* e) {
	e->accept();
    m_isPlaneSelectionByTool = false;
    m_isAxisSelectionByTool = false;
    m_toolDesc = {};
    if( !m_painter ) return;
    if(QApplication::queryKeyboardModifiers() & Qt::ShiftModifier) {
        return;
    }
	if(m_graph) { 
		switch (e->button()) {
		case Qt::RightButton:
			m_painter->showHelp();
			break;
		case Qt::LeftButton:
			m_conDialog = xqcon_create<XQGraphDialogConnector>(
				m_graph,
                new DlgGraphSetup(parentWidget()));
            //\todo setAttribute Qt::WA_DeleteOnClose
			break;
        case Qt::MiddleButton:
			break;
		default:
			break;
		}
	}
}
void
XQGraph::wheelEvent ( QWheelEvent *e) {
    e->accept();
    makeCurrent();
    if(m_painter )
#if QT_VERSION >= QT_VERSION_CHECK(6,0,0)
        m_painter->wheel(e->position().x(), e->position().y(), (double)e->angleDelta().y() / 8.0);
#else
        m_painter->wheel(e->pos().x(), e->pos().y(), (double)e->delta() / 8.0);
#endif
    doneCurrent();
}
void
XQGraph::showEvent ( QShowEvent *) {
}
void
XQGraph::hideEvent ( QHideEvent * ) {
    m_isPlaneSelectionByTool = false;
    m_isAxisSelectionByTool = false;
    m_toolDesc = {};
    m_conDialog.reset();
}
//! openGL stuff
void
XQGraph::initializeGL() {
    shared_ptr<XGraph> graph = m_graph;
    // m_painter will be re-set in the constructor.
    new XQGraphPainter(graph, this);
    m_painter->initializeGL();
}
void
XQGraph::resizeGL ( int width, int height ) {
    // be aware of retina display.
    double pixel_ratio = devicePixelRatio();
    if(m_painter ) {
        glViewport( 0, 0, (GLint)(width * pixel_ratio),
                (GLint)(height * pixel_ratio));
        m_painter->m_pixel_ratio = pixel_ratio;
        m_painter->resizeGL(width, height);
    }
}

#ifndef QOPENGLWIDGET_QPAINTER_ATEND
void XQGraph::paintEvent(QPaintEvent *event) {
    //overpaint huck
    makeCurrent();
    paintGL();
}
#endif
void
XQGraph::paintGL() {
    if(m_painter ) {
        // be aware of retina display.
        double pixel_ratio = devicePixelRatio();
        if( m_painter->m_pixel_ratio != pixel_ratio) {
            fprintf(stderr, "DevicePixelRatio has been changed to %f\n", pixel_ratio);
            resizeGL(width(), height());
        }
        m_painter->paintGL();
    }
}

