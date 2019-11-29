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
}
void
XQGraph::mousePressEvent ( QMouseEvent* e) {
	if( !m_painter ) return;
	XQGraphPainter::SelectionMode mode;
	switch (e->button()) {
	case Qt::RightButton:
        mode = XQGraphPainter::SelectionMode::SelAxis;
		break;
	case Qt::LeftButton:
        if(QApplication::queryKeyboardModifiers() & Qt::ShiftModifier)
            mode = XQGraphPainter::SelectionMode::TiltTracking;
        else
            mode = XQGraphPainter::SelectionMode::SelPlane;
		break;
	case Qt::MidButton:
        mode = XQGraphPainter::SelectionMode::TiltTracking;
		break;
	default:
        mode = XQGraphPainter::SelectionMode::SelNone;
		break;
	}
    makeCurrent();
    m_painter->selectObjs(e->pos().x(), e->pos().y(), XQGraphPainter::SelectionState::SelStart, mode);
    doneCurrent();
}
void
XQGraph::mouseMoveEvent ( QMouseEvent* e) {
    static XTime lasttime = XTime::now();
	if(XTime::now() - lasttime < 0.033) return;
	if( !m_painter ) return;
    makeCurrent();
    m_painter->selectObjs(e->pos().x(), e->pos().y(), XQGraphPainter::SelectionState::Selecting);
    doneCurrent();
}
void
XQGraph::mouseReleaseEvent ( QMouseEvent* e) {
	if( !m_painter ) return;
    makeCurrent();
    m_painter->selectObjs(e->pos().x(), e->pos().y(), XQGraphPainter::SelectionState::SelFinish);
    doneCurrent();
}
void
XQGraph::mouseDoubleClickEvent ( QMouseEvent* e) {
	e->accept();
	if( !m_painter ) return;
    if(QApplication::queryKeyboardModifiers() & Qt::ShiftModifier)
        return;
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
		case Qt::MidButton:
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
		m_painter->wheel(e->pos().x(), e->pos().y(), (double)e->delta() / 8.0);
    doneCurrent();
}
void
XQGraph::showEvent ( QShowEvent *) {
}
void
XQGraph::hideEvent ( QHideEvent * ) {
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

