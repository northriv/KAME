/***************************************************************************
		Copyright (C) 2002-2015 Kentaro Kitagawa
		                   kitag@issp.u-tokyo.ac.jp
		
		This program is free software; you can redistribute it and/or
		modify it under the terms of the GNU Library General Public
		License as published by the Free Software Foundation; either
		version 2 of the License, or (at your option) any later version.
		
		You should have received a copy of the GNU Library General 
		Public License and a list of authors along with this program; 
		see the files COPYING and AUTHORS.
***************************************************************************/
#ifndef GRAPHWIDGET_H
#define GRAPHWIDGET_H

#include "graph.h"
class XQGraphPainter;
class OnScreenObject;

#include "support.h"
#include "xnodeconnector.h"

#ifdef USE_QGLWIDGET
    #include <QtOpenGL> //needs +opengl
#else
    #include <QOpenGLWidget>
#endif

class OnScreenObjectWithMarker;

//! Graph widget with a dialog which is initially hidden.
//! \sa XGraph, XQGraphPainter
class DECLSPEC_KAME XQGraph :
#ifdef USE_QGLWIDGET
        public QGLWidget {
#else
        public QOpenGLWidget {
#endif
    Q_OBJECT
public:
    XQGraph( QWidget* parent = 0, Qt::WindowFlags fl = {} );
	virtual ~XQGraph();
	//! register XGraph instance just after creating
	void setGraph(const shared_ptr<XGraph> &);

    void activateAxisSelectionTool(XAxis::AxisDirection dir, const XString &tool_desc);
    void activatePlaneSelectionTool(XAxis::AxisDirection dirx, XAxis::AxisDirection diry, const XString &tool_desc);

    Talker<std::tuple<XString, XGraph::VFloat, XGraph::VFloat, XQGraph*>> &onAxisSelectedByTool() {return m_onAxisSelectedByTool;}
    Talker<std::tuple<XString, XGraph::ValPoint,XGraph::ValPoint, XQGraph*>> &onPlaneSelectedByTool() {return m_onPlaneSelectedByTool;}

    weak_ptr<XQGraphPainter> painter() const {return m_painter;}
    const shared_ptr<XGraph> &graph() const {return m_graph;}
protected:
    virtual void mousePressEvent ( QMouseEvent*) override;
    virtual void mouseReleaseEvent ( QMouseEvent*) override;
    virtual void mouseDoubleClickEvent ( QMouseEvent*) override;
    virtual void mouseMoveEvent ( QMouseEvent*) override;
    virtual void wheelEvent ( QWheelEvent *) override;
    virtual void showEvent ( QShowEvent * ) override;
    virtual void hideEvent ( QHideEvent * ) override;
#ifndef QOPENGLWIDGET_QPAINTER_ATEND
    virtual void paintEvent(QPaintEvent *event) override;
#endif
    virtual void paintGL() override;
    //! openGL stuff
    virtual void initializeGL() override;
    virtual void resizeGL ( int width, int height ) override;
private:  
	friend class XQGraphPainter;
	shared_ptr<XGraph> m_graph;
	shared_ptr<XQGraphPainter> m_painter;


    Talker<std::tuple<XString, XGraph::VFloat, XGraph::VFloat, XQGraph*>> m_onAxisSelectedByTool;
    Talker<std::tuple<XString, XGraph::ValPoint,XGraph::ValPoint, XQGraph*>> m_onPlaneSelectedByTool;

	xqcon_ptr m_conDialog;

    bool m_isAxisSelectionByTool = false;
    bool m_isPlaneSelectionByTool = false;
    XAxis::AxisDirection m_toolDirX, m_toolDirY;
    XString m_toolDesc;
};

class Ui_FrmGraph;
typedef QForm<QWidget, Ui_FrmGraph> FrmGraph;

#endif // GRAPHWIDGET_H
