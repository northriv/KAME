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
#ifndef GRAPHWIDGET_H
#define GRAPHWIDGET_H

class XGraph;
class XQGraphPainter;

#include "support.h"
#include "xnodeconnector.h"
#include <QOpenGLWidget>

//! Graph widget with a dialog which is initially hidden.
//! \sa XGraph, XQGraphPainter
class DECLSPEC_KAME XQGraph : public QOpenGLWidget {
	Q_OBJECT
public:
    XQGraph( QWidget* parent = 0, Qt::WindowFlags fl = 0 );
	virtual ~XQGraph();
	//! register XGraph instance just after creating
	void setGraph(const shared_ptr<XGraph> &);

protected:
    virtual void mousePressEvent ( QMouseEvent*) override;
    virtual void mouseReleaseEvent ( QMouseEvent*) override;
    virtual void mouseDoubleClickEvent ( QMouseEvent*) override;
    virtual void mouseMoveEvent ( QMouseEvent*) override;
    virtual void wheelEvent ( QWheelEvent *) override;
    virtual void showEvent ( QShowEvent * ) override;
    virtual void hideEvent ( QHideEvent * ) override;
    virtual void paintGL() override;
    //! openGL stuff
    virtual void initializeGL() override;
    virtual void resizeGL ( int width, int height ) override;
private:  
	friend class XQGraphPainter;
	shared_ptr<XGraph> m_graph;
	shared_ptr<XQGraphPainter> m_painter;
	xqcon_ptr m_conDialog;
};

class Ui_FrmGraph;
typedef QForm<QWidget, Ui_FrmGraph> FrmGraph;

#endif // GRAPHWIDGET_H
