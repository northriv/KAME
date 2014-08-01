/***************************************************************************
		Copyright (C) 2002-2014 Kentaro Kitagawa
		                   kitag@kochi-u.ac.jp
		
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

//#define USE_OVERPAINT

class XGraph;
class XQGraphPainter;

#include "support.h"
#include "xnodeconnector.h"
#include <QGLWidget>

//! Graph widget with a dialog which is initially hidden.
//! \sa XGraph, XQGraphPainter
class XQGraph : public QGLWidget {
	Q_OBJECT
public:
    XQGraph( QWidget* parent = 0, Qt::WindowFlags fl = 0 );
	virtual ~XQGraph();
	//! register XGraph instance just after creating
	void setGraph(const shared_ptr<XGraph> &);

protected:
	virtual void mousePressEvent ( QMouseEvent*);
	virtual void mouseReleaseEvent ( QMouseEvent*);
	virtual void mouseDoubleClickEvent ( QMouseEvent*);
	virtual void mouseMoveEvent ( QMouseEvent*);
	virtual void wheelEvent ( QWheelEvent *);
	virtual void showEvent ( QShowEvent * );
	virtual void hideEvent ( QHideEvent * );  
#ifdef USE_OVERPAINT
    virtual void paintEvent(QPaintEvent *event);
#else
    virtual void paintGL ();
#endif
    //! openGL stuff
	virtual void initializeGL ();
	virtual void resizeGL ( int width, int height );
private:  
	friend class XQGraphPainter;
	shared_ptr<XGraph> m_graph;
	shared_ptr<XQGraphPainter> m_painter;
	xqcon_ptr m_conDialog;
};

class Ui_FrmGraph;
typedef QForm<QMainWindow, Ui_FrmGraph> FrmGraph;

#endif // GRAPHWIDGET_H
