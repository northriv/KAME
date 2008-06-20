/***************************************************************************
		Copyright (C) 2002-2008 Kentaro Kitagawa
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

class XGraph;
class XQGraphPainter;

#include "support.h"
#include "xnodeconnector.h"
#include <qgl.h>

//! Graph widget with a dialog which is initially hidden.
//! \sa XGraph, XQGraphPainter
class XQGraph : public QGLWidget
{
	Q_OBJECT

public:
	XQGraph( QWidget* parent = 0, const char* name = 0, WFlags fl = 0 );
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
	//! openGL stuff
	virtual void initializeGL ();
	virtual void resizeGL ( int width, int height );
	virtual void paintGL ();
private:  
	friend class XQGraphPainter;
	shared_ptr<XGraph> m_graph;
	shared_ptr<XQGraphPainter> m_painter;
	xqcon_ptr m_conDialog;
};

#endif // GRAPHWIDGET_H
