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
#include "graphwidget.h"
#include <QTimer>
#include <GL/glu.h>

using std::min;
using std::max;

#include<stdio.h>
#include <QString>
#include <errno.h>

#define checkGLError() \
{	 \
	GLenum err = glGetError(); \
	if(err != GL_NO_ERROR) { \
		switch(err) \
		{ \
		case GL_INVALID_ENUM: \
			dbgPrint("GL_INVALID_ENUM"); \
			break; \
		case GL_INVALID_VALUE: \
			dbgPrint("GL_INVALID_VALUE"); \
			break; \
		case GL_INVALID_OPERATION: \
			dbgPrint("GL_INVALID_OPERATION"); \
			break; \
		case GL_STACK_OVERFLOW: \
			dbgPrint("GL_STACK_OVERFLOW"); \
			break; \
		case GL_STACK_UNDERFLOW: \
			dbgPrint("GL_STACK_UNDERFLOW"); \
			break; \
		case GL_OUT_OF_MEMORY: \
			dbgPrint("GL_OUT_OF_MEMORY"); \
			break; \
		} \
	} \
} 

#define DEFAULT_FONT_SIZE 12

XQGraphPainter::~XQGraphPainter() {
    m_pItem->makeCurrent();
    
    if(m_listplanes) glDeleteLists(m_listplanes, 1);
    if(m_listgrids) glDeleteLists(m_listgrids, 1);
    if(m_listaxes) glDeleteLists(m_listaxes, 1);
    if(m_listpoints) glDeleteLists(m_listpoints, 1);
}

int
XQGraphPainter::windowToScreen(int x, int y, double z, XGraph::ScrPoint *scr) {
	GLdouble nx, ny, nz;
    int ret = gluUnProject(x * m_pixel_ratio, (double)m_viewport[3] - y * m_pixel_ratio, z * m_pixel_ratio,
            m_model, m_proj, m_viewport, &nx, &ny, &nz);
	scr->x = nx;
	scr->y = ny;
	scr->z = nz;
	return (ret != GL_TRUE);
}
int
XQGraphPainter::screenToWindow(const XGraph::ScrPoint &scr, double *x, double *y, double *z) {
	GLdouble nx, ny, nz;
	int ret = gluProject(scr.x, scr.y, scr.z, m_model, m_proj, m_viewport, &nx, &ny, &nz);
    *x = nx / m_pixel_ratio;
    *y = (m_viewport[3] - ny) / m_pixel_ratio;
    *z = nz / m_pixel_ratio;
	return (ret != GL_TRUE);
}

void
XQGraphPainter::repaintBuffer(int x1, int y1, int x2, int y2) {
	if((x1 != x2) || (y1 != y2)) {
		m_pItem->updateGL();
	}
}
void
XQGraphPainter::redrawOffScreen() {
	m_bIsRedrawNeeded = true;
}
  
void
XQGraphPainter::beginLine(double size) {
	glLineWidth(size);
    checkGLError(); 
	glBegin(GL_LINES);
}
void
XQGraphPainter::endLine() {
	glEnd();
    checkGLError(); 
}

void
XQGraphPainter::beginPoint(double size) {
	glPointSize(size);
    checkGLError(); 
	glBegin(GL_POINTS);
}
void
XQGraphPainter::endPoint() {
	glEnd();
    checkGLError(); 
}
void
XQGraphPainter::beginQuad(bool ) {
	glBegin(GL_QUADS);
    checkGLError(); 
}
void
XQGraphPainter::endQuad() {
	glEnd();
    checkGLError(); 
}

void
XQGraphPainter::defaultFont() {
	m_curAlign = 0;
	m_curFontSize = DEFAULT_FONT_SIZE;
}
int
XQGraphPainter::selectFont(const XString &str,
	const XGraph::ScrPoint &start, const XGraph::ScrPoint &dir, const XGraph::ScrPoint &swidth, int sizehint) {
	XGraph::ScrPoint d = dir;
	d.normalize();
	XGraph::ScrPoint s1 = start;
	double x, y, z;
    if(screenToWindow(s1, &x, &y, &z)) return -1;
	XGraph::ScrPoint s2 = s1;
	d *= 0.001;
	s2 += d;
	double x1, y1, z1;
	if(screenToWindow(s2, &x1, &y1, &z1)) return -1;
	XGraph::ScrPoint s3 = s1;
	XGraph::ScrPoint wo2 = swidth;
	wo2 *= 0.5;
	s3 += wo2;
	double x2, y2, z2;
	if(screenToWindow(s3, &x2, &y2, &z2)) return -1;	
	XGraph::ScrPoint s4 = s1;
	s4 -= wo2;
	double x3, y3, z3;
	if(screenToWindow(s4, &x3, &y3, &z3)) return -1;	
	int align = 0;
// width and height, restrict text
	double w = fabs(x3 - x2), h = fabs(y3 - y2);	
	if( fabs(x - x1) > fabs( y - y1) ) {
		//dir is horizontal
		align |= Qt::AlignVCenter;
		h = min(h, 2 * min(y, m_pItem->height() - y));
		if( x > x1 ) {
			align |= Qt::AlignRight;
			w = x;
		}
		else {
			align |= Qt::AlignLeft;
			w = m_pItem->width() - x;
		}
	}
	else {
		//dir is vertical
		align |= Qt::AlignHCenter;
		w = min(w, 2 * min(x, m_pItem->width() - x));
		if( y < y1 ) {
			align |= Qt::AlignTop;
			h = m_pItem->height() - y;
		}
		else {
			align |= Qt::AlignBottom;
			h = y;
		}
	}
	m_curFontSize = DEFAULT_FONT_SIZE + sizehint;
	m_curAlign = align;
    
    QFont font(m_pItem->font());
    for(;;) {
        font.setPointSize(m_curFontSize);
        QFontMetrics fm(font);
        QRect bb = fm.boundingRect(str);
		if(m_curFontSize < DEFAULT_FONT_SIZE + sizehint - 4) return -1;
        if((bb.width() < w ) && (bb.height() < h)) break;
		m_curFontSize--;
	}
    
	return 0;
}
void
XQGraphPainter::drawText(const XGraph::ScrPoint &p, const XString &str) {
    double x,y,z;
    screenToWindow(p, &x, &y, &z);
    QFont font(m_pItem->font());
    font.setPointSize(m_curFontSize);
    QFontMetrics fm(font);
    QRect bb = fm.boundingRect(str);
    if( (m_curAlign & Qt::AlignBottom) ) y -= bb.bottom();
    if( (m_curAlign & Qt::AlignVCenter) ) y += -bb.bottom() + bb.height() / 2;
    if( (m_curAlign & Qt::AlignTop) ) y -= bb.top();
    if( (m_curAlign & Qt::AlignHCenter) ) x -= bb.left() + bb.width() / 2;
    if( (m_curAlign & Qt::AlignRight) ) x -= bb.right();
    m_pItem->renderText(lrint(x), lrint(y), str, font); //window coord. from top-left end.
}

#define VIEW_NEAR -1.5
#define VIEW_FAR 0.5


void
XQGraphPainter::setInitView() {
	glLoadIdentity();
	glOrtho(0.0,1.0,0.0,1.0,VIEW_NEAR,VIEW_FAR);
}
void
XQGraphPainter::viewRotate(double angle, double x, double y, double z, bool init) {
	m_pItem->makeCurrent();
    glGetError(); //reset error
    
	glMatrixMode(GL_PROJECTION);
	if(init) {
		glLoadIdentity();
		glGetDoublev(GL_PROJECTION_MATRIX, m_proj_rot);	
		setInitView();
	}
	if(angle != 0.0) {
		glLoadIdentity();
		glTranslated(0.5, 0.5, 0.5);
		glRotatef(angle, x, y, z);
		glTranslated(-0.5, -0.5, -0.5);
		glMultMatrixd(m_proj_rot);
		glGetDoublev(GL_PROJECTION_MATRIX, m_proj_rot);
		setInitView();
		glMultMatrixd(m_proj_rot);
	}
	checkGLError();
	bool ov = m_bTilted;
	m_bTilted = !init;
	if(ov != m_bTilted) m_bIsRedrawNeeded = true;
	
	m_bIsAxisRedrawNeeded = true;
// 	if(m_viewport[3] != height() ) return; //firsttime
	//save projection matrix
}


#define MAX_SELECTION 100

double
XQGraphPainter::selectGL(int x, int y, int dx, int dy, GLint list,
						 XGraph::ScrPoint *scr, XGraph::ScrPoint *dsdx, XGraph::ScrPoint *dsdy ) {
	m_pItem->makeCurrent();
      
	glGetError(); //reset error
      
	GLuint selections[MAX_SELECTION];
	glGetDoublev(GL_PROJECTION_MATRIX, m_proj);
	glGetDoublev(GL_MODELVIEW_MATRIX, m_model);
	glGetIntegerv(GL_VIEWPORT, m_viewport);
	glSelectBuffer(MAX_SELECTION, selections);
	glRenderMode(GL_SELECT);
	glInitNames();
	glPushName((unsigned int)-1);
	glMatrixMode(GL_PROJECTION);
	glPushMatrix();
	//pick up small region
	glLoadIdentity();
    gluPickMatrix((double)(x - dx) * m_pixel_ratio, (double)m_viewport[3] - (y + dy) * m_pixel_ratio,
            2 * dx * m_pixel_ratio, 2 * dy * m_pixel_ratio, m_viewport);
	glMultMatrixd(m_proj);
      
	glEnable(GL_DEPTH_TEST);
	glLoadName(1);
	glCallList(list);
      
	glMatrixMode(GL_PROJECTION);
	glPopMatrix();
	int hits = glRenderMode(GL_RENDER);
	double zmin = 1.1;
	double zmax = -0.1;
	GLuint *ptr = selections;
	for (int i = 0; i < hits; i++) {
    	double zmin1 = (double)ptr[1] / (double)0xffffffffu;
    	double zmax1  = (double)ptr[2] / (double)0xffffffffu;
    	int n = ptr[0];
    	ptr += 3;
    	for (int j = 0; j < n; j++) {
			int k = *(ptr++);
    	  	if(k != -1) {
    			zmin = min(zmin1, zmin);
    			zmax = max(zmax1, zmax);
    		}
    	}
	}
	if((zmin < 1.0) && (zmax > 0.0) ) {
        windowToScreen(x, y, zmin, scr);
        windowToScreen(x + 1, y, zmin, dsdx);
        windowToScreen(x, y + 1, zmin, dsdy);
    }
    checkGLError();
    return zmin;
}

double
XQGraphPainter::selectPlane(int x, int y, int dx, int dy,
							XGraph::ScrPoint *scr, XGraph::ScrPoint *dsdx, XGraph::ScrPoint *dsdy ) {
	return selectGL(x, y, dx, dy, m_listplanes, scr, dsdx, dsdy);
}
double
XQGraphPainter::selectAxis(int x, int y, int dx, int dy,
                           XGraph::ScrPoint *scr, XGraph::ScrPoint *dsdx, XGraph::ScrPoint *dsdy ) {
    return selectGL(x, y, dx, dy, m_listaxes, scr, dsdx, dsdy);
}
double
XQGraphPainter::selectPoint(int x, int y, int dx, int dy,
							XGraph::ScrPoint *scr, XGraph::ScrPoint *dsdx, XGraph::ScrPoint *dsdy ) {
	return selectGL(x, y, dx, dy, m_listpoints, scr, dsdx, dsdy);
}
void
XQGraphPainter::initializeGL () {
//	m_pItem->makeCurrent();
	
//    glClearColor( 1.0, 1.0, 1.0, 1.0 );
//    glClearDepth( 1.0 );
    // Set up the rendering context, define display lists etc.:
    glHint(GL_POINT_SMOOTH_HINT,GL_FASTEST);
    glHint(GL_LINE_SMOOTH_HINT,GL_FASTEST);
    glDisable(GL_LINE_SMOOTH);
    glDisable(GL_POINT_SMOOTH);
    glEnable(GL_BLEND);
    glEnable(GL_DEPTH_TEST);
    glDepthFunc(GL_LEQUAL);
    glBlendFunc(GL_SRC_ALPHA,GL_ONE_MINUS_SRC_ALPHA);
    if(m_listplanes) glDeleteLists(m_listplanes, 1);
    if(m_listgrids) glDeleteLists(m_listgrids, 1);
    if(m_listaxes) glDeleteLists(m_listaxes, 1);
    if(m_listpoints) glDeleteLists(m_listpoints, 1);
    m_listplanes = glGenLists(1);
    m_listgrids = glGenLists(1);
    m_listaxes = glGenLists(1);
    m_listpoints = glGenLists(1);
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    //save model view matrix
    viewRotate(0.0, 0.0, 0.0, 0.0, true);
}
void
XQGraphPainter::resizeGL ( int width  , int height ) {
//	m_pItem->makeCurrent();

    // setup viewport, projection etc.:
    glMatrixMode(GL_PROJECTION);
    // be aware of retina display.
    glViewport( 0, 0, (GLint)(width * m_pixel_ratio),
                (GLint)(height * m_pixel_ratio));
    m_bIsRedrawNeeded = true;
//  drawLists();
}
void
XQGraphPainter::paintGL () {
//	m_pItem->makeCurrent();
    glGetError(); // flush error
    
    glMatrixMode(GL_PROJECTION);
    // be aware of retina display.
    glViewport( 0, 0, (GLint)(m_pItem->width() * m_pixel_ratio),
                (GLint)(m_pItem->height() * m_pixel_ratio));
    glGetDoublev(GL_PROJECTION_MATRIX, m_proj);
	glGetDoublev(GL_MODELVIEW_MATRIX, m_model);
    glGetIntegerv(GL_VIEWPORT, m_viewport);

    checkGLError(); 

	// Ghost stuff.
	XTime time_started = XTime::now();
    if(m_bIsRedrawNeeded || m_bIsAxisRedrawNeeded) {
		m_modifiedTime = time_started;    
		if(m_lastFrame.size())
			m_updatedTime = time_started;
		else
			m_updatedTime = XTime();
    }
        
    Snapshot shot( *m_graph);

    if(m_bIsRedrawNeeded) {
        shot = startDrawing();
        
        QColor bgc = (QRgb)shot[ *m_graph->backGround()];
        glClearColor( bgc.red() /255.0f, bgc.green() /255.0f, bgc.blue() /255.0f, 1.0 );
        
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        
        glMatrixMode(GL_MODELVIEW);
        glEnable(GL_DEPTH_TEST);
        
        checkGLError(); 

        glNewList(m_listplanes, GL_COMPILE);
        drawOffScreenPlanes(shot);
        glEndList();
        
        checkGLError(); 

        glNewList(m_listgrids, GL_COMPILE_AND_EXECUTE);
        drawOffScreenGrids(shot);
        glEndList();
        
        checkGLError(); 

        glNewList(m_listpoints, GL_COMPILE_AND_EXECUTE);
        drawOffScreenPoints(shot);
        glEndList();
        
        checkGLError(); 

//        glDisable(GL_DEPTH_TEST);
        glNewList(m_listaxes, GL_COMPILE_AND_EXECUTE);
        drawOffScreenAxes(shot);
        glEndList();
        
        checkGLError(); 

        m_bIsRedrawNeeded = false;
        m_bIsAxisRedrawNeeded = false;
    }
    else {
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        
        glMatrixMode(GL_MODELVIEW);
        glEnable(GL_DEPTH_TEST);
        
        glCallList(m_listgrids);
        glCallList(m_listpoints);
//        glDisable(GL_DEPTH_TEST);
#ifdef __APPLE__
        if(1) { //On mac, fonts have to be drawn every time.
#else
        if(m_bIsAxisRedrawNeeded) {
#endif // __APPLE__
            glNewList(m_listaxes, GL_COMPILE_AND_EXECUTE);
            drawOffScreenAxes(shot);
            glEndList();
            m_bIsAxisRedrawNeeded = false;
        }
        else {
            glCallList(m_listaxes);
        }
    }

    drawOnScreenObj(shot);
    
    glMatrixMode(GL_PROJECTION);
    glPushMatrix();
    setInitView();
    glGetDoublev(GL_PROJECTION_MATRIX, m_proj);
    glMatrixMode(GL_MODELVIEW);
    
    double persist = shot[ *m_graph->persistence()];
    if(persist > 0) {
	#define OFFSET 0.1
		double tau = persist / (-log(OFFSET)) * 0.4;
		double scale = exp(-(time_started - m_updatedTime)/tau);
		double offset = -OFFSET*(1.0-scale);
		bool update = (time_started - m_modifiedTime) < persist; 
		GLint accum;
		glGetIntegerv(GL_ACCUM_ALPHA_BITS, &accum);
		checkGLError();
		//! \todo QGLContext might clear accumration buffer.
		if(0) {
			if(update) {
				glAccum(GL_MULT, scale);
			    checkGLError(); 
				glAccum(GL_ACCUM, 1.0 - scale);
			    checkGLError(); 
				glAccum(GL_RETURN, 1.0);
			    checkGLError(); 
			}
			else {
				glAccum(GL_LOAD, 1.0);
			    checkGLError(); 
			}
		}
		else {
			m_lastFrame.resize(m_pItem->width() * m_pItem->height() * 4);
			if(update) {
				glPixelTransferf(GL_ALPHA_SCALE, scale);
			    checkGLError(); 
				glPixelTransferf(GL_ALPHA_BIAS, offset);
			    checkGLError(); 
				glRasterPos2i(0,0);
			    checkGLError(); 
				glDrawPixels((GLint)m_pItem->width(), (GLint)m_pItem->height(), 
							 GL_RGBA, GL_UNSIGNED_BYTE, &m_lastFrame[0]);
			    checkGLError(); 
				glPixelTransferf(GL_ALPHA_SCALE, 1.0);
			    checkGLError(); 
				glPixelTransferf(GL_ALPHA_BIAS, 0.0);
			    checkGLError(); 
			}
			glReadPixels(0, 0, (GLint)m_pItem->width(), (GLint)m_pItem->height(),
						 GL_RGBA, GL_UNSIGNED_BYTE, &m_lastFrame[0]);
		    checkGLError();     
		}
		m_updatedTime = time_started;
		if(update) {
			QTimer::singleShot(50, m_pItem, SLOT(update()));
		}
    }
    else {
    	m_lastFrame.clear();
		m_updatedTime = XTime();
    }

    drawOnScreenViewObj(shot);
    glMatrixMode(GL_PROJECTION);
    glPopMatrix();
//    glFlush();
}
