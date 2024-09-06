/***************************************************************************
        Copyright (C) 2002-2023 Kentaro Kitagawa
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
#include <QTimer>
#ifdef __APPLE__
    #include <OpenGL/glu.h>
//    #include <GLUT/glut.h>
#else
    #include <GL/glu.h>
//    #include <GL/glut.h>
#endif
#include <QPainter>

#if QT_VERSION >= QT_VERSION_CHECK(5,0,0)
    #include <QWindow>
#endif

using std::min;
using std::max;

#include<stdio.h>
#include <QString>
#include <errno.h>

#define DEFAULT_FONT_SIZE 12

#define VIEW_NEAR -1.5
#define VIEW_FAR 0.5

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

XQGraphPainter::XQGraphPainter(const shared_ptr<XGraph> &graph, XQGraph* item) :
	m_graph(graph),
	m_pItem(item),
    m_selectionStateNow(SelectionState::Selecting),
    m_selectionModeNow(SelectionMode::SelNone),
	m_listpoints(0),
	m_listaxes(0),
	m_listgrids(0),
	m_listplanemarkers(0),
	m_listaxismarkers(0),
	m_bIsRedrawNeeded(true),
	m_bIsAxisRedrawNeeded(false),
	m_bTilted(false),
	m_bReqHelp(false) {
	item->m_painter.reset(this);
    graph->iterate_commit([=](Transaction &tr){
		m_lsnRedraw = tr[ *graph].onUpdate().connectWeakly(
            shared_from_this(), &XQGraphPainter::onRedraw);
    });
    m_lsnRepaint = m_tlkRepaint.connectWeakly(
        shared_from_this(), &XQGraphPainter::onRepaint,
        Listener::FLAG_MAIN_THREAD_CALL | Listener::FLAG_AVOID_DUP | Listener::FLAG_DELAY_ADAPTIVE);
    m_pixel_ratio = m_pItem->devicePixelRatio();
}
XQGraphPainter::~XQGraphPainter() {
    m_pItem->makeCurrent();

    if(m_listplanemarkers) glDeleteLists(m_listplanemarkers, 1);
    if(m_listaxismarkers) glDeleteLists(m_listaxismarkers, 1);
    if(m_listgrids) glDeleteLists(m_listgrids, 1);
    if(m_listaxes) glDeleteLists(m_listaxes, 1);
    if(m_listpoints) glDeleteLists(m_listpoints, 1);

    m_listedOSOs.clear();
    m_paintedOSOs.clear();
#ifdef USE_PBO
    glBindBuffer(GL_ARRAY_BUFFER, m_persistentPBO);
    glDeleteBuffers(1, &m_persistentPBO);
#endif

    m_pItem->doneCurrent();
}

int
XQGraphPainter::windowToScreen(int x, int y, double z, XGraph::ScrPoint *scr) {
	GLdouble nx, ny, nz;
    int ret = gluUnProject(x * m_pixel_ratio, (double)m_viewport[3] - y * m_pixel_ratio, z,
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
    *z = nz;
	return (ret != GL_TRUE);
}

void
XQGraphPainter::requestRepaint() {
    m_tlkRepaint.talk(Snapshot( *m_graph)); //defers update.
}
void
XQGraphPainter::onRepaint(const Snapshot &shot) {
    m_pItem->update();
}
void
XQGraphPainter::beginLine(double size, unsigned short stipple) {
    glLineWidth(size * m_pixel_ratio);
    checkGLError(); 
    if(stipple) {
        glLineStipple(1, stipple);
        glBegin(GL_LINE_STIPPLE);
    }
    else
        glBegin(GL_LINES);
}
void
XQGraphPainter::endLine() {
    glEnd();
    checkGLError(); 
}

void
XQGraphPainter::beginPoint(double size) {
    glPointSize(size * m_pixel_ratio);
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
}
void
XQGraphPainter::endQuad() {
	glEnd();
    checkGLError(); 
}

void
XQGraphPainter::secureWindow(const XGraph::ScrPoint &p) {
    double x,y,z;
    screenToWindow(p, &x, &y, &z);
    m_minX = std::min(m_minX, (int)lrint(x));
    m_maxX = std::max(m_maxX, (int)lrint(x));
    m_minY = std::min(m_minY, (int)lrint(y));
    m_maxY = std::max(m_maxY, (int)lrint(y));
}

void
XQGraphPainter::setInitView() {
	glLoadIdentity();
	glOrtho(0.0,1.0,0.0,1.0,VIEW_NEAR,VIEW_FAR);
}
void
XQGraphPainter::viewRotate(double angle, double x, double y, double z, bool init) {
	glMatrixMode(GL_PROJECTION);
	if(init) {
		glLoadIdentity();
		glGetDoublev(GL_PROJECTION_MATRIX, m_proj_rot);	
		setInitView();
	}
    glLoadIdentity();
    if(fabs(angle) > 0.0001) {
        glTranslated(0.5, 0.5, 0.5);
        glRotatef(angle, x, y, z);
        glTranslated(-0.5, -0.5, -0.5);
    }
    glMultMatrixd(m_proj_rot);
    glGetDoublev(GL_PROJECTION_MATRIX, m_proj_rot);
    setInitView();
    glMultMatrixd(m_proj_rot);
    glGetDoublev(GL_PROJECTION_MATRIX, m_proj);
    checkGLError();
	bool ov = m_bTilted;
	m_bTilted = !init;
	if(ov != m_bTilted) m_bIsRedrawNeeded = true;
	
	m_bIsAxisRedrawNeeded = true;
}


#define MAX_SELECTION 100

double
XQGraphPainter::selectGL(int x, int y, int dx, int dy, GLint list,
						 XGraph::ScrPoint *scr, XGraph::ScrPoint *dsdx, XGraph::ScrPoint *dsdy ) {
    glGetError(); //reset error

    GLuint selections[MAX_SELECTION];
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
        windowToScreen(x, y, zmax, scr);
        windowToScreen(x + 1, y, zmax, dsdx);
        windowToScreen(x, y + 1, zmax, dsdy);
    }
    checkGLError();

    return zmin;
}

double
XQGraphPainter::selectPlane(int x, int y, int dx, int dy,
							XGraph::ScrPoint *scr, XGraph::ScrPoint *dsdx, XGraph::ScrPoint *dsdy ) {
    return selectGL(x, y, dx, dy, m_listplanemarkers, scr, dsdx, dsdy);
}
double
XQGraphPainter::selectAxis(int x, int y, int dx, int dy,
                           XGraph::ScrPoint *scr, XGraph::ScrPoint *dsdx, XGraph::ScrPoint *dsdy ) {
    return selectGL(x, y, dx, dy, m_listaxismarkers, scr, dsdx, dsdy);
}
double
XQGraphPainter::selectPoint(int x, int y, int dx, int dy,
							XGraph::ScrPoint *scr, XGraph::ScrPoint *dsdx, XGraph::ScrPoint *dsdy ) {
	return selectGL(x, y, dx, dy, m_listpoints, scr, dsdx, dsdy);
}
void
XQGraphPainter::initializeGL () {
#ifndef USE_QGLWIDGET
    initializeOpenGLFunctions();
#endif
//    if(m_pixel_ratio < 2)
//        glEnable(GL_MULTISAMPLE);
//    else
//        glDisable(GL_MULTISAMPLE);

    //define display lists etc.:
    if(m_listplanemarkers) glDeleteLists(m_listplanemarkers, 1);
    if(m_listaxismarkers) glDeleteLists(m_listaxismarkers, 1);
    if(m_listgrids) glDeleteLists(m_listgrids, 1);
    if(m_listaxes) glDeleteLists(m_listaxes, 1);
    if(m_listpoints) glDeleteLists(m_listpoints, 1);
    m_listplanemarkers = glGenLists(1);
    m_listaxismarkers = glGenLists(1);
    m_listgrids = glGenLists(1);
    m_listaxes = glGenLists(1);
    m_listpoints = glGenLists(1);
    glLoadIdentity();
    //saves model view matrix
    viewRotate(0.0, 0.0, 0.0, 0.0, true);
}
void
XQGraphPainter::resizeGL ( int width  , int height ) {
    m_bIsRedrawNeeded = true;
    m_updatedTime = {};

    //readPixels may exceed the boundary.
    size_t bufsize = (m_pItem->width() + 1) * (m_pItem->height() + 1)
            * m_pixel_ratio * m_pixel_ratio * 3;
#ifdef USE_PBO
    if(m_persistentPBO) {
        glBindBuffer(GL_ARRAY_BUFFER, m_persistentPBO);
        glDeleteBuffers(1, &m_persistentPBO);
        m_persistentPBO = 0;
    }
    glGenBuffers(1, &m_persistentPBO);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER_ARB, m_persistentPBO);
    glBufferData(GL_PIXEL_UNPACK_BUFFER_ARB, bufsize, 0, GL_DYNAMIC_DRAW);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER_ARB, 0);
    if(glGetError() != GL_NO_ERROR) {
        //Sometimes binding to PBO fails.
        glBindBuffer(GL_ARRAY_BUFFER, m_persistentPBO);
        glDeleteBuffers(1, &m_persistentPBO);
        m_persistentPBO = 0;
    }
#endif
    if( !m_persistentPBO) {
        m_persistentFrame.clear();
        m_persistentFrame.resize(bufsize);
        m_persistentFrame.shrink_to_fit();
    }
}
void
XQGraphPainter::drawPersistentFrame(double persist_scale, const QColor &bgc) {
    glDepthMask(GL_FALSE);
    glPushMatrix();
    glLoadIdentity();
    glPixelZoom(1,1);
    glRasterPos2i(-1, -1);
    glBlendFunc(GL_SRC_ALPHA_SATURATE, GL_ONE);
    glClearColor(bgc.redF() * (1.0f - persist_scale), bgc.greenF() * (1.0f - persist_scale), bgc.blueF() * (1.0f - persist_scale),
                 1.0f - persist_scale);
    glClear(GL_COLOR_BUFFER_BIT);
    //Foolish Windows does not comply GL_CONSTANT_ALPHA
//            glBlendColor(0, 0, 0, persist_scale);
//            glBlendFunc(GL_CONSTANT_ALPHA, GL_ONE_MINUS_CONSTANT_ALPHA);
    if(m_persistentPBO) {
        glBindBuffer(GL_PIXEL_UNPACK_BUFFER_ARB, m_persistentPBO);
        checkGLError();
        glDrawPixels(m_pItem->width() * m_pixel_ratio, m_pItem->height() * m_pixel_ratio,
                     GL_BGR, GL_UNSIGNED_BYTE,
                    nullptr); //from PBO
        glBindBuffer(GL_PIXEL_UNPACK_BUFFER_ARB, 0);
    }
    else {
        glDrawPixels(m_pItem->width() * m_pixel_ratio, m_pItem->height() * m_pixel_ratio,
                     GL_BGR, GL_UNSIGNED_BYTE,
                     &m_persistentFrame[0]);
    }
    checkGLError();
    glPopMatrix();
    glDepthMask(GL_TRUE);
}

void
XQGraphPainter::storePersistentFrame() {
    if(m_persistentPBO) {
        glBindBuffer(GL_PIXEL_PACK_BUFFER_ARB, m_persistentPBO);
        checkGLError();
        glReadPixels(0, 0, m_pItem->width() * m_pixel_ratio, m_pItem->height() * m_pixel_ratio,
                     GL_BGR, GL_UNSIGNED_BYTE,
                     nullptr); //to PBO
        glBindBuffer(GL_PIXEL_PACK_BUFFER_ARB, 0);
    }
    else {
        GLint buf;
        glGetIntegerv(GL_DRAW_BUFFER, &buf);
        glReadBuffer(buf);
        checkGLError();
        glReadPixels(0, 0, m_pItem->width() * m_pixel_ratio, m_pItem->height() * m_pixel_ratio,
                     GL_BGR, GL_UNSIGNED_BYTE,
                     &m_persistentFrame[0]);
    }
    checkGLError();
}

void
XQGraphPainter::paintGL () {
#if !defined USE_QGLWIDGET && !defined QOPENGLWIDGET_QPAINTER_ATEND
//    QOpenGLPaintDevice fboPaintDev(width(), height());
    QPainter qpainter(m_pItem);
    qpainter.setRenderHints(QPainter::Antialiasing | QPainter::TextAntialiasing | QPainter::SmoothPixmapTransform);
    // | QPainter::LosslessImageRendering | QPainter::VerticalSubpixelPositioning | QPainter::NonCosmeticBrushPatterns
//    qpainter.setCompositionMode(QPainter::CompositionMode_SourceOver); //This might cause huge memory leak on intel's GPU in OSX.
    qpainter.beginNativePainting();
#endif
    Snapshot shot( *m_graph);

    QColor bgc = (QRgb)shot[ *m_graph->backGround()];
    glClearColor(bgc.redF(), bgc.greenF(), bgc.blueF(), bgc.alphaF());
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    //stores states
    glPushClientAttrib(GL_CLIENT_ALL_ATTRIB_BITS);
    glPushAttrib(GL_ALL_ATTRIB_BITS);
    glMatrixMode(GL_TEXTURE);
    glPushMatrix();
    glLoadIdentity();
    glMatrixMode(GL_PROJECTION);
    glPushMatrix();
    glMatrixMode(GL_MODELVIEW);
    glPushMatrix();
    glLoadIdentity(); //QOpenGLWidget may collapse modelview matrix?
    glGetDoublev(GL_MODELVIEW_MATRIX, m_model); //stores model-view matrix for gluUnproject().
    glMatrixMode(GL_PROJECTION);

    glGetError(); // flush error

    bool texen = glIsEnabled(GL_TEXTURE_2D);
    GLint depth_func_org, blend_func_org;
    glGetIntegerv(GL_DEPTH_FUNC, &depth_func_org);
    glGetIntegerv(GL_BLEND_SRC_ALPHA, &blend_func_org);
    GLint texwraps, texwrapt, texmagfil, texminfil;
    glGetTexParameteriv(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, &texwraps);
    glGetTexParameteriv(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, &texwrapt);
    glGetTexParameteriv(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, &texmagfil);
    glGetTexParameteriv(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, &texminfil);
    GLint boundTexture;
    glGetIntegerv(GL_TEXTURE_BINDING_2D, &boundTexture);

    glDepthFunc(GL_LEQUAL);


    // be aware of retina display.
    glViewport( 0, 0, (GLint)(m_pItem->width() * m_pixel_ratio),
                (GLint)(m_pItem->height() * m_pixel_ratio));
    glLoadMatrixd(m_proj); //restores our projection matrix.
    glGetIntegerv(GL_VIEWPORT, m_viewport);

    // Set up the rendering context,

//    glEnable(GL_MULTISAMPLE);
    glEnable(GL_BLEND);
    glEnable(GL_DEPTH_TEST);

    checkGLError();

	// Ghost stuff.
	XTime time_started = XTime::now();
    if(m_bIsRedrawNeeded || m_bIsAxisRedrawNeeded) {
		m_modifiedTime = time_started;    
    }

    double persist = shot[ *m_graph->persistence()]; //sec.
    if(persist > 0.0) {
        if(m_updatedTime.isSet()) {
            double tau = persist / (-log(0.1)) * 2.0;
            double persist_scale = exp(-(time_started - m_updatedTime)/tau);
            drawPersistentFrame(persist_scale, bgc);
        }
        m_updatedTime = time_started;
    }
    else {
        m_updatedTime = {};
    }

    glBlendFunc(GL_SRC_ALPHA,GL_ONE_MINUS_SRC_ALPHA);
    glMatrixMode(GL_MODELVIEW);

    if(m_bIsRedrawNeeded.compare_set_strong(true, false)) {// || m_bAvoidCallingLists
        shot = startDrawing();

        m_listedOSOs.clear();
//        //For stupid OpenGL implementations.
//        if(m_listplanemarkers) glDeleteLists(m_listplanemarkers, 1);
//        if(m_listaxismarkers) glDeleteLists(m_listaxismarkers, 1);
//        if(m_listgrids) glDeleteLists(m_listgrids, 1);
//        if(m_listaxes) glDeleteLists(m_listaxes, 1);
//        if(m_listpoints) glDeleteLists(m_listpoints, 1);
//        m_listplanemarkers = glGenLists(1);
//        m_listaxismarkers = glGenLists(1);
//        m_listgrids = glGenLists(1);
//        m_listaxes = glGenLists(1);
//        m_listpoints = glGenLists(1);

        checkGLError(); 

        glNewList(m_listgrids, GL_COMPILE_AND_EXECUTE);
        drawOffScreenGrids(shot);
        glEndList();

        checkGLError();

        glNewList(m_listpoints, GL_COMPILE_AND_EXECUTE);
        drawOffScreenPoints(shot);
        glEndList();
        
        checkGLError(); 

        if(persist > 0.0)
            storePersistentFrame();

//        glDisable(GL_DEPTH_TEST);
        glNewList(m_listaxes, GL_COMPILE_AND_EXECUTE);
        drawOffScreenAxes(shot);
        glEndList();

        checkGLError();

        glNewList(m_listaxismarkers, GL_COMPILE);
        drawOffScreenAxisMarkers(shot);
        glEndList();

        checkGLError();

        glNewList(m_listplanemarkers, GL_COMPILE);
        drawOffScreenPlaneMarkers(shot);
        glEndList();

        checkGLError();

//        for(auto &&osobj_w: m_persistentOSOs) {
//            if(auto osobj = osobj_w.lock()) {
//                osobj->drawOffScreenMarker();
//            }
//        }
//        checkGLError();

        m_bIsAxisRedrawNeeded = false;
    }
    else {        
        if(persist > 0.0)
            storePersistentFrame();
        glCallList(m_listgrids);
        glCallList(m_listpoints);
//        glDisable(GL_DEPTH_TEST);
        if(1) { //renderText() have to be called every time.
//        if(m_bIsAxisRedrawNeeded) {
            //For stupid OpenGL implementations.
//            if(m_listaxes) glDeleteLists(m_listaxes, 1);
//            m_listaxes = glGenLists(1);

            glNewList(m_listaxes, GL_COMPILE_AND_EXECUTE);
            drawOffScreenAxes(shot);
            glEndList();
            m_bIsAxisRedrawNeeded = false;
        }
        else {
            glCallList(m_listaxes);
        }
    }

    if(time_started - m_modifiedTime < persist) {
        QTimer::singleShot(50, m_pItem, SLOT(update()));
    }

    try {
        drawOnScreenObj(shot);
    }
    catch(std::domain_error &) {
        fprintf(stderr, "Screen Object not found\n");
    }

//    glMatrixMode(GL_PROJECTION);
//GLdouble proj_orig[16];
//    glGetDoublev(GL_PROJECTION_MATRIX, proj_orig);
//    setInitView();
//    glGetDoublev(GL_PROJECTION_MATRIX, m_proj);
//    glMatrixMode(GL_MODELVIEW);

    drawOnScreenViewObj(shot);

    {
        XScopedLock<XMutex> lock(m_mutexOSO);
        //texture objects
        for(auto &&oso: m_weakptrOSOs) {
            if(auto o = oso.lock()) {
                if(o->hasTexture())
                    o->drawNative();
            }
        }
        for(auto &&osobj: m_paintedOSOs) {
            if(osobj->hasTexture())
                osobj->drawNative();
        }
        for(auto &&osobj: m_listedOSOs) {
            if(osobj->hasTexture())
                osobj->drawNative();
        }
        //better to be drawn after textures
        for(auto it = m_weakptrOSOs.begin(); it != m_weakptrOSOs.end();) {
            if(auto o = it->lock()) {
                if( !o->hasTexture())
                    o->drawNative();
                it++;
            }
            else
                it = m_weakptrOSOs.erase(it);
        }
        for(auto &&osobj: m_paintedOSOs) {
            if( !osobj->hasTexture())
                osobj->drawNative();
        }
        for(auto &&osobj: m_listedOSOs) {
            if( !osobj->hasTexture())
                osobj->drawNative();
        }
    }
    checkGLError();

    glDisable(GL_DEPTH_TEST);

    //restores states
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, texwraps); //not necessary
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, texwrapt); //not necessary
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, texmagfil); //not necessary
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, texminfil); //not necessary
    glBindTexture(GL_TEXTURE_2D, boundTexture); //might be important
    glShadeModel(GL_FLAT);
    glDisable(GL_CULL_FACE);
    glDisable(GL_DEPTH_TEST);
    glDisable(GL_LIGHTING);
    glDepthFunc(depth_func_org);
    glDepthMask(false);//might be important
    glBlendFunc(GL_SRC_ALPHA,blend_func_org);
    glMatrixMode(GL_TEXTURE);
    glPopMatrix();
    glMatrixMode(GL_PROJECTION);
    glPopMatrix();
    glMatrixMode(GL_MODELVIEW);
    glPopMatrix();
    glPopAttrib();
    glPopClientAttrib();
    if(texen)
        glEnable(GL_TEXTURE_2D);

#if !defined USE_QGLWIDGET && !defined QOPENGLWIDGET_QPAINTER_ATEND
    qpainter.endNativePainting();
#else
    QPainter qpainter(m_pItem);
#endif
    {
        XScopedLock<XMutex> lock(m_mutexOSO);
        for(auto &&osobj: m_paintedOSOs) {
            osobj->drawByPainter( &qpainter);
        }
        m_paintedOSOs.clear();
        for(auto &&osobj: m_listedOSOs) {
            osobj->drawByPainter( &qpainter);
        }
        for(auto &&osobj: m_weakptrOSOs) {
            if(auto o = osobj.lock())
                o->drawByPainter( &qpainter);
        }
    }
    if(m_bReqHelp) {
        //native drawing is not supported here.
        drawOnScreenHelp(shot, &qpainter);
        XScopedLock<XMutex> lock(m_mutexOSO);
        for(auto &&osobj: m_paintedOSOs) {
            osobj->drawByPainter( &qpainter);
        }
        m_paintedOSOs.clear();
    }
//    qpainter.end();

//    memcpy(m_proj, proj_orig, sizeof(proj_orig));
}
