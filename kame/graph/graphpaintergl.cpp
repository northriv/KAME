/***************************************************************************
        Copyright (C) 2002-2023 Kentaro Kitagawa
		                   kitagawa@phys.s.u-tokyo.ac.jp
		
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
XQGraphPainter::defaultFont() {
	m_curAlign = 0;
    m_curFontSize = std::min(14L, std::max(9L,
        lrint(DEFAULT_FONT_SIZE * m_pItem->height() / m_pItem->logicalDpiY() / 3.5)));
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
    defaultFont();

    m_curFontSize += sizehint;
    int fontsize_org = m_curFontSize;
	m_curAlign = align;
    
    {
        QFont font(m_pItem->font());
        for(;;) {
            font.setPointSize(m_curFontSize);
            QFontMetrics fm(font);
            QRect bb = fm.boundingRect(str);
            if(m_curFontSize < fontsize_org - 6) return -1;
            if((bb.width() < w ) && (bb.height() < h)) break;
            m_curFontSize -= 2;
        }
    }
	return 0;
}
void
XQGraphPainter::drawText(const XGraph::ScrPoint &p, QString &&str) {
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

    //draws texts later.
    Text txt;
    txt.text = std::move(str);
    txt.x = lrint(x);
    txt.y = lrint(y);
    txt.fontsize = m_curFontSize;
    txt.rgba = m_curTextColor;
    m_textOverpaint.push_back(std::move(txt));
}

OnScreenTexture::~OnScreenTexture() {
    painter()->glDeleteTextures(1, &id);
}

static const std::map<QImage::Format, GLenum> s_texture_aligns = {{QImage::Format_Grayscale8, 1}, {QImage::Format_Grayscale16, 2}, {QImage::Format_RGB888, 1},
                                           {QImage::Format_BGR888, 1}, {QImage::Format_RGBA8888, 4}, {QImage::Format_ARGB32, 4}};
static const std::map<QImage::Format, GLenum> s_texture_int_fmts = {{QImage::Format_Grayscale8, GL_LUMINANCE8}, {QImage::Format_Grayscale16, GL_LUMINANCE16}, {QImage::Format_RGB888, GL_RGB8},
                                             {QImage::Format_BGR888, GL_RGB8}, {QImage::Format_RGBA8888, GL_RGBA8}, {QImage::Format_ARGB32, GL_RGBA8}};
static const std::map<QImage::Format, GLenum> s_texture_fmts = {{QImage::Format_Grayscale8, GL_LUMINANCE}, {QImage::Format_Grayscale16, GL_LUMINANCE}, {QImage::Format_RGB888, GL_RGB},
                                         {QImage::Format_BGR888, GL_BGR}, {QImage::Format_RGBA8888, GL_RGBA}, {QImage::Format_ARGB32, GL_RGBA}};
static const std::map<QImage::Format, GLenum> s_texture_data_fmts = {{QImage::Format_Grayscale8, GL_UNSIGNED_BYTE}, {QImage::Format_Grayscale16, GL_UNSIGNED_SHORT}, {QImage::Format_RGB888, GL_UNSIGNED_BYTE},
                                              {QImage::Format_BGR888, GL_UNSIGNED_BYTE}, {QImage::Format_RGBA8888, GL_UNSIGNED_BYTE}, {QImage::Format_ARGB32, GL_UNSIGNED_INT_8_8_8_8_REV}};

void
OnScreenTexture::repaint(const shared_ptr<QImage> &image) {
//    auto image = std::make_shared<QImage>(256, 256, QImage::Format_RGB888);
//    QRgb value;
//    value = qRgb(0, 0x40, 0x40);
//    for(int x = 0; x < 3; ++x)
//        for(int y = 0; y < 30; ++y)
//            image->setPixel(x, y, value);
//    value = qRgb(0, 0, 0);
//    for(int x = 0; x < 2; ++x)
//        for(int y = 40; y < 100; ++y)
//            image->setPixel(x, y, value);
    glBindTexture(GL_TEXTURE_2D, 0);
    glActiveTexture(GL_TEXTURE1);
    glBindTexture(GL_TEXTURE_2D, id);
    glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, image->width(), image->height(),
        s_texture_fmts.at(image->format()), s_texture_data_fmts.at(image->format()), image->constBits());
    glBindTexture(GL_TEXTURE_2D, 0);
    glActiveTexture(GL_TEXTURE0);
    checkGLError();
    qimage = image;
}
shared_ptr<OnScreenTexture>
XQGraphPainter::createTextureWeakly(const shared_ptr<QImage> &image) {
//    m_bAvoidCallingLists = true; //bindTexture cannot be called inside list.
    glBindTexture(GL_TEXTURE_2D, 0);
    glActiveTexture(GL_TEXTURE1);
    GLuint id;
    glGenTextures(1, &id);
    glBindTexture(GL_TEXTURE_2D, id);
    glPixelStorei(GL_UNPACK_ALIGNMENT, s_texture_aligns.at(image->format()));
//    glTexStorage2D(GL_TEXTURE_2D, 1, GL_RGBA8, image->width(), image->height());
//    glTexSubImage2D(GL_TEXTURE_2D, 0​, 0, 0, image->width(), image->height(),
    glTexImage2D(GL_TEXTURE_2D, 0, s_texture_int_fmts.at(image->format()), image->width(), image->height(),
               0, s_texture_fmts.at(image->format()), s_texture_data_fmts.at(image->format()), image->constBits());
//    glGenerateMipmap(GL_TEXTURE_2D);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST); //GL_LINEAR
//    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_BASE_LEVEL, 0);
//    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAX_LEVEL, 0);
    glBindTexture(GL_TEXTURE_2D, 0);
    glActiveTexture(GL_TEXTURE0);
    checkGLError();
    return createOnScreenObjectWeakly<OnScreenTexture>(id, image);
}
void
OnScreenTexture::drawNative() {
//    static const GLfloat color[] = {1.0, 1.0, 1.0, 1.0};
//    glMaterialfv(GL_FRONT, GL_AMBIENT_AND_DIFFUSE, color);
    glBindTexture(GL_TEXTURE_2D, 0);
    glActiveTexture(GL_TEXTURE1);
    glEnable(GL_TEXTURE_2D);

    glBindTexture(GL_TEXTURE_2D, id);
    painter()->beginQuad(true);
//    glNormal3f(0, 0, 1);
//    glTexCoord2f(0.0, 0.0);
    glMultiTexCoord2f(GL_TEXTURE1, 0.0, 1.0);
    auto pt = leftTop();
    pt += XGraph::ScrPoint(0, 0, -0.01, 0); //drawing at the same Z-order is buggy.
    painter()->setVertex(pt);
//    glTexCoord2f(1, 0.0);
    glMultiTexCoord2f(GL_TEXTURE1, 1.0, 1.0);
    pt = rightTop();
    pt += XGraph::ScrPoint(0, 0, -0.01, 0);
    painter()->setVertex(pt);
//    glTexCoord2f(1, 1);
    glMultiTexCoord2f(GL_TEXTURE1, 1.0, 0.0);
    pt = rightBottom();
    pt += XGraph::ScrPoint(0, 0, -0.01, 0);
    painter()->setVertex(pt);
//    glTexCoord2f(0.0, 1);
    glMultiTexCoord2f(GL_TEXTURE1, 0.0, 0.0);
    pt = leftBottom();
    pt += XGraph::ScrPoint(0, 0, -0.01, 0);
    painter()->setVertex(pt);
    painter()->endQuad();
    glBindTexture(GL_TEXTURE_2D, 0);
    glDisable(GL_TEXTURE_2D);
    glActiveTexture(GL_TEXTURE0);
    checkGLError();
}

void
OnScreenObjectWithMarker::drawOffScreenMarker() {
    painter()->beginQuad(true);
    painter()->setVertex(leftTop());
    painter()->setVertex(rightTop());
    painter()->setVertex(rightBottom());
    painter()->setVertex(leftBottom());
    painter()->endQuad();
}

void
OnScreenRectObject::drawNative() {
    Snapshot shot_graph( *painter()->m_graph);
    switch(m_type) {
    case Type::Selection:
        for(auto c: {(unsigned int)shot_graph[ *painter()->m_graph->backGround()], baseColor()}) {
            painter()->beginQuad(true);
            painter()->setColor(c, 0.1);
            painter()->setVertex(leftTop());
            painter()->setVertex(rightTop());
            painter()->setVertex(rightBottom());
            painter()->setVertex(leftBottom());
            painter()->endQuad();
        }
        break;
    case Type::AreaTool:
//        glEnable(GL_LINE_STIPPLE);
//        unsigned short pat = 0x0f0fu;
        for(auto c: {(unsigned int)shot_graph[ *painter()->m_graph->backGround()], baseColor()}) {
//            painter()->beginLine(1.0, pat);
            painter()->beginLine(1.0);
            painter()->setColor(c, 0.3);
            painter()->setVertex(leftTop());
            painter()->setVertex(rightTop());
            painter()->setVertex(rightTop());
            painter()->setVertex(rightBottom());
            painter()->setVertex(rightBottom());
            painter()->setVertex(leftBottom());
            painter()->setVertex(leftBottom());
            painter()->setVertex(leftTop());
            painter()->endLine();
//            pat = ~pat;
        }
//        glDisable(GL_LINE_STIPPLE);
        break;
    }
}

void
OnScreenTextObject::drawOffScreenMarker() {
    for(auto &&txt: m_textOverpaint) {
        painter()->beginQuad(true);
        painter()->setVertex(txt.corners[0]);
        painter()->setVertex(txt.corners[1]);
        painter()->setVertex(txt.corners[2]);
        painter()->setVertex(txt.corners[3]);
        painter()->endQuad();
    }
}


void
OnScreenObjectWithMarker::placeObject(const XGraph::ScrPoint &init_lefttop, const XGraph::ScrPoint &init_righttop,
                                 const XGraph::ScrPoint &init_rightbottom, const XGraph::ScrPoint &init_leftbottom,
                                 HowToEvade direction, XGraph::SFloat space) {
    m_leftTop = init_lefttop;
    m_rightTop = init_righttop;
    m_rightBottom = init_rightbottom;
    m_leftBottom = init_leftbottom;
    m_direction = direction;
    m_space = space;
    switch (direction) {
    case HowToEvade::Never:
        break;
    case HowToEvade::ByAscent:
        break;
    case HowToEvade::ByDescent:
        break;
    case HowToEvade::ToLeft:
        break;
    case HowToEvade::ToRight:
        break;
//    case HowToEvade::ByCorner:
//        break;
    case HowToEvade::Hide:
        break;
    default:
        break;
    }
    painter()->secureWindow(m_leftTop);
    painter()->secureWindow(m_rightTop);
    painter()->secureWindow(m_leftBottom);
    painter()->secureWindow(m_rightBottom);
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

    glMatrixMode(GL_PROJECTION);

    glGetError(); // flush error
    //stores states
    GLint depth_func_org, blend_func_org;
//    glPushAttrib(GL_ALL_ATTRIB_BITS);
    glGetIntegerv(GL_DEPTH_FUNC, &depth_func_org);
    glGetIntegerv(GL_BLEND_SRC_ALPHA, &blend_func_org);

    glMatrixMode(GL_MODELVIEW);
//    glPushMatrix();
    glLoadIdentity(); //QOpenGLWidget may collapse modelview matrix?
    glGetDoublev(GL_MODELVIEW_MATRIX, m_model); //stores model-view matrix for gluUnproject().
    m_textOverpaint.clear();

    glDepthFunc(GL_LEQUAL);

    glMatrixMode(GL_PROJECTION);
    glPushMatrix();

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

    {
        XScopedLock<XMutex> lock(m_mutexOSO);
        for(auto &&osobj: m_paintedOSOs) {
            osobj->drawNative();
        }
        for(auto &&osobj: m_persistentOSOs) {
            osobj->drawNative();
        }
        for(auto &&osobj: m_weakptrOSOs) {
            if(auto o = osobj.lock())
                o->drawNative();
        }
    }
    checkGLError();

    drawOnScreenObj(shot);

//    glMatrixMode(GL_PROJECTION);
//GLdouble proj_orig[16];
//    glGetDoublev(GL_PROJECTION_MATRIX, proj_orig);
//    setInitView();
//    glGetDoublev(GL_PROJECTION_MATRIX, m_proj);
//    glMatrixMode(GL_MODELVIEW);
    
    drawOnScreenViewObj(shot);

    glDisable(GL_DEPTH_TEST);
    glMatrixMode(GL_PROJECTION);
    //    glFlush();
    glPopMatrix(); //original state for Qt.
    glMatrixMode(GL_MODELVIEW);
//    glPopMatrix(); //original state for Qt.

    //restores states
    glShadeModel(GL_FLAT);
    glDisable(GL_CULL_FACE);
    glDisable(GL_DEPTH_TEST);
    glDisable(GL_LIGHTING);
    glDepthFunc(depth_func_org);
    glDepthMask(false);
    glBlendFunc(GL_SRC_ALPHA,blend_func_org);
//    glPopAttrib();

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
        for(auto &&osobj: m_persistentOSOs) {
            osobj->drawByPainter( &qpainter);
        }
        for(auto &&osobj: m_weakptrOSOs) {
            if(auto o = osobj.lock())
                o->drawByPainter( &qpainter);
        }
    }
    drawTextOverpaint(qpainter);
    if(m_bReqHelp) {
        drawOnScreenHelp(shot, &qpainter);
        drawTextOverpaint(qpainter);
    }
    qpainter.end();

//    memcpy(m_proj, proj_orig, sizeof(proj_orig));
}

void
XQGraphPainter::drawTextOverpaint(QPainter &qpainter) {
    QFont font(qpainter.font());
    bool firsttime = true;
    for(auto &&text: m_textOverpaint) {
        if((QColor(text.rgba) != qpainter.pen().color()) || firsttime)
            qpainter.setPen(QColor(text.rgba));
        if((font.pointSize() != text.fontsize) || firsttime) {
            font.setPointSize(text.fontsize);
            qpainter.setFont(font);
        }
        firsttime = false;
        qpainter.drawText(text.x, text.y, text.text);
    }
    m_textOverpaint.clear();
}
