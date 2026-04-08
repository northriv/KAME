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
#include <QFont>
#include <QFontMetrics>
#include <QPainter>

XMutex OnScreenTexture::garbagemutex;
std::deque<GLuint> OnScreenTexture::unusedIDs;

#if defined(WIN32)
    #include <windows.h>
    PFNGLACTIVETEXTUREPROC glActiveTexture = (PFNGLACTIVETEXTUREPROC)wglGetProcAddress("glActiveTexture");
    PFNGLMULTITEXCOORD2FPROC glMultiTexCoord2f = (PFNGLMULTITEXCOORD2FPROC)wglGetProcAddress("glMultiTexCoord2f");
#endif

using std::min;
using std::max;

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

void
OnScreenRectObject::drawNative(bool colorpicking) {
    auto type = m_type;
    if(colorpicking)
        type = Type::Selection;
    Snapshot shot_graph( *painter()->graph());
    switch(type) {
    case Type::Selection: {
        double w = 0.1;
        for(auto c: {(unsigned int)shot_graph[ *painter()->graph()->backGround()], baseColor()}) {
            painter()->beginQuad(true);
            if(!colorpicking)
                painter()->setColor(c, w);
            painter()->setVertex(leftTop());
            painter()->setVertex(rightTop());
            painter()->setVertex(rightBottom());
            painter()->setVertex(leftBottom());
            painter()->endQuad();
            w = 0.3;
        }
    }
        break;
    case Type::Legends: {
        double w = 0.85;
        for(auto c: {(unsigned int)shot_graph[ *painter()->graph()->backGround()], baseColor()}) {
            painter()->beginQuad(true);
            if(!colorpicking)
                painter()->setColor(c, w);
            painter()->setVertex(leftTop());
            painter()->setVertex(rightTop());
            painter()->setVertex(rightBottom());
            painter()->setVertex(leftBottom());
            painter()->endQuad();
            w = 0.05;
        }
    }
        break;
    case Type::AreaTool:
//        glEnable(GL_LINE_STIPPLE);
//        unsigned short pat = 0x0f0fu;
        for(auto c: {(unsigned int)shot_graph[ *painter()->graph()->backGround()], baseColor()}) {
//            painter()->beginLine(1.0, pat);
            painter()->beginLine(1.0);
            if(!colorpicking)
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
    case Type::BorderLines:
        for(auto c: {(unsigned int)shot_graph[ *painter()->graph()->backGround()], baseColor()}) {
            painter()->beginLine(1.0);
            if(!colorpicking)
                painter()->setColor(c, 0.3);
            painter()->setVertex(leftTop());
            painter()->setVertex(leftBottom());
            painter()->setVertex(rightTop());
            painter()->setVertex(rightBottom());
            painter()->endLine();
        }
        break;
    // EllipseTool removed — Ellipse contour is now drawn by OnPlotMaskObject.
    }
}

void
OnScreenPickableObject::placeObject(const XGraph::ScrPoint &init_lefttop, const XGraph::ScrPoint &init_righttop,
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


static const std::map<QImage::Format, GLenum> s_texture_aligns = {{QImage::Format_Grayscale8, 1}, {QImage::Format_Grayscale16, 2}, {QImage::Format_RGB888, 1},
                                           {QImage::Format_BGR888, 1}, {QImage::Format_RGBA8888, 4}, {QImage::Format_ARGB32, 4},
                                           {QImage::Format_RGBA64, 8}};
static const std::map<QImage::Format, GLenum> s_texture_int_fmts = {{QImage::Format_Grayscale8, GL_LUMINANCE8}, {QImage::Format_Grayscale16, GL_LUMINANCE16}, {QImage::Format_RGB888, GL_RGB8},
                                             {QImage::Format_BGR888, GL_RGB8}, {QImage::Format_RGBA8888, GL_RGBA8}, {QImage::Format_ARGB32, GL_RGBA8},
                                             {QImage::Format_RGBA64, GL_RGB16}};
static const std::map<QImage::Format, GLenum> s_texture_fmts = {{QImage::Format_Grayscale8, GL_LUMINANCE}, {QImage::Format_Grayscale16, GL_LUMINANCE}, {QImage::Format_RGB888, GL_RGB},
                                         {QImage::Format_BGR888, GL_BGR}, {QImage::Format_RGBA8888, GL_RGBA}, {QImage::Format_ARGB32, GL_RGBA},
                                         {QImage::Format_RGBA64, GL_RGBA}};
static const std::map<QImage::Format, GLenum> s_texture_data_fmts = {{QImage::Format_Grayscale8, GL_UNSIGNED_BYTE}, {QImage::Format_Grayscale16, GL_UNSIGNED_SHORT}, {QImage::Format_RGB888, GL_UNSIGNED_BYTE},
                                              {QImage::Format_BGR888, GL_UNSIGNED_BYTE}, {QImage::Format_RGBA8888, GL_UNSIGNED_BYTE}, {QImage::Format_ARGB32, GL_UNSIGNED_INT_8_8_8_8_REV},
                                              {QImage::Format_RGBA64, GL_UNSIGNED_SHORT}};

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
    glActiveTexture(GL_TEXTURE1);

    glEnable(GL_TEXTURE_2D);

    glBindTexture(GL_TEXTURE_2D, id);
    glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, image->width(), image->height(),
        s_texture_fmts.at(image->format()), s_texture_data_fmts.at(image->format()), image->constBits());

    glDisable(GL_TEXTURE_2D);

    glBindTexture(GL_TEXTURE_2D, 0);
    glActiveTexture(GL_TEXTURE0);
    checkGLError();
    qimage = image;
}
weak_ptr<OnScreenTexture>
XQGraphPainter::createTextureDuringListing(const shared_ptr<QImage> &image) {
//    m_bAvoidCallingLists = true; //bindTexture cannot be called inside list.
    glActiveTexture(GL_TEXTURE1);
    glEnable(GL_TEXTURE_2D);
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
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
//    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
//    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    //    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_BASE_LEVEL, 0);
//    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAX_LEVEL, 0);

    glTexEnvf(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_REPLACE);

    glDisable(GL_TEXTURE_2D);
    glBindTexture(GL_TEXTURE_2D, 0);
    glActiveTexture(GL_TEXTURE0);
    checkGLError();
    return createListedOnScreenObject<OnScreenTexture>(id, image, graph());
}

OnScreenTexture::~OnScreenTexture() {
    XScopedLock<XMutex> lock(garbagemutex);
    if(id) {
        unusedIDs.push_back(id); //glDeleteTexture needs to be called during the context.
    }
}

void
OnScreenTexture::drawNative(bool colorpicking) {
    {
        XScopedLock<XMutex> lock(garbagemutex);
        while(unusedIDs.size()) {
            painter()->glDeleteTextures(1, &unusedIDs.front());
            unusedIDs.pop_front();
        }
    }
    if(colorpicking) {
        //just a rectangular.
        painter()->beginQuad(true);
        painter()->setVertex(leftTop());
        painter()->setVertex(rightTop());
        painter()->setVertex(rightBottom());
        painter()->setVertex(leftBottom());
        painter()->endQuad();
        return;
    }

//    static const GLfloat color[] = {1.0, 1.0, 1.0, 1.0};
//    glMaterialfv(GL_FRONT, GL_AMBIENT_AND_DIFFUSE, color);
    glActiveTexture(GL_TEXTURE1);
    glEnable(GL_TEXTURE_2D); //be called just before binding.
    glBindTexture(GL_TEXTURE_2D, id);
    painter()->beginQuad(true);
//    glNormal3f(0, 0, 1);
//    glTexCoord2f(0.0, 0.0);
    glMultiTexCoord2f(GL_TEXTURE1, 0.0, 1.0);
//    auto pt = leftTop();
    auto pt = leftBottom(); //inverted Y
    pt += XGraph::ScrPoint(0, 0, -0.01, 0); //drawing at the same Z-order is buggy.
    painter()->setVertex(pt);
//    glTexCoord2f(1, 0.0);
    glMultiTexCoord2f(GL_TEXTURE1, 1.0, 1.0);
//    pt = rightTop();
    pt = rightBottom(); //inverted Y
    pt += XGraph::ScrPoint(0, 0, -0.01, 0);
    painter()->setVertex(pt);
//    glTexCoord2f(1, 1);
    glMultiTexCoord2f(GL_TEXTURE1, 1.0, 0.0);
//    pt = rightBottom();
    pt = rightTop(); //inverted Y
    pt += XGraph::ScrPoint(0, 0, -0.01, 0);
    painter()->setVertex(pt);
//    glTexCoord2f(0.0, 1);
    glMultiTexCoord2f(GL_TEXTURE1, 0.0, 0.0);
//    pt = leftBottom();
    pt = leftTop(); //inveted Y
    pt += XGraph::ScrPoint(0, 0, -0.01, 0);
    painter()->setVertex(pt);
    painter()->endQuad();
    glDisable(GL_TEXTURE_2D);

    glBindTexture(GL_TEXTURE_2D, 0);
    glActiveTexture(GL_TEXTURE0);
    checkGLError();
}

template <class OSO>
void
OnPlotObject<OSO>::valToScreen() {
    XScopedLock<XMutex> lock( m_mutex);
    if(auto plot = m_plot.lock()) {
        XGraph::GPoint g[4];
        std::array<XGraph::ScrPoint, 4> s;
        for(unsigned int i = 0; i < 4; ++i) {
            plot->valToGraphFast(m_corners[i], &g[i]);
            plot->graphToScreenFast(g[i], &s[i]);
        }
        std::sort(s.begin(), s.end(), [](const XGraph::ScrPoint &a, const XGraph::ScrPoint &b) {
            return std::tie(a.x, a.y) < std::tie(b.x, b.y);
          });
        for(auto &p: s)
            p += m_offset;
        OSO::placeObject(s[1], s[3], s[2], s[0], this->m_direction, this->m_space);
    }
}

template <class OSO>
void
OnPlotObject<OSO>::drawNative(bool colorpicking) {
    valToScreen();
    this->OSO::drawNative(colorpicking);
}
template <class OSO>
void
OnPlotObject<OSO>::drawByPainter(QPainter *p) {
    valToScreen();
    this->OSO::drawByPainter(p);
}

template class OnPlotObject<OnScreenRectObject>;
template class OnPlotObject<OnScreenTextObject>;

template <class OSO, bool IsXAxis>
void
OnAxisObject<OSO, IsXAxis>::toScreen() {
    XScopedLock<XMutex> lock( m_mutex);
    if(auto plot = m_plot.lock()) {
        XGraph::GFloat bgx, bgy, edx, edy;
        Snapshot shot( *plot);
        shared_ptr<XAxis> axisx = shot[ *plot->axisX()];
        shared_ptr<XAxis> axisy = shot[ *plot->axisY()];
        if(IsXAxis) {
            bgx = axisx->valToAxis(m_bg1);
            edx = axisx->valToAxis(m_ed1);
            bgy = m_bg2;
            edy = m_ed2;
        }
        else {
            bgy = axisy->valToAxis(m_bg1);
            edy = axisy->valToAxis(m_ed1);
            bgx = m_bg2;
            edx = m_ed2;
        }
        XGraph::GPoint g[4] = {{bgx, edy}, {edx, edy}, {edx, bgy}, {bgx, bgy}};
        std::array<XGraph::ScrPoint, 4> s;
        for(unsigned int i = 0; i < 4; ++i) {
            plot->graphToScreenFast(g[i], &s[i]);
        }
//        std::sort(s.begin(), s.end(), [](const XGraph::ScrPoint &a, const XGraph::ScrPoint &b) {
//            return std::tie(a.x, a.y) < std::tie(b.x, b.y);
//          });
        for(auto &p: s)
            p += m_offset;
//        OSO::placeObject(s[1], s[3], s[2], s[0], OSO::HowToEvade::Never, 0);
        OSO::placeObject(s[0], s[1], s[2], s[3], this->m_direction, this->m_space);
    }
}

template <class OSO, bool IsXAxis>
void
OnAxisObject<OSO, IsXAxis>::drawNative(bool colorpicking) {
    toScreen();
    this->OSO::drawNative(colorpicking);
}
template <class OSO, bool IsXAxis>
void
OnAxisObject<OSO, IsXAxis>::drawByPainter(QPainter *p) {
    toScreen();
    this->OSO::drawByPainter(p);
}

template class OnAxisObject<OnScreenRectObject, true>;
template class OnAxisObject<OnScreenRectObject, false>;
template class OnAxisObject<OnScreenTextObject, true>;
template class OnAxisObject<OnScreenTextObject, false>;

template <bool IsXAxis>
void OnAxisFuncObject<IsXAxis>::drawNative(bool colorpicking) {
    this->toScreen();
    Snapshot shot_graph( *this->painter()->graph());
    double x1,y1,z1,x2,y2,z2;
    this->painter()->screenToWindow(this->leftBottom(), &x1, &y1, &z1);
    this->painter()->screenToWindow(this->rightBottom(), &x2, &y2, &z2);
    double window_len = std::max(x1, x2) - std::min(x1, x2);
    unsigned int len = std::max(50u, (unsigned int)window_len * 2);
    m_xvec.resize(len);
    m_yvec.resize(len);
    XGraph::VFloat xfront, xback;
    std::tie(xfront, xback) = this->axis1ValueRange();
    XGraph::VFloat dx = (xback - xfront) / (len - 1);
    XGraph::VFloat x = xfront;
    for(auto &&v: m_xvec) {
        v = x;
        x += dx;
    }
    m_yvec = this->func(m_xvec, std::move(m_yvec));
    if(m_yvec.size() != m_xvec.size())
        return;
    XScopedLock<XMutex> lock( this->m_mutex);
    if(auto plot = this->m_plot.lock()) {
        Snapshot shot( *plot);
        XGraph::ScrPoint s;
        double w = 0.85;
        for(auto c: {this->baseColor()}) {
            this->painter()->beginLine(1.0);
            if( !colorpicking)
                this->painter()->setColor(c, w);

            for(unsigned int i = 0; i < len; ++i) {
                if(i >= 2)
                    this->painter()->setVertex(s);
                XGraph::ValPoint v(m_xvec[i], m_yvec[i]);
                if( !IsXAxis)
                    v = {m_yvec[i], m_xvec[i]};
                XGraph::GPoint g;
                plot->valToGraphFast(v, &g);
                plot->graphToScreenFast(g, &s);
                this->painter()->setVertex(s);
            }
            this->painter()->endLine();
            w = 0.9;
        }
    }
}
template class OnAxisFuncObject<true>;
template class OnAxisFuncObject<false>;


OnScreenTextObject::OnScreenTextObject(XQGraphPainter* p, const std::__1::shared_ptr<XNode> &pickable_node) :
    OnScreenPickableObject(p, pickable_node),
    m_minX(0xffff), m_minY(0xffff), m_maxX(-0xffff), m_maxY(-0xffff) {
    defaultFont();
}

void
OnScreenTextObject::drawNative(bool colorpicking) {
    if(m_hidden) return;
    // For the drawTextAtPlacedPosition path, drawText() is never called, so
    // m_minX/Y/m_maxX/Y are never set.  Compute AABB here from the placed
    // position and font metrics so that resolveOverlaps() can move this OSO.
    if( !colorpicking) {
        if(local_shared_ptr<std::tuple<int, int, XString>> txt_ptr = m_textThreadSafe) {
            // Reset shift from previous frame so resolveOverlaps() starts fresh.
            m_shiftX = m_shiftY = 0;
            auto &[alignment_unused, sizehint, str] = *txt_ptr;
            QFont font(painter()->widget()->font());
            int fsz = std::min(16L, std::max(8L,
                lrint(font.pointSize() * painter()->widget()->height() / painter()->widget()->logicalDpiY() / 3.5)));
            font.setPointSize(fsz + sizehint);
            QFontMetrics fm(font);
            QRect bb = fm.boundingRect(str);
            double x, y, z;
            if( !painter()->screenToWindow(leftTop(), &x, &y, &z)) {
                m_minX = x + bb.left();
                m_minY = y + bb.top();
                m_maxX = x + bb.right();
                m_maxY = y + bb.bottom();
            }
        }
    }
    if(colorpicking) {
        for(auto &&txt: m_textOverpaint) {
            if(txt.item_hidden) continue;
            //just a rectangular.
            painter()->beginQuad(true);
            painter()->setVertex(txt.corners[0]);
            painter()->setVertex(txt.corners[1]);
            painter()->setVertex(txt.corners[2]);
            painter()->setVertex(txt.corners[3]);
            painter()->endQuad();
        }
        return;
    }
}
void
OnScreenTextObject::drawByPainter(QPainter *qpainter) {
    if(m_hidden) return;
    QFont font(qpainter->font());
    if(local_shared_ptr<std::tuple<int, int, XString>> txt = m_textThreadSafe) {
        //OSO treated as marker, text stored by drawTextAtPlacedPosition().
        auto [alignment, sizehint, str] = *txt;
        defaultFont();
        font.setPointSize(m_curFontSize + sizehint);
        //todo alignment support
        qpainter->setFont(font);
        qpainter->setPen(QColor(baseColor()));
        double x,y,z;
        if( !painter()->screenToWindow(leftTop(), &x, &y, &z))
            qpainter->drawText(x + m_shiftX, y + m_shiftY, str);
    }
    if(m_textOverpaint.size()) {
        //text stored by drawText().
        font.setPointSize(m_curFontSize);
        qpainter->setFont(font);
        bool firsttime = true;
        for(auto &&text: m_textOverpaint) {
            if(text.item_hidden) continue;
            auto str = m_text.mid(text.strpos, text.length);

            if((QColor(text.rgba) != qpainter->pen().color()) || firsttime)
                qpainter->setPen(QColor(text.rgba));
            firsttime = false;
            qpainter->drawText(text.x + m_shiftX, text.y + m_shiftY, str);
        }
    }
}

void
OnScreenTextObject::clear() {
    m_text.clear();
    m_textOverpaint.clear();
    m_shiftX = m_shiftY = 0;
    m_hidden = false;
    m_minX = 0xffff; m_minY = 0xffff; m_maxX = -0xffff; m_maxY = -0xffff;
}
void
OnScreenTextObject::drawTextAtPlacedPosition(const XString &str, int alignment, int sizehint) {
    m_textThreadSafe = make_local_shared<std::tuple<int, int, XString>>(alignment, sizehint, str);
}
void
OnScreenTextObject::drawText(const XGraph::ScrPoint &p, const XString &str) {
    Text txt;
    txt.strpos = m_text.length();
    m_text.append((QString)str);
    txt.length = str.length();
    txt.pos = p;
    txt.rgba = baseColor();

    QFont font(painter()->widget()->font());
    font.setPointSize(m_curFontSize);
    QFontMetrics fm(font);
    QRect bb = fm.boundingRect(str);
    //draws texts later.
    double x,y,z;
    if(painter()->screenToWindow(p, &x, &y, &z))
        return;
    if( (m_curAlign & Qt::AlignBottom) ) y -= bb.bottom();
    if( (m_curAlign & Qt::AlignVCenter) ) y += -bb.bottom() + bb.height() / 2;
    if( (m_curAlign & Qt::AlignTop) ) y -= bb.top();
    if( (m_curAlign & Qt::AlignHCenter) ) x -= bb.left() + bb.width() / 2;
    if( (m_curAlign & Qt::AlignRight) ) x -= bb.right();

    txt.x = lrint(x);
    txt.y = lrint(y);
    // Track AABB using actual glyph bounds relative to the baseline (bb.top() is
    // negative = above baseline, bb.bottom() is positive = below baseline).
    // Using plain y / y+bb.height() would be wrong because y is the baseline
    // position after alignment, not the visual top edge.
    txt.bbx1 = lrint(x + bb.left());
    txt.bby1 = lrint(y + bb.top());
    txt.bbx2 = lrint(x + bb.right());
    txt.bby2 = lrint(y + bb.bottom());
    m_minX = std::min(m_minX, (double)txt.bbx1);
    m_maxX = std::max(m_maxX, (double)txt.bbx2);
    m_minY = std::min(m_minY, (double)txt.bby1);
    m_maxY = std::max(m_maxY, (double)txt.bby2);
    painter()->windowToScreen(x + bb.left(),  y + bb.top(),    z, &txt.corners[0]);
    painter()->windowToScreen(x + bb.right(), y + bb.top(),    z, &txt.corners[1]);
    painter()->windowToScreen(x + bb.right(), y + bb.bottom(), z, &txt.corners[2]);
    painter()->windowToScreen(x + bb.left(),  y + bb.bottom(), z, &txt.corners[3]);

    m_textOverpaint.push_back(std::move(txt));
}

void
OnScreenTextObject::defaultFont() {
    m_curAlign = 0;
    QFont font(painter()->widget()->font());
    m_curFontSize = std::min(14L, std::max(9L,
        lrint(font.pointSize() * painter()->widget()->height() / painter()->widget()->logicalDpiY() / 3.5)));
}

int
OnScreenTextObject::selectFont(const XString &str, const XGraph::ScrPoint &start,
                          const XGraph::ScrPoint &dir, const XGraph::ScrPoint &width, int sizehint) {
    XGraph::ScrPoint d = dir;
    d.normalize();
    XGraph::ScrPoint s1 = start;
    double x, y, z;
    if(painter()->screenToWindow(s1, &x, &y, &z)) return -1;
    XGraph::ScrPoint s2 = s1;
    d *= 0.001;
    s2 += d;
    double x1, y1, z1;
    if(painter()->screenToWindow(s2, &x1, &y1, &z1)) return -1;
    XGraph::ScrPoint s3 = s1;
    XGraph::ScrPoint wo2 = width;
    wo2 *= 0.5;
    s3 += wo2;
    double x2, y2, z2;
    if(painter()->screenToWindow(s3, &x2, &y2, &z2)) return -1;
    XGraph::ScrPoint s4 = s1;
    s4 -= wo2;
    double x3, y3, z3;
    if(painter()->screenToWindow(s4, &x3, &y3, &z3)) return -1;
    int align = 0;
// width and height, restrict text
    double w = fabs(x3 - x2), h = fabs(y3 - y2);
    if( fabs(x - x1) > fabs( y - y1) ) {
        //dir is horizontal
        align |= Qt::AlignVCenter;
        h = min(h, 2 * min(y, painter()->widget()->height() - y));
        if( x > x1 ) {
            align |= Qt::AlignRight;
            w = x;
        }
        else {
            align |= Qt::AlignLeft;
            w = painter()->widget()->width() - x;
        }
    }
    else {
        //dir is vertical
        align |= Qt::AlignHCenter;
        w = min(w, 2 * min(x, painter()->widget()->width() - x));
        if( y < y1 ) {
            align |= Qt::AlignTop;
            h = painter()->widget()->height() - y;
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
        QFont font(painter()->widget()->font());
        for(;;) {
            font.setPointSize(m_curFontSize);
            QFontMetrics fm(font);
            QRect bb = fm.boundingRect(str);
            if(m_curFontSize < fontsize_org - 6) return -1;
            if((bb.width() < w ) && (bb.height() < h)) {
                break;
            }
            m_curFontSize -= 2;
        }
    }

    return 0;
}

void
OnPlotMaskObject::setMask(const shared_ptr<XPlot> &plot,
                           const XGraph::ValPoint corners[4],
                           const shared_ptr<std::vector<uint8_t>> &mask,
                           unsigned int width, unsigned int height,
                           XGraph::ScrPoint offset) {
    XScopedLock<XMutex> lock(m_mutex);
    m_plot = plot;
    for(unsigned int i = 0; i < 4; ++i)
        m_corners[i] = corners[i];
    m_mask = mask;
    m_width = width;
    m_height = height;
    m_offset = offset;
}
void
OnPlotMaskObject::drawNative(bool colorpicking) {
    XScopedLock<XMutex> lock(m_mutex);
    auto plot = m_plot.lock();
    if( !plot) return;
    // corners: {(bgx,bgy),(edx,bgy),(edx,edy),(bgx,edy)}
    XGraph::ScrPoint s[4];
    for(unsigned int i = 0; i < 4; ++i) {
        XGraph::GPoint g;
        plot->valToGraphFast(m_corners[i], &g);
        plot->graphToScreenFast(g, &s[i]);
        s[i] += m_offset;
    }
    bool hasMask = m_mask && m_width && m_height && (m_mask->size() == (size_t)m_width * m_height);

    glDisable(GL_DEPTH_TEST);
    if(colorpicking) {
        // For picking: draw the mask contour area as quads (no texture).
        if(hasMask) {
            auto lerp = [](const XGraph::ScrPoint &a, const XGraph::ScrPoint &b, float t) -> XGraph::ScrPoint {
                return {a.x + (b.x - a.x) * t, a.y + (b.y - a.y) * t, a.z + (b.z - a.z) * t};
            };
            unsigned int pick_step = std::max(1u, m_height / 64);
            painter()->beginQuad(true);
            for(unsigned int y = 0; y < m_height; y += pick_step) {
                unsigned int y1 = std::min(y + pick_step, m_height);
                // Find xmin/xmax for this band (median of per-row values).
                std::vector<int> rmins, rmaxs;
                for(unsigned int yy = y; yy < y1; ++yy) {
                    int rmin = -1, rmax = -1;
                    for(unsigned int x = 0; x < m_width; ++x) {
                        if((*m_mask)[yy * m_width + x]) {
                            if(rmin < 0) rmin = x;
                            rmax = x;
                        }
                    }
                    if(rmin >= 0) { rmins.push_back(rmin); rmaxs.push_back(rmax); }
                }
                if(rmins.empty()) continue;
                std::sort(rmins.begin(), rmins.end());
                std::sort(rmaxs.begin(), rmaxs.end());
                int xmin = rmins[rmins.size() / 2];
                int xmax = rmaxs[rmaxs.size() / 2];
                float yt = (float)y / m_height;
                float yb = (float)y1 / m_height;
                float xl = (float)xmin / m_width;
                float xr = (float)(xmax + 1) / m_width;
                auto lt = lerp(s[0], s[3], yb);
                auto rt = lerp(s[1], s[2], yb);
                auto lb = lerp(s[0], s[3], yt);
                auto rb = lerp(s[1], s[2], yt);
                painter()->setVertex(lerp(lb, rb, xl));
                painter()->setVertex(lerp(lb, rb, xr));
                painter()->setVertex(lerp(lt, rt, xr));
                painter()->setVertex(lerp(lt, rt, xl));
            }
            painter()->endQuad();
        }
        else {
            painter()->beginQuad(true);
            painter()->setVertex(s[0]);
            painter()->setVertex(s[1]);
            painter()->setVertex(s[2]);
            painter()->setVertex(s[3]);
            painter()->endQuad();
        }
    }
    else if( !hasMask) {
        // No mask — filled rectangle highlight (two-pass bg+fg).
        unsigned int fgc = baseColor();
        Snapshot shot_graph( *painter()->graph());
        unsigned int bgc = shot_graph[ *painter()->graph()->backGround()];
        double w = 0.1;
        for(auto c: {bgc, fgc}) {
            painter()->beginQuad(true);
            painter()->setColor(c, w);
            painter()->setVertex(s[0]);
            painter()->setVertex(s[1]);
            painter()->setVertex(s[2]);
            painter()->setVertex(s[3]);
            painter()->endQuad();
            w = 0.3;
        }
    }
    else if(m_highlighted) {
        drawFilledTexture(s, colorpicking);
    }
    else {
        drawContourLines(s, colorpicking);
    }
    glEnable(GL_DEPTH_TEST);
}

void
OnPlotMaskObject::drawContourLines(const XGraph::ScrPoint s[4], bool colorpicking) {
    // Scan each row for [xmin, xmax] of mask, draw left and right contour lines.
    auto lerp = [](const XGraph::ScrPoint &a, const XGraph::ScrPoint &b, float t) -> XGraph::ScrPoint {
        return {a.x + (b.x - a.x) * t, a.y + (b.y - a.y) * t, a.z + (b.z - a.z) * t};
    };
    Snapshot shot_graph( *painter()->graph());
    unsigned int bgc = shot_graph[ *painter()->graph()->backGround()];
    unsigned int fgc = baseColor();

    // Build left and right contour points from per-row-band boundaries.
    // Downsample to ~128 bands. Use median of per-row xmin/xmax within each
    // band to reject sparse outlier pixels.
    unsigned int step = std::max(1u, m_height / 128);
    struct BandPt {
        XGraph::ScrPoint left, right;
    };
    // Use negative index to mark empty bands (no mask pixels).
    std::vector<int> band_valid; // 1 = has points, 0 = empty
    std::vector<BandPt> band_pts;
    for(unsigned int y = 0; y < m_height; y += step) {
        unsigned int y1 = std::min(y + step, m_height);
        std::vector<int> row_xmins, row_xmaxs;
        for(unsigned int yy = y; yy < y1; ++yy) {
            const uint8_t *row = m_mask->data() + yy * m_width;
            int rmin = -1, rmax = -1;
            for(unsigned int x = 0; x < m_width; ++x) {
                if(row[x]) {
                    if(rmin < 0) rmin = x;
                    rmax = x;
                }
            }
            if(rmin >= 0) {
                row_xmins.push_back(rmin);
                row_xmaxs.push_back(rmax);
            }
        }
        if(row_xmins.empty()) {
            band_valid.push_back(0);
            band_pts.push_back({});
            continue;
        }
        // Use median to reject outliers.
        std::sort(row_xmins.begin(), row_xmins.end());
        std::sort(row_xmaxs.begin(), row_xmaxs.end());
        int xmin = row_xmins[row_xmins.size() / 2];
        int xmax = row_xmaxs[row_xmaxs.size() / 2];
        float yt = ((float)(y + y1) * 0.5f) / m_height;
        auto left_edge = lerp(s[0], s[3], yt);
        auto right_edge = lerp(s[1], s[2], yt);
        band_valid.push_back(1);
        band_pts.push_back({
            lerp(left_edge, right_edge, (float)xmin / m_width),
            lerp(left_edge, right_edge, (float)(xmax + 1) / m_width)
        });
    }

    // Draw two-pass (bg + fg) contour lines. Break lines at empty bands.
    auto drawContour = [&](auto getPt) {
        for(auto c: {bgc, fgc}) {
            bool in_segment = false;
            for(unsigned int i = 0; i < band_pts.size(); ++i) {
                if( !band_valid[i]) {
                    if(in_segment) {
                        painter()->endLine();
                        in_segment = false;
                    }
                    continue;
                }
                if( !in_segment) {
                    painter()->beginLine(1.0);
                    if( !colorpicking)
                        painter()->setColor(c, 0.3);
                    in_segment = true;
                }
                else {
                    // Duplicate vertex for line segment pair.
                    painter()->setVertex(getPt(band_pts[i]));
                }
                painter()->setVertex(getPt(band_pts[i]));
            }
            if(in_segment)
                painter()->endLine();
        }
    };
    // Left contour
    drawContour([](const BandPt &bp) -> const XGraph::ScrPoint& { return bp.left; });
    // Right contour
    drawContour([](const BandPt &bp) -> const XGraph::ScrPoint& { return bp.right; });
    // Connect top and bottom — close the contour where left and right meet.
    int first = -1, last = -1;
    for(unsigned int i = 0; i < band_pts.size(); ++i) {
        if(band_valid[i]) {
            if(first < 0) first = i;
            last = i;
        }
    }
    if(first >= 0) {
        for(auto c: {bgc, fgc}) {
            painter()->beginLine(1.0);
            if( !colorpicking)
                painter()->setColor(c, 0.3);
            painter()->setVertex(band_pts[first].left);
            painter()->setVertex(band_pts[first].right);
            painter()->endLine();
            if(last != first) {
                painter()->beginLine(1.0);
                if( !colorpicking)
                    painter()->setColor(c, 0.3);
                painter()->setVertex(band_pts[last].left);
                painter()->setVertex(band_pts[last].right);
                painter()->endLine();
            }
        }
    }
}

void
OnPlotMaskObject::drawFilledTexture(const XGraph::ScrPoint s[4], bool colorpicking) {
    unsigned int fgc = baseColor();
    unsigned int r = (fgc >> 16) & 0xff, g = (fgc >> 8) & 0xff, b = fgc & 0xff;
    unsigned int alpha_val = colorpicking ? 255 : std::min(255u, (unsigned int)(0.4 * 255));
    std::vector<uint8_t> texdata(m_width * m_height * 4);
    for(unsigned int i = 0; i < m_width * m_height; ++i) {
        texdata[i * 4 + 0] = colorpicking ? 255 : r;
        texdata[i * 4 + 1] = colorpicking ? 255 : g;
        texdata[i * 4 + 2] = colorpicking ? 255 : b;
        texdata[i * 4 + 3] = (*m_mask)[i] ? alpha_val : 0;
    }
    GLuint tex;
    glGenTextures(1, &tex);
    glBindTexture(GL_TEXTURE_2D, tex);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, m_width, m_height, 0,
        GL_RGBA, GL_UNSIGNED_BYTE, texdata.data());

    glEnable(GL_TEXTURE_2D);
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    if( !colorpicking)
        glColor4f(1.0f, 1.0f, 1.0f, 1.0f);
    bool yNormal = (s[0].y < s[2].y);
    float v0 = yNormal ? 1.0f : 0.0f;
    float v1 = yNormal ? 0.0f : 1.0f;
    glBegin(GL_QUADS);
    glTexCoord2f(0.0f, v0); glVertex3f(s[0].x, s[0].y, s[0].z);
    glTexCoord2f(1.0f, v0); glVertex3f(s[1].x, s[1].y, s[1].z);
    glTexCoord2f(1.0f, v1); glVertex3f(s[2].x, s[2].y, s[2].z);
    glTexCoord2f(0.0f, v1); glVertex3f(s[3].x, s[3].y, s[3].z);
    glEnd();
    glDisable(GL_TEXTURE_2D);
    glBindTexture(GL_TEXTURE_2D, 0);
    glDeleteTextures(1, &tex);
}

