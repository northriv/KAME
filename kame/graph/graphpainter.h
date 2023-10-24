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
#ifndef GRAPH_PAINTER_H
#define GRAPH_PAINTER_H

#include "graph.h"
#include "graphwidget.h"

#include <Qt>

#ifdef USE_QGLWIDGET
    #include <qgl.h>
#else
    #include <QOpenGLFunctions>
#endif

#ifdef __APPLE__
    #define USE_PBO
#endif

class OnScreenObject {
public:
    OnScreenObject(XQGraphPainter* p) : m_painter(p) {}
    virtual ~OnScreenObject() {}
    //! draws in OpenGL.
    virtual void drawNative() = 0;
    //! draws by QPainter.
    virtual void drawByPainter(QPainter *) = 0;
    virtual void drawOffScreenMarker() {}
    //! unlinks from XQGraphPainter.
    void release(const shared_ptr<OnScreenObject> &me);
protected:
    XQGraphPainter *painter() const {return m_painter;}
private:
    XQGraphPainter *const m_painter;
};

template <class OSO>
class OnPlotObject : public OSO {
public:
    template <typename... Args>
    OnPlotObject(XQGraphPainter* p, Args&&... args) : OSO(p, std::forward<Args>(args)...) {}

    void placeObject(const shared_ptr<XPlot> &plot,
                     const XGraph::ValPoint corners[4]);

    virtual void drawNative() override;
    virtual void drawByPainter(QPainter *) override;
    virtual void drawOffScreenMarker() override;
private:
    void valToScreen();
    weak_ptr<XPlot> m_plot;
    XGraph::ValPoint m_corners[4];
};


class OnScreenObjectWithMarker : public OnScreenObject {
public:
    OnScreenObjectWithMarker(XQGraphPainter* p) : OnScreenObject(p) {}
    //draws objects/bounding box for GL_SELECT
    virtual void drawOffScreenMarker() override;
    enum class HowToEvade {Never, ByAscent, ByDescent, ToLeft, ToRight, Hide};
    void placeObject(const XGraph::ScrPoint &init_lefttop, const XGraph::ScrPoint &init_righttop,
        const XGraph::ScrPoint &init_rightbottom, const XGraph::ScrPoint &init_leftbottom, HowToEvade direction, XGraph::SFloat space);
    void placeObject(const XGraph::ValPoint corners[4]);
//    void evadeOnScreenObjects(const std::deque<std::weak_ptr<OnScreenObject>> &list, XGraph::SFloat space);
//    static bool evadeMousePointer(const std::deque<std::weak_ptr<OnScreenObject>> &list);
    XGraph::ScrPoint &leftTop() {return m_leftTop;}
    XGraph::ScrPoint &rightTop() {return m_rightTop;}
    XGraph::ScrPoint &rightBottom() {return m_rightBottom;}
    XGraph::ScrPoint &leftBottom() {return m_leftBottom;}
protected:
    XGraph::ScrPoint m_leftTop, m_rightBottom, m_leftBottom, m_rightTop;
    XGraph::SFloat m_space;
    HowToEvade m_direction;
};

class OnScreenRectObject : public OnScreenObjectWithMarker {
public:
    enum class Type {Selection, AreaTool};
    OnScreenRectObject(XQGraphPainter* p, Type type, unsigned int basecolor = 0x0000ffu) :
        OnScreenObjectWithMarker(p), m_type(type), m_baseColor(basecolor) {}
    //! draws in OpenGL.
    virtual void drawNative() override;
    //! draws by QPainter.
    virtual void drawByPainter(QPainter *) override {}
private:
    Type m_type;
    unsigned int m_baseColor;
};

using OnPlotRectObject = OnPlotObject<OnScreenRectObject>;

#include <QImage>
class OnScreenTexture : public OnScreenObjectWithMarker {
public:
   OnScreenTexture(XQGraphPainter *const item, GLuint tid, const shared_ptr<QImage> &image)
       : OnScreenObjectWithMarker(item), id(tid), qimage(image) {}
   virtual ~OnScreenTexture();
   //! update texture by new image.
   void repaint(const shared_ptr<QImage> &image);
   //! draws in OpenGL.
   virtual void drawNative() override;
   //! draws by QPainter.
   virtual void drawByPainter(QPainter *) override {}
private:
   const GLuint id;
   shared_ptr<QImage> qimage;
};

class OnScreenTextObject : public OnScreenObjectWithMarker {
public:
    virtual void drawNative() override;
    virtual void drawByPainter(QPainter *) override;
    virtual void drawOffScreenMarker() override;

    void updateText(const XString &text);

    void clear();
    void drawText(const XGraph::ScrPoint &p, const XString &str);
    void defaultFont();
    //! \param start where text be aligned
    //! \param dir a direction where text be aligned
    //! \param width perp. to \a dir, restricting font size
    //! \return return 0 if succeeded
    int selectFont(const XString &str, const XGraph::ScrPoint &start,
                   const XGraph::ScrPoint &dir, const XGraph::ScrPoint &width, int sizehint = 0);
private:
    QString m_text;
    int m_curFontSize;
    int m_curAlign;
    struct Text {
        XGraph::ScrPoint pos, corners[4];
        QRgb rgba;
        ssize_t strpos, length;
    };
    std::vector<Text> m_textOverpaint; //stores text to be overpainted.
};

//! A painter which holds off-screen pixmap
//! and provides a way to draw
//! not thread-safe
class XQGraphPainter : public enable_shared_from_this<XQGraphPainter>
#ifndef USE_QGLWIDGET
        , protected QOpenGLFunctions
#endif
{
 friend class OnScreenTexture;
public:
	XQGraphPainter(const shared_ptr<XGraph> &graph, XQGraph* item);
 virtual ~XQGraphPainter();
 
 //! Selections  
 enum class SelectionMode {SelNone, SelPoint, SelAxis, SelPlane, TiltTracking};
 enum class SelectionState {SelStart, SelFinish, SelFinishByTool, Selecting};
 using SelectedResult = std::tuple<shared_ptr<XAxis>, XGraph::ScrPoint, XGraph::ScrPoint>;
 std::pair<SelectedResult, SelectedResult> selectObjs(int x, int y, SelectionState state, SelectionMode mode = SelectionMode::SelNone,
    const XString &tool_desc = {});
 
 void wheel(int x, int y, double deg);
 void zoom(double zoomscale, int x, int y);
 void showHelp();
 
 //! view
 //! \param angle CCW
 //! \param x,y,z rotate axis
 //! \param init initialize view matrix or not
 void viewRotate(double angle, double x, double y, double z, bool init = false);  
 
 //! drawings
 
 void setColor(float r, float g, float b, float a = 1.0f) {
    glColor4f(r, g, b, a );
    m_curTextColor = QColor(lrintf(r * 256.0), lrintf(g * 256.0), lrintf(b * 256.0), a).rgba();
}
 void setColor(unsigned int rgb, float a = 1.0f) {
    QColor qc = QRgb(rgb);
    glColor4f(qc.red() / 256.0, qc.green() / 256.0, qc.blue() / 256.0, a );
    qc.setAlpha(lrintf(a * 255));
    m_curTextColor = qc.rgba();
}
 void setVertex(const XGraph::ScrPoint &p) {
    glVertex3f(p.x, p.y, p.z);
 }
 
 void beginLine(double size = 1.0, unsigned short stipple = 0u);
 void endLine();
 
 void beginPoint(double size = 1.0);
 void endPoint();
 
 void beginQuad(bool fill = false);
 void endQuad();

 void secureWindow(const XGraph::ScrPoint &p);

 //obsolete
 void drawText(const XGraph::ScrPoint &p, QString &&str);
 //obsolete
 void drawText(const XGraph::ScrPoint &p, const XString &str) {
     drawText(p, QString(str));
 }

 //On-screen Objects
 shared_ptr<OnScreenTexture> createTexture(const shared_ptr<QImage> &image, bool onetime = false);
 void drawTexture(const OnScreenTexture& texture, const XGraph::ScrPoint p[4]);
 template <class T, typename...Args>
 shared_ptr<T> createOnScreenObject(bool onetime, Args&&... args) {
     auto p = std::make_shared<T>(this, std::forward<Args>(args)...);
     if(onetime)
         m_paintedOSOs.push_back(p);
     else
         m_persistentOSOs.push_back(p);
     return p;
 }
 void removeOnScreenObject(const shared_ptr<OnScreenObject> &p);

 //! make point outer perpendicular to \a dir by offset
 //! \param offset > 0 for outer, < 0 for inner. unit is of screen coord.
 void posOffAxis(const XGraph::ScrPoint &dir, XGraph::ScrPoint *src, XGraph::SFloat offset);
 void defaultFont();
 //! \param start where text be aligned
 //! \param dir a direction where text be aligned
 //! \param width perp. to \a dir, restricting font size
 //! \return return 0 if succeeded
 //obsolete
 int selectFont(const XString &str, const XGraph::ScrPoint &start,
				const XGraph::ScrPoint &dir, const XGraph::ScrPoint &width, int sizehint = 0);
 
 //! minimum resolution of screen coordinate.
 float resScreen();
 //! openGL stuff
 void initializeGL ();
 void resizeGL ( int width, int height );
 void paintGL ();

 //for retina support.
 double m_pixel_ratio;
private:
 friend class OnScreenObject;
 friend class OnScreenRectObject;
 friend class OnScreenTextObject;

 //! coordinate conversions
 //! \ret zero for success
 int windowToScreen(int x, int y, double z, XGraph::ScrPoint *scr);
 int screenToWindow(const XGraph::ScrPoint &scr, double *x, double *y, double *z);
 
 //! Selections  
 //! \param x,y center of clipping area
 //! \param dx,dy clipping width,height
 //! \param dz window of depth
 //! \param scr hits
 //! \param dsdx,dsdy diff. of 1 pixel
 //! \return found depth
 double selectPlane(int x, int y, int dx, int dy,
					XGraph::ScrPoint *scr, XGraph::ScrPoint *dsdx, XGraph::ScrPoint *dsdy );
 double selectAxis(int x, int y, int dx, int dy,
				   XGraph::ScrPoint *scr, XGraph::ScrPoint *dsdx, XGraph::ScrPoint *dsdy );
 double selectPoint(int x, int y, int dx, int dy,
					XGraph::ScrPoint *scr, XGraph::ScrPoint *dsdx, XGraph::ScrPoint *dsdy );
 
 shared_ptr<Listener> m_lsnRedraw;
 void onRedraw(const Snapshot &shot, XGraph *graph);
 
 shared_ptr<Listener> m_lsnRepaint;
 Transactional::TalkerOnce<Snapshot> m_tlkRepaint;
 void requestRepaint();
 void onRepaint(const Snapshot &shot);
 
 //! Draws plots, axes.
Snapshot startDrawing();
 void drawOffScreenGrids(const Snapshot &shot);
 void drawOffScreenPlaneMarkers(const Snapshot &shot); //!< for \a selectGL()
 void drawOffScreenPoints(const Snapshot &shot);
 void drawOffScreenAxes(const Snapshot &shot);
 void drawOffScreenAxisMarkers(const Snapshot &shot); //!< for \a selectGL()
 //! depends on viewpoint
 void drawOnScreenObj(const Snapshot &shot);
 //! independent of viewpoint. For coordinate, legend, hints. title,...
 void drawOnScreenViewObj(const Snapshot &shot);
 void drawOnScreenHelp(const Snapshot &shot, QPainter *qpainter);

 const shared_ptr<XGraph> m_graph;
 XQGraph *const m_pItem;
 
 shared_ptr<XPlot> m_foundPlane;
 shared_ptr<XAxis> m_foundPlaneAxis1, m_foundPlaneAxis2;
 shared_ptr<XAxis> m_foundAxis;
 
 shared_ptr<XAxis> findAxis(const Snapshot &shot, const XGraph::ScrPoint &s1);
 shared_ptr<XPlot> findPlane(const Snapshot &shot, const XGraph::ScrPoint &s1,
							 shared_ptr<XAxis> *axis1, shared_ptr<XAxis> *axis2);
 SelectionState m_selectionStateNow;
 SelectionMode m_selectionModeNow;
 XGraph::ScrPoint m_startScrPos, m_startScrDX, m_startScrDY;
 XGraph::ScrPoint m_finishScrPos, m_finishScrDX, m_finishScrDY;
 XString m_onScreenMsg;
 XString m_toolDescForSelection;
 int m_selStartPos[2];
 int m_tiltLastPos[2];
 int m_pointerLastPos[2];
 
 double selectGL(int x, int y, int dx, int dy, GLint list,
				 XGraph::ScrPoint *scr, XGraph::ScrPoint *dsdx, XGraph::ScrPoint *dsdy );
 void setInitView();
 
 GLint m_listpoints, m_listaxes,
    m_listaxismarkers, m_listgrids, m_listplanemarkers;
 
 atomic<bool> m_bIsRedrawNeeded;
// bool m_bAvoidCallingLists = false;
 bool m_bIsAxisRedrawNeeded;
 bool m_bTilted;
 bool m_bReqHelp;
 
 GLdouble m_proj_rot[16]; // Current Rotation matrix
 GLdouble m_proj[16]; // Current Projection matrix
 GLdouble m_model[16]; // Current Modelview matrix
 GLint m_viewport[4]; // Current Viewport
	
	//! ghost stuff
    void drawPersistentFrame(double persist_scale, const QColor &bgc);
    void storePersistentFrame();
    XTime m_modifiedTime;
	XTime m_updatedTime;
//   XGraph::ScrPoint DirProj; //direction vector of z of window coord.
	int m_curFontSize;
	int m_curAlign;
    GLuint m_persistentPBO = 0;
    std::vector<GLubyte> m_persistentFrame;

    struct Text {
        QString text;
        int x; int y;
        int fontsize;
        QRgb rgba;
    };
    std::vector<Text> m_textOverpaint; //stores text to be overpainted.
    QRgb m_curTextColor;
    void drawTextOverpaint(QPainter &qpainter);

    std::deque<shared_ptr<OnScreenObject>> m_persistentOSOs;
    std::deque<shared_ptr<OnScreenObject>> m_paintedOSOs;

    int m_minX, m_maxX, m_minY, m_maxY; //!< to determin window size by secureWinwdow().
};

#endif
