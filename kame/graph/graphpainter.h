/***************************************************************************
        Copyright (C) 2002-2025 Kentaro Kitagawa
		                   kitag@issp.u-tokyo.ac.jp
		
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

#include "onscreenobject.h"

#ifdef USE_QGLWIDGET
    #include <qgl.h>
#else
    #include <QOpenGLFunctions>
#endif

#ifdef __APPLE__
    #define USE_PBO
#endif

//! A painter which holds off-screen pixmap
//! and provides a way to draw
//! not thread-safe
class DECLSPEC_KAME XQGraphPainter : public enable_shared_from_this<XQGraphPainter>
#ifndef USE_QGLWIDGET
        , protected QOpenGLFunctions
#endif
{
    friend class OnScreenTexture; //to access GL fn.
public:
        XQGraphPainter(const shared_ptr<XGraph> &graph, XQGraph* item);
     virtual ~XQGraphPainter();

     //! Selections
     enum class SelectionMode {SelNone, SelPoint, SelAxis, SelPlane, TiltTracking};
     enum class SelectionState {SelStart, SelFinish, SelFinishByTool, Selecting};
     using SelectedResult = std::tuple<shared_ptr<XAxis>, XGraph::VFloat, XGraph::VFloat>;
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
    }
     void setColor(unsigned int rgb, float a = 1.0f) {
        QColor qc = QRgb(rgb);
        glColor4f(qc.red() / 256.0, qc.green() / 256.0, qc.blue() / 256.0, a );
        qc.setAlpha(lrintf(a * 255));
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

     //On-screen Objects
     std::weak_ptr<OnScreenTexture> createTextureDuringListing(const shared_ptr<QImage> &image);
     void drawTexture(const OnScreenTexture& texture, const XGraph::ScrPoint p[4]);
     template <class T, typename...Args>
     shared_ptr<T> createOneTimeOnScreenObject(Args&&... args) {
         auto p = std::make_shared<T>(this, std::forward<Args>(args)...);
         assert(Transactional::isMainThread());
         m_paintedOSOs.push_back(p);
         return p;
     }
     template <class T, typename...Args>
     weak_ptr<T> createListedOnScreenObject(Args&&... args) {
         auto p = std::make_shared<T>(this, std::forward<Args>(args)...);
         XScopedLock<XMutex> lock(m_mutexOSO);
         m_listedOSOs.push_back(p);
         return p;
     }
     template <class T, typename...Args>
     shared_ptr<T> createOnScreenObjectWeakly(Args&&... args) {
         auto p = std::make_shared<T>(this, std::forward<Args>(args)...);
         XScopedLock<XMutex> lock(m_mutexOSO);
         m_weakptrOSOs.push_back(p);
         return p;
     }

     //! make point outer perpendicular to \a dir by offset
     //! \param offset > 0 for outer, < 0 for inner. unit is of screen coord.
     void posOffAxis(const XGraph::ScrPoint &dir, XGraph::ScrPoint *src, XGraph::SFloat offset);

     //! minimum resolution of screen coordinate.
     float resScreen();
     //! openGL stuff
     void initializeGL ();
     void resizeGL ( int width, int height );
     void paintGL ();

     //for retina support.
     double m_pixel_ratio;

     const shared_ptr<XGraph> graph() const {return m_graph;}
     XQGraph *widget() const {return m_pItem;}

     //! coordinate conversions
     //! \ret zero for success
     int windowToScreen(int x, int y, double z, XGraph::ScrPoint *scr);
     int screenToWindow(const XGraph::ScrPoint &scr, double *x, double *y, double *z);

     void requestRepaint();
 private:
     //! Selections
     //! \param x,y center of clipping area
     //! \param dx,dy clipping width,height
     //! \param dz window of depth
     //! \return found depth., object id, screen coord., dsdx,dsdy diff. by 1 pixel
    std::tuple<double, int, XGraph::ScrPoint, XGraph::ScrPoint, XGraph::ScrPoint> selectPlane(int x, int y, int dx, int dy);
    std::tuple<double, int, XGraph::ScrPoint, XGraph::ScrPoint, XGraph::ScrPoint> selectAxis(int x, int y, int dx, int dy);
    std::tuple<double, int, XGraph::ScrPoint, XGraph::ScrPoint, XGraph::ScrPoint> selectPoint(int x, int y, int dx, int dy);

     shared_ptr<Listener> m_lsnRedraw;
     void onRedraw(const Snapshot &shot, XGraph *graph);

     shared_ptr<Listener> m_lsnRepaint;
     Transactional::TalkerOnce<Snapshot> m_tlkRepaint;
     void onRepaint(const Snapshot &shot);

     //! Draws plots, axes.
    Snapshot startDrawing();
    //For color picking. Red is given by ObjClassColorR, Green is by ID (# in axes, for examplee).
    enum class ObjClassColorR {None, Point, Axis, Plane, Grid, OSO};
     void drawOffScreenGrids(const Snapshot &shot);
     void drawOffScreenPlanes(const Snapshot &shot, ObjClassColorR red_color_picking); //Color picking only.
     void drawOffScreenPoints(const Snapshot &shot, ObjClassColorR red_color_picking = ObjClassColorR::None); //ObjClassColorR::None if not color picking.
     void drawOffScreenAxes(const Snapshot &shot, ObjClassColorR red_color_picking = ObjClassColorR::None); //ObjClassColorR::None if not color picking.
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

     //\return depth, class, object id, screen coord., dsdx,dsdy diff. by 1 pixe
     //If select buffer is used, object class and id are zero.
     std::tuple<double, ObjClassColorR, int, XGraph::ScrPoint, XGraph::ScrPoint, XGraph::ScrPoint>
        pickObjectGL(int x, int y, int dx, int dy, GLint list);
     void setInitView();

     GLint m_listpoints = 0, m_listaxes = 0, m_listgrids = 0;

     GLint m_listpoints_picker = 0,
        m_listaxes_picker = 0, m_listplane_picker = 0;

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
    GLuint m_persistentPBO = 0;
    std::vector<GLubyte> m_persistentFrame;

    XMutex m_mutexOSO;
    std::deque<shared_ptr<OnScreenObject>> m_listedOSOs;
    std::deque<weak_ptr<OnScreenObject>> m_weakptrOSOs;
    std::vector<shared_ptr<OnScreenObject>> m_paintedOSOs;

    int m_minX, m_maxX, m_minY, m_maxY; //!< to determin window size by secureWinwdow().
};

#endif
