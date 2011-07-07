/***************************************************************************
		Copyright (C) 2002-2011 Kentaro Kitagawa
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

#include "graph.h"

class FTFont;
class XQGraph;

#include <Qt>
#include <qgl.h>

//! A painter which holds off-screen pixmap
//! and provides a way to draw
//! not thread-safe
class XQGraphPainter : public enable_shared_from_this<XQGraphPainter> {
public:
	XQGraphPainter(const shared_ptr<XGraph> &graph, XQGraph* item);
 virtual ~XQGraphPainter();
 
 //! Selections  
 enum SelectionMode {SelNone, SelPoint, SelAxis, SelPlane, TiltTracking};
 enum SelectionState {SelStart, SelFinish, Selecting};
 void selectObjs(int x, int y, SelectionState state, SelectionMode mode = SelNone);
 
 void wheel(int x, int y, double deg);
 void zoom(double zoomscale, int x, int y);
 void showHelp();
 
 //! Repaint Off-screen obj, Paint On-screen obj. 
 void repaintGraph(int x1, int y1, int x2, int y2);
 
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
		 }
 void setVertex(const XGraph::ScrPoint &p) {
 glVertex3f(p.x, p.y, p.z);
	 }
 
 void beginLine(double size = 1.0);
 void endLine();
 
 void beginPoint(double size = 1.0);
 void endPoint();
 
 void beginQuad(bool fill = false);
 void endQuad();

 void drawText(const XGraph::ScrPoint &p, const XString &str);
 
 //! make point outer perpendicular to \a dir by offset
 //! \param offset > 0 for outer, < 0 for inner. unit is of screen coord.
 void posOffAxis(const XGraph::ScrPoint &dir, XGraph::ScrPoint *src, XGraph::SFloat offset);
 void defaultFont();
 //! \param start where text be aligned
 //! \param dir a direction where text be aligned
 //! \param width perp. to \a dir, restricting font size
 //! \return return 0 if succeeded
 int selectFont(const XString &str, const XGraph::ScrPoint &start,
				const XGraph::ScrPoint &dir, const XGraph::ScrPoint &width, int sizehint = 0);
 
 //! minimum resolution of screen coordinate.
 float resScreen();
 //! openGL stuff
 void initializeGL ();
 void resizeGL ( int width, int height );
 void paintGL ();
private:
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
 
 shared_ptr<XListener> m_lsnRedraw;
 void onRedraw(const Snapshot &shot, XGraph *graph);
 
 void repaintBuffer(int x1, int y1, int x2, int y2);
 //! do as possible as you can without screen.
 //! e.g. compile primitives, or make pixmap.
 void redrawOffScreen();
 
 //! Draws plots, axes.
 Snapshot startDrawing();
 void drawOffScreenGrids(const Snapshot &shot);
 void drawOffScreenPlanes(const Snapshot &shot);
 void drawOffScreenPoints(const Snapshot &shot);
 void drawOffScreenAxes(const Snapshot &shot);
 //! depends on viewpoint
 void drawOnScreenObj(const Snapshot &shot);
 //! independent of viewpoint. For coordinate, legend, hints. title,...
 void drawOnScreenViewObj(const Snapshot &shot);
 void drawOnScreenHelp(const Snapshot &shot);
 
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
 int m_selStartPos[2];
 int m_tiltLastPos[2];
 int m_pointerLastPos[2];
 
 double selectGL(int x, int y, int dx, int dy, GLint list,
				 XGraph::ScrPoint *scr, XGraph::ScrPoint *dsdx, XGraph::ScrPoint *dsdy );
 void setInitView();
 
 GLint m_listpoints, m_listaxes, m_listgrids, m_listplanes;
 
 bool m_bIsRedrawNeeded;
 bool m_bIsAxisRedrawNeeded;
 bool m_bTilted;
 bool m_bReqHelp;
 
 GLdouble m_proj_rot[16]; // Current Rotation matrix
	GLdouble m_proj[16]; // Current Projection matrix
	GLdouble m_model[16]; // Current Modelview matrix
	GLint m_viewport[4]; // Current Viewport
	
	//! ghost stuff
	std::vector<GLubyte> m_lastFrame;
	XTime m_modifiedTime;
	XTime m_updatedTime;
//   XGraph::ScrPoint DirProj; //direction vector of z of window coord.
	int m_curFontSize;
	int m_curAlign;
	static void openFont();
	static void closeFont();
	static std::wstring string2wstring(const XString &str);
	static int s_fontRefCount;
	static FTFont *s_pFont;  
};

#endif
