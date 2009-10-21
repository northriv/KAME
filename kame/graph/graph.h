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
#ifndef graphH
#define graphH

#include "xnode.h"
#include "xlistnode.h"
#include "xitemnode.h"

#include <vector>
#include <deque>

#include <qcolor.h>
#define clWhite (unsigned int)QColor(Qt::white).rgb()
#define clRed (unsigned int)QColor(Qt::red).rgb()
#define clLime (unsigned int)QColor(Qt::darkYellow).rgb()
#define clAqua (unsigned int)QColor(Qt::cyan).rgb()
#define clBlack (unsigned int)QColor(Qt::black).rgb()
#define clGreen (unsigned int)QColor(Qt::green).rgb()
#define clBlue (unsigned int)QColor(Qt::blue).rgb()

template <typename T>
struct Vector4 {
    Vector4() : x(0), y(0), z(0), w(1) {}
    Vector4(const Vector4 &v) : x(v.x), y(v.y), z(v.z), w(v.w) {}
    Vector4(T nx, T ny, T nz = 0, T nw = 1) : x(nx), y(ny), z(nz), w(nw) {} 
    T x, y, z, w;
    //! operators below do not take weights into account.
    bool operator==(const Vector4 &s1)  const {return ((x == s1.x) && (y == s1.y) && (z == s1.z));}
    Vector4 &operator+=(const Vector4<T> &s1) {
        x += s1.x; y += s1.y; z += s1.z;
        return *this;
    }
    Vector4 &operator-=(const Vector4 &s1) {
        x -= s1.x; y -= s1.y; z -= s1.z;
        return *this;
    }
    Vector4 &operator*=(T k) {
        x *= k; y *= k; z *= k;
        return *this;
    }
    //! square of distance between this and a point
    T distance2(const Vector4 &s1) const {
		T x1 = x - s1.x;
		T y1 = y - s1.y;
		T z1 = z - s1.z;
        return x1*x1 + y1*y1 + z1*z1;
    }
    //! square of distance between this and a line from s1 to s2
    T distance2(const Vector4 &s1, const Vector4 &s2) const  {
		T x1 = x - s1.x;
		T y1 = y - s1.y;
		T z1 = z - s1.z;
		T x2 = s2.x - s1.x;
		T y2 = s2.y - s1.y;
		T z2 = s2.z - s1.z;
		T zbab = x1*x2 + y1*y2 + z1*z2;
		T ab2 = x2*x2 + y2*y2 + z2*z2;
		T zb2 = x1*x1 + y1*y1 + z1*z1;
		return (zb2*ab2 - zbab*zbab) / ab2;
    }
    void normalize() {
        T ir = (T)1.0 / sqrtf(x*x + y*y + z*z);
        x *= ir; y *= ir; z *= ir;
    }
    Vector4 &vectorProduct(const Vector4 &s1) {
		Vector4 s2;
        s2.x = y * s1.z - z * s1.y;
        s2.y = z * s1.x - x * s1.z;
        s2.z = x * s1.y - y * s1.x;
        *this = s2;
        return *this;
    }
    T innerProduct(const Vector4 &s1) const {
        return x * s1.x + y * s1.y + z * s1.z;
    }
}; 

class XAxis;
class XGraph;
class XPlot;

class XQGraphPainter;

typedef XAliasListNode<XAxis> XAxisList;
typedef XAliasListNode<XPlot> XPlotList;

//! XGraph object can have one or more plots and two or more axes.
//! \sa XPlot, XAxis, XQGraphPainter
class XGraph : public XNode
{
	XNODE_OBJECT
protected:
	XGraph(const char *name, bool runtime);
public:
	virtual XString getLabel() const {return *label();}

	typedef float SFloat;
	static const SFloat SFLOAT_MAX;
	typedef float GFloat;
	static const GFloat GFLOAT_MAX;
	typedef double VFloat;
	static const VFloat VFLOAT_MAX;
	typedef Vector4<SFloat> ScrPoint;
	typedef Vector4<GFloat> GPoint;
	typedef Vector4<VFloat> ValPoint;
 
	//! call me before redraw graph
	//! Fix axes, take snapshot of points, autoscale axes
	void setupRedraw(float resolution);
  
	void zoomAxes(float resolution, XGraph::SFloat zoomscale,
				  const XGraph::ScrPoint &zoomcenter);

	const shared_ptr<XAxisList> &axes() const {return m_axes;}
	const shared_ptr<XPlotList> &plots() const {return m_plots;} 

	const shared_ptr<XStringNode> &label() const {return m_label;}
	const shared_ptr<XHexNode> &backGround() const {return m_backGround;}
	const shared_ptr<XHexNode> &titleColor() const {return m_titleColor;}

	const shared_ptr<XBoolNode> &drawLegends() const {return m_drawLegends;}

	const shared_ptr<XDoubleNode> &persistence() const {return m_persistence;}
	//! signal to redraw
	void requestUpdate();
	bool isUpdateScheduled() const {return m_bUpdateScheduled;}
	//! postpone signals to redraw
	void lock();
	//! reschedule signals to redraw
	void unlock();

	XTalker<shared_ptr<XGraph> > &onUpdate() {return m_tlkOnUpdate;}
	const shared_ptr<XListener> &lsnPropertyChanged() const {return m_lsnPropertyChanged;}
protected:
private:
	void onPropertyChanged(const shared_ptr<XValueNodeBase> &);

	bool m_bUpdateScheduled;
	XRecursiveRWLock m_graphLock;

	const shared_ptr<XStringNode> m_label;
	shared_ptr<XListener> m_lsnPropertyChanged;
  
	const shared_ptr<XAxisList> m_axes;
	const shared_ptr<XPlotList> m_plots; 

	const shared_ptr<XHexNode> m_backGround;
	const shared_ptr<XHexNode> m_titleColor;

	const shared_ptr<XBoolNode> m_drawLegends;

	const shared_ptr<XDoubleNode> m_persistence;
	XTalker<shared_ptr<XGraph> > m_tlkOnUpdate;
};

class XPlot : public XNode
{
	XNODE_OBJECT
protected:
	XPlot(const char *name, bool runtime, const shared_ptr<XGraph> &graph);
public:  
	virtual XString getLabel() const {return *label();}

	virtual int clearAllPoints(void) = 0;

	//! obtains values from screen coordinate
	//! if \a scr_prec > 0, value will be rounded around scr_prec
	//! \sa XAxis::AxisToVal.
	int screenToVal(const XGraph::ScrPoint &scr, XGraph::ValPoint *val,
					XGraph::SFloat scr_prec = -1);
	void screenToGraph(const XGraph::ScrPoint &pt, XGraph::GPoint *g);
	void graphToScreen(const XGraph::GPoint &pt, XGraph::ScrPoint *scr);
	void graphToVal(const XGraph::GPoint &pt, XGraph::ValPoint *val);

	const shared_ptr<XStringNode> &label() const {return m_label;}
  
	const shared_ptr<XUIntNode> &maxCount() const {return m_maxCount;}
	const shared_ptr<XBoolNode> &displayMajorGrid() const {return m_displayMajorGrid;}
	const shared_ptr<XBoolNode> &displayMinorGrid() const {return m_displayMinorGrid;}
	const shared_ptr<XBoolNode> &drawLines() const {return m_drawLines;}
	const shared_ptr<XBoolNode> &drawBars() const {return m_drawBars;}
	const shared_ptr<XBoolNode> &drawPoints() const {return m_drawPoints;}
	const shared_ptr<XBoolNode> &colorPlot() const {return m_colorPlot;}
	const shared_ptr<XHexNode> &majorGridColor() const {return m_majorGridColor;}
	const shared_ptr<XHexNode> &minorGridColor() const {return m_minorGridColor;}
	const shared_ptr<XHexNode> &pointColor() const {return m_pointColor;}
	const shared_ptr<XHexNode> &lineColor() const {return m_lineColor;}
	const shared_ptr<XHexNode> &barColor() const {return m_barColor;}//, BarInnerColor;
	const shared_ptr<XHexNode> &colorPlotColorHigh() const {return m_colorPlotColorHigh;}
	const shared_ptr<XHexNode> &colorPlotColorLow() const {return m_colorPlotColorLow;}
	const shared_ptr<XNode> &clearPoints() const {return m_clearPoints;}
	const shared_ptr<XItemNode<XAxisList, XAxis> > &axisX() const {return m_axisX;}
	const shared_ptr<XItemNode<XAxisList, XAxis> > &axisY() const {return m_axisY;}
	const shared_ptr<XItemNode<XAxisList, XAxis> > &axisZ() const {return m_axisZ;}
	const shared_ptr<XItemNode<XAxisList, XAxis> > &axisW() const {return m_axisW;}
	//! z value without AxisZ
	const shared_ptr<XDoubleNode> &zwoAxisZ() const {return m_zwoAxisZ;}
	const shared_ptr<XDoubleNode> &intensity() const {return m_intensity;}

	//! auto-scale
	virtual int validateAutoScale();
	//! draw points from snapshot
	int drawPlot(XQGraphPainter *painter);
	//! draw a point for legneds.
	//! \a spt the center of the point.
	//! \a dx,dy the size of the area.
	int drawLegend(XQGraphPainter *painter, const XGraph::ScrPoint &spt, float dx, float dy);
	void drawGrid(XQGraphPainter *painter, bool drawzaxis = true);  
	//! take a snap-shot all points for rendering 
	void snapshot();
  
	//! \return found index, if not return -1 
	int findPoint(int start, const XGraph::GPoint &gmin, const XGraph::GPoint &gmax,
				  XGraph::GFloat width, XGraph::ValPoint *val, XGraph::GPoint *g1);

	//! Lock info of axes.
	//! \return success or not
	bool trylock();
	void unlock();
protected:
	const weak_ptr<XGraph> m_graph;
	shared_ptr<XAxis> m_curAxisX, m_curAxisY, m_curAxisZ, m_curAxisW;

	virtual int setMaxCount(unsigned int count) = 0;
	virtual XGraph::ValPoint points(unsigned int index) const = 0;
	virtual unsigned int count() const = 0;
	//! data are copied to \p snappedPoints before rendering
	virtual XGraph::ValPoint snappedPoints(unsigned int index) const {return m_ptsSnapped[index];}
	virtual unsigned int snappedCount() const {return m_cntSnapped;}
  
	XGraph::ScrPoint m_scr0;
	XGraph::ScrPoint m_len;
  
private:
	struct tCanvasPoint {
		XGraph::GPoint graph; XGraph::ScrPoint scr; bool insidecube; unsigned int color;
	};
  
	const shared_ptr<XStringNode> m_label;
  
	const shared_ptr<XUIntNode> m_maxCount;
	const shared_ptr<XBoolNode> m_displayMajorGrid;
	const shared_ptr<XBoolNode> m_displayMinorGrid;
	const shared_ptr<XBoolNode> m_drawLines;
	const shared_ptr<XBoolNode> m_drawBars;
	const shared_ptr<XBoolNode> m_drawPoints;
	const shared_ptr<XBoolNode> m_colorPlot;
	const shared_ptr<XHexNode> m_majorGridColor;
	const shared_ptr<XHexNode> m_minorGridColor;
	const shared_ptr<XHexNode> m_pointColor;
	const shared_ptr<XHexNode> m_lineColor;
	const shared_ptr<XHexNode> m_barColor;//, BarInnerColor;
	const shared_ptr<XHexNode> m_colorPlotColorHigh;
	const shared_ptr<XHexNode> m_colorPlotColorLow;
	const shared_ptr<XNode> m_clearPoints;
	const shared_ptr<XItemNode<XAxisList, XAxis> > m_axisX;
	const shared_ptr<XItemNode<XAxisList, XAxis> > m_axisY;
	const shared_ptr<XItemNode<XAxisList, XAxis> > m_axisZ;
	const shared_ptr<XItemNode<XAxisList, XAxis> > m_axisW;
	//! z value without AxisZ
	const shared_ptr<XDoubleNode> m_zwoAxisZ;
	const shared_ptr<XDoubleNode> m_intensity;
  
	shared_ptr<XListener> m_lsnMaxCount;
	shared_ptr<XListener> m_lsnClearPoints;
  
	void onSetMaxCount(const shared_ptr<XValueNodeBase> &);
	void onClearPoints(const shared_ptr<XNode> &) {clearAllPoints();}
  
	bool clipLine(const tCanvasPoint &c1, const tCanvasPoint &c2, 
				  XGraph::ScrPoint *s1, XGraph::ScrPoint *s2, 
				  bool blendcolor, unsigned int *color1, unsigned int *color2, float *alpha1, float *alpha2);
	inline bool isPtIncluded(const XGraph::GPoint &pt);
	inline void validateAutoScaleOnePoint(const XGraph::ValPoint &pt);
    
	void drawGrid(XQGraphPainter *painter, shared_ptr<XAxis> &axis1, shared_ptr<XAxis> &axis2);

	std::vector<XGraph::ValPoint> m_ptsSnapped;
	std::vector<tCanvasPoint> m_canvasPtsSnapped; 
	unsigned int m_cntSnapped;
	void graphToScreenFast(const XGraph::GPoint &pt, XGraph::ScrPoint *scr);
	void valToGraphFast(const XGraph::ValPoint &pt, XGraph::GPoint *gr);
	unsigned int blendColor(unsigned int c1, unsigned int c2, float t);
};

class XAxis : public XNode
{
	XNODE_OBJECT
public:
	enum AxisDirection {DirAxisX, DirAxisY, DirAxisZ, AxisWeight};
	enum Tic {MajorTic, MinorTic, NoTics};  
protected:
	XAxis(const char *name, bool runtime,
		  AxisDirection dir, bool rightOrTop, const shared_ptr<XGraph> &graph);
public:
	virtual ~XAxis() {}

	virtual XString getLabel() const {return *label();}
  
	int drawAxis(XQGraphPainter *painter);
	//! obtains axis pos from value
	XGraph::GFloat valToAxis(XGraph::VFloat value);
	//! obtains value from position on axis
	//! \param pos normally, 0 < \a pos < 1
	//! \param axis_prec precision on axis. if > 0, value will be rounded
	XGraph::VFloat axisToVal(XGraph::GFloat pos, XGraph::GFloat axis_prec = -1);
	//! obtains axis pos from screen coordinate
	//! \return pos in axis
	XGraph::GFloat screenToAxis(const XGraph::ScrPoint &scr);
	//! obtains screen position from axis
	void axisToScreen(XGraph::GFloat pos, XGraph::ScrPoint *scr);
	void valToScreen(XGraph::VFloat val, XGraph::ScrPoint *scr);
	XGraph::VFloat screenToVal(const XGraph::ScrPoint &scr);
  
	XString valToString(XGraph::VFloat val);

	const shared_ptr<XStringNode> &label() const {return m_label;}
    
	const shared_ptr<XDoubleNode> &x() const {return m_x;}
	const shared_ptr<XDoubleNode> &y() const {return m_y;}
	const shared_ptr<XDoubleNode> &z() const {return m_z;} // in screen coordinate
	const shared_ptr<XDoubleNode> &length() const {return m_length;} // in screen coordinate
	const shared_ptr<XDoubleNode> &majorTicScale() const {return m_majorTicScale;}
	const shared_ptr<XDoubleNode> &minorTicScale() const {return m_minorTicScale;}
	const shared_ptr<XBoolNode> &displayMajorTics() const {return m_displayMajorTics;}
	const shared_ptr<XBoolNode> &displayMinorTics() const {return m_displayMinorTics;}
	const shared_ptr<XDoubleNode> &maxValue() const {return m_max;}
	const shared_ptr<XDoubleNode> &minValue() const {return m_min;}
	const shared_ptr<XBoolNode> &rightOrTopSided() const {return m_rightOrTopSided;} //sit on right, top

	const shared_ptr<XStringNode> &ticLabelFormat() const {return m_ticLabelFormat;}
	const shared_ptr<XBoolNode> &displayLabel() const {return m_displayLabel;}
	const shared_ptr<XBoolNode> &displayTicLabels() const {return m_displayTicLabels;}
	const shared_ptr<XHexNode> &ticColor() const {return m_ticColor;}
	const shared_ptr<XHexNode> &labelColor() const {return m_labelColor;}
	const shared_ptr<XHexNode> &ticLabelColor() const {return m_ticLabelColor;}
	const shared_ptr<XBoolNode> &autoFreq() const {return m_autoFreq;}
	const shared_ptr<XBoolNode> &autoScale() const {return m_autoScale;}
	const shared_ptr<XBoolNode> &logScale() const {return m_logScale;}

	void zoom(bool minchange, bool maxchange, XGraph::GFloat zoomscale,
			  XGraph::GFloat center = 0.5);

	//! obtains the type of tic and rounded value from position on axis
	Tic queryTic(int length, int pos, XGraph::VFloat *ticnum);

	//! call me befor drawing, autoscaling
	void startAutoscale(float resolution, bool clearscale = false);
	//! preserve changed scale
	void fixScale(float resolution, bool suppressupdate = false);
	//! fixed value
	XGraph::VFloat fixedMin() const {return m_minFixed;}
	XGraph::VFloat fixedMax() const {return m_maxFixed;}
  
	inline bool isIncluded(XGraph::VFloat x);
	inline void tryInclude(XGraph::VFloat x);

	const AxisDirection &direction() const {return m_direction;}
	const XGraph::ScrPoint &dirVector() const {return m_dirVector;}
protected:

private:
	AxisDirection m_direction;
	XGraph::ScrPoint m_dirVector;
  
	const weak_ptr<XGraph> m_graph;
  
	void _startAutoscale(bool clearscale);
	void drawLabel(XQGraphPainter *painter);
	void performAutoFreq(float resolution);
  
	const shared_ptr<XStringNode> m_label;
    
	const shared_ptr<XDoubleNode> m_x;
	const shared_ptr<XDoubleNode> m_y;
	const shared_ptr<XDoubleNode> m_z; // in screen coordinate
	const shared_ptr<XDoubleNode> m_length; // in screen coordinate
	const shared_ptr<XDoubleNode> m_majorTicScale;
	const shared_ptr<XDoubleNode> m_minorTicScale;
	const shared_ptr<XBoolNode> m_displayMajorTics;
	const shared_ptr<XBoolNode> m_displayMinorTics;
	const shared_ptr<XDoubleNode> m_max;
	const shared_ptr<XDoubleNode> m_min;
	const shared_ptr<XBoolNode> m_rightOrTopSided; //sit on right, top

	const shared_ptr<XStringNode> m_ticLabelFormat;
	const shared_ptr<XBoolNode> m_displayLabel;
	const shared_ptr<XBoolNode> m_displayTicLabels;
	const shared_ptr<XHexNode> m_ticColor;
	const shared_ptr<XHexNode> m_labelColor;
	const shared_ptr<XHexNode> m_ticLabelColor;
	const shared_ptr<XBoolNode> m_autoFreq;
	const shared_ptr<XBoolNode> m_autoScale;
	const shared_ptr<XBoolNode> m_logScale;
  
	XGraph::VFloat m_minFixed, m_maxFixed;
	XGraph::VFloat m_majorFixed, m_minorFixed;
	XGraph::VFloat m_invLogMaxOverMinFixed, m_invMaxMinusMinFixed;
	bool m_bLogscaleFixed;
	bool m_bAutoscaleFixed;
};

class XXYPlot : public XPlot
{
	XNODE_OBJECT
protected:
	XXYPlot(const char *name, bool runtime, const shared_ptr<XGraph> &graph) : 
		XPlot(name, runtime, graph) {}
public:
	int clearAllPoints();
	//! adds one point and draws
	int addPoint(XGraph::VFloat x, XGraph::VFloat y, XGraph::VFloat z = 0.0, XGraph::VFloat weight = 1.0);
	//! Direct Access.
	//! use XGraph::suspendUpdate() first.
	std::deque<XGraph::ValPoint> &points() {return m_points;}
protected:
	int setMaxCount(unsigned int count);
	XGraph::ValPoint points(unsigned int index) const;
	unsigned int count() const {return m_points.size();}
private:
	std::deque<XGraph::ValPoint> m_points;
};

class XFuncPlot : public XPlot
{
	XNODE_OBJECT
protected:
	XFuncPlot(const char *name, bool runtime, const shared_ptr<XGraph> &graph);
public:
	int clearAllPoints() {return 0;}
	virtual int validateAutoScale() {return 0;}
  
	virtual double func(double x) const = 0;
protected:
	virtual int setMaxCount(unsigned int count);
	virtual XGraph::ValPoint points(unsigned int index) const;
	virtual unsigned int count() const;
	virtual XGraph::ValPoint snappedPoints(unsigned int index) const;
	virtual unsigned int snappedCount() const;
private:
	int m_count;
};
//---------------------------------------------------------------------------
#endif
