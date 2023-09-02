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
    Vector4() noexcept : x(0), y(0), z(0), w(1) {}
    Vector4(const Vector4 &) noexcept = default;
    Vector4(T nx, T ny, T nz = 0, T nw = 1) noexcept : x(nx), y(ny), z(nz), w(nw) {}
    Vector4& operator=(const Vector4 &) noexcept = default;
    //! operators below do not take weights into account.
    bool operator==(const Vector4 &s1)  const noexcept {
        return ((x == s1.x) && (y == s1.y) && (z == s1.z));
    }
    Vector4 &operator+=(const Vector4<T> &s1) noexcept {
        x += s1.x; y += s1.y; z += s1.z;
        return *this;
    }
    Vector4 &operator-=(const Vector4 &s1) noexcept {
        x -= s1.x; y -= s1.y; z -= s1.z;
        return *this;
    }
    Vector4 &operator*=(T k) noexcept {
        x *= k; y *= k; z *= k;
        return *this;
    }
    //! square of distance between this and a point
    T distance2(const Vector4 &s1) const noexcept {
		T x1 = x - s1.x;
		T y1 = y - s1.y;
		T z1 = z - s1.z;
        return x1*x1 + y1*y1 + z1*z1;
    }
    //! square of distance between this and a line from s1 to s2
    T distance2(const Vector4 &s1, const Vector4 &s2) const noexcept {
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
    void normalize() noexcept {
        T ir = (T)1.0 / sqrtf(x*x + y*y + z*z);
        x *= ir; y *= ir; z *= ir;
    }
    Vector4 &vectorProduct(const Vector4 &s1) noexcept {
		Vector4 s2;
        s2.x = y * s1.z - z * s1.y;
        s2.y = z * s1.x - x * s1.z;
        s2.z = x * s1.y - y * s1.x;
        *this = s2;
        return *this;
    }
    T innerProduct(const Vector4 &s1) const noexcept {
        return x * s1.x + y * s1.y + z * s1.z;
    }
    T x, y, z, w;
}; 

class XAxis;
class XGraph;
class XPlot;

class XQGraphPainter;

typedef XAliasListNode<XAxis> XAxisList;
typedef XAliasListNode<XPlot> XPlotList;

//! XGraph object can have one or more plots and two or more axes.
//! \sa XPlot, XAxis, XQGraphPainter
class DECLSPEC_KAME XGraph : public XNode {
public:
    XGraph(const char *name, bool runtime);
    virtual XString getLabel() const override {return ( **label())->to_str();}

    typedef float SFloat;
    typedef float GFloat;
    typedef double VFloat;
    typedef Vector4<SFloat> ScrPoint;
    typedef Vector4<GFloat> GPoint;
    typedef Vector4<VFloat> ValPoint;

    //! Fixes axes and performs autoscaling of the axes.
    //! Call this function before redrawal of the graph.
    void setupRedraw(Transaction &tr, float resolution);

    void zoomAxes(Transaction &tr, float resolution, XGraph::SFloat zoomscale,
                  const XGraph::ScrPoint &zoomcenter);

    enum class Theme {Night, DayLight, Current};
    void applyTheme(Transaction &tr, bool reset_to_default = false, Theme theme = Theme::Current);
    static Theme currentTheme() {return s_theme;}
    static void setCurrentTheme(Theme theme) {s_theme = theme;}

    const shared_ptr<XAxisList> &axes() const {return m_axes;}
    const shared_ptr<XPlotList> &plots() const {return m_plots;}

    const shared_ptr<XStringNode> &label() const {return m_label;}
    const shared_ptr<XHexNode> &backGround() const {return m_backGround;}
    const shared_ptr<XHexNode> &titleColor() const {return m_titleColor;}

    const shared_ptr<XBoolNode> &drawLegends() const {return m_drawLegends;}

    const shared_ptr<XDoubleNode> &persistence() const {return m_persistence;}

    const shared_ptr<XStringNode> &osdStrings() const {return m_osdStrings;}

    const shared_ptr<Listener> &lsnPropertyChanged() const {return m_lsnPropertyChanged;}

    struct Payload : public XNode::Payload {
        Talker<XGraph*> &onUpdate() {return m_tlkOnUpdate;}
        const Talker<XGraph*> &onUpdate() const {return m_tlkOnUpdate;}
    private:
        TalkerOnce<XGraph*> m_tlkOnUpdate;
    };

protected:
private:
    void onPropertyChanged(const Snapshot &shot, XValueNodeBase *);

    const shared_ptr<XStringNode> m_label;
    const shared_ptr<XAxisList> m_axes;
    const shared_ptr<XPlotList> m_plots;
    const shared_ptr<XHexNode> m_backGround;
    const shared_ptr<XHexNode> m_titleColor;
    const shared_ptr<XBoolNode> m_drawLegends;
    const shared_ptr<XDoubleNode> m_persistence;
    const shared_ptr<XStringNode> m_osdStrings;

    shared_ptr<Listener> m_lsnPropertyChanged;

    static Theme s_theme;
};

class DECLSPEC_KAME XPlot : public XNode {
public:
	XPlot(const char *name, bool runtime, Transaction &tr_graph, const shared_ptr<XGraph> &graph);
    virtual XString getLabel() const override {return ( **label())->to_str();}

	virtual void clearAllPoints(Transaction &tr) = 0;

	//! obtains values from screen coordinate
	//! if \a scr_prec > 0, value will be rounded around scr_prec
	//! \sa XAxis::AxisToVal.
	int screenToVal(const Snapshot &shot, const XGraph::ScrPoint &scr, XGraph::ValPoint *val,
                    XGraph::SFloat scr_prec = -1);
    void screenToGraph(const Snapshot &shot, const XGraph::ScrPoint &pt, XGraph::GPoint *g) const;
    void graphToScreen(const Snapshot &shot, const XGraph::GPoint &pt, XGraph::ScrPoint *scr);
    void graphToVal(const Snapshot &shot, const XGraph::GPoint &pt, XGraph::ValPoint *val);

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
	const shared_ptr<XTouchableNode> &clearPoints() const {return m_clearPoints;}
	const shared_ptr<XItemNode<XAxisList, XAxis> > &axisX() const {return m_axisX;}
	const shared_ptr<XItemNode<XAxisList, XAxis> > &axisY() const {return m_axisY;}
	const shared_ptr<XItemNode<XAxisList, XAxis> > &axisZ() const {return m_axisZ;}
	const shared_ptr<XItemNode<XAxisList, XAxis> > &axisW() const {return m_axisW;}
	//! z value without AxisZ
	const shared_ptr<XDoubleNode> &zwoAxisZ() const {return m_zwoAxisZ;}
	const shared_ptr<XDoubleNode> &intensity() const {return m_intensity;}

	//! auto-scale
	virtual int validateAutoScale(const Snapshot &shot);
	//! Draws points from snapshot
    virtual int drawPlot(const Snapshot &shot, XQGraphPainter *painter);
	//! Draws a point for legneds.
	//! \a spt the center of the point.
	//! \a dx,dy the size of the area.
    int drawLegend(const Snapshot &shot, XQGraphPainter *painter, const XGraph::ScrPoint &spt, float dx, float dy);
	void drawGrid(const Snapshot &shot, XQGraphPainter *painter, bool drawzaxis = true);
	//! Takes a snap-shot all points for rendering
	virtual void snapshot(const Snapshot &shot) = 0;
  
	//! \return found index, if not return -1 
	int findPoint(const Snapshot &shot, int start, const XGraph::GPoint &gmin, const XGraph::GPoint &gmax,
				  XGraph::GFloat width, XGraph::ValPoint *val, XGraph::GPoint *g1);

	//! \return success or not
	bool fixScales(const Snapshot &);
protected:
	const weak_ptr<XGraph> m_graph;
	shared_ptr<XAxis> m_curAxisX, m_curAxisY, m_curAxisZ, m_curAxisW;

	XGraph::ScrPoint m_scr0;
	XGraph::ScrPoint m_len;
	std::vector<XGraph::ValPoint> m_ptsSnapped;
  
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
	const shared_ptr<XTouchableNode> m_clearPoints;
	const shared_ptr<XItemNode<XAxisList, XAxis> > m_axisX;
	const shared_ptr<XItemNode<XAxisList, XAxis> > m_axisY;
	const shared_ptr<XItemNode<XAxisList, XAxis> > m_axisZ;
	const shared_ptr<XItemNode<XAxisList, XAxis> > m_axisW;
	//! z value without AxisZ
	const shared_ptr<XDoubleNode> m_zwoAxisZ;
	const shared_ptr<XDoubleNode> m_intensity;

	shared_ptr<Listener> m_lsnClearPoints;
  
	void onClearPoints(const Snapshot &, XTouchableNode *);
  
	inline bool clipLine(const tCanvasPoint &c1, const tCanvasPoint &c2,
				  XGraph::ScrPoint *s1, XGraph::ScrPoint *s2, 
                  bool blendcolor, unsigned int *color1, unsigned int *color2, float *alpha1, float *alpha2) const;
    inline bool isPtIncluded(const XGraph::GPoint &pt) const;
    
	void drawGrid(const Snapshot &shot,
		XQGraphPainter *painter, shared_ptr<XAxis> &axis1, shared_ptr<XAxis> &axis2);

	std::vector<tCanvasPoint> m_canvasPtsSnapped; 
    inline void graphToScreenFast(const XGraph::GPoint &pt, XGraph::ScrPoint *scr) const;
    inline void valToGraphFast(const XGraph::ValPoint &pt, XGraph::GPoint *gr) const;
    inline unsigned int blendColor(unsigned int c1, unsigned int c2, float t) const;
};

class DECLSPEC_KAME XAxis : public XNode {
public:
    enum class AxisDirection {X, Y, Z, Weight};
    enum class Tic {Major, Minor, None};

	XAxis(const char *name, bool runtime,
		  AxisDirection dir, bool rightOrTop, Transaction &tr_graph, const shared_ptr<XGraph> &graph);

    virtual XString getLabel() const override {return ( **label())->to_str();}
  
    int drawAxis(const Snapshot &shot, XQGraphPainter *painter);
	//! obtains axis pos from value
    XGraph::GFloat valToAxis(XGraph::VFloat value);
	//! obtains value from position on axis
	//! \param pos normally, 0 < \a pos < 1
	//! \param axis_prec precision on axis. if > 0, value will be rounded
    XGraph::VFloat axisToVal(XGraph::GFloat pos, XGraph::GFloat axis_prec = -1) const;
	//! obtains axis pos from screen coordinate
	//! \return pos in axis
    XGraph::GFloat screenToAxis(const Snapshot &shot, const XGraph::ScrPoint &scr) const;
	//! obtains screen position from axis
    void axisToScreen(const Snapshot &shot, XGraph::GFloat pos, XGraph::ScrPoint *scr) const;
    void valToScreen(const Snapshot &shot, XGraph::VFloat val, XGraph::ScrPoint *scr);
    XGraph::VFloat screenToVal(const Snapshot &shot, const XGraph::ScrPoint &scr) const;
  
    //! use formatDouble(fmt, val) for frequent usage.
    XString valToString(XGraph::VFloat val) const;

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

	//! Obtains the type of tic and rounded value from position on axis
    Tic queryTic(int length, int pos, XGraph::VFloat *ticnum);

	//! Call this function before drawing or autoscale.
	void startAutoscale(const Snapshot &shot, float resolution, bool clearscale = false);
	//! Preserves modified scale.
	void fixScale(Transaction &tr, float resolution, bool suppressupdate = false);
	//! fixed value
    XGraph::VFloat fixedMin() const noexcept {return m_minFixed;}
    XGraph::VFloat fixedMax() const noexcept {return m_maxFixed;}
  
    inline bool isIncluded(XGraph::VFloat x) const;
    inline void tryInclude(XGraph::VFloat x);

	const AxisDirection &direction() const {return m_direction;}
	const XGraph::ScrPoint &dirVector() const {return m_dirVector;}

private:
	AxisDirection m_direction;
	XGraph::ScrPoint m_dirVector;
  
	const weak_ptr<XGraph> m_graph;
  
	void startAutoscale_(const Snapshot &shot, bool clearscale);
	void drawLabel(const Snapshot &shot, XQGraphPainter *painter);
	void performAutoFreq(const Snapshot &shot, float resolution);
  
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

class DECLSPEC_KAME XXYPlot : public XPlot {
public:
	XXYPlot(const char *name, bool runtime, Transaction &tr_graph, const shared_ptr<XGraph> &graph) :
		XPlot(name, runtime, tr_graph, graph) {}

    virtual void clearAllPoints(Transaction &tr) override;
	//! Adds one point and draws.
	void addPoint(Transaction &tr,
		XGraph::VFloat x, XGraph::VFloat y, XGraph::VFloat z = 0.0, XGraph::VFloat weight = 1.0);

    struct Payload : public XNode::Payload {
        Payload() : XNode::Payload(), m_startPos(0) {}
        std::vector<XGraph::ValPoint> &points() {return m_points;}
        const std::vector<XGraph::ValPoint> &points() const {return m_points;}
        unsigned int m_startPos;
    private:
        std::vector<XGraph::ValPoint> m_points;
    };
protected:
	//! Takes a snap-shot all points for rendering
    virtual void snapshot(const Snapshot &shot) override;
};

class DECLSPEC_KAME XFuncPlot : public XPlot {
public:
	XFuncPlot(const char *name, bool runtime, Transaction &tr_graph, const shared_ptr<XGraph> &graph);
    void clearAllPoints(Transaction &) override {}
    virtual int validateAutoScale(const Snapshot &) override {return 0;}
  
	virtual double func(double x) const = 0;
protected:
	//! Takes a snap-shot all points for rendering
    virtual void snapshot(const Snapshot &shot) override;
private:
};

class XQGraphTexture;
//! displays pixmap as texture.
class DECLSPEC_KAME X2DImagePlot : public XPlot {
public:
    X2DImagePlot(const char *name, bool runtime, Transaction &tr_graph, const shared_ptr<XGraph> &graph);
    void clearAllPoints(Transaction &) override {}
    virtual int validateAutoScale(const Snapshot &) override {return 0;}

    void setImage(Transaction &tr, QImage&& image, double scr_width = 0.0, double scr_height = 0.0);

    struct Payload : public XNode::Payload {
        Payload() : XNode::Payload() {}
        //! image dimensions in screen coordinate.
        double scrWidth() const {return m_scrWidth;}
        double scrHeight() const {return m_scrHeight;}
        shared_ptr<QImage> image() const {return m_image;}
    private:
        friend class X2DImagePlot;
        shared_ptr<QImage> m_image;
        double m_scrWidth, m_scrHeight;
    };
protected:
    virtual int drawPlot(const Snapshot &shot, XQGraphPainter *painter) override;
    //! Takes a snap-shot all points for rendering
    virtual void snapshot(const Snapshot &shot) override;
private:
    shared_ptr<QImage> m_image, m_image_textured;
    unique_ptr<XQGraphTexture> m_texture;
};
//---------------------------------------------------------------------------
#endif
