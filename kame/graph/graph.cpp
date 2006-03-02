#include "graph.h"
#include "support.h"

#include "graphpainter.h"
#include <qgl.h>
#include <klocale.h>

using std::min;
using std::max;

#define GRAPH_UI_DELAY 10

#include <float.h>
const XGraph::SFloat XGraph::SFLOAT_MAX = FLT_MAX;
const XGraph::GFloat XGraph::GFLOAT_MAX = FLT_MAX;
const XGraph::VFloat XGraph::VFLOAT_MAX = DBL_MAX;

#define SFLOAT_EXP expf
#define GFLOAT_EXP expf
#define VFLOAT_EXP exp

#define SFLOAT_POW powf
#define GFLOAT_POW powf
#define VFLOAT_POW pow

#define SFLOAT_LOG logf
#define GFLOAT_LOG logf
#define VFLOAT_LOG log

#define SFLOAT_LOG10 log10f
#define GFLOAT_LOG10 log10f
#define VFLOAT_LOG10 log10

#define SFLOAT_LRINT lrintf
#define GFLOAT_LRINT lrintf
#define VFLOAT_LRINT lrint

#define VFLOAT_RINT lrint

#define AxisToLabel 0.09
#define AxisToTicLabel 0.015
#define UNZOOM_ABIT 0.9

#define PLOT_POINT_SIZE 5.0

#define PLOT_POINT_INTENS 0.5
#define PLOT_LINE_INTENS 0.7
#define PLOT_BAR_INTENS 0.4

QString
XAxisList::getItemName(unsigned int index) const
{
  return *(*this)[index]->label();
}

XGraph::XGraph(const char *name, bool runtime) : 
    XNode(name, runtime),
    m_bUpdateScheduled(false),
    m_label(create<XStringNode>("Label", true)),
    m_lsnPropertyChanged(label()->onValueChanged().connectWeak(false, shared_from_this(),
        &XGraph::onPropertyChanged)),
    m_axes(create<XAxisList>("Axes", true)),
    m_plots(create<XPlotList>("Plots", true)),
    m_backGround(create<XHexNode>("BackGround", true)),
    m_titleColor(create<XHexNode>("TitleColor", true))
{
    backGround()->onValueChanged().connect(lsnPropertyChanged());
    titleColor()->onValueChanged().connect(lsnPropertyChanged());
    backGround()->value(clWhite);
    titleColor()->value(clBlack);
    
    label()->value(name);
    (axes()->create<XAxis>("XAxis", true, XAxis::DirAxisX
        , false, dynamic_pointer_cast<XGraph>(shared_from_this())))
        ->label()->value(i18n("X Axis"));
    (axes()->create<XAxis>("YAxis", true, XAxis::DirAxisY
        , false, dynamic_pointer_cast<XGraph>(shared_from_this())))
        ->label()->value(i18n("Y Axis"));    
}

void
XGraph::onPropertyChanged(const shared_ptr<XValueNodeBase> &)
{
    suspendUpdate();
	requestUpdate();
    resumeUpdate();
}

void
XGraph::requestUpdate()
{
  ASSERT(m_graphLock.isLocked());
  m_bUpdateScheduled = true;
}
void
XGraph::resumeUpdate()
{
  m_graphLock.writeUnlock();
  if(!m_graphLock.isLocked() && m_bUpdateScheduled)
    {
    	  m_bUpdateScheduled = false;
      onUpdate().talk(dynamic_pointer_cast<XGraph>(shared_from_this()));
    }
}
void
XGraph::suspendUpdate()
{
  m_graphLock.writeLock();
}

void
XGraph::setupRedraw(float resolution)
{
  m_graphLock.readLock();
  
  m_bUpdateScheduled = false;
  
  { XScopedReadLock<XRecursiveRWLock> lock(axes()->childMutex());
      for(unsigned int i = 0; i < axes()->count(); i++)
        {
            (*axes())[i]->startAutoscale(resolution, *(*axes())[i]->autoScale() );
        }
      { XScopedReadLock<XRecursiveRWLock> lock(plots()->childMutex());
      for(unsigned int i = 0; i < plots()->count(); i++)
        {
          if(!(*plots())[i]->lockAxesInfo()) continue;
          (*plots())[i]->snapshot();
          (*plots())[i]->validateAutoScale();
          (*plots())[i]->unlockAxesInfo();
        }
      }
      for(unsigned int i = 0; i < axes()->count(); i++)
        {
          if(*(*axes())[i]->autoScale())
    		(*axes())[i]->zoom(true, true, UNZOOM_ABIT);
          (*axes())[i]->fixScale(resolution, true);
        }
  }
  
  m_graphLock.readUnlock();
}

void
XGraph::zoomAxes(float resolution, 
    XGraph::GFloat scale, const XGraph::ScrPoint &center)
{
  XScopedReadLock<XRecursiveRWLock> lock(axes()->childMutex());

  for(unsigned int i = 0; i < axes()->count(); i++)
    {
      (*axes())[i]->startAutoscale(resolution);
    }
//   validateAutoScale();
  suspendUpdate();
  for(unsigned int i = 0; i < axes()->count(); i++)
    {
    	(*axes())[i]->zoom(true, true, scale, (*axes())[i]->screenToAxis(center));
        (*axes())[i]->fixScale(resolution);
    }
  resumeUpdate();
}

XPlot::XPlot(const char *name, bool runtime, const shared_ptr<XGraph> &graph)
  : XNode(name, runtime),
  m_graph(graph),
  m_maxCount(create<XUIntNode>("MaxCount", true)),
  m_displayMajorGrid(create<XBoolNode>("DisplayMajorGrid", true)),
  m_displayMinorGrid(create<XBoolNode>("DisplayMinorGrid", true)),
  m_drawLines(create<XBoolNode>("DrawLines", true)),
  m_drawBars(create<XBoolNode>("DrawBars", true)),
  m_drawPoints(create<XBoolNode>("DrawPoints", true)),
  m_colorPlot(create<XBoolNode>("ColorPlot", true)),
  m_majorGridColor(create<XHexNode>("MajorGridColor", true)),
  m_minorGridColor(create<XHexNode>("MinorGridColor", true)),
  m_pointColor(create<XHexNode>("PointColor", true)),
  m_lineColor(create<XHexNode>("LineColor", true)),
  m_barColor(create<XHexNode>("BarColor", true)), //, BarInnerColor;
  m_colorPlotColorHigh(create<XHexNode>("ColorPlotColorHigh", true)),
  m_colorPlotColorLow(create<XHexNode>("ColorPlotColorLow", true)),
  m_clearPoints(create<XNode>("ClearPoints", true)),
  m_axisX(create<XItemNode<XAxisList, XAxis> >("AxisX", true, graph->axes())),
  m_axisY(create<XItemNode<XAxisList, XAxis> >("AxisY", true, graph->axes())),
  m_axisZ(create<XItemNode<XAxisList, XAxis> >("AxisZ", true, graph->axes())),
  //! z value without AxisZ
  m_zwoAxisZ(create<XDoubleNode>("ZwoAxisZ", true)),
  m_intensity(create<XDoubleNode>("Intensity", true)),
  m_cntSnapped(0)
{
//  MaxCount.value(0);
  drawLines()->value(true);
  drawBars()->value(false);
  drawPoints()->value(true);
  majorGridColor()->value(clAqua);
  minorGridColor()->value(clLime);
  lineColor()->value(clRed);
  pointColor()->value(clRed);
  barColor()->value(clRed);
  displayMajorGrid()->value(true);
  displayMinorGrid()->value(false);
  intensity()->setFormat("%.2f");
  intensity()->value(1.0);
  colorPlot()->value(false);
  colorPlotColorHigh()->value(clRed);
  colorPlotColorLow()->value(clBlue);
    
  m_lsnMaxCount = maxCount()->onValueChanged().connectWeak
        (false, shared_from_this(), &XPlot::onSetMaxCount);
  m_lsnClearPoints = clearPoints()->onTouch().connectWeak
        (false, shared_from_this(), &XPlot::onClearPoints);

  drawLines()->onValueChanged().connect(graph->lsnPropertyChanged());
  drawBars()->onValueChanged().connect(graph->lsnPropertyChanged());
  drawPoints()->onValueChanged().connect(graph->lsnPropertyChanged());
  majorGridColor()->onValueChanged().connect(graph->lsnPropertyChanged());
  minorGridColor()->onValueChanged().connect(graph->lsnPropertyChanged());
  pointColor()->onValueChanged().connect(graph->lsnPropertyChanged());
  lineColor()->onValueChanged().connect(graph->lsnPropertyChanged());
  barColor()->onValueChanged().connect(graph->lsnPropertyChanged());
  displayMajorGrid()->onValueChanged().connect(graph->lsnPropertyChanged());
  displayMinorGrid()->onValueChanged().connect(graph->lsnPropertyChanged());
  intensity()->onValueChanged().connect(graph->lsnPropertyChanged());
  colorPlot()->onValueChanged().connect(graph->lsnPropertyChanged());
  colorPlotColorHigh()->onValueChanged().connect(graph->lsnPropertyChanged());
  colorPlotColorLow()->onValueChanged().connect(graph->lsnPropertyChanged());
    
  zwoAxisZ()->value(0.15);
}

bool
XPlot::lockAxesInfo()
{
  axisX()->listMutex().readLock();
  axisY()->listMutex().readLock();
  axisZ()->listMutex().readLock();
  shared_ptr<XAxis> axisx = *axisX();
  shared_ptr<XAxis> axisy = *axisY();
  shared_ptr<XAxis> axisz = *axisZ();
  if(!axisx || !axisy) {
      axisX()->listMutex().readUnlock();
      axisY()->listMutex().readUnlock();
      axisZ()->listMutex().readUnlock();
      return false;
  }
  m_curAxisX = axisx;
  m_curAxisY = axisy;
  m_curAxisZ = axisz;
  m_scr0.x = *m_curAxisX->x();
  m_scr0.y = *m_curAxisY->y();
  m_len.x = *m_curAxisX->length();
  m_len.y = *m_curAxisY->length();
  if(!m_curAxisZ) {
  	m_scr0.z = *zwoAxisZ();
	m_len.z = (XGraph::SFloat)0.0;
  }
  else {
  	m_scr0.z = *m_curAxisZ->z();
	m_len.z = *m_curAxisZ->length();
  }
  return true;
}

void
XPlot::unlockAxesInfo()
{
      axisX()->listMutex().readUnlock();
      axisY()->listMutex().readUnlock();
      axisZ()->listMutex().readUnlock();
}
void
XPlot::screenToGraph(const XGraph::ScrPoint &pt, XGraph::GPoint *g)
{
  shared_ptr<XAxis> axisx = *axisX();
  shared_ptr<XAxis> axisy = *axisY();
  shared_ptr<XAxis> axisz = *axisZ();
	if(!axisz)
		g->z = (XGraph::GFloat)0.0;
	else
		g->z = (pt.z - *axisz->z()) / *axisz->length();
	g->x = (pt.x - *axisx->x()) / *axisx->length();
	g->y = (pt.y - *axisy->y()) / *axisy->length();
}
void
XPlot::graphToVal(const XGraph::GPoint &pt, XGraph::ValPoint *val)
{
  if(!lockAxesInfo()) return;
  val->x = m_curAxisX->axisToVal(pt.x);
  val->y = m_curAxisY->axisToVal(pt.y);
  if(m_curAxisZ)
  	val->z = m_curAxisZ->axisToVal(pt.z);
  else
  	val->z = (XGraph::VFloat)0.0;
  unlockAxesInfo();
}
int
XPlot::screenToVal(const XGraph::ScrPoint &scr, XGraph::ValPoint *val, XGraph::SFloat scr_prec)
{
  if(!lockAxesInfo()) return -1;
  val->x = m_curAxisX->axisToVal(m_curAxisX->screenToAxis(scr), scr_prec / m_len.x);
  val->y = m_curAxisY->axisToVal(m_curAxisY->screenToAxis(scr), scr_prec / m_len.y);
  val->z = (m_curAxisZ) ? m_curAxisZ->axisToVal(m_curAxisZ->screenToAxis(scr), scr_prec / m_len.z) : (XGraph::GFloat)0.0;
  unlockAxesInfo();
  return 0;
}
void
XPlot::graphToScreen(const XGraph::GPoint &pt, XGraph::ScrPoint *scr)
{
  if(!lockAxesInfo()) return;
  graphToScreenFast(pt, scr);
  unlockAxesInfo();
}
void
XPlot::graphToScreenFast(const XGraph::GPoint &pt, XGraph::ScrPoint *scr)
{
  scr->x = m_scr0.x + m_len.x * pt.x;
  scr->y = m_scr0.y + m_len.y * pt.y;
  scr->z = m_scr0.z + m_len.z * pt.z;
}
void
XPlot::valToGraphFast(const XGraph::ValPoint &pt, XGraph::GPoint *gr)
{
  gr->x = m_curAxisX->valToAxis(pt.x);
  gr->y = m_curAxisY->valToAxis(pt.y);
  if(m_curAxisZ)
	gr->z = m_curAxisZ->valToAxis(pt.z);
  else
  	gr->z = 0.0;
}
int
XPlot::findPoint(int start, const XGraph::GPoint &gmin, const XGraph::GPoint &gmax, 
		XGraph::GFloat width, XGraph::ValPoint *val, XGraph::GPoint *g1)
{
	if(!lockAxesInfo()) return -1;
	for(unsigned int i = start; i < snappedCount(); i++) {
	XGraph::ValPoint v;
	XGraph::GPoint g2;
		v = snappedPoints(i);
		valToGraphFast(v, &g2);
		if(g2.distance2(gmin, gmax) < width*width) {
			*val = v;
			*g1 = g2;
			unlockAxesInfo();
			return i;
		}
	}
	unlockAxesInfo();
	return -1;
}

void
XPlot::onSetMaxCount(const shared_ptr<XValueNodeBase> &)
{
      int cnt = *maxCount();
      setMaxCount(cnt);
}

void
XPlot::drawGrid(XQGraphPainter *painter, shared_ptr<XAxis> &axis1, shared_ptr<XAxis> &axis2)
{
  int len = SFLOAT_LRINT(1.0/painter->resScreen());
  painter->beginLine(1.0);
  if(*displayMajorGrid() || *displayMinorGrid())
    {
      XGraph::ScrPoint s1, s2;
      for(int i = 0; i < len; i++)
        {
	 XGraph::GFloat x = (XGraph::GFloat)i/len;
	  graphToScreenFast(XGraph::GPoint((axis1 == m_curAxisX) ? x : 0.0,
         (axis1 == m_curAxisY) ? x : 0.0, (axis1 == m_curAxisZ) ? x : 0.0), &s1);
	  graphToScreenFast(XGraph::GPoint(
	  	(axis1 == m_curAxisX) ? x : ((axis2 == m_curAxisX) ? 1.0 : 0.0),
	  	(axis1 == m_curAxisY) ? x : ((axis2 == m_curAxisY) ? 1.0 : 0.0),
	  	(axis1 == m_curAxisZ) ? x : ((axis2 == m_curAxisZ) ? 1.0 : 0.0)),
		&s2);
	  XGraph::VFloat tempx;
	  switch(axis1->queryTic(len, i, &tempx))
            {
            case XAxis::MajorTic:
	      if(*displayMajorGrid())
                {
		  painter->setColor(*majorGridColor(),
             max(0.0, min(*intensity() * 0.7, 0.5)) );
		  painter->setVertex(s1);
		  painter->setVertex(s2);
                }
	      break;
            case XAxis::MinorTic:
	      if(*displayMinorGrid())
                {
		  painter->setColor(*minorGridColor(),
             max(0.0, min(*intensity() * 0.5, 0.5)) );
		  painter->setVertex(s1);
		  painter->setVertex(s2);
                }
	      break;
	    default:
	      break;
            }
        }
    }
    painter->endLine();
}
void
XPlot::drawGrid(XQGraphPainter *painter, bool drawzaxis)
{
  if(!lockAxesInfo()) return;
  
  drawGrid(painter, m_curAxisX, m_curAxisY);
  drawGrid(painter, m_curAxisY, m_curAxisX);
  if(m_curAxisZ && drawzaxis) {
	drawGrid(painter, m_curAxisX, m_curAxisZ);
  	drawGrid(painter, m_curAxisZ, m_curAxisX);
	drawGrid(painter, m_curAxisY, m_curAxisZ);
  	drawGrid(painter, m_curAxisZ, m_curAxisY);
  }
  
  unlockAxesInfo();
}
int
XPlot::drawPlot(XQGraphPainter *painter)
{
  if(!lockAxesInfo()) return -1;
  
  bool colorplot = *colorPlot();
  int cnt = snappedCount();
  tCanvasPoint *cpt;
  {
	XGraph::ScrPoint s1;
	XGraph::GPoint g1;
	unsigned int colorhigh = *colorPlotColorHigh();
	unsigned int colorlow = *colorPlotColorLow();
	cpt = &m_canvasPtsSnapped[0];
	for(int i = 0; i < cnt; i++)
	{
		XGraph::ValPoint pt = snappedPoints(i);
		valToGraphFast(pt, &g1);
		graphToScreenFast(g1, &s1);
		cpt->scr = s1;
		cpt->graph = g1;
		cpt->insidecube = isPtIncluded(g1);
		if(colorplot)
			cpt->color = blendColor(colorlow, colorhigh, cpt->graph.z);
		cpt++;
	}
  }
  if(*drawBars())
    {
	XGraph::ScrPoint s1, s2;
	tCanvasPoint pt2;
	double g0y = m_curAxisY->valToAxis(0.0);
	m_curAxisY->axisToScreen(g0y, &s2);
	double s0y = s2.y;
	float alpha = max(0.0f, min((float)(*intensity() * PLOT_BAR_INTENS), 1.0f));
	painter->setColor(*barColor(), alpha );
	painter->beginLine(1.0);
	cpt = &m_canvasPtsSnapped[0];
	for(int i = 0; i < cnt; i++)
	{
		pt2 = *cpt;
		pt2.graph.y = g0y;
		pt2.scr.y = s0y;
		pt2.insidecube = isPtIncluded(pt2.graph);
		if(clipLine(*cpt, pt2, &s1, &s2, false, NULL, NULL))
		{
			if(colorplot) painter->setColor(cpt->color, alpha); 
			painter->setVertex(s1);
			painter->setVertex(s2);
		}
		cpt++;
	}
	painter->endLine();
  }
  if(*drawLines())
  {
    float alpha = max(0.0f, min((float)(*intensity() * PLOT_LINE_INTENS), 1.0f));
    painter->setColor(*lineColor(), alpha );
    painter->beginLine(1.0);
    XGraph::ScrPoint s1, s2;
    cpt = &m_canvasPtsSnapped[0];
    unsigned int color1, color2;
    for(int i = 1; i < cnt; i++)
    {
		if(clipLine(*cpt, *(cpt + 1), &s1, &s2, colorplot, &color1, &color2)) {
 			if(colorplot) painter->setColor(color1, alpha);
			painter->setVertex(s1);
 			if(colorplot) painter->setColor(color2, alpha);
			painter->setVertex(s2);
		}
		cpt++;
    }
    painter->endLine();
  }
  if(*drawPoints())
  {
    float alpha = max(0.0f, min((float)(*intensity() * PLOT_POINT_INTENS), 1.0f));
    painter->setColor(*pointColor(), alpha );
     painter->beginPoint(PLOT_POINT_SIZE);
    cpt = &m_canvasPtsSnapped[0];
    for(int i = 0; i < cnt; i++)
    {
  	if(cpt->insidecube) {
		if(colorplot) painter->setColor(cpt->color, alpha);
		painter->setVertex(cpt->scr);
	}
	cpt++;
    }
     painter->endPoint();
  }
  
  unlockAxesInfo();

  return 0;

}

unsigned int
XPlot::blendColor(unsigned int c1, unsigned int c2, float t)
{
	unsigned char c1red = qRed((QRgb)c1);
	unsigned char c1green = qGreen((QRgb)c1);
	unsigned char c1blue = qBlue((QRgb)c1);
	unsigned char c2red = qRed((QRgb)c2);
	unsigned char c2green = qGreen((QRgb)c2);
	unsigned char c2blue = qBlue((QRgb)c2);
	return (unsigned int)qRgb(
		lrintf(c2red * t + c1red * (1.0f - t)),
		lrintf(c2green * t + c1green * (1.0f - t)),
		lrintf(c2blue * t + c1blue * (1.0f - t)));
}

bool
XPlot::isPtIncluded(const XGraph::GPoint &pt)
{
  return (pt.x >= 0) && (pt.x <= 1) &&
    (pt.y >= 0) && (pt.y <= 1) &&
    (pt.z >= 0) && (pt.z <= 1);
}

bool
XPlot::clipLine(const tCanvasPoint &c1, const tCanvasPoint &c2,
  	XGraph::ScrPoint *s1, XGraph::ScrPoint *s2, bool colorplot, unsigned int *color1, unsigned int *color2)
{
  if(c1.insidecube && c2.insidecube) {
  	*s1 = c1.scr; *s2 = c2.scr;
	if(colorplot) {
		*color1 = c1.color;
		*color2 = c2.color;
	}
	return true;
  }
  XGraph::GPoint g1 = c1.graph;
  XGraph::GPoint g2 = c2.graph;
  
  XGraph::GFloat idx = 1.0 / (g1.x - g2.x);
  XGraph::GFloat tx0 = -g2.x * idx;
  XGraph::GFloat tx1 = (1.0 - g2.x) * idx;
  XGraph::GFloat txmin = min(tx0, tx1);
  XGraph::GFloat txmax = max(tx0, tx1);
  
  XGraph::GFloat idy = 1.0 / (g1.y - g2.y);
  XGraph::GFloat ty0 = -g2.y * idy;
  XGraph::GFloat ty1 = (1.0 - g2.y) * idy;
  XGraph::GFloat tymin = min(ty0, ty1);
  XGraph::GFloat tymax = max(ty0, ty1);
  
  XGraph::GFloat tmin = max(txmin, tymin);
  XGraph::GFloat tmax = min(txmax, tymax);
  
  if(m_curAxisZ) {
	if(tmin >= tmax) return false;
	XGraph::GFloat idz = 1.0 / (g1.z - g2.z);
	XGraph::GFloat tz0 = -g2.z * idz;
	XGraph::GFloat tz1 = (1.0 - g2.z) * idz;
	XGraph::GFloat tzmin = min(tz0, tz1);
	XGraph::GFloat tzmax = max(tz0, tz1);
	tmin = max(tmin, tzmin);
	tmax = min(tmax, tzmax);
  }  

  if(tmin >= tmax) return false;
  if(tmin >= 1.0) return false;
  if(tmax <= 0.0) return false;
   
   if(tmin > 0.0) {
   	graphToScreenFast(XGraph::GPoint(
		tmin*g1.x + (1.0 - tmin) *g2.x,
		tmin*g1.y + (1.0 - tmin) *g2.y,
		tmin*g1.z + (1.0 - tmin) *g2.z)
		, s2);
	if(colorplot) *color2 = blendColor(c2.color, c1.color, tmin);
   }
   else {
   	*s2 = c2.scr;
	if(colorplot) *color2 = c2.color;
   }
   if(tmax < 1.0) {
   	graphToScreenFast(XGraph::GPoint(
		tmax*g1.x + (1.0 - tmax) *g2.x,
		tmax*g1.y + (1.0 - tmax) *g2.y,
		tmax*g1.z + (1.0 - tmax) *g2.z)
		, s1);
	if(colorplot) *color1 = blendColor(c2.color, c1.color, tmax);
   }
   else {
   	*s1 = c1.scr;
	if(colorplot) *color1 = c1.color;
   }
  return true;
}
void
XPlot::snapshot()
{
  unsigned int cnt = std::min((unsigned int)*maxCount(), count());
  m_cntSnapped = cnt;
  m_ptsSnapped.resize(cnt);
  m_canvasPtsSnapped.resize(cnt);
  for(unsigned int i = 0; i < m_cntSnapped; i++)
    {
      m_ptsSnapped[i] = points(i);
    }
}

int
XPlot::validateAutoScale()
{
  for(unsigned int i = 0; i < snappedCount(); i++)
    {
    XGraph::ValPoint pt = snappedPoints(i);
    	validateAutoScaleOnePoint(pt);
    }
 return 0;
}
void
XPlot::validateAutoScaleOnePoint(const XGraph::ValPoint &pt)
{
bool included = true;
	included &= *m_curAxisX->autoScale() | m_curAxisX->isIncluded(pt.x);
	included &= *m_curAxisY->autoScale() | m_curAxisY->isIncluded(pt.y);
	if(m_curAxisZ)
		included &= *m_curAxisZ->autoScale() | m_curAxisZ->isIncluded(pt.z);
	if(included) {
		m_curAxisX->tryInclude(pt.x);
		m_curAxisY->tryInclude(pt.y);
		if(m_curAxisZ)
			m_curAxisZ->tryInclude(pt.z);
	}
}

XGraph::ValPoint
XXYPlot::points(unsigned int index) const
{
  XGraph::ValPoint point;
//  pts_mutex.ReadLock();
  point = m_points[index];
//  pts_mutex.ReadUnlock();
  return point;
}

int
XXYPlot::setMaxCount(unsigned int )
{
  clearAllPoints();
  return 0;
}

int
XXYPlot::clearAllPoints()
{
  //! this may cause exception, in rare cases
  shared_ptr<XGraph> graph(m_graph);
  
  graph->suspendUpdate();
  m_points.clear();
  graph->requestUpdate();
  graph->resumeUpdate();
  return 0;
}

int
XXYPlot::addPoint(XGraph::VFloat x, XGraph::VFloat y, XGraph::VFloat z)
{
  XGraph::ValPoint npt(x, y, z);

  //! this may cause exception, in rare cases
  shared_ptr<XGraph> graph(m_graph);
  
  graph->suspendUpdate();
  
  if(count() >= *maxCount())
    {
      m_points.pop_front();
    }
  m_points.push_back(npt);
    
  graph->requestUpdate();
  graph->resumeUpdate();
  return 0;
}

XAxis::XAxis(const char *name, bool runtime,
     AxisDirection dir, bool rightOrTop, const shared_ptr<XGraph> &graph) : 
 XNode(name, runtime),
  m_direction(dir),
  m_dirVector( (dir == DirAxisX) ? 1.0 : 0.0, 
    (dir == DirAxisY) ? 1.0 : 0.0, (dir == DirAxisZ) ? 1.0 : 0.0),
  m_graph(graph),
  m_label(create<XStringNode>("Label", true)),
  m_x(create<XDoubleNode>("X", true)),
  m_y(create<XDoubleNode>("Y", true)),
  m_z(create<XDoubleNode>("Z", true)),
  m_length(create<XDoubleNode>("Length", true)), // in screen coordinate
  m_majorTicScale(create<XDoubleNode>("MajorTicScale", true)),
  m_minorTicScale(create<XDoubleNode>("MinorTicScale", true)),
  m_displayMajorTics(create<XBoolNode>("DisplayMajorTics", true)),
  m_displayMinorTics(create<XBoolNode>("DisplayMinorTics", true)),
  m_max(create<XDoubleNode>("Max", true)),
  m_min(create<XDoubleNode>("Min", true)),
  m_rightOrTopSided(create<XBoolNode>("RightOrTopSided", true)), //sit on right, top
  m_ticLabelFormat(create<XStringNode>("TicLabelFormat", true)),
  m_displayLabel(create<XBoolNode>("DisplayLabel", true)),
  m_displayTicLabels(create<XBoolNode>("DisplayTicLabels", true)),
  m_ticColor(create<XHexNode>("TicColor", true)),
  m_labelColor(create<XHexNode>("LabelColor", true)),
  m_ticLabelColor(create<XHexNode>("TicLabelColor", true)),
  m_autoFreq(create<XBoolNode>("AutoFreq", true)),
  m_autoScale(create<XBoolNode>("AutoScale", true)),
  m_logScale(create<XBoolNode>("LogScale", true))
{
  m_ticLabelFormat->setValidator(&formatDoubleValidator);
  
  x()->value(0.15);
  y()->value(0.15);
  z()->value(0.15);
  length()->value(0.7);
  maxValue()->value(0);
  minValue()->value(0);
  ticLabelFormat()->value("");
  logScale()->value(false);
  displayLabel()->value(true);
  displayTicLabels()->value(true);
  displayMajorTics()->value(true);
  displayMinorTics()->value(true);
  autoFreq()->value(true);
  autoScale()->value(true);
  rightOrTopSided()->value(rightOrTop);
  majorTicScale()->value(10);
  minorTicScale()->value(1);
  labelColor()->value(clBlack);
  ticColor()->value(clBlack);
  ticLabelColor()->value(clBlack);

  if(rightOrTop) {
      if(dir == DirAxisY) x()->value(1.0- *x());
      if(dir == DirAxisX) y()->value(1.0- *y());
  }
  _startAutoscale(true);

  maxValue()->onValueChanged().connect(graph->lsnPropertyChanged());
  minValue()->onValueChanged().connect(graph->lsnPropertyChanged());
  label()->onValueChanged().connect(graph->lsnPropertyChanged());
  logScale()->onValueChanged().connect(graph->lsnPropertyChanged());
  autoScale()->onValueChanged().connect(graph->lsnPropertyChanged());
  x()->onValueChanged().connect(graph->lsnPropertyChanged());
  y()->onValueChanged().connect(graph->lsnPropertyChanged());
  z()->onValueChanged().connect(graph->lsnPropertyChanged());
  length()->onValueChanged().connect(graph->lsnPropertyChanged());
  ticLabelFormat()->onValueChanged().connect(graph->lsnPropertyChanged());
  displayLabel()->onValueChanged().connect(graph->lsnPropertyChanged());
  displayTicLabels()->onValueChanged().connect(graph->lsnPropertyChanged());
  displayMajorTics()->onValueChanged().connect(graph->lsnPropertyChanged());
  displayMinorTics()->onValueChanged().connect(graph->lsnPropertyChanged());
  autoFreq()->onValueChanged().connect(graph->lsnPropertyChanged());
  rightOrTopSided()->onValueChanged().connect(graph->lsnPropertyChanged());
  majorTicScale()->onValueChanged().connect(graph->lsnPropertyChanged());
  minorTicScale()->onValueChanged().connect(graph->lsnPropertyChanged());
  labelColor()->onValueChanged().connect(graph->lsnPropertyChanged());
  ticColor()->onValueChanged().connect(graph->lsnPropertyChanged());
  ticLabelColor()->onValueChanged().connect(graph->lsnPropertyChanged());
}


void
XAxis::_startAutoscale(bool clearscale)
{
  m_bLogscaleFixed = *logScale();
  m_bAutoscaleFixed = *autoScale();
  if(clearscale) {
	m_minFixed = XGraph::VFLOAT_MAX;
	m_maxFixed = m_bLogscaleFixed ? 0 : - XGraph::VFLOAT_MAX;
  }
  else {
	m_minFixed = m_bLogscaleFixed ? max((XGraph::VFloat)*minValue(), (XGraph::VFloat)0.0) : 
        (XGraph::VFloat)*minValue();
	m_maxFixed = *maxValue();
  }
  m_invMaxMinusMinFixed = -1; //undef
  m_invLogMaxOverMinFixed = -1; //undef
}
void
XAxis::startAutoscale(float, bool clearscale)
{
	_startAutoscale(clearscale);
}
void
XAxis::fixScale(float resolution, bool suppressupdate)
{
	if(suppressupdate) {
		m_graph.lock()->lsnPropertyChanged()->mask();
	}
	if(m_minFixed == m_maxFixed) {
	XGraph::VFloat x = m_minFixed;
		m_maxFixed = x * 1.01 + 0.01;
		m_minFixed = x * 0.99 - 0.01;
	}
	XGraph::VFloat min_tmp = m_bLogscaleFixed ? 
        max((XGraph::VFloat)*minValue(), (XGraph::VFloat)0.0) : (XGraph::VFloat)*minValue();
	if(m_minFixed != min_tmp) {
        minValue()->setFormat(ticLabelFormat()->to_str().utf8());
		minValue()->value(m_minFixed);
    }
	if(m_maxFixed != *maxValue()) {
        maxValue()->setFormat(ticLabelFormat()->to_str().utf8());
		maxValue()->value(m_maxFixed);
    }
	if(suppressupdate) {
        m_graph.lock()->lsnPropertyChanged()->unmask();
	}
	autoFreq(resolution);
}
void
XAxis::autoFreq(float resolution)
{
  if(*autoFreq() &&
  	(!m_bLogscaleFixed || (m_minFixed >= 0)) &&
	(m_minFixed < m_maxFixed))
    {
      float fac = max(0.7f, log10f(1.0 / resolution / 500.0) );
      m_majorFixed = (VFLOAT_POW((XGraph::VFloat)10.0,
             VFLOAT_RINT(VFLOAT_LOG10(m_maxFixed - m_minFixed) - fac)));
      m_minorFixed = m_majorFixed / (XGraph::VFloat)2.0;
    }
   else {
	m_majorFixed = *majorTicScale();
	m_minorFixed = *minorTicScale();
   }
}

bool
XAxis::isIncluded(XGraph::VFloat x)
{
  return (x >= m_minFixed) && (x <= m_maxFixed);
}
void
XAxis::tryInclude(XGraph::VFloat x)
{
  //omitts negative values in log scaling
  if(!(m_bLogscaleFixed && (x <= 0)))
    {
    	if(m_bAutoscaleFixed)
	{
	      if(x > m_maxFixed)
	      {
        		m_maxFixed = x;
        		m_invMaxMinusMinFixed = -1; //undef
      		m_invLogMaxOverMinFixed = -1; //undef
	      }			
	      if(x < m_minFixed)
	      {
	           m_minFixed = x;
                m_invMaxMinusMinFixed = -1; //undef
                m_invLogMaxOverMinFixed = -1; //undef
	      }
	}
    }
}

void
XAxis::zoom(bool minchange, bool maxchange, XGraph::GFloat prop, XGraph::GFloat center)
{
  if(m_minFixed == m_maxFixed)
    {
      m_minFixed = (1 - 1e-15) * m_minFixed - (m_bLogscaleFixed ? 0 : 1e-50);
      m_maxFixed = (1 + 1e-15) * m_maxFixed + (m_bLogscaleFixed ? 0 : 1e-50);
    }
  if(maxchange) {
  	m_maxFixed = axisToVal(center + (XGraph::GFloat)0.5 / prop);
  }
  if(minchange) {
  	m_minFixed = axisToVal(center - (XGraph::GFloat)0.5 / prop);
  }
  m_invMaxMinusMinFixed = -1; //undef
  m_invLogMaxOverMinFixed = -1; //undef
}

XGraph::GFloat
XAxis::valToAxis(XGraph::VFloat x)
{
  XGraph::GFloat pos;
  if(m_bLogscaleFixed)
    {
      if ((x <= 0) || (m_minFixed <= 0) || (m_maxFixed <= m_minFixed))
	return - XGraph::GFLOAT_MAX;
      if(m_invLogMaxOverMinFixed < 0)
          	m_invLogMaxOverMinFixed = 1 / VFLOAT_LOG(m_maxFixed / m_minFixed);	
      pos = VFLOAT_LOG(x / m_minFixed) * m_invLogMaxOverMinFixed;
    }
  else
    {
      if(m_maxFixed <= m_minFixed) return -1;
      if(m_invMaxMinusMinFixed < 0)
         m_invMaxMinusMinFixed = 1 / (m_maxFixed - m_minFixed);
      pos = (x - m_minFixed) * m_invMaxMinusMinFixed;
    }
  return pos;
}

XGraph::VFloat
XAxis::axisToVal(XGraph::GFloat pos, XGraph::GFloat axis_prec)
{
  XGraph::VFloat x = 0;
  if(axis_prec <= 0)
    {
      if(m_bLogscaleFixed)
	{
	  if((m_minFixed <= 0) || (m_maxFixed <= m_minFixed)) return 0;
	  x = m_minFixed * VFLOAT_EXP(VFLOAT_LOG(m_maxFixed / m_minFixed) * pos);
	}
      else
	{
	  if(m_maxFixed <= m_minFixed) return 0;
	  x = m_minFixed + pos *(m_maxFixed - m_minFixed);
	}
      return x;
    }
  else
    {
      x = axisToVal(pos);
      XGraph::VFloat dx = axisToVal(pos + axis_prec) - x;
      return setprec(x, dx);
    }
}

void
XAxis::axisToScreen(XGraph::GFloat pos, XGraph::ScrPoint *scr)
{
  XGraph::SFloat len = *length();
  pos *= len;
  scr->x = *x() + ((m_direction == DirAxisX) ? pos: (XGraph::SFloat) 0.0);
  scr->y = *y() + ((m_direction == DirAxisY) ? pos: (XGraph::SFloat) 0.0);
  scr->z = *z() + ((m_direction == DirAxisZ) ? pos: (XGraph::SFloat) 0.0);
}
XGraph::GFloat
XAxis::screenToAxis(const XGraph::ScrPoint &scr)
{
  XGraph::SFloat _x = scr.x - *x();
  XGraph::SFloat _y = scr.y - *y();
  XGraph::SFloat _z = scr.z - *z();
  XGraph::GFloat pos = ((m_direction == DirAxisX) ? _x : 
    ((m_direction == DirAxisY) ? _y : _z)) / (XGraph::SFloat)*length();
  return pos;
}
XGraph::VFloat
XAxis::screenToVal(const XGraph::ScrPoint &scr)
{
	return axisToVal(screenToAxis(scr));
}
void
XAxis::valToScreen(XGraph::VFloat val, XGraph::ScrPoint *scr)
{
	axisToScreen(valToAxis(val), scr);
}
QString
XAxis::valToString(XGraph::VFloat val)
{
    return formatDouble(ticLabelFormat()->to_str().utf8(), val);
}

XAxis::Tic
XAxis::queryTic(int len, int pos, XGraph::VFloat *ticnum)
{
  XGraph::VFloat x, t;
  if(m_bLogscaleFixed)
    {
      x = axisToVal((XGraph::GFloat)pos / len);
      if(x <= 0) return NoTics;
      t = VFLOAT_POW((XGraph::VFloat)10.0, VFLOAT_RINT(VFLOAT_LOG10(x)));
      if(GFLOAT_LRINT(valToAxis(t) * len) == pos)
	{
	  *ticnum = t;
	  return MajorTic;
	}
      x = x / t;
      if(x < 1)
	t = VFLOAT_RINT(x / (XGraph::VFloat)0.1) * (XGraph::VFloat)0.1 * t;
      else
	t = VFLOAT_RINT(x) * t;
      if(GFLOAT_LRINT(valToAxis(t) * len) == pos)
	{
	  *ticnum = t;
	  return MinorTic;
	}
      return NoTics;
    }
  else
    {
      x = axisToVal((XGraph::GFloat)pos / len);
      t = VFLOAT_RINT(x / m_majorFixed) * m_majorFixed;
      if(GFLOAT_LRINT(valToAxis(t) * len) == pos)
	{
	  *ticnum = t;
	  return MajorTic;
	}
      t = VFLOAT_RINT(x / m_minorFixed) * m_minorFixed;
      if(GFLOAT_LRINT(valToAxis(t) * len) == pos)
	{
	  *ticnum = t;
	  return MinorTic;
	}
      return NoTics;
    }
}


void
XAxis::drawLabel(XQGraphPainter *painter)
{
const int sizehint = 2;
      painter->setColor(*labelColor());
XGraph::ScrPoint s1, s2, s3;
	axisToScreen(0.5, &s1);
	s2 = s1;
	axisToScreen(1.5, &s3);
	s3 -= s2;
	painter->posOffAxis(m_dirVector, &s1, AxisToLabel);
	s2 -= s1;
	s2 *= -1;
	if(!painter->selectFont(*label(), s1, s2, s3, sizehint)) {
		painter->drawText(s1, *label());
		return;
	}
	
	axisToScreen(1.02, &s1);
	axisToScreen(1.05, &s2);
	s2 -= s1;
	s3 = s1;
	painter->posOffAxis(m_dirVector, &s3, 0.5);
	s3 -= s1;
	if(!painter->selectFont(*label(), s1, s2, s3, sizehint)) {
		painter->drawText(s1, *label());
		return;
	}
	
	axisToScreen(0.5, &s1);
	s2 = s1;
	axisToScreen(1.5, &s3);
	s3 -= s2;
	painter->posOffAxis(m_dirVector, &s1, -AxisToLabel);
	s2 -= s1;
	s2 *= -1;
	if(!painter->selectFont(*label(), s1, s2, s3, sizehint)) {
		painter->drawText(s1, *label());
		return;
	}
}


int
XAxis::drawAxis(XQGraphPainter *painter)
{
  XGraph::SFloat LenMajorTicL = 0.01;
  XGraph::SFloat LenMinorTicL = 0.005;

  painter->setColor(*ticColor());

  XGraph::ScrPoint s1, s2;
  axisToScreen(0.0, &s1);
  axisToScreen(1.0, &s2);

  painter->beginLine();
  painter->setVertex(s1);
  painter->setVertex(s2);
  painter->endLine();
  
  if(*displayLabel())
    {
	drawLabel(painter);
    	
    }
  if(m_bLogscaleFixed && (m_minFixed < 0)) return -1;
  if(m_maxFixed <= m_minFixed) return -1;
  
  int len = SFLOAT_LRINT(*length() / painter->resScreen());
  painter->defaultFont();
  XGraph::GFloat mindx = 2, lastg = -1;
  //!dry-run to determine font
  for(int i = 0; i < len; i++) {
    XGraph::VFloat z;
    XGraph::GFloat x = (XGraph::GFloat)i / len;
      if(queryTic(len, i, &z) == MajorTic)
	{
		if(mindx > x - lastg) {
			axisToScreen(x, &s1);
			s2 = s1;
			XGraph::ScrPoint s3;
			axisToScreen(lastg, &s3);
			s3 -= s2;
			s3 *= 0.7;
			painter->posOffAxis(m_dirVector, &s1, AxisToTicLabel);
			s2 -= s1;
			s2 *= -1;
			
			double var = setprec(z, m_bLogscaleFixed ? (XGraph::VFloat)z :  m_minorFixed);
			painter->selectFont(valToString(var), s1, s2, s3, 0);
			
			mindx = x - lastg;
		}
		lastg = x;
	}
  }
  
  for(int i = 0; i < len; i++) {
    XGraph::VFloat z;
    XGraph::GFloat x = (XGraph::GFloat)i / len;
      switch(queryTic(len, i, &z))
	{
	case MajorTic:
	  if(*displayMajorTics())
	    {
	      axisToScreen(x, &s1);
	      painter->posOffAxis(m_dirVector, &s1, LenMajorTicL);
	      axisToScreen(x, &s2);
	      painter->posOffAxis(m_dirVector, &s2, -LenMajorTicL);
	      painter->setColor(*ticColor());
	      painter->beginLine(1.0);
	      painter->setVertex(s1);
	      painter->setVertex(s2);
	      painter->endLine();
	    }
	  if(*displayTicLabels())
	    {
	      axisToScreen(x, &s1);
	      painter->posOffAxis(m_dirVector, &s1, AxisToTicLabel);
	      double var = setprec(z, m_bLogscaleFixed ? (XGraph::VFloat)z : m_minorFixed);
	      painter->drawText(s1, valToString(var));
	    }
	  break;
	case MinorTic:
	  if(*displayMinorTics())
	    {
	      axisToScreen(x, &s1);
	      painter->posOffAxis(m_dirVector, &s1, LenMinorTicL);
	      axisToScreen(x, &s2);
	      painter->posOffAxis(m_dirVector, &s2, -LenMinorTicL);
	      painter->setColor(*ticColor());
		painter->beginLine(1.0);
		painter->setVertex(s1);
		painter->setVertex(s2);
		painter->endLine();
	    }
	  break;
	case NoTics:
	  break;
	}
    }
  return 0;
}

XFuncPlot::XFuncPlot(const char *name, bool runtime, const shared_ptr<XGraph> &graph)
  : XPlot(name, runtime, graph)
{
  maxCount()->value(300);
}

int
XFuncPlot::setMaxCount(unsigned int cnt)
{
  m_count = cnt;
  return 0;
}
XGraph::ValPoint
XFuncPlot::snappedPoints(unsigned int index) const
{
  XGraph::ValPoint pt;
  pt.x = m_curAxisX->axisToVal((XGraph::GFloat)index / count());
  pt.y = func(pt.x);
  pt.z = 0.0;
  return pt;
}
unsigned int
XFuncPlot::snappedCount() const
{
  return m_count;
}
XGraph::ValPoint
XFuncPlot::points(unsigned int index) const
{
  return snappedPoints(index);
}
unsigned int
XFuncPlot::count() const
{
  return snappedCount();
}

