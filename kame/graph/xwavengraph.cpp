#include "xwavengraph.h"

#include "forms/graphnurlform.h"
#include "graphwidget.h"
#include "graph.h"

#include <kurlrequester.h>
#include <qpushbutton.h>
#include <kiconloader.h>
#include <kapplication.h>
#include <qstatusbar.h>
#include <klocale.h>

#define OFSMODE std::ios::out | std::ios::app | std::ios::ate

//---------------------------------------------------------------------------

XWaveNGraph::XWaveNGraph(const char *name, bool runtime, FrmGraphNURL *item)
  : XNode(name, runtime),
  m_btnDump(item->m_btnDump),
  m_graph(create<XGraph>(name, false)),
  m_dump(create<XNode>("Dump", true)),
  m_filename(create<XStringNode>("FileName", true))
{
  item->m_graphwidget->setGraph(m_graph);
  item->statusBar()->hide();
  m_conFilename = xqcon_create<XKURLReqConnector>(m_filename, item->m_url,
				      "*.dat|Data files (*.dat)\n*.*|All files (*.*)", true);
  m_conDump = xqcon_create<XQButtonConnector>(m_dump, item->m_btnDump);
  init();
}
XWaveNGraph::XWaveNGraph(const char *name, bool runtime, 
        XQGraph *graphwidget, KURLRequester *urlreq, QPushButton *btndump)
  : XNode(name, runtime),
  m_btnDump(btndump),
  m_graph(create<XGraph>(name, false)),
  m_dump(create<XNode>("Dump", true)),
  m_filename(create<XStringNode>("FileName", true))
{
  graphwidget->setGraph(m_graph);
  m_conFilename = xqcon_create<XKURLReqConnector>(m_filename, urlreq,
                      "*.dat|Data files (*.dat)\n*.*|All files (*.*)", true);
  m_conDump = xqcon_create<XQButtonConnector>(m_dump, btndump);
  init();
}
void
XWaveNGraph::init()
{
  m_btnDump->setIconSet( KApplication::kApplication()->iconLoader()->loadIconSet("filesave", 
            KIcon::Toolbar, KIcon::SizeSmall, true ) );
  m_lsnOnFilenameChanged = filename()->onValueChanged().connectWeak(
    false, shared_from_this(), &XWaveNGraph::onFilenameChanged);
    
  dump()->setUIEnabled(false);
}
XWaveNGraph::~XWaveNGraph()
{
  if(m_plot1) m_graph->plots()->releaseChild(m_plot1);
  if(m_plot2) m_graph->plots()->releaseChild(m_plot2);
  if(m_axisy2) m_graph->axes()->releaseChild(m_axisy2);
  m_stream.close();
}

void
XWaveNGraph::selectAxes(int x, int y1, int y2, int yweight)
{
  XScopedWriteLock<XRecursiveRWLock> lock(m_mutex);
  {  
      m_colx = x;
      m_coly1 = y1;
      m_coly2 = y2;
      m_colyweight = yweight;
    
      XScopedLock<XGraph> lock(*m_graph);
      if(m_plot1) m_graph->plots()->releaseChild(m_plot1);
      if(m_plot2) m_graph->plots()->releaseChild(m_plot2);
      if(m_axisy2) m_graph->axes()->releaseChild(m_axisy2);
      m_plot1.reset();
      m_plot2.reset();
      m_axisy2.reset();
        
     // m_graph->setName(getName());
      m_graph->label()->value(getLabel());
    
      m_plot1 = m_graph->plots()->create<XXYPlot>("Plot1", true, m_graph);
      m_plot1->label()->value(KAME::i18n("Plot1"));
    
      atomic_shared_ptr<const XNode::NodeList> axes_list(m_graph->axes()->children());
      shared_ptr<XAxis> axisx = dynamic_pointer_cast<XAxis>(axes_list->at(0));
      shared_ptr<XAxis> axisy = dynamic_pointer_cast<XAxis>(axes_list->at(1));
    
      m_plot1->axisX()->value(axisx);
      m_plot1->axisY()->value(axisy);
      m_plot1->maxCount()->value(rowCount());
      m_plot1->maxCount()->setUIEnabled(false);
      m_plot1->clearPoints()->setUIEnabled(false);
      if(m_coly2 >= 0)
        {
          m_axisy2 = m_graph->axes()->create<XAxis>("Y2 Axis", true, XAxis::DirAxisY, true, m_graph);
          m_plot2 = m_graph->plots()->create<XXYPlot>("Plot2", true, m_graph);
          m_plot2->label()->value(KAME::i18n("Plot2"));
          m_plot2->axisX()->value(axisx);
          m_plot2->axisY()->value(m_axisy2);
          m_plot2->pointColor()->value(clGreen);
          m_plot2->lineColor()->value(clGreen);
          m_plot2->barColor()->value(clGreen);
          m_plot2->displayMajorGrid()->value(false);
          m_plot2->maxCount()->value(rowCount());
          m_plot2->maxCount()->setUIEnabled(false);
          m_plot2->clearPoints()->setUIEnabled(false);
        }
  }
}

void
XWaveNGraph::setColCount(unsigned int n, const char **labels)
{
  XScopedWriteLock<XRecursiveRWLock> lock(m_mutex);
  m_colcnt = n;
  m_labels.resize(m_colcnt);
  for(unsigned int i = 0; i < n; i++) {
    m_labels[i] = labels[i];
  }
}
void
XWaveNGraph::setLabel(unsigned int col, const char *label)
{
    m_labels[col] = label;
}
void
XWaveNGraph::setRowCount(unsigned int n) {
  XScopedWriteLock<XRecursiveRWLock> lock(m_mutex);
  m_cols.resize(m_colcnt * n);
  if(m_plot1) m_plot1->maxCount()->value(n);
  if(m_plot2) m_plot2->maxCount()->value(n);
}

void
XWaveNGraph::readLock() const
{
    m_mutex.readLock();
}
void
XWaveNGraph::readUnlock() const
{
    m_mutex.readUnlock();
}
void
XWaveNGraph::writeLock()
{
  m_mutex.writeLock();
  m_graph->lock();
}

void
XWaveNGraph::writeUnlock(bool updategraph)
{
  if(updategraph) {
      m_mutex.writeUnlockNReadLock();
      drawGraph();
      m_graph->unlock();
      m_mutex.readUnlock();
    }
  else {
      m_graph->unlock();
      m_mutex.writeUnlock();
  }
}

int
XWaveNGraph::colX() const {return m_colx;}
int
XWaveNGraph::colY1() const {return m_coly1;}
int
XWaveNGraph::colY2() const {return m_coly2;}
int
XWaveNGraph::colYWeight() const {return m_colyweight;}
void
XWaveNGraph::onFilenameChanged(const shared_ptr<XValueNodeBase> &)
{
  m_btnDump->setIconSet( KApplication::kApplication()->iconLoader()->loadIconSet("filesave", 
            KIcon::Toolbar, KIcon::SizeSmall, true ) );

   {   XScopedLock<XMutex> lock(m_filemutex);
      
      if(m_stream.is_open()) m_stream.close();
      m_stream.clear();
      m_stream.open((const char*)QString(filename()->to_str()).local8Bit(), OFSMODE);
    
      if(m_stream.good()) {
          m_lsnOnDumpTouched = dump()->onTouch().connectWeak(
            false, shared_from_this(), &XWaveNGraph::onDumpTouched);
          dump()->setUIEnabled(true);
      }
      else {
          m_lsnOnDumpTouched.reset();
          dump()->setUIEnabled(false);      
      }
  }
}
void
XWaveNGraph::onDumpTouched(const shared_ptr<XNode> &)
{
  XScopedLock<XMutex> filelock(m_filemutex);
  if(m_stream.good()) {
    XScopedReadLock<XRecursiveRWLock> lock(m_mutex);
	
	m_stream << "#dumping...  "
		<< (XTime::now()).getTimeFmtStr("%Y/%m/%d %H:%M:%S")
		<< std::endl;
	m_stream << "#";
	for(unsigned int i = 0; i < colCount(); i++)
	{
        	m_stream << m_labels[i] << " ";
	}
	m_stream << std::endl;
	
	for(unsigned int i = 0; i < rowCount(); i++)
	{
        	if(colYWeight() >= 0)
        	if(cols(colYWeight())[i] < 0.1) continue;
        	for(unsigned int j = 0; j < colCount(); j++)
        	{
        		m_stream << cols(j)[i] << " ";
        	}
        	m_stream << std::endl;
	}
	m_stream << std::endl;
    
	m_stream.flush();

    m_btnDump->setIconSet( KApplication::kApplication()->iconLoader()->loadIconSet("redo", 
            KIcon::Toolbar, KIcon::SizeSmall, true ) );
  }
}
void
XWaveNGraph::clear()
{
  XScopedWriteLock<XWaveNGraph> lock(*this);
  setRowCount(0);
  { XScopedLock<XGraph> lock(*m_graph);
      if(m_plot1) m_plot1->clearAllPoints();
      if(m_plot2) m_plot2->clearAllPoints();
      m_graph->requestUpdate();
  }
}
void
XWaveNGraph::drawGraph()
{
  XScopedReadLock<XRecursiveRWLock> lock(m_mutex);
  {

      ASSERT(m_plot1);
        
      XScopedLock<XGraph> lock(*m_graph);
      
      atomic_shared_ptr<const XNode::NodeList> axes_list(m_graph->axes()->children());
      shared_ptr<XAxis> axisx = dynamic_pointer_cast<XAxis>(axes_list->at(0));
      shared_ptr<XAxis> axisy = dynamic_pointer_cast<XAxis>(axes_list->at(1));
    
      axisx->label()->value(m_labels[m_colx]);
      axisy->label()->value(m_labels[m_coly1]);
      if(colY2() > 0)
        m_axisy2->label()->value(m_labels[m_coly2]);
        
      int rowcnt = rowCount();
      double *colx = cols(colX());
      double *coly1 = cols(colY1());
      double *coly2 = (colY2() > 0) ? cols(colY2()) : NULL;
      double *colyweight = (colYWeight() >= 0) ? cols(colYWeight()) : NULL;
    
      std::deque<XGraph::ValPoint> &points_plot1(m_plot1->points());
      points_plot1.clear();
      for(int i = 0; i < rowcnt; i++)
        {
          if(colyweight)
    	       if(colyweight[i] <= 0) continue;
           points_plot1.push_back( XGraph::ValPoint(colx[i], coly1[i]) );
        }
      if(coly2)
        {
          std::deque<XGraph::ValPoint> &points_plot2 = m_plot2->points();
          points_plot2.clear();
          for(int i = 0; i < rowcnt; i++)
        	  {
    	    if(colyweight)
    	       if(colyweight[i] <= 0) continue;
            points_plot2.push_back( XGraph::ValPoint(colx[i], coly2[i]) );
        	  }
        }
      m_graph->requestUpdate();
  }
}

unsigned int
XWaveNGraph::rowCount() const {return m_colcnt ? m_cols.size() / m_colcnt : 0;}
unsigned int
XWaveNGraph::colCount() const {return m_colcnt;}
double *
XWaveNGraph::cols(unsigned int n) {return &(m_cols[rowCount() * n]);}
