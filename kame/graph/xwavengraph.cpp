/***************************************************************************
		Copyright (C) 2002-2009 Kentaro Kitagawa
		                   kitag@issp.u-tokyo.ac.jp
		
		This program is free software; you can redistribute it and/or
		modify it under the terms of the GNU Library General Public
		License as published by the Free Software Foundation; either
		version 2 of the License, or (at your option) any later version.
		
		You should have received a copy of the GNU Library General 
		Public License and a list of authors along with this program; 
		see the files COPYING and AUTHORS.
 ***************************************************************************/
#include "xwavengraph.h"

#include "ui_graphnurlform.h"
#include "graphwidget.h"
#include "graph.h"

#include <kurlrequester.h>
#include <qpushbutton.h>
#include <kiconloader.h>
#include <kapplication.h>
#include <qstatusbar.h>

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
  m_conFilename = xqcon_create<XKURLReqConnector>(m_filename, item->m_url,
                      "*.dat|Data files (*.dat)\n*.*|All files (*.*)", true);
  m_conDump = xqcon_create<XQButtonConnector>(m_dump, item->m_btnDump);
  init();
}
XWaveNGraph::XWaveNGraph(const char *name, bool runtime, 
        XQGraph *graphwidget, KUrlRequester *urlreq, QPushButton *btndump)
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
  m_lsnOnFilenameChanged = filename()->onValueChanged().connectWeak(
    shared_from_this(), &XWaveNGraph::onFilenameChanged);
  m_lsnOnIconChanged = m_tlkOnIconChanged.connectWeak(
    shared_from_this(), &XWaveNGraph::onIconChanged,
    XListener::FLAG_MAIN_THREAD_CALL || XListener::FLAG_AVOID_DUP);
    
  m_tlkOnIconChanged.talk(false);
  dump()->setUIEnabled(false);
  
  m_graph->persistence()->value(0.0);
  clearPlots();
}
XWaveNGraph::~XWaveNGraph()
{
	m_stream.close();
}

void
XWaveNGraph::clearPlots()
{
  XScopedWriteLock<XRecursiveRWLock> lock(m_mutex); {
	  XScopedLock<XGraph> lock(*m_graph);
		for(std::deque<Plot>::iterator it = m_plots.begin(); it != m_plots.end(); it++) {
			m_graph->plots()->releaseChild(it->xyplot);
		}
	  if(m_axisw) m_graph->axes()->releaseChild(m_axisw);
	  if(m_axisz) m_graph->axes()->releaseChild(m_axisz);
	  if(m_axisy2) m_graph->axes()->releaseChild(m_axisy2);
      if(m_axisx) m_axisx->label()->value("");
      if(m_axisy) m_axisy->label()->value("");
	  m_plots.clear();
	  m_axisy2.reset();
	  m_axisz.reset();
	  m_axisw.reset();
	  m_colw = -1;
  }
}
void
XWaveNGraph::insertPlot(const XString &label, int x, int y1, int y2, int weight, int z)
{
	ASSERT( (y1 < 0) || (y2 < 0) );
  XScopedWriteLock<XRecursiveRWLock> lock(m_mutex);
  {  
  	Plot plot;
      plot.colx = x;
      plot.coly1 = y1;
      plot.coly2 = y2;
      plot.colweight = weight;
      plot.colz = z;
      
      if(weight >= 0) {
      	if((m_colw >= 0) && (m_colw != weight))
      		m_colw = -1;
      	else
      		m_colw = weight;
      }
    
      XScopedLock<XGraph> lock(*m_graph);
        
     // m_graph->setName(getName());
      m_graph->label()->value(getLabel());
    
      unsigned int plotnum = m_plots.size() + 1;
      plot.xyplot = m_graph->plots()->create<XXYPlot>(formatString("Plot%u", plotnum).c_str(),
      	 true, m_graph);
      plot.xyplot->label()->value(label);
    
      XNode::NodeList::reader axes_list(m_graph->axes()->children());
      m_axisx = dynamic_pointer_cast<XAxis>(axes_list->at(0));
      m_axisy = dynamic_pointer_cast<XAxis>(axes_list->at(1));
      plot.xyplot->axisX()->value(m_axisx);
      m_axisx->label()->value(m_labels[plot.colx]);
      if(plot.coly1 >= 0) {
	      plot.xyplot->axisY()->value(m_axisy);
	      m_axisy->label()->value(m_labels[plot.coly1]);
      }
      if(plot.colz >= 0) {
          if(!m_axisz)
          	m_axisz = m_graph->axes()->create<XAxis>("Z Axis", true, XAxis::DirAxisZ, true, m_graph);
          plot.xyplot->axisZ()->value(m_axisz);
	      m_axisz->label()->value(m_labels[plot.colz]);
      }
      if(plot.colweight >= 0) {
          if(!m_axisw) {
          	m_axisw = m_graph->axes()->create<XAxis>("Weight", true, XAxis::AxisWeight, true, m_graph);
          }
	      m_axisw->autoScale()->value(false);
	      m_axisw->autoScale()->setUIEnabled(false);
          plot.xyplot->axisW()->value(m_axisw);
	      m_axisw->label()->value(m_labels[plot.colweight]);
      }
      if(plot.coly2 >= 0)
        {
          if(!m_axisy2)
	          m_axisy2 = m_graph->axes()->create<XAxis>("Y2 Axis", true, XAxis::DirAxisY, true, m_graph);
          plot.xyplot->axisY()->value(m_axisy2);
	      m_axisy2->label()->value(m_labels[plot.coly2]);
        }
      plot.xyplot->maxCount()->value(rowCount());
      plot.xyplot->maxCount()->setUIEnabled(false);
      plot.xyplot->clearPoints()->setUIEnabled(false);
      plot.xyplot->intensity()->value(1.0);
      if(m_plots.size()) {
          plot.xyplot->pointColor()->value(clGreen);
          plot.xyplot->lineColor()->value(clGreen);
          plot.xyplot->barColor()->value(clGreen);
          plot.xyplot->displayMajorGrid()->value(false);
        }
     
     m_plots.push_back(plot);
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
	for(std::deque<Plot>::iterator it = m_plots.begin(); it != m_plots.end(); it++) {
		it->xyplot->maxCount()->value(n);
	}
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

void
XWaveNGraph::onIconChanged(const bool &v)
{
    KIconLoader *loader = KIconLoader::global();
	if(!v)
		m_btnDump->setIcon( loader->loadIcon("filesave",
	            KIconLoader::Toolbar, KIconLoader::SizeSmall, true ) );
	else
	    m_btnDump->setIcon( loader->loadIcon("redo",
	            KIconLoader::Toolbar, KIconLoader::SizeSmall, true ) );
}
void
XWaveNGraph::onFilenameChanged(const shared_ptr<XValueNodeBase> &)
{
   {   XScopedLock<XMutex> lock(m_filemutex);
      
      if(m_stream.is_open()) m_stream.close();
      m_stream.clear();
      m_stream.open((const char*)QString(filename()->to_str().c_str()).toLocal8Bit().data(), OFSMODE);
    
      if(m_stream.good()) {
          m_lsnOnDumpTouched = dump()->onTouch().connectWeak(
            shared_from_this(), &XWaveNGraph::onDumpTouched);
          dump()->setUIEnabled(true);
      }
      else {
          m_lsnOnDumpTouched.reset();
          dump()->setUIEnabled(false);      
      }
	  m_tlkOnIconChanged.talk(false);
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
    	if((m_colw < 0) || (cols(m_colw)[i] > 0)) {
            for(unsigned int j = 0; j < colCount(); j++)
            {
                m_stream << cols(j)[i] << " ";
            }
            m_stream << std::endl;
    	}
    }
    m_stream << std::endl;
    
    m_stream.flush();

	m_tlkOnIconChanged.talk(true);
  }
}
void
XWaveNGraph::clear()
{
  XScopedWriteLock<XWaveNGraph> lock(*this);
  setRowCount(0);
  { XScopedLock<XGraph> lock(*m_graph);
	for(std::deque<Plot>::iterator it = m_plots.begin(); it != m_plots.end(); it++) {
      it->xyplot->clearAllPoints();
	}
      m_graph->requestUpdate();
  }
}
void
XWaveNGraph::drawGraph()
{
  XScopedReadLock<XRecursiveRWLock> lock(m_mutex);
  {
      XScopedLock<XGraph> lock(*m_graph);
      
	for(std::deque<Plot>::iterator it = m_plots.begin(); it != m_plots.end(); it++) {        
      int rowcnt = rowCount();
      double *colx = cols(it->colx);
      double *coly = NULL;
      if(it->coly1 >= 0) coly = cols(it->coly1);
      if(it->coly2 >= 0) coly = cols(it->coly2);
      double *colweight = (it->colweight >= 0) ? cols(it->colweight) : NULL;
      double *colz = (it->colz >= 0) ? cols(it->colz) : NULL;
    
      if(colweight) {
	      double weight_max = 0.0;
	      for(int i = 0; i < rowcnt; i++)
		      	weight_max = std::max(weight_max, colweight[i]);
	      m_axisw->maxValue()->value(weight_max);
	      m_axisw->minValue()->value(-0.4*weight_max);
      }
            
      std::deque<XGraph::ValPoint> &points_plot(it->xyplot->points());
      points_plot.clear();
      for(int i = 0; i < rowcnt; i++)
        {
          double z = 0.0;
          if(colz)
            z = colz[i];
          if(colweight) {
          	if(colweight[i] > 0)
               points_plot.push_back( XGraph::ValPoint(colx[i], coly[i], z, colweight[i]) );
          }
          else
               points_plot.push_back( XGraph::ValPoint(colx[i], coly[i], z) );
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
