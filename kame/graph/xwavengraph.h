/***************************************************************************
		Copyright (C) 2002-2007 Kentaro Kitagawa
		                   kitagawa@scphys.kyoto-u.ac.jp
		
		This program is free software; you can redistribute it and/or
		modify it under the terms of the GNU Library General Public
		License as published by the Free Software Foundation; either
		version 2 of the License, or (at your option) any later version.
		
		You should have received a copy of the GNU Library General 
		Public License and a list of authors along with this program; 
		see the files COPYING and AUTHORS.
 ***************************************************************************/
//---------------------------------------------------------------------------

#ifndef xwavengraphH
#define xwavengraphH
//---------------------------------------------------------------------------

#include "xnodeconnector.h"
#include <vector>
#include "graph.h"
#include <fstream>

class FrmGraphNURL;
class XQGraph;
class KURLRequester;
class QPushButton;
class XAxis;
class XXYPlot;

class XWaveNGraph : public XNode
{
 XNODE_OBJECT
 protected:
  XWaveNGraph(const char *name, bool runtime, FrmGraphNURL *item);
  XWaveNGraph(const char *name, bool runtime, 
        XQGraph *graphwidget, KURLRequester *urlreq, QPushButton *btndump);
 public:
  virtual ~XWaveNGraph();

  void setRowCount(unsigned int rowcnt);
  void setColCount(unsigned int colcnt, const char **lables);
  void setLabel(unsigned int col, const char *label);
  void selectAxes(int colx = 0, int coly1 = 1, int coly2 = -1, int colweight = -1, int colz = -1);

  unsigned int rowCount() const;
  unsigned int colCount() const;
  double *cols(unsigned int n);
  
  //! clear all data points
  void clear();
  
  void readLock() const;
  void readUnlock() const;
  //! now allow user to access data
  void writeLock();
  //! unlock and update graph
  void writeUnlock(bool updategraph = true);

  int colX() const;
  int colY1() const;
  int colY2() const;
  int colWeight() const; 
  int colZ() const; 

  const shared_ptr<XGraph> &graph() const {return m_graph;}
  const shared_ptr<XXYPlot> &plot1() const {return m_plot1;}
  const shared_ptr<XXYPlot> &plot2() const {return m_plot2;}
  const shared_ptr<XAxis> &axisx() const {return m_axisx;}  
  const shared_ptr<XAxis> &axisy() const {return m_axisy;}  
  const shared_ptr<XAxis> &axisy2() const {return m_axisy2;}  
  const shared_ptr<XAxis> &axisz() const {return m_axisz;}  
  const shared_ptr<XAxis> &axisw() const {return m_axisw;}  
  const shared_ptr<XNode> &dump() const {return m_dump;}
  const shared_ptr<XStringNode> &filename() const {return m_filename;}
 private:
  XRecursiveRWLock m_mutex;
  unsigned int m_colcnt;
  std::vector<std::string> m_labels;
  std::vector<double> m_cols;

  void init();
  void drawGraph();

  QPushButton *const m_btnDump;
  
  const shared_ptr<XGraph> m_graph;
  shared_ptr<XXYPlot> m_plot1, m_plot2;
  shared_ptr<XAxis> m_axisx, m_axisy, m_axisy2, m_axisw, m_axisz;
  
  const shared_ptr<XNode> m_dump;
  const shared_ptr<XStringNode> m_filename;

  shared_ptr<XListener> m_lsnOnDumpTouched, m_lsnOnFilenameChanged;

  void onDumpTouched(const shared_ptr<XNode> &);
  void onFilenameChanged(const shared_ptr<XValueNodeBase> &);
  
  std::fstream m_stream;
  XMutex m_filemutex;
  
  xqcon_ptr m_conFilename, m_conDump;

  int m_colx, m_coly1, m_coly2, m_colweight, m_colz;
};

#endif
