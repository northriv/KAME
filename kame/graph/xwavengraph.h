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

//! Graph widget with internal data sets. The data can be saved as a text file.
//! \sa XQGraph, XGraph
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
 void insertPlot(const std::string &label,
	int colx = 0, int coly1 = 1, int coly2 = -1, int colweight = -1, int colz = -1);
 void clearPlots();
 unsigned int numPlots() const {return m_plots.size();}
 
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
 
 //! \arg plotnum start with zero.
 int colX(unsigned int plotnum) const {return m_plots[plotnum].colx;}
 //! \arg plotnum start with zero.
 int colY1(unsigned int plotnum) const {return m_plots[plotnum].coly1;}
 //! \arg plotnum start with zero.
 int colY2(unsigned int plotnum) const {return m_plots[plotnum].coly2;}
 //! \arg plotnum start with zero.
 int colWeight(unsigned int plotnum) const {return m_plots[plotnum].colweight;}
 //! \arg plotnum start with zero.
 int colZ(unsigned int plotnum) const {return m_plots[plotnum].colz;}
 
 const shared_ptr<XGraph> &graph() const {return m_graph;}
 //! \arg plotnum start with zero.
 const shared_ptr<XXYPlot> &plot(unsigned int plotnum) const {return m_plots[plotnum].xyplot;}
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
 struct Plot {
 	shared_ptr<XXYPlot> xyplot;
	int colx, coly1, coly2, colweight, colz;
 };
 int m_colw;
 std::deque<Plot> m_plots;
 shared_ptr<XAxis> m_axisx, m_axisy, m_axisy2, m_axisw, m_axisz;
 
 const shared_ptr<XNode> m_dump;
 const shared_ptr<XStringNode> m_filename;
 
 shared_ptr<XListener> m_lsnOnDumpTouched, m_lsnOnFilenameChanged, m_lsnOnIconChanged;
 XTalker<bool> m_tlkOnIconChanged;
 
 void onDumpTouched(const shared_ptr<XNode> &);
 void onFilenameChanged(const shared_ptr<XValueNodeBase> &);
 void onIconChanged(const bool &);
 
 std::fstream m_stream;
 XMutex m_filemutex;
 
 xqcon_ptr m_conFilename, m_conDump;
};

#endif
