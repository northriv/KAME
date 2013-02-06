/***************************************************************************
		Copyright (C) 2002-2013 Kentaro Kitagawa
		                   kitag@kochi-u.ac.jp
		
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

class XQGraph;
class KUrlRequester;
class QPushButton;
class XAxis;
class XXYPlot;
class Ui_FrmGraphNURL;
typedef QForm<QMainWindow, Ui_FrmGraphNURL> FrmGraphNURL;

//! Graph widget with internal data sets. The data can be saved as a text file.
//! \sa XQGraph, XGraph

class XWaveNGraph: public XNode {
public:
	XWaveNGraph(const char *name, bool runtime, FrmGraphNURL *item);
	XWaveNGraph(const char *name, bool runtime, XQGraph *graphwidget,
		KUrlRequester *urlreq, QPushButton *btndump);
	virtual ~XWaveNGraph();

	const shared_ptr<XGraph> &graph() const { return m_graph;}
	const shared_ptr<XStringNode> &filename() const { return m_filename;}

	const shared_ptr<XTouchableNode> &dump() const { return m_dump;}
	void drawGraph(Transaction &tr);

	struct Payload : public XNode::Payload {
		void clearPoints();
		void clearPlots();
		void insertPlot(const XString &label, int colx = 0, int coly1 = 1,
			int coly2 = -1, int colweight = -1, int colz = -1);

		void setLabel(unsigned int col, const char *label);
		void setRowCount(unsigned int rowcnt);
		void setColCount(unsigned int colcnt, const char **lables);
		unsigned int rowCount() const;
		unsigned int colCount() const;
		unsigned int numPlots() const { return m_plots.size();}

		const double *cols(unsigned int n) const;
		double *cols(unsigned int n);
		//! \param plotnum start with zero.
		int colX(unsigned int plotnum) const { return m_plots[plotnum].colx;}
		//! \param plotnum start with zero.
		int colY1(unsigned int plotnum) const { return m_plots[plotnum].coly1;}
		//! \param plotnum start with zero.
		int colY2(unsigned int plotnum) const { return m_plots[plotnum].coly2;}
		//! \param plotnum start with zero.
		int colWeight(unsigned int plotnum) const { return m_plots[plotnum].colweight;}
		//! \param plotnum start with zero.
		int colZ(unsigned int plotnum) const { return m_plots[plotnum].colz;}
		//! \param plotnum start with zero.
		const shared_ptr<XXYPlot> &plot(unsigned int plotnum) const { return m_plots[plotnum].xyplot;}
		const shared_ptr<XAxis> &axisx() const { return m_axisx;}
		const shared_ptr<XAxis> &axisy() const { return m_axisy;}
		const shared_ptr<XAxis> &axisy2() const { return m_axisy2;}
		const shared_ptr<XAxis> &axisz() const { return m_axisz;}
		const shared_ptr<XAxis> &axisw() const { return m_axisw;}

		void dump(std::fstream &);

		const Talker<bool, bool> &onIconChanged() const { return m_tlkOnIconChanged;}
		Talker<bool, bool> &onIconChanged() { return m_tlkOnIconChanged;}
	private:
//		friend class XWaveNGraph;
		struct Plot {
			shared_ptr<XXYPlot> xyplot;
			int colx, coly1, coly2, colweight, colz;
		};
		unsigned int m_colcnt;
		std::vector<XString> m_labels;
		std::vector<double> m_cols;
		int m_colw;
		std::deque<Plot> m_plots;
		shared_ptr<XAxis> m_axisx, m_axisy, m_axisy2, m_axisw, m_axisz;

		Talker<bool, bool> m_tlkOnIconChanged;
	};
private:
	void init();

	QPushButton * const m_btnDump;

	const shared_ptr<XGraph> m_graph;

	const shared_ptr<XTouchableNode> m_dump;
	const shared_ptr<XStringNode> m_filename;

	shared_ptr<XListener> m_lsnOnDumpTouched, m_lsnOnFilenameChanged,
		m_lsnOnIconChanged;

	void onDumpTouched(const Snapshot &shot, XTouchableNode *);
	void onFilenameChanged(const Snapshot &shot, XValueNodeBase *);
	void onIconChanged(const Snapshot &shot, bool );

	xqcon_ptr m_conFilename, m_conDump;

	std::fstream m_stream;
	XMutex m_filemutex;
};

#endif
