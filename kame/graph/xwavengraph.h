/***************************************************************************
		Copyright (C) 2002-2015 Kentaro Kitagawa
		                   kitagawa@phys.s.u-tokyo.ac.jp
		
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
class QLineEdit;
class QAbstractButton;
class QPushButton;
class XAxis;
class XXYPlot;
class Ui_FrmGraphNURL;
typedef QForm<QWidget, Ui_FrmGraphNURL> FrmGraphNURL;

//! Graph widget with internal data sets. The data can be saved as a text file.
//! \sa XQGraph, XGraph

class DECLSPEC_KAME XWaveNGraph: public XNode {
public:
	XWaveNGraph(const char *name, bool runtime, FrmGraphNURL *item);
	XWaveNGraph(const char *name, bool runtime, XQGraph *graphwidget,
        QLineEdit *ed, QAbstractButton *btn, QPushButton *btndump);
	virtual ~XWaveNGraph();

	const shared_ptr<XGraph> &graph() const { return m_graph;}
	const shared_ptr<XStringNode> &filename() const { return m_filename;}

	const shared_ptr<XTouchableNode> &dump() const { return m_dump;}
	void drawGraph(Transaction &tr);

    struct DECLSPEC_KAME Payload : public XNode::Payload {
		void clearPoints();
		void clearPlots();
		void insertPlot(const XString &label, int colx = 0, int coly1 = 1,
			int coly2 = -1, int colweight = -1, int colz = -1);

		void setLabel(unsigned int col, const char *label);
        const std::vector<XString> &labels() const {return m_labels;}
        unsigned int precision(unsigned int col) const {return m_cols[col]->precision;}
        void setRowCount(unsigned int rowcnt);
        void setColCount(unsigned int colcnt, const char** labels);
        unsigned int rowCount() const {return m_rowCount; }
        unsigned int colCount() const {return m_cols.size();}
        unsigned int numPlots() const { return m_plots.size();}

        template <typename VALUE>
        void setColumn(unsigned int n, std::vector<VALUE> &&column, unsigned int precision = std::numeric_limits<VALUE>::digits10 + 1);
        const double *weight() const;
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

        const Talker<bool> &onIconChanged() const { return m_tlkOnIconChanged;}
        Talker<bool> &onIconChanged() { return m_tlkOnIconChanged;}
	private:
        friend class XWaveNGraph;
        int m_colw;
        std::vector<XString> m_labels;
        std::vector<unsigned int> m_precisions;
        size_t m_rowCount;
        struct ColumnBase {
            ColumnBase(unsigned int prec) : precision(prec) {}
            virtual ~ColumnBase() = default;
            virtual bool moreThanZero(size_t i) const = 0;
            virtual void fillGraphPoints(XGraph::VFloat *) const = 0;
            virtual void toOFStream(std::fstream &s, size_t idx) = 0;
            unsigned int precision;
        };
        template <typename VALUE>
        struct Column : public ColumnBase {
            Column(std::vector<VALUE> &&vec, unsigned int prec) : ColumnBase(prec), vector(std::move(vec)) {}
            virtual ~Column() = default;
            virtual bool moreThanZero(size_t i) const {
                return vector[i] > 0;
            }
            virtual void fillGraphPoints(XGraph::VFloat *vec) const {
                std::copy(vector.cbegin(), vector.cend(), vec);
            }
            virtual void toOFStream(std::fstream &s, size_t idx) {
                s << vector[idx];
            }
            std::vector<VALUE> vector;
        };
        std::vector<shared_ptr<ColumnBase>> m_cols;
        struct Plot {
            shared_ptr<XXYPlot> xyplot;
            int colx, coly1, coly2, colweight, colz;
        };
        std::vector<Plot> m_plots;
		shared_ptr<XAxis> m_axisx, m_axisy, m_axisy2, m_axisw, m_axisz;

        Talker<bool> m_tlkOnIconChanged;
	};
private:
	void init();

	QPushButton * const m_btnDump;

	const shared_ptr<XGraph> m_graph;

	const shared_ptr<XTouchableNode> m_dump;
	const shared_ptr<XStringNode> m_filename;

	shared_ptr<Listener> m_lsnOnDumpTouched, m_lsnOnFilenameChanged,
		m_lsnOnIconChanged;

	void onDumpTouched(const Snapshot &shot, XTouchableNode *);
	void onFilenameChanged(const Snapshot &shot, XValueNodeBase *);
	void onIconChanged(const Snapshot &shot, bool );

	xqcon_ptr m_conFilename, m_conDump;

    unique_ptr<XThread> m_threadDump;
	std::fstream m_stream;
	XMutex m_filemutex;
};

template <typename VALUE>
void
XWaveNGraph::Payload::setColumn(unsigned int n, std::vector<VALUE> &&column, unsigned int precision) {
    assert(column.size() == m_rowCount);
    m_cols.at(n) = std::make_shared<Column<VALUE>>(std::move(column), precision);
}

#endif
