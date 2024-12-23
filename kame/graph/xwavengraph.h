/***************************************************************************
        Copyright (C) 2002-2023 Kentaro Kitagawa
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
#include "graphntoolbox.h"

class XAxis;
class XXYPlot;

class XDriver;
class XMeasure;
class XGraph1DMathToolList;
class XQGraph1DMathToolConnector;

//! Graph widget with internal data sets. The data can be saved as a text file.
//! \sa XQGraph, XGraph
class DECLSPEC_KAME XWaveNGraph: public XGraphNToolBox {
public:
    XWaveNGraph(const char *name, bool runtime, FrmGraphNURL *item);
    XWaveNGraph(const char *name, bool runtime, XQGraph *graphwidget,
        QLineEdit *ed, QAbstractButton *btn, QPushButton *btndump,
        QToolButton *m_btnmath, const shared_ptr<XMeasure> &meas, const shared_ptr<XDriver> &driver);
    XWaveNGraph(const char *name, bool runtime, XQGraph *graphwidget,
        QLineEdit *ed, QAbstractButton *btn, QPushButton *btndump);
    virtual ~XWaveNGraph();

	void drawGraph(Transaction &tr);

    bool clearPlots(Transaction &tr);

    struct DECLSPEC_KAME Payload : public XGraphNToolBox::Payload {
		void clearPoints();
        bool insertPlot(Transaction &tr, const XString &label, int colx = 0, int coly1 = 1,
			int coly2 = -1, int colweight = -1, int colz = -1);

		void setLabel(unsigned int col, const char *label);
        const std::vector<XString> &labels() const {return m_labels;}
        unsigned int precision(unsigned int col) const {return m_cols.at(col)->precision;}
        void setRowCount(unsigned int rowcnt);
        void setCols(const std::initializer_list<std::string> &labels);
        void setColCount(unsigned int colcnt, const char** labels);
        unsigned int rowCount() const {return m_rowCount; }
        unsigned int colCount() const {return m_cols.size();}
        unsigned int numPlots() const { return m_plots.size();}

        template <typename VALUE>
        void setColumn(unsigned int n, std::vector<VALUE> &&column, unsigned int precision = std::numeric_limits<VALUE>::digits10 + 1);

        shared_ptr<XPlot> plot(unsigned int plotnum) const { return m_plots[plotnum];}
		const shared_ptr<XAxis> &axisx() const { return m_axisx;}
		const shared_ptr<XAxis> &axisy() const { return m_axisy;}
		const shared_ptr<XAxis> &axisy2() const { return m_axisy2;}
		const shared_ptr<XAxis> &axisz() const { return m_axisz;}
		const shared_ptr<XAxis> &axisw() const { return m_axisw;}
	private:
        friend class XWaveNGraph;
        int m_colw;
        std::vector<XString> m_labels;
        std::vector<unsigned int> m_precisions;
        size_t m_rowCount;
        struct ColumnBase {
            ColumnBase(unsigned int prec) : precision(prec) {}
            virtual ~ColumnBase() = default;
            virtual double max() const = 0;
            virtual bool moreThanZero(size_t i) const = 0;
            virtual const std::vector<XGraph::VFloat> &fillOrPointToGraphPoints(std::vector<XGraph::VFloat>& buf) const = 0;
            virtual void toOFStream(std::fstream &s, size_t idx) = 0;
            unsigned int precision;
            template <typename VALUE>
            static const std::vector<XGraph::VFloat> &fillOrPointToGraphPointsBasic(std::vector<XGraph::VFloat>& buf,
                const std::vector<VALUE>& vector) {
                buf.resize(vector.size());
                std::copy(vector.cbegin(), vector.cend(), buf.begin());
                return buf;
            }
            static const std::vector<XGraph::VFloat> &fillOrPointToGraphPointsBasic(std::vector<XGraph::VFloat>&,
                const std::vector<XGraph::VFloat>&vector) {
                return vector;
            }
        };
        template <typename VALUE>
        struct Column : public ColumnBase {
            Column(std::vector<VALUE> &&vec, unsigned int prec) : ColumnBase(prec), vector(std::move(vec)) {}
            virtual ~Column() = default;
            virtual double max() const {
                if(vector.empty()) return 0.0;
                return *std::max_element(vector.cbegin(), vector.cend());
            }
            virtual bool moreThanZero(size_t i) const {
                return vector[i] > 0;
            }
            virtual const std::vector<XGraph::VFloat> &fillOrPointToGraphPoints(std::vector<XGraph::VFloat>& buf) const {
                return fillOrPointToGraphPointsBasic(buf, vector);
            }
            virtual void toOFStream(std::fstream &s, size_t idx) {
                s << vector[idx];
            }
            std::vector<VALUE> vector;
        };
        std::vector<shared_ptr<ColumnBase>> m_cols;

        struct XPlotWrapper : public XPlot {
            XPlotWrapper(const char *name, bool runtime, Transaction &tr_graph, const shared_ptr<XGraph> &graph);
            virtual void clearAllPoints(Transaction &) override {}
            //! Takes a snap-shot all points for rendering
            virtual void snapshot(const Snapshot &shot) override;
            weak_ptr<XWaveNGraph> m_parent;
            int m_colx, m_coly1, m_coly2, m_colweight, m_colz;
        };
        std::vector<shared_ptr<XPlotWrapper>> m_plots;
        shared_ptr<XAxis> m_axisx, m_axisy, m_axisy2, m_axisw, m_axisz;
        std::deque<shared_ptr<XGraph1DMathToolList>> m_toolLists;
    };
protected:
    virtual void dumpToFileThreaded(std::fstream &, const Snapshot &, const std::string &ext) override;
private:
    weak_ptr<XMeasure> m_meas;
    weak_ptr<XDriver> m_driver;
    QToolButton *m_btnMathTool = nullptr;    
    XQGraph *m_graphwidget;

    TalkerOnce<XWaveNGraph*> m_tlkOnPlotInsertion;
    shared_ptr<Listener> m_lsnOnPlotInsertion;
    void OnPlotInsertion(const Snapshot &shot, XWaveNGraph *wave);
    shared_ptr<XQGraph1DMathToolConnector> m_conTools;
};

template <typename VALUE>
void
XWaveNGraph::Payload::setColumn(unsigned int n, std::vector<VALUE> &&column, unsigned int precision) {
    if(column.size() != m_rowCount)
        throw std::range_error("Invalid row count.");
    m_cols.at(n) = std::make_shared<Column<VALUE>>(std::move(column), precision);
}

#endif
