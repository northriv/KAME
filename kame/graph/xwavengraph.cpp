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
#include "xwavengraph.h"

#include "ui_graphnurlform.h"
#include "graphwidget.h"
#include "graph.h"

#include <QPushButton>
#include <QStatusBar>

#define OFSMODE (std::ios::out | std::ios::app | std::ios::ate)

//---------------------------------------------------------------------------

XWaveNGraph::XWaveNGraph(const char *name, bool runtime, FrmGraphNURL *item) :
	XNode(name, runtime), m_btnDump(item->m_btnDump), m_graph(create<XGraph> (
		name, false)), m_dump(create<XTouchableNode> ("Dump", true)), m_filename(create<
		XStringNode> ("FileName", true)) {
	item->m_graphwidget->setGraph(m_graph);
    m_conFilename = xqcon_create<XFilePathConnector> (m_filename, item->m_edUrl, item->m_btnUrl,
		"Data files (*.dat);;All files (*.*)", true);
	m_conDump = xqcon_create<XQButtonConnector> (m_dump, item->m_btnDump);
	init();
}
XWaveNGraph::XWaveNGraph(const char *name, bool runtime, XQGraph *graphwidget,
    QLineEdit *ed, QAbstractButton *btn, QPushButton *btndump) :
	XNode(name, runtime), m_btnDump(btndump), m_graph(create<XGraph> (name,
		false)), m_dump(create<XTouchableNode> ("Dump", true)), m_filename(create<
		XStringNode> ("FileName", true)) {
	graphwidget->setGraph(m_graph);
    m_conFilename = xqcon_create<XFilePathConnector> (m_filename, ed, btn,
		"Data files (*.dat);;All files (*.*)", true);
	m_conDump = xqcon_create<XQButtonConnector> (m_dump, btndump);
	init();
}
void XWaveNGraph::init() {
	iterate_commit([=](Transaction &tr){
		m_lsnOnFilenameChanged = tr[ *filename()].onValueChanged().connectWeakly(
			shared_from_this(), &XWaveNGraph::onFilenameChanged);
    });

    iterate_commit([=](Transaction &tr){
        m_lsnOnIconChanged = tr[ *this].onIconChanged().connectWeakly(
			shared_from_this(),
			&XWaveNGraph::onIconChanged, Listener::FLAG_MAIN_THREAD_CALL
				| Listener::FLAG_AVOID_DUP);
		tr.mark(tr[ *this].onIconChanged(), false);

		tr[ *dump()].setUIEnabled(false);
        tr[ *m_graph->persistence()] = 0.4;
		tr[ *this].clearPlots();
    });
}
XWaveNGraph::~XWaveNGraph() {
	m_stream.close();
}

void
XWaveNGraph::Payload::clearPoints() {
	setRowCount(0);
	for(int i = 0; i < numPlots(); ++i)
		tr()[ *plot(i)].points().clear();

	shared_ptr<XGraph> graph(static_cast<XWaveNGraph*>( &node())->graph());
	tr().mark(tr()[ *graph].onUpdate(), graph.get());
}
void
XWaveNGraph::Payload::clearPlots() {
	const auto &graph(static_cast<XWaveNGraph &>(node()).m_graph);
	for(auto it = m_plots.begin(); it != m_plots.end(); it++) {
		graph->plots()->release(tr(), it->xyplot);
	}
	if(m_axisw)
		graph->axes()->release(tr(), m_axisw);
	if(m_axisz)
		graph->axes()->release(tr(), m_axisz);
	if(m_axisy2)
		graph->axes()->release(tr(), m_axisy2);

	if(m_axisx)
		tr()[ *m_axisx->label()] = "";
	if(m_axisy)
		tr()[ *m_axisy->label()] = "";
	m_plots.clear();
	m_axisy2.reset();
	m_axisz.reset();
	m_axisw.reset();
    m_colw = -1;
}
void
XWaveNGraph::Payload::insertPlot(const XString &label, int x, int y1, int y2,
	int weight, int z) {
	const auto &graph(static_cast<XWaveNGraph &>(node()).m_graph);
	assert( (y1 < 0) || (y2 < 0) );
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

	// graph->setName(getName());
	tr()[ *graph->label()] = node().getLabel();

	unsigned int plotnum = m_plots.size() + 1;
	plot.xyplot = graph->plots()->create<XXYPlot>(tr(), formatString("Plot%u",
		plotnum).c_str(), true, ref(tr()), graph);

	tr()[ *plot.xyplot->label()] = label;
	const XNode::NodeList &axes_list( *tr().list(graph->axes()));
	m_axisx = static_pointer_cast<XAxis>(axes_list.at(0));
	m_axisy = static_pointer_cast<XAxis>(axes_list.at(1));
	tr()[ *plot.xyplot->axisX()] = m_axisx;
	tr()[ *m_axisx->label()] = m_labels[plot.colx];
	if(plot.coly1 >= 0) {
		tr()[ *plot.xyplot->axisY()] = m_axisy;
		tr()[ *m_axisy->label()] = m_labels[plot.coly1];
	}
	tr()[ *plot.xyplot->maxCount()] = rowCount();
	tr()[ *plot.xyplot->maxCount()] = false;
	tr()[ *plot.xyplot->clearPoints()].setUIEnabled(false);
	tr()[ *plot.xyplot->intensity()] = 1.0;
//	if(m_plots.size()) {
//        tr()[ *plot.xyplot->pointColor()] = clAqua; //Green;
//        tr()[ *plot.xyplot->lineColor()] = clAqua; //Green;
//        tr()[ *plot.xyplot->barColor()] = clAqua; //Green;
//		tr()[ *plot.xyplot->displayMajorGrid()] = false;
//	}

	if(plot.colz >= 0) {
		if( !m_axisz) {
			m_axisz = graph->axes()->create<XAxis>(tr(), "Z Axis", true,
                XAxis::AxisDirection::Z, true, ref(tr()), graph);
		}
		tr()[ *plot.xyplot->axisZ()] = m_axisz;
		tr()[ *m_axisz->label()] = m_labels[plot.colz];
	}
	if(plot.colweight >= 0) {
		if( !m_axisw) {
			m_axisw = graph->axes()->create<XAxis>(tr(), "Weight", true,
                XAxis::AxisDirection::Weight, true, ref(tr()), graph);
		}
		tr()[ *m_axisw->autoScale()] = false;
		tr()[ *m_axisw->autoScale()].setUIEnabled(false);
		tr()[ *plot.xyplot->axisW()] = m_axisw;
		tr()[ *m_axisw->label()] = m_labels[plot.colweight];
	}
	if(plot.coly2 >= 0) {
		if( !m_axisy2) {
			m_axisy2 = graph->axes()->create<XAxis>(tr(), "Y2 Axis", true,
                XAxis::AxisDirection::Y, true, ref(tr()), graph);
		}
		tr()[ *plot.xyplot->axisY()] = m_axisy2;
		tr()[ *m_axisy2->label()] = m_labels[plot.coly2];
	}

	m_plots.push_back(plot);

    graph->applyTheme(tr(), true);
}

void
XWaveNGraph::Payload::setColCount(unsigned int n, const char **labels) {
    m_colcnt = n;
    m_labels.resize(m_colcnt);
    for(unsigned int i = 0; i < n; i++) {
        m_labels[i] = labels[i];
    }
}
void
XWaveNGraph::Payload::setLabel(unsigned int col, const char *label) {
	m_labels[col] = label;
}
void
XWaveNGraph::Payload::setRowCount(unsigned int n) {
	m_cols.resize(m_colcnt * n);
    for(auto it = m_plots.begin(); it != m_plots.end(); it++) {
		tr()[ *it->xyplot->maxCount()] = n;
	}
}
double *
XWaveNGraph::Payload::cols(unsigned int n) {
    return rowCount() ? &(m_cols[rowCount() * n]) : 0;
}
const double *
XWaveNGraph::Payload::cols(unsigned int n) const {
    return rowCount() ? &(m_cols[rowCount() * n]) : 0;
}
const double *
XWaveNGraph::Payload::weight() const {
    if(m_colw < 0) return 0L;
    return cols(m_colw);
}

void
XWaveNGraph::onIconChanged(const Snapshot &shot, bool v) {
	if( !m_conDump->isAlive()) return;
	if( !v)
        m_btnDump->setIcon(QApplication::style()->
            standardIcon(QStyle::SP_DialogSaveButton));
	else
        m_btnDump->setIcon(QApplication::style()->
            standardIcon(QStyle::SP_BrowserReload));
}
void
XWaveNGraph::onFilenameChanged(const Snapshot &shot, XValueNodeBase *) {
	{
		XScopedLock<XMutex> lock(m_filemutex);

		if(m_stream.is_open())
			m_stream.close();
		m_stream.clear();
		m_stream.open(
            (const char*)QString(shot[ *filename()].to_str().c_str()).toLocal8Bit().data(),
			OFSMODE);

		iterate_commit([=](Transaction &tr){
			if(m_stream.good()) {
				m_lsnOnDumpTouched = tr[ *dump()].onTouch().connectWeakly(
					shared_from_this(), &XWaveNGraph::onDumpTouched);
				tr[ *dump()].setUIEnabled(true);
			}
			else {
				m_lsnOnDumpTouched.reset();
				tr[ *dump()].setUIEnabled(false);
				gErrPrint(i18n("Failed to open file."));
			}
			tr.mark(tr[ *this].onIconChanged(), false);
        });
	}
}

void
XWaveNGraph::onDumpTouched(const Snapshot &, XTouchableNode *) {
    Snapshot shot( *this);
    {
        XScopedLock<XMutex> filelock(m_filemutex);
        if( !m_stream.good()) return;

        int rowcnt = shot[ *this].rowCount();
        int colcnt = shot[ *this].colCount();

        m_stream << "#";
        for(unsigned int i = 0; i < colcnt; i++) {
            m_stream << shot[ *this].labels()[i] << KAME_DATAFILE_DELIMITER;
        }
//		stream << std::endl;
        m_stream << "#at " << (XTime::now()).getTimeFmtStr(
            "%Y/%m/%d %H:%M:%S") << std::endl;

        auto &p(shot[ *this]);
        for(unsigned int i = 0; i < rowcnt; i++) {
            if( !p.weight() || (p.weight()[i] > 0)) {
                for(unsigned int j = 0; j < colcnt; j++) {
                    m_stream << p.cols(j)[i] << KAME_DATAFILE_DELIMITER;
                }
                m_stream << std::endl;
            }
        }
        m_stream << std::endl;

        m_stream.flush();
    }

    iterate_commit([=](Transaction &tr){
        tr.mark(tr[ *this].onIconChanged(), true);
    });
}
void XWaveNGraph::drawGraph(Transaction &tr) {
	const Snapshot &shot(tr);
	for(int i = 0; i < shot[ *this].numPlots(); ++i) {
		int rowcnt = shot[ *this].rowCount();
		double *colx = tr[ *this].cols(shot[ *this].colX(i));
		double *coly = NULL;
		if(shot[ *this].colY1(i) >= 0)
			coly = tr[ *this].cols(shot[ *this].colY1(i));
		if(shot[ *this].colY2(i) >= 0)
			coly = tr[ *this].cols(shot[ *this].colY2(i));
		double *colweight = (shot[ *this].colWeight(i) >= 0) ? tr[ *this].cols(shot[ *this].colWeight(i)) : NULL;
		double *colz = (shot[ *this].colZ(i) >= 0) ? tr[ *this].cols(shot[ *this].colZ(i)) : NULL;

		if(colweight) {
			double weight_max = 0.0;
			for(int i = 0; i < rowcnt; i++)
				weight_max = std::max(weight_max, colweight[i]);
			tr[ *shot[ *this].axisw()->maxValue()] = weight_max;
			tr[ *shot[ *this].axisw()->minValue()] =  -0.4 * weight_max;
		}

        auto &points_plot(tr[ *shot[ *this].plot(i)].points());
		points_plot.clear();
		for(int i = 0; i < rowcnt; ++i) {
			double z = 0.0;
			if(colz)
				z = colz[i];
			if(colweight) {
				if(colweight[i] > 0)
					points_plot.push_back(XGraph::ValPoint(colx[i],
						coly[i], z, colweight[i]));
			}
			else
				points_plot.push_back(XGraph::ValPoint(colx[i], coly[i], z));
		}
	}
	tr.mark(tr[ *m_graph].onUpdate(), m_graph.get());
}
