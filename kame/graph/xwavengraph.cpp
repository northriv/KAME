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
#include <iomanip>

#include <QPushButton>
#include <QStatusBar>
#include <QStyle>

#define OFSMODE (std::ios::out | std::ios::app | std::ios::ate)

//---------------------------------------------------------------------------

XWaveNGraph::XWaveNGraph(const char *name, bool runtime, FrmGraphNURL *item) :
    XWaveNGraph(name, runtime, item->m_graphwidget, item->m_edUrl,
        item->m_btnUrl, item->m_btnDump) {

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

XWaveNGraph::Payload::XPlotWrapper::XPlotWrapper(const char *name, bool runtime,
    Transaction &tr_graph, const shared_ptr<XGraph> &graph) :
    XPlot(name, runtime, tr_graph, graph) {

}
void
XWaveNGraph::Payload::XPlotWrapper::snapshot(const Snapshot &shot_graph) {
    auto waves = m_parent.lock();
    if( !waves) return;
    //Snapshot only for the parent. Otherwise, transaction of graph will fail.
    SingleSnapshot<XWaveNGraph> shot_waves( *waves);
    int rowcnt = shot_waves->rowCount();
    m_ptsSnapped.clear();
    m_ptsSnapped.reserve(rowcnt);
    if( !rowcnt)
        return;

    std::vector<XGraph::VFloat> cols[4];
    const XGraph::VFloat *pcolx = nullptr, *pcoly = nullptr,
            *pcolz = nullptr, *pcolweight = nullptr;

    auto prepare_column_data =
        [&](int colidx, std::vector<XGraph::VFloat>&buf, const XGraph::VFloat *&pcolumn) {
        if(colidx >= 0) {
            pcolumn = shot_waves->m_cols[colidx]->fillOrPointToGraphPoints(buf);
        }
    };
    prepare_column_data(m_colx, cols[0], pcolx);
    prepare_column_data(m_coly1, cols[1], pcoly);
    prepare_column_data(m_coly2, cols[1], pcoly);
    prepare_column_data(m_colz, cols[2], pcolz);
    prepare_column_data(m_colweight, cols[3], pcolweight);

    for(int i = 0; i < rowcnt; ++i) {
        double z = 0.0;
        if(pcolz)
            z = pcolz[i];
        if(pcolweight) {
            if(pcolweight[i] > 0) {
                m_ptsSnapped.push_back(XGraph::ValPoint(pcolx[i],
                    pcoly[i], z, pcolweight[i]));
            }
        }
        else {
            m_ptsSnapped.push_back(XGraph::ValPoint(pcolx[i], pcoly[i], z));
        }
    }
}

void
XWaveNGraph::Payload::clearPoints() {
	setRowCount(0);

	shared_ptr<XGraph> graph(static_cast<XWaveNGraph*>( &node())->graph());
	tr().mark(tr()[ *graph].onUpdate(), graph.get());
}
void
XWaveNGraph::Payload::clearPlots() {
	const auto &graph(static_cast<XWaveNGraph &>(node()).m_graph);
    for(auto &&x: m_plots) {
        graph->plots()->release(tr(), x);
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

	if(weight >= 0) {
		if((m_colw >= 0) && (m_colw != weight))
			m_colw = -1;
		else
			m_colw = weight;
	}

	// graph->setName(getName());
	tr()[ *graph->label()] = node().getLabel();

	unsigned int plotnum = m_plots.size() + 1;
    auto plot = graph->plots()->create<XPlotWrapper>(tr(), formatString("Plot%u",
		plotnum).c_str(), true, ref(tr()), graph);
    plot->m_parent = static_pointer_cast<XWaveNGraph>(node().shared_from_this());
    plot->m_colx = x;
    plot->m_coly1 = y1;
    plot->m_coly2 = y2;
    plot->m_colweight = weight;
    plot->m_colz = z;

    tr()[ *plot->label()] = label;
	const XNode::NodeList &axes_list( *tr().list(graph->axes()));
	m_axisx = static_pointer_cast<XAxis>(axes_list.at(0));
	m_axisy = static_pointer_cast<XAxis>(axes_list.at(1));
    tr()[ *plot->axisX()] = m_axisx;
    tr()[ *m_axisx->label()] = m_labels[plot->m_colx];
    if(plot->m_coly1 >= 0) {
        tr()[ *plot->axisY()] = m_axisy;
        tr()[ *m_axisy->label()] = m_labels[plot->m_coly1];
	}
    tr()[ *plot->maxCount()] = rowCount();
    tr()[ *plot->maxCount()].setUIEnabled(false);
    tr()[ *plot->clearPoints()].setUIEnabled(false);
    tr()[ *plot->intensity()] = 1.0;
//	if(m_plots.size()) {
//        tr()[ *plot.xyplot->pointColor()] = clAqua; //Green;
//        tr()[ *plot.xyplot->lineColor()] = clAqua; //Green;
//        tr()[ *plot.xyplot->barColor()] = clAqua; //Green;
//		tr()[ *plot.xyplot->displayMajorGrid()] = false;
//	}

    if(plot->m_colz >= 0) {
		if( !m_axisz) {
			m_axisz = graph->axes()->create<XAxis>(tr(), "Z Axis", true,
                XAxis::AxisDirection::Z, true, ref(tr()), graph);
		}
        tr()[ *plot->axisZ()] = m_axisz;
        tr()[ *m_axisz->label()] = m_labels[plot->m_colz];
	}
    if(plot->m_colweight >= 0) {
		if( !m_axisw) {
			m_axisw = graph->axes()->create<XAxis>(tr(), "Weight", true,
                XAxis::AxisDirection::Weight, true, ref(tr()), graph);
		}
		tr()[ *m_axisw->autoScale()] = false;
		tr()[ *m_axisw->autoScale()].setUIEnabled(false);
        tr()[ *plot->axisW()] = m_axisw;
        tr()[ *m_axisw->label()] = m_labels[plot->m_colweight];
	}
    if(plot->m_coly2 >= 0) {
		if( !m_axisy2) {
			m_axisy2 = graph->axes()->create<XAxis>(tr(), "Y2 Axis", true,
                XAxis::AxisDirection::Y, true, ref(tr()), graph);
		}
        tr()[ *plot->axisY()] = m_axisy2;
        tr()[ *m_axisy2->label()] = m_labels[plot->m_coly2];
	}

	m_plots.push_back(plot);

    graph->applyTheme(tr(), true);
}

void
XWaveNGraph::Payload::setColCount(unsigned int colcnt, const char **labels) {
    m_cols.resize(colcnt);
    m_labels.resize(colcnt);
    m_precisions.resize(colcnt, 6);
    for(auto &&label: m_labels)
        label = *labels++;
}
void
XWaveNGraph::Payload::setLabel(unsigned int col, const char *label) {
    m_labels.at(col) = label;
}
void
XWaveNGraph::Payload::setRowCount(unsigned int n) {
    m_rowCount = n;
    for(auto &&plot: m_plots) {
        tr()[ *plot->maxCount()] = n;
	}
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
    if(m_filemutex.trylock()) {
        m_filemutex.unlock();
    }
    else {
        gWarnPrint(i18n("Previous dump is still on going. It is deferred."));
    }
    m_threadDump.reset(new XThread{shared_from_this(),
        [this](const atomic<bool>&, Snapshot &&shot){
        XScopedLock<XMutex> filelock(m_filemutex);
        if( !m_stream.good()) {
            gErrPrint(i18n("File cannot open."));
            return;
        }
        Transactional::setCurrentPriorityMode(Priority::UI_DEFERRABLE);

        int rowcnt = shot[ *this].rowCount();
        int colcnt = shot[ *this].colCount();

        m_stream << "#" << getLabel() << std::endl;
        m_stream << "#";
        for(unsigned int i = 0; i < colcnt; i++) {
            m_stream << shot[ *this].labels()[i] << KAME_DATAFILE_DELIMITER;
        }
//		stream << std::endl;
        m_stream << "#at " << (XTime::now()).getTimeFmtStr(
            "%Y/%m/%d %H:%M:%S") << std::endl;

        auto &p(shot[ *this]);
        shared_ptr<Payload::ColumnBase> colw;
        if(p.m_colw >= 0) colw = p.m_cols[p.m_colw];
        int written_rows = 0;
        for(unsigned int i = 0; i < rowcnt; i++) {
            if( !colw || (colw->moreThanZero(i))) {
                for(unsigned int j = 0; j < colcnt; j++) {
                    m_stream << std::setprecision(p.m_cols[j]->precision);
                    p.m_cols[j]->toOFStream(m_stream, i);
                    m_stream << KAME_DATAFILE_DELIMITER;
                }
                m_stream << std::endl;
                written_rows++;
            }
        }
        m_stream << std::endl;

        m_stream.flush();

        gMessagePrint(formatString_tr(I18N_NOOP("Succesfully %d rows written into %s."), written_rows, shot[ *filename()].to_str().c_str()));
    }, Snapshot( *this)});

    iterate_commit([=](Transaction &tr){
        tr.mark(tr[ *this].onIconChanged(), true);
    });
}
void XWaveNGraph::drawGraph(Transaction &tr) {
	const Snapshot &shot(tr);
    if(shot[ *this].m_colw >= 0) {
        XGraph::VFloat weight_max = shot[ *this].m_cols[shot[ *this].m_colw]->max();
        tr[ *shot[ *this].axisw()->maxValue()] = weight_max;
        tr[ *shot[ *this].axisw()->minValue()] =  -0.4 * weight_max;
    }
	tr.mark(tr[ *m_graph].onUpdate(), m_graph.get());
}
