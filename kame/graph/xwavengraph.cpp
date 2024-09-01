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
#include "xwavengraph.h"

#include "ui_graphnurlform.h"
#include "graphwidget.h"
#include "graph.h"
#include <iomanip>
#include "graphmathtoolconnector.h"

XWaveNGraph::XWaveNGraph(const char *name, bool runtime, FrmGraphNURL *item) :
    XWaveNGraph(name, runtime, item->m_graphwidget, item->m_edUrl,
        item->m_btnUrl, item->m_btnDump) {

}
XWaveNGraph::XWaveNGraph(const char *name, bool runtime, XQGraph *graphwidget,
    QLineEdit *ed, QAbstractButton *btn, QPushButton *btndump,
    QToolButton *btnmath,
    const shared_ptr<XMeasure> &meas, const shared_ptr<XDriver> &driver) :
    XWaveNGraph(name, runtime, graphwidget, ed, btn, btndump) {
    m_btnMathTool = btnmath;
    m_meas = meas;
    m_driver = driver;
}
XWaveNGraph::XWaveNGraph(const char *name, bool runtime, XQGraph *graphwidget,
    QLineEdit *ed, QAbstractButton *btn, QPushButton *btndump) :
    XGraphNToolBox(name, runtime, graphwidget, ed, btn, btndump),
    m_graphwidget(graphwidget) {
    iterate_commit([=](Transaction &tr){
        tr[ *dump()].setUIEnabled(false);
        tr[ *graph()->persistence()] = 0.4;
        clearPlots(tr);
    });
}
XWaveNGraph::~XWaveNGraph() {}

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
            pcolumn = &shot_waves->m_cols[colidx]->fillOrPointToGraphPoints(buf)[0];
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
bool
XWaveNGraph::clearPlots(Transaction &tr) {
    for(auto &&x: tr[ *this].m_toolLists) {
        if( !release(tr, x))
            return false; //transaction has failed.
    }
    tr[ *this].m_toolLists.clear();

    for(auto &&x: tr[ *this].m_plots) {
        if( !graph()->plots()->release(tr, x))
            return false;
	}
    if(tr[ *this].m_axisw)
        if( !graph()->axes()->release(tr, tr[ *this].m_axisw))
            return false;
    if(tr[ *this].m_axisz)
        if( !graph()->axes()->release(tr, tr[ *this].m_axisz))
            return false;
    if(tr[ *this].m_axisy2)
        if( !graph()->axes()->release(tr, tr[ *this].m_axisy2))
            return false;

    if(tr[ *this].m_axisx)
        tr[ *tr[ *this].m_axisx->label()] = "";
    if(tr[ *this].m_axisy)
        tr[ *tr[ *this].m_axisy->label()] = "";

    tr[ *this].m_plots.clear();
    tr[ *this].m_axisy2.reset();
    tr[ *this].m_axisz.reset();
    tr[ *this].m_axisw.reset();
    tr[ *this].m_colw = -1;
    return true;
}
bool
XWaveNGraph::Payload::insertPlot(Transaction &tr, const XString &label, int x, int y1, int y2,
	int weight, int z) {
    const auto &graph(static_cast<XWaveNGraph &>(node()).graph());
	assert( (y1 < 0) || (y2 < 0) );

	if(weight >= 0) {
		if((m_colw >= 0) && (m_colw != weight))
			m_colw = -1;
		else
			m_colw = weight;
	}

	// graph->setName(getName());
    tr[ *graph->label()] = node().getLabel();

	unsigned int plotnum = m_plots.size() + 1;
    auto plot = graph->plots()->create<XPlotWrapper>(tr, formatString("Plot%u",
        plotnum).c_str(), true, ref(tr), graph);
    if( !plot) return false; //transaction has faield.
    plot->m_parent = static_pointer_cast<XWaveNGraph>(node().shared_from_this());
    plot->m_colx = x;
    plot->m_coly1 = y1;
    plot->m_coly2 = y2;
    plot->m_colweight = weight;
    plot->m_colz = z;

    tr[ *plot->label()] = label;
    const XNode::NodeList &axes_list( *tr.list(graph->axes()));
	m_axisx = static_pointer_cast<XAxis>(axes_list.at(0));
	m_axisy = static_pointer_cast<XAxis>(axes_list.at(1));
    tr[ *plot->axisX()] = m_axisx;
    tr[ *m_axisx->label()] = m_labels[plot->m_colx];
    if(plot->m_coly1 >= 0) {
        tr[ *plot->axisY()] = m_axisy;
        tr[ *m_axisy->label()] = m_labels[plot->m_coly1];
	}
    tr[ *plot->maxCount()] = rowCount();
    tr[ *plot->maxCount()].setUIEnabled(false);
    tr[ *plot->clearPoints()].setUIEnabled(false);
    tr[ *plot->intensity()] = 1.0;
//	if(m_plots.size()) {
//        tr()[ *plot.xyplot->pointColor()] = clAqua; //Green;
//        tr()[ *plot.xyplot->lineColor()] = clAqua; //Green;
//        tr()[ *plot.xyplot->barColor()] = clAqua; //Green;
//		tr()[ *plot.xyplot->displayMajorGrid()] = false;
//	}

    if(plot->m_colz >= 0) {
		if( !m_axisz) {
            m_axisz = graph->axes()->create<XAxis>(tr, "Z Axis", true,
                XAxis::AxisDirection::Z, true, ref(tr), graph);
		}
        if( !m_axisz) return false; //transaction has failed.
        tr[ *plot->axisZ()] = m_axisz;
        tr[ *m_axisz->label()] = m_labels[plot->m_colz];
	}
    if(plot->m_colweight >= 0) {
		if( !m_axisw) {
            m_axisw = graph->axes()->create<XAxis>(tr, "Weight", true,
                XAxis::AxisDirection::Weight, true, ref(tr), graph);
		}
        if( !m_axisw) return false; //transaction has failed.
        tr[ *m_axisw->autoScale()] = false;
        tr[ *m_axisw->autoScale()].setUIEnabled(false);
        tr[ *plot->axisW()] = m_axisw;
        tr[ *m_axisw->label()] = m_labels[plot->m_colweight];
	}
    if(plot->m_coly2 >= 0) {
		if( !m_axisy2) {
            m_axisy2 = graph->axes()->create<XAxis>(tr, "Y2 Axis", true,
                XAxis::AxisDirection::Y, true, ref(tr), graph);
		}
        if( !m_axisy2) return false; //transaction has failed.
        tr[ *plot->axisY()] = m_axisy2;
        tr[ *m_axisy2->label()] = m_labels[plot->m_coly2];
	}

	m_plots.push_back(plot);

    graph->applyTheme(tr, true);

    auto &wave{static_cast<XWaveNGraph&>(node())};
    if(auto meas = wave.m_meas.lock())
        if(auto driver = wave.m_driver.lock()) {
            m_toolLists.push_back(wave.create<XGraph1DMathToolList>(tr,
                label.c_str(), false, meas, driver, plot));
            if( !m_toolLists.back()) return false; //transaction has failed.
            m_conTools = std::make_shared<XQGraph1DMathToolConnector>(m_toolLists, wave.m_btnMathTool, wave.m_graphwidget);
        }
    return true;
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
XWaveNGraph::dumpToFileThreaded(std::fstream &stream, const Snapshot &shot, const std::string &ext) {
    int rowcnt = shot[ *this].rowCount();
    int colcnt = shot[ *this].colCount();

    stream << "#" << getLabel() << std::endl;
    stream << "#";
    for(unsigned int i = 0; i < colcnt; i++) {
        stream << shot[ *this].labels()[i] << KAME_DATAFILE_DELIMITER;
    }
//		stream << std::endl;
    stream << "#at " << (XTime::now()).getTimeFmtStr(
        "%Y/%m/%d %H:%M:%S") << std::endl;

    auto &p(shot[ *this]);
    shared_ptr<Payload::ColumnBase> colw;
    if(p.m_colw >= 0) colw = p.m_cols[p.m_colw];
    int written_rows = 0;
    for(unsigned int i = 0; i < rowcnt; i++) {
        if( !colw || (colw->moreThanZero(i))) {
            for(unsigned int j = 0; j < colcnt; j++) {
                stream << std::setprecision(p.m_cols[j]->precision);
                p.m_cols[j]->toOFStream(stream, i);
                stream << KAME_DATAFILE_DELIMITER;
            }
            stream << std::endl;
            written_rows++;
        }
    }
    stream << std::endl;
    gMessagePrint(formatString_tr(I18N_NOOP("Succesfully %d rows written into %s."), written_rows, shot[ *filename()].to_str().c_str()));
}
void XWaveNGraph::drawGraph(Transaction &tr) {
	const Snapshot &shot(tr);
    if(shot[ *this].m_colw >= 0) {
        XGraph::VFloat weight_max = shot[ *this].m_cols[shot[ *this].m_colw]->max();
        tr[ *shot[ *this].axisw()->maxValue()] = weight_max;
        tr[ *shot[ *this].axisw()->minValue()] =  -0.4 * weight_max;
    }
    tr.mark(tr[ *graph()].onUpdate(), graph().get());

    int cnt = shot[ *this].m_plots.size();
    cnt = std::min(cnt, (int)shot[ *this].m_toolLists.size());
    for(unsigned int j = 0; j < cnt; j++) {
        auto plot = shot[ *this].m_plots[j];
        std::vector<XGraph::VFloat> colx, coly;
        auto colxcref = shot[ *this].m_cols[plot->m_colx]->fillOrPointToGraphPoints(colx);
        if(std::max(plot->m_coly1, plot->m_coly2) > 0) {
            auto colycref =
                shot[ *this].m_cols[(plot->m_coly1 > 0) ? plot->m_coly1 : plot->m_coly2]->fillOrPointToGraphPoints(coly);
            shot[ *this].m_toolLists[j]->update(tr, m_graphwidget, colxcref.begin(), colxcref.end(), colycref.begin(), colycref.end());
        }
    }
}
