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
#include "x2dimage.h"
#include "xnodeconnector.h"
#include "ui_graphnurlform.h"
#include "graphwidget.h"
#include "graph.h"
#include <iomanip>
#include "graphmathtool.h"
#include <QToolButton>
#include "graphmathtoolconnector.h"
#include <QBuffer>

X2DImage::X2DImage(const char *name, bool runtime, FrmGraphNURL *item) :
    X2DImage(name, runtime, item->m_graphwidget, item->m_edUrl,
        item->m_btnUrl, item->m_btnDump) {

}
X2DImage::~X2DImage() {}

X2DImage::X2DImage(const char *name, bool runtime, XQGraph *graphwidget,
    QLineEdit *ed, QAbstractButton *btn, QPushButton *btndump,
    unsigned int max_color_index, QToolButton *btnmath,
    const shared_ptr<XMeasure> &meas, const shared_ptr<XDriver> &driver) :
    X2DImage(name, runtime, graphwidget, ed, btn, btndump) {
    m_btnMathTool = btnmath;
    for(unsigned int i = 0; i < max_color_index; ++i)
        m_toolLists.push_back(create<XGraph2DMathToolList>(formatString("CH%u", i).c_str(), false, meas, driver, plot()));

    m_conTools = std::make_unique<XQGraph2DMathToolConnector>(m_toolLists, m_btnMathTool, graphwidget);
}

X2DImage::X2DImage(const char *name, bool runtime, XQGraph *graphwidget,
    QLineEdit *ed, QAbstractButton *btn, QPushButton *btndump) : XGraphNToolBox(name, runtime, graphwidget, ed, btn, btndump),
    m_graphwidget(graphwidget) {
    iterate_commit([=](Transaction &tr){
        tr[ *graph()->label()] = getLabel();
        tr[ *graph()->persistence()] = 0;
        tr[ *graph()->drawLegends()] = false;
        auto plot = graph()->plots()->create<X2DImagePlot>(tr, "ImagePlot", true, ref(tr), graph());
        tr[ *plot->label()] = getLabel();
        m_plot = plot;
        const XNode::NodeList &axes_list( *tr.list(graph()->axes()));
        auto axisx = static_pointer_cast<XAxis>(axes_list.at(0));
        auto axisy = static_pointer_cast<XAxis>(axes_list.at(1));
        tr[ *plot->axisX()] = axisx;
        tr[ *axisx->label()] = "X";
        tr[ *plot->axisY()] = axisy;
        tr[ *axisy->label()] = "Y";
    });
}

void
X2DImage::dumpToFileThreaded(std::fstream &stream) {
    Snapshot shot( *plot());
    QByteArray ba;
    QBuffer buffer(&ba);
    buffer.open(QIODevice::WriteOnly);
    shot[ *plot()].image()->save( &buffer, "PNG");
    try {
        stream.write(ba.constData(), ba.size());
        gMessagePrint(formatString_tr(I18N_NOOP("Succesfully written into %s."), shot[ *filename()].to_str().c_str()));
    }
    catch(const std::ios_base::failure& e) {
        gErrPrint(e.what());
    }
}

void
X2DImage::updateImage(Transaction &tr, const shared_ptr<QImage> &image,
    const std::vector<const uint32_t *> &rawimages, unsigned int raw_stride, const std::vector<double> coefficients) {
    m_plot->setImage(tr, image);
    if(m_toolLists.size())
        for(unsigned int cidx = 0; cidx < rawimages.size(); ++cidx) {
            m_toolLists[cidx]->update(tr, m_graphwidget,
                rawimages[cidx], image->width(), raw_stride, image->height(), coefficients[cidx]);
        }
}
