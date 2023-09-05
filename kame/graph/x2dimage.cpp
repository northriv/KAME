/***************************************************************************
        Copyright (C) 2002-2023 Kentaro Kitagawa
		                   kitagawa@phys.s.u-tokyo.ac.jp
		
		This program is free software; you can redistribute it and/or
		modify it under the terms of the GNU Library General Public
		License as published by the Free Software Foundation; either
		version 2 of the License, or (at your option) any later version.
		
		You should have received a copy of the GNU Library General 
		Public License and a list of authors along with this program; 
		see the files COPYING and AUTHORS.
 ***************************************************************************/
#include "x2dimage.h"

#include "ui_graphnurlform.h"
#include "graphwidget.h"
#include "graph.h"
#include <iomanip>

X2DImage::X2DImage(const char *name, bool runtime, FrmGraphNURL *item) :
    X2DImage(name, runtime, item->m_graphwidget, item->m_edUrl,
        item->m_btnUrl, item->m_btnDump) {

}
X2DImage::X2DImage(const char *name, bool runtime, XQGraph *graphwidget,
    QLineEdit *ed, QAbstractButton *btn, QPushButton *btndump) : XGraphNToolBox(name, runtime, graphwidget, ed, btn, btndump) {
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
    Snapshot shot( *this);

}

void
X2DImage::setImage(Transaction &tr, const shared_ptr<QImage> &image) {
    m_plot->setImage(tr, image);
}
