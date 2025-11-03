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
#include <QColorSpace>

X2DImage::X2DImage(const char *name, bool runtime, FrmGraphNURL *item, bool hascolorbar) :
    X2DImage(name, runtime, item->m_graphwidget, item->m_edUrl,
        item->m_btnUrl, item->m_btnDump, nullptr, hascolorbar) {

}
X2DImage::~X2DImage() {}

X2DImage::X2DImage(const char *name, bool runtime, XQGraph *graphwidget,
    QLineEdit *ed, QAbstractButton *btn, QPushButton *btndump,
    unsigned int max_color_index, QDoubleSpinBox *dblgamma, QToolButton *btnmath,
    const shared_ptr<XMeasure> &meas, const shared_ptr<XDriver> &driver, bool hascolorbar) :
    X2DImage(name, runtime, graphwidget, ed, btn, btndump, dblgamma, hascolorbar) {
    m_btnMathTool = btnmath;
    for(unsigned int i = 0; i < max_color_index; ++i)
        m_toolLists.push_back(create<XGraph2DMathToolList>(formatString("CH%u", i).c_str(), false, meas, driver, plot()));

    m_conTools = std::make_unique<XQGraph2DMathToolConnector>(m_toolLists, m_btnMathTool, graphwidget);
}

X2DImage::X2DImage(const char *name, bool runtime, XQGraph *graphwidget,
    QLineEdit *ed, QAbstractButton *btn, QPushButton *btndump, QDoubleSpinBox *dblgamma, bool hascolorbar) :
    XGraphNToolBox(name, runtime, graphwidget, ed, btn, btndump,
        "Images (*.png *.jpg *.jpeg);;Data files (*.dat);;All files (*.*)"),
    m_gamma(create<XDoubleNode>("Gamma", false)),
    m_graphwidget(graphwidget),
    m_dblGamma(dblgamma) {
    iterate_commit([=](Transaction &tr){
        tr[ *graph()->label()] = getLabel();
        tr[ *graph()->persistence()] = 0;
        tr[ *graph()->drawLegends()] = false;
        auto plot = graph()->plots()->create<X2DImagePlot>(tr, "ImagePlot", true, ref(tr), graph());
        if( !plot) return; //transaction has failed.
        tr[ *plot->label()] = getLabel();
        m_plot = plot;
        const XNode::NodeList &axes_list( *tr.list(graph()->axes()));
        auto axisx = static_pointer_cast<XAxis>(axes_list.at(0));
        auto axisy = static_pointer_cast<XAxis>(axes_list.at(1)); //new yaxis
        tr[ *plot->axisX()] = axisx;
        tr[ *axisx->label()] = "X";
        tr[ *plot->axisY()] = axisy;
        tr[ *axisy->label()] = "Y";
        tr[ *plot->keepXYAspectRatioToOne()] = true;
        tr[ *axisx->invisible()] = true;
        tr[ *axisy->invisible()] = true;
        tr[ *axisy->invertAxis()] = true;
        tr[ *axisx->x()] = 0.02; //0.15 was default
        tr[ *axisx->y()] = 0.055; //0.15 was default
        tr[ *axisx->length()] = 1.0 - tr[ *axisx->x()] * 2; //0.7 was default
        tr[ *axisy->x()] = (double)tr[ *axisx->x()];
        tr[ *axisy->y()] = (double)tr[ *axisx->y()];
        tr[ *axisy->length()] = 1.0 - tr[ *axisx->y()] * 2; //0.7 was default
        tr[ *axisx->marginDuringAutoScale()] = 0.0;
        tr[ *axisy->marginDuringAutoScale()] = 0.0;

        tr[ *gamma()] = 2.2;

        if(hascolorbar) {
        //Colorbar
            auto cplot = graph()->plots()->create<XColorBarImagePlot>(tr, "ColorBar", true, ref(tr), graph());
            if( !cplot) return; //transaction has failed.
            tr[ *cplot->label()] = getLabel() + "-ColorBar";
            m_colorbarplot = cplot;
            auto axisc = graph()->axes()->create<XAxis>(tr, "ColorAxis", true, XAxis::AxisDirection::X, false, ref(tr), graph());
            auto axisc2 = graph()->axes()->create<XAxis>(tr, "ColorBarAxis", true, XAxis::AxisDirection::Y, false, ref(tr), graph());
            tr[ *cplot->axisX()] = axisc;
            tr[ *axisc->label()] = "Color";
            tr[ *cplot->axisY()] = axisc2;
            tr[ *axisc2->label()] = "CY";
            tr[ *axisc->autoScale()] = false;
            tr[ *axisc->autoScale()].setUIEnabled(false);
            tr[ *axisc->maxValue()].setUIEnabled(false);
            tr[ *axisc->minValue()].setUIEnabled(false);
            tr[ *axisc2->autoScale()].setUIEnabled(false);
            tr[ *axisc2->maxValue()].setUIEnabled(false);
            tr[ *axisc2->minValue()].setUIEnabled(false);
            tr[ *axisc->invisible()] = false;
            tr[ *axisc->displayLabel()] = false;
            tr[ *axisc->rightOrTopSided()] = true;
            tr[ *axisc2->invisible()] = true;
            tr[ *axisc2->length()] = 0.03;
            tr[ *axisy->length()] = (double)tr[ *axisy->length()] - tr[ *axisc2->length()] - 0.05;
            tr[ *axisc2->x()] = (double)tr[ *axisx->x()];
            tr[ *axisc2->y()] = (double)tr[ *axisy->y()] + tr[ *axisy->length()] + 0.01;
            tr[ *axisc->length()] = (double)tr[ *axisx->length()];
            tr[ *axisc->x()] = (double)tr[ *axisc2->x()];
            tr[ *axisc->y()] = (double)tr[ *axisc2->y()] + tr[ *axisc2->length()];
        }
        graph()->applyTheme(tr, true);
    });
    m_conUIs = {
        xqcon_create<XQDoubleSpinBoxConnector>(gamma(), m_dblGamma),
    };
}

void
X2DImage::dumpToFileThreaded(std::fstream &stream, const Snapshot &shot, const std::string &ext) {
    if((ext == "DAT") || (ext == "dat")) {
        auto image = shot[ *plot()].image();
        stream << "#at " << (XTime::now()).getTimeFmtStr(
            "%Y/%m/%d %H:%M:%S") << std::endl;
        stream << image->width() << KAME_DATAFILE_DELIMITER
            << image->height() << std::endl
            << image->depth() << std::endl;

        const uint16_t* p = reinterpret_cast<const uint16_t*>(image->constBits());
        const uint16_t* p_start = p;
        for(unsigned int i = 0; i < image->height(); ++i) {
            for(unsigned int j = 0; j < image->width(); ++j) {
                stream << *p++;
                stream << KAME_DATAFILE_DELIMITER;
                stream << *p++;
                stream << KAME_DATAFILE_DELIMITER;
                stream << *p++;
                stream << KAME_DATAFILE_DELIMITER;
                p++; //alpha channel
            }
            stream << std::endl;
        }
        gMessagePrint(formatString_tr(I18N_NOOP("Succesfully %ld words written into %s."), p - p_start, shot[ *filename()].to_str().c_str()));
    }
    else {
        QByteArray ba;
        QBuffer buffer(&ba);
        buffer.open(QIODevice::WriteOnly);
        {
            const auto &image_org = shot[ *plot()].image();
            QImage image;
            if(m_colorbarplot) {
                const auto &barimage_org = shot[ *m_colorbarplot].image();
                if(image_org->bytesPerLine() == barimage_org->bytesPerLine()) {
                    int blanks = 6, barwidth = 32;
                    QImage barimage = barimage_org->scaled(image_org->width(), barwidth);
                    image = QImage(image_org->width(),
                        image_org->height() + blanks + barimage.height(), image_org->format()); //scaled copy to add a colorbar.
                    //writes original image.
                    std::copy(image_org->bits(), image_org->bits() + image_org->bytesPerLine() * image_org->height(), image.bits());
                    //writes colorbar pixels.
                    uchar *bits = image.bits() + image.bytesPerLine() * (image_org->height() + blanks);
                    std::copy(barimage.bits(), barimage.bits() + barimage.bytesPerLine() * barimage.height(), bits);
                }
                else
                    image = image_org->copy();
            }
            else
                image = image_org->copy();

        //    image.setColorSpace(QColorSpace::SRgbLinear);
            if(shot[ *m_gamma] == 2.2)
                image.convertToColorSpace(QColorSpace::SRgb);
            else if(shot[ *m_gamma] == 1.0)
                image.convertToColorSpace(QColorSpace::SRgbLinear);
            else
                image.convertToColorSpace(QColorSpace{QColorSpace::Primaries::SRgb, (float)(double)shot[ *m_gamma]});
            image.save( &buffer, ext.c_str(), 100); //uncompressed full quality.
        }
        try {
            stream.write(ba.constData(), ba.size());
            gMessagePrint(formatString_tr(I18N_NOOP("Succesfully written into %s."), shot[ *filename()].to_str().c_str()));
        }
        catch(const std::ios_base::failure& e) {
            gErrPrint(e.what());
        }
    }
}

void
X2DImage::updateRawImages(Transaction &tr,unsigned int width, unsigned int height,
                      const std::vector<const uint32_t *> &rawimages, unsigned int raw_stride, const std::vector<double> coefficients_given, const std::vector<double> offsets_given) {
    auto coeffs = coefficients_given;
    auto offsets = offsets_given;
    if(coeffs.empty())
        coeffs.resize(rawimages.size(), 1.0);
    if(offsets.empty())
        offsets.resize(rawimages.size(), 0.0);
    for(unsigned int cidx = 0; cidx < std::min(m_toolLists.size(), rawimages.size()); ++cidx) {
        m_toolLists[cidx]->update(tr, m_graphwidget,
                                  rawimages[cidx], width, raw_stride, height, coeffs[cidx], offsets[cidx]);
    }
}
void
X2DImage::updateQImage(Transaction &tr, const shared_ptr<QImage> &image) {
    m_plot->setImage(tr, image);
}
void
X2DImage::updateColorBarImage(Transaction &tr, double cmin, double cmax, const shared_ptr<QImage> &image) {
    m_colorbarplot->setImage(tr, image);
    shared_ptr<XAxis> axis = tr[ *m_colorbarplot->axisX()];
    auto axisc = static_pointer_cast<XAxis>(axis);
    tr[ *axisc->maxValue()] = cmax;
    tr[ *axisc->minValue()] = cmin;
}
