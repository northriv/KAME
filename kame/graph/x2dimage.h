/***************************************************************************
        Copyright (C) 2002-2025 Kentaro Kitagawa
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

#ifndef x2dimageH
#define x2dimageH
//---------------------------------------------------------------------------

#include "xnodeconnector.h"
#include "graphntoolbox.h"

class XDriver;
class XMeasure;
class XGraph2DMathToolList;
class XQGraph2DMathToolConnector;
class QDoubleSpinBox;

//! Graph widget with internal data sets. The data can be saved as a text file.
//! \sa XQGraph, XGraph
class DECLSPEC_KAME X2DImage: public XGraphNToolBox {
public:
    X2DImage(const char *name, bool runtime, FrmGraphNURL *item, bool hascolorbar = false);
    X2DImage(const char *name, bool runtime, XQGraph *graphwidget,
        QLineEdit *ed, QAbstractButton *btn, QPushButton *btndump,
        unsigned int max_color_index, QDoubleSpinBox *dblgamma,
        QToolButton *m_btnmath, const shared_ptr<XMeasure> &meas, const shared_ptr<XDriver> &driver, bool hascolorbar = false);
    X2DImage(const char *name, bool runtime, XQGraph *graphwidget,
        QLineEdit *ed = nullptr, QAbstractButton *btn = nullptr, QPushButton *btndump = nullptr,
        QDoubleSpinBox *dblgamma = nullptr, bool hascolorbar = false);

    virtual ~X2DImage();

    const shared_ptr<XDoubleNode> &gamma() const {return m_gamma;}

    //! update internal high-colordepth images for math tools. updateQImage (and updateColorBarImage) should be performed asap.
    void updateRawImages(Transaction &tr, unsigned int width, unsigned int height,
                     const std::vector<const uint32_t *> &rawimages, unsigned int raw_stride = 0, const std::vector<double> coefficients = {}, const std::vector<double> offsets = {});
    void updateQImage(Transaction &tr, const shared_ptr<QImage> &image);
    void updateColorBarImage(Transaction &tr, double cmin, double cmax, const shared_ptr<QImage> &image);

    const shared_ptr<X2DImagePlot> &plot() const {return m_plot;}
    const shared_ptr<X2DImagePlot> &colorBarPlot() const {return m_colorbarplot;}
protected:
    virtual void dumpToFileThreaded(std::fstream &, const Snapshot &, const std::string &ext) override;
private:
    shared_ptr<X2DImagePlot> m_plot, m_colorbarplot;
    const shared_ptr<XDoubleNode> m_gamma;

    XQGraph *m_graphwidget;
    QToolButton *m_btnMathTool = nullptr;
    QDoubleSpinBox *m_dblGamma = nullptr;
    std::deque<xqcon_ptr> m_conUIs;
    std::deque<shared_ptr<XGraph2DMathToolList>> m_toolLists;
    unique_ptr<XQGraph2DMathToolConnector> m_conTools;
};

#endif
