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

//! Graph widget with internal data sets. The data can be saved as a text file.
//! \sa XQGraph, XGraph
class DECLSPEC_KAME X2DImage: public XGraphNToolBox {
public:
    X2DImage(const char *name, bool runtime, FrmGraphNURL *item);
    X2DImage(const char *name, bool runtime, XQGraph *graphwidget,
        QLineEdit *ed, QAbstractButton *btn, QPushButton *btndump,
        unsigned int max_color_index, QToolButton *m_btnmath, const shared_ptr<XMeasure> &meas, const shared_ptr<XDriver> &driver);
    X2DImage(const char *name, bool runtime, XQGraph *graphwidget,
        QLineEdit *ed = nullptr, QAbstractButton *btn = nullptr, QPushButton *btndump = nullptr);

    virtual ~X2DImage();

    void updateImage(Transaction &tr, const shared_ptr<QImage> &image,
        const std::vector<const uint32_t *> &rawimages = {}, const std::vector<double> coefficients = {});

    const shared_ptr<X2DImagePlot> &plot() const {return m_plot;}
protected:
    virtual void dumpToFileThreaded(std::fstream &) override;
private:
    shared_ptr<X2DImagePlot> m_plot;

    XQGraph *m_graphwidget;
    QToolButton *m_btnMathTool = nullptr;
    std::deque<shared_ptr<XGraph2DMathToolList>> m_toolLists;
    unique_ptr<XQGraph2DMathToolConnector> m_conTools;
};

#endif
