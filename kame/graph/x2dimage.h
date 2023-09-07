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

//! Graph widget with internal data sets. The data can be saved as a text file.
//! \sa XQGraph, XGraph
class DECLSPEC_KAME X2DImage: public XGraphNToolBox {
public:
    X2DImage(const char *name, bool runtime, FrmGraphNURL *item);
    X2DImage(const char *name, bool runtime, XQGraph *graphwidget,
        QLineEdit *ed = nullptr, QAbstractButton *btn = nullptr, QPushButton *btndump = nullptr);
    virtual ~X2DImage() {}

    void setImage(Transaction &tr, const shared_ptr<QImage> &image);

    struct DECLSPEC_KAME Payload : public XGraphNToolBox::Payload {
    private:
        friend class X2DImage;
    };
protected:
    virtual void dumpToFileThreaded(std::fstream &) override;
private:
    shared_ptr<X2DImagePlot> m_plot;
};

#endif
