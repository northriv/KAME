/***************************************************************************
        Copyright (C) 2002-2014 Kentaro Kitagawa
                           kitag@kochi-u.ac.jp

        This program is free software; you can redistribute it and/or
        modify it under the terms of the GNU Library General Public
        License as published by the Free Software Foundation; either
        version 2 of the License, or (at your option) any later version.

        You should have received a copy of the GNU Library General
        Public License and a list of authors along with this program;
        see the files COPYING and AUTHORS.
 ***************************************************************************/

#ifndef MESSAGEBOX_H
#define MESSAGEBOX_H

#include "support.h"
#include <QObject>

class QIcon;
class QWidget;

class XMessageBox : public QObject {
    Q_OBJECT
public:
    XMessageBox(QWidget *parent);
    static QWidget *form();
    static void post(XString msg, const QIcon &icon, bool popup = false, int duration_ms = 0);
protected slots:
    void hide();
private:
};

#endif // MESSAGEBOX_H
