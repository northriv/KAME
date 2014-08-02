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

#include "messagebox.h"
#include "xnodeconnector.h"
#include <QMainWindow>

#include "ui_messageform.h"

typedef QForm<QMainWindow, Ui_FrmMessage> FrmMessage;
static FrmMessage *s_pFrmMessage = 0L;

XMessageBox::XMessageBox(QWidget *parent) {
    s_pFrmMessage = new FrmMessage(parent, Qt::Tool | Qt::WindowStaysOnBottomHint);
    s_pFrmMessage->show();

    s_pFrmMessage->m_btn->hide();

}

QWidget*
XMessageBox::form() {
    return s_pFrmMessage;
}
void
XMessageBox::post(XString msg, const QIcon &icon, int duration_ms, bool popup) {
    if( !icon.isNull())
       new QListWidgetItem(icon, msg, s_pFrmMessage->m_list);
    else
       s_pFrmMessage->m_list->addItem(msg);

    if(popup) {
        s_pFrmMessage->m_btn->setText(msg);
        s_pFrmMessage->m_btn->setIcon(icon);
        s_pFrmMessage->m_btn->show();
    }
//    else {
//        s_pFrmMessage->m_btn->hide();
//    }
}
