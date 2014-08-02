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
#include <QApplication>
#include <QDesktopWidget>
#include <QTimer>

#include "ui_messageform.h"

typedef QForm<QMainWindow, Ui_FrmMessage> FrmMessage;
static FrmMessage *s_pFrmMessage = 0L;
static QTimer *s_timer = 0L;

XMessageBox::XMessageBox(QWidget *parent) {
    s_pFrmMessage = new FrmMessage(parent, Qt::Tool | Qt::WindowStaysOnBottomHint);
    s_pFrmMessage->show();

    s_pFrmMessage->m_btn->hide();

    QRect rect = QApplication::desktop()->availableGeometry(s_pFrmMessage);
    s_pFrmMessage->move(0, rect.bottom() - s_pFrmMessage->height());

    s_timer = new QTimer(this);
    connect(s_timer, SIGNAL(timeout()), this, SLOT(hide()));
    s_timer->setSingleShot(true);
}

QWidget*
XMessageBox::form() {
    return s_pFrmMessage;
}
void
XMessageBox::hide() {
    s_pFrmMessage->m_btn->hide();
}
void
XMessageBox::post(XString msg, const QIcon &icon, bool popup, int duration_ms) {
    if(popup) {
        s_pFrmMessage->m_btn->setText(msg);
        s_pFrmMessage->m_btn->setIcon(icon);
        s_pFrmMessage->m_btn->show();
        s_timer->stop();
        if(duration_ms) {
            s_timer->setInterval(duration_ms);
            s_timer->start();
        }

    }
//    else {
//        s_pFrmMessage->m_btn->hide();
//    }

    msg = XTime::now().getTimeFmtStr("%H:%M:%S ", false) + msg;
    if( !icon.isNull())
       new QListWidgetItem(icon, msg, s_pFrmMessage->m_list);
    else
       s_pFrmMessage->m_list->addItem(msg);
    if(s_pFrmMessage->m_list->count() > 100)
        s_pFrmMessage->m_list->takeItem(0);
    s_pFrmMessage->m_list->scrollToBottom();
}
