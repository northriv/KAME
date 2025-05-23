/***************************************************************************
        Copyright (C) 2002-2014 Kentaro Kitagawa
                           kitag@issp.u-tokyo.ac.jp

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
#include <QWindow>
#include <QScreen>
#include <QTimer>

#include "ui_messageform.h"

typedef QForm<QWidget, Ui_FrmMessage> FrmMessage;
static FrmMessage *s_pFrmMessage = 0L;
static QTimer *s_timer = 0L;

XMessageBox::XMessageBox(QWidget *parent) {
    s_pFrmMessage = new FrmMessage((QWidget*)0, Qt::Tool |
#if defined __MACOSX__ || defined __APPLE__
        Qt::WindowStaysOnBottomHint  |   //not working with windows10
#endif
        Qt::CustomizeWindowHint | Qt::WindowTitleHint | Qt::WindowMinimizeButtonHint);
    s_pFrmMessage->show();

    s_pFrmMessage->m_widget->hide();

    QRect rect = s_pFrmMessage->window()->windowHandle()->screen()->availableGeometry();
    int y = rect.bottom() - s_pFrmMessage->frameSize().height();
    s_pFrmMessage->move(0, y);
//    s_pFrmMessage->m_list->setMouseTracking(true); //for statusTip.

    {
        QFont font(s_pFrmMessage->font());
        font.setPointSize(font.pointSize() * 4 / 5);
        s_pFrmMessage->m_list->setFont(font);
    }
    {
        QFont font(s_pFrmMessage->font());
        font.setPointSize(font.pointSize() * 5 / 6);
        s_pFrmMessage->m_label->setFont(font);
    }

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
    s_pFrmMessage->m_widget->hide();
}
void
XMessageBox::post(XString msg, const QIcon &icon, bool popup, int duration_ms, XString tooltip) {
    if( !msg.length()) {
        s_pFrmMessage->m_widget->hide();
        return;
    }
    if(popup) {
        s_pFrmMessage->m_label->setText(msg);
        s_pFrmMessage->m_label->repaint();
        s_pFrmMessage->m_btn->setIcon(icon);
        if(duration_ms) {
            static XTime last_raise = {};
            s_pFrmMessage->m_widget->show();
            s_pFrmMessage->showNormal();
            if(XTime::now().diff_msec(last_raise) > std::min(20000, duration_ms)) {
                last_raise = XTime::now(); //surpress frequent raise.
                s_pFrmMessage->raise();
            }
        }
        else {
            s_pFrmMessage->m_widget->hide();
        }
        s_timer->stop();
        if(duration_ms > 0) {
            s_timer->setInterval(duration_ms);
            s_timer->start();
        }

    }

    msg = XTime::now().getTimeFmtStr("%H:%M:%S ", false) + msg;
    QListWidgetItem *item;
    item = new QListWidgetItem(icon, msg, s_pFrmMessage->m_list);
    item->setToolTip(tooltip);
    if(s_pFrmMessage->m_list->count() > 100)
        s_pFrmMessage->m_list->takeItem(0);
    s_pFrmMessage->m_list->scrollToBottom();
}
