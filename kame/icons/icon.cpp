/***************************************************************************
		Copyright (C) 2002-2013 Kentaro Kitagawa
		                   kitag@kochi-u.ac.jp
		
		This program is free software; you can redistribute it and/or
		modify it under the terms of the GNU Library General Public
		License as published by the Free Software Foundation; either
		version 2 of the License, or (at your option) any later version.
		
		You should have received a copy of the GNU Library General 
		Public License and a list of authors along with this program; 
		see the files COPYING and AUTHORS.
***************************************************************************/
#include "icon.h"
#include <QApplication>
#include <QPixmap>
#include <QStyle>
#include <QIcon>

//#include "kame-24x24-png.c"
extern const unsigned char icon_kame_24x24_png[1065];

QPixmap *g_pIconKame24x24;
QPixmap *g_pIconKame;
QPixmap *g_pIconWarn;
QPixmap *g_pIconError;
QPixmap *g_pIconInfo;
QPixmap *g_pIconStop;
QPixmap *g_pIconClose;
QPixmap *g_pIconInterface;
QPixmap *g_pIconDriver;
QPixmap *g_pIconReader;
QPixmap *g_pIconScalar;
QPixmap *g_pIconGraph;
QPixmap *g_pIconScript;
QPixmap *g_pIconRoverT;
QPixmap *g_pIconLEDOn;
QPixmap *g_pIconLEDOff;

void makeIcons()
{
	g_pIconKame24x24 = new QPixmap;
    g_pIconKame24x24->loadFromData( icon_kame_24x24_png, sizeof( icon_kame_24x24_png ), "PNG" );
	
    g_pIconKame = new QPixmap(":/icons/kame.png");

    g_pIconRoverT = new QPixmap(":/icons/rovert.png");

    g_pIconLEDOn = new QPixmap(":/icons/ledon.png");

    g_pIconLEDOff = new QPixmap(":/icons/ledoff.png");

    g_pIconInfo = new QPixmap(QApplication::style()->standardIcon(QStyle::SP_MessageBoxInformation).pixmap(48,48));
	
    g_pIconWarn = new QPixmap(QApplication::style()->standardIcon(QStyle::SP_MessageBoxWarning).pixmap(48,48));

    g_pIconError = new QPixmap(QApplication::style()->standardIcon(QStyle::SP_MessageBoxCritical).pixmap(48,48));

    g_pIconStop = new QPixmap(QApplication::style()->standardIcon(QStyle::SP_BrowserStop).pixmap(48,48));

    g_pIconClose = new QPixmap(QApplication::style()->standardIcon(QStyle::SP_TitleBarCloseButton).pixmap(48,48));

    g_pIconDriver = g_pIconKame;

    g_pIconInterface = new QPixmap(QApplication::style()->standardIcon(QStyle::SP_ComputerIcon).pixmap(48,48));

    g_pIconReader = new QPixmap(QApplication::style()->standardIcon(QStyle::SP_MediaPlay).pixmap(48,48));

    g_pIconScalar = new QPixmap(QApplication::style()->standardIcon(QStyle::SP_FileDialogDetailedView).pixmap(48,48));

    g_pIconGraph = new QPixmap(":/icons/graph.png");;

    g_pIconScript = new QPixmap(":/icons/ruby.png");
}
