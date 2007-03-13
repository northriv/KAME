/***************************************************************************
		Copyright (C) 2002-2007 Kentaro Kitagawa
		                   kitagawa@scphys.kyoto-u.ac.jp
		
		This program is free software; you can redistribute it and/or
		modify it under the terms of the GNU Library General Public
		License as published by the Free Software Foundation; either
		version 2 of the License, or (at your option) any later version.
		
		You should have received a copy of the GNU Library General 
		Public License and a list of authors along with this program; 
		see the files COPYING and AUTHORS.
***************************************************************************/
#include "icon.h"
#include <qpixmap.h>
#include <kiconloader.h>

#include "kame-24x24-png.c"

QPixmap *g_pIconKame24x24;
QPixmap *g_pIconKame48x48;
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

void makeIcons(KIconLoader *loader)
{
	g_pIconKame24x24 = new QPixmap;
	g_pIconKame24x24->loadFromData( icon_kame_24x24_png, sizeof( icon_kame_24x24_png ), "PNG" );
	
	g_pIconKame48x48 = new QPixmap(
		loader->loadIcon("kame", KIcon::Toolbar, 0, KIcon::DefaultState, 0, true ));
	if(g_pIconKame48x48->isNull() ) g_pIconKame48x48 = g_pIconKame24x24;

	g_pIconInfo = new QPixmap(
		loader->loadIcon("messagebox_info", KIcon::Toolbar, 0, KIcon::DefaultState, 0, false ));
	
	g_pIconWarn = new QPixmap(
		loader->loadIcon("messagebox_warning", KIcon::Toolbar, 0, KIcon::DefaultState, 0, false ));
	
	g_pIconError = new QPixmap(
		loader->loadIcon("messagebox_critical", KIcon::Toolbar, 0, KIcon::DefaultState, 0, false ));
	
	g_pIconStop = new QPixmap(
		loader->loadIcon("stop", KIcon::Toolbar, 0, KIcon::DefaultState, 0, false ));
	
	g_pIconClose = new QPixmap(
		loader->loadIcon("fileclose", KIcon::Toolbar, 0, KIcon::DefaultState, 0, false ));
	
	g_pIconDriver = new QPixmap(
		loader->loadIcon("exec", KIcon::Toolbar, 0, KIcon::DefaultState, 0, false ));
	
	g_pIconInterface = new QPixmap(
		loader->loadIcon("mouse", KIcon::Toolbar, 0, KIcon::DefaultState, 0, false ));
	
	g_pIconReader = new QPixmap(
		loader->loadIcon("player_play", KIcon::Toolbar, 0, KIcon::DefaultState, 0, false ));
	
	g_pIconScalar = new QPixmap(
		loader->loadIcon("math_abs", KIcon::Toolbar, 0, KIcon::DefaultState, 0, false ));
	
	g_pIconGraph = new QPixmap(
		loader->loadIcon("graph", KIcon::Toolbar, 0, KIcon::DefaultState, 0, false ));
	
	g_pIconScript = new QPixmap(
		loader->loadIcon("ruby", KIcon::Toolbar, 0, KIcon::DefaultState, 0, false ));
}
