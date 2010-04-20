/***************************************************************************
		Copyright (C) 2002-2010 Kentaro Kitagawa
		                   kitag@issp.u-tokyo.ac.jp
		
		This program is free software; you can redistribute it and/or
		modify it under the terms of the GNU Library General Public
		License as published by the Free Software Foundation; either
		version 2 of the License, or (at your option) any later version.
		
		You should have received a copy of the GNU Library General 
		Public License and a list of authors along with this program; 
		see the files COPYING and AUTHORS.
***************************************************************************/
/***************************************************************************
                          icon  -  description
                             -------------------
    begin                : ? 11? 24 2004
    copyright            : (C) 2004 by Kentaro Kitagawa
    email                : kitag@issp.u-tokyo.ac.jp
***************************************************************************/

/***************************************************************************
 *                                                                         *
 *   This program is free software; you can redistribute it and/or modify  *
 *   it under the terms of the GNU General Public License as published by  *
 *   the Free Software Foundation; either version 2 of the License, or     *
 *   (at your option) any later version.                                   *
 *                                                                         *
 ***************************************************************************/
#ifndef ICON_H
#define ICON_H

class QPixmap;

extern QPixmap *g_pIconKame24x24;
extern QPixmap *g_pIconKame48x48;
extern QPixmap *g_pIconInfo;
extern QPixmap *g_pIconWarn;
extern QPixmap *g_pIconError;
extern QPixmap *g_pIconStop;
extern QPixmap *g_pIconClose;
extern QPixmap *g_pIconInterface;
extern QPixmap *g_pIconDriver;
extern QPixmap *g_pIconReader;
extern QPixmap *g_pIconScalar;
extern QPixmap *g_pIconGraph;
extern QPixmap *g_pIconScript;

class KIconLoader;

void makeIcons(KIconLoader *loader);

#endif
