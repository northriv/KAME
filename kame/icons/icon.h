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
#ifndef ICON_H
#define ICON_H

class QPixmap;

extern DECLSPEC_KAME QPixmap *g_pIconKame24x24;
extern DECLSPEC_KAME QPixmap *g_pIconKame;
extern DECLSPEC_KAME QPixmap *g_pIconInfo;
extern DECLSPEC_KAME QPixmap *g_pIconWarn;
extern DECLSPEC_KAME QPixmap *g_pIconError;
extern DECLSPEC_KAME QPixmap *g_pIconStop;
extern DECLSPEC_KAME QPixmap *g_pIconClose;
extern DECLSPEC_KAME QPixmap *g_pIconInterface;
extern DECLSPEC_KAME QPixmap *g_pIconDriver;
extern DECLSPEC_KAME QPixmap *g_pIconReader;
extern DECLSPEC_KAME QPixmap *g_pIconScalar;
extern DECLSPEC_KAME QPixmap *g_pIconGraph;
extern DECLSPEC_KAME QPixmap *g_pIconScript;
extern DECLSPEC_KAME QPixmap *g_pIconRoverT;
extern DECLSPEC_KAME QPixmap *g_pIconLEDOn;
extern DECLSPEC_KAME QPixmap *g_pIconLEDOff;

void makeIcons();

#endif
