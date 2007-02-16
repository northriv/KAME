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
//---------------------------------------------------------------------------
#include "recordreaderform.h"
#include "recordreaderconnector.h"
#include "recordreader.h"
#include <qpushbutton.h>
#include <kiconloader.h>
#include <kapplication.h>
#include <klocale.h>

XRawStreamRecordReaderConnector::XRawStreamRecordReaderConnector(
    const shared_ptr<XRawStreamRecordReader> &reader, FrmRecordReader *form) :
  XQConnector(reader, form),
  m_reader(reader),
  m_pForm(form),
  m_conRecordFile(xqcon_create<XKURLReqConnector>(
                    reader->filename(), form->urlBinRec,
                    ("*.bin|Binary files (*.bin)\n*.*|All files (*.*)"), false)),
  m_conFF(xqcon_create<XQToggleButtonConnector>(reader->fastForward(), form->btnFF)),
  m_conRW(xqcon_create<XQToggleButtonConnector>(reader->rewind(), form->btnRW)),
  m_conStop(xqcon_create<XQButtonConnector>(reader->stop(), form->btnStop)),
  m_conFirst(xqcon_create<XQButtonConnector>(reader->first(), form->btnFirst)),
  m_conNext(xqcon_create<XQButtonConnector>(reader->next(), form->btnNext)),
  m_conBack(xqcon_create<XQButtonConnector>(reader->back(), form->btnBack)),
  m_conPosString(xqcon_create<XQLineEditConnector>(reader->posString(), form->edTime)),
  m_conSpeed(xqcon_create<XQComboBoxConnector>(reader->speed(), form->cmbSpeed))
{
    KApplication *app = KApplication::kApplication();
    form->btnNext->setIconSet( app->iconLoader()->loadIconSet("forward", 
            KIcon::Toolbar, KIcon::SizeSmall, true ) );
    form->btnBack->setIconSet( app->iconLoader()->loadIconSet("previous", 
            KIcon::Toolbar, KIcon::SizeSmall, true ) );
    form->btnFF->setIconSet( app->iconLoader()->loadIconSet("player_fwd", 
            KIcon::Toolbar, KIcon::SizeSmall, true ) );
    form->btnRW->setIconSet( app->iconLoader()->loadIconSet("player_rew", 
            KIcon::Toolbar, KIcon::SizeSmall, true ) );
    form->btnFirst->setIconSet( app->iconLoader()->loadIconSet("player_start", 
            KIcon::Toolbar, KIcon::SizeSmall, true ) );
    form->btnStop->setIconSet( app->iconLoader()->loadIconSet("player_stop", 
            KIcon::Toolbar, KIcon::SizeSmall, true ) );
}
