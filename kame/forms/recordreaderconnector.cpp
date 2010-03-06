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
//---------------------------------------------------------------------------
#include "recordreaderconnector.h"
#include "recordreader.h"
#include <qpushbutton.h>
#include <kiconloader.h>
#include "ui_recordreaderform.h"

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
	m_conSpeed(xqcon_create<XQComboBoxConnector>(reader->speed(), form->cmbSpeed, Snapshot( *reader->speed()))) {

    KIconLoader *loader = KIconLoader::global();
    form->btnNext->setIcon( loader->loadIcon("forward",
															  KIconLoader::Toolbar, KIconLoader::SizeSmall, true ) );
    form->btnBack->setIcon( loader->loadIcon("previous",
															  KIconLoader::Toolbar, KIconLoader::SizeSmall, true ) );
    form->btnFF->setIcon( loader->loadIcon("player_fwd",
															KIconLoader::Toolbar, KIconLoader::SizeSmall, true ) );
    form->btnRW->setIcon( loader->loadIcon("player_rew",
															KIconLoader::Toolbar, KIconLoader::SizeSmall, true ) );
    form->btnFirst->setIcon( loader->loadIcon("player_start",
															   KIconLoader::Toolbar, KIconLoader::SizeSmall, true ) );
    form->btnStop->setIcon( loader->loadIcon("player_stop",
															  KIconLoader::Toolbar, KIconLoader::SizeSmall, true ) );
}
