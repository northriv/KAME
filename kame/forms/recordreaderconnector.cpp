/***************************************************************************
		Copyright (C) 2002-2015 Kentaro Kitagawa
		                   kitagawa@phys.s.u-tokyo.ac.jp
		
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
#include <QPushButton>

#include "ui_recordreaderform.h"

XRawStreamRecordReaderConnector::XRawStreamRecordReaderConnector(
    const shared_ptr<XRawStreamRecordReader> &reader, FrmRecordReader *form) :
	XQConnector(reader, form),
	m_reader(reader),
	m_pForm(form),
    m_conRecordFile(xqcon_create<XFilePathConnector>(
                        reader->filename(), form->m_edPath, form->m_btnPath,
                        ("Binary files (*.bin);;All files (*.*)"), false)),
	m_conFF(xqcon_create<XQToggleButtonConnector>(reader->fastForward(), form->btnFF)),
	m_conRW(xqcon_create<XQToggleButtonConnector>(reader->rewind(), form->btnRW)),
	m_conStop(xqcon_create<XQButtonConnector>(reader->stop(), form->btnStop)),
	m_conFirst(xqcon_create<XQButtonConnector>(reader->first(), form->btnFirst)),
	m_conNext(xqcon_create<XQButtonConnector>(reader->next(), form->btnNext)),
	m_conBack(xqcon_create<XQButtonConnector>(reader->back(), form->btnBack)),
	m_conPosString(xqcon_create<XQLineEditConnector>(reader->posString(), form->edTime)),
	m_conSpeed(xqcon_create<XQComboBoxConnector>(reader->speed(), form->cmbSpeed, Snapshot( *reader->speed()))) {

    form->btnNext->setIcon(
        QApplication::style()->standardIcon(QStyle::SP_MediaSeekForward));
    form->btnBack->setIcon(
        QApplication::style()->standardIcon(QStyle::SP_MediaSeekBackward));
    form->btnFF->setIcon(
        QApplication::style()->standardIcon(QStyle::SP_MediaSkipForward));
    form->btnRW->setIcon(
        QApplication::style()->standardIcon(QStyle::SP_MediaSkipBackward));
    form->btnFirst->setIcon(
        QApplication::style()->standardIcon(QStyle::SP_MediaPlay));
    form->btnStop->setIcon(
        QApplication::style()->standardIcon(QStyle::SP_MediaPause));
}
