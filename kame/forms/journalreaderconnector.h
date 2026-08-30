/***************************************************************************
		Copyright (C) 2002-2015 Kentaro Kitagawa
		                   kitag@issp.u-tokyo.ac.jp
		
		This program is free software; you can redistribute it and/or
		modify it under the terms of the GNU General Public
		License as published by the Free Software Foundation; either
		version 2 of the License, or (at your option) any later version.
		
		You should have received a copy of the GNU General 
		Public License and a list of authors along with this program; 
		see the files COPYING and AUTHORS.
***************************************************************************/
//---------------------------------------------------------------------------

#ifndef journalreaderconnectorH
#define journalreaderconnectorH
//---------------------------------------------------------------------------
#include "xnodeconnector.h"

class Ui_FrmJournalReader;
typedef QForm<QWidget, Ui_FrmJournalReader> FrmJournalReader;

class XJournalReader;
class XJournalReaderConnector : public XQConnector {
	Q_OBJECT
public:
	XJournalReaderConnector(
		const shared_ptr<XJournalReader> &reader, FrmJournalReader *form);
	virtual ~XJournalReaderConnector() {}

private slots:
	//! The user let go of the scrub bar, or paged it.  Seeking is expensive,
	//! so the slider does not track: this arrives once, not per pixel.
	void onSliderChanged(int value);

private:
	//! Where the reader has got to, back onto the slider.  Skipped while the
	//! user has hold of it, so the two do not fight over the handle.
	void onPositionChanged(const Snapshot &shot, XValueNodeBase *);

	const shared_ptr<XJournalReader> m_reader;
	FrmJournalReader *const m_pForm;
  
	const xqcon_ptr m_conRecordFile, m_conFF, m_conRW, m_conStop,
		m_conFirst, m_conNext, m_conBack, m_conRecordTime, m_conSpeed, m_conRestore;    
	//! Last, so that it is destroyed first: members go in reverse order of
	//! declaration, and a listener that outlived m_reader could still be
	//! dispatched into a callback that reads it.
	shared_ptr<Listener> m_lsnPosition;
};
  
#endif
