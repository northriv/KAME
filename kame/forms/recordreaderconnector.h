/***************************************************************************
		Copyright (C) 2002-2011 Kentaro Kitagawa
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

#ifndef recordreaderconnectorH
#define recordreaderconnectorH
//---------------------------------------------------------------------------
#include "xnodeconnector.h"

class Ui_FrmRecordReader;
typedef QForm<QWidget, Ui_FrmRecordReader> FrmRecordReader;

class XRawStreamRecordReader;
class XRawStreamRecordReaderConnector : public XQConnector {
	Q_OBJECT
public:
	XRawStreamRecordReaderConnector(
		const shared_ptr<XRawStreamRecordReader> &reader, FrmRecordReader *form);
	virtual ~XRawStreamRecordReaderConnector() {}

private:
	const shared_ptr<XRawStreamRecordReader> m_reader;
	FrmRecordReader *const m_pForm;
  
	const xqcon_ptr m_conRecordFile, m_conFF, m_conRW, m_conStop,
		m_conFirst, m_conNext, m_conBack, m_conPosString, m_conSpeed;    
};
  
#endif
