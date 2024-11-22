/***************************************************************************
        Copyright (C) 2002-2024 Kentaro Kitagawa
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

#ifndef nodebrowserH
#define nodebrowserH
//---------------------------------------------------------------------------
#include "xnodeconnector.h"

class Ui_FrmNodeBrowser;
typedef QForm<QWidget, Ui_FrmNodeBrowser> FrmNodeBrowser;

class XMeasure;
class XNodeBrowser : public XQConnector {
	Q_OBJECT
public:
	XNodeBrowser(
        const shared_ptr<XMeasure> &root, FrmNodeBrowser *form);
	virtual ~XNodeBrowser();

private slots:	
	virtual void process();
private:
	QTimer *m_pTimer;

    const weak_ptr<XMeasure> m_root;
	FrmNodeBrowser *const m_pForm;

    const shared_ptr<XStringNode> m_desc;

	shared_ptr<XNode> connectedNode(QWidget *widget);

	const xqcon_ptr m_conDesc;
	xqcon_ptr m_conValue;
};

#endif
