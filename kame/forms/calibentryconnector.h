/***************************************************************************
        Copyright (C) 2002-2026 Kentaro Kitagawa
		                   kitag@issp.u-tokyo.ac.jp

		This program is free software; you can redistribute it and/or
		modify it under the terms of the GNU General Public
		License as published by the Free Software Foundation; either
		version 2 of the License, or (at your option) any later version.

		You should have received a copy of the GNU General
		Public License and a list of authors along with this program;
		see the files COPYING and AUTHORS.
***************************************************************************/
#ifndef calibentryconnectorH
#define calibentryconnectorH

#include "xnodeconnector.h"

class QPushButton;
class QTableWidget;
class XCalibratedEntryList;

class XCalibratedEntryListConnector : public XListQConnector {
    Q_OBJECT
public:
    XCalibratedEntryListConnector(const shared_ptr<XCalibratedEntryList> &list,
        QTableWidget *table, QPushButton *btnNew, QPushButton *btnDelete);
    virtual ~XCalibratedEntryListConnector() {}
protected:
    void onCatch(const Snapshot &, const XListNodeBase::Payload::CatchEvent &) override;
    void onRelease(const Snapshot &, const XListNodeBase::Payload::ReleaseEvent &) override;
protected slots:
    void cellClicked(int row, int col);
private:
    const shared_ptr<XCalibratedEntryList> m_list;
    const shared_ptr<XTouchableNode> m_newEntry, m_deleteEntry;
    struct tcons {
        xqcon_ptr consrc, concurve;
        shared_ptr<XNode> node;
        QWidget *widget;
    };
    std::deque<tcons> m_cons;
    const xqcon_ptr m_conNew, m_conDelete;
    shared_ptr<Listener> m_lsnNew, m_lsnDelete;
    void onNew(const Snapshot &, XTouchableNode *);
    void onDelete(const Snapshot &, XTouchableNode *);
};

#endif
