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

#ifndef graphntoolboxH
#define graphntoolboxH
//---------------------------------------------------------------------------

#include "xnodeconnector.h"
#include "graph.h"
#include <fstream>

class XQGraph;
class QLineEdit;
class QAbstractButton;
class QPushButton;
class Ui_FrmGraphNURL;
typedef QForm<QWidget, Ui_FrmGraphNURL> FrmGraphNURL;

class DECLSPEC_KAME XGraphNToolBox: public XNode {
public:
    XGraphNToolBox(const char *name, bool runtime, FrmGraphNURL *item);
    XGraphNToolBox(const char *name, bool runtime, XQGraph *graphwidget,
        QLineEdit *ed, QAbstractButton *btn, QPushButton *btndump);
    virtual ~XGraphNToolBox();

    const shared_ptr<XGraph> &graph() const { return m_graph;}
    const shared_ptr<XStringNode> &filename() const { return m_filename;}

    const shared_ptr<XTouchableNode> &dump() const { return m_dump;}

    struct DECLSPEC_KAME Payload : public XNode::Payload {
        const Talker<bool> &onIconChanged() const { return m_tlkOnIconChanged;}
        Talker<bool> &onIconChanged() { return m_tlkOnIconChanged;}
    private:
        friend class XGraphNToolBox;
        Talker<bool> m_tlkOnIconChanged;
    };
protected:
    virtual void dumpToFileThreaded(std::fstream &) = 0;

    std::deque<xqcon_ptr> m_conUIs;
private:
    QPushButton * const m_btnDump;

    const shared_ptr<XGraph> m_graph;

    const shared_ptr<XTouchableNode> m_dump;
    const shared_ptr<XStringNode> m_filename;

    shared_ptr<Listener> m_lsnOnDumpTouched, m_lsnOnFilenameChanged,
        m_lsnOnIconChanged;

    void onDumpTouched(const Snapshot &shot, XTouchableNode *);
    void onFilenameChanged(const Snapshot &shot, XValueNodeBase *);
    void onIconChanged(const Snapshot &shot, bool );

    xqcon_ptr m_conFilename, m_conDump;

    unique_ptr<XThread> m_threadDump;
    std::fstream m_stream;
    XMutex m_filemutex;
};
#endif
