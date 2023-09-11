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

#ifndef graphmathtoolconnectorH
#define graphmathtoolconnectorH
//---------------------------------------------------------------------------

#include "xnodeconnector.h"
#include "graphmathtool.h"

class QPushButton;
class QAction;
class QActionGroup;
class QMenu;
class XGraph1DMathToolList;
class XGraph2DMathToolList;
class XQGraph;

//Q_OBJECT cannot be used in template class!!!!

class DECLSPEC_KAME XQGraph1DMathToolConnector : public QObject {
    Q_OBJECT
public:
    XQGraph1DMathToolConnector(const std::deque<shared_ptr<XGraph1DMathToolList>> &lists, QToolButton* item, XQGraph *graphwidget);
    virtual ~XQGraph1DMathToolConnector();
private:
    QToolButton *const m_pItem;
    QMenu *m_menu;
    XQGraph *m_graphwidget;

    std::map<QAction *, XString> m_actionToToolMap;
    std::multimap<QAction *, std::pair<shared_ptr<XGraph1DMathToolList>, shared_ptr<XNode>>> m_actionToExisitingToolMap;
public:

private:
    std::deque<shared_ptr<XGraph1DMathToolList>> m_lists;
    std::deque<shared_ptr<Listener>> m_activeListeners;
public slots:
    virtual void menuOpenActionActivated();
    virtual void toolActivated(QAction *);
};

class DECLSPEC_KAME XQGraph2DMathToolConnector : public QObject {
    Q_OBJECT
public:
    XQGraph2DMathToolConnector(const std::deque<shared_ptr<XGraph2DMathToolList>> &lists, QToolButton* item, XQGraph *graphwidget);
    virtual ~XQGraph2DMathToolConnector();
private:
    QToolButton *const m_pItem;
    QMenu *m_menu;
    XQGraph *m_graphwidget;

    std::map<QAction *, XString> m_actionToToolMap;
    std::multimap<QAction *, std::pair<shared_ptr<XGraph2DMathToolList>, shared_ptr<XNode>>> m_actionToExisitingToolMap;
public:

private:
    std::deque<shared_ptr<XGraph2DMathToolList>> m_lists;
    std::deque<shared_ptr<Listener>> m_activeListeners;
public slots:
    virtual void menuOpenActionActivated();
    virtual void toolActivated(QAction *);
};


#endif
