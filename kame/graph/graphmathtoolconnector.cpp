/***************************************************************************
        Copyright (C) 2002-2023 Kentaro Kitagawa
		                   kitagawa@phys.s.u-tokyo.ac.jp
		
		This program is free software; you can redistribute it and/or
		modify it under the terms of the GNU Library General Public
		License as published by the Free Software Foundation; either
		version 2 of the License, or (at your option) any later version.
		
		You should have received a copy of the GNU Library General 
		Public License and a list of authors along with this program; 
		see the files COPYING and AUTHORS.
 ***************************************************************************/
#include "graphmathtoolconnector.h"
#include "graphwidget.h"
#include <QMenu>
#include <QAction>
#include <QToolButton>
#include "measure.h"
//---------------------------------------------------------------------------
XQGraph2DMathToolConnector::XQGraph2DMathToolConnector
(const std::deque<shared_ptr<XGraph2DMathToolList>> &lists, QToolButton* item, XQGraph *graphwidget) :
    m_pItem(item), m_graphwidget(graphwidget), m_lists(lists) {
    m_menu = new QMenu();
    item->setMenu(m_menu);
    item->setPopupMode(QToolButton::InstantPopup);
    connect( m_menu, SIGNAL( aboutToShow() ), this, SLOT( menuOpenActionActivated() ) );
}

void XQGraph2DMathToolConnector::toolActivated(QAction *act) {
    if(m_actionToToolMap.count(act)) {
        auto [toollist, label] = m_actionToToolMap.at(act);
        m_graphwidget->activatePlaneSelectionTool(XAxis::AxisDirection::X, XAxis::AxisDirection::Y, label);
        m_activeListener = m_graphwidget->onPlaneSelectedByTool().connectWeakly(
            toollist, &XGraph2DMathToolList::onPlaneSelectedByTool);
    }
    if(m_actionToExisitingToolMap.count(act)) {
        auto r = m_actionToExisitingToolMap.at(act);
        auto toollist = r.first;
        auto tool = r.second;
        toollist->m_measure.lock()->iterate_commit([&](Transaction &tr){
            toollist->release(tr, tool);
            toollist->m_measure.lock()->scalarEntries()->release(tr, tool);
        });
    }
    m_actionToToolMap.clear();
    m_actionToExisitingToolMap.clear();
}


void XQGraph2DMathToolConnector::menuOpenActionActivated() {
    m_menu->clear();
//    for(auto &&a: m_actionToToolMap)
//        delete a.first;
    m_actionToToolMap.clear();
    m_actionToExisitingToolMap.clear();
    std::deque<QMenu*> submenus;
    for(auto &list: m_lists) {
        QMenu *submenu = m_menu->addMenu(list->getLabel());
        submenus.push_back(submenu);
        Snapshot shot( *list);
        if(shot.size()) {
            for(auto it = shot.list()->begin(); it != shot.list()->end(); ++it) {
                QMenu *menuoftool = submenu->addMenu((*it)->getLabel());
                QAction *act = new QAction(i18n("Delete Tool"), menuoftool);
                menuoftool->addAction(act);
                m_actionToExisitingToolMap.emplace(act,
                    std::pair<shared_ptr<XGraph2DMathToolList>, shared_ptr<XNode>>(list, *it));
            }
        }
        submenu->addSeparator();
        for(auto &type: list->typelabels()) {
            QAction *act = new QAction(type, submenu);
            submenu->addAction(act);
            m_actionToToolMap.emplace(act, std::pair<shared_ptr<XGraph2DMathToolList>, XString>(list, type));
        }
        connect( submenu, SIGNAL( triggered(QAction*) ), this, SLOT( toolActivated(QAction*) ) );
    }

}
XQGraph2DMathToolConnector::~XQGraph2DMathToolConnector() {
    m_menu->clear();
    m_actionToToolMap.clear();
}
