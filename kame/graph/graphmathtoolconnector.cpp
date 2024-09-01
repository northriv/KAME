/***************************************************************************
        Copyright (C) 2002-2023 Kentaro Kitagawa
		                   kitag@issp.u-tokyo.ac.jp
		
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

std::deque<shared_ptr<Listener>> XQGraph1DMathToolConnector::s_activeListeners;
std::deque<shared_ptr<Listener>> XQGraph2DMathToolConnector::s_activeListeners;

XQGraph1DMathToolConnector::XQGraph1DMathToolConnector
(const std::deque<shared_ptr<XGraph1DMathToolList>> &lists, QToolButton* item, XQGraph *graphwidget) :
    m_pItem(item), m_graphwidget(graphwidget), m_lists(lists) {
    m_menu = new QMenu();
    item->setMenu(m_menu);
    item->setPopupMode(QToolButton::InstantPopup);
    connect( m_menu, SIGNAL( aboutToShow() ), this, SLOT( menuOpenActionActivated() ) );
}

void XQGraph1DMathToolConnector::toolActivated(QAction *act) {
    if(m_actionToToolMap.count(act)) {
        auto label = m_actionToToolMap.at(act);
        m_graphwidget->activateAxisSelectionTool(XAxis::AxisDirection::X, label);
        s_activeListeners.clear(); //cancels all the remaining selections.
        for(auto &toollist: m_lists)
            s_activeListeners.push_back( m_graphwidget->onAxisSelectedByTool().connectWeakly(
                toollist, &XGraph1DMathToolList::onAxisSelectedByTool));
    }
    if(m_actionToExisitingToolMap.count(act)) {
        auto [begin, end] = m_actionToExisitingToolMap.equal_range(act);
        for(auto it = begin; it != end; it++) {
            auto toollist = it->second.first;
            auto tool = it->second.second;
            toollist->m_measure.lock()->iterate_commit([&](Transaction &tr){
                if( !static_pointer_cast<XGraph1DMathTool>(tool)->releaseEntries(tr))
                    return;
                toollist->release(tr, tool);
            });
        }
    }
    m_actionToToolMap.clear();
    m_actionToExisitingToolMap.clear();
}


void XQGraph1DMathToolConnector::menuOpenActionActivated() {
    m_menu->clear();
//    for(auto &&a: m_actionToToolMap)
//        delete a.first;
    m_actionToToolMap.clear();
    m_actionToExisitingToolMap.clear();
    if(m_lists.empty())
        return;
    auto &list = m_lists[0];
    Snapshot shot( *list);
    for(unsigned int i = 0; i < shot.size(); ++i) {
        QMenu *menuoftool = m_menu->addMenu(shot.list()->at(i)->getLabel());
        QAction *act = new QAction(i18n("Delete Tool"), menuoftool);
        menuoftool->addAction(act);
        for(auto &toollist: m_lists) {
            Snapshot shot( *toollist);
            m_actionToExisitingToolMap.emplace(act,
                std::pair<shared_ptr<XGraph1DMathToolList>, shared_ptr<XNode>>(toollist, shot.list()->at(i)));
        }
    }
    m_menu->addSeparator();
    for(auto &type: list->typelabels()) {
        QAction *act = new QAction(type, m_menu);
        m_menu->addAction(act);
        m_actionToToolMap.emplace(act, type);
    }
    connect( m_menu, SIGNAL( triggered(QAction*) ), this, SLOT( toolActivated(QAction*) ) );

}
XQGraph1DMathToolConnector::~XQGraph1DMathToolConnector() {
    m_menu->clear();
    m_actionToToolMap.clear();
}



XQGraph2DMathToolConnector::XQGraph2DMathToolConnector
(const std::deque<shared_ptr<XGraph2DMathToolList>> &lists, QToolButton* item, XQGraph *graphwidget) :
    m_pItem(item), m_graphwidget(graphwidget), m_lists(lists) {
    m_menu = new QMenu();
    item->setMenu(m_menu);
    item->setPopupMode(QToolButton::InstantPopup);
    connect( m_menu, SIGNAL( aboutToShow() ), this, SLOT( menuOpenActionActivated() ) );
}
XQGraph2DMathToolConnector::XQGraph2DMathToolConnector
(const shared_ptr<XGraph2DMathToolList> &list, QToolButton* item, XQGraph *graphwidget) :
    XQGraph2DMathToolConnector(std::deque<shared_ptr<XGraph2DMathToolList>>{list}, item, graphwidget) {}

void XQGraph2DMathToolConnector::toolActivated(QAction *act) {
    if(m_actionToToolMap.count(act)) {
        auto label = m_actionToToolMap.at(act);
        m_graphwidget->activatePlaneSelectionTool(XAxis::AxisDirection::X, XAxis::AxisDirection::Y, label);
        s_activeListeners.clear(); //cancels all the remaining selections.
        for(auto &toollist: m_lists)
            s_activeListeners.push_back( m_graphwidget->onPlaneSelectedByTool().connectWeakly(
                toollist, &XGraph2DMathToolList::onPlaneSelectedByTool));
    }
    if(m_actionToExisitingToolMap.count(act)) {
        auto [begin, end] = m_actionToExisitingToolMap.equal_range(act);
        for(auto it = begin; it != end; it++) {
            auto toollist = it->second.first;
            auto tool = it->second.second;
            toollist->m_measure.lock()->iterate_commit([&](Transaction &tr){
                if( !static_pointer_cast<XGraph2DMathTool>(tool)->releaseEntries(tr))
                    return;
                toollist->release(tr, tool);
            });
        }
    }
    m_actionToToolMap.clear();
    m_actionToExisitingToolMap.clear();
}


void XQGraph2DMathToolConnector::menuOpenActionActivated() {
    m_menu->clear();
    m_actionToToolMap.clear();
    m_actionToExisitingToolMap.clear();
    if(m_lists.empty())
        return;
    auto &list = m_lists[0];
    Snapshot shot( *list);
    for(unsigned int i = 0; i < shot.size(); ++i) {
        QMenu *menuoftool = m_menu->addMenu(shot.list()->at(i)->getLabel());
        QAction *act = new QAction(i18n("Delete Tool"), menuoftool);
        menuoftool->addAction(act);
        for(auto &toollist: m_lists) {
            Snapshot shot( *toollist);
            m_actionToExisitingToolMap.emplace(act,
                std::pair<shared_ptr<XGraph2DMathToolList>, shared_ptr<XNode>>(toollist, shot.list()->at(i)));
        }
    }
    m_menu->addSeparator();
    for(auto &type: list->typelabels()) {
        QAction *act = new QAction(type, m_menu);
        m_menu->addAction(act);
        m_actionToToolMap.emplace(act, type);
    }
    connect( m_menu, SIGNAL( triggered(QAction*) ), this, SLOT( toolActivated(QAction*) ) );
}
XQGraph2DMathToolConnector::~XQGraph2DMathToolConnector() {
    m_menu->clear();
    m_actionToToolMap.clear();
}
