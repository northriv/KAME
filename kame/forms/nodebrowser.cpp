/***************************************************************************
		Copyright (C) 2002-2014 Kentaro Kitagawa
		                   kitag@kochi-u.ac.jp

		This program is free software; you can redistribute it and/or
		modify it under the terms of the GNU Library General Public
		License as published by the Free Software Foundation; either
		version 2 of the License, or (at your option) any later version.

		You should have received a copy of the GNU Library General 
		Public License and a list of authors along with this program; 
		see the files COPYING and AUTHORS.
 ***************************************************************************/
//---------------------------------------------------------------------------
#include "nodebrowser.h"
#include "measure.h"
#include <QLineEdit>
#include <QCursor>
#include <QTimer>
#include <QTextBrowser>
#include <QApplication>

#include "ui_nodebrowserform.h"

XNodeBrowser::XNodeBrowser
(const shared_ptr<XNode> &root, FrmNodeBrowser *form)
	: XQConnector(root, form),
	m_root(root),
	m_pForm(form),
	m_desc(XNode::createOrphan<XStringNode>("Desc", true)),
	m_conDesc(xqcon_create<XQTextBrowserConnector>(m_desc, form->m_txtDesc)) {

	m_pTimer = new QTimer(this);
	connect(m_pTimer, SIGNAL (timeout() ), this, SLOT(process()));
	m_pTimer->start(500);
    form->m_txtDesc->setAcceptRichText(true);
}

XNodeBrowser::~XNodeBrowser() {
	m_pTimer->stop();
}
shared_ptr<XNode>
XNodeBrowser::connectedNode(QWidget *widget) {
	if(!widget || (widget == m_pForm->m_txtDesc) ||
		(widget == m_pForm->m_edValue) || (widget == m_pForm)) {
		return shared_ptr<XNode>();
	}
	return XQConnector::connectedNode(widget);
}

void
XNodeBrowser::process() {
	QWidget *widget;
	shared_ptr<XNode> node;
	//	widget = KApplication::kApplication()->focusWidget();
	//		node = connectedNode(widget);
	if( !node) {
        widget = QApplication::widgetAt(QCursor::pos());
		node = connectedNode(widget);
		if( !node && widget) {
			widget = widget->parentWidget();
			node = connectedNode(widget);
			if( !node && widget) {
				widget = widget->parentWidget();
				node = connectedNode(widget);
			}
		}
	}

	if( !node)
		node = m_lastPointed;
	if((node != m_lastPointed) && node) {
		Snapshot shot( *node);
		auto valuenode(dynamic_pointer_cast<XValueNodeBase>(node));
		auto listnode(dynamic_pointer_cast<XListNodeBase>(node));

		m_conValue.reset();
		if(valuenode)
			m_conValue = xqcon_create<XQLineEditConnector>(valuenode, m_pForm->m_edValue);
		QString str;
		str += "<font color=#005500>Label:</font> ";
		str += node->getLabel().c_str();
		//		str += "\nName: ";
		//		str += node->getName();
		str += "<br>";
		if( !shot[ *node].isUIEnabled()) str+= "UI/scripting disabled.<br>";
		if(shot[ *node].isRuntime()) str+= "For run-time only.<br>";
		str += "<font color=#005500>Type:</font> ";
		str += node->getTypename().c_str();
		str += "<br>";
		XString rbpath;
		XNode *cnode = node.get();
		shared_ptr<XNode> rootnode(m_root);
		Snapshot rootshot( *rootnode);
		while(cnode) {
			if((rbpath.length() > 64) ||
				(cnode == m_root.lock().get())) {
				str += "<font color=#550000>Ruby object:</font><br> Measurement";
				str += rbpath.c_str();
				str += "<br><font color=#550000>Supported Ruby methods:</font>"
					" name() touch() child(<font color=#000088><i>name/idx</i></font>)"
					" [](<font color=#000088><i>name/idx</i></font>) count() each() to_ary()";
				if(valuenode)
					str += " set(<font color=#000088><i>x</i></font>)"
						" value=<font color=#000088><i>x</i></font> load(<font color=#000088><i>x</i></font>)"
						" &lt;&lt;<font color=#000088><i>x</i></font> get() value() to_str()";
				if(listnode)
					str += " create(<font color=#000088><i>type</i></font>, <font color=#000088><i>name</i></font>)"
						" release()";
				str += "<br>";
				break;
			}
			rbpath = formatString("[\"%s\"]%s", cnode->getName().c_str(), rbpath.c_str());
			cnode = cnode->upperNode(rootshot);
		}
		if( !cnode) {
			//			str += rbpath;
			str += "Inaccessible from the root.<br>";		
		}
		if(shot.size()) {
			str += formatString("<font color=#005500>%u Child(ren):</font> <br>", (unsigned int)shot.list()->size()).c_str();
			for(auto it = shot.list()->begin(); it != shot.list()->end(); ++it) {
				str += " ";
				str += ( *it)->getName().c_str();
			}
			str += "<br>";
		}
		trans( *m_desc).str(str);
	}
	m_lastPointed = node;
}
