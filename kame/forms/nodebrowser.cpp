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
#include "nodebrowser.h"
#include "measure.h"
#include <QLineEdit>
#include <QCursor>
#include <QTimer>
#include <QTextBrowser>
#include <QApplication>

#include "ui_nodebrowserform.h"

XNodeBrowser::XNodeBrowser
(const shared_ptr<XMeasure> &root, FrmNodeBrowser *form)
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
    if( !widget || (widget == m_pForm->m_txtDesc) ||
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

    shared_ptr<XMeasure> rootnode(m_root);

    if((node != rootnode->lastPointedByNodeBrowser()) && node) {
        trans( *rootnode->pyInfoForNodeBrowser()) = ""; //erases old info.

		Snapshot shot( *node);
		auto valuenode(dynamic_pointer_cast<XValueNodeBase>(node));
		auto listnode(dynamic_pointer_cast<XListNodeBase>(node));

		m_conValue.reset();
		if(valuenode)
			m_conValue = xqcon_create<XQLineEditConnector>(valuenode, m_pForm->m_edValue);
        else
            m_pForm->m_edValue->setText("");
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
        XString nodeabspath;
		XNode *cnode = node.get();
        Snapshot rootshot( *rootnode);
		while(cnode) {
            if((nodeabspath.length() > 64) ||
				(cnode == m_root.lock().get())) {
#ifdef USE_PYBIND11
                str += "<font color=#550000>Python object:</font><br> Root()";
#else
                str += "<font color=#550000>Ruby object:</font><br> Measurement";
#endif
                str += nodeabspath.c_str();
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
            nodeabspath = formatString("[\"%s\"]%s", cnode->getName().c_str(), nodeabspath.c_str());
			cnode = cnode->upperNode(rootshot);
		}
		if( !cnode) {
            //			str += nodeabspath;
            str += "Inaccessible from the root.<br><br>";
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
    else if((node == rootnode->lastPointedByNodeBrowser()) && node) {
        XString s = ***rootnode->pyInfoForNodeBrowser();
        if(s.length())
            m_desc->iterate_commit([=](Transaction &tr){
                if(tr[ *m_desc].to_str().find("<br><br>") == std::string::npos) {
                    tr[ *m_desc].str(tr[ *m_desc].to_str() + "<br><font color=#550000>Supported Python methods:</font>"
                        + s + "<br><br>"); //python info has not yet been added.
                }
            });
    }
    rootnode->lastPointedByNodeBrowser() = node;
}
