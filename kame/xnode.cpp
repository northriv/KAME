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

#include "transaction_impl.h"

#include "xnode.h"
#include <typeinfo>

DECLSPEC_KAME XThreadLocal<std::deque<shared_ptr<XNode> > > XNode::stl_thisCreating;

void
XNode::Payload::setUIEnabled(bool var) {
	if(isDisabled()) return;
	m_flags = (m_flags & ~NODE_UI_ENABLED) | (var ? NODE_UI_ENABLED : 0);
	tr().mark(onUIFlagsChanged(), &node());
}
void
XNode::Payload::disable() {
	m_flags = (m_flags & ~(NODE_DISABLED | NODE_UI_ENABLED)) | NODE_DISABLED;
	tr().mark(onUIFlagsChanged(), &node());
}

XNode::XNode(const char *name, bool runtime)
	: Transactional::Node<XNode>(), m_name(name ? name : "") {
	// temporaly shared_ptr to be able to use shared_from_this() in constructors
    XNode::stl_thisCreating->push_back(shared_ptr<XNode>(this));
	assert(shared_from_this());

	trans( *this).setRuntime(runtime);

	dbgPrint(QString("xnode %1 is created., addr=0x%2, size=0x%3")
			 .arg(getLabel())
			 .arg((uintptr_t)this, 0, 16)
			 .arg((uintptr_t)sizeof(XNode), 0, 16));
}
XNode::~XNode() {
	dbgPrint(QString("xnode %1 is being deleted., addr=0x%2").arg(getLabel()).arg((uintptr_t)this, 0, 16));
}
XString
XNode::getName() const {
    return m_name;
}
XString
XNode::getTypename() const {
    XString name = typeid( *this).name();
    int i = name.find('X');
    assert(i != std::string::npos);
    assert(i + 1 < name.length());
    return name.substr(i + 1);
}

void
XNode::disable() {
	trans(*this).disable();
}
void
XNode::setUIEnabled(bool v) {
	trans( *this).setUIEnabled(v);
}

shared_ptr<XNode>
XNode::getChild(const XString &var) const {
	Snapshot shot( *this);
	shared_ptr<XNode> node;
	shared_ptr<const NodeList> list(shot.list());
	if(list) {
		for(auto it = list->begin(); it != list->end(); it++) {
			if(dynamic_pointer_cast<XNode>( *it)->getName() == var) {
                node = dynamic_pointer_cast<XNode>( *it);
                break;
			}
		}
	}
	return node;
}

void
XTouchableNode::Payload::touch() {
	tr().mark(onTouch(), static_cast<XTouchableNode *>(&node()));
}

template <typename T, int base>
void
XIntNodeBase<T, base>::Payload::str_(const XString &str) {
}
template <>
void
XIntNodeBase<int, 10>::Payload::str_(const XString &str) {
    bool ok;
    int var = QString(str).toInt(&ok, 10);
    if( !ok)
		throw XKameError(i18n("Ill string conversion to integer."), __FILE__, __LINE__);
    *this = var;
}
template <>
void
XIntNodeBase<unsigned int, 10>::Payload::str_(const XString &str) {
    bool ok;
    unsigned int var = QString(str).toUInt(&ok);
    if( !ok)
		throw XKameError(i18n("Ill string conversion to unsigned integer."), __FILE__, __LINE__);
    *this = var;
}
template <>
void
XIntNodeBase<long, 10>::Payload::str_(const XString &str) {
    bool ok;
    long var = QString(str).toLong(&ok, 10);
    if( !ok)
		throw XKameError(i18n("Ill string conversion to integer."), __FILE__, __LINE__);
    *this = var;
}
template <>
void
XIntNodeBase<unsigned long, 10>::Payload::str_(const XString &str) {
    bool ok;
    unsigned long var = QString(str).toULong(&ok);
    if( !ok)
		throw XKameError(i18n("Ill string conversion to unsigned integer."), __FILE__, __LINE__);
    *this = var;
}
template <>
void
XIntNodeBase<unsigned long, 16>::Payload::str_(const XString &str) {
    bool ok;
    unsigned int var = QString(str).toULong(&ok, 16);
    if( !ok)
		throw XKameError(i18n("Ill string conversion to hex."), __FILE__, __LINE__);
    *this = var;
}
template <>
void
XIntNodeBase<bool, 10>::Payload::str_(const XString &str) {
	bool ok;
	bool x = QString(str).toInt(&ok);
    if(ok) {
		*this =  x ? true : false ;
		return;
    }
	if(QString(str).trimmed().toLower() == "true") {
        *this = true; return;
	}
	if(QString(str).trimmed().toLower() == "false") {
        *this = false; return;
	}
	throw XKameError(i18n("Ill string conversion to boolean."), __FILE__, __LINE__);
}

template <typename T, int base>
XString
XIntNodeBase<T, base>::Payload::to_str() const {
    return QString::number(m_var, base);
}
template <>
XString
XIntNodeBase<bool, 10>::Payload::to_str() const {
    return m_var ? "true" : "false";
}

template class XIntNodeBase<int, 10>;
template class XIntNodeBase<unsigned int, 10>;
template class XIntNodeBase<long, 10>;
template class XIntNodeBase<unsigned long, 10>;
template class XIntNodeBase<unsigned long, 16>;
template class XIntNodeBase<bool, 10>;

XStringNode::XStringNode(const char *name, bool runtime)
	: XValueNodeBase(name, runtime) {}

XDoubleNode::XDoubleNode(const char *name, bool runtime, const char *format)
	: XValueNodeBase(name, runtime) {
	setFormat(format);
}

XString
XDoubleNode::Payload::to_str() const {
    return formatDouble(
    	static_cast<const XDoubleNode&>(node()).format(), m_var);
}
void
XDoubleNode::Payload::str_(const XString &str) {
	bool ok;
    double var = QString(str).toDouble(&ok);
    if( !ok)
		throw XKameError(i18n("Ill string conversion to double float."), __FILE__, __LINE__);
    *this = var;
}

void
XDoubleNode::setFormat(const char* format) {
    XString fmt;
    if(format)
    	fmt = format;
    try {
        formatDoubleValidator(fmt);
        m_format.reset(new XString(fmt));
    }
    catch (XKameError &e) {
        e.print();
    }
}

template DECLSPEC_KAME class Transactional::Node<class XNode>;
