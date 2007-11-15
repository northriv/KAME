/***************************************************************************
		Copyright (C) 2002-2007 Kentaro Kitagawa
		                   kitag@issp.u-tokyo.ac.jp
		
		This program is free software; you can redistribute it and/or
		modify it under the terms of the GNU Library General Public
		License as published by the Free Software Foundation; either
		version 2 of the License, or (at your option) any later version.
		
		You should have received a copy of the GNU Library General 
		Public License and a list of authors along with this program; 
		see the files COPYING and AUTHORS.
***************************************************************************/
#include "xnode.h"
#include <klocale.h>
#include <typeinfo>

XThreadLocal<std::deque<shared_ptr<XNode> > > XNode::stl_thisCreating;

//---------------------------------------------------------------------------
XNode::XNode(const char *name, bool runtime)
	: m_children(), m_name(name ? name : "")
{
	// temporaly shared_ptr to be able to use shared_from_this() in constructors
	XNode::stl_thisCreating->push_back(shared_ptr<XNode>(this));
	ASSERT(shared_from_this());
	m_flags = (runtime ? FLAG_RUNTIME : 0) | FLAG_UI_ENABLED | FLAG_ENABLED;
	dbgPrint(QString("xnode %1 is created., addr=0x%2, size=0x%3")
			 .arg(getName())
			 .arg((unsigned int)this, 0, 16)
			 .arg((unsigned int)sizeof(XNode), 0, 16));
}
XNode::~XNode() {
	dbgPrint(QString("xnode %1 is being deleted., addr=0x%2").arg(getName()).arg((unsigned int)this, 0, 16));
}
std::string
XNode::getName() const {
    return m_name;
}
std::string
XNode::getTypename() const {
    std::string name = typeid(*this).name();
    unsigned int i = name.find('X');
    ASSERT(i != std::string::npos);
    ASSERT(i + 1 < name.length());
    return name.substr(i + 1);
}
void
XNode::insert(const shared_ptr<XNode> &ptr)
{
    ASSERT(ptr);
    if(!ptr->m_parent.lock())
    	ptr->m_parent = shared_from_this();
    for(;;) {
        atomic_shared_ptr<NodeList> old_list(m_children);
        atomic_shared_ptr<NodeList> new_list(old_list ? (new NodeList(*old_list)) : (new NodeList));        
        new_list->push_back(ptr);
        if(new_list.compareAndSwap(old_list, m_children)) break;
    }
}
void
XNode::disable() {
	for(;;) {
		int flag = m_flags;
		if(atomicCompareAndSet(flag, flag & ~(FLAG_ENABLED | FLAG_UI_ENABLED), &m_flags))
			break;
	}
    onUIEnabled().talk(shared_from_this());
}
void
XNode::setUIEnabled(bool v) {
	if(!isEnabled()) return;
	for(;;) {
		int flag = m_flags;
		if(atomicCompareAndSet(flag, v ? (flag | FLAG_UI_ENABLED) : (flag & ~FLAG_UI_ENABLED), &m_flags))
			break;
	}
    onUIEnabled().talk(shared_from_this());
}
void
XNode::touch() {
    onTouch().talk(shared_from_this());
}

void
XNode::clearChildren()
{
    m_children.reset();
}
int
XNode::releaseChild(const shared_ptr<XNode> &node)
{
//	node->m_parent.reset();
    for(;;) {
        atomic_shared_ptr<NodeList> old_list(m_children);
        if(!old_list) return -1;
        atomic_shared_ptr<NodeList> new_list(new NodeList(*old_list));
        NodeList::iterator it = find(new_list->begin(), new_list->end(), node);
        if(it == new_list->end()) return -1;
        new_list->erase(it);
        if(new_list->empty())
            new_list.reset();
        if(new_list.compareAndSwap(old_list, m_children)) break;
    }
    return 0;
}

shared_ptr<XNode>
XNode::getChild(const std::string &var) const
{
	shared_ptr<XNode> node;
	atomic_shared_ptr<const XNode::NodeList> list(children());
	if(list) { 
		for(XNode::NodeList::const_iterator it = list->begin(); it != list->end(); it++) {
			if((*it)->getName() == var) {
                node = *it;
                break;
			}
		}
	}
	return node;
}
shared_ptr<XNode>
XNode::getParent() const
{
	return m_parent.lock();
}

XValueNodeBase::XValueNodeBase(const char *name, bool runtime) : 
    XNode(name, runtime), m_validator(0L)
{
}
void
XValueNodeBase::str(const std::string &s) throw (XKameError &) {
    std::string sc(s);
    if(m_validator)
		(*m_validator)(sc);
    _str(sc);
}
void
XValueNodeBase::str(const QString &s) throw (XKameError &) {
    str(std::string((const char*)s));
}
void
XValueNodeBase::setValidator(Validator v) {
    m_validator = v;
}

template <typename T, int base>
void
XValueNode<T, base>::value(const T &t) {
    if(m_tlkBeforeValueChanged.empty() && m_tlkOnValueChanged.empty()) {
        m_var = t;
    }
    else {
        XScopedLock<XRecursiveMutex> lock(m_valuemutex);
        shared_ptr<XValueNodeBase> ptr = 
            dynamic_pointer_cast<XValueNodeBase>(shared_from_this());
        m_tlkBeforeValueChanged.talk(ptr);
        m_var = t;
        m_tlkOnValueChanged.talk(ptr); //, 1, &statusmutex);
    }
}

template <>
void
XValueNode<int, 10>::_str(const std::string &str) throw (XKameError &) {
    bool ok;
    int var = QString(str).toInt(&ok, 10);
    if(!ok)
		throw XKameError(KAME::i18n("Ill string conversion to integer."), __FILE__, __LINE__);
    value(var);
}
template <>
void
XValueNode<unsigned int, 10>::_str(const std::string &str) throw (XKameError &) {
    bool ok;
    unsigned int var = QString(str).toUInt(&ok);
    if(!ok)
		throw XKameError(KAME::i18n("Ill string conversion to unsigned integer."), __FILE__, __LINE__);
    value(var);
}
template <>
void
XValueNode<unsigned int, 16>::_str(const std::string &str) throw (XKameError &) {
    bool ok;
    unsigned int var = QString(str).toUInt(&ok, 16);
    if(!ok)
		throw XKameError(KAME::i18n("Ill string conversion to hex."), __FILE__, __LINE__);
    value(var);
}
template <>
void
XValueNode<bool, 10>::_str(const std::string &str) throw (XKameError &) {
	bool ok;
	bool x = QString(str).toInt(&ok);
    if(ok) {
		value( x ? true : false );
		return;
    }
	if(QString(str).stripWhiteSpace().lower() == "true") {
        value(true); return;
	}
	if(QString(str).stripWhiteSpace().lower() == "false") {
        value(false); return;
	}
	throw XKameError(KAME::i18n("Ill string conversion to boolean."), __FILE__, __LINE__);
}

template <typename T, int base>
std::string
XValueNode<T, base>::to_str() const {
    return QString::number(m_var, base);
}
template <>
std::string
XValueNode<bool, 10>::to_str() const {
    return m_var ? "true" : "false";
}

template class XValueNode<int, 10>;
template class XValueNode<unsigned int, 10>;
template class XValueNode<unsigned int, 16>;
template class XValueNode<bool, 10>;

XStringNode::XStringNode(const char *name, bool runtime)
	: XValueNodeBase(name, runtime), m_var(std::string()) {}

std::string
XStringNode::to_str() const
{
    return m_var;
}
void
XStringNode::operator=(const std::string &var)
{
    value(var);
}
void
XStringNode::_str(const std::string &var) throw (XKameError &)
{
    value(var);
}

XStringNode::operator std::string() const {
    return to_str();
}
void
XStringNode::value(const std::string &t) {
    if(beforeValueChanged().empty() && onValueChanged().empty()) {
        m_var = t;
    }
    else {
        XScopedLock<XRecursiveMutex> lock(m_valuemutex);
        beforeValueChanged().talk(dynamic_pointer_cast<XValueNodeBase>(shared_from_this()));
        m_var = t;
        onValueChanged().talk(dynamic_pointer_cast<XValueNodeBase>(shared_from_this()));
    }
}

XDoubleNode::XDoubleNode(const char *name, bool runtime, const char *format)
	: XValueNodeBase(name, runtime), m_var(0.0)
{
	if(format)
		setFormat(format);
	else
		setFormat("");
}
std::string
XDoubleNode::to_str() const
{
    return formatDouble(m_format.c_str(), m_var);
}
void
XDoubleNode::_str(const std::string &str) throw (XKameError &)
{
	bool ok;
    double var = QString(str).toDouble(&ok);
    if(!ok) 
		throw XKameError(KAME::i18n("Ill string conversion to double float."), __FILE__, __LINE__);
    value(var);
}
void
XDoubleNode::value(const double &t) {
    if(beforeValueChanged().empty() && onValueChanged().empty()) {
        m_var = t;
    }
    else {
        XScopedLock<XRecursiveMutex> lock(m_valuemutex);
        beforeValueChanged().talk(dynamic_pointer_cast<XValueNodeBase>(shared_from_this()));
        m_var = t;
        onValueChanged().talk(dynamic_pointer_cast<XValueNodeBase>(shared_from_this()));
    }
}
XDoubleNode::operator double() const
{
    return m_var;
}

const char *
XDoubleNode::format() const {
    return m_format.c_str();
}
void
XDoubleNode::setFormat(const char* format) {
    std::string fmt;
    if(format) fmt = format;
    try {
        formatDoubleValidator(fmt);
        m_format = fmt;
    }
    catch (XKameError &e) {
        e.print();
    }
}

