/***************************************************************************
		Copyright (C) 2002-2010 Kentaro Kitagawa
		                   kitag@issp.u-tokyo.ac.jp

		This program is free software; you can redistribute it and/or
		modify it under the terms of the GNU Library General Public
		License as published by the Free Software Foundation; either
		version 2 of the License, or (at your option) any later version.

		You should have received a copy of the GNU Library General 
		Public License and a list of authors along with this program; 
		see the files COPYING and AUTHORS.
 ***************************************************************************/
#ifndef xnodeH
#define xnodeH

#include "transaction.h"
#include "xsignal.h"
#include "threadlocal.h"

template <class T>
shared_ptr<T> createOrphan(const char *name, bool runtime = false);
template <class T, typename X>
shared_ptr<T> createOrphan(const char *name, bool runtime, X x);
template <class T, typename X, typename Y>
shared_ptr<T> createOrphan(const char *name, bool runtime, X x, Y y);
template <class T, typename X, typename Y, typename Z>
shared_ptr<T> createOrphan(const char *name, bool runtime, X x, Y y, Z z);
template <class T, typename X, typename Y, typename Z, typename ZZ>
shared_ptr<T> createOrphan(const char *name, bool runtime, X x, Y y, Z z, ZZ zz);

#define XNODE_OBJECT  template <class _T> \
	friend shared_ptr<_T> createOrphan(const char *name, bool runtime); \
	template <class _T, typename _X> \
	friend shared_ptr<_T> createOrphan(const char *name, bool runtime, _X x); \
	template <class _T, typename _X, typename _Y> \
	friend shared_ptr<_T> createOrphan(const char *name, bool runtime, _X x, _Y y); \
	template <class _T, typename _X, typename _Y, typename _Z> \
	friend shared_ptr<_T> createOrphan(const char *name, bool runtime, _X x, _Y y, _Z z); \
	template <class _T, typename _X, typename _Y, typename _Z, typename _ZZ> \
	friend shared_ptr<_T> createOrphan(const char *name, bool runtime, _X x, _Y y, _Z z, _ZZ zz);

//! XNode supports load/save for scripts/GUI, and signaling among threads.
//! \sa create(), createOrphan()
class XNode : public enable_shared_from_this<XNode>, public Node {
	//! Use XNODE_OBJECT macro in sub-classes.
	XNODE_OBJECT
protected:
	explicit XNode(const char *name, bool runtime = false);
public:
	virtual ~XNode();  

	template <class T>
	shared_ptr<T> create(const char *name, bool runtime = false);
	template <class T, typename X>
	shared_ptr<T> create(const char *name, bool runtime, X x);
	template <class T, typename X, typename Y>
	shared_ptr<T> create(const char *name, bool runtime, X x, Y y);
	template <class T, typename X, typename Y, typename Z>
	shared_ptr<T> create(const char *name, bool runtime, X x, Y y, Z z);
	template <class T, typename X, typename Y, typename Z, typename ZZ>
	shared_ptr<T> create(const char *name, bool runtime, X x, Y y, Z z, ZZ zz);

	//! \return internal/script name. Use latin1 chars.
	XString getName() const;
	//! \return i18n name for UI.
	virtual XString getLabel() const {return getName();}
	XString getTypename() const;

	shared_ptr<XNode> getChild(const XString &var) const;
	shared_ptr<XNode> getParent() const;

	shared_ptr<const NodeList> children() const {
		Snapshot shot(*this);
		return shot.list();
	}

	void clearChildren();
	int releaseChild(const shared_ptr<XNode> &node);

	bool isRunTime() const {return (**shared_from_this())->isRuntime();}

	//! If true, operation allowed by GUI
	//! \sa SetUIEnabled()
	bool isUIEnabled() const {return (**shared_from_this())->isUIEnabled();}
	//! Enable/Disable control over GUI
	void setUIEnabled(bool v);
	//! Disable all operations on this node forever.
	void disable();
	//! Touch signaling
	void touch();

	//! After touching
	//! \sa touch()
	XTalker<shared_ptr<XNode> > &onTouch() {return m_tlkOnTouch;}
	//! If true, operation allowed by GUI
	//! \sa setUIEnabled
	XTalker<shared_ptr<XNode> > &onUIEnabled() {return m_tlkOnUIEnabled;}

	virtual void insert(const shared_ptr<XNode> &ptr);

	//! Data holder.
	struct Payload : public Node::Payload {
		Payload(Node &node) : Node::Payload(node), m_flags(NODE_UI_ENABLED) {}
		Payload(const Payload &x) : Node::Payload(x), m_flags(x.m_flags) {}
		virtual ~Payload() {
		}

		bool isUIEnabled() const {return m_flags & NODE_UI_ENABLED;}
		void setUIEnabled(bool var) {
			if(isDisabled()) return;
			m_flags = (m_flags & ~NODE_UI_ENABLED) | (var ? NODE_UI_ENABLED : 0);
		}
		bool isDisabled() const {return m_flags & NODE_DISABLED;}
		void disable() {m_flags = (m_flags & ~NODE_DISABLED) | NODE_DISABLED;}
		bool isRuntime() const {return m_flags & NODE_RUNTIME;}
		void setRuntime(bool var) {m_flags = (m_flags & ~NODE_RUNTIME) | (var ? NODE_RUNTIME : 0);}

	private:
		int m_flags;
	};
	enum FLAG {NODE_UI_ENABLED = 0x1, NODE_DISABLED = 0x2, NODE_RUNTIME = 0x4};

protected: 
	XTalker<shared_ptr<XNode> > m_tlkOnTouch;
	XTalker<shared_ptr<XNode> > m_tlkOnUIEnabled;
private:
	const XString m_name;

	static XThreadLocal<std::deque<shared_ptr<XNode> > > stl_thisCreating;
	weak_ptr<XNode> m_parent;
};


template <class T>
shared_ptr<T>
XNode::create(const char *name, bool runtime)
{
	shared_ptr<T> ptr(createOrphan<T>(name, runtime));
	insert(ptr);
	return ptr;
}
template <class T, typename X>
shared_ptr<T>
XNode::create(const char *name, bool runtime, X x)
{
	shared_ptr<T> ptr(createOrphan<T>(name, runtime, x));
	insert(ptr);
	return ptr;
}
template <class T, typename X, typename Y>
shared_ptr<T>
XNode::create(const char *name, bool runtime, X x, Y y)
{
	shared_ptr<T> ptr(createOrphan<T>(name, runtime, x, y));
	insert(ptr);
	return ptr;
}
template <class T, typename X, typename Y, typename Z>
shared_ptr<T>
XNode::create(const char *name, bool runtime, X x, Y y, Z z)
{
	shared_ptr<T> ptr(createOrphan<T>(name, runtime, x, y, z));
	insert(ptr);
	return ptr;
}
template <class T, typename X, typename Y, typename Z, typename ZZ>
shared_ptr<T>
XNode::create(const char *name, bool runtime, X x, Y y, Z z, ZZ zz)
{
	shared_ptr<T> ptr(createOrphan<T>(name, runtime, x, y, z, zz));
	insert(ptr);
	return ptr;
}

template <class T>
shared_ptr<T>
createOrphan(const char *name, bool runtime)
{
	new T(name, runtime);
	shared_ptr<T> ptr = dynamic_pointer_cast<T>(XNode::stl_thisCreating->back());
	XNode::stl_thisCreating->pop_back();
	return ptr;
}
template <class T, typename X>
shared_ptr<T>
createOrphan(const char *name, bool runtime, X x)
{
	new T(name, runtime, x);
	shared_ptr<T> ptr = dynamic_pointer_cast<T>(XNode::stl_thisCreating->back());
	XNode::stl_thisCreating->pop_back();
	return ptr;
}
template <class T, typename X, typename Y>
shared_ptr<T>
createOrphan(const char *name, bool runtime, X x, Y y)
{
	new T(name, runtime, x, y);
	shared_ptr<T> ptr = dynamic_pointer_cast<T>(XNode::stl_thisCreating->back());
	XNode::stl_thisCreating->pop_back();
	return ptr;
}
template <class T, typename X, typename Y, typename Z>
shared_ptr<T>
createOrphan(const char *name, bool runtime, X x, Y y, Z z)
{
	new T(name, runtime, x, y, z);
	shared_ptr<T> ptr = dynamic_pointer_cast<T>(XNode::stl_thisCreating->back());
	XNode::stl_thisCreating->pop_back();
	return ptr;
}
template <class T, typename X, typename Y, typename Z, typename ZZ>
shared_ptr<T>
createOrphan(const char *name, bool runtime, X x, Y y, Z z, ZZ zz)
{
	new T(name, runtime, x, y, z, zz);
	shared_ptr<T> ptr = dynamic_pointer_cast<T>(XNode::stl_thisCreating->back());
	XNode::stl_thisCreating->pop_back();
	return ptr;
}

//---------------------------------------------------------------------------
#endif
