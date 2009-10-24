/***************************************************************************
		Copyright (C) 2002-2009 Kentaro Kitagawa
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

#include "support.h"
#include "xsignal.h"
#include "rwlock.h"
#include "atomic_list.h"

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
class XNode : public enable_shared_from_this<XNode>
#ifdef HAVE_LIBGCCPP
, public kame_gc
#endif
{
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

	typedef atomic_list<shared_ptr<XNode> > NodeList;

	NodeList::reader children() const {return NodeList::reader(m_children);}

	void clearChildren();
	int releaseChild(const shared_ptr<XNode> &node);

	bool isRunTime() const {return m_flags & FLAG_RUNTIME;}

	//! If true, operation allowed by GUI
	//! \sa SetUIEnabled()
	bool isUIEnabled() const {return m_flags & FLAG_UI_ENABLED;}
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
protected: 
	//! If false, all operations have to be disabled. 
	bool isEnabled() const {return m_flags & FLAG_ENABLED;}

	NodeList m_children;

	XTalker<shared_ptr<XNode> > m_tlkOnTouch;
	XTalker<shared_ptr<XNode> > m_tlkOnUIEnabled;
private:
	const XString m_name;
	enum {FLAG_RUNTIME = 0x1, FLAG_ENABLED = 0x2, FLAG_UI_ENABLED = 0x4};
	int m_flags;

	static XThreadLocal<std::deque<shared_ptr<XNode> > > stl_thisCreating;
	weak_ptr<XNode> m_parent;
};

//! Base class containing values
class XValueNodeBase : public XNode
{
	XNODE_OBJECT
protected:
	explicit XValueNodeBase(const char *name, bool runtime = false);
public:
	//! Get value as a string, which is used as XML meta data.
	virtual XString to_str() const = 0;
	//! Set value as a string, which is used as XML meta data.
	//! throw exception when validator throws.
	void str(const XString &str) throw (XKameError &);

	typedef void (*Validator)(XString &);
	//! validator can throw \a XKameError, if it detects conversion errors.
	//! never insert when str() may be called.
	void setValidator(Validator);

	XTalker<shared_ptr<XValueNodeBase> > &beforeValueChanged()
	{return m_tlkBeforeValueChanged;}
	XTalker<shared_ptr<XValueNodeBase> > &onValueChanged() 
	{return m_tlkOnValueChanged;}
protected:
	virtual void _str(const XString &str) throw (XKameError &) = 0;

	XTalker<shared_ptr<XValueNodeBase> > m_tlkBeforeValueChanged;
	XTalker<shared_ptr<XValueNodeBase> > m_tlkOnValueChanged;
	Validator m_validator;
};

class XDoubleNode : public XValueNodeBase
{
	XNODE_OBJECT
protected:
	explicit XDoubleNode(const char *name, bool runtime = false, const char *format = 0L);
public:
	virtual ~XDoubleNode() {}
	virtual XString to_str() const;
	virtual void value(const double &t);
	virtual operator double() const;
	const char *format() const;
	void setFormat(const char* format);
protected:
	virtual void _str(const XString &str) throw (XKameError &);
private:
	atomic<double> m_var;
	XString m_format;
	XRecursiveMutex m_valuemutex;
};

class XStringNode : public XValueNodeBase
{
	XNODE_OBJECT
protected:
	explicit XStringNode(const char *name, bool runtime = false);
public:
	virtual ~XStringNode() {}
	virtual XString to_str() const;
	virtual operator XString() const;
	virtual void value(const XString &t);
protected:
	virtual void _str(const XString &str) throw (XKameError &);
private:
	atomic_shared_ptr<XString> m_var;
	XRecursiveMutex m_valuemutex;
};

//! Base class for value node.
template <typename T, int base = 10>
class XValueNode : public XValueNodeBase
{
	XNODE_OBJECT
protected:
	explicit XValueNode(const char *name, bool runtime = false)
	: XValueNodeBase(name, runtime), m_var(0) {}
public:
	virtual ~XValueNode() {}
	virtual operator T() const {return m_var;}
	virtual XString to_str() const;
	virtual void value(const T &t);
protected:
	virtual void _str(const XString &str) throw (XKameError &);
	atomic<T> m_var;
	XRecursiveMutex m_valuemutex;
private:
};

typedef XValueNode<int> XIntNode;
typedef XValueNode<unsigned int> XUIntNode;
typedef XValueNode<bool> XBoolNode;
typedef XValueNode<unsigned int, 16> XHexNode;

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
