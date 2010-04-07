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

class XNode;

typedef Transactional::Snapshot<XNode> Snapshot;
typedef Transactional::Transaction<XNode> Transaction;
template <class T>
struct SingleSnapshot : public Transactional::SingleSnapshot<XNode, T> {
	explicit SingleSnapshot(const T&node) :  Transactional::SingleSnapshot<XNode, T>(node) {}
};
template <class T>
struct SingleTransaction : public Transactional::SingleTransaction<XNode, T> {
	explicit SingleTransaction(T&node) :  Transactional::SingleTransaction<XNode, T>(node) {}
};

#define trans(node) for(Transaction \
	__implicit_tr(node, false); !__implicit_tr.isModified() || !__implicit_tr.commitOrNext(); ) __implicit_tr[node]

template <class T>
typename boost::enable_if<boost::is_base_of<XNode, T>,
	const SingleSnapshot<T> >::type
 operator*(T &node) {
	return SingleSnapshot<T>(node);
}

template <typename tArg, typename tArgRef = const tArg &>
class Talker : public Transactional::Talker<XNode, tArg, tArgRef> {};
template <typename tArg, typename tArgRef = const tArg &>
class TalkerSingleton : public virtual Transactional::TalkerSingleton<XNode, tArg, tArgRef>, public virtual Talker<tArg, tArgRef>  {};

#define XNODE_OBJECT #warning XNODE_OBJECT is obsolete.

//! XNode supports load/save for scripts/GUI, and signaling among threads.
//! \sa create(), createOrphan()
class XNode : public enable_shared_from_this<XNode>, public Transactional::Node<XNode> {
public:
	explicit XNode(const char *name, bool runtime = false);
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

	template <class T>
	shared_ptr<T> create(Transaction &tr, const char *name, bool runtime = false);
	template <class T, typename X>
	shared_ptr<T> create(Transaction &tr, const char *name, bool runtime, X x);
	template <class T, typename X, typename Y>
	shared_ptr<T> create(Transaction &tr, const char *name, bool runtime, X x, Y y);
	template <class T, typename X, typename Y, typename Z>
	shared_ptr<T> create(Transaction &tr, const char *name, bool runtime, X x, Y y, Z z);
	template <class T, typename X, typename Y, typename Z, typename ZZ>
	shared_ptr<T> create(Transaction &tr, const char *name, bool runtime, X x, Y y, Z z, ZZ zz);

	template <class _T>
	static shared_ptr<_T> createOrphan(const char *name, bool runtime = false);
	template <class _T, typename _X>
	static shared_ptr<_T> createOrphan(const char *name, bool runtime, _X x);
	template <class _T, typename _X, typename _Y>
	static shared_ptr<_T> createOrphan(const char *name, bool runtime, _X x, _Y y);
	template <class _T, typename _X, typename _Y, typename _Z>
	static shared_ptr<_T> createOrphan(const char *name, bool runtime, _X x, _Y y, _Z z);
	template <class _T, typename _X, typename _Y, typename _Z, typename _ZZ>
	static shared_ptr<_T> createOrphan(const char *name, bool runtime, _X x, _Y y, _Z z, _ZZ zz);

	//! \return internal/script name. Use latin1 chars.
	XString getName() const;
	//! \return i18n name for UI.
	virtual XString getLabel() const {return getName();}
	XString getTypename() const;

	shared_ptr<XNode> getChild(const XString &var) const;
	shared_ptr<XNode> getParent() const;

	void clearChildren();
	int releaseChild(const shared_ptr<XNode> &node);

	//! Enables/Disables control over GUI
	void setUIEnabled(bool v);
	//! Disables all operations on this node forever.
	void disable();

	virtual void insert(const shared_ptr<XNode> &ptr);
	virtual bool insert(Transaction &tr, const shared_ptr<XNode> &ptr, bool online_after_insertion = false);

	//! Data holder.
	struct Payload : public Transactional::Node<XNode>::Payload {
		Payload() : Transactional::Node<XNode>::Payload(), m_flags(NODE_UI_ENABLED) {}
		//! If true, operations are allowed by UI and scripts.
		bool isUIEnabled() const {return m_flags & NODE_UI_ENABLED;}
		void setUIEnabled(bool var);
		bool isDisabled() const {return m_flags & NODE_DISABLED;}
		void disable();
		bool isRuntime() const {return m_flags & NODE_RUNTIME;}
		void setRuntime(bool var) {m_flags = (m_flags & ~NODE_RUNTIME) | (var ? NODE_RUNTIME : 0);}
		//! \sa setUIEnabled
		Talker<XNode*, XNode*> &onUIFlagsChanged() {return m_tlkOnUIFlagsChanged;}
		const Talker<XNode*, XNode*> &onUIFlagsChanged() const {return m_tlkOnUIFlagsChanged;}
	private:
		int m_flags;
		TalkerSingleton<XNode*, XNode*> m_tlkOnUIFlagsChanged;
	};
	enum FLAG {NODE_UI_ENABLED = 0x1, NODE_DISABLED = 0x2, NODE_RUNTIME = 0x4};

protected: 
private:
	XNode(); //inhibited.
	const XString m_name;

	static XThreadLocal<std::deque<shared_ptr<XNode> > > stl_thisCreating;
};

class XTouchableNode : public XNode {
public:
	explicit XTouchableNode(const char *name, bool runtime) : XNode(name, runtime) {}

	struct Payload : public XNode::Payload {
		Payload() : XNode::Payload() {}
		void touch();
		//! \sa touch()
		Talker<XTouchableNode*, XTouchableNode*> &onTouch() {return m_tlkOnTouch;}
		const Talker<XTouchableNode*, XTouchableNode*> &onTouch() const {return m_tlkOnTouch;}
	protected:
		Talker<XTouchableNode*, XTouchableNode*> m_tlkOnTouch;
	};
protected:
};

//! Interface class containing values
class XValueNodeBase : public XNode {
protected:
	explicit XValueNodeBase(const char *name, bool runtime) : XNode(name, runtime), m_validator(0) {}
public:
	//! gets value as a string, which is used for scripting.
	XString to_str() const { return (**this)->to_str();}
	//! sets value as a string, which is used for scripting.
	//! It throws exception when the validator throws.
	void str(const XString &str) throw (XKameError &);

	typedef void (*Validator)(XString &);
	void setValidator(Validator x) {m_validator = x;}

	struct Payload : public XNode::Payload {
		Payload() : XNode::Payload() {}
		//! Gets value as a string, which is used for scripting.
		virtual XString to_str() const = 0;
		//! Sets value as a string, which is used for scripting.
		//! This throws exception when the validator throws.
		void str(const XString &str) throw (XKameError &) {
		    XString sc(str);
		    if(static_cast<XValueNodeBase&>(node()).m_validator)
		    	(*static_cast<XValueNodeBase&>(node()).m_validator)(sc);
		    _str(sc);
		    tr().mark(onValueChanged(), static_cast<XValueNodeBase*>(&node()));
		}
		Talker<XValueNodeBase*, XValueNodeBase*> &onValueChanged() {return m_tlkOnValueChanged;}
		const Talker<XValueNodeBase*, XValueNodeBase*> &onValueChanged() const {return m_tlkOnValueChanged;}
	protected:
		//! This may throw exception due to format issues.
		virtual void _str(const XString &) = 0;
		TalkerSingleton<XValueNodeBase*, XValueNodeBase*> m_tlkOnValueChanged;
	};
protected:
	Validator m_validator;
};

//! Base class for integer node.
template <typename T, int base = 10>
class XIntNodeBase : public XValueNodeBase {
public:
	explicit XIntNodeBase(const char *name, bool runtime = false)
	: XValueNodeBase(name, runtime) {}
	virtual ~XIntNodeBase() {}

	operator T() const {return **this;}
	void value(T x);

	struct Payload : public XValueNodeBase::Payload {
		Payload() : XValueNodeBase::Payload() {this->m_var = 0;}
		virtual XString to_str() const;
		operator T() const {return m_var;}
		Payload &operator=(T x) {
			m_var = x;
		    tr().mark(onValueChanged(), static_cast<XValueNodeBase*>(&node()));
			return *this;
		}
	protected:
		virtual void _str(const XString &);
		T m_var;
	};
};

class XDoubleNode : public XValueNodeBase {
public:
	explicit XDoubleNode(const char *name, bool runtime = false, const char *format = 0L);
	virtual ~XDoubleNode() {}

	operator double() const {return **this;}
	void value(double x);

	const char *format() const {return local_shared_ptr<XString>(m_format)->c_str();}
	void setFormat(const char* format);

	struct Payload : public XValueNodeBase::Payload {
		Payload() : XValueNodeBase::Payload() {this->m_var = 0.0;}
		virtual XString to_str() const;
		operator double() const {return m_var;}
		Payload &operator=(double x) {
			m_var = x;
		    tr().mark(onValueChanged(), static_cast<XValueNodeBase*>(&node()));
			return *this;
		}
	protected:
		virtual void _str(const XString &);
		double m_var;
	};
	atomic_shared_ptr<XString> m_format;
};

class XStringNode : public XValueNodeBase {
public:
	explicit XStringNode(const char *name, bool runtime = false);
	virtual ~XStringNode() {}

	operator XString() const {return **this;}
	void value(const XString &x);

	struct Payload : public XValueNodeBase::Payload {
		Payload() : XValueNodeBase::Payload() {}
		virtual XString to_str() const {return this->m_var;}
		operator const XString&() const {return m_var;}
		Payload &operator=(const XString &x) {
			m_var = x;
			tr().mark(onValueChanged(), static_cast<XValueNodeBase*>(&node()));
			return *this;
		}
	protected:
		virtual void _str(const XString &str) {m_var = str;}
		XString m_var;
	};
};

typedef XIntNodeBase<int> XIntNode;
typedef XIntNodeBase<unsigned int> XUIntNode;
typedef XIntNodeBase<long> XLongNode;
typedef XIntNodeBase<unsigned long> XULongNode;
typedef XIntNodeBase<bool> XBoolNode;
typedef XIntNodeBase<unsigned long, 16> XHexNode;

template <class T>
shared_ptr<T>
XNode::createOrphan(const char *name, bool runtime) {
	Transactional::Node<XNode>::create<T>(name, runtime);
	shared_ptr<T> ptr = dynamic_pointer_cast<T>(XNode::stl_thisCreating->back());
	XNode::stl_thisCreating->pop_back();
	return ptr;
}
template <class T, typename X>
shared_ptr<T>
XNode::createOrphan(const char *name, bool runtime, X x) {
	Transactional::Node<XNode>::create<T>(name, runtime, x);
	shared_ptr<T> ptr = dynamic_pointer_cast<T>(XNode::stl_thisCreating->back());
	XNode::stl_thisCreating->pop_back();
	return ptr;
}
template <class T, typename X, typename Y>
shared_ptr<T>
XNode::createOrphan(const char *name, bool runtime, X x, Y y) {
	Transactional::Node<XNode>::create<T>(name, runtime, x, y);
	shared_ptr<T> ptr = dynamic_pointer_cast<T>(XNode::stl_thisCreating->back());
	XNode::stl_thisCreating->pop_back();
	return ptr;
}
template <class T, typename X, typename Y, typename Z>
shared_ptr<T>
XNode::createOrphan(const char *name, bool runtime, X x, Y y, Z z) {
	Transactional::Node<XNode>::create<T>(name, runtime, x, y, z);
	shared_ptr<T> ptr = dynamic_pointer_cast<T>(XNode::stl_thisCreating->back());
	XNode::stl_thisCreating->pop_back();
	return ptr;
}
template <class T, typename X, typename Y, typename Z, typename ZZ>
shared_ptr<T>
XNode::createOrphan(const char *name, bool runtime, X x, Y y, Z z, ZZ zz) {
	Transactional::Node<XNode>::create<T>(name, runtime, x, y, z, zz);
	shared_ptr<T> ptr = dynamic_pointer_cast<T>(XNode::stl_thisCreating->back());
	XNode::stl_thisCreating->pop_back();
	return ptr;
}

template <class T>
shared_ptr<T>
XNode::create(Transaction &tr, const char *name, bool runtime) {
	shared_ptr<T> ptr(createOrphan<T>(name, runtime));
	if(ptr) insert(tr, ptr, true);
	return ptr;
}
template <class T, typename X>
shared_ptr<T>
XNode::create(Transaction &tr, const char *name, bool runtime, X x) {
	shared_ptr<T> ptr(createOrphan<T>(name, runtime, x));
	if(ptr) insert(tr, ptr, true);
	return ptr;
}
template <class T, typename X, typename Y>
shared_ptr<T>
XNode::create(Transaction &tr, const char *name, bool runtime, X x, Y y) {
	shared_ptr<T> ptr(createOrphan<T>(name, runtime, x, y));
	if(ptr) insert(tr, ptr, true);
	return ptr;
}
template <class T, typename X, typename Y, typename Z>
shared_ptr<T>
XNode::create(Transaction &tr, const char *name, bool runtime, X x, Y y, Z z) {
	shared_ptr<T> ptr(createOrphan<T>(name, runtime, x, y, z));
	if(ptr) insert(tr, ptr, true);
	return ptr;
}
template <class T, typename X, typename Y, typename Z, typename ZZ>
shared_ptr<T>
XNode::create(Transaction &tr, const char *name, bool runtime, X x, Y y, Z z, ZZ zz) {
	shared_ptr<T> ptr(createOrphan<T>(name, runtime, x, y, z, zz));
	if(ptr) insert(tr, ptr, true);
	return ptr;
}

template <class T>
shared_ptr<T>
XNode::create(const char *name, bool runtime) {
	shared_ptr<T> ptr(createOrphan<T>(name, runtime));
	if(ptr) insert(ptr);
	return ptr;
}
template <class T, typename X>
shared_ptr<T>
XNode::create(const char *name, bool runtime, X x) {
	shared_ptr<T> ptr(createOrphan<T>(name, runtime, x));
	if(ptr) insert(ptr);
	return ptr;
}
template <class T, typename X, typename Y>
shared_ptr<T>
XNode::create(const char *name, bool runtime, X x, Y y) {
	shared_ptr<T> ptr(createOrphan<T>(name, runtime, x, y));
	if(ptr) insert(ptr);
	return ptr;
}
template <class T, typename X, typename Y, typename Z>
shared_ptr<T>
XNode::create(const char *name, bool runtime, X x, Y y, Z z) {
	shared_ptr<T> ptr(createOrphan<T>(name, runtime, x, y, z));
	if(ptr) insert(ptr);
	return ptr;
}
template <class T, typename X, typename Y, typename Z, typename ZZ>
shared_ptr<T>
XNode::create(const char *name, bool runtime, X x, Y y, Z z, ZZ zz) {
	shared_ptr<T> ptr(createOrphan<T>(name, runtime, x, y, z, zz));
	if(ptr) insert(ptr);
	return ptr;
}

//---------------------------------------------------------------------------
#endif
