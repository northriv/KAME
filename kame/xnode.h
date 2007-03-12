/***************************************************************************
		Copyright (C) 2002-2007 Kentaro Kitagawa
		                   kitagawa@scphys.kyoto-u.ac.jp
		
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

template <class T>
shared_ptr<T> createOrphan(const char *name, bool runtime = false);
template <class T, typename X>
shared_ptr<T> createOrphan(const char *name, bool runtime, X x);
template <class T, typename X, typename Y>
shared_ptr<T> createOrphan(const char *name, bool runtime, X x, Y y);
template <class T, typename X, typename Y, typename Z>
shared_ptr<T> createOrphan(const char *name, bool runtime, X x, Y y, Z z);
template <class T, typename X, typename Y, typename Z, typename ZZ>
shared_ptr<T> createOrphan(const char *name, bool runtime, X x, Y y, Z z, ZZ z);

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
  
//! XNode supports loading/saveing to XML, GUI framework, basic signaling among threads
class XNode : public enable_shared_from_this<XNode>
#ifdef HAVE_LIBGCCPP
			, public kame_gc
#endif
{
	//! use XNODE_OBJECT in sub-classes
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
	shared_ptr<T> create(const char *name, bool runtime, X x, Y y, Z z, ZZ z);
  
	//! \return internal/script name. Use latin1 chars.
	std::string getName() const;
	//! \return i18n name for UI.
	virtual std::string getLabel() const {return getName();}
	std::string getTypename() const;
 
	shared_ptr<XNode> getChild(const std::string &var) const;

	typedef std::deque<shared_ptr<XNode> > NodeList;
	//! \return null if node has no child.
	atomic_shared_ptr<const NodeList> children() const {return m_children;}
  
	void clearChildren();
	int releaseChild(const shared_ptr<XNode> &node);

	bool isRunTime() const {return m_bRunTime;}

	//! If true, operation allowed by GUI
	//! \sa SetUIEnabled()
	bool isUIEnabled() const {return m_bUIEnabled;}
	//! Enable/Disable control over GUI
	void setUIEnabled(bool v);
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
	atomic_shared_ptr<NodeList> m_children;
  
	XTalker<shared_ptr<XNode> > m_tlkOnTouch;
	XTalker<shared_ptr<XNode> > m_tlkOnUIEnabled;
private:
	const std::string m_name;
	const bool m_bRunTime;
	bool m_bUIEnabled;
  
	static XThreadLocal<NodeList> stl_thisCreating;
};

//! Base class containing values
class XValueNodeBase : public XNode
{
	XNODE_OBJECT
protected:
	explicit XValueNodeBase(const char *name, bool runtime = false);
public:
	//! Get value as a string, which is used as XML meta data.
	virtual std::string to_str() const = 0;
	//! Set value as a string, which is used as XML meta data.
	//! throw exception when validator throws.
	void str(const QString &str) throw (XKameError &);
	void str(const std::string &str) throw (XKameError &);

	typedef void (*Validator)(std::string &);
	//! validator can throw \a XKameError, if it detects conversion errors.
	//! never insert when str() may be called.
	void setValidator(Validator);

	XTalker<shared_ptr<XValueNodeBase> > &beforeValueChanged()
		{return m_tlkBeforeValueChanged;}
	XTalker<shared_ptr<XValueNodeBase> > &onValueChanged() 
		{return m_tlkOnValueChanged;}
protected:
	virtual void _str(const std::string &str) throw (XKameError &) = 0;
  
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
	virtual std::string to_str() const;
	virtual void value(const double &t);
	virtual operator double() const;
	const char *format() const;
	void setFormat(const char* format);
protected:
	virtual void _str(const std::string &str) throw (XKameError &);
private:
	atomic_shared_ptr<double> m_var;
	std::string m_format;
	XRecursiveMutex m_valuemutex;
};

class XStringNode : public XValueNodeBase
{
	XNODE_OBJECT
protected:
	explicit XStringNode(const char *name, bool runtime = false);
public:
	virtual ~XStringNode() {}
	virtual std::string to_str() const;
	virtual operator std::string() const;
	void operator=(const std::string &str);
	virtual void value(const std::string &t);
protected:
	virtual void _str(const std::string &str) throw (XKameError &);
private:
	atomic_shared_ptr<std::string> m_var;
	XRecursiveMutex m_valuemutex;
};

//! Base class for atomic reading value node.
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
	virtual std::string to_str() const;
	virtual void value(const T &t);
protected:
	virtual void _str(const std::string &str) throw (XKameError &);
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
