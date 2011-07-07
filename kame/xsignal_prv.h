/***************************************************************************
		Copyright (C) 2002-2011 Kentaro Kitagawa
		                   kitag@issp.u-tokyo.ac.jp
		
		This program is free software; you can redistribute it and/or
		modify it under the terms of the GNU Library General Public
		License as published by the Free Software Foundation; either
		version 2 of the License, or (at your option) any later version.
		
		You should have received a copy of the GNU Library General 
		Public License and a list of authors along with this program; 
		see the files COPYING and AUTHORS.
***************************************************************************/
#ifndef XSIGNAL_PRV_H_
#define XSIGNAL_PRV_H_

template <class Arg>
class XListenerImpl__ : public XListener {
protected:
	XListenerImpl__(XListener::FLAGS flags)
		: XListener(flags), arg() {}
public:
	virtual ~XListenerImpl__() {}
	virtual void operator() (const Arg &) const = 0;
	//! is used when m_bAvoidDup is on.
	atomic_scoped_ptr<Arg> arg;
};
template <class Arg>
class XListenerStatic__ : public XListenerImpl__<Arg> {
	friend class XTalker<Arg>;
protected:
	XListenerStatic__(void (*func)(const Arg &),
					 XListener::FLAGS flags) :
		XListenerImpl__<Arg>(flags), m_func(func) {
    }
public:
	virtual void operator() (const Arg &x) const {
		(*m_func)(x);
	}
private:
	void (*const m_func)(const Arg &);
};
template <class tClass, class Arg>
class XListenerWeak__ : public XListenerImpl__<Arg> {
	friend class XTalker<Arg>;
protected:
	XListenerWeak__(const shared_ptr<tClass> &obj, void (tClass::*func)(const Arg &),
				   XListener::FLAGS flags) :
		XListenerImpl__<Arg>(flags), m_func(func), m_obj(obj) {
    }
public:
	virtual void operator() (const Arg &x) const {
		if(shared_ptr<tClass> p = m_obj.lock() ) ((p.get())->*m_func)(x);
	}
private:
	void (tClass::*const m_func)(const Arg &);
	const weak_ptr<tClass> m_obj;
};
template <class tClass, class Arg>
class XListenerShared__ : public XListenerImpl__<Arg> {
	friend class XTalker<Arg>;
protected:
	XListenerShared__(const shared_ptr<tClass> &obj, void (tClass::*func)(const Arg &),
					 XListener::FLAGS flags) :
		XListenerImpl__<Arg>(flags), m_obj(obj), m_func(func)   {
        ASSERT(obj);
	}
public:
	virtual void operator() (const Arg &x) const {((m_obj.get())->*m_func)(x);}
private:
	void (tClass::*m_func)(const Arg &);
	const shared_ptr<tClass> m_obj;
};

#endif /*XSIGNAL_PRV_H_*/
