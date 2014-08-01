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
#ifndef INTERFACE_H_
#define INTERFACE_H_

#include "xnode.h"
#include "xlistnode.h"
#include "xitemnode.h"
#include <vector>

class XDriver;
//! virtual class for communication devices.
//! \sa XCharInterface
class XInterface : public XNode {
public:
	XInterface(const char *name, bool runtime, const shared_ptr<XDriver> &driver);
	virtual ~XInterface() {}
 
	struct XInterfaceError : public XKameError {
		XInterfaceError(const XString &msg, const char *file, int line);
        virtual ~XInterfaceError() throw() {}
	};
	struct XConvError : public XInterfaceError {
		XConvError(const char *file, int line);
        virtual ~XConvError() throw() {}
	};
	struct XCommError : public XInterfaceError {
		XCommError(const XString &, const char *file, int line);
        virtual ~XCommError() throw() {}
	};
	struct XOpenInterfaceError : public XInterfaceError {
		XOpenInterfaceError(const char *file, int line);
        virtual ~XOpenInterfaceError() throw() {}
	};
 
	void setLabel(const XString& str) {m_label = str;}
	virtual XString getLabel() const;
 
	shared_ptr<XDriver> driver() const {return m_driver.lock();}
	//! type of interface or driver.
	const shared_ptr<XComboNode> &device() const {return m_device;}
	//! port number or device name.
	const shared_ptr<XStringNode> &port() const {return m_port;}
	//! e.g. GPIB address.
	const shared_ptr<XUIntNode> &address() const {return m_address;}
	//! True if interface is opened. Start/stop interface.
	const shared_ptr<XBoolNode> &control() const {return m_control;}

	void lock() {m_mutex.lock();}
	void unlock() {m_mutex.unlock();}
	bool isLocked() const {return m_mutex.isLockedByCurrentThread();}

	XRecursiveMutex &mutex() {return m_mutex;}
    
	virtual bool isOpened() const = 0;

	void start();
	void stop();
  
	struct Payload : public XNode::Payload {
		Talker<XInterface*, XInterface*> &onOpen() {return m_tlkOnOpen;}
		const Talker<XInterface*, XInterface*> &onOpen() const {return m_tlkOnOpen;}
		Talker<XInterface*, XInterface*> &onClose() {return m_tlkOnClose;}
		const Talker<XInterface*, XInterface*> &onClose() const {return m_tlkOnClose;}
	protected:
		Talker<XInterface*, XInterface*> m_tlkOnOpen;
		Talker<XInterface*, XInterface*> m_tlkOnClose;
	};
protected:  
	virtual void open() throw (XInterfaceError &) = 0;
	//! This can be called even if has already closed.
	virtual void close() throw (XInterfaceError &) = 0;
private:
	void onControlChanged(const Snapshot &shot, XValueNodeBase *);

	const weak_ptr<XDriver> m_driver;
	const shared_ptr<XComboNode> m_device;
	const shared_ptr<XStringNode> m_port;
	const shared_ptr<XUIntNode> m_address;
	const shared_ptr<XBoolNode> m_control;

	shared_ptr<XListener> lsnOnControlChanged;
      
	XRecursiveMutex m_mutex;
  
	XString m_label;
};

class XInterfaceList : public XAliasListNode<XInterface> {
public:
	XInterfaceList(const char *name, bool runtime) : XAliasListNode<XInterface>(name, runtime) {}
};

#endif /*INTERFACE_H_*/
