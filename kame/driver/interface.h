/***************************************************************************
		Copyright (C) 2002-2015 Kentaro Kitagawa
		                   kitag@issp.u-tokyo.ac.jp
		
		This program is free software; you can redistribute it and/or
		modify it under the terms of the GNU General Public
		License as published by the Free Software Foundation; either
		version 2 of the License, or (at your option) any later version.
		
		You should have received a copy of the GNU General 
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
class DECLSPEC_KAME XInterface : public XNode {
public:
	XInterface(const char *name, bool runtime, const shared_ptr<XDriver> &driver);
 
    struct DECLSPEC_KAME XInterfaceError : public XKameError {
		XInterfaceError(const XString &msg, const char *file, int line);
	};
    struct DECLSPEC_KAME XConvError : public XInterfaceError {
		XConvError(const char *file, int line);
	};
    struct DECLSPEC_KAME XCommError : public XInterfaceError {
		XCommError(const XString &, const char *file, int line);
	};
    struct DECLSPEC_KAME XOpenInterfaceError : public XInterfaceError {
		XOpenInterfaceError(const char *file, int line);
	};
    struct DECLSPEC_KAME XUnsupportedFeatureError : public XInterfaceError {
        XUnsupportedFeatureError(const char *file, int line);
    };

	void setLabel(const XString& str) {m_label = str;}
    virtual XString getLabel() const override;
 
	shared_ptr<XDriver> driver() const {return m_driver.lock();}
	//! type of interface or driver.
	const shared_ptr<XComboNode> &device() const {return m_device;}
	//! port number or device name.
	const shared_ptr<XStringNode> &port() const {return m_port;}
	//! e.g. GPIB address.
	const shared_ptr<XUIntNode> &address() const {return m_address;}
	//! True if interface is opened. Start/stop interface.
	const shared_ptr<XBoolNode> &control() const {return m_control;}

    //! Taking this lock while a Transaction is alive on the calling thread is
    //! CLAUDE.md driver-authoring rule 5: interface I/O inside a transaction.
    //! It re-runs on every CAS retry (re-issuing device commands), and it takes
    //! a plain mutex from inside an in-flight transaction, which is the
    //! deadlock class behind the 2026-07-10 negotiation stall.
    //!
    //! This is the one place that catches it completely.  Every I/O path passes
    //! here: the 185 explicit `XScopedLock<XInterface>` sites in drivers, and
    //! the verbs' own internal locks (e.g. XCharInterface::send).  So the check
    //! sees arbitrary indirection depth and virtual dispatch, which no static
    //! analysis of ours can — tools/audit/check_stm_closures.py only reaches
    //! one call from the closure, and "am I inside a transaction" is a dynamic
    //! property of the call stack that C++ has no way to express in a type.
    //!
    //! The predicate is exact rather than approximate: `detail::s_tx_nest` is
    //! held for a Transaction's whole lifetime (its AcquireOneCount is a value
    //! member) but only during a Snapshot's *construction* (a ctor local), so
    //! ordinary `Snapshot shot(*this); ... interface()->query(...)` driver code
    //! — the XDSO acquisition loop, for one — does not trip it.
    virtual void lock() {
#ifndef NDEBUG
        reportIfInTransaction_();
#endif
        m_mutex.lock();
    }
    virtual void unlock() {m_mutex.unlock();}

	XRecursiveMutex &mutex() {return m_mutex;}

#ifndef NDEBUG
    //! Debug-only diagnostic for lock().  Reports once per (driver, priority)
    //! and counts; set KAME_STM_ABORT_IO_IN_TX=1 to abort on the first hit.
    //!
    //! Reports rather than aborting by default because this is a diagnostic and
    //! not a safeguard: an unknown number of sites may still reach it through
    //! indirection the audit cannot see, and a debug build that aborts on the
    //! first one is a debug build nobody runs.  The abort is there for whoever
    //! is actually hunting one.
    void reportIfInTransaction_() const noexcept;
#endif
    
	virtual bool isOpened() const = 0;

	void start();
	void stop();
  
    struct DECLSPEC_KAME Payload : public XNode::Payload {
        Talker<XInterface*> &onOpen() {return m_tlkOnOpen;}
        const Talker<XInterface*> &onOpen() const {return m_tlkOnOpen;}
        Talker<XInterface*> &onClose() {return m_tlkOnClose;}
        const Talker<XInterface*> &onClose() const {return m_tlkOnClose;}
	protected:
        Talker<XInterface*> m_tlkOnOpen;
        Talker<XInterface*> m_tlkOnClose;
	};
protected:  
    virtual void open() = 0;
	//! This can be called even if has already closed.
    virtual void close() = 0;
private:
	void onControlChanged(const Snapshot &shot, XValueNodeBase *);

	const weak_ptr<XDriver> m_driver;
	const shared_ptr<XComboNode> m_device;
	const shared_ptr<XStringNode> m_port;
	const shared_ptr<XUIntNode> m_address;
	const shared_ptr<XBoolNode> m_control;

	shared_ptr<Listener> lsnOnControlChanged;
      
	XRecursiveMutex m_mutex;
    unique_ptr<XThread> m_threadStart;
	XString m_label;
};

class DECLSPEC_KAME XInterfaceList : public XAliasListNode<XInterface> {
public:
	XInterfaceList(const char *name, bool runtime) : XAliasListNode<XInterface>(name, runtime) {}
};

#endif /*INTERFACE_H_*/
