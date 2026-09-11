/***************************************************************************
        Copyright (C) 2002-2024 Kentaro Kitagawa
		                   kitag@issp.u-tokyo.ac.jp
		
		This program is free software; you can redistribute it and/or
		modify it under the terms of the GNU General Public
		License as published by the Free Software Foundation; either
		version 2 of the License, or (at your option) any later version.
		
		You should have received a copy of the GNU General 
		Public License and a list of authors along with this program; 
		see the files COPYING and AUTHORS.
***************************************************************************/
#ifndef PRIMARYDRIVERWITHTHREAD_H_
#define PRIMARYDRIVERWITHTHREAD_H_

#include "primarydriver.h"

class XPrimaryDriverWithThread : public XPrimaryDriver {
public:
    using XPrimaryDriver::XPrimaryDriver; //inherits constructors.
    virtual ~XPrimaryDriverWithThread() = default;
  
	//! Shuts down your threads, unconnects GUI, and deactivates signals.\n
	//! This function may be called even if driver has already stopped.
	//! This should not cause an exception.
    virtual void stop() override;

    class Payload : public XPrimaryDriver::Payload {};
protected:
	//! Starts up your threads, connects GUI, and activates signals.
	//! This function should not cause an exception.
    virtual void start() override;

	virtual void *execute(const atomic<bool> &terminated) = 0;

protected:
    //! RAII guard marking an acquisition loop for the OS scheduler.
    //!
    //! Construct it immediately before the `while( !terminated)` loop.  It
    //! spans the loop, the loop spans the thread, and the thread dies with the
    //! driver — so this is a set-once thread property in RAII clothing, which
    //! is what an OS scheduling class has to be (POSIX RT attributes are set at
    //! thread setup; MMCSS registers a thread once).
    //!
    //! **It grants no STM priority.**  CPU preference is a thread property; an
    //! STM tier is a transaction property.  This is only the former, so it keeps
    //! the acquisition thread scheduled without giving its commits any standing
    //! against other negotiators.  `execute_internal` below declares
    //! `Priority::NORMAL` at thread entry and nothing here changes it.
    //!
    //! An alias, not a class of its own: a driver's EXTRA acquisition threads
    //! — DMA writers, async chunk readers, DSO read loops — need exactly this
    //! object and cannot all name it here (one is not under this class at all;
    //! see \a ScopedAcquisitionOSPriority in primarydriver.h, whose doc block
    //! carries the rest of the history).  One implementation, so the two
    //! cannot drift into meaning different things.
    using AcquisitionPriority = ScopedAcquisitionOSPriority;

private:
    unique_ptr<XThread> m_thread;
	void *execute_internal(const atomic<bool> &terminated) {
        Transactional::setCurrentPriorityMode(Priority::NORMAL);

		void *ret = NULL;
		try {
			ret = execute(terminated);
		}
		catch(XKameError &e) {
			e.print(getLabel() + i18n(" Error: "));
		}
		closeInterface(); //closes interface if any.
        // One summary line per acquisition thread, and only when the driver
        // actually saw a slow record commit.  Off the hot path by construction
        // (the thread is exiting), and silent on a healthy driver — the same
        // report-at-scope-exit shape as kame::rt_section.
        if(slowRecordCommits())
            dbgPrint(formatString(
                "%s: %llu of %llu record commits took over %llu us "
                "(max %llu us) in the STM",
                getLabel().c_str(),
                (unsigned long long)slowRecordCommits(),
                (unsigned long long)recordCommits(),
                (unsigned long long)(SLOW_RECORD_COMMIT_NS / 1000ull),
                (unsigned long long)(maxRecordCommitNS() / 1000ull)));
		return ret;
	}
};

inline void
XPrimaryDriverWithThread::start() {
    m_thread.reset(new XThread(shared_from_this(),
        &XPrimaryDriverWithThread::execute_internal));
}

inline void
XPrimaryDriverWithThread::stop() {
    unique_ptr<XThread> thread = std::move(m_thread);
    if(thread && !thread->isTerminated()) {
        thread->terminate();
        m_thread = std::move(thread);
    }
	else
		closeInterface(); //closes interface if any.
}

#endif /*PRIMARYDRIVERWITHTHREAD_H_*/
