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
    //! **It grants no STM priority, and that is the whole history of this
    //! class.**  It used to raise the loop to `Priority::HIGHEST`, and the
    //! field and the lab converged on a structural incompatibility at the tier
    //! contracts' meeting point: HIGHEST never waits (its fair-mode immunity IS
    //! the contract), so it is the one contender a NORMAL transaction's
    //! privilege cannot stop.  When any privilege-holding transaction's closure
    //! takes longer than the HIGHEST commit period (closure x rate >= 1 — e.g.
    //! a 20 ms PNR analysis against a 50 /s record stream), it resonates into
    //! quasi-starvation, re-running its closure every record (measured: 1.1 ->
    //! 15.5 closure runs per commit) while its privilege pins every OTHER
    //! negotiator — a system-wide freeze that ends in the HANG watchdog.  At
    //! NORMAL the acquisition commits negotiate like everyone, fair-mode works
    //! on them, and the same load runs clean.  (User verdict, 2026-07-31; the
    //! reasoning is in kamestm/design/RT_READINESS.md.)
    //!
    //! The STM half was a `ScopedPriority(Priority::NORMAL)` base for a while
    //! after that verdict, which was dead weight: `execute_internal` below
    //! already declares NORMAL at thread entry, so the base saved NORMAL and
    //! restored NORMAL.  Removed 2026-08-14 — an RAII that restores what was
    //! never changed only invites the reader to believe a tier is in play here.
    //!
    //! CPU preference is a thread property with no fair-mode immunity, so the
    //! OS half keeps the acquisition thread scheduled without letting it starve
    //! anyone at the STM level.  The kamestm HIGHEST tier itself remains
    //! available to hosts that can honour its deployment precondition (HIGHEST
    //! commit rate x longest peer closure << 1); KAME with per-record analyses
    //! cannot.
    //! Spelt as an alias because a driver's EXTRA acquisition threads — DMA
    //! writers, async chunk readers, DSO read loops — need exactly this object
    //! and cannot all name it here (one of them is not under this class at all;
    //! see \a ScopedAcquisitionOSPriority in primarydriver.h).  One
    //! implementation, so the two cannot drift into meaning different things.
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
