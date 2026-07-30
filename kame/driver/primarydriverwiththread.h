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
    //! RAII guard raising an acquisition loop to `Priority::HIGHEST`.
    //!
    //! Construct it immediately before the `while( !terminated)` loop — never
    //! around the setup commit that precedes it.  That commit runs once at
    //! driver start, often while a .kam load is starting many drivers at once,
    //! which is the one case where several impolite threads hurt each other.
    //! Everything inside the loop, by contrast, belongs to the record: the
    //! settings Snapshots (`***someNode()` expands to a SingleSnapshot, see
    //! kame/xnode.h, so those negotiate too), the hardware I/O, and the record
    //! commit(s).  None of it should be polite.
    //!
    //! **Unconditional on purpose.**  This is a safeguard against unforeseen
    //! contention — a .kam load, a script or an MCP session snapshotting the
    //! measurement root, a graph redraw bundling an ancestor — and a safeguard
    //! that has to be switched on ahead of time is not one, because nobody
    //! predicts the unforeseen.  It is also free until it is needed:
    //! `ScopedNegotiateLinkage::_negotiate()` returns `[[likely]]` early when
    //! no peer has tagged the linkage, so `_negotiate_internal()` — the only
    //! place that looks at the priority at all — is reached only under real
    //! contention.  Until then a HIGHEST acquisition thread behaves bit-for-bit
    //! like a NORMAL one.
    //!
    //! **What it costs when it does act, stated plainly.**  HIGHEST breaks out
    //! of the negotiator's round loop before the sleep path, which is where
    //! `fair_mode_blocks_me` gates on a peer's privilege stamp — so a HIGHEST
    //! thread ignores privilege entirely.  An ancestor-scope operation can
    //! therefore no longer be protected by privilege against acquisition
    //! threads, and with several drivers acquiring it can be starved for as
    //! long as they keep acquiring.  That is a deliberate policy choice:
    //! measurement beats UI and scripting.  Note the record-commit counters
    //! above do NOT see it — they only count the acquisition side — so a
    //! starved .kam load or redraw has to be noticed by other means.
    //! **STM-HIGHEST is retired for KAME (user verdict, 2026-07-31)** — this
    //! RAII now grants only the OS-level elevation.  The field and the lab
    //! converged on a structural incompatibility at the tier contracts'
    //! meeting point: HIGHEST never waits (its fair-mode immunity IS the
    //! contract), so it is the one contender a NORMAL transaction's privilege
    //! cannot stop.  When any privilege-holding transaction's closure takes
    //! longer than the HIGHEST commit period (closure x rate >= 1 — e.g. a
    //! 20 ms PNR analysis against a 50 /s record stream), it resonates into
    //! quasi-starvation, re-running its closure every record (measured: 1.1
    //! -> 15.5 closure runs per commit) while its privilege pins every OTHER
    //! negotiator — a system-wide freeze that ends in the HANG watchdog.
    //! At NORMAL the acquisition commits negotiate like everyone, fair-mode
    //! works on them, and the same load runs clean.
    //!
    //! The OS half stays: CPU preference is a thread property with no
    //! fair-mode immunity, so it keeps the acquisition thread scheduled
    //! without letting it starve anyone at the STM level.  The kamestm
    //! HIGHEST tier itself remains available to hosts that can honour its
    //! deployment precondition (HIGHEST commit rate x longest peer closure
    //! << 1); KAME with per-record analyses cannot.
    class AcquisitionPriority : public Transactional::ScopedPriority {
    public:
        AcquisitionPriority()
            : Transactional::ScopedPriority(
                  Transactional::Priority::NORMAL) {
            raiseAcquisitionOSPriority_();
        }
        ~AcquisitionPriority() {
            restoreAcquisitionOSPriority_();
        }
    };

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
