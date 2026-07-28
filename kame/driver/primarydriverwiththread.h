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
#include <cstdlib>

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

    //! Whether an acquisition loop should raise itself to
    //! `Priority::HIGHEST` for the duration of the loop (NOT for the setup
    //! commit that precedes it — that one runs once at driver start, often
    //! while a .kam load is starting many drivers at once, and is exactly the
    //! case where several impolite threads hurt).
    //!
    //! HIGHEST is not a priority: it only stops this thread from waiting, and
    //! nothing makes its peers wait for it.  For an acquisition loop that is
    //! the right shape anyway — the record commit has no give-up path, so
    //! "keep attempting" beats "sleep", and a wait budget would only convert
    //! waiting into retrying without releasing the loop.  The risk is
    //! entirely in the count: one such thread among polite peers measured
    //! 5 slow commits in 4 s, two collided and went to 1334, four cost 10x
    //! throughput.  Real drivers commit disjoint subtrees so they should not
    //! collide, but that has not been measured on real hardware, which is why
    //! this is opt-in per driver AND gated at runtime.
    //!
    //! Set `KAME_ACQ_HIGHEST=1` to enable.  Read once per process.
    static bool acqHighestEnabled() {
        static const bool s_enabled = []{
            const char *v = std::getenv("KAME_ACQ_HIGHEST");
            return v && *v && *v != '0';
        }();
        return s_enabled;
    }
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
