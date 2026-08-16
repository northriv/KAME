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
#ifndef PRIMARYDRIVER_H_
#define PRIMARYDRIVER_H_

#include "driver.h"
#include "interface.h"
#include <atomic>
#include <cstdint>

//! OS half of \a AcquisitionPriority (primarydriverwiththread.h): marks the
//! CALLING thread as an acquisition thread for the OS scheduler, for the
//! thread's lifetime.  Deliberately not coupled to STM priority changes -- an
//! OS scheduling class is a per-thread property, set once at thread setup
//! (POSIX RT and MMCSS practice alike), while STM priority toggles with
//! transaction phases.  Toggling the OS class along with it (the historic
//! Windows behaviour) handed the CPU to arbitrary threads for the demoted
//! downstream half of every acquisition cycle, which is backwards for meeting
//! the next trigger: the loop should finish its iteration at acquisition
//! priority and yield naturally in the device wait.
//!
//! Windows: THREAD_PRIORITY_TIME_CRITICAL (level 15 in the normal class,
//! documented, no privilege needed).  PREEMPT_RT, when it comes, goes HERE and
//! only here -- SCHED_FIFO/RR/DEADLINE and the numeric level are deployment
//! decisions (relative to threaded irqs, needing RLIMIT_RTPRIO), so this is a
//! single visible place to make them.  Elsewhere: no-op.
//! Returns the caller's previous OS priority so nested/inline use can restore
//! exactly what it inherited rather than assuming THREAD_PRIORITY_NORMAL.
DECLSPEC_KAME int raiseAcquisitionOSPriority_() noexcept;
DECLSPEC_KAME void restoreAcquisitionOSPriority_(int saved_priority) noexcept;

//! RAII for the OS half ALONE, for a realtime thread that never enters the STM.
//!
//! Pulser DMA writers, free-run trigger predictors and async chunk readers feed
//! hardware and take no Snapshot, so they have no use for an STM tier — but
//! they do need the CPU.  Historically they asked for it by calling
//! `Transactional::setCurrentPriorityMode(Priority::HIGHEST)`, whose Windows arm
//! mapped HIGHEST to THREAD_PRIORITY_TIME_CRITICAL.  That arm was removed when
//! STM priority was decoupled from the OS scheduler, which silently turned every
//! such call into a no-op: the thread lost TIME_CRITICAL and nothing replaced it
//! (a Windows thread does not inherit its creator's priority, and these threads
//! are separate XThreads that never construct \a AcquisitionPriority).  This is
//! what those call sites meant, said directly.
//!
//! Use this — not `setCurrentPriorityMode` — whenever the thread is realtime but
//! STM-free.  It is also the safer spelling for a function that can additionally
//! be called inline on someone else's thread: `setCurrentPriorityMode` is a
//! PERSISTENT thread mode with no restore, so one such call leaks its tier into
//! whatever runs next on that thread (`ScopedDemoteRealtime` cannot catch it —
//! it arms only when the thread was ALREADY HIGHEST on entry).
class DECLSPEC_KAME ScopedAcquisitionOSPriority {
public:
    ScopedAcquisitionOSPriority() noexcept
        : m_savedPriority(raiseAcquisitionOSPriority_()) {}
    ~ScopedAcquisitionOSPriority() noexcept {
        restoreAcquisitionOSPriority_(m_savedPriority);
    }
    ScopedAcquisitionOSPriority(const ScopedAcquisitionOSPriority &) = delete;
    ScopedAcquisitionOSPriority &operator=(const ScopedAcquisitionOSPriority &) = delete;
private:
    int m_savedPriority;
};

class DECLSPEC_KAME XPrimaryDriver : public XDriver {
public:
	XPrimaryDriver(const char *name, bool runtime, Transaction &tr_meas, const shared_ptr<XMeasure> &meas);
  
	//! Shows all forms belonging to driver
	virtual void showForms() = 0;
  
	//! Shuts down your threads, unconnects GUI, and deactivates signals.\n
	//! This function may be called even if driver has already stopped.
	//! This should not cause an exception.
	virtual void stop() = 0;

private:
	friend class XRawStreamRecordReader;
	friend class XRawStreamRecorder;
protected:
	//! Starts up your threads, connects GUI, and activates signals.
	//! This function should not cause an exception.
	virtual void start() = 0;
	//! Be called for closing interfaces.
	//! This function should not cause an exception.
	virtual void closeInterface() = 0;

public:
	//! These are FIFO.
    struct RawData : public std::vector<char> {
		//! Pushes raw data to raw record
		//! Use signed/unsigned char, int16_t(16bit), and int32_t for integers.
		//! IEEE 754 float and double for floting point numbers.
		//! Little endian bytes will be stored into thread-local \sa rawData().
		//! \sa pop(), rawData()
		template <typename tVar>
		inline void push(tVar);
	private:
		inline void push_char(char);
		inline void push_int16_t(int16_t);
		inline void push_int32_t(int32_t);
        inline void push_int64_t(int64_t);
        inline void push_double(double);
	};

    struct DECLSPEC_KAME RawDataReader {
		typedef std::vector<char>::const_iterator const_iterator;
		//! reads raw record
		//! \sa push(), rawData()
        //! XBufferUnderflowRecordError will be thrown if buffer shorts.
		template <typename tVar>
        inline tVar pop();

		const_iterator begin() const {return m_data.begin();}
		const_iterator end() const {return m_data.end();}
		unsigned int size() const {return m_data.size();}
		const std::vector<char> &data() const {return m_data;}
		const_iterator &popIterator() {return it;}
	private:
		friend class XPrimaryDriver;
		friend class XRawStreamRecordReader;
		RawDataReader(const std::vector<char> &data) : m_data(data) {it = data.begin();}
		RawDataReader();
		const_iterator it;
		const std::vector<char> &m_data;
		inline char pop_char();
		inline int16_t pop_int16_t();
		inline int32_t pop_int32_t();
        inline int64_t pop_int64_t();
        inline double pop_double();
	};

protected:
	//! This function will be called when raw data are written.
	//! Implement this function to convert the raw data to the record (Payload).
	//! \sa analyze()
    //! XRecordError will be thrown if data is not propertly formatted.
    virtual void analyzeRaw(RawDataReader &reader, Transaction &tr) = 0;

	//! will call analyzeRaw()
	//! \param rawdata the data being processed.
	//! \param time_awared time when a visible phenomenon started
	//! \param time_recorded usually pass \p XTime::now()
	//! \sa Payload::timeAwared()
	//! \sa Payload::time()
	void finishWritingRaw(const shared_ptr<const RawData> &rawdata,
		const XTime &time_awared, const XTime &time_recorded);

    //! How long this driver is willing to spend WAITING inside one
    //! finishWritingRaw, in µs.  0 (the default) = unbounded, i.e. today's
    //! behaviour.
    //!
    //! Only realtime acquisition loops need this, and the reason is specific.
    //! An acquisition loop at Priority::HIGHEST is demoted to NORMAL for
    //! everything downstream of the record -- the marked-message dispatch inside
    //! the commit, then visualize() and onVisualization -- because that is other
    //! people's work and must not inherit an exemption from politeness.  But once
    //! demoted it can WAIT, so the loop's period becomes exposed to whatever
    //! downstream contends with, which is the one thing HIGHEST was supposed to
    //! protect.
    //!
    //! A wait budget closes that without handing downstream any privilege: the
    //! loop declares how much of its period it will lend, and beyond that the
    //! negotiator stops waiting.  The budget is a thread-local ABSOLUTE limit, so
    //! one guard at the top of finishWritingRaw covers both demoted regions --
    //! and it is inert during the HIGHEST part, since HIGHEST leaves the round
    //! loop before it can sleep.  (An earlier note here claimed a budget was
    //! simply inert on a HIGHEST thread; that holds only while it IS HIGHEST.)
    //!
    //! **Default 20 ms, on every primary driver, realtime or not.**  KAME is a
    //! measurement instrument: a record whose commit stalls for a third of a
    //! second is a bad data point, not merely a slow one, and past about 20 ms
    //! that starts to show up in the measurement whatever the driver's priority.
    //! So the bound is not a realtime luxury to be gated on Priority::HIGHEST --
    //! it is the acquisition path's contract.
    //!
    //! It is not free.  Grand-scope arm, 8 threads --
    //!
    //!                throughput   p99.99    p99.999   MAX
    //!     no budget    2.36 M/s   3.67 ms   67.1 ms   326.6 ms
    //!     20 ms        2.25 M/s   16.8 ms   21.0 ms    20.3 ms
    //!
    //! -- so it costs 4.7 % of commit throughput, because a clipped commit stops
    //! waiting and retries and the retry adds CAS pressure.  (8-of-8 and 1-of-8
    //! budgeted measured the same, 2.25 vs 2.26 M/s, so that is the clipping
    //! itself and not a cascade.)  The p99.99 rising from 3.67 to 16.8 ms is
    //! movement *within* the budget, not a regression against it: with the budget
    //! every percentile including MAX lands under the 20 ms line, which is the
    //! property being bought.  Throughput is the thing traded away.
    //!
    //! No record is lost: the budget bounds *waiting*, and the clipped commit
    //! retries through iterate_commit until it succeeds.
    //! One wait is exempt from the bound (2026-07-31): the wait behind a
    //! LIVE privileged peer — privilege is the completion guarantee and a
    //! budget that declined it froze the whole system in the field.  A record
    //! can therefore be late by that holder's closure; still never lost.
    //!
    //! Override to pick it from the cycle -- comfortably less than the
    //! acquisition period, so a blown budget costs a late record rather than a
    //! lost one.  Return 0 to disable.
    virtual unsigned int downstreamWaitBudgetUS() const {return 20000;}
public:
    //! \name Record-commit latency telemetry
    //!
    //! An acquisition thread's `finishWritingRaw` is where the hardware loop
    //! meets the STM, so it is the one commit whose latency can stall
    //! acquisition.  These count how often that commit was slow, so the
    //! question "does this driver ever wait on the STM at all?" can be
    //! answered from a real session instead of extrapolated from a synthetic
    //! benchmark (where the whole-tree arm shows 250 ms tails and the
    //! per-subtree arm shows 1.5 us).
    //!
    //! Counted, never printed on the hot path — the same discipline as
    //! `kame_pool_rt_violations()`: one comparison per record, and a single
    //! summary line at thread exit if the count is nonzero.  Threshold is
    //! deliberately coarse; a driver that never trips it needs no realtime
    //! treatment at all.
    //! \{
    static constexpr std::uint64_t SLOW_RECORD_COMMIT_NS = 1000000ull; //!< 1 ms
    //! Records whose finishWritingRaw commit exceeded SLOW_RECORD_COMMIT_NS.
    std::uint64_t slowRecordCommits() const noexcept {
        return m_slowRecordCommits.load(std::memory_order_relaxed);
    }
    //! Longest finishWritingRaw commit seen, in ns.
    std::uint64_t maxRecordCommitNS() const noexcept {
        return m_maxRecordCommitNS.load(std::memory_order_relaxed);
    }
    //! Total records committed through finishWritingRaw.
    std::uint64_t recordCommits() const noexcept {
        return m_recordCommits.load(std::memory_order_relaxed);
    }
    //! \}
private:
    std::atomic<std::uint64_t> m_slowRecordCommits{0};
    std::atomic<std::uint64_t> m_maxRecordCommitNS{0};
    std::atomic<std::uint64_t> m_recordCommits{0};
public:
    struct DECLSPEC_KAME Payload : public XDriver::Payload {
		const RawData &rawData() const {return *m_rawData;}
	private:
		friend class XPrimaryDriver;
		shared_ptr<const RawData> m_rawData;
	};
};

inline void
XPrimaryDriver::RawData::push_char(char x) {
    push_back(x);
}
inline void
XPrimaryDriver::RawData::push_int16_t(int16_t x) {
    int16_t y = x;
    char *p = reinterpret_cast<char *>(&y);
#ifdef __BIG_ENDIAN__
    for(char *z = p + sizeof(x) - 1; z >= p; z--) {
#else
	for(char *z = p; z < p + sizeof(x); z++) {
#endif
		push_back( *z);
	}
}
inline void
XPrimaryDriver::RawData::push_int32_t(int32_t x) {
	int32_t y = x;
	char *p = reinterpret_cast<char *>(&y);
#ifdef __BIG_ENDIAN__
	for(char *z = p + sizeof(x) - 1; z >= p; z--) {
#else
	for(char *z = p; z < p + sizeof(x); z++) {
#endif
		push_back( *z);
	}
}
inline void
XPrimaryDriver::RawData::push_int64_t(int64_t x) {
    int64_t y = x;
    char *p = reinterpret_cast<char *>(&y);
#ifdef __BIG_ENDIAN__
    for(char *z = p + sizeof(x) - 1; z >= p; z--) {
#else
    for(char *z = p; z < p + sizeof(x); z++) {
#endif
        push_back( *z);
    }
}
inline void
XPrimaryDriver::RawData::push_double(double x) {
	static_assert(sizeof(double) == 8, "Not 8-byte sized double"); // for compatibility.
	double y = x;
	char *p = reinterpret_cast<char *>( &y);
#ifdef __BIG_ENDIAN__
	for(char *z = p + sizeof(x) - 1; z >= p; z--) {
#else
	for(char *z = p; z < p + sizeof(x); z++) {
#endif
		push_back( *z);
	}
}
inline char
XPrimaryDriver::RawDataReader::pop_char() {
	char c = *(it++);
	return c;
}
inline int16_t
XPrimaryDriver::RawDataReader::pop_int16_t() {
	union {
		int16_t x;
		char p[sizeof(int16_t)];
	} uni;
#ifdef __BIG_ENDIAN__
	for(char *z = uni.p + sizeof(uni) - 1; z >= uni.p; z--) {
#else
	for(char *z = uni.p; z < uni.p + sizeof(uni); z++) {
#endif
		*z = *(it++);
	}
	return uni.x;
}
inline int32_t
XPrimaryDriver::RawDataReader::pop_int32_t() {
	union {
		int32_t x;
		char p[sizeof(int32_t)];
	} uni;
#ifdef __BIG_ENDIAN__
	for(char *z = uni.p + sizeof(uni) - 1; z >= uni.p; z--) {
#else
	for(char *z = uni.p; z < uni.p + sizeof(uni); z++) {
#endif
		*z = *(it++);
	}
	return uni.x;
}
inline int64_t
XPrimaryDriver::RawDataReader::pop_int64_t() {
    union {
        int64_t x;
        char p[sizeof(int64_t)];
    } uni;
#ifdef __BIG_ENDIAN__
    for(char *z = uni.p + sizeof(uni) - 1; z >= uni.p; z--) {
#else
    for(char *z = uni.p; z < uni.p + sizeof(uni); z++) {
#endif
        *z = *(it++);
    }
    return uni.x;
}
inline double
XPrimaryDriver::RawDataReader::pop_double() {
	union {
		double x;
		char p[sizeof(double)];
	} uni;
#ifdef __BIG_ENDIAN__
	for(char *z = uni.p + sizeof(uni) - 1; z >= uni.p; z--) {
#else
	for(char *z = uni.p; z < uni.p + sizeof(uni); z++) {
#endif
		*z = *(it++);
	}
	return uni.x;
}

template <>
inline char XPrimaryDriver::RawDataReader::pop() {
	if(it + sizeof(char) > end()) throw XBufferUnderflowRecordError(__FILE__, __LINE__);
	return pop_char();
}
template <>
inline unsigned char XPrimaryDriver::RawDataReader::pop() {
	if(it + sizeof(char) > end()) throw XBufferUnderflowRecordError(__FILE__, __LINE__);
	return static_cast<unsigned char>(pop_char());
}
template <>
inline int16_t XPrimaryDriver::RawDataReader::pop() {
	if(it + sizeof(int16_t) > end()) throw XBufferUnderflowRecordError(__FILE__, __LINE__);
	return pop_int16_t();
}
template <>
inline uint16_t XPrimaryDriver::RawDataReader::pop() {
	if(it + sizeof(int16_t) > end()) throw XBufferUnderflowRecordError(__FILE__, __LINE__);
	return static_cast<uint16_t>(pop_int16_t());
}
template <>
inline int32_t XPrimaryDriver::RawDataReader::pop() {
	if(it + sizeof(int32_t) > end()) throw XBufferUnderflowRecordError(__FILE__, __LINE__);
	return pop_int32_t();
}
template <>
inline uint32_t XPrimaryDriver::RawDataReader::pop() {
	if(it + sizeof(int32_t) > end()) throw XBufferUnderflowRecordError(__FILE__, __LINE__);
	return static_cast<uint32_t>(pop_int32_t());
}
template <>
inline int64_t XPrimaryDriver::RawDataReader::pop() {
    if(it + sizeof(int64_t) > end()) throw XBufferUnderflowRecordError(__FILE__, __LINE__);
    return pop_int64_t();
}
template <>
inline uint64_t XPrimaryDriver::RawDataReader::pop() {
    if(it + sizeof(int64_t) > end()) throw XBufferUnderflowRecordError(__FILE__, __LINE__);
    return static_cast<uint64_t>(pop_int64_t());
}
// `int64_t` is `long` on LP64 Linux but `long long` on macOS (and on Windows),
// so the specialization tables here cover a DIFFERENT pair of C++ types on each
// platform: `push<long long>` / `pop<size_t>` compiles on exactly one of them
// and is an undefined symbol on the other.  Add the OTHER 8-byte spelling —
// whichever of {long, long long} is not already int64_t — so driver source is
// portable either way.  Naming it with conditional_t rather than an #ifdef is
// what keeps this from becoming a duplicate specialization on the platform
// where the two spellings coincide.
//
// Guarded on __SIZEOF_LONG__ == 8: under Windows LLP64 `long` is 32-bit, so
// there is no second 8-byte spelling to add (and pushing a 32-bit `long`
// through push_int64_t would be wrong).
#include <type_traits>
#if defined __SIZEOF_LONG__ && (__SIZEOF_LONG__ == 8)
//! The 8-byte signed/unsigned integer type that is NOT int64_t/uint64_t.
using kame_alt_int64_t = std::conditional<
    std::is_same<int64_t, long long>::value, long, long long>::type;
using kame_alt_uint64_t = std::conditional<
    std::is_same<uint64_t, unsigned long long>::value,
    unsigned long, unsigned long long>::type;
#endif

// See the matching note next to RawData::push(uint64_t): int64_t is `long` on
// LP64 Linux and `long long` on macOS, so cover the other 64-bit spelling too.
#if defined __SIZEOF_LONG__ && (__SIZEOF_LONG__ == 8)
template <>
inline kame_alt_int64_t XPrimaryDriver::RawDataReader::pop() {
    if(it + sizeof(int64_t) > end()) throw XBufferUnderflowRecordError(__FILE__, __LINE__);
    return static_cast<kame_alt_int64_t>(pop_int64_t());
}
template <>
inline kame_alt_uint64_t XPrimaryDriver::RawDataReader::pop() {
    if(it + sizeof(int64_t) > end()) throw XBufferUnderflowRecordError(__FILE__, __LINE__);
    return static_cast<kame_alt_uint64_t>(pop_int64_t());
}
#endif
template <>
inline float XPrimaryDriver::RawDataReader::pop() {
	if(it + sizeof(float) > end()) throw XBufferUnderflowRecordError(__FILE__, __LINE__);
	union {
		int32_t x;
		float y;
	} uni;
	static_assert(sizeof(uni.x) == sizeof(uni.y), "Size mismatch");
	uni.x = pop_int32_t();
	return uni.y;
}
template <>
inline double XPrimaryDriver::RawDataReader::pop() {
	if(it + sizeof(double) > end()) throw XBufferUnderflowRecordError(__FILE__, __LINE__);
	static_assert(sizeof(double) == 8, "Not 8-byte sized double");
	return pop_double();
}

template <>
inline void XPrimaryDriver::RawData::push(char x) {
	push_char(x);
}
template <>
inline void XPrimaryDriver::RawData::push(unsigned char x) {
	push_char(static_cast<char>(x));
}
template <>
inline void XPrimaryDriver::RawData::push(int16_t x) {
	push_int16_t(x);
}
template <>
inline void XPrimaryDriver::RawData::push(uint16_t x) {
	push_int16_t(static_cast<int16_t>(x));
}
template <>
inline void XPrimaryDriver::RawData::push(int32_t x) {
	push_int32_t(x);
}
template <>
inline void XPrimaryDriver::RawData::push(uint32_t x) {
	push_int32_t(static_cast<int32_t>(x));
}
template <>
inline void XPrimaryDriver::RawData::push(int64_t x) {
    push_int64_t(x);
}
template <>
inline void XPrimaryDriver::RawData::push(uint64_t x) {
    push_int64_t(static_cast<int64_t>(x));
}
// kame_alt_int64_t / kame_alt_uint64_t: see the note above the pop() pair.
#if defined __SIZEOF_LONG__ && (__SIZEOF_LONG__ == 8)
template <>
inline void XPrimaryDriver::RawData::push(kame_alt_int64_t x) {
    push_int64_t(static_cast<int64_t>(x));
}
template <>
inline void XPrimaryDriver::RawData::push(kame_alt_uint64_t x) {
    push_int64_t(static_cast<int64_t>(x));
}
#endif
template <>
inline void XPrimaryDriver::RawData::push(float f) {
	union {
		int32_t x;
		float y;
	} uni;
	static_assert(sizeof(uni.x) == sizeof(uni.y), "Size mismatch");
	uni.y = f;
	push_int32_t(uni.x);
}
template <>
inline void XPrimaryDriver::RawData::push(double x) {
	push_double(x);
}

#endif /*PRIMARYDRIVER_H_*/
