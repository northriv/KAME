/***************************************************************************
        Copyright (C) 2002-2017 Kentaro Kitagawa
                           kitagawa@phys.s.u-tokyo.ac.jp

        This program is free software; you can redistribute it and/or
        modify it under the terms of the GNU Library General Public
        License as published by the Free Software Foundation; either
        version 2 of the License, or (at your option) any later version.

        You should have received a copy of the GNU Library General
        Public License and a list of authors along with this program;
        see the files COPYING and AUTHORS.
***************************************************************************/

#ifndef SOFTTRIGGER_H
#define SOFTTRIGGER_H

#include "transaction.h"
#include "xthread.h"
#include <deque>
#include "atomic_queue.h"

class SoftwareTriggerManager;
//! Stores and reads time stamps between synchronized devices.
class SoftwareTrigger : public enable_shared_from_this<SoftwareTrigger> {
protected:
    friend class SoftwareTriggerManager;
    SoftwareTrigger(const char *label, unsigned int bits);
public:
    const char *label() const {return m_label.c_str();}
    //! Registers a name of "ARM" terminal to determine time origin.
    void setArmTerm(const char *arm_term) {m_armTerm = arm_term;}
    const char *armTerm() const {return m_armTerm.c_str();}
    bool isPersistentCoherentMode() const {return m_isPersistentCoherent;}
    void setPersistentCoherentMode(bool x) {m_isPersistentCoherent = x;}
    void start(double freq);
    double freq() const {return m_freq;} //!< [Hz].
    unsigned int bits() const {return m_bits;}
    void stop();
    double timeForBufferredTriggersRequired() const {return 0.5;} //sec.
    //! issues trigger anyway.
    void forceStamp(uint64_t now, double freq);
    //! issues trigger if possible.
    //! \return true if trigger is issued.
    bool stamp(uint64_t cnt);
    //! Edge triggering.
    //! \param time unit in 1/freq().
    //! \return true if trigger is issued.
    template <typename T>
    bool changeValue(T oldval, T val, uint64_t time) {
        if(((m_risingEdgeMask & val) & (m_risingEdgeMask & ~oldval))
           || ((m_fallingEdgeMask & ~val) & (m_fallingEdgeMask & oldval))) {
            if(time < m_endOfBlank) return false;
            return stamp(time);
        }
        return false;
    }

    void connect(uint32_t rising_edge_mask, uint32_t falling_edge_mask);
    void disconnect();
    //! \param blankterm in seconds, not to stamp so frequently.
    void setBlankTerm(double blankterm) {
        m_blankTerm = llrint(blankterm * freq());
        memoryBarrier();
    }
    using STRGTalker = Transactional::Talker<shared_ptr<SoftwareTrigger>>;
    //! for restarting connected task.
    STRGTalker &onStart() {return m_tlkStart;}

    using TRTalker = Transactional::Talker<uint64_t>;
    TRTalker &onTriggerRequested() {return m_tlkTriggerRequested;}
    //! clears all time stamps.
    void clear();
    //! clears past time stamps.
    void clear(uint64_t now, double freq);
    //! \return if not, zero will be returned.
    //! \param freq frequency of reader.
    //! \param threshold upper bound to be pop, unit in 1/\a freq (2nd param.).
    uint64_t tryPopFront(uint64_t threshold, double freq);
private:
    void clear_();
    const XString m_label;
    XString m_armTerm;
    unsigned int m_bits;
    uint32_t m_risingEdgeMask, m_fallingEdgeMask;
    uint64_t m_blankTerm;
    uint64_t m_lastThresholdRequested;
    uint64_t m_endOfBlank; //!< next stamp must not be less than this.
    double m_freq; //!< [Hz].
    enum {QUEUE_SIZE = 8192};
    typedef atomic_queue_reserved<uint64_t, QUEUE_SIZE> FastQueue;
    FastQueue m_fastQueue; //!< recorded stamps.
    typedef std::deque<uint64_t> SlowQueue;
    SlowQueue m_slowQueue; //!< recorded stamps, when \a m_fastQueue is full.
    atomic<unsigned int> m_slowQueueSize;
    XMutex m_mutex; //!< for \a m_slowQueue.
    STRGTalker m_tlkStart;
    TRTalker m_tlkTriggerRequested;
    bool m_isPersistentCoherent;
};

class SoftwareTriggerManager {
public:
    SoftwareTriggerManager();

    shared_ptr<SoftwareTrigger> create(const char *label, unsigned int bits);
    void unregister(const shared_ptr<SoftwareTrigger> &);

    using SoftwareTriggerList = std::deque<shared_ptr<SoftwareTrigger>>;
    const local_shared_ptr<SoftwareTriggerList> list() const {return m_list;}
    //! for changing list.
    using STRGTalker = Transactional::Talker<shared_ptr<SoftwareTrigger>>;
    STRGTalker &onListChanged() {return m_tlkListChanged;}
private:
    atomic_shared_ptr<SoftwareTriggerList> m_list;
    STRGTalker m_tlkListChanged;
};


#endif // SOFTTRIGGER_H
