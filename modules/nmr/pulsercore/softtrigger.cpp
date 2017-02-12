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

#include "softtrigger.h"

//#include <boost/math/common_factor.hpp>
//using boost::math::lcm;

inline int unsigned gcd(unsigned int a, unsigned int b){
    if( !b) return a;
    return gcd(b, a % b);
}
inline unsigned int lcm(unsigned int a, unsigned int b){
    return a * b / gcd(a,b);
}

SoftwareTriggerManager::SoftwareTriggerManager() :
    m_list(new SoftwareTriggerList) {

}

shared_ptr<SoftwareTrigger>
SoftwareTriggerManager::create(const char *label, unsigned int bits) {
    shared_ptr<SoftwareTrigger> p(new SoftwareTrigger(label, bits));

    //inserting the new trigger source to the list atomically.
    for(local_shared_ptr<SoftwareTriggerList> old_list(m_list);;) {
        local_shared_ptr<SoftwareTriggerList> new_list(new SoftwareTriggerList( *old_list));
        new_list->push_back(p);
        if(m_list.compareAndSwap(old_list, new_list)) break;
    }
    onListChanged().talk(p);
    return p;
}
void
SoftwareTriggerManager::unregister(const shared_ptr<SoftwareTrigger> &p) {
    //performing it atomically.
    for(local_shared_ptr<SoftwareTriggerList> old_list(m_list);;) {
        local_shared_ptr<SoftwareTriggerList> new_list(new SoftwareTriggerList( *old_list));
        new_list->erase(std::find(new_list->begin(), new_list->end(), p));
        if(m_list.compareAndSwap(old_list, new_list)) break;
    }
    onListChanged().talk(p);
}

SoftwareTrigger::SoftwareTrigger(const char *label, unsigned int bits)
    : m_label(label), m_bits(bits),
      m_risingEdgeMask(0u), m_fallingEdgeMask(0u) {
    clear_();
    m_isPersistentCoherent = false;
}

void
SoftwareTrigger::clear_() {
    uint64_t x;
    while(FastQueue::key t = m_fastQueue.atomicFront(&x)) {
        m_fastQueue.atomicPop(t);
    }
    m_slowQueue.clear();
    m_slowQueueSize = 0;
}
bool
SoftwareTrigger::stamp(uint64_t cnt) {
    readBarrier();
    if(cnt < m_endOfBlank) return false;
    if(cnt == 0) return false; //ignore.
    try {
        m_fastQueue.push(cnt);
    }
    catch (FastQueue::nospace_error&) {
        XScopedLock<XMutex> lock(m_mutex);
        fprintf(stderr, "Slow queue!\n");
        m_slowQueue.push_back(cnt);
        if(m_slowQueue.size() > 100000u)
            m_slowQueue.pop_front();
        else
            ++m_slowQueueSize;
    }
    m_endOfBlank = cnt + m_blankTerm;
    return true;
}
void
SoftwareTrigger::start(double freq) {
    {
        XScopedLock<XMutex> lock(m_mutex);
        m_endOfBlank = 0;
        if(!m_blankTerm) m_blankTerm = lrint(0.02 * freq);
        m_freq = freq;
        clear_();
    }
    onStart().talk(shared_from_this());
}

void
SoftwareTrigger::stop() {
    XScopedLock<XMutex> lock(m_mutex);
    clear_();
    m_endOfBlank = (uint64_t)-1LL;
}
void
SoftwareTrigger::connect(uint32_t rising_edge_mask, uint32_t falling_edge_mask) {
    XScopedLock<XMutex> lock(m_mutex);
    clear_();
    if(m_risingEdgeMask || m_fallingEdgeMask)
        throw XKameError(
            i18n_noncontext("Duplicated connection to virtual trigger is not supported."), __FILE__, __LINE__);
    m_risingEdgeMask = rising_edge_mask;
    m_fallingEdgeMask = falling_edge_mask;
}
void
SoftwareTrigger::disconnect() {
    XScopedLock<XMutex> lock(m_mutex);
    clear_();
    m_risingEdgeMask = 0;
    m_fallingEdgeMask = 0;
}
uint64_t
SoftwareTrigger::tryPopFront(uint64_t threshold, double freq__) {
    unsigned int freq_em = lrint(freq());
    unsigned int freq_rc = lrint(freq__);
    unsigned int gcd__ = gcd(freq_em, freq_rc);

    uint64_t cnt;
    if(m_slowQueueSize) {
        XScopedLock<XMutex> lock(m_mutex);
        if(FastQueue::key t = m_fastQueue.atomicFront(&cnt)) {
            if((cnt < m_slowQueue.front()) || !m_slowQueueSize) {
                cnt = (cnt * (freq_rc / gcd__)) / (freq_em / gcd__);
                if(cnt >= threshold)
                    return 0uLL;
                if(m_fastQueue.atomicPop(t))
                    return cnt;
                return 0uLL;
            }
        }
        if( !m_slowQueueSize)
            return 0uLL;
        cnt = m_slowQueue.front();
        cnt = (cnt * (freq_rc / gcd__)) / (freq_em / gcd__);
        if(cnt >= threshold)
            return 0uLL;
        m_slowQueue.pop_front();
        --m_slowQueueSize;
        return cnt;
    }
    if(FastQueue::key t = m_fastQueue.atomicFront(&cnt)) {
        cnt = (cnt * (freq_rc / gcd__)) / (freq_em / gcd__);
        if(cnt >= threshold)
            return 0uLL;
        if(m_fastQueue.atomicPop(t))
            return cnt;
    }
    return 0uLL;
}

void
SoftwareTrigger::clear() {
    XScopedLock<XMutex> lock(m_mutex);
    clear_();
    m_endOfBlank =  0;
}
void
SoftwareTrigger::clear(uint64_t now, double freq__) {
    unsigned int freq_em= lrint(freq());
    unsigned int freq_rc = lrint(freq__);
    unsigned int gcd__ = gcd(freq_em, freq_rc);
    now = (now * (freq_em / gcd__)) / (freq_rc / gcd__);

    XScopedLock<XMutex> lock(m_mutex);
    uint64_t x;
    while(FastQueue::key t = m_fastQueue.atomicFront(&x)) {
        if(x <= now)
            m_fastQueue.atomicPop(t);
        else
            break;
    }
    while(m_slowQueue.size() && (m_slowQueue.front() <= now)) {
        m_slowQueue.pop_front();
        --m_slowQueueSize;
    }
}
void
SoftwareTrigger::forceStamp(uint64_t now, double freq__) {
    unsigned int freq_em= lrint(freq());
    unsigned int freq_rc = lrint(freq__);
    unsigned int gcd__ = gcd(freq_em, freq_rc);
    now = (now * (freq_em / gcd__)) / (freq_rc / gcd__);

    XScopedLock<XMutex> lock(m_mutex);
    ++m_slowQueueSize;
    m_slowQueue.push_front(now);
    std::sort(m_slowQueue.begin(), m_slowQueue.end());
}
