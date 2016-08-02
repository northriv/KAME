/***************************************************************************
        Copyright (C) 2002-2015 Kentaro Kitagawa
                           kitagawa@phys.s.u-tokyo.ac.jp

        This program is free software; you can redistribute it and/or
        modify it under the terms of the GNU Library General Public
        License as published by the Free Software Foundation; either
        version 2 of the License, or (at your option) any later version.

        You should have received a copy of the GNU Library General
        Public License and a list of authors along with this program;
        see the files COPYING and AUTHORS.
 ***************************************************************************/
#ifndef signalH
#define signalH

#include "support.h"
#include "xtime.h"
#include "xthread.h"
#include "atomic_smart_ptr.h"
#include <deque>

//! Detect whether the current thread is the main thread.
DECLSPEC_KAME bool isMainThread();

namespace Transactional {
template <class SS, typename...Args>
class Talker;}

//! Base class of listener, which holds pointers to object and function.
//! Hold instances by shared_ptr.
class DECLSPEC_KAME XListener {
public:
    virtual ~XListener();
    //! \return an appropriate delay for delayed transactions.
    unsigned int delay_ms() const;

    enum FLAGS : int {
        FLAG_MAIN_THREAD_CALL = 0x01, FLAG_AVOID_DUP = 0x02,
        FLAG_DELAY_SHORT = 0x100, FLAG_DELAY_ADAPTIVE = 0x200
    };

    int flags() const {return (int)m_flags;}
protected:
    template <class SS, typename...Args>
    friend class Transactional::Talker;
    XListener(FLAGS flags);
    atomic<int> m_flags;
};

class DECLSPEC_KAME XTalkerBase_ {
protected:
public:
    virtual ~XTalkerBase_() = default;
protected:
};

struct XTransaction_ {
    XTransaction_() : registered_time(XTime::now()) {}
    virtual ~XTransaction_() = default;
    const XTime registered_time;
    virtual bool talkBuffered() = 0;
};

DECLSPEC_KAME void registerTransactionList(XTransaction_ *);

#endif
