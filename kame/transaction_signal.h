/***************************************************************************
        Copyright (C) 2002-2015 Kentaro Kitagawa
                           kitag@issp.u-tokyo.ac.jp

        This program is free software; you can redistribute it and/or
        modify it under the terms of the GNU Library General Public
        License as published by the Free Software Foundation; either
        version 2 of the License, or (at your option) any later version.

        You should have received a copy of the GNU Library General
        Public License and a list of authors along with this program;
        see the files COPYING and AUTHORS.
 ***************************************************************************/
#ifndef TRANSACTION_SIGNAL_H
#define TRANSACTION_SIGNAL_H

#include "support.h"
#include "threadlocal.h"
#include "atomic_smart_ptr.h"
#include "allocator.h"
#include "xtime.h"

namespace Transactional {

struct ProcessCounter {
    using cnt_t = uint16_t;
    ProcessCounter();
    operator cnt_t() const noexcept {return m_var;}
    enum : cnt_t { MAINTHREADID = 1 };
    static cnt_t id() noexcept {return *stl_processID;}
private:
    cnt_t m_var;
    static atomic<cnt_t> s_count;
    //! Holds the current porcess(thread) internal ID. Taking non-zero value.
    static XThreadLocal<ProcessCounter> stl_processID;
};

inline bool isMainThread() noexcept {return ProcessCounter::id() == ProcessCounter::MAINTHREADID;}

template <int N>
struct CallByTuple {
    template <class Func, class R, typename TPL, typename... Args>
    CallByTuple(Func f, R &r, const TPL& t, const Args&...args) {
        CallByTuple<N - 1>(f, r, t, std::get<N - 1>(t), args...);
    }
};
template <>
struct CallByTuple<0> {
    template <class Func, class R, typename TPL, typename... Args>
    CallByTuple(Func f, R &r, const TPL&, const Args&...args) {
        (r.*f)(args...);
    }
};

template <typename...Args>
struct Event {
    explicit Event(std::tuple<Args...>&& tpl) noexcept : tuple(std::move(tpl)) {}
    Event(const Event&) = default;
    Event(Event&&) = default;
    Event &operator=(const Event&) = delete;
private:
    std::tuple<Args...> tuple;
public:
    template <class Func, class T>
    void operator()(Func f, T &t) const {
        CallByTuple<sizeof...(Args)>(f, t, tuple);
    }
};

template <class SS, typename...Args> class Talker;

//! Base class of listener, which holds pointers to object and function.
//! Hold instances by shared_ptr.
class DECLSPEC_KAME Listener {
public:
    virtual ~Listener() = default;
    //! \return an appropriate delay for delayed events.
    unsigned int delay_ms() const;

    enum FLAGS : int {
        FLAG_MAIN_THREAD_CALL = 0x01, FLAG_AVOID_DUP = 0x02,
        FLAG_DELAY_SHORT = 0x100, FLAG_DELAY_ADAPTIVE = 0x200
    };
    int flags() const {return (int)m_flags;}
protected:
    template <class SS, typename...Args>
    friend class Transactional::Talker;
    Listener(FLAGS flags);
    const int m_flags;
};

template <class Event>
class ListenerBase : public Listener {
protected:
    explicit ListenerBase(Listener::FLAGS flags) : Listener(flags), event() {}
public:
    virtual void operator() (const Event&) const = 0;
protected:
    template <class SS, typename...Args>
    friend class Talker;
    atomic_unique_ptr<Event> event;
};

template<class Event, class R, class Func>
struct ListenerRef : public ListenerBase<Event> {
    ListenerRef(R &obj, Func f, Listener::FLAGS flags) noexcept :
        ListenerBase<Event>(flags), m_func(f), m_obj(obj) { }
    virtual void operator() (const Event& e) const override {
        e(m_func, m_obj);
    }
private:
    Func m_func;
    R &m_obj;
};
template<class Event, class R, class Func>
struct ListenerWeak : public ListenerBase<Event> {
    ListenerWeak(const shared_ptr<R> &obj, Func f, Listener::FLAGS flags) noexcept :
         ListenerBase<Event>(flags), m_func(f), m_obj(obj) { }
    virtual void operator() (const Event& e) const override {
        if(auto p = m_obj.lock() ) {
            e(m_func, *p);
        }
    }
private:
    Func m_func;
    const weak_ptr<R> m_obj;
};

template <class SS>
struct Message_ {
    virtual ~Message_() = default;
    virtual void talk(const SS &shot) = 0;
    virtual int unmark(const shared_ptr<Listener> &x) = 0;
};

struct BufferedEvent {
    BufferedEvent() : registered_time(XTime::now()) {}
    virtual ~BufferedEvent() = default;
    const XTime registered_time;
    virtual bool talkBuffered() = 0;
    static DECLSPEC_KAME void registerEvent(std::unique_ptr<BufferedEvent>);
};

//! M/M Listener and Talker model
//! \sa Listener
//! \p tArg: value which will be derivered
template <class SS, typename...Args>
class Talker {
public:
    virtual ~Talker() = default;

    template <class R, class T, typename...ArgRefs>
    shared_ptr<Listener> connect(R& obj, void(T::*func)(ArgRefs...), int flags = 0);
    template <class R, class T, typename...ArgRefs>
    shared_ptr<Listener> connectWeakly(const shared_ptr<R> &obj,
        void (T::*func)(ArgRefs...), int flags = 0);

    void connect(const shared_ptr<Listener> &x);
    void disconnect(const shared_ptr<Listener> &);

    //! Requests a talk to connected listeners.
    //! If a listener is not mainthread model, the listener will be called later.
    //! \param arg passing argument to all listeners
    //! If listener avoids duplication, lock won't be passed to listener.
    struct Message;
    template <typename...ArgRefs>
    shared_ptr<Message> createMessage(int64_t tr_serial, ArgRefs&&... arg) const;
    template <typename...ArgRefs>
    void talk(const SS &shot, ArgRefs&&...args) const {
        Message m(m_listeners, std::forward<ArgRefs>(args)...);
        m.talk(shot);
    }

    bool empty() const noexcept {return !m_listeners;}
private:
    using Event_ = Event<SS, Args...>;
    typedef fast_vector<weak_ptr<ListenerBase<Event_>> > ListenerList;
    typedef fast_vector<shared_ptr<Listener> > UnmarkedListenerList;
    shared_ptr<ListenerList> m_listeners;

    void connect(const shared_ptr<ListenerBase<Event_>> &);

    struct EventWrapper : public BufferedEvent {
        EventWrapper(const shared_ptr<ListenerBase<Event_>> &l) noexcept :
            BufferedEvent(), listener(l) {}
        virtual ~EventWrapper() = default;
        const shared_ptr<ListenerBase<Event_>> listener;
    };
    struct EventWrapperAllowDup : public EventWrapper {
        EventWrapperAllowDup(const shared_ptr<ListenerBase<Event_>> &l, const Event_ &e) noexcept :
            EventWrapper(l), event(e) {}
        Event_ event;
        virtual bool talkBuffered() override {
            ( *this->listener)(std::move(event));
            return false;
        }
    };
    struct EventWrapperAvoidDup : public EventWrapper {
        EventWrapperAvoidDup(const shared_ptr<ListenerBase<Event_>> &l) : EventWrapper(l) {}
            virtual bool talkBuffered() override {
                bool skip = false;
                if(this->listener->delay_ms()) {
                    long elapsed_ms = XTime::now().diff_msec(this->registered_time);
                    skip = ((long)this->listener->delay_ms() > elapsed_ms);
                }
                if( !skip) {
                    atomic_unique_ptr<Event_> e;
                    e.swap(this->listener->event);
                    assert(e.get());
                    ( *this->listener)( std::move(*e));
                }
                return skip;
            }
    };
public:
    struct Message : public Message_<SS> {
        template <class...ArgRefs>
        Message(const shared_ptr<ListenerList> &l, ArgRefs&&...as) noexcept :
            Message_<SS>(), listeners(l), args(std::forward<ArgRefs>(as)...) {}
        shared_ptr<ListenerList> listeners;
        std::tuple<Args...> args;
        shared_ptr<UnmarkedListenerList> listeners_unmarked;
        virtual void talk(const SS &shot) override;
        virtual int unmark(const shared_ptr<Listener> &x) override {
            if( !listeners)
                return 0;
            int canceled = 0;
            for(auto &&y: *listeners) {
                if(auto listener = y.lock()) {
                    if(listener == x) {
                        if( !listeners_unmarked)
                            listeners_unmarked = std::make_shared<UnmarkedListenerList>();
                        listeners_unmarked->push_back(x);
                        ++canceled;
                    }
                }
            }
            return canceled;
        }
    };
};

template <class SS, typename...Args>
class TalkerOnce : public Talker<SS, Args...> {
public:
    TalkerOnce() : Talker<SS, Args...>(), m_transaction_serial(0) {}
    TalkerOnce(const TalkerOnce &x) : Talker<SS, Args...>(x), m_transaction_serial(0) {}
    template <typename...ArgRefs>
    shared_ptr<typename TalkerOnce::Message> createMessage(int64_t tr_serial, ArgRefs&&...args) const {
        if(m_transaction_serial == tr_serial) {
            if(auto m = m_marked.lock()) {
                m->args = std::forward_as_tuple(args...);
                return nullptr;
            }
        }
        auto m = Talker<SS, Args...>::createMessage(tr_serial, std::forward<ArgRefs>(args)...);
        m_transaction_serial = tr_serial;
        m_marked = m;
        return m;
    }
private:
    mutable weak_ptr<typename Talker<SS, Args...>::Message> m_marked;
    mutable int64_t m_transaction_serial;
};

template <class SS, typename...Args>
template <typename...ArgRefs>
shared_ptr<typename Talker<SS, Args...>::Message> Talker<SS, Args...>::createMessage(int64_t, ArgRefs&&...args) const {
    if( !m_listeners)
        return nullptr;
    return std::make_shared<Message>(m_listeners, std::forward<ArgRefs>(args)...);
}

template <class SS, typename...Args>
template <class R, class T, typename...ArgRefs>
shared_ptr<Listener>
Talker<SS, Args...>::connect(R &obj, void(T::*func)(ArgRefs...), int flags) {
    shared_ptr<ListenerBase<Event_>> listener =
            std::make_shared<ListenerRef<Talker<SS, Args...>::Event_, T, decltype(func)>>(
                static_cast<T&>(obj), func, (Listener::FLAGS)flags);
    connect(listener);
    return listener;
}

template <class SS, typename...Args>
template <class R, class T, typename...ArgRefs>
shared_ptr<Listener>
Talker<SS, Args...>::connectWeakly(const shared_ptr<R> &obj,
    void(T::*func)(ArgRefs...), int flags) {
    shared_ptr<ListenerBase<Event_>> listener =
            std::make_shared<ListenerWeak<Talker<SS, Args...>::Event_, T, decltype(func)>>(
            static_pointer_cast<T>(obj), func, (Listener::FLAGS)flags);
    connect(listener);
    return listener;
}
template <class SS, typename...Args>
void
Talker<SS, Args...>::connect(const shared_ptr<Listener> &lx) {
    auto listener = dynamic_pointer_cast<ListenerBase<Event_>>(lx);
    connect(listener);
}
template <class SS, typename...Args>
void
Talker<SS, Args...>::connect(const shared_ptr<ListenerBase<Event_>> &lx) {
    auto new_list = m_listeners ? std::make_shared<ListenerList>( *m_listeners) : std::make_shared<ListenerList>();
    // clean-up dead listeners.
    for(auto it = new_list->begin(); it != new_list->end();) {
        if( !it->lock())
            it = new_list->erase(it);
        else
            ++it;
    }
    new_list->push_back(lx);
    new_list->shrink_to_fit();
    m_listeners = new_list;
}
template <class SS, typename...Args>
void
Talker<SS, Args...>::disconnect(const shared_ptr<Listener> &lx) {
    auto new_list = m_listeners ? std::make_shared<ListenerList>( *m_listeners) : std::make_shared<ListenerList>();
    for(auto it = new_list->begin(); it != new_list->end();) {
        if(auto listener = it->lock()) {
            // clean dead listeners and matching one.
            if( !listener || (lx == listener)) {
                it = new_list->erase(it);
                continue;
            }
        }
        ++it;
    }
    if(new_list->empty())
        new_list.reset();
    else
        new_list->shrink_to_fit();
    m_listeners = new_list;
}

template <class SS, typename...Args>
void
Talker<SS, Args...>::Message::talk(const SS &shot) {
    if( !listeners) return;
    Event_ event(std::tuple_cat(std::forward_as_tuple(shot), std::move(args)));
    //Writing deferred events to event pool.
    for(auto &&x: *listeners) {
        if(auto listener = x.lock()) {
            if(listeners_unmarked &&
                (std::find(listeners_unmarked->begin(), listeners_unmarked->end(), listener) != listeners_unmarked->end()))
                continue;
            if(listener->flags() & Listener::FLAG_MAIN_THREAD_CALL) {
                if(listener->flags() & Listener::FLAG_AVOID_DUP) {
                    atomic_unique_ptr<Event_> newevent(new Event_(event) );
                    newevent.swap(listener->event);
                    if( !newevent.get())
                        BufferedEvent::registerEvent(std::unique_ptr<BufferedEvent>(
                                        new EventWrapperAvoidDup(listener)));
                }
                else {
                    if(isMainThread()) {
                        try {
                            ( *listener)(event);
                        }
                        catch (XKameError &e) {
                            e.print();
                        }
                    }
                    else {
                        BufferedEvent::registerEvent(std::unique_ptr<BufferedEvent>(
                                        new EventWrapperAllowDup(listener, event)));
                    }
                }
            }
        }
    }
    //Immediate events.
    for(auto &&x: *listeners) {
        if(auto listener = x.lock()) {
            if(listeners_unmarked &&
                (std::find(listeners_unmarked->begin(), listeners_unmarked->end(), listener) != listeners_unmarked->end()))
                continue;
            if( !(listener->flags() & Listener::FLAG_MAIN_THREAD_CALL)) {
                try {
                    ( *listener)(event);
                }
                catch (XKameError &e) {
                    e.print();
                }
            }
        }
    }
}

} //namespace Transactional

#endif /*TRANSACTION_SIGNAL_H*/
