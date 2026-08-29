/***************************************************************************
        Copyright (C) 2002-2025 Kentaro Kitagawa
                           kitag@issp.u-tokyo.ac.jp

        This program is free software; you can redistribute it and/or
        modify it under the terms of the GNU General Public
        License as published by the Free Software Foundation; either
        version 2 of the License, or (at your option) any later version.

        You should have received a copy of the GNU General
        Public License and a list of authors along with this program;
        see the files COPYING and AUTHORS.
 ***************************************************************************/
#ifndef xnodeH
#define xnodeH

#include "transaction.h"
#include "threadlocal.h"
#include <deque>

class XNode;

using Snapshot = Transactional::Snapshot<XNode>;
using Transaction = Transactional::Transaction<XNode>;

template <class T>
using SingleSnapshot = Transactional::SingleSnapshot<XNode, T>;
template <class T>
using SingleTransaction = Transactional::SingleTransaction<XNode, T>;

#define trans(node) for(Transaction \
    implicit_tr(node, false); !implicit_tr.isModified() || !implicit_tr.commitOrNext(); ) implicit_tr[node]

template <class T>
typename std::enable_if<std::is_base_of<XNode, T>::value, const SingleSnapshot<T> >::type
 operator*(T &node) {
    return SingleSnapshot<T>(node);
}

template <typename...Args>
using Talker = Transactional::Talker<Snapshot, Args...>;
template <typename...Args>
using TalkerOnce = Transactional::TalkerOnce<Snapshot, Args...>;

using Listener = Transactional::Listener;

using Transactional::Priority;

extern template class DECLSPEC_KAME Transactional::Node<class XNode>;
//! XNode supports accesses from scripts/GUI and shared_from_this(),
//! in addition to the features of Transactional::Node.
//! \sa Transactional::Node, create(), createOrphan().
class DECLSPEC_KAME XNode : public enable_shared_from_this<XNode>, public Transactional::Node<XNode> {
public:
    explicit XNode(const char *name, bool runtime = false);
    virtual ~XNode();

    template <class T>
    shared_ptr<T> create(const char *name) {return create<T>(name, false);}
    template <class T, typename... Args>
    shared_ptr<T> create(const char *name, bool runtime, Args&&... args);

    template <class T>
    shared_ptr<T> create(Transaction &tr, const char *name) {return create<T>(tr, name, false);}
    template <class T, typename... Args>
    shared_ptr<T> create(Transaction &tr, const char *name, bool runtime, Args&&... args);

    template <class T__>
    static shared_ptr<T__> createOrphan(const char *name) {return createOrphan<T__>(name, false);}
    template <class T__, typename... Args_>
    static shared_ptr<T__> createOrphan(const char *name, bool runtime, Args_&&... args);

    //! \return internal/scripting name. Use latin1 chars.
    XString getName() const {return m_name;}
    //! \return i18n name for UI.
    virtual XString getLabel() const {return getName();}
    //! \return the string this node was CREATED with, when it has one, and
    //! otherwise the demangled type name without its leading 'X'.
    //!
    //! The stored key is the identifier that can bring a node back:
    //! `createByTypename()` records it, and it is what `XTypeHolder` accepts.
    //! The typeid-derived fallback agrees with it only by a coincidence of
    //! spelling (`REGISTER_TYPE(list, Foo, …)` registers "Foo" and names the
    //! class `XFoo`) — a coincidence that fails for a template alias, and
    //! across compilers, since MSVC spells `typeid().name()` differently.
    virtual XString getTypename() const;
    //! Records the registry key a node was created with.  Called by
    //! `XListNodeBase::createByTypename()`; there should be no other caller.
    void setStoredTypename(const XString &);

    shared_ptr<XNode> getChild(const XString &var) const;

    //! Enables/disables controls over scripting/GUI.
    void setUIEnabled(bool v);
    //! Disables all scripting/GUI operations on this node hereafter.
    void disable();

    //! Data holder.
    //! \sa Transactional::Node::Payload.
    struct DECLSPEC_KAME Payload : public Transactional::Node<XNode>::Payload {
        Payload() : Transactional::Node<XNode>::Payload(), m_flags((int)FLAG::NODE_UI_ENABLED) {}
        //! If true, operations are allowed by UI and scripts.
        bool isUIEnabled() const {return m_flags & FLAG::NODE_UI_ENABLED;}
        void setUIEnabled(bool var);
        bool isDisabled() const {return m_flags & FLAG::NODE_DISABLED;}
        void disable();
        bool isRuntime() const {return m_flags & FLAG::NODE_RUNTIME;}
        void setRuntime(bool var) {m_flags = (m_flags & ~FLAG::NODE_RUNTIME) | (var ? FLAG::NODE_RUNTIME : 0);}
        //! \sa setUIEnabled
        Talker<XNode*> &onUIFlagsChanged() {return m_tlkOnUIFlagsChanged;}
        const Talker<XNode*> &onUIFlagsChanged() const {return m_tlkOnUIFlagsChanged;}
    private:
        enum FLAG : int {NODE_UI_ENABLED = 0x1, NODE_DISABLED = 0x2, NODE_RUNTIME = 0x4};
        int m_flags;
        TalkerOnce<XNode*> m_tlkOnUIFlagsChanged;
    };

    XNode() = delete;
private:
    const XString m_name;
    //! Written once, just after creation, on a node other threads can
    //! already see -- so it is published the way XDoubleNode::m_format is,
    //! not as a bare XString.
    atomic_shared_ptr<XString> m_storedTypename;
    static XThreadLocal<std::deque<shared_ptr<XNode> > > stl_thisCreating;
};

class DECLSPEC_KAME XTouchableNode : public XNode {
public:
    XTouchableNode(const char *name, bool runtime) : XNode(name, runtime) {}

    struct DECLSPEC_KAME Payload : public XNode::Payload {
        void touch();
        //! \sa touch()
        Talker<XTouchableNode*> &onTouch() {return m_tlkOnTouch;}
        const Talker<XTouchableNode*> &onTouch() const {return m_tlkOnTouch;}
    protected:
        Talker<XTouchableNode*> m_tlkOnTouch;
    };
};

//! Interface class containing values
class DECLSPEC_KAME XValueNodeBase : public XNode {
protected:
    XValueNodeBase(const char *name, bool runtime) : XNode(name, runtime), m_validator(0) {}
public:
    using Validator = void (*)(XString &);
    void setValidator(Validator x) {m_validator = x;}

    struct DECLSPEC_KAME Payload : public XNode::Payload {
        Payload() : XNode::Payload() {}
        //! Gets value as a string, which is used for scripting.
        virtual XString to_str() const = 0;
        //! Sets value as a string, which is used for scripting.
        //! This throws exception when the validator throws.
        void str(const XString &str) {
            XString sc(str);
            if(static_cast<XValueNodeBase&>(node()).m_validator)
                (*static_cast<XValueNodeBase&>(node()).m_validator)(sc);
            str_(sc);
        }
        Talker<XValueNodeBase*> &onValueChanged() {return m_tlkOnValueChanged;}
        const Talker<XValueNodeBase*> &onValueChanged() const {return m_tlkOnValueChanged;}
    protected:
        //! \a str_() can throw exception due to format issues.
        //! A marking to \a onValueChanged() is necessary.
        virtual void str_(const XString &) = 0;
        TalkerOnce<XValueNodeBase*> m_tlkOnValueChanged;
    };
protected:
    Validator m_validator;
};

//! Base class for integer node.
template <typename T, int base = 10>
class DECLSPEC_KAME XIntNodeBase : public XValueNodeBase {
public:
    explicit XIntNodeBase(const char *name, bool runtime = false) : XValueNodeBase(name, runtime) {}
    virtual ~XIntNodeBase() = default;

    struct DECLSPEC_KAME Payload : public XValueNodeBase::Payload {
        Payload() : XValueNodeBase::Payload() {this->m_var = 0;}
        virtual XString to_str() const override;
        operator T() const {return m_var;}
        Payload &operator=(T x) {
            m_var = x;
            tr().mark(onValueChanged(), static_cast<XValueNodeBase*>(&node()));
            return *this;
        }
    protected:
        virtual void str_(const XString &) override;
        T m_var;
    };
};

class DECLSPEC_KAME XDoubleNode : public XValueNodeBase {
public:
    explicit XDoubleNode(const char *name, bool runtime = false, const char *format = 0L);
    virtual ~XDoubleNode() = default;

    const char *format() const {return local_shared_ptr<XString>(m_format)->c_str();}
    void setFormat(const char* format);

    struct DECLSPEC_KAME Payload : public XValueNodeBase::Payload {
        Payload() : XValueNodeBase::Payload() {this->m_var = 0.0;}
        virtual XString to_str() const override;
        operator double() const {return m_var;}
        Payload &operator=(double x) {
            m_var = x;
            tr().mark(onValueChanged(), static_cast<XValueNodeBase*>(&node()));
            return *this;
        }
    protected:
        virtual void str_(const XString &) override;
        double m_var;
    };
private:
    atomic_shared_ptr<XString> m_format;
};

class DECLSPEC_KAME XStringNode : public XValueNodeBase {
public:
    explicit XStringNode(const char *name, bool runtime = false);
    virtual ~XStringNode() = default;

    struct DECLSPEC_KAME Payload : public XValueNodeBase::Payload {
        virtual XString to_str() const override {return this->m_var;}
        operator const XString&() const {return m_var;}
        Payload &operator=(const XString &x) {
            m_var = x;
            tr().mark(onValueChanged(), static_cast<XValueNodeBase*>(&node()));
            return *this;
        }
    protected:
        virtual void str_(const XString &str) override { *this = str;}
        XString m_var;
    };
};

using XIntNode = XIntNodeBase<int>;
using XUIntNode = XIntNodeBase<unsigned int>;
using XLongNode = XIntNodeBase<long>;
using XULongNode = XIntNodeBase<unsigned long>;
using XBoolNode = XIntNodeBase<bool>;
using XHexNode = XIntNodeBase<unsigned long, 16>;

template <typename T, int base>
inline void
XIntNodeBase<T, base>::Payload::str_(const XString &) {
}
template <>
inline void
XIntNodeBase<int, 10>::Payload::str_(const XString &str) {
    bool ok;
    int var = QString(str).toInt(&ok, 10);
    if( !ok)
        throw XKameError(i18n("Ill string conversion to integer."), __FILE__, __LINE__);
    *this = var;
}
template <>
inline void
XIntNodeBase<unsigned int, 10>::Payload::str_(const XString &str) {
    bool ok;
    unsigned int var = QString(str).toUInt(&ok);
    if( !ok)
        throw XKameError(i18n("Ill string conversion to unsigned integer."), __FILE__, __LINE__);
    *this = var;
}
template <>
inline void
XIntNodeBase<long, 10>::Payload::str_(const XString &str) {
    bool ok;
    long var = QString(str).toLong(&ok, 10);
    if( !ok)
        throw XKameError(i18n("Ill string conversion to integer."), __FILE__, __LINE__);
    *this = var;
}
template <>
inline void
XIntNodeBase<unsigned long, 10>::Payload::str_(const XString &str) {
    bool ok;
    unsigned long var = QString(str).toULong(&ok);
    if( !ok)
        throw XKameError(i18n("Ill string conversion to unsigned integer."), __FILE__, __LINE__);
    *this = var;
}
template <>
inline void
XIntNodeBase<unsigned long, 16>::Payload::str_(const XString &str) {
    bool ok;
    // Was `unsigned int var`: QString::toULong() returns ulong (64-bit on
    // LP64) and the node stores `unsigned long`, so a hex string above
    // 0xFFFFFFFF was silently reduced mod 2^32 on the way in while to_str()
    // printed the full width on the way out.  Matches the XULongNode
    // specialisation now.
    unsigned long var = QString(str).toULong(&ok, 16);
    if( !ok)
        throw XKameError(i18n("Ill string conversion to hex."), __FILE__, __LINE__);
    *this = var;
}
template <>
inline void
XIntNodeBase<bool, 10>::Payload::str_(const XString &str) {
    bool ok;
    bool x = QString(str).toInt(&ok);
    if(ok) {
        *this =  x ? true : false ;
        return;
    }
    if(QString(str).trimmed().toLower() == "true") {
        *this = true; return;
    }
    if(QString(str).trimmed().toLower() == "false") {
        *this = false; return;
    }
    throw XKameError(i18n("Ill string conversion to boolean."), __FILE__, __LINE__);
}

template <typename T, int base>
inline XString
XIntNodeBase<T, base>::Payload::to_str() const {
    return QString::number(m_var, base);
}
template <>
inline XString
XIntNodeBase<bool, 10>::Payload::to_str() const {
    return m_var ? "true" : "false";
}

template <class T, typename... Args>
shared_ptr<T>
XNode::createOrphan(const char *name, bool runtime, Args&&... args) {
    // XNode's constructor pushes shared_ptr(this) so that constructors can use
    // shared_from_this(); this pops it.  A constructor that throws after that
    // push leaves an entry that is BOTH dangling and owning: `new T` has
    // already run the base destructor and freed the memory, yet the entry's
    // shared_ptr still holds a refcount on it.  Popping such an entry
    // double-frees; leaving it makes the next createOrphan adopt freed memory.
    // Node constructors can throw for real — they run transactions (the
    // documented child-init pattern), so the STM starvation throw reaches
    // them, as do XKameError and bad_alloc.
    const size_t depth = XNode::stl_thisCreating->size();
    T *raw;
    try {
        raw = Transactional::Node<XNode>::create<T>(
            name, runtime, std::forward<Args>(args)...);
    }
    catch(...) {
        // Neutralise every entry the failed construction left (its own, plus
        // any child it had created and not yet popped) by leaking the control
        // block: refcount never reaches zero, so the deleter never runs on the
        // freed memory.  A few dozen bytes on an error path, versus a double
        // free.
        while(XNode::stl_thisCreating->size() > depth) {
            new shared_ptr<XNode>(std::move(XNode::stl_thisCreating->back()));
            XNode::stl_thisCreating->pop_back();
        }
        throw;
    }
    // Positional pop verified by identity: if these ever disagree the deque
    // has been desynchronised and adopting the back() entry would hand out
    // someone else's object.
    if(XNode::stl_thisCreating->empty() ||
        (XNode::stl_thisCreating->back().get() != static_cast<XNode *>(raw)))
        throw std::runtime_error(
            "XNode::createOrphan: the creation stack is desynchronised "
            "(an exception escaped a node constructor).");
    shared_ptr<T> ptr = dynamic_pointer_cast<T>(XNode::stl_thisCreating->back());
    XNode::stl_thisCreating->pop_back();
    return ptr;
}

template <class T, typename... Args>
shared_ptr<T>
XNode::create(Transaction &tr, const char *name, bool runtime, Args&&... args) {
    shared_ptr<T> ptr(createOrphan<T>(name, runtime, std::forward<Args>(args)...));
    if(ptr) {
        if( !insert(tr, ptr, true))
            ptr = nullptr; //online insertion has failed.
    }
    return ptr;
}

template <class T, typename... Args>
shared_ptr<T>
XNode::create(const char *name, bool runtime, Args&&... args) {
    shared_ptr<T> ptr(createOrphan<T>(name, runtime, std::forward<Args>(args)...));
    if(ptr) insert(ptr);
    return ptr;
}

//---------------------------------------------------------------------------
#endif
