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
#ifndef xnodeH
#define xnodeH

#include "transaction.h"
#include "threadlocal.h"
#include "xthread.h"
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
using TalkerSingleton = Transactional::TalkerSingleton<Snapshot, Args...>;

using Listener = Transactional::Listener;

extern template class Transactional::Node<class XNode>;
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
    XString getTypename() const;

    shared_ptr<XNode> getChild(const XString &var) const;
    shared_ptr<XNode> getParent() const;

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
        TalkerSingleton<XNode*> m_tlkOnUIFlagsChanged;
    };

    XNode() = delete;
private:
    const XString m_name;
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
        void str(const XString &str) throw (XKameError &) {
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
        TalkerSingleton<XValueNodeBase*> m_tlkOnValueChanged;
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

template <class T, typename... Args>
shared_ptr<T>
XNode::createOrphan(const char *name, bool runtime, Args&&... args) {
    Transactional::Node<XNode>::create<T>(name, runtime, std::forward<Args>(args)...);
    shared_ptr<T> ptr = dynamic_pointer_cast<T>(XNode::stl_thisCreating->back());
    XNode::stl_thisCreating->pop_back();
    return ptr;
}

template <class T, typename... Args>
shared_ptr<T>
XNode::create(Transaction &tr, const char *name, bool runtime, Args&&... args) {
    shared_ptr<T> ptr(createOrphan<T>(name, runtime, std::forward<Args>(args)...));
    if(ptr) insert(tr, ptr, true);
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
