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
#ifndef XITEMNODE_H_
#define XITEMNODE_H_

#include "xnode.h"
#include "xlistnode.h"

//! Posses a pointer to a member of a list
class DECLSPEC_KAME XItemNodeBase : public XValueNodeBase {
public:
    explicit XItemNodeBase(const char *name, bool runtime = false, bool auto_set_any = false);
    virtual ~XItemNodeBase() = default;

    struct Item { XString name, label; };
    virtual std::vector<Item> itemStrings(const Snapshot &shot_of_list) const = 0;

    bool autoSetAny() const {return !!m_lsnTryAutoSet;}

    struct DECLSPEC_KAME Payload : public XValueNodeBase::Payload {
        Payload() : XValueNodeBase::Payload() {}
        struct ListChangeEvent {
            ListChangeEvent(const Snapshot &s, XItemNodeBase *e) : shot_of_list(s), emitter(e) {}
            Snapshot shot_of_list;
            XItemNodeBase *emitter;
        };
        Talker<ListChangeEvent> &onListChanged() {return m_tlkOnListChanged;}
        const Talker<ListChangeEvent> &onListChanged() const {return m_tlkOnListChanged;}
    private:
        TalkerSingleton<ListChangeEvent> m_tlkOnListChanged;
    };
private:
    shared_ptr<XListener> m_lsnTryAutoSet;
    void onTryAutoSet(const Snapshot &shot, const Payload::ListChangeEvent &e);
};

DECLSPEC_KAME void
xpointeritemnode_throwConversionError_();

template <class TL>
class XPointerItemNode : public XItemNodeBase {
public:
    XPointerItemNode(const char *name, bool runtime, Transaction &tr_list,
        const shared_ptr<TL> &list, bool auto_set_any = false)
        :  XItemNodeBase(name, runtime, auto_set_any), m_list(list) {
        m_lsnOnItemReleased = tr_list[ *list].onRelease().connect( *this, &XPointerItemNode<TL>::onItemReleased);
        m_lsnOnListChanged = tr_list[ *list].onListChanged().connect( *this, &XPointerItemNode<TL>::lsnOnListChanged);
    }
    virtual ~XPointerItemNode() = default;

    struct Payload : public XItemNodeBase::Payload {
        operator shared_ptr<XNode>() const { return m_var.lock();}
        virtual XString to_str() const {
            shared_ptr<XNode> node( *this);
            if(node)
                return node->getLabel();
            else
                return XString();
        }
        Payload &operator=(const shared_ptr<XNode> &t) {
            m_var = t;
            tr().mark(onValueChanged(), static_cast<XValueNodeBase*>( &node()));
            return *this;
        }
    protected:
        virtual void str_(const XString &var) {
            if(var.empty()) {
                *this = shared_ptr<XNode>();
                return;
            }
            if(auto list = static_cast<const XPointerItemNode&>(node()).m_list.lock()) {
                Snapshot shot( *list);
                if(shot.size()) {
                    for(auto it = shot.list()->begin(); it != shot.list()->end(); ++it) {
                        if(( *it)->getLabel() == var) {
                            *this = *it;
                            return;
                        }
                    }
                }
            }
            xpointeritemnode_throwConversionError_();
        }
        weak_ptr<XNode> m_var;
    };
private:
    void onItemReleased(const Snapshot& /*shot*/, const XListNodeBase::Payload::ReleaseEvent &e) {
        for(Snapshot shot( *this);;) {
            if(e.released != (shared_ptr<XNode>)shot[ *this])
                break;
            Transaction tr(shot);
            tr[ *this] = shared_ptr<XNode>();
            if(tr.commit()) break;
        }
    }
    void lsnOnListChanged(const Snapshot& shot, XListNodeBase* node) {
        if(auto list = m_list.lock()) {
            assert(node == list.get());
            typename Payload::ListChangeEvent e(shot, this);
            Snapshot( *this).talk(( **this)->onListChanged(), std::move(e));
        }
    }
    shared_ptr<XListener> m_lsnOnItemReleased, m_lsnOnListChanged;
protected:
    weak_ptr<TL> m_list;
};

template <class TL, class... VT>
class XItemNode;

template <class TL>
class XItemNode<TL> : public XPointerItemNode<TL> {
protected:
    XItemNode(const char *name, bool runtime, Transaction &tr_list,
        const shared_ptr<TL> &list, bool auto_set_any = false)
        :  XPointerItemNode<TL>(name, runtime, tr_list, list, auto_set_any) {
    }
    virtual ~XItemNode() = default;
    virtual std::vector<XItemNodeBase::Item> itemStrings(const Snapshot &) const {
        return std::vector<XItemNodeBase::Item>();
    }
};

//! A pointer to a XListNode TL, T1 (and VT) are value types
//! template <class TL, class... VT>
template <class TL, class T1, class... VT>
class XItemNode<TL, T1, VT...> : public XItemNode<TL, VT...> {
public:
    XItemNode(const char *name, bool runtime, Transaction &tr_list,
        const shared_ptr<TL> &list, bool auto_set_any = false)
        :  XItemNode<TL, VT...>(name, runtime, tr_list, list, auto_set_any) {
    }
    virtual ~XItemNode() = default;

    struct Payload : public XItemNode<TL, VT...>::Payload {
        Payload() : XItemNode<TL, VT...>::Payload() {}
        operator shared_ptr<T1>() const {
            return dynamic_pointer_cast<T1>(shared_ptr<XNode>( *this));
        }
        Payload &operator=(const shared_ptr<XNode> &t) {
            XPointerItemNode<TL>::Payload::operator=(t);
            return *this;
        }
    };

    virtual std::vector<XItemNodeBase::Item> itemStrings(const Snapshot &shot) const {
        auto items = this->XItemNode<TL, VT...>::itemStrings(shot);
        if(auto list = this->m_list.lock()) {
            if(shot.size(list)) {
                for(auto it = shot.list(list)->begin(); it != shot.list(list)->end(); ++it) {
                    if(dynamic_pointer_cast<T1>( *it)) {
                        XItemNodeBase::Item item;
                        item.name = ( *it)->getName();
                        item.label = ( *it)->getLabel();
                        items.push_back(std::move(item));
                    }
                }
            }
        }
        return std::move(items);
    }
};

//! Contains strings, value is one of strings
class DECLSPEC_KAME XComboNode : public XItemNodeBase {
public:
    explicit XComboNode(const char *name, bool runtime = false, bool auto_set_any = false);
    virtual ~XComboNode() = default;

    virtual std::vector<XItemNodeBase::Item> itemStrings(const Snapshot &shot) const {
        return shot[ *this].itemStrings();
    }

    struct DECLSPEC_KAME Payload : public XItemNodeBase::Payload {
        Payload() : XItemNodeBase::Payload(), m_strings(std::make_shared<std::deque<XString>>()),
            m_var(std::pair<XString, int>("", -1)) {}
        void add(const XString &str);
        void add(std::initializer_list<XString> strlist) {for(auto &&x: strlist){ add(x);}}
        void clear();
        operator int() const { return m_var.second;}
        virtual XString to_str() const { return m_var.first;}
        Payload &operator=(int t);
        Payload &operator=(const XString &);
        virtual std::vector<XItemNodeBase::Item> itemStrings() const;
    protected:
        virtual void str_(const XString &);
    private:
        shared_ptr<std::deque<XString> > m_strings;
        std::pair<XString, int> m_var;
    };
};

#endif /*XITEMNODE_H_*/
