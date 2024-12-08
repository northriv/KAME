/***************************************************************************
        Copyright (C) 2002-2024 Kentaro Kitagawa
		                   kitag@issp.u-tokyo.ac.jp
		
		This program is free software; you can redistribute it and/or
		modify it under the terms of the GNU Library General Public
		License as published by the Free Software Foundation; either
		version 2 of the License, or (at your option) any later version.
		
		You should have received a copy of the GNU Library General 
		Public License and a list of authors along with this program; 
		see the files COPYING and AUTHORS.
***************************************************************************/
//---------------------------------------------------------------------------

#ifndef xpythonsupportH
#define xpythonsupportH

#ifdef USE_PYBIND11

#include "xpythonmodule.h"
#include "xscriptingthread.h"
#include <map>

class XMeasure;

//! Python scripting support, containing a thread running python monitor program.
//! The monitor program synchronize Ruby threads and XScriptingThread objects.
//! \sa XScriptingThread
class XPython : public XScriptingThreadList {
public:
    XPython(const char *name, bool runtime, const shared_ptr<XMeasure> &measure);
    virtual ~XPython();

    //!internal use. down casters.
    static pybind11::object cast_to_pyobject(shared_ptr<XNode> y);
    static pybind11::object cast_to_pyobject(shared_ptr<XNode::Payload> y);

    template <class N, class Base, class Trampoline>
    using classtype_xnode_with_trampoline = std::tuple<unique_ptr<pybind11::class_<N, Base, Trampoline, shared_ptr<N>>>,
    unique_ptr<pybind11::class_<typename N::Payload, typename Base::Payload, typename Trampoline::Payload>>>;
    template <class N, class Base>
    using classtype_xnode = std::tuple<unique_ptr<pybind11::class_<N, Base, shared_ptr<N>>>,
    unique_ptr<pybind11::class_<typename N::Payload, typename Base::Payload>>>;
    //! Wraps C++ XNode-derived classes N, along with N::Payload.
    //! N derived from Base, and N::Payload derived from Base::Payload.
    //! \return to be used by .def or else. use auto [node, payload] =....
    template <class N, class Base, class Trampoline, typename...Args>
    classtype_xnode_with_trampoline<N, Base, Trampoline>
    static export_xnode_with_trampoline(const char *name = nullptr);

    template <class N, class Base, typename...Args>
    classtype_xnode<N, Base> static export_xnode(const char *name = nullptr);

    template <class N, class V, typename...Args>
    classtype_xnode<N, XValueNodeBase> static export_xvaluenode(const char *name = nullptr);

    template <class D, class Base, class Trampoline>
    classtype_xnode_with_trampoline<D, Base, Trampoline>
    static export_xpythondriver(const char *name = nullptr);

    static pybind11::module_&kame_module() {return s_kame_module;}
    static pybind11::module_ s_kame_module;
protected:
    virtual void *execute(const atomic<bool> &) override;
    void my_defout(shared_ptr<XNode> node, const std::string &msg);
    std::string my_defin(shared_ptr<XNode> node);

private:
    static std::map<size_t, std::function<pybind11::object(const shared_ptr<XNode>&)>> s_xnodeDownCasters;
    static std::map<size_t, std::function<pybind11::object(const shared_ptr<XNode::Payload>&)>> s_payloadDownCasters;

    XCondition m_mainthread_cb_cond;
    Transactional::Talker<pybind11::object*, pybind11::object*, pybind11::object*, pybind11::object*> m_mainthread_cb_tlk;
    shared_ptr<Listener> m_mainthread_cb_lsn;

    void mainthread_callback(pybind11::object *scrthread, pybind11::object *func, pybind11::object *ret, pybind11::object *status);

    template <class N>
    static std::string declare_xnode_downcasters();

//    template <class T> friend class XPythonDriver;
//    static XThreadLocal<shared_ptr<XNode>> stl_nodeCreating; //to be used inside lambda creation fn of exportClass().
};

#endif //USE_PYBIND11
//---------------------------------------------------------------------------
#endif //
