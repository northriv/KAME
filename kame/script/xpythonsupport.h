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

struct KAMEPyBind {
    template <class N, class Base, class Trampoline>
//    using classtype_xnode_with_trampoline = typename std::conditional<
//        !std::is_base_of<typename N::Payload, typename Base::Payload>::value,
//        std::tuple<unique_ptr<pybind11::class_<N, Base, Trampoline, shared_ptr<N>>>, //N::Payload is defined.
//            unique_ptr<pybind11::class_<typename N::Payload, typename Base::Payload, typename Trampoline::Payload>>>,
//        unique_ptr<pybind11::class_<N, Base, Trampoline, shared_ptr<N>>>>; //N::Payload is NOT defined.
    using classtype_xnode_with_trampoline =
        std::tuple<unique_ptr<pybind11::class_<N, Base, Trampoline, shared_ptr<N>>>,
            unique_ptr<pybind11::class_<typename N::Payload, typename Base::Payload, typename Trampoline::Payload>>>; //N::Payload is NOT defined.

    template <class N, class Base>
    using classtype_xnode = typename std::conditional<
        !std::is_base_of<typename N::Payload, typename Base::Payload>::value,
            std::tuple<unique_ptr<pybind11::class_<N, Base, shared_ptr<N>>>, //N::Payload is defined.
                unique_ptr<pybind11::class_<typename N::Payload, typename Base::Payload>>>,
            unique_ptr<pybind11::class_<N, Base, shared_ptr<N>>>>::type; //N::Payload is NOT defined.
    //! Wraps C++ XNode-derived classes N, along with N::Payload, with Trampoline class.
    //! N derived from Base, and N::Payload derived from Base::Payload.
    //! \return to be used by .def or else. use auto [node, payload] =....
    template <class N, class Base, class Trampoline, typename...Args>
    classtype_xnode_with_trampoline<N, Base, Trampoline>
    export_xnode_with_trampoline(const char *name = nullptr);

    //! Wraps C++ XNode-derived classes N, (along with N::Payload if exists).
    //! N derived from Base, (and N::Payload derived from Base::Payload).
    //! \return to be used by .def or else. use auto [node, payload] =...., or auto node = ....
    template <class N, class Base, typename...Args>
    classtype_xnode<N, Base> export_xnode(const char *name = nullptr);

    //! For node with setter/getter of type V.
    template <class N, class V, class Base, typename...Args>
    classtype_xnode<N, Base> export_xvaluenode(const char *name = nullptr);

    //! For (abstract) driver classes open to python.
    template <class D, class Base>
    classtype_xnode<D, Base>
    export_xdriver(const char *name = nullptr);

    template <class D, class Base, class Trampoline>
    classtype_xnode_with_trampoline<D, Base, Trampoline>
    export_xpythondriver(const char *name = nullptr);

    pybind11::module_& kame_module() {return s_kame_module;}
    pybind11::module_ s_kame_module;

    //!internal use. down casters.
    pybind11::object cast_to_pyobject(shared_ptr<XNode> y);
    pybind11::object cast_to_pyobject(XNode::Payload *y);
    pybind11::object cast_to_pyobject(const XNode::Payload *y) {
        return cast_to_pyobject(const_cast<XNode::Payload *>(y));
    }

    static void export_embedded_module_basic(pybind11::module_&);
    static void export_embedded_module_graph(pybind11::module_&);
    static void export_embedded_module_basic_drivers(pybind11::module_&);
    static void export_embedded_module_interface(pybind11::module_&);
    static void export_embedded_module_xqcon(pybind11::module_&);

private:
    //std::type_index(typeid(x)), serialno, down_caster_func.
    std::unordered_map<std::type_index, std::pair<size_t, std::function<pybind11::object(const shared_ptr<XNode>&)>>> m_xnodeDownCasters;
    std::unordered_map<std::type_index, std::pair<size_t, std::function<pybind11::object(XNode::Payload *)>>> m_payloadDownCasters;

    template <class N, bool IS_PAYLOAD_DEFINED = true>
    std::string declare_xnode_downcasters();
};

//! Python scripting support, containing a thread running python monitor program.
//! The monitor program synchronize Ruby threads and XScriptingThread objects.
//! \sa XScriptingThread
class XPython : public XScriptingThreadList {
public:
    XPython(const char *name, bool runtime, const shared_ptr<XMeasure> &measure);
    virtual ~XPython();

    static KAMEPyBind bind;
protected:
    virtual void *execute(const atomic<bool> &) override;
    void my_defout(shared_ptr<XNode> node, const std::string &msg);
    std::string my_defin(shared_ptr<XNode> node);

private:
    XCondition m_mainthread_cb_cond;
    Transactional::Talker<pybind11::object*, pybind11::object*, pybind11::object*, pybind11::object*> m_mainthread_cb_tlk;
    shared_ptr<Listener> m_mainthread_cb_lsn;

    void mainthread_callback(pybind11::object *scrthread, pybind11::object *func, pybind11::object *ret, pybind11::object *status);
};

#endif //USE_PYBIND11
//---------------------------------------------------------------------------
#endif //
