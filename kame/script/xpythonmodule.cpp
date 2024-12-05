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
#include "xpythonmodule.h"
#include "xpythonsupport.h"

#include <pybind11/iostream.h>
#include <pybind11/stl.h>
#include <pybind11/complex.h>
#include <pybind11/functional.h>
#include <pybind11/chrono.h>

#include "xlistnode.h"
#include "xitemnode.h"

#include "recorder.h"
#include "driver.h"
#include "analyzer.h"
#include "primarydriver.h"
#include "primarydriverwiththread.h"
#include "secondarydriver.h"

#include "xnodeconnector.h"
#include <QWidget>

/*TODO

with interfacelock (	 py::gil_scoped_release pyguard and spin)
send (	 py::gil_scoped_release pyguard)
receive(	 py::gil_scoped_release pyguard
query(	 py::gil_scoped_release pyguard
PyQt?

PY_MOD
XScalarEntr
xtime
XWaven
Payload->prop by macro?
push
pop
finishWriting ((	 py::gil_scoped_release pyguard??recursive?)
execute
     py::gil_scoped_aquire pyguard
    or PYBIND11_OVERRIDE_PURE?

DECLARE_TYPE ->
EXPORTXDRIVER(???Driver, "notes")
    .def(bar()).def_readwrite(foo)
    creator of XItemNode<XDriverList, X???Driver>
    add to hashlist of py::object, for dynamic_pointer_cast<X???Driver>,
        tag = typeid().hash
    define Payload class
Snapshot
    .def(__getitem__ py::object) return Snapshot[cast_to_pyobject]
Payload

EXPORTXQCON_TO_PYBOJ(XQ??, Q??)


 */
PYBIND11_DECLARE_HOLDER_TYPE(T, local_shared_ptr<T>, true)

namespace py = pybind11;
std::map<size_t, std::function<py::object(const shared_ptr<XNode>&)>> XPython::s_xnodeDownCasters;
std::map<size_t, std::function<py::object(const shared_ptr<XNode::Payload>&)>> XPython::s_payloadDownCasters;

template <class N, class Base>
XPython::classtype_xnode<N, Base>
XPython::export_xnode(pybind11::module_ &m) {
    XPython::s_xnodeDownCasters.insert(std::make_pair(typeid(N).hash_code(), [](const shared_ptr<XNode>&x)->py::object{
        return py::cast(dynamic_pointer_cast<N>(x));
    }));
    XPython::s_payloadDownCasters.insert(std::make_pair(typeid(typename N::Payload).hash_code(), [](const shared_ptr<XNode::Payload>&x)->py::object{
        return py::cast(dynamic_pointer_cast<typename N::Payload>(x));
    }));
    XString name = typeid(N).name();
    int i = name.find('X');
    name = name.substr(i + 1);
    auto pynode = std::make_unique<py::class_<N, Base, shared_ptr<N>>>(m, name.c_str());
    pynode->def(py::init([](const shared_ptr<XNode> &x){return dynamic_pointer_cast<N>(x);}));
    auto pypayload = std::make_unique<py::class_<typename N::Payload, typename Base::Payload>>(m, (name + "::Payload").c_str());
    return {std::move(pynode), std::move(pypayload)};
}

template <class N, class V>
XPython::classtype_xnode<N, XValueNodeBase>
XPython::export_xvaluenode(pybind11::module_ &m) {
    constexpr const char *pyv = (std::is_integral<V>::value || std::is_same<V, bool>::value) ? "__int__" :
        (std::is_floating_point<V>::value ? "__double__" :
        (std::is_convertible<V, std::string>::value ? "__str__" : ""));
    auto [pynode, pypayload] = export_xnode<N, XValueNodeBase>(m);
    (*pynode)
        .def(pyv, [](shared_ptr<N> &self)->V{return ***self;})
        .def("set", [](shared_ptr<N> &self, V x){trans(*self) = x;});
    (*pypayload)
        .def(pyv, [](typename N::Payload &self)->V{ return self;})
        .def("set", [](typename N::Payload &self, V x){self.operator=(x);});
    return {std::move(pynode), std::move(pypayload)};
}

py::object XPython::cast_to_pyobject(shared_ptr<XNode::Payload> y) {
    auto it = s_payloadDownCasters.find(typeid(y).hash_code());
    if(it != s_payloadDownCasters.end()) {
        return (it->second)(y);
    }
    return py::cast(y);
}
py::object XPython::cast_to_pyobject(shared_ptr<XNode> y) {
    auto it = s_xnodeDownCasters.find(typeid(y).hash_code());
    if(it != s_xnodeDownCasters.end()) {
        return (it->second)(y);
    }
    //manages to use its super class.
    if(auto x = dynamic_pointer_cast<XValueNodeBase>(y)) {
        if(auto x = dynamic_pointer_cast<XIntNode>(y))
            return py::cast(x);
        if(auto x = dynamic_pointer_cast<XUIntNode>(y))
            return py::cast(x);
        if(auto x = dynamic_pointer_cast<XLongNode>(y))
            return py::cast(x);
        if(auto x = dynamic_pointer_cast<XULongNode>(y))
            return py::cast(x);
        if(auto x = dynamic_pointer_cast<XHexNode>(y))
            return py::cast(x);
        if(auto x = dynamic_pointer_cast<XBoolNode>(y))
            return py::cast(x);
        if(auto x = dynamic_pointer_cast<XDoubleNode>(y))
            return py::cast(x);
        if(auto x = dynamic_pointer_cast<XStringNode>(y))
            return py::cast(x);
        if(auto x = dynamic_pointer_cast<XItemNodeBase>(y)) {
            if(auto z = dynamic_pointer_cast<XComboNode>(x))
                return py::cast(z);
            return py::cast(x);
        }
        return py::cast(x);
    }
    if(auto x = dynamic_pointer_cast<XListNodeBase>(y))
        return py::cast(x);
    if(auto x = dynamic_pointer_cast<XTouchableNode>(y))
        return py::cast(x);
    //manages to use its base class.
    for(auto &&c: s_xnodeDownCasters) {
        auto x = (c.second)(y);
        if(x.cast<shared_ptr<XNode>>())
            return x;
    }
    //end up with XNode.
    return py::cast(y);
};


//For XQ**Connector
PYBIND11_DECLARE_HOLDER_TYPE(T, qshared_ptr<T>, true)

template <class QN, class Base, class N>
auto
export_xqcon(pybind11::module_ &m) {
    auto pyc = py::class_<XQConnectorHolder_, qshared_ptr<XQConnectorHolder_>>(m, typeid(QN).name());

    pyc.def(py::init([](const shared_ptr<N> &node, py::object widget){
        if( !isMainThread())
            throw std::runtime_error("Be called from main thread.");
        //todo getChild, main thread
        return xqcon_create<QN>(node, py::cast<QWidget*>(widget));}));
    return std::move(pyc);
}


PYBIND11_EMBEDDED_MODULE(kame, m) {

    auto bound_xnode = py::class_<XNode, shared_ptr<XNode>>(m, "Node")
        .def("__repr__", [](shared_ptr<XNode> &self)->std::string{
            return formatString("<node[%s]\"%s\"@%p>", self->getTypename().c_str(), self->getName().c_str(), &*self);
        })
        .def("__len__", [](shared_ptr<XNode> &self){return Snapshot( *self).size();})
        .def("getLabel", [](shared_ptr<XNode> &self)->std::string{return self->getLabel();})
        .def("getName", [](shared_ptr<XNode> &self)->std::string{return self->getName();})
        .def("getTypename", [](shared_ptr<XNode> &self)->std::string{return self->getTypename();})
        .def("isRuntime", [](shared_ptr<XNode> &self)->bool{return Snapshot( *self)[ *self].isRuntime();})
        .def("isDisabled", [](shared_ptr<XNode> &self)->bool{return Snapshot( *self)[ *self].isDisabled();})
        .def("disable", [](shared_ptr<XNode> &self) {self->disable();})
        .def("setUIEnabled", [](shared_ptr<XNode> &self, bool v) {self->setUIEnabled(v);})
        .def("isUIEnabled", [](shared_ptr<XNode> &self)->bool{return Snapshot( *self)[ *self].isUIEnabled();});
    py::class_<XNode::Payload>(m, "Node::Payload")
        .def("disable", [](XNode::Payload &self) {self.disable();})
        .def("setUIEnabled", [](XNode::Payload &self, bool v) {self.setUIEnabled(v);})
        .def("isRuntime", [](XNode::Payload &self)->bool{return self.isRuntime();})
        .def("isDisabled", [](XNode::Payload &self)->bool{return self.isDisabled();})
        .def("isUIEnabled", [](XNode::Payload &self)->bool{return self.isUIEnabled();});
    py::class_<Snapshot>(m, "Snapshot")
        .def(py::init([](const shared_ptr<XNode> &x){return Snapshot(*x);}), py::keep_alive<1, 2>())
        .def("__repr__", [](Snapshot &self)->std::string{
            return formatString("<Snapshot@%p>", &self);
        })
        .def("size", [](Snapshot &self, const shared_ptr<XNode>&node){return self.size(node);})
        .def("list", [](Snapshot &self)->std::vector<shared_ptr<XNode>>{
            std::vector<shared_ptr<XNode>> v;
            if(self.size())
                v.insert(v.begin(), self.list()->begin(), self.list()->end());
            return v;
        })
        .def("list", [](Snapshot &self, shared_ptr<XNode>&node){
            std::vector<shared_ptr<XNode>> v;
            if(self.size(node))
                v.insert(v.begin(), self.list(node)->begin(), self.list(node)->end());
            return v;
        })
        .def("__len__", [](Snapshot &self){return self.size();})
        .def("__getitem__", [](Snapshot &self, unsigned int pos)->shared_ptr<XNode>{
            return self.size() ? self.list()->at(pos) : shared_ptr<XNode>();}
        )
        .def("__getitem__", [](Snapshot &self, shared_ptr<XNode> &node)->py::object{
            return py::cast( &self[ *node]);
        }, py::return_value_policy::reference_internal)
        .def("isUpperOf", &Snapshot::isUpperOf);
    py::class_<Transaction, Snapshot>(m, "Transaction")
        .def(py::init([](const shared_ptr<XNode> &x){return Transaction(*x);}), py::keep_alive<1, 2>())
//        .def("__enter__", ([](const shared_ptr<XNode> &x){return Transaction(*x);}), py::keep_alive<1, 2>())
//        .def("__exit__", [](Transaction &self, pybind11::args){})
        .def("__repr__", [](Transaction &self)->std::string{
            return formatString("<Transaction@%p>", &self);
        })
        .def("commit", [](Transaction &self) {
            return self.commit();
        })
        .def("commitOrNext", [](Transaction &self) {
            return self.commitOrNext();
        })
        .def("__getitem__", [](Snapshot &self, shared_ptr<XNode> &node)->py::object{
            return py::cast( &self[ *node]);
        }, py::return_value_policy::reference_internal)
        .def("__setitem__", [](Transaction &self, const shared_ptr<XNode> &y, int v){
            if(auto x = dynamic_pointer_cast<XValueNodeBase>(y)) {
                if(auto x = dynamic_pointer_cast<XIntNode>(y))
                    self[ *x] = v;
                else if(auto x = dynamic_pointer_cast<XUIntNode>(y))
                    self[ *x] = v;
                else if(auto x = dynamic_pointer_cast<XLongNode>(y))
                    self[ *x] = v;
                else if(auto x = dynamic_pointer_cast<XULongNode>(y))
                    self[ *x] = v;
                else if(auto x = dynamic_pointer_cast<XHexNode>(y))
                    self[ *x] = v;
                else if(auto x = dynamic_pointer_cast<XBoolNode>(y))
                    self[ *x] = v;
                else if(auto x = dynamic_pointer_cast<XDoubleNode>(y))
                    self[ *x] = v;
                else if(auto x = dynamic_pointer_cast<XComboNode>(y))
                    self[ *x] = v;
                else throw std::runtime_error("Error: type mismatch.");
            }
            else
                throw std::runtime_error("Error: not a value node.");
        })
        .def("__setitem__", [](Transaction &self, const shared_ptr<XNode> &y, const std::string &v){
            if(auto x = dynamic_pointer_cast<XValueNodeBase>(y))
                self[ *x].str(v);
            else
                throw std::runtime_error("Error: not a value node.");
        })
        .def("__setitem__", [](Transaction &self, const shared_ptr<XNode> &y, double v){
            if(auto x = dynamic_pointer_cast<XValueNodeBase>(y)) {
                if(auto x = dynamic_pointer_cast<XDoubleNode>(y))
                    self[ *x] = v;
                else
                    throw std::runtime_error("Error: type mismatch.");
            }
            else
                throw std::runtime_error("Error: not a value node.");
        });

    {   auto [node, payload] = XPython::export_xnode<XListNodeBase, XNode>(m);
        (*node)
        .def("release", [](shared_ptr<XListNodeBase> &self, shared_ptr<XNode> &child){self->release(child);})
        .def("createByTypename", &XListNodeBase::createByTypename);}
    {   auto [node, payload] = XPython::export_xnode<XTouchableNode, XNode>(m);
        (*node)
        .def("touch", [](shared_ptr<XTouchableNode> &self){trans(*self).touch();});}
    {   auto [node, payload] = XPython::export_xnode<XValueNodeBase, XNode>(m);
        (*node)
        .def("__str__", [](shared_ptr<XValueNodeBase> &self)->std::string{return Snapshot( *self)[*self].to_str();})
        .def("set", [](shared_ptr<XValueNodeBase> &self, const std::string &s){trans(*self).str(s);});}
    {   auto [node, payload] = XPython::export_xnode<XItemNodeBase, XValueNodeBase>(m);
        (*node)
        .def("itemStrings", &XItemNodeBase::itemStrings)
        .def("autoSetAny", &XItemNodeBase::autoSetAny);}
    XPython::export_xvaluenode<XIntNode, int>(m);
    XPython::export_xvaluenode<XUIntNode, unsigned int>(m);
    XPython::export_xvaluenode<XLongNode, long>(m);
    XPython::export_xvaluenode<XULongNode, unsigned long>(m);
    XPython::export_xvaluenode<XHexNode, unsigned long>(m);
    XPython::export_xvaluenode<XBoolNode, bool>(m);
    XPython::export_xvaluenode<XDoubleNode, double>(m);
    XPython::export_xvaluenode<XStringNode, std::string>(m);
    {   auto [node, payload] = XPython::export_xnode<XComboNode, XItemNodeBase>(m);
        (*node)
        .def("add", [](shared_ptr<XComboNode> &self, const std::string &s){trans(*self).add(s);})
        .def("add", [](shared_ptr<XComboNode> &self, const std::vector<std::string> &strlist){
            self->iterate_commit([=](Transaction &tr){
                for(auto &s: strlist)
                    tr[ *self].add(s);
            });
        })
        .def("set", [](shared_ptr<XComboNode> &self, const std::string &s){trans(*self) = s;})
        .def("set", [](shared_ptr<XComboNode> &self, int x){trans(*self) = x;});}
    bound_xnode.def("__getitem__", [](shared_ptr<XNode> &self, unsigned int pos)->py::object {
            Snapshot shot( *self);
            if( !shot.size())
                throw std::out_of_range("Empty node.");
            return XPython::cast_to_pyobject(shot.list()->at(pos));
        })
        .def("dynamic_cast", [](shared_ptr<XNode> &self)->py::object {return XPython::cast_to_pyobject(self);})
        .def("__getitem__", [](shared_ptr<XNode> &self, const std::string &str)->py::object{
            auto y = self->getChild(str);
            return XPython::cast_to_pyobject(y);
        })
        .def("__setitem__", [](shared_ptr<XNode> &self, const std::string &str, int v){
            auto y = self->getChild(str);
            if(auto x = dynamic_pointer_cast<XValueNodeBase>(y)) {
                if(auto x = dynamic_pointer_cast<XIntNode>(y))
                    trans( *x) = v;
                else if(auto x = dynamic_pointer_cast<XUIntNode>(y))
                    trans( *x) = v;
                else if(auto x = dynamic_pointer_cast<XLongNode>(y))
                    trans( *x) = v;
                else if(auto x = dynamic_pointer_cast<XULongNode>(y))
                    trans( *x) = v;
                else if(auto x = dynamic_pointer_cast<XHexNode>(y))
                    trans( *x) = v;
                else if(auto x = dynamic_pointer_cast<XBoolNode>(y))
                    trans( *x) = v;
                else if(auto x = dynamic_pointer_cast<XDoubleNode>(y))
                    trans( *x) = v;
                else if(auto x = dynamic_pointer_cast<XComboNode>(y))
                    trans( *x) = v;
                else throw std::runtime_error("Error: type mismatch.");
            }
            else
                throw std::runtime_error("Error: not a value node.");
        })
        .def("__setitem__", [](shared_ptr<XNode> &self, const std::string &str, const std::string &v){
            auto y = self->getChild(str);
            if(auto x = dynamic_pointer_cast<XValueNodeBase>(y))
                trans( *x).str(v);
            else
                throw std::runtime_error("Error: not a value node.");
        })
        .def("__setitem__", [](shared_ptr<XNode> &self, const std::string &str, double v){
            auto y = self->getChild(str);
            if(auto x = dynamic_pointer_cast<XValueNodeBase>(y)) {
                if(auto x = dynamic_pointer_cast<XDoubleNode>(y))
                    trans( *x) = v;
                else
                    throw std::runtime_error("Error: type mismatch.");
            }
            else
                throw std::runtime_error("Error: not a value node.");
        })
        .def("iterate_commit", [](shared_ptr<XNode> &self, py::object pyfunc)->Snapshot {
            return self->iterate_commit([=](Transaction &tr){
                pyfunc(tr);
            });
        });

    //Driver classes
    {   auto [node, payload] = XPython::export_xnode<XDriver, XNode>(m);
        (*node)
        .def("showForms", [](shared_ptr<XDriver> &driver){
            if( !isMainThread())
                throw std::runtime_error("Be called from main thread.");
            driver->showForms();});
        (*payload)
        .def("time", [](XDriver::Payload &self)->system_clock::time_point{return self.time();})
        .def("timeAwared", [](XDriver::Payload &self)->system_clock::time_point{return self.timeAwared();});}
    {   auto [node, payload] = XPython::export_xnode<XScalarEntry, XNode>(m);
        (*node)
            .def("driver", &XScalarEntry::driver)
            .def("value", [](shared_ptr<XScalarEntry> &self, Transaction &tr, double val){self->value(tr, val);})
            .def("storeValue", [](shared_ptr<XScalarEntry> &self, Transaction &tr){ self->storeValue(tr);});
        (*payload)
            .def("isTriggered", &XScalarEntry::Payload::isTriggered);
    }

    XPython::export_xnode<XPointerItemNode<XDriverList>, XItemNodeBase>(m);

//    {   auto [node, payload] = XPython::export_xnode<XSecondaryDriver, XDriver>(m);
//        (*node)
//            .def("requestAnalysis", &XSecondaryDriver::requestAnalysis)
//            .def("connect", [](const shared_ptr<XPointerItemNode<XDriverList> > &selecter){self->connect(selecter);});
//        (*payload);
//    }


    //XQ**Connector
    py::class_<XQConnectorHolder_, qshared_ptr<XQConnectorHolder_>>(m, "XQConnector")
        .def("__repr__", [](qshared_ptr<XQConnectorHolder_> &self)->std::string{
            return formatString("<xqconnector[%s] @%p>", typeid(self).name(), &*self);
        });

}
