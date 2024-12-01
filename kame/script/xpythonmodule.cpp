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

namespace py = pybind11;
std::map<size_t, std::function<py::object(const shared_ptr<XNode>&)>> XPython::s_xnodeDownCasters;

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
    //manages to use its super class.
    for(auto &&c: s_xnodeDownCasters) {
        auto x = (c.second)(y);
        if(py::cast<bool>(x))
            return x;
    }
    //end up with XNode.
    return py::cast(y);
};
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
        .def("isUIEnabled", [](shared_ptr<XNode> &self)->bool{return Snapshot( *self)[ *self].isUIEnabled();});
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

    py::class_<XListNodeBase, XNode, shared_ptr<XListNodeBase>>(m, "ListNode")
        .def(py::init([](const shared_ptr<XNode> &x){return dynamic_pointer_cast<XListNodeBase>(x);}))
        .def("release", [](shared_ptr<XListNodeBase> &self, shared_ptr<XNode> &child){self->release(child);})
        .def("createByTypename", &XListNodeBase::createByTypename);
    py::class_<XTouchableNode, XNode, shared_ptr<XTouchableNode>>(m, "TouchableNode")
        .def(py::init([](const shared_ptr<XNode> &x){return dynamic_pointer_cast<XTouchableNode>(x);}))
        .def("touch", [](shared_ptr<XTouchableNode> &self){trans(*self).touch();});
    py::class_<XValueNodeBase, XNode, shared_ptr<XValueNodeBase>>(m, "ValueNode")
        .def(py::init([](const shared_ptr<XNode> &x){return dynamic_pointer_cast<XValueNodeBase>(x);}))
        .def("__str__", [](shared_ptr<XValueNodeBase> &self)->std::string{return Snapshot( *self)[*self].to_str();})
        .def("set", [](shared_ptr<XValueNodeBase> &self, const std::string &s){trans(*self).str(s);});
    py::class_<XItemNodeBase, XValueNodeBase, shared_ptr<XItemNodeBase>>(m, "ItemNode")
        .def(py::init([](const shared_ptr<XNode> &x){return dynamic_pointer_cast<XItemNodeBase>(x);}))
        .def("itemStrings", &XItemNodeBase::itemStrings)
        .def("autoSetAny", &XItemNodeBase::autoSetAny);
    py::class_<XIntNode, XNode, shared_ptr<XIntNode>>(m, "IntNode")
        .def(py::init([](const shared_ptr<XNode> &x){return dynamic_pointer_cast<XIntNode>(x);}))
        .def("__int__", [](shared_ptr<XIntNode> &self)->int{return ***self;})
        .def("set", [](shared_ptr<XIntNode> &self, int x){trans(*self) = x;});
    py::class_<XUIntNode, XNode, shared_ptr<XUIntNode>>(m, "UIntNode")
        .def(py::init([](const shared_ptr<XNode> &x){return dynamic_pointer_cast<XUIntNode>(x);}))
        .def("__int__", [](shared_ptr<XIntNode> &self)->unsigned int{return ***self;})
        .def("set", [](shared_ptr<XUIntNode> &self, unsigned int x){trans(*self) = x;});
    py::class_<XLongNode, XNode, shared_ptr<XLongNode>>(m, "LongNode")
        .def(py::init([](const shared_ptr<XNode> &x){return dynamic_pointer_cast<XLongNode>(x);}))
        .def("__int__", [](shared_ptr<XLongNode> &self)->long{return ***self;})
        .def("set", [](shared_ptr<XLongNode> &self, long x){trans(*self) = x;});
    py::class_<XULongNode, XNode, shared_ptr<XULongNode>>(m, "ULongNode")
        .def(py::init([](const shared_ptr<XNode> &x){return dynamic_pointer_cast<XULongNode>(x);}))
        .def("__int__", [](shared_ptr<XULongNode> &self)->unsigned long{return ***self;})
        .def("set", [](shared_ptr<XULongNode> &self, unsigned long x){trans(*self) = x;});
    py::class_<XHexNode, XNode, shared_ptr<XHexNode>>(m, "HexNode")
        .def(py::init([](const shared_ptr<XNode> &x){return dynamic_pointer_cast<XHexNode>(x);}))
        .def("__int__", [](shared_ptr<XHexNode> &self)->unsigned long{return ***self;})
        .def("set", [](shared_ptr<XHexNode> &self, unsigned long x){trans(*self) = x;});
    py::class_<XBoolNode, XNode, shared_ptr<XBoolNode>>(m, "BoolNode")
        .def(py::init([](const shared_ptr<XNode> &x){return dynamic_pointer_cast<XBoolNode>(x);}))
        .def("__int__", [](shared_ptr<XBoolNode> &self)->bool{return ***self;})
        .def("set", [](shared_ptr<XBoolNode> &self, bool x){trans(*self) = x;});
    py::class_<XDoubleNode, XNode, shared_ptr<XDoubleNode>>(m, "DoubleNode")
        .def(py::init([](const shared_ptr<XNode> &x){return dynamic_pointer_cast<XDoubleNode>(x);}))
        .def("__float__", [](shared_ptr<XDoubleNode> &self)->double{return ***self;})
        .def("set", [](shared_ptr<XDoubleNode> &self, double x){trans(*self) = x;});
    py::class_<XStringNode, XNode, shared_ptr<XStringNode>>(m, "StringNode")
        .def(py::init([](const shared_ptr<XNode> &x){return dynamic_pointer_cast<XStringNode>(x);}))
        .def("__str__", [](shared_ptr<XStringNode> &self)->std::string{return ***self;})
        .def("set", [](shared_ptr<XStringNode> &self, const std::string &s){trans(*self) = s;});
    py::class_<XComboNode, XNode, shared_ptr<XComboNode>>(m, "ComboNode")
        .def(py::init([](const shared_ptr<XNode> &x){return dynamic_pointer_cast<XComboNode>(x);}))
        .def("add", [](shared_ptr<XComboNode> &self, const std::string &s){trans(*self).add(s);})
        .def("add", [](shared_ptr<XComboNode> &self, const std::vector<std::string> &strlist){
            self->iterate_commit([=](Transaction &tr){
                for(auto &s: strlist)
                    tr[ *self].add(s);
            });
        })
        .def("set", [](shared_ptr<XComboNode> &self, const std::string &s){trans(*self) = s;})
        .def("set", [](shared_ptr<XComboNode> &self, int x){trans(*self) = x;});
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
}
