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
#include <pybind11/embed.h> //include before kame headers
#include <pybind11/iostream.h>
#include <pybind11/stl.h>
#include <pybind11/complex.h>
#include <pybind11/functional.h>
#include <pybind11/chrono.h>

namespace py = pybind11;

#include "xpythonsupport.h"
#include "xscriptingthread.h"
#include "measure.h"
#include <QFile>
#include <QDataStream>
#include <math.h>
#include <iostream>
#include "xlistnode.h"
#include "xitemnode.h"

auto py_dynamic_cast = [](shared_ptr<XNode> y)->py::object{
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
        if(auto x = dynamic_pointer_cast<XItemNodeBase>(y))
            return py::cast(x);
    }
    if(auto x = dynamic_pointer_cast<XListNodeBase>(y))
        return py::cast(x);
    if(auto x = dynamic_pointer_cast<XTouchableNode>(y))
        return py::cast(x);
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
        .def("createByTupename", &XListNodeBase::createByTypename);
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
    //todo combo
    bound_xnode.def("__getitem__", [](shared_ptr<XNode> &self, unsigned int pos)->py::object {
            Snapshot shot( *self);
            if( !shot.size())
                throw std::out_of_range("Empty node.");
            return py_dynamic_cast(shot.list()->at(pos));
        })
        .def("dynamic_cast", [](shared_ptr<XNode> &self)->py::object {return py_dynamic_cast(self);})
        .def("__getitem__", [](shared_ptr<XNode> &self, const std::string &str)->py::object{
            auto y = self->getChild(str);
            return py_dynamic_cast(y);
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

//
#define XPYTHONSUPPORT_RB ":/script/xpythonsupport.py" //in the qrc.

XPython::XPython(const char *name, bool runtime, const shared_ptr<XMeasure> &measure)
    : XScriptingThreadList(name, runtime, measure) {
}
XPython::~XPython() {
}

void
XPython::my_defout(shared_ptr<XNode> node, const std::string &msg) {
//    shared_ptr<XNode> p = Snapshot(*m_measure.lock()->python()).list()->at(0);
    auto scriptthread = dynamic_pointer_cast<XScriptingThread>(node);
    if(scriptthread) {
        Snapshot shot( *scriptthread);
        shot.talk(shot[ *scriptthread].onMessageOut(), std::make_shared<XString>(msg));
        dbgPrint(QString("Python [%1]; %2").arg(shot[ *scriptthread->filename()].to_str()).arg(msg.c_str()));
    }
    else
        fprintf(stderr, "%s\n", msg.c_str());
}
std::string
XPython::my_defin(shared_ptr<XNode> node) {
//    shared_ptr<XNode> p = Snapshot(*m_measure.lock()->python()).list()->at(0);
    auto scriptthread = dynamic_pointer_cast<XScriptingThread>(node);
    if(scriptthread) {
        XString line = scriptthread->gets();
        return line;
    }
    return "";
}

void *
XPython::execute(const atomic<bool> &terminated) {
    Transactional::setCurrentPriorityMode(Transactional::Priority::UI_DEFERRABLE);

    {
        QFile scriptfile(XPYTHONSUPPORT_RB);
        if( !scriptfile.open(QIODevice::ReadOnly | QIODevice::Text)) {
            gErrPrint("No KAME python support file installed.");
            return NULL;
        }
        fprintf(stderr, "Loading python scripting monitor.\n");
        char data[65536];
        QDataStream( &scriptfile).readRawData(data, sizeof(data));

        py::scoped_interpreter guard{}; // start the interpreter and keep it alive

        auto kame_module = py::module_::import("kame");
        shared_ptr<XMeasure> measure = m_measure.lock();
        assert(measure);
        XString name = measure->getName();
        name[0] = toupper(name[0]);
        kame_module.def("Root", [=]()->shared_ptr<XNode>{return measure;});
        kame_module.def("Measurement", [=]()->shared_ptr<XNode>{return measure;});
        kame_module.def("PyInfoForNodeBrowser", [=]()->shared_ptr<XStringNode>{return measure->pyInfoForNodeBrowser();});
        kame_module.def("LastPointedByNodeBrowser", [=]()->py::object {return py_dynamic_cast(measure->lastPointedByNodeBrowser());});
        kame_module.def("my_defout", [=](shared_ptr<XNode> scrthread, const std::string &str){this->my_defout(scrthread, str);});
        kame_module.def("my_defin", [=](shared_ptr<XNode> scrthread)->std::string{return this->my_defin(scrthread);});
        kame_module.def("is_main_terminated", [=](){return this->m_thread->isTerminated();});
        kame_module.def("XScriptingThreads", [=]()->shared_ptr<XListNodeBase>{return dynamic_pointer_cast<XListNodeBase>(this->shared_from_this());});

        py::print("Hello, World!"); // use the Python API
        while( !terminated) {
            try {
                py::object main_scope = py::module_::import("__main__").attr("__dict__");
                py::exec(data, main_scope);
            }
            catch (py::error_already_set& e) {
                std::cout << "Python error.\n" << e.what() << "\n";
            }
            catch (...) {
                std::cout << "Python unknown error.\n" << "\n";
            }
            msecsleep(1000);
        }
    }

    fprintf(stderr, "python fin");
    fprintf(stderr, "ished\n");
    return NULL;
}
