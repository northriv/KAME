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
#include "measure.h"

#include "recorder.h"
#include "driver.h"
#include "analyzer.h"
#include "primarydriver.h"
#include "primarydriverwiththread.h"
#include "secondarydriver.h"
#include "pythondriver.h"
#include "xnodeconnector.h"
#include "graph.h"
#include "xwavengraph.h"
#include "ui_graphnurlform.h"
#include <QWidget>
#include <QAbstractButton>
#include <QLineEdit>
#include <QTextEdit>
#include <QSlider>
#include <QLabel>
#include <QSpinBox>
#include <QPushButton>
#include <QIcon>
#include <QToolButton>
#include <QTableWidget>
#include <QComboBox>
#include <QListWidget>
#include <QLCDNumber>

/*TODO
with interfacelock (	 py::gil_scoped_release pyguard and spin)
send (	 py::gil_scoped_release pyguard)
receive(	 py::gil_scoped_release pyguard
query(	 py::gil_scoped_release pyguard


XWaven
Payload->prop by macro?
push
pop
finishWriting ((	 py::gil_scoped_release pyguard??recursive?)
execute
     py::gil_scoped_aquire pyguard
    or PYBIND11_OVERRIDE_PURE?

 */
PYBIND11_DECLARE_HOLDER_TYPE(T, local_shared_ptr<T>, true)

namespace py = pybind11;

KAMEPyBind XPython::bind; //should be here before PYBIND11_EMBEDDED_MODULE.

PYBIND11_EMBEDDED_MODULE(kame, m) {
    XPython::bind.s_kame_module = m;

    //Binding XNode, Snapshot, Transaction, X***ValueNode, XTime, ...
    KAMEPyBind::export_embedded_module_basic(m);

    //XWaveNGraph
    KAMEPyBind::export_embedded_module_graph(m);

    KAMEPyBind::export_embedded_module_basic_drivers(m);

    KAMEPyBind::export_embedded_module_interface(m);

    KAMEPyBind::export_embedded_module_xqcon(m);
}

py::object KAMEPyBind::cast_to_pyobject(XNode::Payload *y) {
    auto it = m_payloadDownCasters.find(typeid(y).hash_code());
    if(it != m_payloadDownCasters.end()) {
        return (it->second.second)(y);
    }
    //manages to use its downmost base class.
    std::map<size_t, py::object> cand;
    for(auto &c: m_payloadDownCasters) {
        auto x = (c.second.second)(y);
        if(x.cast<XNode::Payload*>())
            cand.insert(std::make_pair(c.second.first, x));
    }
    if(cand.size())
        return cand.rbegin()->second; //the oldest choice.
    return py::cast(y); //end up with XNode::Payload*
}
py::object KAMEPyBind::cast_to_pyobject(shared_ptr<XNode> y) {
    auto it = m_xnodeDownCasters.find(typeid(y).hash_code());
    if(it != m_xnodeDownCasters.end()) {
        return (it->second.second)(y);
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
    //manages to use its downmost base class.
    std::map<size_t, py::object> cand;
    for(auto &c: m_xnodeDownCasters) {
        auto x = (c.second.second)(y);
        if(x.cast<shared_ptr<XNode>>())
            cand.insert(std::make_pair(c.second.first, x));
    }
    if(cand.size())
        return cand.rbegin()->second; //the oldest choice.
    //end up with XNode.
    return py::cast(y);
};

//For XQ**Connector, with customized deleter.
PYBIND11_DECLARE_HOLDER_TYPE(T, qshared_ptr<T>, true)

template <class QN, class N, class QW, typename...Args>
auto
export_xqcon() {
    auto &m = XPython::bind.kame_module();
    XString name = typeid(QN).name();
    int i = name.find('X');
    name = name.substr(i);

    //pybind11 does not allow to wrap the same class.
    //one holder class for one connector class.
    struct XQConnectorHolder_TMP : public XQConnectorHolder_ {
        XQConnectorHolder_TMP(XQConnector *con) : XQConnectorHolder_(con) {}
        ~XQConnectorHolder_TMP() {}
    };

    auto pyc = py::class_<XQConnectorHolder_TMP, XQConnectorHolder_, qshared_ptr<XQConnectorHolder_TMP>>(m, name.c_str());
    pyc.def(py::init([](const shared_ptr<N> &node, py::object widget, Args&&...args){
        if( !isMainThread())
            throw std::runtime_error("Be called from main thread.");
        if(auto x = dynamic_cast<QW*>(py::cast<QWidget*>(widget)))
            return
                qshared_ptr<XQConnectorHolder_TMP>(new XQConnectorHolder_TMP(
                    new QN(node, x, std::forward<Args>(args)...)));
        else
            throw std::runtime_error("Type mismatch.");
        })); //, py::keep_alive<0, 1>()
    return std::move(pyc);
}

void
KAMEPyBind::export_embedded_module_basic(pybind11::module_& m) {
    auto bound_xnode = py::class_<XNode, shared_ptr<XNode>>(m, "XNode")
        .def("__repr__", [](shared_ptr<XNode> &self)->std::string{
            return formatString("<node[%s]\"%s\"@%p>", ("X" + self->getTypename()).c_str(), self->getName().c_str(), &*self);
        })
        .def("insert", [](shared_ptr<XNode> &self, shared_ptr<XNode> &child){self->insert(child);})
        .def("insert", [](shared_ptr<XNode> &self, Transaction &tr, shared_ptr<XNode> &child){self->insert(tr, child);})
        .def("release", [](shared_ptr<XNode> &self, shared_ptr<XNode> &child){self->release(child);})
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
            if(self.size()) return self.list()->at(pos);
            throw XNode::NodeNotFoundError("out of range");}
        )
        .def("__getitem__", [](Snapshot &self, shared_ptr<XNode> &node)->py::object{
            return XPython::bind.cast_to_pyobject( &self.at( *node));
        }, py::return_value_policy::reference_internal)
        .def("isUpperOf", &Snapshot::isUpperOf);
    py::class_<Transaction, Snapshot>(m, "Transaction")
        .def(py::init([](const shared_ptr<XNode> &x){return Transaction(*x);}), py::keep_alive<1, 2>())
        .def("__iter__", [](Transaction &self)->Transaction &{ return self; })
        .def("__next__", [](Transaction &self)->Transaction &{
            if(self.isModified() && self.commitOrNext())
                throw pybind11::stop_iteration();
            else
                return self;
        })
        .def("__repr__", [](Transaction &self)->std::string{
            return formatString("<Transaction@%p>", &self);
        })
        .def("commit", [](Transaction &self) {
            return self.commit();
        })
        .def("commitOrNext", [](Transaction &self) {
            return self.commitOrNext();
        })
        .def("__getitem__", [](Transaction &self, shared_ptr<XNode> &node)->py::object{
            return XPython::bind.cast_to_pyobject( &self[ *node]); //Transaction has no at().
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
                    self[ *x] = v; //deprecated
                else if(auto x = dynamic_pointer_cast<XDoubleNode>(y))
                    self[ *x] = v;
                else if(auto x = dynamic_pointer_cast<XComboNode>(y))
                    self[ *x] = v;
                else throw std::runtime_error("Error: type mismatch.");
            }
            else
                throw std::runtime_error("Error: not a value node.");
        })
        .def("__setitem__", [](Transaction &self, const shared_ptr<XNode> &y, bool v){
            if(auto x = dynamic_pointer_cast<XBoolNode>(y))
                self[ *x] = v; //deprecated
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

    {   auto [node, payload] = XPython::bind.export_xnode<XListNodeBase, XNode>();
        (*node)
        .def("createByTypename", &XListNodeBase::createByTypename);}
    {   auto [node, payload] = XPython::bind.export_xnode<XTouchableNode, XNode>();
        (*node)
        .def("touch", [](shared_ptr<XTouchableNode> &self){trans(*self).touch();});}
    {   auto [node, payload] = XPython::bind.export_xnode<XValueNodeBase, XNode>();
        (*node)
        .def("__str__", [](shared_ptr<XValueNodeBase> &self)->std::string{return Snapshot( *self)[*self].to_str();})
        .def("set", [](shared_ptr<XValueNodeBase> &self, const std::string &s){trans(*self).str(s);});}
    {   auto [node, payload] = XPython::bind.export_xnode<XItemNodeBase, XValueNodeBase>();
        (*node)
        .def("itemStrings", &XItemNodeBase::itemStrings)
        .def("autoSetAny", &XItemNodeBase::autoSetAny);}
    XPython::bind.export_xvaluenode<XIntNode, int, XValueNodeBase>("XIntNode");
    XPython::bind.export_xvaluenode<XUIntNode, unsigned int, XValueNodeBase>("XUIntNode");
    XPython::bind.export_xvaluenode<XLongNode, long, XValueNodeBase>("XLongNode");
    XPython::bind.export_xvaluenode<XULongNode, unsigned long, XValueNodeBase>("XULongNode");
    XPython::bind.export_xvaluenode<XHexNode, unsigned long, XValueNodeBase>("XHexNode");
    XPython::bind.export_xvaluenode<XBoolNode, bool, XValueNodeBase>("XBoolNode");
    XPython::bind.export_xvaluenode<XDoubleNode, double, XValueNodeBase>("XDoubleNode");
    XPython::bind.export_xvaluenode<XStringNode, std::string, XValueNodeBase>("XStringNode");
    {   auto [node, payload] = XPython::bind.export_xnode<XComboNode, XItemNodeBase, bool>("XComboNode");
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
                throw XNode::NodeNotFoundError("out of range");
            return XPython::bind.cast_to_pyobject(shot.list()->at(pos));
        })
        .def("dynamic_cast", [](shared_ptr<XNode> &self)->py::object {return XPython::bind.cast_to_pyobject(self);})
        .def("__getitem__", [](shared_ptr<XNode> &self, const std::string &str)->py::object{
            auto y = self->getChild(str);
            return XPython::bind.cast_to_pyobject(y);
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
                    trans( *x) = v; //deprecated feature.
                else if(auto x = dynamic_pointer_cast<XDoubleNode>(y))
                    trans( *x) = v;
                else if(auto x = dynamic_pointer_cast<XComboNode>(y))
                    trans( *x) = v;
                else throw std::runtime_error("Error: type mismatch.");
            }
            else
                throw std::runtime_error("Error: not a value node.");
        })
        .def("__setitem__", [](shared_ptr<XNode> &self, const std::string &str, bool v){
            auto y = self->getChild(str);
            if(auto x = dynamic_pointer_cast<XBoolNode>(y))
                trans( *x) = v;
            else throw std::runtime_error("Error: type mismatch.");
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

    py::class_<XTime>(m, "XTime")
        .def(py::init([](const system_clock::time_point &t)->XTime{return {t};}));
    py::implicitly_convertible<system_clock::time_point, XTime>();
//    py::implicitly_convertible<XTime, system_clock::time_point>();
    //Exceptions
    py::register_exception<XNode::NodeNotFoundError>(m, "KAMENodeNotFoundError", PyExc_KeyError);
    py::register_exception<XKameError>(m, "KAMEError", PyExc_RuntimeError);
}
void
KAMEPyBind::export_embedded_module_graph(pybind11::module_& m) {
    {   auto [node, payload] = XPython::bind.export_xnode<XGraph, XNode>();
        (*node);
        (*payload);
    }
    XPython::bind.export_xnode<XAxis, XNode>();
    XPython::bind.export_xnode<XPlot, XNode>();
    XPython::bind.export_xnode<XXYPlot, XPlot>();
    XPython::bind.export_xnode<X2DImagePlot, XPlot>();
    XPython::bind.export_xnode<XGraphNToolBox, XNode>();
    {   auto [node, payload] = XPython::bind.export_xnode<XWaveNGraph, XGraphNToolBox,
                XQGraph *, QLineEdit *, QAbstractButton *, QPushButton *>();
    (*node)
        .def("drawGraph", &XWaveNGraph::drawGraph)
        .def("clearPlots", &XWaveNGraph::clearPlots);
    (*payload)
        .def("clearPoints", &XWaveNGraph::Payload::clearPoints)
        .def("insertPlot", [](XWaveNGraph::Payload &self,
             Transaction &tr, const std::string &label, int colx, int coly1, int coly2, int colw, int colz){
            return self.insertPlot(tr, label, colx, coly1, coly2, colw, colz);
        })
        .def("setCols", [](XWaveNGraph::Payload &self, std::initializer_list<std::string> &labels){
            self.setCols(labels);
        })
        .def("colCount", &XWaveNGraph::Payload::colCount)
        .def("rowCount", &XWaveNGraph::Payload::rowCount)
        .def("numPlots", &XWaveNGraph::Payload::numPlots)
        .def("setLabel", &XWaveNGraph::Payload::setLabel)
        .def("setColumn", [](XWaveNGraph::Payload &self, unsigned int n, std::vector<double> &&data, unsigned int prec){
            self.setColumn(n, std::move(data), prec);
        });
    }
}

void
KAMEPyBind::export_embedded_module_basic_drivers(pybind11::module_& m) {
    //Driver classes
    {   auto [node, payload] = XPython::bind.export_xnode<XDriver, XNode>();
        (*node)
        .def("showForms", [](shared_ptr<XDriver> &driver){
            if( !isMainThread())
                throw std::runtime_error("Be called from main thread.");
            driver->showForms();});
        (*payload)
        .def("time", [](XDriver::Payload &self)->system_clock::time_point{return self.time();})
        .def("timeAwared", [](XDriver::Payload &self)->system_clock::time_point{return self.timeAwared();});}
    {   auto [node, payload] = XPython::bind.export_xnode<XScalarEntry, XNode,
            const shared_ptr<XDriver> &, const char *>();
        (*node)
            .def("driver", &XScalarEntry::driver)
            .def("value", [](shared_ptr<XScalarEntry> &self, Transaction &tr, double val){self->value(tr, val);})
            .def("storeValue", [](shared_ptr<XScalarEntry> &self, Transaction &tr){ self->storeValue(tr);});
        (*payload)
            .def("isTriggered", &XScalarEntry::Payload::isTriggered);
    }
    XPython::bind.export_xnode<XDriverList, XListNodeBase>("XDriverList"); //needed to be used as an argument.
    XPython::bind.export_xnode<XPointerItemNode<XDriverList>, XItemNodeBase>("XDriverPointerItemNode");
    //for Driver selection
    XPython::bind.export_xvaluenode<XItemNode<XDriverList, XDriver>,
            shared_ptr<XDriver>, XPointerItemNode<XDriverList>,
            Transaction &, shared_ptr<XDriverList> &, bool>("XDriverItemNode");
//    XPython::bind.export_xvaluenode<XItemNode<XDriverList, XMagnetPS, XDMM, XQDPPMS>,
//            shared_ptr<XDriver>, XPointerItemNode<XDriverList>,
//            Transaction &, shared_ptr<XDriverList> &, bool>("MagnetPSLikeItemNode");
    XPython::bind.export_xnode<XMeasure, XNode>();
    XPython::bind.export_xnode<XPrimaryDriver, XDriver>();
    XPython::bind.export_xnode<XPrimaryDriverWithThread, XPrimaryDriver>();


    py::class_<XPrimaryDriver::RawData, shared_ptr<XPrimaryDriver::RawData>>(m, "RawData")
        .def(py::init())
        .def("push_int16", [](shared_ptr<XPrimaryDriver::RawData> &self, int16_t x){self->push(x);})
        .def("push_uint16", [](shared_ptr<XPrimaryDriver::RawData> &self, uint16_t x){self->push(x);})
        .def("push_int32", [](shared_ptr<XPrimaryDriver::RawData> &self, int32_t x){self->push(x);})
        .def("push_uint32", [](shared_ptr<XPrimaryDriver::RawData> &self, uint32_t x){self->push(x);})
        .def("push_int64", [](shared_ptr<XPrimaryDriver::RawData> &self, int64_t x){self->push(x);})
        .def("push_uint64", [](shared_ptr<XPrimaryDriver::RawData> &self, uint64_t x){self->push(x);})
        .def("push_double", [](shared_ptr<XPrimaryDriver::RawData> &self, double x){self->push(x);});
    py::class_<XPrimaryDriver::RawDataReader>(m, "RawDataReader")
        .def("pop_int16", [](XPrimaryDriver::RawDataReader &self){return self.pop<int16_t>();})
        .def("pop_uint16", [](XPrimaryDriver::RawDataReader &self){return self.pop<uint16_t>();})
        .def("pop_int32", [](XPrimaryDriver::RawDataReader &self){return self.pop<int32_t>();})
        .def("pop_uint32", [](XPrimaryDriver::RawDataReader &self){return self.pop<uint32_t>();})
        .def("pop_int64", [](XPrimaryDriver::RawDataReader &self){return self.pop<int64_t>();})
        .def("pop_uint64", [](XPrimaryDriver::RawDataReader &self){return self.pop<uint64_t>();})
        .def("pop_double", [](XPrimaryDriver::RawDataReader &self){return self.pop<double>();});


    XPython::bind.export_xnode<XSecondaryDriver, XDriver>("XSecondaryDriver");
//    XPython::bind.export_xnode<XPythonDriver<XSecondaryDriver>, XSecondaryDriver>();
    {   auto [node, payload] = XPython::bind.export_xpythondriver
        <XPythonSecondaryDriver, XSecondaryDriver, XPythonSecondaryDriverHelper>("XPythonSecondaryDriver");
        (*node)
            .def("visualize", &XPythonSecondaryDriver::visualize)
//            .def("analyze", &XPythonSecondaryDriver::analyze)
            .def("checkDependency", &XPythonSecondaryDriver::checkDependency)
            .def("requestAnalysis", [](shared_ptr<XPythonSecondaryDriver> &self){
                self->requestAnalysis();})
            .def("connect", [](shared_ptr<XPythonSecondaryDriver> &self,
                 const shared_ptr<XPointerItemNode<XDriverList> > &selecter){self->connect(selecter);});
        (*payload);
    }

    //Exceptions
    py::register_exception<XDriver::XRecordError>(m, "KAMERecordError", PyExc_RuntimeError);
    py::register_exception<XDriver::XSkippedRecordError>(m, "KAMESkippedRecordError", PyExc_RuntimeError);
    py::register_exception<XDriver::XBufferUnderflowRecordError>(m, "KAMEBufferUnderflowRecordError", PyExc_RuntimeError);
}
void
KAMEPyBind::export_embedded_module_interface(pybind11::module_& m) {
    {   auto [node, payload] = XPython::bind.export_xnode<XInterface, XNode>();
        (*node)
            .def("__enter__", [](shared_ptr<XInterface> &self){
                py::gil_scoped_release pyguard;
                self->lock();})
            .def("__exit__", [](shared_ptr<XInterface> &self, pybind11::args){self->unlock();});
    }

    py::register_exception<XInterface::XInterfaceError>(m, "KAMEInterfaceError", PyExc_RuntimeError);
    py::register_exception<XInterface::XConvError>(m, "KAMEInterfaceConvError", PyExc_RuntimeError);
    py::register_exception<XInterface::XCommError>(m, "KAMEInterfaceCommError", PyExc_RuntimeError);
    py::register_exception<XInterface::XOpenInterfaceError>(m, "KAMEInterfaceOpenError", PyExc_RuntimeError);
    py::register_exception<XInterface::XUnsupportedFeatureError>(m, "KAMEInterfaceUnsupportedFeatureError", PyExc_RuntimeError);
}
void
KAMEPyBind::export_embedded_module_xqcon(pybind11::module_& m) {
    //QWidget
    py::class_<QWidget>(m, "QWidget")
        .def("objectName", [](const QWidget *self)->std::string{return self->objectName().toStdString();})
        .def("children", [](const QWidget *self)->std::vector<QWidget*>{
            if( !isMainThread())
                throw std::runtime_error("Be called from main thread.");
            std::vector<QWidget*> list;
            for(auto &&x: self->children())
                if(auto y = dynamic_cast<QWidget*>(x))
                    list.push_back(y);
            return list;
        })
        .def("findChildWidget", [](const QWidget *self, const std::string &name)->QWidget*{
            if( !isMainThread())
                throw std::runtime_error("Be called from main thread.");
            return self->findChild<QWidget*>(name.c_str());
        }, py::return_value_policy::reference);


    //XQ**Connector
    py::class_<XQConnectorHolder_, qshared_ptr<XQConnectorHolder_>>(m, "XQConnector")
        .def_static("connectedNode", [](const QWidget *item)->shared_ptr<XNode>{
            return XQConnector::connectedNode(item);
        })
        .def("__repr__", [](qshared_ptr<XQConnectorHolder_> &self)->std::string{
            return formatString("<xqconnector[%s] @%p>", typeid(self).name(), &*self);
        })
        .def("isAlive", [](qshared_ptr<XQConnectorHolder_> &self)->bool{
            return self->isAlive();
        });
    export_xqcon<XQButtonConnector, XTouchableNode, QAbstractButton>();
    export_xqcon<XQLineEditConnector, XValueNodeBase, QLineEdit>();
    export_xqcon<XQTextEditConnector, XValueNodeBase, QTextEdit>();
    export_xqcon<XQSpinBoxConnector, XIntNode, QSpinBox>();
    export_xqcon<XQSpinBoxUnsignedConnector, XUIntNode, QSpinBox>();
    export_xqcon<XQDoubleSpinBoxConnector, XDoubleNode, QDoubleSpinBox>();
    export_xqcon<XQLabelConnector, XValueNodeBase, QLabel>();
    export_xqcon<XQLCDNumberConnector, XDoubleNode, QLCDNumber>();
    export_xqcon<XQLedConnector, XBoolNode, QPushButton>();
    export_xqcon<XQToggleButtonConnector, XBoolNode, QAbstractButton>();
    export_xqcon<XQListWidgetConnector, XItemNodeBase, QListWidget, const Snapshot &>();
    export_xqcon<XQComboBoxConnector, XItemNodeBase, QComboBox, const Snapshot &>();
    export_xqcon<XColorConnector, XHexNode, QPushButton>();

}

