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
#include "graphmathtool.h"
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
    std::map<size_t, decltype(m_payloadDownCasters.begin()->second.second)> cand;
    for(auto &c: m_payloadDownCasters) {
        auto x = (c.second.second)(y);
        if(x.cast<XNode::Payload*>())
            cand.emplace(c.second.first, c.second.second);
    }
    if(cand.size()) {
//        //caches the best result.
//        m_payloadDownCasters.insert(std::make_pair(typeid(y).hash_code(),
//            std::make_pair(m_payloadDownCasters.size(), cand.rbegin()->second)
//        ));
        return cand.rbegin()->second(y); //the oldest choice.
    }
    return py::cast(y); //end up with XNode::Payload*
}
py::object KAMEPyBind::cast_to_pyobject(shared_ptr<XNode> y) {
    if( !y) return py::none();
    auto it = m_xnodeDownCasters.find(typeid(y).hash_code());
    if(it != m_xnodeDownCasters.end()) {
        return (it->second.second)(y);
    }
//    //manages to use its super class.
//    if(auto x = dynamic_pointer_cast<XValueNodeBase>(y)) {
//        if(auto x = dynamic_pointer_cast<XIntNode>(y))
//            return py::cast(x);
//        if(auto x = dynamic_pointer_cast<XUIntNode>(y))
//            return py::cast(x);
//        if(auto x = dynamic_pointer_cast<XLongNode>(y))
//            return py::cast(x);
//        if(auto x = dynamic_pointer_cast<XULongNode>(y))
//            return py::cast(x);
//        if(auto x = dynamic_pointer_cast<XHexNode>(y))
//            return py::cast(x);
//        if(auto x = dynamic_pointer_cast<XBoolNode>(y))
//            return py::cast(x);
//        if(auto x = dynamic_pointer_cast<XDoubleNode>(y))
//            return py::cast(x);
//        if(auto x = dynamic_pointer_cast<XStringNode>(y))
//            return py::cast(x);
//        if(auto x = dynamic_pointer_cast<XItemNodeBase>(y)) {
//            if(auto z = dynamic_pointer_cast<XComboNode>(x))
//                return py::cast(z);
//            return py::cast(x);
//        }
//        return py::cast(x);
//    }
//    if(auto x = dynamic_pointer_cast<XListNodeBase>(y))
//        return py::cast(x);
//    if(auto x = dynamic_pointer_cast<XTouchableNode>(y))
//        return py::cast(x);

    //manages to use its downmost base class.
    std::map<size_t, decltype(m_xnodeDownCasters.begin()->second.second)> cand;
    for(auto &c: m_xnodeDownCasters) {
        auto x = (c.second.second)(y);
        if(x.cast<shared_ptr<XNode>>())
            cand.emplace(c.second.first, c.second.second);
    }
    if(cand.size()) {
//TODO ??? NoneType thrown in support code. std::function breaks?
//        //caches the best result.
//        m_xnodeDownCasters.emplace(typeid(y).hash_code(),
//            std::make_pair(m_xnodeDownCasters.size() + 10000u, cand.rbegin()->second)
//        );
        return cand.rbegin()->second(y); //the oldest choice.
    }

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

    //XValueNodeBase and cousins.
    {   auto [node, payload] = XPython::bind.export_xnode<XValueNodeBase, XNode>();
        (*node)
        .def("__str__", [](shared_ptr<XValueNodeBase> &self)->std::string{return Snapshot( *self)[*self].to_str();})
        .def("set", [](shared_ptr<XValueNodeBase> &self, const std::string &s){trans(*self).str(s);});}
    XPython::bind.export_xvaluenode<XIntNode, int, XValueNodeBase>("XIntNode");
    XPython::bind.export_xvaluenode<XUIntNode, unsigned int, XValueNodeBase>("XUIntNode");
    XPython::bind.export_xvaluenode<XLongNode, long, XValueNodeBase>("XLongNode");
    XPython::bind.export_xvaluenode<XULongNode, unsigned long, XValueNodeBase>("XULongNode");
    XPython::bind.export_xvaluenode<XHexNode, unsigned long, XValueNodeBase>("XHexNode");
    XPython::bind.export_xvaluenode<XBoolNode, bool, XValueNodeBase>("XBoolNode");
    XPython::bind.export_xvaluenode<XDoubleNode, double, XValueNodeBase>("XDoubleNode");
    XPython::bind.export_xvaluenode<XStringNode, std::string, XValueNodeBase>("XStringNode");
    {   auto [node, payload] = XPython::bind.export_xnode<XItemNodeBase, XValueNodeBase>();
        (*node)
        .def("itemStrings", &XItemNodeBase::itemStrings)
        .def("autoSetAny", &XItemNodeBase::autoSetAny);}
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

    {   auto [node, payload] = XPython::bind.export_xnode<XListNodeBase, XNode>();
        (*node)
        .def("createByTypename", &XListNodeBase::createByTypename);}
    {   auto [node, payload] = XPython::bind.export_xnode<XTouchableNode, XNode>();
        (*node)
        .def("touch", [](shared_ptr<XTouchableNode> &self){trans(*self).touch();});}

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

struct PyFunc1DMathTool {
    ~PyFunc1DMathTool() {
        if( !pyfunc) return;
        pybind11::gil_scoped_acquire guard;
        pyfunc.reset();
    }
    using cv_iterator = std::vector<XGraph::VFloat>::const_iterator;
    using ret_type = typename XGraph1DMathToolX<PyFunc1DMathTool, false>::ret_type;

    ret_type
    operator()(cv_iterator xbegin, cv_iterator xend, cv_iterator ybegin, cv_iterator yend){
        pybind11::gil_scoped_acquire guard;
        auto pf = pyfunc;
        if( !pf)
            return {};
        try {
            using namespace Eigen;
            auto xvec = Map<const VectorXd, 0>( &*xbegin, xend - xbegin);
            auto yvec = Map<const VectorXd, 0>( &*ybegin, yend - ybegin);
            return py::cast<ret_type>((*pf)(Ref<const VectorXd>(xvec), Ref<const VectorXd>(yvec)));
        }
        catch (pybind11::error_already_set& e) {
            gErrPrint(i18n("Python error: ") + e.what());
        }
        catch (std::runtime_error &e) {
            gErrPrint(i18n("Python KAME binding error: ") + e.what());
        }
        catch (...) {
            gErrPrint(i18n("Unknown python error."));
        }
        return {};
    }
    std::shared_ptr<py::object> pyfunc;
};

template <class PyFunc, class MathTool, class MathToolList>
class XPythonGraphMathTool : public MathTool {
public:
    using MathTool::MathTool;
    virtual XString getTypename() const override { return m_creation_key;}

    virtual bool releaseEntries(Transaction &tr) override {
        bool ret = MathTool::releaseEntries(tr);
        if(ret) {
            //clears an extra reference counting.
            pybind11::gil_scoped_acquire guard;
            m_self_creating = pybind11::none();
            //now python will free this.
        }
        return ret;
    }

    //! registers run-time driver class defined in python, into XGraph1DMathToolList.
    //! \sa XListNodeBase::createByTypename(), XNode::getTypename(), XTypeHolder<>.
    static void exportClass(const std::string &key, pybind11::object cls, const std::string &label) {
        MathToolList::s_types.eraseCreator(key); //erase previous info.
        MathToolList::s_types.insertCreator(key, [key, cls](const char *name, bool runtime,
            std::reference_wrapper<Transaction> tr,
            const shared_ptr<XScalarEntryList> &entries, const shared_ptr<XDriver> &driver,
            const shared_ptr<XPlot> &plot, const std::vector<std::string> &entrynames)->shared_ptr<XNode> {
            pybind11::gil_scoped_acquire guard;
            pybind11::object obj = cls(name, runtime, ref(tr), entries, driver, plot, entrynames); //createOrphan in python side.
            auto pytool = dynamic_pointer_cast<XPythonGraphMathTool>
                (obj.cast<shared_ptr<XNode>>());
            if( !driver)
                throw std::runtime_error("Tool creation failed.");
            pytool->m_self_creating = obj; //pybind11::cast(driver); //for persistence of python-side class.
            pytool->m_creation_key = key;
            return pytool;
        }, label);
    }
private:
    pybind11::object m_self_creating; //to increase reference counter.
    XString m_creation_key;
};

template class XPythonGraphMathTool<PyFunc1DMathTool, XGraph1DMathToolX<PyFunc1DMathTool, false>, XGraph1DMathToolList>;

using XPythonGraph1DMathTool = XPythonGraphMathTool<PyFunc1DMathTool, XGraph1DMathToolX<PyFunc1DMathTool, false>, XGraph1DMathToolList>;

struct PyFunc2DMathTool {
    ~PyFunc2DMathTool() {
        if( !pyfunc) return;
        pybind11::gil_scoped_acquire guard;
        pyfunc.reset();
    }

    using ret_type = typename XGraph2DMathToolX<PyFunc2DMathTool, false>::ret_type;

    ret_type
    operator()(const uint32_t *leftupper, unsigned int width,
                      unsigned int stride, unsigned int numlines, double coefficient){
        using namespace Eigen;
        using RMatrixXu32 = Matrix<uint32_t, Dynamic, Dynamic, RowMajor>;
        auto cmatrix = Map<const RMatrixXu32, 0, Stride<Dynamic, 1>>(
            leftupper, numlines, width, Stride<Dynamic, 1>(stride, 1));
        pybind11::gil_scoped_acquire guard;
        auto pf = pyfunc;
        if( !pf)
            return {};
        try {
            return py::cast<ret_type>((*pf)(Ref<const RMatrixXu32>(cmatrix),
                width, stride, numlines, coefficient));
        }
        catch (pybind11::error_already_set& e) {
            gErrPrint(i18n("Python error: ") + e.what());
        }
        catch (std::runtime_error &e) {
            gErrPrint(i18n("Python KAME binding error: ") + e.what());
        }
        catch (...) {
            gErrPrint(i18n("Unknown python error."));
        }
        return {};
    }
    std::shared_ptr<py::object> pyfunc;
};

template class XPythonGraphMathTool<PyFunc2DMathTool, XGraph2DMathToolX<PyFunc2DMathTool, false>, XGraph2DMathToolList>;

using XPythonGraph2DMathTool = XPythonGraphMathTool<PyFunc2DMathTool, XGraph2DMathToolX<PyFunc2DMathTool, false>, XGraph2DMathToolList>;

template <class PyMathTool, class MathTool>
void export_mathtool(const char *name) {
    auto [node, payload] = XPython::bind.export_xnode<PyMathTool, MathTool,
            Transaction&, const shared_ptr<XScalarEntryList> &,
            const shared_ptr<XDriver> &, const shared_ptr<XPlot> &, const std::vector<std::string> &>(name);
    (*node)
        .def_static("exportClass", &PyMathTool::exportClass)
        .def("setFunctor", [](shared_ptr<PyMathTool> &self, py::object f){
            trans( *self).functor.pyfunc = std::make_shared<py::object>(f);
        });
    (*payload)
        .def("setFunctor", [](typename PyMathTool::Payload &self, py::object f){
            self.functor.pyfunc = std::make_shared<py::object>(f);
        });
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
        .def("labels", [](XWaveNGraph::Payload &self){
            std::vector<std::string> list;
            for(auto &x: self.labels())
                list.push_back(x);
            return list;
        })
        .def("setColumn", [](XWaveNGraph::Payload &self, unsigned int n, std::vector<double> &&data, unsigned int prec){
            self.setColumn(n, std::move(data), prec);
        });
    }

    XPython::bind.export_xnode<XGraphMathTool, XNode>();
    XPython::bind.export_xnode<XGraph1DMathTool, XGraphMathTool>();
    XPython::bind.export_xnode<XGraph2DMathTool, XGraphMathTool>();

    export_mathtool<XPythonGraph1DMathTool, XGraph1DMathTool>("XPythonGraph1DMathTool");

    export_mathtool<XPythonGraph2DMathTool, XGraph2DMathTool>("XPythonGraph2DMathTool");
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
    XPython::bind.export_xnode<XScalarEntryList, XListNodeBase>("XScalarEntryList"); //needed to be used as an argument.
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
                py::gil_scoped_release unguard;
                self->lock();})
            .def("__exit__", [](shared_ptr<XInterface> &self, pybind11::args){
                self->unlock();})
            .def("isOpened", &XInterface::isOpened);
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

