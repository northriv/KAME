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
#ifndef pythondriverH
#define pythondriverH

#ifdef USE_PYBIND11

#include "xpythonsupport.h"
#include "driver.h"
#include <QWidget>
#include <QtUiTools>
#include <QFile>
#include "secondarydriver.h"

//! Base class for python based drivers
template <class T>
class DECLSPEC_KAME XPythonDriver : public T {
public:
    using T::T; //inherits constructors.

    virtual ~XPythonDriver() = default;

    virtual XString getTypename() const override { return m_creation_key;}

	//! Shows all forms belonging to the driver.
    virtual void showForms() override {
        if(m_form) {
            m_form->show();
            m_form->raise();
        }
    }
 
    //! setups a form window using Qt designer .ui file.
    //! \sa showForms(), form().
    void loadUIFile(const std::string &loc) {
        QFile file(loc.c_str());
        file.open(QIODevice::ReadOnly);

        QUiLoader loader;
        m_form.reset(loader.load(&file));//, g_pFrmMain
    }
    QWidget *form() const {return m_form.get();}

    //! registers run-time driver class defined in python, into XDriverList.
    //! \sa XListNodeBase::createByTypename(), XNode::getTypename(), XTypeHolder<>.
    static void exportClass(const std::string &key, pybind11::object cls, const std::string &label) {
        XDriverList::s_types.eraseCreator(key); //erase previous info.
        XDriverList::s_types.insertCreator(key, [key, cls](const char *name, bool runtime,
            std::reference_wrapper<Transaction> tr, const shared_ptr<XMeasure> &meas)->shared_ptr<XNode> {
            pybind11::gil_scoped_acquire guard;
            pybind11::object obj = cls(name, runtime, ref(tr), meas); //createOrphan in python side.
//            auto driver = obj.cast<shared_ptr<XPythonDriver<T>>>(); //casting from python-side class to C++ class fails.
            auto driver = dynamic_pointer_cast<XPythonDriver<T>>(obj.cast<shared_ptr<XNode>>());
//            auto driver = dynamic_pointer_cast<XPythonDriver<T>>( *XPython::stl_nodeCreating); //using temporally value set in "__init__" at export_xnode().
//            XPython::stl_nodeCreating->reset();
            if( !driver)
                throw std::runtime_error("Driver creation failed.");
            driver->m_self = obj; //for persistence of python-side class.
            driver->m_creation_key = key;
            return driver;
        }, label);
    }

    struct DECLSPEC_KAME Payload : public T::Payload {};

protected:
    qshared_ptr<QWidget> m_form;
    XString m_creation_key;
    pybind11::object m_self; //to increase reference counter.
};

class XCharInterface;
template<class tDriver, class tInterface>
class XCharDeviceDriver;
class XPrimaryDriverWithThread;

template<class tInterface = XCharInterface>
struct XPythonCharDeviceDriverWithThread : public XPythonDriver<XCharDeviceDriver<XPrimaryDriverWithThread, tInterface>> {
    using tBaseDriver = XPythonDriver<XCharDeviceDriver<XPrimaryDriverWithThread, tInterface>>;
    using tBaseDriver::XPythonDriver; //inherits constructors.

    ////originally protected, opened for public.
    virtual void analyzeRaw(typename tBaseDriver::RawDataReader &reader, Transaction &tr) override = 0;
    virtual void visualize(const Snapshot &shot) override = 0;

    struct DECLSPEC_KAME Payload : public tBaseDriver::Payload {};
protected:
    virtual void *execute(const atomic<bool> &terminated) = 0;
};
template<class tInterface = XCharInterface>
struct XPythonCharDeviceDriverWithThreadHelper : public XPythonCharDeviceDriverWithThread<tInterface> {
    using XPythonCharDeviceDriverWithThreadHelper::XPythonCharDeviceDriverWithThreadHelper; //inherits constructors.
    using tBaseDriver = XPythonCharDeviceDriverWithThread<tInterface>;

    //! This function will be called when raw data are written.
    //! Implement this function to convert the raw data to the record (Payload).
    //! \sa analyze()
    //! XRecordError will be thrown if data is not propertly formatted.
    virtual void analyzeRaw(typename tBaseDriver::RawDataReader &reader, Transaction &tr) override {
        PYBIND11_OVERRIDE_PURE(
            bool, /* Return type */
            tBaseDriver,      /* Parent class */
            checkDependency,          /* Name of function in C++ (must match Python name) */
            reader, tr      /* Argument(s) */
        );
    }

    //! This function is called after committing XPrimaryDriver::analyzeRaw() or XSecondaryDriver::analyze().
    //! This might be called even if the record is invalid (time() == false).
    virtual void visualize(const Snapshot &shot) override {
        PYBIND11_OVERRIDE_PURE(
            void, /* Return type */
            tBaseDriver,      /* Parent class */
            visualize,          /* Name of function in C++ (must match Python name) */
            shot      /* Argument(s) */
        );
    }

    struct Payload : public tBaseDriver::Payload {};
};

struct DECLSPEC_KAME XPythonSecondaryDriver : public XPythonDriver<XSecondaryDriver> {
    using XPythonDriver<XSecondaryDriver>::XPythonDriver; //inherits constructors.

    ////originally protected, opened for public.

    //! Call this to receive signal/data.
    void connect(const shared_ptr<XPointerItemNode<XDriverList> > &selecter) {XPythonDriver<XSecondaryDriver>::connect(selecter);} //to open for public.
    //! check dependencies and lock all records and analyze
    //! null pointer will be passed to analyze()
    //! emitter is driver itself.
    //! \sa analyze(), checkDependency()
    void requestAnalysis() {XPythonDriver<XSecondaryDriver>::requestAnalysis();}

    virtual bool checkDependency(const Snapshot &shot_this,
        const Snapshot &shot_emitter, const Snapshot &shot_others,
        XDriver *emitter) const override = 0;
    virtual void analyze(Transaction &tr, const Snapshot &shot_emitter, const Snapshot &shot_others,
        XDriver *emitter) override = 0;
    virtual void visualize(const Snapshot &shot) override = 0;

    struct DECLSPEC_KAME Payload : public XPythonDriver<XSecondaryDriver>::Payload {};
};

struct XPythonSecondaryDriverHelper : public XPythonSecondaryDriver {
    using XPythonSecondaryDriver::XPythonSecondaryDriver; //inherits constructors.

    //! Checks if the connected drivers have valid time stamps.
    //! \return true if dependency is resolved.
    //! This function must be reentrant unlike analyze().
    virtual bool checkDependency(const Snapshot &shot_this,
        const Snapshot &shot_emitter, const Snapshot &shot_others,
        XDriver *emitter) const override {
        PYBIND11_OVERRIDE_PURE(
            bool, /* Return type */
            XPythonSecondaryDriver,      /* Parent class */
            checkDependency,          /* Name of function in C++ (must match Python name) */
            shot_this, shot_emitter, shot_others, emitter      /* Argument(s) */
        );
    }

    //! This function is called after committing XPrimaryDriver::analyzeRaw() or XSecondaryDriver::analyze().
    //! This might be called even if the record is invalid (time() == false).
    virtual void visualize(const Snapshot &shot) override {
        PYBIND11_OVERRIDE_PURE(
            void, /* Return type */
            XPythonSecondaryDriver,      /* Parent class */
            visualize,          /* Name of function in C++ (must match Python name) */
            shot      /* Argument(s) */
        );
    }
    //! This function is called when a connected driver emit a signal
    virtual void analyze(Transaction &tr, const Snapshot &shot_emitter, const Snapshot &shot_others,
        XDriver *emitter) override {
        PYBIND11_OVERRIDE_PURE(
            void, /* Return type */
            XPythonSecondaryDriver,      /* Parent class */
            analyze,          /* Name of function in C++ (must match Python name) */
            tr, shot_emitter, shot_others, emitter      /* Argument(s) */
        );
    }


    struct Payload : public XPythonSecondaryDriver::Payload {};
};

template <class N>
std::string
XPython::declare_xnode_downcasters() {
    XPython::s_xnodeDownCasters.insert(std::make_pair(typeid(N).hash_code(), [](const shared_ptr<XNode>&x)->pybind11::object{
        return pybind11::cast(dynamic_pointer_cast<N>(x));
    }));
    XPython::s_payloadDownCasters.insert(std::make_pair(typeid(typename N::Payload).hash_code(), [](const shared_ptr<XNode::Payload>&x)->pybind11::object{
        return pybind11::cast(dynamic_pointer_cast<typename N::Payload>(x));
    }));
    XString name = typeid(N).name();
    int i = name.find('X');
    name = name.substr(i); //squeezes C++ class name.
    return name;
}

template <class N, class Base, class Trampoline, typename...Args>
XPython::classtype_xnode_with_trampoline<N, Base, Trampoline>
XPython::export_xnode_with_trampoline(const char *name_) {
    auto &m = XPython::kame_module();
    auto name = declare_xnode_downcasters<N>();
    if(name_) name = name_; //overrides name given by typeid().name
    auto pynode = std::make_unique<pybind11::class_<N, Base, Trampoline, shared_ptr<N>>>(m, name.c_str());
//for initialization of trampoline codes. N is base of Base(Trampoline class).
    ( *pynode)
//            .def(py::init_alias<const char *, bool, Args&&...>())
//            .def(py::init_alias<const shared_ptr<XNode> &, const char *, bool, Args&&...>())
//            .def(py::init_alias<const shared_ptr<XNode> &, Transaction &, const char *, bool, Args&&...>())
        .def(pybind11::init([](const char *name, bool runtime, Args&&... args){
            auto node = XNode::createOrphan<Trampoline>(name, runtime, std::forward<Args>(args)...);
//            *stl_nodeCreating = node; //to be used inside lambda creation fn of exportClass().
            return node;
        }))
        .def(pybind11::init([](const shared_ptr<XNode> &parent, const char *name, bool runtime, Args&&... args){
            auto node = parent->create<Trampoline>(name, runtime, std::forward<Args>(args)...);
//            *stl_nodeCreating = node;
            return node;
        }))
        .def(pybind11::init([](const shared_ptr<XNode> &parent, Transaction &tr, const char *name, bool runtime, Args&&... args){
            auto node = parent->create<Trampoline>(tr, name, runtime, std::forward<Args>(args)...);
//            *stl_nodeCreating = node;
            return node;
        }));

    pynode->def(pybind11::init([](const shared_ptr<XNode> &x){return dynamic_pointer_cast<N>(x);}));
//! todo inheritance of Payload is awkward.
//    if constexpr( !std::is_same<typename Base::Payload, typename N::Payload>::value) {
    auto pypayload = std::make_unique<pybind11::class_<typename N::Payload, typename Base::Payload, typename Trampoline::Payload>>(m, (name + "::Payload").c_str());
    return {std::move(pynode), std::move(pypayload)};
}

template <class N, class Base, typename...Args>
XPython::classtype_xnode<N, Base>
XPython::export_xnode(const char *name_) {
    auto &m = XPython::kame_module();
    auto name = declare_xnode_downcasters<N>();
    if(name_) name = name_; //overrides name given by typeid().name
    auto pynode = std::make_unique<pybind11::class_<N, Base, shared_ptr<N>>>(m, name.c_str());
    if constexpr(std::is_constructible<N, const char *, bool, Args&&...>::value
        || sizeof...(Args)) { //when Args exists. constructor is assumed to be present.
        //avoids compile error against abstract class creation.
        if constexpr(sizeof...(Args) > 8) //disabled.
        //Debug, For readable typeerror, unpacks and casts.
        ( *pynode)
            .def(pybind11::init([](pybind11::args args){
                auto it = args.begin();
                //!todo isinstance
                std::string name = (it++)->cast<std::string>();
                bool runtime = (it++)->cast<bool>();
                return XNode::createOrphan<N>(name.c_str(), runtime,
                     (it++)->cast<Args>()...);
            }));
        else
        ( *pynode)
            .def(pybind11::init([](const char *name, bool runtime, Args&&... args){
            return XNode::createOrphan<N>(name, runtime, std::forward<Args>(args)...);}))
            .def(pybind11::init([](const shared_ptr<XNode> &parent, const char *name, bool runtime, Args&&... args){
            return parent->create<N>(name, runtime, std::forward<Args>(args)...);}))
            .def(pybind11::init([](const shared_ptr<XNode> &parent, Transaction &tr, const char *name, bool runtime, Args&&... args){
            return parent->create<N>(tr, name, runtime, std::forward<Args>(args)...);}));
    }
    pynode->def(pybind11::init([](const shared_ptr<XNode> &x){return dynamic_pointer_cast<N>(x);}));
//! todo inheritance of Payload is awkward.
//    if constexpr( !std::is_same<typename Base::Payload, typename N::Payload>::value) {
    auto pypayload = std::make_unique<pybind11::class_<typename N::Payload, typename Base::Payload>>(m, (name + "::Payload").c_str());
    return {std::move(pynode), std::move(pypayload)};
}

template <class N, class V, class Base, typename...Args>
XPython::classtype_xnode<N, Base>
XPython::export_xvaluenode(const char *name) {
    constexpr const char *pyv = (std::is_integral<V>::value ? "__int__" :
        (std::is_same<V, bool>::value ? "__bool__" :
        (std::is_floating_point<V>::value ? "__double__" :
        (std::is_convertible<V, std::string>::value ? "__str__" : "get"))));
    auto [pynode, pypayload] = export_xnode<N, Base, Args...>(name);
    (*pynode)
        .def(pyv, [](shared_ptr<N> &self)->V{return ***self;})
        .def("set", [](shared_ptr<N> &self, V x){trans(*self) = x;});
    (*pypayload)
        .def(pyv, [](typename N::Payload &self)->V{ return self;})
        .def("set", [](typename N::Payload &self, V x){self.operator=(x);});
    return {std::move(pynode), std::move(pypayload)};
}

//! Trampoline should be helper class, with PYBIND11_OVERRIDE_PURE, PYBIND11_OVERRIDE...
//! D should open pure virtual functions for public, to be called from python side.
template <class D, class Base, class Trampoline>
XPython::classtype_xnode_with_trampoline<D, Base, Trampoline>
XPython::export_xpythondriver(const char *name) {
    auto [pynode, pypayload] = export_xnode_with_trampoline<D, Base, Trampoline,
            std::reference_wrapper<Transaction>, const shared_ptr<XMeasure>&>(name);
    (*pynode)
        .def_static("exportClass", &D::exportClass)
        .def("form", &D::form, pybind11::return_value_policy::reference_internal)
        .def("loadUIFile", [](shared_ptr<D> &self, const std::string &loc)->QWidget* {
            if( !isMainThread())
                throw std::runtime_error("Be called from main thread.");
            self->loadUIFile(loc);
            return self->form();
        }, pybind11::return_value_policy::reference_internal);
    return {std::move(pynode), std::move(pypayload)};
}

//! For abstract driver classes open to python.
template <class D, class Base>
XPython::classtype_xnode<D, Base>
XPython::export_xdriver(const char *name) {
    auto [pynode, pypayload] = export_xnode<D, Base>(name);
    //exports XItemNode<XDriverList, D> for secondary driver.
    XPython::export_xvaluenode<XItemNode<XDriverList, D>,
            shared_ptr<D>, XPointerItemNode<XDriverList>,
            Transaction &, shared_ptr<XDriverList> &, bool>((std::string(typeid(D).name()) + "ItemNode").c_str());
//!todo py::type
    return {std::move(pynode), std::move(pypayload)};
}

#endif //USE_PYBIND11

//---------------------------------------------------------------------------
#endif
