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
    template <typename... Args>
    XPythonDriver(const char *name, bool runtime,
        Transaction &tr_meas, const shared_ptr<XMeasure> &meas, Args&&... args)
        : T(name, runtime, ref(tr_meas), meas, std::forward<Args>(args)...) {
        m_lsnOnRelease = tr_meas[ *meas->drivers()].onRelease().connectWeakly(
            this->shared_from_this(), &XPythonDriver::onRelease);
    }

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
            if( !driver)
                throw std::runtime_error("Driver creation failed.");
            driver->m_self_creating = obj; //pybind11::cast(driver); //for persistence of python-side class.
            driver->m_creation_key = key;
            return driver;
        }, label);
    }

    struct DECLSPEC_KAME Payload : public T::Payload {
        virtual ~Payload() {
            if( !dict) return;
            pybind11::gil_scoped_acquire guard;
            dict.reset();
        }
        pybind11::object local() {
            pybind11::gil_scoped_acquire guard;
            if( !dict)
                dict = std::make_shared<pybind11::object>(pybind11::dict());
            else
                dict = std::make_shared<pybind11::object>(dict->attr("copy")());
            return *dict;
        }
        shared_ptr<pybind11::object> dict; //GIL is mandatory.
    };

protected:
    pybind11::object m_self_creating; //to increase reference counter.

    qshared_ptr<QWidget> m_form;
    XString m_creation_key;
private:
    void onRelease(const Snapshot &shot, const XListNodeBase::Payload::ReleaseEvent &e) {
        if(e.released != this->shared_from_this())
            return;
        //clears an extra reference counting.
        pybind11::gil_scoped_acquire guard;
        m_self_creating = pybind11::none();
        //now python will free this.
    }
    shared_ptr<Listener> m_lsnOnRelease;
};

class XCharInterface;
template<class tDriver, class tInterface>
class XCharDeviceDriver;
class XPrimaryDriverWithThread;

template<class tInterface = XCharInterface>
struct XPythonCharDeviceDriverWithThread : public XPythonDriver<XCharDeviceDriver<XPrimaryDriverWithThread, tInterface>> {
    using tBaseDriver = XPythonDriver<XCharDeviceDriver<XPrimaryDriverWithThread, tInterface>>;
    using tBaseDriver::tBaseDriver; //inherits constructors.

    ////originally protected, opened for public.
    virtual void analyzeRaw(std::reference_wrapper<typename tBaseDriver::RawDataReader> reader, std::reference_wrapper<Transaction> tr) = 0;
    virtual void visualize(const Snapshot &shot) override = 0;
    //! will call analyzeRaw()
    //! \param rawdata the data being processed.
    //! \param time_awared time when a visible phenomenon started
    //! \param time_recorded usually pass \p XTime::now()
    //! \sa Payload::timeAwared()
    //! \sa Payload::time()
    void finishWritingRaw(const shared_ptr<const typename tBaseDriver::RawData> &rawdata,
        const XTime &time_awared, const XTime &time_recorded) {
        tBaseDriver::finishWritingRaw(rawdata, time_awared, time_recorded);}

    struct DECLSPEC_KAME Payload : public tBaseDriver::Payload {};

    virtual void executeInPython(const std::function<bool()> &is_terminated) = 0;
protected:
    virtual void analyzeRaw(typename tBaseDriver::RawDataReader &reader, Transaction &tr) override {
        analyzeRaw(std::ref(reader), ref(tr)); //pybind11 does not accept reference to non-copyable obj for OVERRIDE func..
    }

    virtual void *execute(const atomic<bool> &terminated) override {
        try {
            executeInPython([&]()->bool{return terminated;});
        }
        catch (pybind11::error_already_set& e) {
            pybind11::gil_scoped_acquire guard;
            gErrPrint(i18n("Python error: ") + e.what());
        }
        catch (std::runtime_error &e) {
            pybind11::gil_scoped_acquire guard;
            gErrPrint(i18n("Python KAME binding error: ") + e.what());
        }
        catch (...) {
            gErrPrint(i18n("Unknown python error."));
        }
        return nullptr;
    }
};
template<class tInterface = XCharInterface>
struct XPythonCharDeviceDriverWithThreadHelper : public XPythonCharDeviceDriverWithThread<tInterface> {
    using tBaseDriver = XPythonCharDeviceDriverWithThread<tInterface>;
    using tBaseDriver::tBaseDriver; //inherits constructors.

    //! This function will be called when raw data are written.
    //! Implement this function to convert the raw data to the record (Payload).
    //! \sa analyze()
    //! XRecordError will be thrown if data is not propertly formatted.
    virtual void analyzeRaw(std::reference_wrapper<typename tBaseDriver::RawDataReader> reader, std::reference_wrapper<Transaction> tr) override {
        PYBIND11_OVERRIDE_PURE(
            void, /* Return type */
            tBaseDriver,      /* Parent class */
            analyzeRaw,          /* Name of function in C++ (must match Python name) */
            reader, tr      /* Argument(s) */
        );
    }

    //! This function is called after committing XPrimaryDriver::analyzeRaw() or XSecondaryDriver::analyze().
    //! This might be called even if the record is invalid (time() == false).
    virtual void visualize(const Snapshot &shot) override {
        PYBIND11_OVERRIDE_PURE(
            void, tBaseDriver, visualize, shot);
    }

    virtual void executeInPython(const std::function<bool()> &is_terminated) override {
        //include pybind11/functional.h.
        PYBIND11_OVERRIDE_PURE_NAME(
            void, tBaseDriver, "execute", executeInPython, is_terminated);
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
    virtual void analyze(std::reference_wrapper<Transaction> tr, const Snapshot &shot_emitter, const Snapshot &shot_others,
        XDriver *emitter) = 0;
    virtual void visualize(const Snapshot &shot) override = 0;

    struct DECLSPEC_KAME Payload : public XPythonDriver<XSecondaryDriver>::Payload {};
protected:
    virtual void analyze(Transaction &tr, const Snapshot &shot_emitter, const Snapshot &shot_others,
                         XDriver *emitter) override {
        analyze(ref(tr), shot_emitter, shot_others, emitter); //pybind11 does not accept reference to non-copyable obj for OVERRIDE func..
    }

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
            void, XPythonSecondaryDriver, visualize, shot);
    }
    //! This function is called when a connected driver emit a signal
    virtual void analyze(std::reference_wrapper<Transaction> tr, const Snapshot &shot_emitter, const Snapshot &shot_others,
        XDriver *emitter) override {
        PYBIND11_OVERRIDE_PURE(
            void, XPythonSecondaryDriver, analyze, tr, shot_emitter, shot_others, emitter);
    }


    struct Payload : public XPythonSecondaryDriver::Payload {};
};

template <class N>
std::string
KAMEPyBind::declare_xnode_downcasters() {
    m_xnodeDownCasters.insert(std::make_pair(typeid(N).hash_code(),
        std::make_pair(m_xnodeDownCasters.size(),
            [](const shared_ptr<XNode>&x)->pybind11::object{
                return pybind11::cast(dynamic_pointer_cast<N>(x));
            })
    ));
    m_payloadDownCasters.insert(std::make_pair(typeid(typename N::Payload).hash_code(),
        std::make_pair(m_payloadDownCasters.size(),
            [](XNode::Payload *x)->pybind11::object{
                return pybind11::cast(dynamic_cast<typename N::Payload*>(x));
            })
    ));
    XString name = typeid(N).name();
    int i = name.find('X');
    name = name.substr(i); //squeezes C++ class name.
    return name;
}

template <class N, class Base, class Trampoline, typename...Args>
KAMEPyBind::classtype_xnode_with_trampoline<N, Base, Trampoline>
KAMEPyBind::export_xnode_with_trampoline(const char *name_) {
    auto &m = kame_module();
    declare_xnode_downcasters<Trampoline>(); //not working
//    declare_xnode_downcasters<Base>(); //for XPythonDriver<>
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
            return node;
        }))
        .def(pybind11::init([](const shared_ptr<XNode> &parent, const char *name, bool runtime, Args&&... args){
            auto node = parent->create<Trampoline>(name, runtime, std::forward<Args>(args)...);
            return node;
        }))
        .def(pybind11::init([](const shared_ptr<XNode> &parent, Transaction &tr, const char *name, bool runtime, Args&&... args){
            auto node = parent->create<Trampoline>(tr, name, runtime, std::forward<Args>(args)...);
            return node;
        }));

    pynode->def(pybind11::init([](const shared_ptr<XNode> &x){return dynamic_pointer_cast<N>(x);}));
//! todo inheritance of Payload is awkward.
//    if constexpr( !std::is_same<typename Base::Payload, typename N::Payload>::value) {
    auto pypayload = std::make_unique<pybind11::class_<typename N::Payload, typename Base::Payload, typename Trampoline::Payload>>
            (m, (name + "_Payload").c_str());
    return {std::move(pynode), std::move(pypayload)};
}

template <class N, class Base, typename...Args>
KAMEPyBind::classtype_xnode<N, Base>
KAMEPyBind::export_xnode(const char *name_) {
    auto &m = kame_module();
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
    if constexpr( !std::is_base_of<typename N::Payload, typename Base::Payload>::value) {
        //N::Payload is defined.
        auto pypayload = std::make_unique<pybind11::class_<typename N::Payload, typename Base::Payload>>(m, (name + "_Payload").c_str());
        return {std::move(pynode), std::move(pypayload)};
    }
    else {
        return std::move(pynode);  //N::Payload is NOT defined.
    }
}

template <class N, class V, class Base, typename...Args>
KAMEPyBind::classtype_xnode<N, Base>
KAMEPyBind::export_xvaluenode(const char *name) {
    constexpr const char *pyv = (std::is_integral<V>::value ? "__int__" :
        (std::is_same<V, bool>::value ? "__bool__" :
        (std::is_floating_point<V>::value ? "__double__" :
        (std::is_convertible<V, std::string>::value ? "__str__" : "get"))));
    if constexpr( !std::is_base_of<typename N::Payload, typename Base::Payload>::value) {
        //N::Payload is defined.
        auto [pynode, pypayload] = export_xnode<N, Base, Args...>(name);
        (*pynode)
            .def(pyv, [](shared_ptr<N> &self)->V{return ***self;})
            .def("set", [](shared_ptr<N> &self, V x){trans(*self) = x;});
        (*pypayload)
            .def(pyv, [](typename N::Payload &self)->V{ return self;})
            .def("set", [](typename N::Payload &self, V x){self.operator=(x);});
        return {std::move(pynode), std::move(pypayload)};
    }
    else {
        auto pynode = export_xnode<N, Base, Args...>(name);
        (*pynode)
            .def(pyv, [](shared_ptr<N> &self)->V{return ***self;})
            .def("set", [](shared_ptr<N> &self, V x){trans(*self) = x;});
        return std::move(pynode);  //N::Payload is NOT defined.
    }
}

//! Trampoline should be helper class, with PYBIND11_OVERRIDE_PURE, PYBIND11_OVERRIDE...
//! D should open pure virtual functions for public, to be called from python side.
template <class D, class Base, class Trampoline>
KAMEPyBind::classtype_xnode_with_trampoline<D, Base, Trampoline>
KAMEPyBind::export_xpythondriver(const char *name) {
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
    (*pypayload)
        .def("local", &D::Payload::local);
    return {std::move(pynode), std::move(pypayload)};
}

//! For abstract driver classes open to python.
template <class D, class Base>
KAMEPyBind::classtype_xnode<D, Base>
KAMEPyBind::export_xdriver(const char *name_) {
    auto [pynode, pypayload] = export_xnode<D, Base>(name_);

    //exports XItemNode<XDriverList, D> for secondary driver.
    XString name = typeid(D).name();
    int i = name.find('X');
    name = name.substr(i); //squeezes C++ class name.
    if(name_) name = name_;
    export_xvaluenode<XItemNode<XDriverList, D>,
            shared_ptr<D>, XPointerItemNode<XDriverList>,
            Transaction &, shared_ptr<XDriverList> &, bool>((name + "ItemNode").c_str());
    return {std::move(pynode), std::move(pypayload)};
}

//! Helper struct to declare xnode-based classes to python side.
//! Classes (NAME), (NAME)_Payload will be defined.
template <class N, class Base>
struct PyXNodeExporter {
    PyXNodeExporter(const char *name = nullptr) {
        pybind11::gil_scoped_acquire guard;
        pycls = XPython::bind.export_xnode<N, Base>(name);
    }

    //! to additionally define methods.
    template <class Fn>
    PyXNodeExporter(Fn fn)
        : PyXNodeExporter() {
        pybind11::gil_scoped_acquire guard;
        if constexpr( !std::is_base_of<typename N::Payload, typename Base::Payload>::value)
            fn( *std::get<0>(pycls), *std::get<1>(pycls));
        else
            fn( *pycls);
    }
    //! to additionally define methods and with preferred typename.
    template <class Fn>
    PyXNodeExporter(const char *name, Fn fn)
        : PyXNodeExporter(name) {
        pybind11::gil_scoped_acquire guard;
        if constexpr( !std::is_base_of<typename N::Payload, typename Base::Payload>::value)
            fn( *std::get<0>(pycls), *std::get<1>(pycls));
        else
            fn( *pycls);
    }
    KAMEPyBind::classtype_xnode<N, Base> pycls;
};

//! Helper struct to declare abstract driver classes to python side.
//! Classes (NAME), (NAME)_Payload, (NAME)ItemNode will be defined.
template <class D, class Base>
struct PyDriverExporter {
    PyDriverExporter(const char *name = nullptr) {
        pybind11::gil_scoped_acquire guard;
        pycls = XPython::bind.export_xdriver<D, Base>(name);
    }
    //! to additionally define methods.
    template <class Fn>
    PyDriverExporter(Fn fn)
        : PyDriverExporter() {
        pybind11::gil_scoped_acquire guard;
        fn( *std::get<0>(pycls), *std::get<1>(pycls));
    }
    //! to additionally define methods and with preferred typename.
    template <class Fn>
    PyDriverExporter(const char *name, Fn fn)
        : PyDriverExporter(name) {
        pybind11::gil_scoped_acquire guard;
        fn( *std::get<0>(pycls), *std::get<1>(pycls));
    }

    KAMEPyBind::classtype_xnode<D, Base> pycls;
};
template <class D, class Base, class Trampoline>
struct PyDriverExporterWithTrampoline {
    PyDriverExporterWithTrampoline(const char *name = nullptr) {
        pybind11::gil_scoped_acquire guard;
        pycls = XPython::bind.export_xpythondriver<D, Base, Trampoline>(name);
    }
    //! to additionally define methods.
    template <class Fn>
    PyDriverExporterWithTrampoline(Fn fn)
        : PyDriverExporterWithTrampoline() {
        pybind11::gil_scoped_acquire guard;
        fn( *std::get<0>(pycls), *std::get<1>(pycls));
    }
    //! to additionally define methods and with preferred typename.
    template <class Fn>
    PyDriverExporterWithTrampoline(const char *name, Fn fn)
        : PyDriverExporterWithTrampoline(name) {
        pybind11::gil_scoped_acquire guard;
        fn( *std::get<0>(pycls), *std::get<1>(pycls));
    }

    KAMEPyBind::classtype_xnode_with_trampoline<D, Base, Trampoline> pycls;
};

#endif //USE_PYBIND11

//---------------------------------------------------------------------------
#endif
