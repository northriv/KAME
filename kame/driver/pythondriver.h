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

#include "xpythonmodule.h"  //include before kame headers
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
//    XPythonDriver(const char *name, bool runtime, Transaction &tr_meas, const shared_ptr<XMeasure> &meas)
//        : T(name, runtime, ref(tr_meas), meas) {}

    virtual ~XPythonDriver() = default;

    virtual XString getTypename() const override { return m_creation_key;}

	//! Shows all forms belonging to the driver.
    virtual void showForms() override {

        if(m_form) {
            m_form->show();
            m_form->raise();
        }
    }
 
    void loadUIFile(const std::string &loc) {
        QFile file(loc.c_str());
        file.open(QIODevice::ReadOnly);

        QUiLoader loader;
        m_form = loader.load(&file, g_pFrmMain);
    }
    QWidget *form() const {return m_form;}

    //! registers run-time driver class defined in python, into XDriverList.
    //! \sa XListNodeBase::createByTypename(), XNode::getTypename(), XTypeHolder<>.
    static void exportClass(const std::string &key, pybind11::object cls, const std::string &label) {
        XDriverList::s_types.eraseCreator(key); //erase previous info.
        XDriverList::s_types.insertCreator(key, [key, cls](const char *name, bool runtime,
            std::reference_wrapper<Transaction> tr, const shared_ptr<XMeasure> &meas)->shared_ptr<XNode> {
            pybind11::gil_scoped_acquire guard;
            pybind11::object obj = cls(name, runtime, ref(tr), meas); //createOrphan in python side.
//            auto driver = obj.cast<shared_ptr<XPythonDriver<T>>>(); //casting from python-side class to C++ class fails.
            auto driver = dynamic_pointer_cast<XPythonDriver<T>>( *XPython::stl_nodeCreating); //using temporally value set in "__init__" at export_xnode().
            XPython::stl_nodeCreating->reset();
            if( !driver)
                throw std::runtime_error("Driver creation failed.");
            driver->m_creation_key = key;
            return driver;
        }, label);
    }

    struct DECLSPEC_KAME Payload : public T::Payload {
    private:
        friend class XPythonDriver;
    };
 
protected:
    QWidget *m_form = nullptr;
    XString m_creation_key;
};

struct DECLSPEC_KAME XPythonSecondaryDriver : public XPythonDriver<XSecondaryDriver> {
    using XPythonDriver<XSecondaryDriver>::XPythonDriver; //inherits constructors.
//    XPythonSecondaryDriver(const char *name, bool runtime, Transaction &tr_meas, const shared_ptr<XMeasure> &meas)
//        : XPythonDriver<XSecondaryDriver>(name, runtime, ref(tr_meas), meas) {}

    //! Call this to receive signal/data.
    void connect(const shared_ptr<XPointerItemNode<XDriverList> > &selecter) {XPythonDriver<XSecondaryDriver>::connect(selecter);} //to open for public.
    //! check dependencies and lock all records and analyze
    //! null pointer will be passed to analyze()
    //! emitter is driver itself.
    //! \sa analyze(), checkDependency()
    void requestAnalysis() {XPythonDriver<XSecondaryDriver>::requestAnalysis();} //to open for public.

    virtual bool checkDependency(const Snapshot &shot_this,
        const Snapshot &shot_emitter, const Snapshot &shot_others,
        XDriver *emitter) const override = 0;//opened for public.
    virtual void analyze(Transaction &tr, const Snapshot &shot_emitter, const Snapshot &shot_others,
        XDriver *emitter) override = 0; //opened for public.
    virtual void visualize(const Snapshot &shot) override = 0;//opened for public.

    struct DECLSPEC_KAME Payload : public XPythonDriver<XSecondaryDriver>::Payload {
    private:
    };
};
struct DECLSPEC_KAME XPythonSecondaryDriverHelper : public XPythonSecondaryDriver {
    using XPythonSecondaryDriver::XPythonSecondaryDriver; //inherits constructors.
//    XPythonSecondaryDriverHelper(const char *name, bool runtime, Transaction &tr_meas, const shared_ptr<XMeasure> &meas)
//        : XPythonSecondaryDriver(name, runtime, ref(tr_meas), meas) {}

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


    struct DECLSPEC_KAME Payload : public XPythonSecondaryDriver::Payload {
    private:
    };

};

#endif //USE_PYBIND11

//---------------------------------------------------------------------------
#endif
