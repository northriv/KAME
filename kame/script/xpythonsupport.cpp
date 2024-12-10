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
#include <QWidget>
#include <math.h>
#include <iostream>

//
#define XPYTHONSUPPORT_PY ":/script/xpythonsupport.py" //in the qrc.
#define XPYTHONEXT_TEST_PY ":/script/pytestdriver.py" //in the qrc.

XPython::XPython(const char *name, bool runtime, const shared_ptr<XMeasure> &measure)
    : XScriptingThreadList(name, runtime, measure) {
}
XPython::~XPython() {
}

void XPython::mainthread_callback(py::object *scrthread, py::object *func, py::object *ret, py::object *status) {
    if( !func) {
        //for the first time, empty callback is issued.
        *status = py::cast(false);
    }
    else {
        pybind11::gil_scoped_acquire guard;
        try {
            py::object tls = py::eval("TLS");
            auto setattr = tls.attr("__setattr__");
            setattr("xscrthread", scrthread);
            setattr("logfile", pybind11::none());
            *ret = py::reinterpret_borrow<py::function>( *func)();
            *status = py::cast(false);
        }
        catch (py::error_already_set& e) {
            std::cerr << "Python error.\n" << e.what() << "\n";
            *status = py::cast(e.what());
        }
        catch (...) {
            std::cerr << "Python unknown error.\n" << "\n";
            *status = py::cast("Python unknown error.\n");
        }
    }
    XScopedLock<XCondition> lock(m_mainthread_cb_cond);
    m_mainthread_cb_cond.signal();
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
        m_mainthread_cb_lsn = m_mainthread_cb_tlk.connectWeakly(
            shared_from_this(), &XPython::mainthread_callback, Listener::FLAG_MAIN_THREAD_CALL);
        //Wait for main event loop, using dry run.
        {
            py::object status;
            status = py::cast(true);
            m_mainthread_cb_tlk.talk(nullptr, nullptr, nullptr, &status);
            XScopedLock<XCondition> lock(m_mainthread_cb_cond);
            while(status.is(py::cast(true)))
                m_mainthread_cb_cond.wait();
        }

        py::scoped_interpreter guard{}; // start the interpreter and keep it alive

        auto kame_module = py::module_::import("kame");
//        bind.s_kame_module = kame_module; //not needed.

        shared_ptr<XMeasure> measure = m_measure.lock();
        assert(measure);
        XString name = measure->getName();
        name[0] = toupper(name[0]);
        kame_module.def("Root", [=]()->shared_ptr<XNode>{return measure;});
        kame_module.def("Measurement", [=]()->shared_ptr<XNode>{return measure;});
        kame_module.def("PyInfoForNodeBrowser", [=]()->shared_ptr<XStringNode>{return measure->pyInfoForNodeBrowser();});
        kame_module.def("LastPointedByNodeBrowser", [=]()->py::object {return bind.cast_to_pyobject(measure->lastPointedByNodeBrowser());});
        kame_module.def("my_defout", [=](shared_ptr<XNode> scrthread, const std::string &str){this->my_defout(scrthread, str);});
        kame_module.def("my_defin", [=](shared_ptr<XNode> scrthread)->std::string{return this->my_defin(scrthread);});
        kame_module.def("is_main_terminated", [=](){return this->m_thread->isTerminated();});
        kame_module.def("XScriptingThreads", [=]()->shared_ptr<XListNodeBase>{return dynamic_pointer_cast<XListNodeBase>(this->shared_from_this());});
        kame_module.def("MainWindow", [=]()->QWidget*{return g_pFrmMain;}, py::return_value_policy::reference);
#ifdef PYBIND11_NO_ASSERT_GIL_HELD_INCREF_DECREF
        kame_module.def("kame_mainthread", [=](py::object closure)->py::object{
            py::object ret, status;
            status = py::cast(true);
            py::object scrthread = py::eval("TLS.xscrthread");
            pybind11::gil_scoped_release guard;
            m_mainthread_cb_tlk.talk( &scrthread, &closure, &ret, &status);
            XScopedLock<XCondition> lock(m_mainthread_cb_cond);
            while(status.is(py::cast(true)))
                m_mainthread_cb_cond.wait();
            if( !status.is(py::cast(false))) {
                pybind11::gil_scoped_acquire guard;
                py::set_error(PyExc_RuntimeError, py::cast<std::string>(status).c_str());
                throw py::error_already_set();
            }
            return ret;
        });
#endif

        for(auto &filename: {XPYTHONEXT_TEST_PY, XPYTHONSUPPORT_PY}) {
            QFile scriptfile(filename);
            if( !scriptfile.open(QIODevice::ReadOnly | QIODevice::Text)) {
                gErrPrint("No KAME python support file installed.");
                return NULL;
            }
            fprintf(stderr, "Loading python scripting support from %s.\n", filename);
            char data[65536];
            QDataStream( &scriptfile).readRawData(data, sizeof(data));
//            py::print("Hello, World!"); // use the Python API
            for(;;) {
                try {
                    py::object main_scope = py::module_::import("__main__").attr("__dict__");
                    py::exec(data, main_scope);
                }
                catch (py::error_already_set& e) {
                    std::cerr << "Python error.\n" << e.what() << "\n";
                }
                catch (...) {
                    std::cerr << "Python unknown error.\n" << "\n";
                }
                if(terminated || (std::string(filename) != XPYTHONSUPPORT_PY))
                    break;
                //support routine may exit accidentally. Retries.
                msecsleep(500);
            }
        }
//        stl_nodeCreating->reset();
        m_mainthread_cb_lsn.reset();
    }

    fprintf(stderr, "python fin");
    fprintf(stderr, "ished\n");
    return NULL;
}
