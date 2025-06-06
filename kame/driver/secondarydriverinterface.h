/***************************************************************************
        Copyright (C) 2002-2025 Kentaro Kitagawa
		                   kitag@issp.u-tokyo.ac.jp
		
		This program is free software; you can redistribute it and/or
		modify it under the terms of the GNU Library General Public
		License as published by the Free Software Foundation; either
		version 2 of the License, or (at your option) any later version.
		
		You should have received a copy of the GNU Library General 
		Public License and a list of authors along with this program; 
		see the files COPYING and AUTHORS.
***************************************************************************/
#ifndef SECONDARYDRIVERINTERFACE_H_
#define SECONDARYDRIVERINTERFACE_H_

#ifdef USE_PYBIND11
    #include <pybind11/pybind11.h>
#endif

#include "secondarydriver.h"

template <class T>
XSecondaryDriverInterface<T>::XSecondaryDriverInterface(const char *name, bool runtime,
	Transaction &tr_meas, const shared_ptr<XMeasure> &meas) :
    T(name, runtime, ref(tr_meas), meas),
    m_drivers(meas->drivers()) {
}

template <class T>
void
XSecondaryDriverInterface<T>::requestAnalysis() {
	Snapshot shot( *this);
    onConnectedRecorded(shot, this);
}
template <class T>
void
XSecondaryDriverInterface<T>::onConnectedRecorded(const Snapshot &shot_emitter, XDriver *driver) {
	Snapshot shot_all_drivers( *m_drivers.lock());
	if( !shot_all_drivers.isUpperOf( *this))
		return;
	Snapshot shot_this( *this, shot_all_drivers);
	Transaction tr(shot_this);
	bool firsttime = true;
	for(;;) {
		if( !firsttime) {
			try {
				shot_all_drivers = tr.newTransactionUsingSnapshotFor( *m_drivers.lock());
				shot_this = tr;
			}
			catch (typename T::NodeNotFoundError &) {
				return; //has been freed from the list.
			}
		}
		firsttime = false;
		if( !shot_all_drivers.isUpperOf( *driver))
			return; //driver has been freed from the list.

		if(driver != this) {
		//checking if emitter has already connected unless self-emitted.
			bool found = false;
			for(auto it = shot_this[ *this].m_connections.begin(); it != shot_this[ *this].m_connections.end(); ++it) {
				if((shared_ptr<XNode>(shot_this[ *it->m_selecter]).get() == driver) &&
                    (shot_emitter[ *driver].time().isSet())) {
					found = true;
					break;
				}
			}
			if( !found)
				return;
		}
		//checking if the selecters point to existing drivers.
		for(auto it = shot_this[ *this].m_connections.begin();
			it != shot_this[ *this].m_connections.end(); ++it) {
			shared_ptr<XNode> node = shot_this[ *it->m_selecter];
			if(node) {
				if( !shot_all_drivers.isUpperOf( *node))
					return;
				if((node.get() != driver) &&
					!shot_all_drivers[ *static_pointer_cast<XDriver>(node)].time())
					return; //Record is invalid.
			}
		}

        try {
            //driver-side dependency check
            if( !checkDependency(tr, shot_emitter, shot_all_drivers, driver))
                return;
        }
#ifdef USE_PYBIND11
        catch (pybind11::error_already_set& e) {
            pybind11::gil_scoped_acquire guard;
            gErrPrint(i18n("Python error: ") + e.what());
            return;
        }
#endif
        catch (std::runtime_error &e) {
            gErrPrint(std::string("Python KAME binding error: ") + e.what());
            return;
        }

		bool skipped = false;
		XKameError err;
		XTime time_recorded = shot_emitter[ *driver].time();
		try {
			analyze(tr, shot_emitter, shot_all_drivers, driver);
		}
#ifdef USE_PYBIND11
        catch (pybind11::error_already_set& e) {
            pybind11::gil_scoped_acquire guard;
            if(e.matches(PyExc_InterruptedError)) {
                skipped = true;
                err = typename T::XSkippedRecordError("", __FILE__, __LINE__);
            }
            else if(e.matches(PyExc_ValueError)) {
                time_recorded = XTime(); //record is invalid
                err = typename T::XRecordError(e.what(), __FILE__, __LINE__);
            }
            else {
                gErrPrint(i18n("Python error: ") + e.what());
                return;
            }
        }
#endif
//        catch (std::runtime_error &e) {
//            gErrPrint(std::string("Python KAME binding error: ") + e.what());
//            return;
//        }
        catch (typename T::XSkippedRecordError& e) {
			skipped = true;
			err = e;
		}
		catch (typename T::XRecordError& e) {
			time_recorded = XTime(); //record is invalid
			err = e;
		}
		if( !skipped)
			this->record(tr, shot_emitter[ *driver].timeAwared(), time_recorded);
		if(tr.commit()) {
			if(err.msg().length())
				err.print(this->getLabel() + ": ");
            try {
                this->visualize(tr);
            }
#ifdef USE_PYBIND11
            catch (pybind11::error_already_set& e) {
                pybind11::gil_scoped_acquire guard;
                gErrPrint(i18n("Python error: ") + e.what());
            }
#endif
            catch (std::runtime_error &e) {
                gErrPrint(std::string("Python KAME binding error: ") + e.what());
            }
            break;
		}
	}
}
template <class T>
void
XSecondaryDriverInterface<T>::connect(const shared_ptr<XPointerItemNode<XDriverList> > &selecter) {
    this->iterate_commit([=](Transaction &tr){
    	typename Payload::Connection con;
		con.m_selecter = selecter;
		tr[ *this].m_connections.push_back(con);
    });

    selecter->iterate_commit([=](Transaction &tr){
		if(m_lsnOnItemChanged)
			tr[ *selecter].onValueChanged().connect(m_lsnOnItemChanged);
		else
			m_lsnOnItemChanged = tr[ *selecter].onValueChanged().connectWeakly(this->shared_from_this(),
				&XSecondaryDriverInterface<T>::onItemChanged);
    });
}
template <class T>
void
XSecondaryDriverInterface<T>::onItemChanged(const Snapshot &shot, XValueNodeBase *node) {
    auto *item = static_cast<XPointerItemNode<XDriverList>*>(node);
    shared_ptr<XNode> nd = shot[ *item];
    auto driver = static_pointer_cast<XDriver>(nd);

    shared_ptr<Listener> lsnonrecord;
	if(driver) {
        driver->iterate_commit([=, &lsnonrecord](Transaction &tr){
			lsnonrecord = tr[ *driver].onRecord().connectWeakly(
				this->shared_from_this(), &XSecondaryDriverInterface<T>::onConnectedRecorded);
        });
	}
    this->iterate_commit([=](Transaction &tr){
		auto it = std::find(tr[ *this].m_connections.begin(), tr[ *this].m_connections.end(), item);
		it->m_lsnOnRecord = lsnonrecord;
    });
}

#endif /*SECONDARYDRIVERINTERFACE_H_*/
