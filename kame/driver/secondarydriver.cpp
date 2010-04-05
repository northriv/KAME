/***************************************************************************
		Copyright (C) 2002-2010 Kentaro Kitagawa
		                   kitag@issp.u-tokyo.ac.jp
		
		This program is free software; you can redistribute it and/or
		modify it under the terms of the GNU Library General Public
		License as published by the Free Software Foundation; either
		version 2 of the License, or (at your option) any later version.
		
		You should have received a copy of the GNU Library General 
		Public License and a list of authors along with this program; 
		see the files COPYING and AUTHORS.
***************************************************************************/
#include "secondarydriver.h"

XSecondaryDriver::XSecondaryDriver(const char *name, bool runtime,
	Transaction &tr_meas, const shared_ptr<XMeasure> &meas) :
    XDriver(name, runtime, ref(tr_meas), meas),
    m_drivers(meas->drivers()) {
}
XSecondaryDriver::~XSecondaryDriver() {
}

void
XSecondaryDriver::requestAnalysis() {
	Snapshot shot( *this);
    onConnectedRecorded(shot, this);
}
void
XSecondaryDriver::onConnectedRecorded(const Snapshot &shot_emitter, XDriver *driver) {
	for(;;) {
		Snapshot shot_others( *m_drivers.lock());
		if( !shot_others.isUpperOf( *this))
			return;
		if( !shot_others.isUpperOf( *driver))
			return;
		Transaction tr( *this, shot_others);
		Snapshot &shot_this(tr);

		if(driver != this) {
		//checking if emitter has already connected unless self-emitted.
			bool found = false;
			for(Payload::ConnectionList::const_iterator it = shot_this[ *this].m_connections.begin();
				it != shot_this[ *this].m_connections.end(); ++it) {
				if((shared_ptr<XNode>(shot_this[ *it->m_selecter]).get() == driver) &&
					(shot_emitter[ *driver].time())) {
					found = true;
					break;
				}
			}
			if( !found)
				return;
		}
		//checking if the selecters point to existing drivers.
		for(Payload::ConnectionList::const_iterator it = shot_this[ *this].m_connections.begin();
			it != shot_this[ *this].m_connections.end(); ++it) {
			shared_ptr<XNode> node = shot_this[ *it->m_selecter];
			if(node) {
				if( !shot_others.isUpperOf( *node))
					return;
				if((node.get() != driver) &&
					!shot_others[ *static_pointer_cast<XDriver>(node)].time())
					return; //Record is invalid.
			}
		}

		//driver-side dependency check
		if( !checkDependency(tr, shot_emitter, shot_others, driver))
			return;

		bool skipped = false;
		XTime time_recorded = shot_emitter[ *driver].time();
		try {
			analyze(tr, shot_emitter, shot_others, driver);
		}
		catch (XSkippedRecordError& e) {
			skipped = true;
			if(e.msg().length())
				e.print(getLabel() + ": " + i18n("Skipped, because "));
		}
		catch (XRecordError& e) {
			time_recorded = XTime(); //record is invalid
			e.print(getLabel() + ": " + i18n("Record Error, because "));
		}
		if( !skipped)
			record(tr, shot_emitter[ *driver].timeAwared(), time_recorded);
		if(tr.commit()) {
			visualize(tr);
			break;
		}
	}
}
void
XSecondaryDriver::connect(const shared_ptr<XPointerItemNode<XDriverList> > &selecter) {
    for(Transaction tr( *this);; ++tr) {
    	Payload::Connection con;
		con.m_selecter = selecter;
		tr[ *this].m_connections.push_back(con);

		if(tr.commit())
			break;
	}

    for(Transaction tr( *selecter);; ++tr) {
		if(m_lsnOnItemChanged)
			tr[ *selecter].onValueChanged().connect(m_lsnOnItemChanged);
		else
			m_lsnOnItemChanged = tr[ *selecter].onValueChanged().connectWeakly(shared_from_this(),
				&XSecondaryDriver::onItemChanged);
		if(tr.commit())
			break;
	}
}
void
XSecondaryDriver::onItemChanged(const Snapshot &shot, XValueNodeBase *node) {
    XPointerItemNode<XDriverList> *item =
        static_cast<XPointerItemNode<XDriverList>*>(node);
    shared_ptr<XNode> nd = shot[ *item];
    shared_ptr<XDriver> driver = static_pointer_cast<XDriver>(nd);

    shared_ptr<XListener> lsnonrecord;
	if(driver) {
		for(Transaction tr( *driver);; ++tr) {
			lsnonrecord = tr[ *driver].onRecord().connectWeakly(
					shared_from_this(), &XSecondaryDriver::onConnectedRecorded);
			if(tr.commit())
				break;
		}
	}
    for(Transaction tr( *this);; ++tr) {
		Payload::ConnectionList::iterator it
			= std::find(tr[ *this].m_connections.begin(), tr[ *this].m_connections.end(), item);
		it->m_lsnOnRecord = lsnonrecord;
		if(tr.commit())
			break;
    }
}
