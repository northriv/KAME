/***************************************************************************
		Copyright (C) 2002-2009 Kentaro Kitagawa
		                   kitag@issp.u-tokyo.ac.jp

		This program is free software; you can redistribute it and/or
		modify it under the terms of the GNU Library General Public
		License as published by the Free Software Foundation; either
		version 2 of the License, or (at your option) any later version.

		You should have received a copy of the GNU Library General 
		Public License and a list of authors along with this program; 
		see the files COPYING and AUTHORS.
 ***************************************************************************/
#ifndef TRANSACTION_H
#define TRANSACTION_H

//! \todo metamonitor.
//! \todo wrapper.


//! Example 1
//! shared_ptr<Subscriber> ss1 = monitor1->monitorData();
//! sleep(1);
//! if(Snapshot<MonitorA> shot1(monitor1)) { // checking consistency (i.e. requiring at least one transaction).
//! double x = shot1[node1]; //implicit conversion defined in Node1::Passage.
//! double y = shot1[node1]->y(); }
//!
//! Example 2
//! double x = *node1; // for an immediate access, same as (double)(const Node1::Passage&)(*node1)
//!
//! Example 3
//! { Transaction<MonitorA> tr1(monitor1);
//! tr1[node1] = tr1[node1] * 2.0; }
//! 
//! Example 4
//! node1->value(1.0); // for an immediate access.
//! 
//! Example 5
//! //Obsolete, for backward compatibility.
//! monitor1.readLock(); // or use lock(), a snapshot will be copied to TLS.
//! double x = node1->y();
//! monitor1.readUnock(); // or use unlock()
//! 
//! Example 6
//! //Obsolete, for backward compatibility.
//! monitor1.writeLock(); // a transaction will be prepared in TLS.
//! node1->value(1.0);
//! monitor1.writeUnock(); // commit the transaction.

//! Watch point for transactional memeory access.
//! The list of the pointers to data is atomically read/written.
class Monitor {
public:
	Monitor() {}
	virtual ~Monitor() {}

	//! Data holder/accessor.
	struct Passage {
		virtual ~Passage();
	protected:
	private:
	};

	template <class T>
	operator T::Passage&() {return dynamic_cast<T::Passage&>(*m_passage);}

	Passage &resolve(const shared_ptr<Snapshot> &) const;
private:
	typedef std::deque<weak_ptr<Monitor> > SubscriberList;
	atomic_shared_ptr<SubscriberList> m_subscribers;
	atomic_shared_ptr<Passage> m_passage;
};

class Metamonitor : public Monitor {
public:
	struct Packet : public Monitor::Packet {
		virtual shared_ptr<DataMap> dataMap() {return m_dataMap;}
		virtual Packet &resolve(const Monitor &monitor) {
			return dataMap()->find(monitor).second;
		}
		typedef std::deque<Passage> PassageList;
		PassageList m_passages;
		shared_ptr<MonitorList> m_monitorList;
		shared_ptr<DataMap> m_dataMap;
	};
private:
};


//! Take a snapshot for a monitored data set.
class Snapshot {
public:
	Sanpshot(const Snapshot&x);
	Sanpshot(const shared_ptr<Monitor>&mon);
	~Snapshot() {}

	template <class T>
	const T::Passage &operator[](const shared_ptr<T> &monitor) const {
		return dynamic_cast<const T::Passage&>(monitor->resolve(shared_from_this()));}
private:
	//! The snapshot.
	const shared_ptr<Passage> m_passage;
	const shared_ptr<Monitor> m_monitor;
};

//! Transactional writing for a monitored data set.
//! The revision will be committed implicitly on leaving the scope.
class Transaction : public Snapshot {
public:
	Transaction(const Transaction&x) : Snapshot(x), m_newdata(x.m_newdata) {}
	Transaction(const shared_ptr<Monitor>&mon) : 
		Snapshot(mon), m_newdata(new Monitor::DataList(data())) {}
	~Transaction() {}
	//! Explicit commit.
	void commit();
	//! Abandon revision.
	void abort();

	template <class T>
	struct accessor {
		accessor(const shared_ptr<T> &t) : m_var(t) {}
		template <class X>
		operator X() const {return (X)*m_var;}
		T &operator->() const {return *m_var;}
		template <class X>
		accessor &operator=(const X &x) {m_var->value(x);}
	private:
		shared_ptr<T> m_var;
	};
	//! For implicit casting.
	template <class T>
	const accessor &operator[](const shared_ptr<T> &t) const {
		return accessor<T>(dynamic_pointer_cast<T>(m_newpacket->dataMap().find(t.ptr())->second->resolve()));}
private:
	shared_ptr<Monitor::Packet> m_newpacket;
};

class XMonitor : public Monitor {
};
class XGroupMonitor : public XMonitor {
};
template <typename T>
class Transactional : public _transactional {
	struct Data : public Metadata {
		shared_ptr<T> var;
	};
	atomic_shared_ptr<Data> m_data;
	void _commit(const shared_ptr<T>&t, const packed* = NULL);
	shared_ptr<const T> read(const Snapshot &shot) const;
};

template <typename T>
void Transactional::_commit(const shared_ptr<T>&t, const snapshot*) {
	shared_ptr<WatcherList> new_list;
	shared_ptr<Data> newone(new Data(*m_data));
	newone->var = t;
	for(int j = 0; j < newone->watchers.size(); j++) {
		Watcher watcher = newone->watchers->at(j);
		shared_ptr<Monitor> mon = watcher.monitor.lock();
		if(!mon) {
			//Remove from the list;
			if(!new_list)
				new_list.reset(new WatcherList(*newone->watchers));
			new_list.erase(std::remove(new_list.begin(), new_list.end(), watcher), new_list.end());
			continue;
		}
		if(mon->m_bActive) {
			if(packed && (packed->m_monitor == mon)) {
			//Write in the working set.
				packed->m_newdata->at(watcher.index) = newone;
			}
			else {
				for(;;) {
					atomic_shared_ptr<Monitor::TransactionalList> oldrecord(mon->m_watchpoint);
					atomic_shared_ptr<Monitor::TransactionalList> newrecord(new Monitor::TransactionalList(*mon->m_watchpoint));
					ASSERT(watcher.index < oldrecord->size());
					newrecord->at(watcher.index) = newone;
					if(mon->m_watchpoint.compareAndSet(oldrecord, newrecord))
						break;
				}
			}
		}
	}
	for(;;) {
		atomic_shared_ptr<Data> oldone(m_data);
		atomic_shared_ptr<Data> newone2(new Data(*m_data));
		newone2->var = t;
		if(new_list && (newone->watchers == newone2->watchers))
			newone2->watchers = new_list;
		if(m_data.compareAndSet(oldone, newone2))
			break;
	}
}
template <typename T>
void Transactional::write(const PackedWrite &packed, const shared_ptr<T>&t) {
	_commit(t, &packed);
}

template <typename T>
shared_ptr<const T> Transactional::read(const Snapshot &shot) const {
}

class XNode {
	typedef std::deque<shared_ptr<XNode> > NodeList;
	struct Property {
		int flags;
		NodeList children;
		weak_ptr<XNode> parent;
	};
	Transactional<Property> m_property;
};

#endif /*TRANSACTION_H*/
