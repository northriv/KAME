/***************************************************************************
		Copyright (C) 2002-2014 Kentaro Kitagawa
		                   kitag@kochi-u.ac.jp

		This program is free software; you can redistribute it and/or
		modify it under the terms of the GNU Library General Public
		License as published by the Free Software Foundation; either
		version 2 of the License, or (at your option) any later version.

		You should have received a copy of the GNU Library General
		Public License and a list of authors along with this program;
		see the files COPYING and AUTHORS.
***************************************************************************/
#ifndef ATOMIC_LIST_H_
#define ATOMIC_LIST_H_

#include "atomic_smart_ptr.h"
#include <deque>

template <typename T, class LIST = std::deque<T> >
class atomic_list : public atomic_shared_ptr<LIST> {
public:
	typedef typename LIST::iterator iterator;
	typedef typename LIST::const_iterator const_iterator;
	typedef local_shared_ptr<const LIST> reader;

	atomic_list() {}
	~atomic_list() {}

	class writer : public local_shared_ptr<LIST> {
	public:
		writer(atomic_list &x) :
			local_shared_ptr<LIST>(),
			m_target(x),
			m_old_var(x) {
			reset_unsafe(m_old_var ? (new LIST(*m_old_var)) : (new LIST()));
		}
		~writer() {}
		bool commit() {
			return (m_target.compareAndSet(m_old_var, *this));
		}
	protected:
		writer();
	private:
		atomic_list &m_target;
		const local_shared_ptr<LIST> m_old_var;
	};

	void push_back(const T &x) {
		for(;;) {
			auto tr(*this);
			tr->push_back(x);
			if(tr.commit())
				break;
		}
	}
};

#endif /* ATOMIC_LIST_H_ */
