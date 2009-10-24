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
#ifndef ATOMIC_LIST_H_
#define ATOMIC_LIST_H_

#include "transactional.h"

template <typename T, class LIST = std::deque<T> >
class atomic_list : public transactional<LIST > {
public:
	typedef typename LIST::iterator iterator;
	typedef typename LIST::const_iterator const_iterator;

	atomic_list() {}
	~atomic_list() {}

	void push_back(const T &x) {
		for(;;) {
			typename atomic_list::writer tr(*this);
			tr->push_back(x);
			if(tr.commit())
				break;
		}
	}
};

#endif /* ATOMIC_LIST_H_ */
