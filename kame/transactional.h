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
#ifndef TRANSACTIONAL_H_
#define TRANSACTIONAL_H_

#include "atomic_smart_ptr.h"

//! Transactional memory access on \a atomic_shared_ptr.
//! Examples:\n
/*!
 * for(;;) {\n
 * 	transactional<X>::writer tr(target);\n
 * 	tr->x++;\n
 * 	tr->y--;\n
 * 	if(tr.commit())\n
 * 		break; //Success.\n
 *  //Committing failed because another thread has made changes.\n
 * }\n
 *\n
 *	transactional<X>::reader snapshot(target);\n
 *	if(snapshot) {\n
 *	//Good reference. \n
 *		a = snapshot->x;\n
 *		b = snapshot->y;\n
 *	}\n
 */
//! \sa atomic_shared_ptr, atomic_list
template <typename X>
class transactional : public atomic_shared_ptr<X> {
public:
	typedef atomic_shared_ptr<X> shared_ptr;
	typedef atomic_shared_ptr<const X> shared_const_ptr;
	typedef atomic_shared_ptr<const X> reader;

	class writer : public atomic_shared_ptr<X> {
	public:
		writer(transactional &x) :
			atomic_shared_ptr<X>(),
			m_target(x),
			m_old_var(x) {
			reset_unsafe(m_old_var ? (new X(*m_old_var)) : (new X()));
		}
		~writer() {}
		bool commit() {
			return (compareAndSet(m_old_var, m_target));
		}
	protected:
		writer();
	private:
		transactional &m_target;
		const shared_ptr m_old_var;
	};

	transactional() : atomic_shared_ptr<X>() {}
	transactional(const transactional &x) : atomic_shared_ptr<X>(x) {}
	~transactional() {}
};

#endif /* TRANSACTIONAL_H_ */
