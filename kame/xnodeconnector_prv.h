/***************************************************************************
		Copyright (C) 2002-2015 Kentaro Kitagawa
		                   kitagawa@phys.s.u-tokyo.ac.jp
		
		This program is free software; you can redistribute it and/or
		modify it under the terms of the GNU Library General Public
		License as published by the Free Software Foundation; either
		version 2 of the License, or (at your option) any later version.
		
		You should have received a copy of the GNU Library General 
		Public License and a list of authors along with this program; 
		see the files COPYING and AUTHORS.
***************************************************************************/
#ifndef XNODECONNECTOR_PRV_H_
#define XNODECONNECTOR_PRV_H_


//! function for creating XQConnector instances
template <class T, class A, class B>
xqcon_ptr xqcon_create(const shared_ptr<A> &a, B *b) {
    xqcon_ptr pHolder(new XQConnectorHolder_( new
											  T(a, b)));
    return pHolder;
}
//! function for creating XQConnector instances
template <class T, class A, class B, typename C>
xqcon_ptr xqcon_create(const shared_ptr<A> &a, B *b, C c) {
    xqcon_ptr pHolder(new XQConnectorHolder_( new
											  T(a, b, c)));        
    return pHolder;
}
//! function for creating XQConnector instances
template <class T, class A, class B, typename C, typename D>
xqcon_ptr xqcon_create(const shared_ptr<A> &a, B *b, C c, D d) {
    xqcon_ptr pHolder(new XQConnectorHolder_( new
											  T(a, b, c, d)));        
    return pHolder;
}
//! function for creating XQConnector instances
template <class T, class A, class B, typename C, typename D, typename E>
xqcon_ptr xqcon_create(const shared_ptr<A> &a, B *b, C c, D d, E e) {
    xqcon_ptr pHolder(new XQConnectorHolder_( new
                                              T(a, b, c, d, e)));
    return pHolder;
}

#endif /*XNODECONNECTOR_PRV_H_*/
