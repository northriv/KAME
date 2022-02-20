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
#ifndef TCP_H_
#define TCP_H_

#include "charinterface.h"

#if  defined __linux__ || defined __APPLE__
    #define TCP_SOCKET
#endif //__linux__ || LINUX

#if defined WINDOWS || defined __WIN32__ || defined _WIN32
    #define TCP_SOCKET
#endif // WINDOWS || __WIN32__ || defined _WIN32

#if defined TCP_SOCKET
    #define USE_TCP
#endif

#ifdef TCP_SOCKET
class XTCPSocketPort : public XPort {
public:
    XTCPSocketPort(XCharInterface *interface);
    virtual ~XTCPSocketPort();
 
    virtual void open(const XCharInterface *pInterface);
    virtual void send(const char *str);
    virtual void write(const char *sendbuf, int size);
    virtual void receive();
    virtual void receive(unsigned int length);
private:
    void reopen_socket();
	int m_socket;
};

typedef XTCPSocketPort XTCPPort;

#endif // TCP_SOCKET

#endif /*TCP_H_*/
