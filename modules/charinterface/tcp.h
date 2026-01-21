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
 
    virtual shared_ptr<XPort> open(const XCharInterface *pInterface) override;
    virtual void send(const char *str) override;
    virtual void write(const char *sendbuf, int size) override;
    virtual void receive() override;
    virtual void receive(unsigned int length) override;
private:
    void reopen_socket();
	int m_socket;
    int m_timeout_sec = 5;
    int m_eosPosInBuffer = -1;
    char m_byteAfterEOS;
    unsigned int rearrangeBufferForNextReceive();
};

typedef XTCPSocketPort XTCPPort;

#endif // TCP_SOCKET

#endif /*TCP_H_*/
