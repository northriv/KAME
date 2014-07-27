/***************************************************************************
		Copyright (C) 2002-2013 Kentaro Kitagawa
		                   kitag@kochi-u.ac.jp
		
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
#define TCP_POSIX
#endif //__linux__ || LINUX

#if defined WINDOWS || defined __WIN32__
#define TCP_QT
#endif // WINDOWS || __WIN32__

#if defined TCP_QT || defined TCP_POSIX
#define USE_TCP_PORT
#endif

#ifdef TCP_POSIX
class XPosixTCPPort : public XPort {
public:
	XPosixTCPPort(XCharInterface *interface);
	virtual ~XPosixTCPPort();
 
	virtual void open() throw (XInterface::XCommError &);
	virtual void send(const char *str) throw (XInterface::XCommError &);
	virtual void write(const char *sendbuf, int size) throw (XInterface::XCommError &);
	virtual void receive() throw (XInterface::XCommError &);
	virtual void receive(unsigned int length) throw (XInterface::XCommError &);  
private:
	int m_socket;
};

typedef XPosixTCPPort XTCPPort;

#endif // TCP_POSIX

#endif /*TCP_H_*/
