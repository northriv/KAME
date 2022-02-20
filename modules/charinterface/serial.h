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
#ifndef SERIAL_H_
#define SERIAL_H_

#include "charinterface.h"

#if  defined __linux__ || defined __APPLE__
#define SERIAL_POSIX
#endif //__linux__ || LINUX

#if defined WINDOWS || defined __WIN32__ || defined _WIN32
#define SERIAL_WIN32
#endif // WINDOWS || __WIN32__ || defined _WIN32

#if defined SERIAL_WIN32 || defined SERIAL_POSIX
#define USE_SERIAL
#endif

class XSerialPort : public XPort {
public:
    XSerialPort(XCharInterface *interface);
    virtual ~XSerialPort();

    virtual void open(const XCharInterface *pInterface);
    virtual void send(const char *str);
    virtual void write(const char *sendbuf, int size);
    virtual void receive();
    virtual void receive(unsigned int length);
private:
#ifdef SERIAL_POSIX
    void flush();
    int m_scifd;
#endif /*SERIAL_POSIX*/
#ifdef SERIAL_WIN32
    void *m_handle;
#endif /*SERIAL_WIN32*/
    bool m_serialFlushBeforeWrite;
    bool m_serialHasEchoBack;
};

#endif /*SERIAL_H_*/
