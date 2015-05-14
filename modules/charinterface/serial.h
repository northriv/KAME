/***************************************************************************
		Copyright (C) 2002-2014 Kentaro Kitagawa
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

#ifdef SERIAL_POSIX

class XPosixSerialPort : public XPort {
public:
	XPosixSerialPort(XCharInterface *interface);
	virtual ~XPosixSerialPort();
 
	virtual void open() throw (XInterface::XCommError &);
	virtual void send(const char *str) throw (XInterface::XCommError &);
	virtual void write(const char *sendbuf, int size) throw (XInterface::XCommError &);
	virtual void receive() throw (XInterface::XCommError &);
	virtual void receive(unsigned int length) throw (XInterface::XCommError &);  
private:
	void flush();
	int m_scifd;
};

typedef XPosixSerialPort XSerialPort;

#endif /*SERIAL_POSIX*/

#ifdef SERIAL_WIN32

class QSerialPort;

class XWin32SerialPort : public XPort {
public:
    XWin32SerialPort(XCharInterface *interface);
    virtual ~XWin32SerialPort();

    virtual void open() throw (XInterface::XCommError &);
    virtual void send(const char *str) throw (XInterface::XCommError &);
    virtual void write(const char *sendbuf, int size) throw (XInterface::XCommError &);
    virtual void receive() throw (XInterface::XCommError &);
    virtual void receive(unsigned int length) throw (XInterface::XCommError &);
private:
    void *m_handle;
};

typedef XWin32SerialPort XSerialPort;

#endif /*SERIAL_WIN32*/

#endif /*SERIAL_H_*/
