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
#ifndef SERIAL_H_
#define SERIAL_H_

#include "charinterface.h"

#if  defined __linux__ || defined __APPLE__
#define SERIAL_POSIX
#endif //__linux__ || LINUX

#if defined WINDOWS || defined __WIN32__
#define SERIAL_QT
#endif // WINDOWS || __WIN32__

#if defined SERIAL_QT || defined SERIAL_POSIX
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

#ifdef SERIAL_QT

class QSerialPort;

class XQtSerialPort : public XPort {
public:
    XQtSerialPort(XCharInterface *interface);
    virtual ~XQtSerialPort();

    virtual void open() throw (XInterface::XCommError &);
    virtual void send(const char *str) throw (XInterface::XCommError &);
    virtual void write(const char *sendbuf, int size) throw (XInterface::XCommError &);
    virtual void receive() throw (XInterface::XCommError &);
    virtual void receive(unsigned int length) throw (XInterface::XCommError &);
private:
    shared_ptr<QSerialPort> m_qport;
};

typedef XQtSerialPort XSerialPort;

#endif /*SERIAL_QT*/

#endif /*SERIAL_H_*/
