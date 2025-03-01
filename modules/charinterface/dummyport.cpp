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
#include "dummyport.h"

#if defined __WIN32__ || defined WINDOWS || defined _WIN32
	#define DUMMYPORT_FILENAME "kamedummyport.log"
#else
	#define DUMMYPORT_FILENAME "/tmp/kamedummyport.log"
#endif

XDummyPort::XDummyPort(XCharInterface *interface) :
    XPort(interface),
    m_stream()
{
}
XDummyPort::~XDummyPort()
{
    m_stream.close();
}
shared_ptr<XPort> XDummyPort::open(const XCharInterface *pInterface)
{
    m_stream.open(DUMMYPORT_FILENAME, std::ios::out);
    return shared_from_this();
}
void
XDummyPort::send(const char *str)
{
    m_stream << "send:"
			 << str << std::endl;
}
void
XDummyPort::write(const char *sendbuf, int size)
{
    m_stream << "write:";
    m_stream.write(sendbuf, size);
    m_stream << std::endl;
}
void
XDummyPort::receive()
{
    m_stream << "receive:"
			 << std::endl;
    buffer().resize(1);
    buffer()[0] = '\0';
}
void
XDummyPort::receive(unsigned int length)
{
    m_stream << "receive length = :"
			 << length << std::endl;
    buffer().resize(length);
    buffer()[0] = '\0';
}
