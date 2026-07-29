/***************************************************************************
        Copyright (C) 2002-2024 Kentaro Kitagawa
		                   kitag@issp.u-tokyo.ac.jp
		
		This program is free software; you can redistribute it and/or
		modify it under the terms of the GNU General Public
		License as published by the Free Software Foundation; either
		version 2 of the License, or (at your option) any later version.
		
		You should have received a copy of the GNU General 
		Public License and a list of authors along with this program; 
		see the files COPYING and AUTHORS.
***************************************************************************/
#include "dummyport.h"

#if defined __WIN32__ || defined WINDOWS || defined _WIN32
	static XString dummyPortFilename() {return "kamedummyport.log";}
#else
	#include <unistd.h>
	// Same reasoning as kame/support.cpp's debug log: a fixed name in a
	// world-writable /tmp cannot be opened by a second user on a shared
	// machine, and the failure is silent.  Qualify by uid.
	static XString dummyPortFilename() {
		const char *tmp = getenv("TMPDIR");
		if( !tmp || !tmp[0]) tmp = "/tmp";
		return formatString("%s/kamedummyport-%lu.log", tmp, (unsigned long)getuid());
	}
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
    m_stream.open(dummyPortFilename().c_str(), std::ios::out);
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
