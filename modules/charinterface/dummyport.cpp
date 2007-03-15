/***************************************************************************
		Copyright (C) 2002-2007 Kentaro Kitagawa
		                   kitagawa@scphys.kyoto-u.ac.jp
		
		This program is free software; you can redistribute it and/or
		modify it under the terms of the GNU Library General Public
		License as published by the Free Software Foundation; either
		version 2 of the License, or (at your option) any later version.
		
		You should have received a copy of the GNU Library General 
		Public License and a list of authors along with this program; 
		see the files COPYING and AUTHORS.
***************************************************************************/
#include "dummyport.h"

XDummyPort::XDummyPort(XCharInterface *interface) :
    XPort(interface),
    m_stream()
{
}
XDummyPort::~XDummyPort()
{
    m_stream.close();
}
void
XDummyPort::open() throw (XInterface::XCommError &)
{
    m_stream.open("/tmp/kamedummyport.log", std::ios::out);
}
void
XDummyPort::send(const char *str) throw (XInterface::XCommError &)
{
    m_stream << "send:"
			 << str << std::endl;
}
void
XDummyPort::write(const char *sendbuf, int size) throw (XInterface::XCommError &)
{
    m_stream << "write:";
    m_stream.write(sendbuf, size);
    m_stream << std::endl;
}
void
XDummyPort::receive() throw (XInterface::XCommError &)
{
    m_stream << "receive:"
			 << std::endl;
    buffer().resize(1);
    buffer()[0] = '\0';
}
void
XDummyPort::receive(unsigned int length) throw (XInterface::XCommError &)
{
    m_stream << "receive length = :"
			 << length << std::endl;
    buffer().resize(length);
    buffer()[0] = '\0';
}
