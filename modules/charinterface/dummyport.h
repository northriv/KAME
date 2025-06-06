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
#ifndef DUMMYPORT_H_
#define DUMMYPORT_H_
#include "charinterface.h"

#include <fstream>

class XDummyPort : public XPort {
public:
	XDummyPort(XCharInterface *interface);
	virtual ~XDummyPort();
    virtual shared_ptr<XPort> open(const XCharInterface *pInterface) override;
    virtual void send(const char *str) override;
    virtual void write(const char *sendbuf, int size) override;
    virtual void receive() override;
    virtual void receive(unsigned int length) override;
private:
    std::ofstream m_stream;
};

#endif /*DUMMYPORT_H_*/
