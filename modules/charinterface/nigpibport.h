/***************************************************************************
        Copyright (C) 2002-2026 Kentaro Kitagawa
		                   kitag@issp.u-tokyo.ac.jp

		This program is free software; you can redistribute it and/or
		modify it under the terms of the GNU Library General Public
		License as published by the Free Software Foundation; either
		version 2 of the License, or (at your option) any later version.

		You should have received a copy of the GNU Library General
		Public License and a list of authors along with this program;
		see the files COPYING and AUTHORS.
***************************************************************************/
#ifndef NIGPIBPORT_H_
#define NIGPIBPORT_H_

#include "charinterface.h"
#include <memory>

class NiGpibDriver; //!< defined in usermode-linux-gpib/NiGpibDriver.h

//! Physical board port — one NiGpibDriver instance per NI USB-GPIB adapter.
//! \sa XNIUsermodeGPIBPort
class XNIUsermodeGPIBBoardPort : public XPort {
public:
    explicit XNIUsermodeGPIBBoardPort(XCharInterface *intf);
    virtual ~XNIUsermodeGPIBBoardPort(); //!< defined in .cpp (NiGpibDriver is incomplete here)

    virtual shared_ptr<XPort> open(const XCharInterface *pInterface) override;

    NiGpibDriver &driver();

protected:
    //! These are never called directly — XNIUsermodeGPIBPort handles all I/O via sendTo/receiveFrom.
    virtual void send(const char *) override { assert(false); }
    virtual void write(const char *, int) override { assert(false); }
    virtual void receive() override { assert(false); }
    virtual void receive(unsigned int) override { assert(false); }

private:
    std::unique_ptr<NiGpibDriver> m_drv;
};

//! Addressed GPIB port backed by the usermode NI USB-GPIB driver (macOS).
//! One XNIUsermodeGPIBBoardPort (= one NiGpibDriver) is shared across all
//! instruments on the same adapter; instruments are distinguished by address.
class XNIUsermodeGPIBPort : public XAddressedPort<XNIUsermodeGPIBBoardPort> {
public:
    explicit XNIUsermodeGPIBPort(XCharInterface *intf)
        : XAddressedPort<XNIUsermodeGPIBBoardPort>(intf) {}
    virtual ~XNIUsermodeGPIBPort() = default;

    virtual shared_ptr<XPort> open(const XCharInterface *pInterface) override;

    virtual void sendTo(XCharInterface *intf, const char *str) override;
    virtual void writeTo(XCharInterface *intf, const char *sendbuf, int size) override;
    virtual void receiveFrom(XCharInterface *intf) override;
    virtual void receiveFrom(XCharInterface *intf, unsigned int length) override;

private:
    //! Single-shot SRQ check before write: drains output buffer if MAV is asserted.
    void spollBeforeWrite(XCharInterface *intf);
    //! Wait for SRQ (via cheap line_status loop) then serial-poll to verify MAV before read.
    void spollBeforeRead(XCharInterface *intf);
};

#endif /*NIGPIBPORT_H_*/
