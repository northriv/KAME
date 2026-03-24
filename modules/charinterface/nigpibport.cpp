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
// Include NiGpibDriver first — osx_compat.h must not see C++ keywords.
#include "nigpibport.h"
#include "usermode-linux-gpib/NiGpibDriver.h"

/* -----------------------------------------------------------------------
 * XNIUsermodeGPIBBoardPort
 * ----------------------------------------------------------------------- */

XNIUsermodeGPIBBoardPort::XNIUsermodeGPIBBoardPort(XCharInterface *intf)
    : XPort(intf) {}

XNIUsermodeGPIBBoardPort::~XNIUsermodeGPIBBoardPort() = default; //NiGpibDriver destructor runs here

NiGpibDriver &XNIUsermodeGPIBBoardPort::driver() {
    assert(m_drv);
    return *m_drv;
}

shared_ptr<XPort>
XNIUsermodeGPIBBoardPort::open(const XCharInterface *intf) {
    Snapshot shot( *intf);
    m_drv = std::make_unique<NiGpibDriver>(
        atoi(shot[ *intf->port()].to_str().c_str())/*board controller address*/,
        3000000 /*3 s timeout*/);
    if(!m_drv->open())
        throw XInterface::XCommError(
            i18n("NI USB-GPIB: no supported adapter found or open failed"), __FILE__, __LINE__);
    m_drv->interfaceClear();
    return shared_from_this();
}

/* -----------------------------------------------------------------------
 * XNIUsermodeGPIBPort helpers
 * ----------------------------------------------------------------------- */

static int gpibAddr(XCharInterface *intf) {
    return (int)(int)Snapshot(*intf)[ *intf->address()];
}

// Read terminator character for gpibNoEOI devices.
// Uses the penultimate EOS byte (e.g. CR of "\r\n") to avoid premature
// termination if the device prepends the final EOS char (Oxford prepends LF).
static uint8_t noEoiReadTerm(XCharInterface *intf) {
    const auto &eos = intf->eos();
    return (eos.size() >= 2) ? (uint8_t)eos[eos.size() - 2] : (uint8_t)eos.back();
}

/* -----------------------------------------------------------------------
 * XNIUsermodeGPIBPort
 * ----------------------------------------------------------------------- */

shared_ptr<XPort>
XNIUsermodeGPIBPort::open(const XCharInterface *pInterface) {
    auto p = static_pointer_cast<XNIUsermodeGPIBPort>(
        XAddressedPort<XNIUsermodeGPIBBoardPort>::open(pInterface));
    XScopedLock<XNIUsermodeGPIBPort> lock(*p);
    try {
        p->driver().deviceClear(gpibAddr(const_cast<XCharInterface *>(pInterface)));
    }
    catch(const std::exception &e) {
        throw XInterface::XCommError(e.what(), __FILE__, __LINE__);
    }
    return p;
}

// Wait for SRQ using cheap line_status checks, then do a full serial poll
// to read the STB.  Returns the STB, or 0 if SRQ never asserted within timeout.
// timeout_ms==0 means a single non-blocking check.
static uint8_t waitSRQThenSPoll(NiGpibDriver &drv, int addr,
                                 int wait_before_ms, unsigned int timeout_ms)
{
    if(wait_before_ms)
        msecsleep(wait_before_ms);
    // Poll cheaply (one USB round-trip) until SRQ is asserted or time expires.
    const unsigned int poll_interval_ms = 5;
    unsigned int elapsed = 0;
    while(!drv.checkSRQ()) {
        if(elapsed >= timeout_ms) return 0; // timed out, no SRQ
        msecsleep(poll_interval_ms);
        elapsed += poll_interval_ms;
    }
    return drv.serialPoll(addr);
}

void XNIUsermodeGPIBPort::spollBeforeWrite(XCharInterface *intf) {
    if(!intf->gpibUseSerialPollOnWrite() || !intf->gpibMAVbit()) return;
    int addr = gpibAddr(intf);
    uint8_t stb;
    try {
        // Single non-blocking SRQ check (timeout=0) — don't wait for the device.
        stb = waitSRQThenSPoll(driver(), addr, intf->gpibWaitBeforeSPoll(), 0);
    }
    catch(const std::exception &e) {
        throw XInterface::XCommError(e.what(), __FILE__, __LINE__);
    }
    if(stb & intf->gpibMAVbit()) {
        gErrPrint(i18n("ibrd before ibwrt asserted"));
        msecsleep(40);
        try {
            if(intf->gpibNoEOI() && !intf->eos().empty())
                driver().readEOS(addr, noEoiReadTerm(intf));
            else
                driver().read(addr);
        }
        catch(const std::exception &e) {
            throw XInterface::XCommError(e.what(), __FILE__, __LINE__);
        }
    }
}

void XNIUsermodeGPIBPort::spollBeforeRead(XCharInterface *intf) {
    if(!intf->gpibUseSerialPollOnRead() || !intf->gpibMAVbit()) return;
    int addr = gpibAddr(intf);
    try {
        // Single non-blocking SRQ check. If SRQ is already asserted, confirm MAV
        // via serial poll; if not, fall straight through to read() (which blocks
        // until the device drives EOI or the board timeout fires).
        // A 3-second SRQ wait loop would cause ~3 s overhead for instruments that
        // respond via EOI rather than SRQ, which is unacceptable on a shared bus.
        uint8_t stb = waitSRQThenSPoll(driver(), addr, intf->gpibWaitBeforeSPoll(), 0);
        if(stb && !(stb & intf->gpibMAVbit()))
            gWarnPrint(i18n("SRQ asserted but MAV not set"));
    }
    catch(const std::exception &e) {
        throw XInterface::XCommError(e.what(), __FILE__, __LINE__);
    }
}

void XNIUsermodeGPIBPort::sendTo(XCharInterface *intf, const char *str) {
    spollBeforeWrite(intf);
    if(intf->gpibWaitBeforeWrite())
        msecsleep(intf->gpibWaitBeforeWrite());
    int addr = gpibAddr(intf);
    const char *term = (intf->gpibNoEOI() && !intf->eos().empty())
                       ? intf->eos().c_str() : ""; //empty term → assert EOI
    try { driver().send(addr, std::string(str), term); }
    catch(const std::exception &e) {
        throw XInterface::XCommError(e.what(), __FILE__, __LINE__);
    }
}

void XNIUsermodeGPIBPort::writeTo(XCharInterface *intf, const char *sendbuf, int size) {
    spollBeforeWrite(intf);
    if(intf->gpibWaitBeforeWrite())
        msecsleep(intf->gpibWaitBeforeWrite());
    int addr = gpibAddr(intf);
    try { driver().send(addr, std::string(sendbuf, size), ""); } //assert EOI for raw writes
    catch(const std::exception &e) {
        throw XInterface::XCommError(e.what(), __FILE__, __LINE__);
    }
}

void XNIUsermodeGPIBPort::receiveFrom(XCharInterface *intf) {
    spollBeforeRead(intf);
    if(intf->gpibWaitBeforeRead())
        msecsleep(intf->gpibWaitBeforeRead());
    int addr = gpibAddr(intf);
    std::string response;
    try {
        if(intf->gpibNoEOI() && !intf->eos().empty())
            response = driver().readEOS(addr, noEoiReadTerm(intf));
        else
            response = driver().read(addr); //EOI termination
    }
    catch(const std::exception &e) {
        throw XInterface::XCommError(e.what(), __FILE__, __LINE__);
    }
    auto &buf = buffer();
    buf.assign(response.begin(), response.end());
    buf.push_back('\0');
}

void XNIUsermodeGPIBPort::receiveFrom(XCharInterface *intf, unsigned int length) {
    spollBeforeRead(intf);
    if(intf->gpibWaitBeforeRead())
        msecsleep(intf->gpibWaitBeforeRead());
    int addr = gpibAddr(intf);
    std::string response;
    try { response = driver().read(addr, length); }
    catch(const std::exception &e) {
        throw XInterface::XCommError(e.what(), __FILE__, __LINE__);
    }
    auto &buf = buffer();
    buf.assign(response.begin(), response.end());
}
