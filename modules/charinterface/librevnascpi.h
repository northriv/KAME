/***************************************************************************
        Copyright (C) 2002-2026 Kentaro Kitagawa
                           kitag@issp.u-tokyo.ac.jp

        This program is free software; you can redistribute it and/or
        modify it under the terms of the GNU General Public
        License as published by the Free Software Foundation; either
        version 2 of the License, or (at your option) any later version.

        You should have received a copy of the GNU General
        Public License and a list of authors along with this program;
        see the files COPYING and AUTHORS.
***************************************************************************/
#ifndef LIBREVNASCPI_H_
#define LIBREVNASCPI_H_

#include "charinterface.h"
#include <cctype>
#include <cstdio>

//! Speaks to the SCPI server of the LibreVNA-GUI, which both the network
//! analyzer driver and the signal generator driver connect to (TCP 19542).
//!
//! It exists for one reason: what that server does with an *event* -- a
//! command that changes a setting or triggers an action, i.e. everything
//! without a '?' -- changed incompatibly in GUI v1.6.0.
//!
//!  - up to v1.5.1 an event answered with an empty line, or with "ERROR"
//!  - from v1.6.0 an event answers with nothing whatsoever, and a command the
//!    server could not parse is reported through the CME bit of the event
//!    status register instead
//!
//! So an event must not be sent with query() to a modern GUI: no answer ever
//! arrives, and the read waits out the socket timeout (5 s, see tcp.cpp)
//! before failing -- which is what every setting in these drivers did against
//! any GUI from v1.6.0 on. Queries need none of this; both protocols answer
//! those identically, so only events go through here.
class LibreVNASCPI {
public:
	//! Reads the GUI version through *IDN?, which both protocols answer alike.
	//! Call once the interface is open, and before sending any event.
	void probeAPI(const shared_ptr<XCharInterface> &intf);
	//! Sends one event, and reports a rejected command the way this GUI reports it.
	void sendEvent(const shared_ptr<XCharInterface> &intf, const XString &cmd);
private:
	//! true once the GUI is known to answer an event with nothing at all.
	bool m_eventsAreSilent = false;
};

inline void
LibreVNASCPI::probeAPI(const shared_ptr<XCharInterface> &intf) {
	XScopedLock<XInterface> lock( *intf);
	intf->query("*IDN?");
	XString idn = intf->toStrSimplified();
	//LibreVNA,LibreVNA-GUI,<serial>,<software version>
	unsigned int major = 0, minor = 0;
	bool parsed = false;
	auto pos = idn.rfind(',');
	if(pos != std::string::npos) {
		const char *ver = idn.c_str() + pos + 1;
		while( *ver && !isdigit((unsigned char) *ver))
			ver++; //steps over a leading "v", should the version carry one.
		parsed = (sscanf(ver, "%u.%u", &major, &minor) == 2);
	}
	if( !parsed) {
		//Reading this wrong towards the old protocol would leave every event
		//waiting out the socket timeout, so assume what the releases do now,
		//and name the answer that could not be read.
		gWarnPrint(i18n("Unreadable LibreVNA-GUI version, assuming the current protocol: ") + idn);
		m_eventsAreSilent = true;
		return;
	}
	m_eventsAreSilent = (major > 1) || ((major == 1) && (minor >= 6));
}
inline void
LibreVNASCPI::sendEvent(const shared_ptr<XCharInterface> &intf, const XString &cmd) {
	XScopedLock<XInterface> lock( *intf); //holds the event and its check together.
	if( !m_eventsAreSilent) {
		intf->query(cmd);
		if(intf->toStr() == "ERROR\n")
			throw XInterface::XConvError(__FILE__, __LINE__);
		return;
	}
	intf->send(cmd);
	//Nothing comes back, so the event status register is the only place a
	//command it could not parse shows up. Reading the register clears it.
	intf->query("*ESR?");
	//CME is the only error bit this GUI sets today; the others are documented
	//as unused, so covering them costs nothing and keeps this honest if a
	//later version starts to use them.
	if(intf->toUInt() & (4u | 8u | 16u | 32u))
		throw XInterface::XConvError(__FILE__, __LINE__);
}

#endif /*LIBREVNASCPI_H_*/
