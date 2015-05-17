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
#ifndef FUJIKININTERFACE_H_
#define FUJIKININTERFACE_H_

#include "charinterface.h"
#include "chardevicedriver.h"

class XFujikinInterface : public XCharInterface {
public:
	XFujikinInterface(const char *name, bool runtime, const shared_ptr<XDriver> &driver);
	virtual ~XFujikinInterface();

	template <typename T>
	void send(uint8_t classid, uint8_t instanceid, uint8_t attributeid, T data);
	template <typename T>
	T query(uint8_t classid, uint8_t instanceid, uint8_t attributeid);
protected:
	virtual void open() throw (XInterfaceError &);
	//! This can be called even if has already closed.
	virtual void close() throw (XInterfaceError &);

	virtual bool isOpened() const {return !!m_master;}
private:
	void communicate(uint8_t classid, uint8_t instanceid, uint8_t attributeid,
		const std::vector<uint8_t> &data, std::vector<uint8_t> *response = 0);

	shared_ptr<XFujikinInterface> m_master;
	static XMutex s_lock;
	static std::deque<weak_ptr<XFujikinInterface> > s_masters; //guarded by s_lock.
	int m_openedCount; //guarded by s_lock.

	enum {STX = 0x02, ACK = 0x06, 	NAK = 0x16};
};

template <class T>
class XFujikinProtocolDriver : public XCharDeviceDriver<T, XFujikinInterface> {
public:
	XFujikinProtocolDriver(const char *name, bool runtime,
	Transaction &tr_meas, const shared_ptr<XMeasure> &meas) :
		XCharDeviceDriver<T, XFujikinInterface>(name, runtime, ref(tr_meas), meas) {}
	virtual ~XFujikinProtocolDriver() {};
};

#endif /*FUJIKININTERFACE_H_*/
