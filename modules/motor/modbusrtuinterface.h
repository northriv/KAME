/***************************************************************************
		Copyright (C) 2002-2012 Kentaro Kitagawa
		                   kitag@kochi-u.ac.jp

		This program is free software; you can redistribute it and/or
		modify it under the terms of the GNU Library General Public
		License as published by the Free Software Foundation; either
		version 2 of the License, or (at your option) any later version.

		You should have received a copy of the GNU Library General
		Public License and a list of authors along with this program;
		see the files COPYING and AUTHORS.
***************************************************************************/
#ifndef MODBUSRTUINTERFACE_H_
#define MODBUSRTUINTERFACE_H_

#include "charinterface.h"
#include "chardevicedriver.h"

class XModbusRTUInterface : public XCharInterface {
public:
	XModbusRTUInterface(const char *name, bool runtime, const shared_ptr<XDriver> &driver);
	virtual ~XModbusRTUInterface();

	void readHoldingResistors(uint16_t res_addr, int count, std::vector<uint16_t> &data);
	void presetSingeResistor(uint16_t res_addr, uint16_t data);
	void presetMultipleResistors(uint16_t res_no, int count, const std::vector<uint16_t> &data);
	void diagnostics();

	uint32_t readHoldingSingleResistor(uint16_t res_addr) {
		std::vector<uint16_t> data(1);
		readHoldingResistors(res_addr, 1, data);
		return data[0];
	}
	uint32_t readHoldingTwoResistors(uint16_t res_addr) {
		std::vector<uint16_t> data(2);
		readHoldingResistors(res_addr, 2, data);
		return data[0] * 0x10000uL + data[1];
	}
	void presetTwoResistors(uint16_t res_addr, uint32_t dword) {
		std::vector<uint16_t> data(2);
		data[0] = dword / 0x10000u;
		data[1] = dword % 0x10000u;
		presetMultipleResistors(res_addr, 2, data);
	}
protected:
	virtual void open() throw (XInterfaceError &);
	//! This can be called even if has already closed.
	virtual void close() throw (XInterfaceError &);

	virtual bool isOpened() const {return m_master;}

	void query_unicast(unsigned int func_code, const std::vector<unsigned char> &bytes, std::vector<unsigned char> &buf);
private:
	//Modbus utilizes Big endian.
	static void set_word(unsigned char *ptr, uint16_t word) {
		ptr[0] = static_cast<unsigned char>(word / 0x100u);
		ptr[1] = static_cast<unsigned char>(word % 0x100u);
	}
	static void set_dword(unsigned char *ptr, uint32_t dword) {
		set_word(ptr, static_cast<uint16_t>(dword / 0x10000u));
		set_word(ptr + 2, static_cast<uint16_t>(dword % 0x10000u));
	}
	static uint16_t get_word(unsigned char *ptr) {
		return ptr[1] +ptr[0] * 0x100u;
	}
	static uint32_t get_dword(unsigned char *ptr) {
		return get_word(ptr + 2) + get_word(ptr) * 0x10000uL;
	}
	uint16_t crc16(const unsigned char *bytes, ssize_t count);

	shared_ptr<XModbusRTUInterface> m_master;
	static XMutex s_lock;
	static std::deque<weak_ptr<XModbusRTUInterface> > s_masters; //guarded by s_lock.
	int m_openedCount; //guarded by s_lock.
};

template <class T>
class XModbusRTUDriver : public XCharDeviceDriver<T, XModbusRTUInterface> {
public:
	XModbusRTUDriver(const char *name, bool runtime,
	Transaction &tr_meas, const shared_ptr<XMeasure> &meas) :
		XCharDeviceDriver<T, XModbusRTUInterface>(name, runtime, ref(tr_meas), meas) {}
	virtual ~XModbusRTUDriver() {};
};

#endif /*MODBUSRTUINTERFACE_H_*/
