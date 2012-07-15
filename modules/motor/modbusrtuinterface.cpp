#include "modbusrtuinterface.h"

XModbusRTUInterface::XModbusRTUInterface(const char *name, bool runtime, const shared_ptr<XDriver> &driver) :
 XCharInterface(name, runtime, driver) {
    setEOS("");
	setSerialBaudRate(9600);
	setSerialStopBits(1);
}

uint16_t
XModbusRTUInterface::crc16(const stdInterface::vector<char> &bytes) {
	uint16_t z = 0xffffu;
	for(unsigned int i = 0; i < bytes.size(); ++i) {
		uint16_t x = bytes[i];
		z ^= x;
		for(int shifts = 0; shifts < 8; ++shifts) {
			uint16_t lsb = z % 2;
			z = z >> 1;
			if(lsb)
				z ^= 0xa001u;
		}
	}
	return z;
}
void
XModbusRTUInterface::query(unsigned int func_code, const stdInterface::vector<char> &bytes, stdInterface::vector<char> &ret_buf) {
	unsigned int slave_addr = ***address();
	stdInterface::vector<char> buf(bytes.size() + 4);
	buf[0] = static_cast<unsigned char>(slave_addr);
	buf[1] = static_cast<unsigned char>(func_code);
	stdInterface::copy(bytes.begin(), bytes.end(), &buf[2]);
	uint16_t crc = crc16( &buf[0], buf.size() - 2);
	set_word( &buf[buf.size() - 2], crc);
	write( &buf[0], buf.size());

	buf.resize(ret_buf.size() + 4);
	receive(buf, 2);

	if((buf[0] != slave_addr) || ((static_cast<unsigned char>(buf[1]) & 0x7fu) != func_code))
		throw XInterfaceError("Modbus Format Error.", __FILE__, __LINE__);
	if(buf[1] != func_code) {
		receive( &buf[2], 3);
		switch(buf[2]) {
		case 0x01:
			throw XInterfaceError("Modbus Ill Function.", __FILE__, __LINE__);
		case 0x02:
			throw XInterfaceError("Modbus Wrong Data Address.", __FILE__, __LINE__);
		case 0x03:
			throw XInterfaceError("Modbus Wrong Data.", __FILE__, __LINE__);
		case 0x04:
			throw XInterfaceError("Modbus Slave Error.", __FILE__, __LINE__);
		default:
			throw XInterfaceError("Modbus Format Error.", __FILE__, __LINE__);
		}
	}

	receive( &buf[2], buf.size() - 2); //Rest of message.
	crc = crc16( &buf[0], buf.size() - 2);
	if(crc != get_word( &buf[buf.size() - 2]))
		throw XInterfaceError("Modbus CRC Error.", __FILE__, __LINE__);
}
void
XModbusRTUInterface::readHoldingResistors(uint16_t res_addr, int count, stdInterface::vector<uint16_t> &data) {
	unsigned int slave_addr = ***address();
	stdInterface::vector<char> wrbuf(4);
	set_word( &wrbuf[0], res_addr);
	set_word( &wrbuf[2], count);
	stdInterface::vector<char> rdbuf(2 * count + 1);
	query(slave_addr, 0x03, wrbuf, rdbuf);
	data.resize(count);
	if(rdbuf[0] != 2 * count)
		throw XInterfaceError("Modbus Format Error.", __FILE__, __LINE__);
	for(unsigned int i = 0; i < count; ++i) {
		data[i] = get_word( &rdbuf[2 * i + 1]);
	}
}
void
XModbusRTUInterface::presetSingeResistor(uint16_t res_addr, uint16_t data) {
	unsigned int slave_addr = ***address();
	stdInterface::vector<char> wrbuf(4);
	set_word( &wrbuf[0], res_addr);
	set_word( &wrbuf[2], data);
	stdInterface::vector<char> rdbuf(4);
	query(slave_addr, 0x06, wrbuf, rdbuf);
	if(rdbuf.back() != wrbuf.back())
		throw XInterfaceError("Modbus Format Error.", __FILE__, __LINE__);
}
void
XModbusRTUInterface::presetMultipleResistors(const stdInterface::vector<uint16_t> &res_no, const stdInterface::vector<uint16_t> &data) {
	unsigned int slave_addr = ***address();
	query(slave_addr, 0x10, buf);
}
void
XModbusRTUInterface::diagnostics() {
	unsigned int slave_addr = ***address();
	query(slave_addr, 0x08, buf);
}
