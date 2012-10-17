#include "modbusrtuinterface.h"

std::deque<weak_ptr<XModbusRTUInterface> > XModbusRTUInterface::s_masters;
XMutex XModbusRTUInterface::s_lock;

XModbusRTUInterface::XModbusRTUInterface(const char *name, bool runtime, const shared_ptr<XDriver> &driver) :
 XCharInterface(name, runtime, driver),
 m_openedCount(0) {
    setEOS("");
	setSerialBaudRate(9600);
	setSerialStopBits(1);
}

XModbusRTUInterface::~XModbusRTUInterface() {
}
void
XModbusRTUInterface::open() throw (XInterfaceError &) {
	{
		XScopedLock<XModbusRTUInterface> lock( *this);
		m_master = dynamic_pointer_cast<XModbusRTUInterface>(shared_from_this());
		Snapshot shot( *this);
		XScopedLock<XMutex> glock(s_lock);
		for(auto it = s_masters.begin(); it != s_masters.end(); ++it) {
			if(auto mint = it->lock()) {
				if((XString)Snapshot( *mint)[ *mint->port()] == (XString)shot[ *port()]) {
					m_master =mint;
					assert(m_master->m_openedCount);
					//The port has been already opened by m_master.
					m_master->m_openedCount++;
					return;
				}
			}
		}
		s_masters.push_back(m_master);
		m_master->m_openedCount = 1;
	}
	XCharInterface::open();
}
void
XModbusRTUInterface::close() throw (XInterfaceError &) {
	XScopedLock<XModbusRTUInterface> lock( *this);
	XScopedLock<XMutex> glock(s_lock);
	if(m_master) {
		m_master->m_openedCount--;
		if( !m_master->m_openedCount) {
			for(auto it = s_masters.begin(); it != s_masters.end();) {
				if(m_master == it->lock())
					it = s_masters.erase(it);
				else
					++it;
			}
			m_master->XCharInterface::close();
		}
		m_master.reset();
	}
}

uint16_t
XModbusRTUInterface::crc16(const unsigned char *bytes, ssize_t count) {
	uint16_t z = 0xffffu;
	for(ssize_t i = 0; i < count; ++i) {
		uint16_t x = bytes[i];
		z ^= x;
		for(int shifts = 0; shifts < 8; ++shifts) {
			uint16_t lsb = z % 2;
			z = z >> 1;
			if(lsb)
				z ^= 0xa001u;
		}
	}
	return (z % 0x100u) * 0x100u + z / 0x100u;
}
void
XModbusRTUInterface::query_unicast(unsigned int func_code,
		const std::vector<unsigned char> &bytes, std::vector<unsigned char> &ret_buf) {
	auto master = m_master;
	double msec_per_char = 1e3 / serialBaudRate() * 12;
	XScopedLock<XModbusRTUInterface> lock( *master);
	unsigned int slave_addr = ***address();
	std::vector<unsigned char> buf(bytes.size() + 4);
	buf[0] = static_cast<unsigned char>(slave_addr);
	buf[1] = static_cast<unsigned char>(func_code);
	std::copy(bytes.begin(), bytes.end(), &buf[2]);
	uint16_t crc = crc16( &buf[0], buf.size() - 2);
	set_word( &buf[buf.size() - 2], crc);

	msecsleep(std::max(1.75, 3.5 * msec_per_char)); //puts silent interval.
	master->write( reinterpret_cast<char*>(&buf[0]), buf.size());
	msecsleep(buf.size() * msec_per_char); //For O_NONBLOCK.
	msecsleep(std::max(1.75, 3.5 * msec_per_char)); //puts silent interval.

	buf.resize(ret_buf.size() + 4);
	master->receive(2); //addr + func_code.
	std::copy(buffer().begin(), buffer().end(), buf.begin());

	if((buf[0] != slave_addr) || ((buf[1] & 0x7fu) != func_code))
		throw XInterfaceError("Modbus RTU Format Error.", __FILE__, __LINE__);
	if(buf[1] != func_code) {
		master->receive(3);
		switch(buffer()[0]) {
		case 0x01:
			throw XInterfaceError("Modbus RTU Ill Function.", __FILE__, __LINE__);
		case 0x02:
			throw XInterfaceError("Modbus RTU Wrong Data Address.", __FILE__, __LINE__);
		case 0x03:
			throw XInterfaceError("Modbus RTU Wrong Data.", __FILE__, __LINE__);
		case 0x04:
			throw XInterfaceError("Modbus RTU Slave Error.", __FILE__, __LINE__);
		default:
			throw XInterfaceError("Modbus RTU Format Error.", __FILE__, __LINE__);
		}
	}

	master->receive( ret_buf.size() + 2); //Rest of message.
	std::copy(buffer().begin(), buffer().end(), buf.begin() + 2);
	crc = crc16( &buf[0], buf.size() - 2);
	if(crc != get_word( &buf[buf.size() - 2]))
		throw XInterfaceError("Modbus RTU CRC Error.", __FILE__, __LINE__);
	std::copy(buffer().begin(), buffer().end() - 2, ret_buf.begin());
}
void
XModbusRTUInterface::readHoldingResistors(uint16_t res_addr, int count, std::vector<uint16_t> &data) {
	std::vector<unsigned char> wrbuf(4);
	set_word( &wrbuf[0], res_addr);
	set_word( &wrbuf[2], count);
	std::vector<unsigned char> rdbuf(2 * count + 1);
	query_unicast(0x03, wrbuf, rdbuf);
	data.resize(count);
	if(rdbuf[0] != 2 * count)
		throw XInterfaceError("Modbus RTU Format Error.", __FILE__, __LINE__);
	for(unsigned int i = 0; i < count; ++i) {
		data[i] = get_word( &rdbuf[2 * i + 1]);
	}
}
void
XModbusRTUInterface::presetSingleResistor(uint16_t res_addr, uint16_t data) {
	std::vector<unsigned char> wrbuf(4);
	set_word( &wrbuf[0], res_addr);
	set_word( &wrbuf[2], data);
	std::vector<unsigned char> rdbuf(4);
	query_unicast(0x06, wrbuf, rdbuf);
	if(rdbuf.back() != wrbuf.back())
		throw XInterfaceError("Modbus Format Error.", __FILE__, __LINE__);
}
void
XModbusRTUInterface::presetMultipleResistors(uint16_t res_no, int count, const std::vector<uint16_t> &data) {
	std::vector<unsigned char> wrbuf(5 + 2 * count);
	set_word( &wrbuf[0], res_no);
	set_word( &wrbuf[2], count);
	wrbuf[4] = count * 2;
	int idx = 5;
	for(auto it = data.begin(); it != data.end(); ++it) {
		set_word( &wrbuf[idx], *it);
		idx += 2;
	}
	std::vector<unsigned char> rdbuf(4);
	query_unicast(0x10, wrbuf, rdbuf);
}
void
XModbusRTUInterface::diagnostics() {
	std::vector<unsigned char> wrbuf(4);
	set_word( &wrbuf[0], 0);
	set_word( &wrbuf[2], 0x1234);
	std::vector<unsigned char> rdbuf(4);
	query_unicast(0x08, wrbuf, rdbuf);
}
