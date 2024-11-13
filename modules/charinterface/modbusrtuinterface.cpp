#include "modbusrtuinterface.h"

std::deque<weak_ptr<XModbusRTUInterface::PortWrapper>> XModbusRTUInterface::s_openedPorts;
XMutex XModbusRTUInterface::s_globalMutex;

XModbusRTUInterface::XModbusRTUInterface(const char *name, bool runtime, const shared_ptr<XDriver> &driver) :
 XCharInterface(name, runtime, driver) {
    setEOS("");
    setSerialEOS("");
	setSerialBaudRate(9600);
    setSerialStopBits(1);
    trans( *device()) = "SERIAL";
}

XModbusRTUInterface::~XModbusRTUInterface() {
}
void
XModbusRTUInterface::open() {
    XScopedLock<XModbusRTUInterface> lock( *this);
    {
		Snapshot shot( *this);
        XScopedLock<XMutex> glock( s_globalMutex);
        for(auto it = s_openedPorts.begin(); it != s_openedPorts.end();) {
            if(auto pt = it->lock()) {
                if(pt->port->portString() == shot[ *port()].to_str()) {
//                    if(pt->m_ != serialBaudRate())
//                        throw XInterface::XOpenInterfaceError(__FILE__, __LINE__);
//                    if(pt->serialStopBits() != serialStopBits())
//                        throw XInterface::XOpenInterfaceError(__FILE__, __LINE__);
//                    if(pt->serialParity() != serialParity())
//                        throw XInterface::XOpenInterfaceError(__FILE__, __LINE__);
                    m_openedPort = pt;
                    //The COMM port has been already opened by m_openedPort.
					return;
				}
                ++it;
			}
            else
                it = s_openedPorts.erase(it); //cleans garbage.
        }
        //Opens new COMM device.
        XCharInterface::open();
        m_openedPort = std::make_shared<PortWrapper>();
        m_openedPort->port = openedPort();
        m_openedPort->lastTimeStamp = XTime::now();
        s_openedPorts.push_back(m_openedPort);
    }
}
void
XModbusRTUInterface::close() {
	XScopedLock<XModbusRTUInterface> lock( *this);
    m_openedPort.reset(); //release shared_ptr to the port if any.
    XCharInterface::close(); //release shared_ptr to the port if any.
}

uint16_t
XModbusRTUInterface::crc16(const unsigned char *bytes, uint32_t count) {
	uint16_t z = 0xffffu;
    for(uint32_t i = 0; i < count; ++i) {
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
    auto port = m_openedPort;

    double msec_per_char = 1e3 / serialBaudRate() * 12;
    double silent = 1e-3 * std::max(3.0, 3.5 * msec_per_char);
    for(;; msecsleep(silent * 1e3 / 2 + 1)) {
        {
            XScopedLock<XMutex> portlock(port->mutex);

            if(XTime::now() - port->lastTimeStamp < silent)
                continue; //puts silent interval. Waits outside the lock.

            port->lastTimeStamp = XTime::now();
            port->lastTimeStamp += 0.1; //for possible throwing.

            unsigned int slave_addr = ***address();
            std::vector<unsigned char> buf(bytes.size() + 4);
            buf[0] = static_cast<unsigned char>(slave_addr);
            buf[1] = static_cast<unsigned char>(func_code);
            std::copy(bytes.begin(), bytes.end(), &buf[2]);
            uint16_t crc = crc16( &buf[0], buf.size() - 2);
            set_word( &buf[buf.size() - 2], crc);

            port->port->writeTo(this, reinterpret_cast<char*>(&buf[0]), buf.size());
        //    msecsleep(buf.size() * msec_per_char + std::max(3.0, 3.5 * msec_per_char) + 0.5); //For O_NONBLOCK.

            buf.resize(ret_buf.size() + 4);
            port->port->receiveFrom(this, 2); //addr + func_code.
            std::copy(port->port->buffer().begin(), port->port->buffer().end(), buf.begin());

            if(buf[0] != slave_addr) {
                if(buf[1] == slave_addr) {
                    buf[0] = buf[1];
                    port->port->receiveFrom(this, 1); //func_code.
                    buf[1] = port->port->buffer()[0];
                }
                else if((buf[1] & 0x7fu) == func_code) {
                    buf[0] = slave_addr;
                }
                else
                    throw XInterfaceError(formatString("Modbus RTU Address Error, got %u instead of %u, and func. = %u.", buf[0], slave_addr, buf[1]), __FILE__, __LINE__);
                //met spurious start bit.
                gWarnPrint(formatString("Modbus RTU, ignores spurious start bit before a response for slave %u.", slave_addr));
            }
            if((buf[1] & 0x7fu) != func_code)
                throw XInterfaceError("Modbus RTU Format Error.", __FILE__, __LINE__);
            if(buf[1] != func_code) {
                port->port->receiveFrom(this, 3);
                switch(port->port->buffer()[0]) {
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

            port->port->receiveFrom(this, ret_buf.size() + 2); //Rest of message.
            std::copy(port->port->buffer().begin(), port->port->buffer().end(), buf.begin() + 2);
            crc = crc16( &buf[0], buf.size() - 2);
            if(crc != get_word( &buf[buf.size() - 2]))
                throw XInterfaceError("Modbus RTU CRC Error.", __FILE__, __LINE__);
            std::copy(port->port->buffer().begin(), port->port->buffer().end() - 2, ret_buf.begin());

            port->lastTimeStamp = XTime::now();
        }
        break;
    }
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
void
XModbusRTUInterface::send(const char *str) {
    auto port = m_openedPort;
    XScopedLock<XMutex> portlock(port->mutex);
    port->port->sendTo(this, str);
}
void
XModbusRTUInterface::receive() {
    auto port = m_openedPort;
    XScopedLock<XMutex> portlock(port->mutex);
    port->port->receiveFrom(this);
}
