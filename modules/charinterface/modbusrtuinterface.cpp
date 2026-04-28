#include "modbusrtuinterface.h"

#include "serial.h"
#include <atomic>

// Inter-transaction silent gap on a multidrop RS485 bus is enforced
// via an atomic timestamp shared by every XModbusRTUInterface that
// resolves to the same XAddressedPort instance (s_openedPorts shares
// the port across interfaces with identical port strings).
//
// Each query_unicast CAS-claims the slot — winning thread sets the
// timestamp to "now + 100ms grace" so concurrent claimants spin until
// the in-flight transaction releases by writing the real "now"
// timestamp. The per-interface XInterface mutex no longer arbitrates
// the bus (cef9a427 dropped the cascade to port-mutex for performance);
// the CAS plays that role without serialising single-port driver work.
class XModbusRTUPort : public XAddressedPort<XSerialPort> {
public:
    XModbusRTUPort(XCharInterface *interface) : XAddressedPort<XSerialPort>(interface) {
        m_lastTimeStamp.store(XTime::now(), std::memory_order_relaxed);
    }
    virtual ~XModbusRTUPort() {}

    virtual void sendTo(XCharInterface *intf, const char *str) override {send(str);}
    virtual void writeTo(XCharInterface *intf, const char *sendbuf, int size) override {write(sendbuf, size);}
    virtual void receiveFrom(XCharInterface *intf) override {receive();}
    virtual void receiveFrom(XCharInterface *intf, unsigned int length) override {receive(length);}

    std::atomic<XTime> m_lastTimeStamp;
};

XModbusRTUInterface::XModbusRTUInterface(const char *name, bool runtime, const shared_ptr<XDriver> &driver) :
 XCharInterface(name, runtime, driver) {
    setEOS("");
    setSerialTCPIPEOS("");
	setSerialBaudRate(9600);
    setSerialStopBits(1);
    trans( *device()) = "SERIAL";
}

XModbusRTUInterface::~XModbusRTUInterface() {
}
void
XModbusRTUInterface::open() {
    close();
    shared_ptr<XPort> port = std::make_shared<XModbusRTUPort>(this);
    const char *seos = (eos().length()) ? eos().c_str() : serialTCPIPEOS().c_str();
    port->setEOS(seos);
    openPort(port);
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
    auto port = static_pointer_cast<XModbusRTUPort>(openedPort());

    double msec_per_char = 1e3 / serialBaudRate() * 12;
    double silent = 1e-3 * std::max(3.0, 3.5 * msec_per_char);
    unsigned int sleep_ms = std::max<unsigned int>(1, (unsigned int)(silent * 500 + 1));

    // Bus-claim CAS loop — see XModbusRTUPort header comment.
    for(;;) {
        XTime old_ts = port->m_lastTimeStamp.load(std::memory_order_acquire);
        XTime now = XTime::now();
        if(now - old_ts < silent) {
            msecsleep(sleep_ms);
            continue;
        }
        // Hold the slot for up to 100 ms while we run write+receives;
        // a successful CAS publishes (now + 100ms) so other claimants spin.
        XTime grace = now;
        grace += 0.1;
        if( !port->m_lastTimeStamp.compare_exchange_strong(
                old_ts, grace,
                std::memory_order_acq_rel, std::memory_order_acquire))
            continue;
        break;
    }
    try {
        XScopedLock<XInterface> portlock( *this);

        unsigned int slave_addr = ***address();
        std::vector<unsigned char> buf(bytes.size() + 4);
        buf[0] = static_cast<unsigned char>(slave_addr);
        buf[1] = static_cast<unsigned char>(func_code);
        std::copy(bytes.begin(), bytes.end(), &buf[2]);
        uint16_t crc = crc16( &buf[0], buf.size() - 2);
        set_word( &buf[buf.size() - 2], crc);

        write(reinterpret_cast<char*>(&buf[0]), buf.size());
    //    msecsleep(buf.size() * msec_per_char + std::max(3.0, 3.5 * msec_per_char) + 0.5); //For O_NONBLOCK.

        buf.resize(ret_buf.size() + 4);
        receive(2); //addr + func_code.
        std::copy(buffer_receive().begin(), buffer_receive().end(), buf.begin());

        if(buf[0] != slave_addr) {
            if(buf[1] == slave_addr) {
                buf[0] = buf[1];
                receive(1); //func_code.
                buf[1] = buffer_receive()[0];
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
            receive(3);
            switch(buffer_receive()[0]) {
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

        receive(ret_buf.size() + 2); //Rest of message.
        std::copy(buffer_receive().begin(), buffer_receive().end(), buf.begin() + 2);
        crc = crc16( &buf[0], buf.size() - 2);
        if(crc != get_word( &buf[buf.size() - 2]))
            throw XInterfaceError("Modbus RTU CRC Error.", __FILE__, __LINE__);
        std::copy(buffer_receive().begin(), buffer_receive().end() - 2, ret_buf.begin());

        // Successful transaction: publish real "now" so other waiters
        // can claim after the silent gap.
        port->m_lastTimeStamp.store(XTime::now(), std::memory_order_release);
    }
    catch( ...) {
        // On exception, also release the slot promptly so the bus
        // doesn't sit idle for the full 100ms grace.
        port->m_lastTimeStamp.store(XTime::now(), std::memory_order_release);
        throw;
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
