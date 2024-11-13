#include "fujikininterface.h"

std::deque<weak_ptr<XPort> > XFujikinInterface::s_openedPorts;
XMutex XFujikinInterface::s_lock;

XFujikinInterface::XFujikinInterface(const char *name, bool runtime, const shared_ptr<XDriver> &driver) :
 XCharInterface(name, runtime, driver) {
    setEOS("");
    setSerialEOS("");
	setSerialBaudRate(38400);
	setSerialStopBits(1);
    trans( *device()) = "SERIAL";
}

XFujikinInterface::~XFujikinInterface() {
}

void
XFujikinInterface::open() {
    XScopedLock<XFujikinInterface> lock( *this);
    {
        Snapshot shot( *this);
        XScopedLock<XMutex> glock(s_lock);
        for(auto it = s_openedPorts.begin(); it != s_openedPorts.end();) {
            if(auto pt = it->lock()) {
                if(pt->portString() == (XString)shot[ *port()]) {
                    m_openedPort = pt;
                    //The COMM port has been already opened by m_master.
                    return;
                }
                ++it;
            }
            else
                it = s_openedPorts.erase(it); //cleans garbage.
        }
    }
    //Opens new COMM device.
    XCharInterface::open();
    m_openedPort = openedPort();
    s_openedPorts.push_back(m_openedPort);
}
void
XFujikinInterface::close() {
	XScopedLock<XFujikinInterface> lock( *this);
	XScopedLock<XMutex> glock(s_lock);
    m_openedPort.reset(); //release shared_ptr to the port if any.
    XCharInterface::close(); //release shared_ptr to the port if any.
}

template <typename T>
void
XFujikinInterface::send(uint8_t classid, uint8_t instanceid, uint8_t attributeid, T data) {
}
template <>
void
XFujikinInterface::send(uint8_t classid, uint8_t instanceid, uint8_t attributeid, uint8_t data) {
std::vector<uint8_t> wbuf(1);
	wbuf[0] = data;
	communicate(classid, instanceid, attributeid, wbuf);
}
template <>
void
XFujikinInterface::send(uint8_t classid, uint8_t instanceid, uint8_t attributeid, uint16_t data) {
std::vector<uint8_t> wbuf(2);
	wbuf[0] = data % 0x100u;
	wbuf[1] = data / 0x100u;
	communicate(classid, instanceid, attributeid, wbuf);
}
template <>
void
XFujikinInterface::send(uint8_t classid, uint8_t instanceid, uint8_t attributeid, uint32_t data) {
std::vector<uint8_t> wbuf(4);
	wbuf[0] = data % 0x100u;
	wbuf[1] = (data / 0x100uL) % 0x100u;
	wbuf[1] = (data / 0x10000uL) % 0x100u;
	wbuf[1] = data / 0x1000000uL;
	communicate(classid, instanceid, attributeid, wbuf);
}
template <typename T>
T
XFujikinInterface::query(uint8_t classid, uint8_t instanceid, uint8_t attributeid) {
}
template <>
uint8_t
XFujikinInterface::query(uint8_t classid, uint8_t instanceid, uint8_t attributeid) {
	std::vector<uint8_t> wbuf(0), rbuf;
		communicate(classid, instanceid, attributeid, wbuf, &rbuf);
		if(rbuf.size() != 1)
			throw XInterfaceError("Fujikin Protocol Wrong Data-Size Error.", __FILE__, __LINE__);
		return rbuf[0];
}
template <>
uint16_t
XFujikinInterface::query(uint8_t classid, uint8_t instanceid, uint8_t attributeid) {
	std::vector<uint8_t> wbuf(0), rbuf;
		communicate(classid, instanceid, attributeid, wbuf, &rbuf);
		if(rbuf.size() != 2)
			throw XInterfaceError("Fujikin Protocol Wrong Data-Size Error.", __FILE__, __LINE__);
		return rbuf[0] + (uint16_t)rbuf[1] * 0x100u;
}
template <>
XString
XFujikinInterface::query(uint8_t classid, uint8_t instanceid, uint8_t attributeid) {
	std::vector<uint8_t> wbuf(0), rbuf;
		communicate(classid, instanceid, attributeid, wbuf, &rbuf);
		rbuf.push_back(0); //null
		return reinterpret_cast<char *>( &rbuf[0]);
}
void
XFujikinInterface::communicate(uint8_t classid, uint8_t instanceid, uint8_t attributeid,
	const std::vector<uint8_t> &data, std::vector<uint8_t> *response) {
    for(int retry = 0; ; retry++) {
        try {
            communicate_once(classid, instanceid, attributeid, data, response);
            break;
        }
        catch (XInterfaceError &e) {
            if(retry < 1) {
                e.print("Retrying after an error: ");
                msecsleep(20);
                continue;
            }
            throw e;
        }
    }
}
void
XFujikinInterface::communicate_once(uint8_t classid, uint8_t instanceid, uint8_t attributeid,
    const std::vector<uint8_t> &data, std::vector<uint8_t> *response) {

    bool write = !response;
    std::vector<uint8_t> buf;
    buf.push_back( ***address());
    buf.push_back(STX);
    uint8_t commandcode = write ? 0x81 : 0x80;
    buf.push_back(commandcode);
    buf.push_back(3 + data.size());
    buf.push_back(classid);
    buf.push_back(instanceid);
    buf.push_back(attributeid);
    for(auto it = data.begin(); it != data.end(); ++it)
        buf.push_back( *it);
    buf.push_back(0); //pad
    uint8_t checksum = 0;
    for(auto it = buf.begin() + 1; it != buf.end(); ++it)
        checksum += *it; //from STX to data.back.
    buf.push_back(checksum);

    auto port = m_openedPort;
    XScopedLock<XMutex> lock(s_lock); //!\todo better to use port-by-port lock.
    msecsleep(1);
    port->writeTo(this, reinterpret_cast<char*>( &buf[0]), buf.size());
    port->receiveFrom(this, 1);
    switch(port->buffer()[0]) {
    case ACK:
        break;
    case NAK:
    default:
        throw XInterfaceError(
            formatString("Fujikin Protocol Command Error ret=%x.", (unsigned int)port->buffer()[0]),
            __FILE__, __LINE__);
    }
    if(write) {
        port->receiveFrom(this, 1);
        switch(port->buffer()[0]) {
        case ACK:
            break;
        case NAK:
        default:
            throw XInterfaceError(
                formatString("Fujikin Protocol Command Error ret=%x.", (unsigned int)port->buffer()[0]),
                __FILE__, __LINE__);
        }
    }
    else {
        port->receiveFrom(this, 4);
        if((port->buffer()[0] != 0) || (port->buffer()[1] != STX))
            throw XInterfaceError(
                formatString("Fujikin Protocol Command Error ret=%4s.", (const char*)&port->buffer()[0]),
                __FILE__, __LINE__);
        int len = port->buffer()[3];
        uint8_t checksum = 0;
        for(auto it = port->buffer().begin(); it != port->buffer().end(); ++it)
            checksum += *it;
        port->receiveFrom(this, len + 2);
//		if((master->buffer()[0] != classid) || (master->buffer()[1] != instanceid) || (master->buffer()[2] != attributeid))
//			throw XInterfaceError("Fujikin Protocol Format Error.", __FILE__, __LINE__);
        if((port->buffer()[len] != 0)) //pad
            throw XInterfaceError("Fujikin Protocol Format Error.", __FILE__, __LINE__);
        for(auto it = port->buffer().begin(); it != port->buffer().end(); ++it)
            checksum += *it;
        checksum -= port->buffer().back() * 2;
        if(checksum != 0)
            throw XInterfaceError("Fujikin Protocol Check-Sum Error.", __FILE__, __LINE__);
        response->resize(len - 3);
        for(int i = 0; i < response->size(); ++i) {
            response->at(i) = port->buffer()[i + 3];
        }
    }
}
