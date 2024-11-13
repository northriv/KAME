#include "fujikininterface.h"

#include "serial.h"

class XFujikinProtocolPort : public XAddressedPort<XSerialPort> {
public:
    XFujikinProtocolPort(XCharInterface *interface) : XAddressedPort<XSerialPort>(interface) {}
    virtual ~XFujikinProtocolPort() {}

    virtual void sendTo(XCharInterface *intf, const char *str) override {send(str);}
    virtual void writeTo(XCharInterface *intf, const char *sendbuf, int size) override {write(sendbuf, size);}
    virtual void receiveFrom(XCharInterface *intf) override {receive();}
    virtual void receiveFrom(XCharInterface *intf, unsigned int length) override {receive(length);}
};

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
    close();
    shared_ptr<XPort> port = std::make_shared<XFujikinProtocolPort>(this);
    port->setEOS(eos().c_str());
    openPort(port);
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

    XScopedLock<XInterface> lock( *this);
    msecsleep(1);
    this->write(reinterpret_cast<char*>( &buf[0]), buf.size());
    receive(1);
    switch(buffer()[0]) {
    case ACK:
        break;
    case NAK:
    default:
        throw XInterfaceError(
            formatString("Fujikin Protocol Command Error ret=%x.", (unsigned int)buffer()[0]),
            __FILE__, __LINE__);
    }
    if(write) {
        receive(1);
        switch(buffer()[0]) {
        case ACK:
            break;
        case NAK:
        default:
            throw XInterfaceError(
                formatString("Fujikin Protocol Command Error ret=%x.", (unsigned int)buffer()[0]),
                __FILE__, __LINE__);
        }
    }
    else {
        receive(4);
        if((buffer()[0] != 0) || (buffer()[1] != STX))
            throw XInterfaceError(
                formatString("Fujikin Protocol Command Error ret=%4s.", (const char*)&buffer()[0]),
                __FILE__, __LINE__);
        int len = buffer()[3];
        uint8_t checksum = 0;
        for(auto it = buffer().begin(); it != buffer().end(); ++it)
            checksum += *it;
        receive(len + 2);
//		if((master->buffer()[0] != classid) || (master->buffer()[1] != instanceid) || (master->buffer()[2] != attributeid))
//			throw XInterfaceError("Fujikin Protocol Format Error.", __FILE__, __LINE__);
        if((buffer()[len] != 0)) //pad
            throw XInterfaceError("Fujikin Protocol Format Error.", __FILE__, __LINE__);
        for(auto it = buffer().begin(); it != buffer().end(); ++it)
            checksum += *it;
        checksum -= buffer().back() * 2;
        if(checksum != 0)
            throw XInterfaceError("Fujikin Protocol Check-Sum Error.", __FILE__, __LINE__);
        response->resize(len - 3);
        for(int i = 0; i < response->size(); ++i) {
            response->at(i) = buffer()[i + 3];
        }
    }
}
