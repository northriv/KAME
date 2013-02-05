#include "fujikininterface.h"

std::deque<weak_ptr<XFujikinInterface> > XFujikinInterface::s_masters;
XMutex XFujikinInterface::s_lock;

XFujikinInterface::XFujikinInterface(const char *name, bool runtime, const shared_ptr<XDriver> &driver) :
 XCharInterface(name, runtime, driver),
 m_openedCount(0) {
    setEOS("");
	setSerialBaudRate(38400);
	setSerialStopBits(1);
}

XFujikinInterface::~XFujikinInterface() {
}
void
XFujikinInterface::open() throw (XInterfaceError &) {
	{
		XScopedLock<XFujikinInterface> lock( *this);
		m_master = dynamic_pointer_cast<XFujikinInterface>(shared_from_this());
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
XFujikinInterface::close() throw (XInterfaceError &) {
	XScopedLock<XFujikinInterface> lock( *this);
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
void
XFujikinInterface::communicate(uint8_t classid, uint8_t instanceid, uint8_t attributeid,
	const std::vector<uint8_t> &data, std::vector<uint8_t> *response) {
	bool write = !response;
	std::vector<uint8_t> buf;
	buf.push_back(0); //master
	buf.push_back(STX);
	uint8_t commandcode = write ? 0x81 : 0x80;
	buf.push_back(commandcode);
	buf.push_back(9 + data.size());
	buf.push_back(classid);
	buf.push_back(instanceid);
	buf.push_back(attributeid);
	for(auto it = data.begin(); it != data.end(); ++it)
		buf.push_back( *it);
	buf.push_back(0); //pad
	uint8_t checksum = 0;
	for(auto it = buf.begin(); it != buf.end(); ++it)
		checksum += *it;
	buf.push_back(checksum);

	auto master = m_master;
	master->write( reinterpret_cast<char*>( &buf[0]), buf.size());
	master->receive(1);
	switch(master->buffer()[0]) {
	case ACK:
		break;
	case NAK:
	default:
		throw XInterfaceError("Fujikin Protocol Communication Error.", __FILE__, __LINE__);
	}
	if(write) {
		master->receive(1);
		switch(master->buffer()[0]) {
		case ACK:
			break;
		case NAK:
		default:
			throw XInterfaceError("Fujikin Protocol Command Error.", __FILE__, __LINE__);
		}
	}
	else {
		master->receive(4);
		if((master->buffer()[0] != 0) || (master->buffer()[1] != STX) || (master->buffer()[2] != commandcode))
			throw XInterfaceError("Fujikin Protocol Format Error.", __FILE__, __LINE__);
		int len = master->buffer()[3];
		uint8_t checksum = 0;
		for(auto it = master->buffer().begin(); it != master->buffer().end(); ++it)
			checksum += *it;
		master->receive(len - 4);
		if((master->buffer()[0] != classid) || (master->buffer()[1] != instanceid) || (master->buffer()[2] != attributeid))
			throw XInterfaceError("Fujikin Protocol Format Error.", __FILE__, __LINE__);
		if((master->buffer()[len - 6] != 0))
			throw XInterfaceError("Fujikin Protocol Format Error.", __FILE__, __LINE__);
		for(auto it = master->buffer().begin(); it != master->buffer().end(); ++it)
			checksum += *it;
		checksum -= master->buffer().back() * 2;
		if(checksum != 0)
			throw XInterfaceError("Fujikin Protocol Check-Sum Error.", __FILE__, __LINE__);
		response->resize(len - 9);
		for(int i = 0; i < response->size(); ++i) {
			response->at(i) = master->buffer()[i + 3];
		}
	}
}