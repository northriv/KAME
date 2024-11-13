/***************************************************************************
        Copyright (C) 2002-2024 Kentaro Kitagawa
		                   kitag@issp.u-tokyo.ac.jp
		
		This program is free software; you can redistribute it and/or
		modify it under the terms of the GNU Library General Public
		License as published by the Free Software Foundation; either
		version 2 of the License, or (at your option) any later version.
		
		You should have received a copy of the GNU Library General 
		Public License and a list of authors along with this program; 
		see the files COPYING and AUTHORS.
***************************************************************************/
#ifndef CHARINTERFACE_H_
#define CHARINTERFACE_H_

#include "interface.h"

//#include <stdarg.h>

class DECLSPEC_SHARED XCustomCharInterface : public XInterface {
public:
    XCustomCharInterface(const char *name, bool runtime, const shared_ptr<XDriver> &driver);
    virtual ~XCustomCharInterface() = default;

    //! Buffer is Thread-Local-Strage.
    //! Therefore, be careful when you access multi-interfaces in one thread.
    //! \sa XThreadLocal
    const std::vector<char> &buffer() const {return *s_tlBuffer;}
    //! error-check is user's responsibility.
    int scanf(const char *format, ...) const
#if defined __GNUC__ || defined __clang__
        __attribute__ ((format(scanf,2,3)))
#endif
    ;
    //! XConvError will be thrown if conversoin fails.
    double toDouble() const;
    int toInt() const;
    unsigned int toUInt() const;
    XString toStr() const;
    XString toStrSimplified() const; //!< returns string white-spaces stripped.

    //! format version of send()
    //! \sa printf()
    //! XInterfaceError will be thrown if IO fails.
    void sendf(const char *format, ...)
#if defined __GNUC__ || defined __clang__
        __attribute__ ((format(printf,2,3)))
#endif
    ;
    //! format version of query()
    //! \sa printf()
    //! XInterfaceError will be thrown if IO fails.
    void queryf(const char *format, ...)
#if defined __GNUC__ || defined __clang__
        __attribute__ ((format(printf,2,3)))
#endif
    ;

    void setEOS(const char *str);
    const XString &eos() const {return m_eos;}

    void query(const XString &str);
    virtual void query(const char *str);
    virtual void send(const char *str) = 0;
    virtual void receive() = 0;

    virtual bool isOpened() const override = 0;

    //! only for XPort and internal use.
    static std::vector<char> &buffer_receive() {return *s_tlBuffer;}
protected:
    //! XInterfaceError will be thrown if IO fails.
    virtual void open() override = 0;
    //! This can be called even if has already closed.
    //! XInterfaceError will be thrown if IO fails.
    virtual void close() override = 0;

private:
    static XThreadLocal<std::vector<char> > s_tlBuffer;
    XString m_eos;
};

class XPort;
//! Standard interface for character devices. e.g. GPIB, serial port, TCP/IP...
class DECLSPEC_SHARED XCharInterface : public XCustomCharInterface {
public:
	XCharInterface(const char *name, bool runtime, const shared_ptr<XDriver> &driver);
    virtual ~XCharInterface() = default;

	void setGPIBUseSerialPollOnWrite(bool x) {m_bGPIBUseSerialPollOnWrite = x;}
	void setGPIBUseSerialPollOnRead(bool x) {m_bGPIBUseSerialPollOnRead = x;}
	void setGPIBWaitBeforeWrite(int msec) {m_gpibWaitBeforeWrite = msec;}
	void setGPIBWaitBeforeRead(int msec) {m_gpibWaitBeforeRead = msec;}
	void setGPIBWaitBeforeSPoll(int msec) {m_gpibWaitBeforeSPoll = msec;}
	void setGPIBMAVbit(unsigned char x) {m_gpibMAVbit = x;}
  
	bool gpibUseSerialPollOnWrite() const {return m_bGPIBUseSerialPollOnWrite;}
	bool gpibUseSerialPollOnRead() const {return m_bGPIBUseSerialPollOnRead;}
	int gpibWaitBeforeWrite() const {return m_gpibWaitBeforeWrite;}
	int gpibWaitBeforeRead() const {return m_gpibWaitBeforeRead;}
	int gpibWaitBeforeSPoll() const {return m_gpibWaitBeforeSPoll;}
	unsigned char gpibMAVbit() const {return m_gpibMAVbit;}
	
	//! These properties should be set before open().
	void setSerialBaudRate(unsigned int rate) {m_serialBaudRate = rate;}
	void setSerialStopBits(unsigned int bits) {m_serialStopBits = bits;}
	enum {PARITY_NONE = 0, PARITY_ODD = 1, PARITY_EVEN = 2};
	void setSerialParity(unsigned int parity) {m_serialParity = parity;}
	void setSerial7Bits(bool enable) {m_serial7Bits = enable;}
	void setSerialFlushBeforeWrite(bool x) {m_serialFlushBeforeWrite = x;}
	void setSerialEOS(const char *str) {m_serialEOS = str;} //!< be overridden by \a setEOS().
	void setSerialHasEchoBack(bool x) {m_serialHasEchoBack = x;}
  
	unsigned int serialBaudRate() const {return m_serialBaudRate;}
	unsigned int serialStopBits() const {return m_serialStopBits;}
	unsigned int serialParity() const {return m_serialParity;}
	bool serial7Bits() const {return m_serial7Bits;}
	bool serialFlushBeforeWrite() const {return m_serialFlushBeforeWrite;}
	const XString &serialEOS() const {return m_serialEOS;}
	bool serialHasEchoBack() const {return m_serialHasEchoBack;}

    virtual void send(const XString &str);
    virtual void send(const char *str) override;
    virtual void write(const char *sendbuf, int size);
    virtual void receive() override;
    virtual void receive(unsigned int length);
    virtual bool isOpened() const override {return !!m_xport;}

    virtual void lock() override; //locks XIterface::m_mutex and port's mutex if needed.
    virtual void unlock() override;
    virtual bool isLocked() const override;

protected:
    virtual void open() override;
	//! This can be called even if has already closed.
    virtual void close() override;

    shared_ptr<XPort> openedPort() const {return m_xport;}
    void openPort(shared_ptr<XPort> p);
private:
    XString m_serialEOS;
	bool m_bGPIBUseSerialPollOnWrite;
	bool m_bGPIBUseSerialPollOnRead;
	int m_gpibWaitBeforeWrite;
	int m_gpibWaitBeforeRead;
	int m_gpibWaitBeforeSPoll;
	unsigned char m_gpibMAVbit; //! don't check if zero
  
	unsigned int m_serialBaudRate;
	unsigned int m_serialStopBits;
	unsigned int m_serialParity;
	bool m_serial7Bits;
	bool m_serialFlushBeforeWrite;
	bool m_serialHasEchoBack;
	
	shared_ptr<XPort> m_xport;

	//! for scripting
	shared_ptr<XStringNode> m_script_send;
	shared_ptr<XStringNode> m_script_query;
    shared_ptr<Listener> m_lsnOnSendRequested;
    shared_ptr<Listener> m_lsnOnQueryRequested;
	void onSendRequested(const Snapshot &shot, XValueNodeBase *);
	void onQueryRequested(const Snapshot &shot, XValueNodeBase *);
};

//! Low-level I/O for XCharInterface
//! \sa XAddressedPort
class XPort : public enable_shared_from_this<XPort> {
public:
    XPort(XCharInterface *interface) {m_portString = ***interface->port();}
    virtual ~XPort() = default;

    //! \return myself, or physical port that has already been opened.
    virtual shared_ptr<XPort> open(const XCharInterface *pInterface) = 0;
    //! Thread-Local-Storage Buffer.
	//! \sa XThreadLocal
    std::vector<char>& buffer() {return XCharInterface::buffer_receive();}
    const XString &portString() const {return m_portString;}
    const XString &eos() const {return m_eos;}
    void setEOS(const char *str) {m_eos = str;}

    virtual bool isAddressedPort() const {return false;} //!< \sa XAddressedPort
    virtual void sendTo(XCharInterface *, const char *str) {send(str);}
    virtual void writeTo(XCharInterface *, const char *sendbuf, int size) {write(sendbuf, size);}
    virtual void receiveFrom(XCharInterface *) {receive();}
    virtual void receiveFrom(XCharInterface *, unsigned int length) {receive(length);}

    virtual void lock() {} //must use XInterface::lock().
    virtual void unlock() {}
    virtual bool isLocked() const {return false;}
protected:
    //! XInterface::XCommError will be thrown if IO fails.
    virtual void send(const char *str) = 0;
    virtual void write(const char *sendbuf, int size) = 0;
    virtual void receive() = 0;
    virtual void receive(unsigned int length) = 0;

    static std::deque<weak_ptr<XPort>> s_openedPorts; //should be guarded by s_mutex.
    static XMutex s_mutex;
private:
    XString m_portString, m_eos;
};

//! \sa XPort, Low-level I/O for XCharInterface, sharing one physical interface. sa GPIB, modbus.
//!
template <class P>
class XAddressedPort : public P {
public:
    XAddressedPort(XCharInterface *interface) : P(interface) {}

    //! \return myself, or physical port that has already been opened.
    virtual shared_ptr<XPort> open(const XCharInterface *pInterface) override {
        XScopedLock<XMutex> lock(P::s_mutex);
        Snapshot shot( *pInterface);
        for(auto it = P::s_openedPorts.begin(); it != P::s_openedPorts.end();) {
            if(auto pt = it->lock()) {
                if(pt->portString() == shot[ pInterface->port()].to_str()) {
                    //The physical port has been already opened.
                    return pt;
                }
                ++it;
            }
            else
                it = P::s_openedPorts.erase(it); //cleans garbage.
        }
        //Opens new port.
        auto p = P::open(pInterface);
        P::s_openedPorts.emplace_back(p);
        return p;
    }

    virtual bool isAddressedPort() const override {return true;}
    virtual void sendTo(XCharInterface *intf, const char *str) override = 0;
    virtual void writeTo(XCharInterface *intf, const char *sendbuf, int size) override = 0;
    virtual void receiveFrom(XCharInterface *intf) override = 0;
    virtual void receiveFrom(XCharInterface *intf, unsigned int length) override = 0;

    virtual void lock() override {m_mutex.lock();}
    virtual void unlock() override {m_mutex.unlock();}
    virtual bool isLocked() const {return m_mutex.isLockedByCurrentThread();}
private:
    XRecursiveMutex m_mutex;
};

#endif /*CHARINTERFACE_H_*/
