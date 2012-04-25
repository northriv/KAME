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
#ifndef CHARINTERFACE_H_
#define CHARINTERFACE_H_

#include "interface.h"

class XPort;
//#include <stdarg.h>

//! Standard interface for character devices. e.g. GPIB, serial port, ....
class XCharInterface : public XInterface {
public:
	XCharInterface(const char *name, bool runtime, const shared_ptr<XDriver> &driver);
	virtual ~XCharInterface() {}

	//! Buffer is Thread-Local-Strage.
	//! Therefore, be careful when you access multi-interfaces in one thread.
	//! \sa XThreadLocal
	const std::vector<char> &buffer() const;
	//! error-check is user's responsibility.
	int scanf(const char *format, ...) const
		__attribute__ ((format(scanf,2,3)));
	double toDouble() const throw (XConvError &);
	int toInt() const throw (XConvError &);
	unsigned int toUInt() const throw (XConvError &);
  
	void send(const XString &str) throw (XCommError &);
	virtual void send(const char *str) throw (XCommError &);
	//! format version of send()
	//! \sa printf()
	void sendf(const char *format, ...) throw (XInterfaceError &)
		__attribute__ ((format(printf,2,3)));
	virtual void write(const char *sendbuf, int size) throw (XCommError &);
	virtual void receive() throw (XCommError &);
	virtual void receive(unsigned int length) throw (XCommError &);
	void query(const XString &str) throw (XCommError &);
	virtual void query(const char *str) throw (XCommError &);
	//! format version of query()
	//! \sa printf()
	void queryf(const char *format, ...) throw (XInterfaceError &)
		__attribute__ ((format(printf,2,3)));
  
	void setEOS(const char *str);
	void setGPIBUseSerialPollOnWrite(bool x) {m_bGPIBUseSerialPollOnWrite = x;}
	void setGPIBUseSerialPollOnRead(bool x) {m_bGPIBUseSerialPollOnRead = x;}
	void setGPIBWaitBeforeWrite(int msec) {m_gpibWaitBeforeWrite = msec;}
	void setGPIBWaitBeforeRead(int msec) {m_gpibWaitBeforeRead = msec;}
	void setGPIBWaitBeforeSPoll(int msec) {m_gpibWaitBeforeSPoll = msec;}
	void setGPIBMAVbit(unsigned char x) {m_gpibMAVbit = x;}
  
	const XString &eos() const {return m_eos;}
	bool gpibUseSerialPollOnWrite() const {return m_bGPIBUseSerialPollOnWrite;}
	bool gpibUseSerialPollOnRead() const {return m_bGPIBUseSerialPollOnRead;}
	int gpibWaitBeforeWrite() const {return m_gpibWaitBeforeWrite;}
	int gpibWaitBeforeRead() const {return m_gpibWaitBeforeRead;}
	int gpibWaitBeforeSPoll() const {return m_gpibWaitBeforeSPoll;}
	unsigned char gpibMAVbit() const {return m_gpibMAVbit;}
	
	void setSerialBaudRate(unsigned int rate) {m_serialBaudRate = rate;}
	void setSerialStopBits(unsigned int bits) {m_serialStopBits = bits;}
	enum {PARITY_NONE = 0, PARITY_ODD = 1, PARITY_EVEN = 2};
	void setSerialParity(unsigned int parity) {m_serialParity = parity;}
	void setSerial7Bits(bool enable) {m_serial7bits = enable;}
  
	unsigned int serialBaudRate() const {return m_serialBaudRate;}
	unsigned int serialStopBits() const {return m_serialStopBits;}
	unsigned int serialParity() const {return m_serialParity;}
	bool serial7Bits() const {return m_serial7Bits;}

	virtual bool isOpened() const {return !!m_xport;}
protected:
	virtual void open() throw (XInterfaceError &);
	//! This can be called even if has already closed.
	virtual void close() throw (XInterfaceError &);
private:
	XString m_eos;
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
	
	shared_ptr<XPort> m_xport;

	//! for scripting
	shared_ptr<XStringNode> m_script_send;
	shared_ptr<XStringNode> m_script_query;
	shared_ptr<XListener> m_lsnOnSendRequested;
	shared_ptr<XListener> m_lsnOnQueryRequested;
	void onSendRequested(const Snapshot &shot, XValueNodeBase *);
	void onQueryRequested(const Snapshot &shot, XValueNodeBase *);
};

class XPort {
public:
	XPort(XCharInterface *interface);
	virtual ~XPort();
	virtual void open() throw (XInterface::XCommError &) = 0;
	virtual void send(const char *str) throw (XInterface::XCommError &) = 0;
	virtual void write(const char *sendbuf, int size) throw (XInterface::XCommError &) = 0;
	virtual void receive() throw (XInterface::XCommError &) = 0;
	virtual void receive(unsigned int length) throw (XInterface::XCommError &) = 0;
	//! Buffer is Thread-Local-Strage.
	//! Therefore, be careful when you access multi-interfaces in one thread.
	//! \sa XThreadLocal
	std::vector<char>& buffer() {return *s_tlBuffer;}
protected:
	static XThreadLocal<std::vector<char> > s_tlBuffer;
	XCharInterface *const m_pInterface;
};

#endif /*CHARINTERFACE_H_*/
