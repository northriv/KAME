/***************************************************************************
		Copyright (C) 2002-2013 Kentaro Kitagawa
		                   kitag@kochi-u.ac.jp
		
		This program is free software; you can redistribute it and/or
		modify it under the terms of the GNU Library General Public
		License as published by the Free Software Foundation; either
		version 2 of the License, or (at your option) any later version.
		
		You should have received a copy of the GNU Library General 
		Public License and a list of authors along with this program; 
		see the files COPYING and AUTHORS.
***************************************************************************/
#ifndef PRIMARYDRIVER_H_
#define PRIMARYDRIVER_H_

#include "driver.h"
#include "interface.h"

class XPrimaryDriver : public XDriver {
public:
	XPrimaryDriver(const char *name, bool runtime, Transaction &tr_meas, const shared_ptr<XMeasure> &meas);
	virtual ~XPrimaryDriver() {}
  
	//! Shows all forms belonging to driver
	virtual void showForms() = 0;
  
	//! Shuts down your threads, unconnects GUI, and deactivates signals.\n
	//! This function may be called even if driver has already stopped.
	//! This should not cause an exception.
	virtual void stop() = 0;

private:
	friend class XRawStreamRecordReader;
	friend class XRawStreamRecorder;
protected:
	//! Starts up your threads, connects GUI, and activates signals.
	//! This function should not cause an exception.
	virtual void start() = 0;
	//! Be called for closing interfaces.
	//! This function should not cause an exception.
	virtual void closeInterface() = 0;

	//! These are FIFO.
	struct RawData : public std::vector<char> {
		//! Pushes raw data to raw record
		//! Use signed/unsigned char, int16_t(16bit), and int32_t for integers.
		//! IEEE 754 float and double for floting point numbers.
		//! Little endian bytes will be stored into thread-local \sa rawData().
		//! \sa pop(), rawData()
		template <typename tVar>
		inline void push(tVar);
	private:
		inline void push_char(char);
		inline void push_int16_t(int16_t);
		inline void push_int32_t(int32_t);
		inline void push_double(double);
	};

	struct RawDataReader {
		typedef std::vector<char>::const_iterator const_iterator;
		//! reads raw record
		//! \sa push(), rawData()
		template <typename tVar>
		inline tVar pop() throw (XBufferUnderflowRecordError&);

		const_iterator begin() const {return m_data.begin();}
		const_iterator end() const {return m_data.end();}
		unsigned int size() const {return m_data.size();}
		const std::vector<char> &data() const {return m_data;}
		const_iterator &popIterator() {return it;}
	private:
		friend class XPrimaryDriver;
		friend class XRawStreamRecordReader;
		RawDataReader(const std::vector<char> &data) : m_data(data) {it = data.begin();}
		RawDataReader();
		const_iterator it;
		const std::vector<char> &m_data;
		inline char pop_char();
		inline int16_t pop_int16_t();
		inline int32_t pop_int32_t();
		inline double pop_double();
	};

	//! This function will be called when raw data are written.
	//! Implement this function to convert the raw data to the record (Payload).
	//! \sa analyze()
	virtual void analyzeRaw(RawDataReader &reader, Transaction &tr) throw (XRecordError&) = 0;

	//! will call analyzeRaw()
	//! \param rawdata the data being processed.
	//! \param time_awared time when a visible phenomenon started
	//! \param time_recorded usually pass \p XTime::now()
	//! \sa Payload::timeAwared()
	//! \sa Payload::time()
	void finishWritingRaw(const shared_ptr<const RawData> &rawdata,
		const XTime &time_awared, const XTime &time_recorded);
public:
	struct Payload : public XDriver::Payload {
		const RawData &rawData() const {return *m_rawData;}
	private:
		friend class XPrimaryDriver;
		shared_ptr<const RawData> m_rawData;
	};
};

inline void
XPrimaryDriver::RawData::push_char(char x) {
    push_back(x);
}
inline void
XPrimaryDriver::RawData::push_int16_t(int16_t x) {
    int16_t y = x;
    char *p = reinterpret_cast<char *>(&y);
#ifdef __BIG_ENDIAN__
    for(char *z = p + sizeof(x) - 1; z >= p; z--) {
#else
	for(char *z = p; z < p + sizeof(x); z++) {
#endif
		push_back( *z);
	}
}
inline void
XPrimaryDriver::RawData::push_int32_t(int32_t x) {
	int32_t y = x;
	char *p = reinterpret_cast<char *>(&y);
#ifdef __BIG_ENDIAN__
	for(char *z = p + sizeof(x) - 1; z >= p; z--) {
#else
	for(char *z = p; z < p + sizeof(x); z++) {
#endif
		push_back( *z);
	}
}
inline void
XPrimaryDriver::RawData::push_double(double x) {
	static_assert(sizeof(double) == 8, "Not 8-byte sized double"); // for compatibility.
	double y = x;
	char *p = reinterpret_cast<char *>( &y);
#ifdef __BIG_ENDIAN__
	for(char *z = p + sizeof(x) - 1; z >= p; z--) {
#else
	for(char *z = p; z < p + sizeof(x); z++) {
#endif
		push_back( *z);
	}
}
inline char
XPrimaryDriver::RawDataReader::pop_char() {
	char c = *(it++);
	return c;
}
inline int16_t
XPrimaryDriver::RawDataReader::pop_int16_t() {
	union {
		int16_t x;
		char p[sizeof(int16_t)];
	} uni;
#ifdef __BIG_ENDIAN__
	for(char *z = uni.p + sizeof(uni) - 1; z >= uni.p; z--) {
#else
	for(char *z = uni.p; z < uni.p + sizeof(uni); z++) {
#endif
		*z = *(it++);
	}
	return uni.x;
}
inline int32_t
XPrimaryDriver::RawDataReader::pop_int32_t() {
	union {
		int32_t x;
		char p[sizeof(int32_t)];
	} uni;
#ifdef __BIG_ENDIAN__
	for(char *z = uni.p + sizeof(uni) - 1; z >= uni.p; z--) {
#else
	for(char *z = uni.p; z < uni.p + sizeof(uni); z++) {
#endif
		*z = *(it++);
	}
	return uni.x;
}
inline double
XPrimaryDriver::RawDataReader::pop_double() {
	union {
		double x;
		char p[sizeof(double)];
	} uni;
#ifdef __BIG_ENDIAN__
	for(char *z = uni.p + sizeof(uni) - 1; z >= uni.p; z--) {
#else
	for(char *z = uni.p; z < uni.p + sizeof(uni); z++) {
#endif
		*z = *(it++);
	}
	return uni.x;
}

template <>
inline char XPrimaryDriver::RawDataReader::pop() throw (XBufferUnderflowRecordError&) {
	if(it + sizeof(char) > end()) throw XBufferUnderflowRecordError(__FILE__, __LINE__);
	return pop_char();
}
template <>
inline unsigned char XPrimaryDriver::RawDataReader::pop() throw (XBufferUnderflowRecordError&) {
	if(it + sizeof(char) > end()) throw XBufferUnderflowRecordError(__FILE__, __LINE__);
	return static_cast<unsigned char>(pop_char());
}
template <>
inline int16_t XPrimaryDriver::RawDataReader::pop() throw (XBufferUnderflowRecordError&) {
	if(it + sizeof(int16_t) > end()) throw XBufferUnderflowRecordError(__FILE__, __LINE__);
	return pop_int16_t();
}
template <>
inline uint16_t XPrimaryDriver::RawDataReader::pop() throw (XBufferUnderflowRecordError&) {
	if(it + sizeof(int16_t) > end()) throw XBufferUnderflowRecordError(__FILE__, __LINE__);
	return static_cast<uint16_t>(pop_int16_t());
}
template <>
inline int32_t XPrimaryDriver::RawDataReader::pop() throw (XBufferUnderflowRecordError&) {
	if(it + sizeof(int32_t) > end()) throw XBufferUnderflowRecordError(__FILE__, __LINE__);
	return pop_int32_t();
}
template <>
inline uint32_t XPrimaryDriver::RawDataReader::pop() throw (XBufferUnderflowRecordError&) {
	if(it + sizeof(int32_t) > end()) throw XBufferUnderflowRecordError(__FILE__, __LINE__);
	return static_cast<uint32_t>(pop_int32_t());
}
template <>
inline float XPrimaryDriver::RawDataReader::pop() throw (XBufferUnderflowRecordError&) {
	if(it + sizeof(float) > end()) throw XBufferUnderflowRecordError(__FILE__, __LINE__);
	union {
		int32_t x;
		float y;
	} uni;
	static_assert(sizeof(uni.x) == sizeof(uni.y), "Size mismatch");
	uni.x = pop_int32_t();
	return uni.y;
}
template <>
inline double XPrimaryDriver::RawDataReader::pop() throw (XBufferUnderflowRecordError&) {
	if(it + sizeof(double) > end()) throw XBufferUnderflowRecordError(__FILE__, __LINE__);
	static_assert(sizeof(double) == 8, "Not 8-byte sized double");
	return pop_double();
}

template <>
inline void XPrimaryDriver::RawData::push(char x) {
	push_char(x);
}
template <>
inline void XPrimaryDriver::RawData::push(unsigned char x) {
	push_char(static_cast<char>(x));
}
template <>
inline void XPrimaryDriver::RawData::push(int16_t x) {
	push_int16_t(x);
}
template <>
inline void XPrimaryDriver::RawData::push(uint16_t x) {
	push_int16_t(static_cast<int16_t>(x));
}
template <>
inline void XPrimaryDriver::RawData::push(int32_t x) {
	push_int32_t(x);
}
template <>
inline void XPrimaryDriver::RawData::push(uint32_t x) {
	push_int32_t(static_cast<int32_t>(x));
}
template <>
inline void XPrimaryDriver::RawData::push(float f) {
	union {
		int32_t x;
		float y;
	} uni;
	static_assert(sizeof(uni.x) == sizeof(uni.y), "Size mismatch");
	uni.y = f;
	push_int32_t(uni.x);
}
template <>
inline void XPrimaryDriver::RawData::push(double x) {
	push_double(x);
}

#endif /*PRIMARYDRIVER_H_*/
