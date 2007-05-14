/***************************************************************************
		Copyright (C) 2002-2007 Kentaro Kitagawa
		                   kitag@issp.u-tokyo.ac.jp
		
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

class XPrimaryDriver : public XDriver
{
	XNODE_OBJECT
protected:
	XPrimaryDriver(const char *name, bool runtime, 
				   const shared_ptr<XScalarEntryList> &scalarentries,
				   const shared_ptr<XInterfaceList> &interfaces,
				   const shared_ptr<XThermometerList> &thermometers,
				   const shared_ptr<XDriverList> &drivers);
public:
	virtual ~XPrimaryDriver() {}
  
	//! show all forms belonging to driver
	virtual void showForms() = 0;
  
	template <typename tVar>
	inline static void push(tVar, std::vector<char> &buf);

	template <typename tVar>
	inline static tVar pop(std::vector<char>::iterator &it);
  
	virtual const shared_ptr<XRecordDependency> dependency() const 
	{return shared_ptr<XRecordDependency>();}


	//! Shut down your threads, unconnect GUI, and deactivate signals
	//! this may be called even if driver has already stopped.
	//! This should not cause an exception.
	virtual void stop() = 0;  
protected:
	//! Start up your threads, connect GUI, and activate signals
	//! This should not cause an exception.
	virtual void start() = 0;
	//! Be called for closing interfaces.
	//! This should not cause an exception.
	virtual void afterStop() = 0;  
    
	//! this is called when raw is written 
	//! unless dependency is broken
	//! convert raw to record
	//! \sa analyze()
	virtual void analyzeRaw() throw (XRecordError&) = 0;
  
	//! clear thread-local raw buffer.
	void clearRaw() {rawData().clear();}
	//! will call analyzeRaw() if dependency.
	//! unless dependency is broken.
	//! \arg time_awared time when a visible phenomenon started
	//! \arg time_recorded usually pass \p XTime::now()
	//! \sa timeAwared()
	//! \sa time()
	void finishWritingRaw(const XTime &time_awared, const XTime &time_recorded);
	//! raw data. Thread-Local storaged.
	std::vector<char> &rawData() {return *s_tlRawData;}

	//! These are FIFO (fast in fast out)
	//! push raw data to raw record
	template <typename tVar>
	inline void push(tVar);
	//! read raw record
	template <typename tVar>
	inline tVar pop() throw (XBufferUnderflowRecordError&);

	std::vector<char>::iterator& rawDataPopIterator() {return *s_tl_pop_it;}
private:
	friend class XRawStreamRecordReader;
	friend class XRawStreamRecorder;

	//! raw data
	static XThreadLocal<std::vector<char> > s_tlRawData;
	typedef std::vector<char>::iterator RawData_it;
	static XThreadLocal<RawData_it> s_tl_pop_it;
  
	inline static void _push_char(char, std::vector<char> &buf);
	inline static void _push_short(short, std::vector<char> &buf);
	inline static void _push_int32(int32_t, std::vector<char> &buf);
	inline static void _push_double(double, std::vector<char> &buf);
	inline static char _pop_char(std::vector<char>::iterator &it);
	inline static short _pop_short(std::vector<char>::iterator &it);
	inline static int32_t _pop_int32(std::vector<char>::iterator &it);
	inline static double _pop_double(std::vector<char>::iterator &it);
};

inline void
XPrimaryDriver::_push_char(char x, std::vector<char> &buf) {
    buf.push_back(x);
}
inline void
XPrimaryDriver::_push_short(short x, std::vector<char> &buf) {
    short y = x;
    char *p = reinterpret_cast<char *>(&y);
#ifdef __BIG_ENDIAN__
    for(char *z = p + sizeof(x) - 1; z >= p; z--) {
	#else
		for(char *z = p; z < p + sizeof(x); z++) {
		#endif
			buf.push_back(*z);
		}
	}
	inline void
		XPrimaryDriver::_push_int32(int32_t x, std::vector<char> &buf) {
		int32_t y = x;
		char *p = reinterpret_cast<char *>(&y);
	#ifdef __BIG_ENDIAN__
		for(char *z = p + sizeof(x) - 1; z >= p; z--) {
		#else
			for(char *z = p; z < p + sizeof(x); z++) {
			#endif
				buf.push_back(*z);
			}
		}
		inline void
			XPrimaryDriver::_push_double(double x, std::vector<char> &buf) {
			double y = x;
			char *p = reinterpret_cast<char *>(&y);
		#ifdef __BIG_ENDIAN__
			for(char *z = p + sizeof(x) - 1; z >= p; z--) {
			#else
				for(char *z = p; z < p + sizeof(x); z++) {
				#endif
					buf.push_back(*z);
				}
			}
			inline char
				XPrimaryDriver::_pop_char(std::vector<char>::iterator &it) {
				char c = *(it++);
				return c;
			}
			inline short
				XPrimaryDriver::_pop_short(std::vector<char>::iterator &it) {
				union {
					short x;
					char p[sizeof(short)];
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
					XPrimaryDriver::_pop_int32(std::vector<char>::iterator &it) {
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
						XPrimaryDriver::_pop_double(std::vector<char>::iterator &it) {
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
							inline char XPrimaryDriver::pop() throw (XBufferUnderflowRecordError&) {
							if(*s_tl_pop_it == rawData().end()) throw XBufferUnderflowRecordError(__FILE__, __LINE__);
							return _pop_char(*s_tl_pop_it);
						}
						template <>
							inline unsigned char XPrimaryDriver::pop() throw (XBufferUnderflowRecordError&) {
							if(*s_tl_pop_it == rawData().end()) throw XBufferUnderflowRecordError(__FILE__, __LINE__);
							return static_cast<unsigned char>(_pop_char(*s_tl_pop_it));
						}
						template <>
							inline short XPrimaryDriver::pop() throw (XBufferUnderflowRecordError&) {
							if(*s_tl_pop_it == rawData().end()) throw XBufferUnderflowRecordError(__FILE__, __LINE__);
							return _pop_short(*s_tl_pop_it);
						}
						template <>
							inline unsigned short XPrimaryDriver::pop() throw (XBufferUnderflowRecordError&) {
							if(*s_tl_pop_it == rawData().end()) throw XBufferUnderflowRecordError(__FILE__, __LINE__);
							return static_cast<unsigned short>(_pop_short(*s_tl_pop_it));
						}
						template <>
							inline int32_t XPrimaryDriver::pop() throw (XBufferUnderflowRecordError&) {
							if(*s_tl_pop_it == rawData().end()) throw XBufferUnderflowRecordError(__FILE__, __LINE__);
							return _pop_int32(*s_tl_pop_it);
						}
						template <>
							inline uint32_t XPrimaryDriver::pop() throw (XBufferUnderflowRecordError&) {
							if(*s_tl_pop_it == rawData().end()) throw XBufferUnderflowRecordError(__FILE__, __LINE__);
							return static_cast<uint32_t>(_pop_int32(*s_tl_pop_it));
						}
						template <>
							inline float XPrimaryDriver::pop() throw (XBufferUnderflowRecordError&) {
							if(*s_tl_pop_it == rawData().end()) throw XBufferUnderflowRecordError(__FILE__, __LINE__);
							union {
								int32_t x;
								float y;
							} uni;
							C_ASSERT(sizeof(uni.x) == sizeof(uni.y));
							uni.x = _pop_int32(*s_tl_pop_it);
							return uni.y;
						}
						template <>
							inline double XPrimaryDriver::pop() throw (XBufferUnderflowRecordError&) {
							if(*s_tl_pop_it == rawData().end()) throw XBufferUnderflowRecordError(__FILE__, __LINE__);
							C_ASSERT(sizeof(double) == 8);
							return _pop_double(*s_tl_pop_it);
						}

						template <>
							inline char XPrimaryDriver::pop(std::vector<char>::iterator &it) {
							return _pop_char(it);
						}
						template <>
							inline unsigned char XPrimaryDriver::pop(std::vector<char>::iterator &it) {
							return static_cast<unsigned char>(_pop_char(it));
						}
						template <>
							inline short XPrimaryDriver::pop(std::vector<char>::iterator &it) {
							return _pop_short(it);
						}
						template <>
							inline unsigned short XPrimaryDriver::pop(std::vector<char>::iterator &it) {
							return static_cast<unsigned short>(_pop_short(it));
						}
						template <>
							inline int32_t XPrimaryDriver::pop(std::vector<char>::iterator &it) {
							return _pop_int32(it);
						}
						template <>
							inline uint32_t XPrimaryDriver::pop(std::vector<char>::iterator &it) {
							return static_cast<uint32_t>(_pop_int32(it));
						}
						template <>
							inline float XPrimaryDriver::pop(std::vector<char>::iterator &it) {
							union {
								int32_t x;
								float y;
							} uni;
							C_ASSERT(sizeof(uni.x) == sizeof(uni.y));
							uni.x = _pop_int32(it);
							return uni.y;
						}
						template <>
							inline double XPrimaryDriver::pop(std::vector<char>::iterator &it) {
							return _pop_double(it);
						}
						template <>
							inline void XPrimaryDriver::push(char x) {
							_push_char(x, rawData());
						}
						template <>
							inline void XPrimaryDriver::push(unsigned char x) {
							_push_char(static_cast<char>(x), rawData());
						}
						template <>
							inline void XPrimaryDriver::push(short x) {
							_push_short(x, rawData());
						}
						template <>
							inline void XPrimaryDriver::push(unsigned short x) {
							_push_short(static_cast<short>(x), rawData());
						}
						template <>
							inline void XPrimaryDriver::push(int32_t x) {
							_push_int32(x, rawData());
						}
						template <>
							inline void XPrimaryDriver::push(uint32_t x) {
							_push_int32(static_cast<int32_t>(x), rawData());
						}
						template <>
							inline void XPrimaryDriver::push(float f) {
							union {
								int32_t x;
								float y;
							} uni;
							C_ASSERT(sizeof(uni.x) == sizeof(uni.y));
							uni.y = f;
							_push_int32(uni.x, rawData());
						}
						template <>
							inline void XPrimaryDriver::push(double x) {
							_push_double(x, rawData());
						}

						template <>
							inline void XPrimaryDriver::push(char x, std::vector<char> &buf) {
							_push_char(x, buf);
						}
						template <>
							inline void XPrimaryDriver::push(unsigned char x, std::vector<char> &buf) {
							_push_char(static_cast<char>(x), buf);
						}
						template <>
							inline void XPrimaryDriver::push(short x, std::vector<char> &buf) {
							_push_short(x, buf);
						}
						template <>
							inline void XPrimaryDriver::push(unsigned short x, std::vector<char> &buf) {
							_push_short(static_cast<short>(x), buf);
						}
						template <>
							inline void XPrimaryDriver::push(int32_t x, std::vector<char> &buf) {
							_push_int32(x, buf);
						}
						template <>
							inline void XPrimaryDriver::push(uint32_t x, std::vector<char> &buf) {
							_push_int32(static_cast<int32_t>(x), buf);
						}
						template <>
							inline void XPrimaryDriver::push(float f, std::vector<char> &buf) {
							union {
								int32_t x;
								float y;
							} uni;
							C_ASSERT(sizeof(uni.x) == sizeof(uni.y));
							uni.y =  f;
							_push_int32(uni.x, buf);
						}
						template <>
							inline void XPrimaryDriver::push(double x, std::vector<char> &buf) {
							_push_double(x, buf);
						}

					#endif /*PRIMARYDRIVER_H_*/
