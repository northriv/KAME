/***************************************************************************
		Copyright (C) 2002-2007 Kentaro Kitagawa
		                   kitagawa@scphys.kyoto-u.ac.jp
		
		This program is free software; you can redistribute it and/or
		modify it under the terms of the GNU Library General Public
		License as published by the Free Software Foundation; either
		version 2 of the License, or (at your option) any later version.
		
		You should have received a copy of the GNU Library General 
		Public License and a list of authors along with this program; 
		see the files COPYING and AUTHORS.
 ***************************************************************************/
#include "primarydriver.h"
#include <klocale.h>

XThreadLocal<std::vector<char> > XPrimaryDriver::s_tlRawData;
XThreadLocal<XPrimaryDriver::RawData_it> XPrimaryDriver::s_tl_pop_it;

XPrimaryDriver::XPrimaryDriver(const char *name, bool runtime, 
   const shared_ptr<XScalarEntryList> &scalarentries,
   const shared_ptr<XInterfaceList> &interfaces,
   const shared_ptr<XThermometerList> &thermometers,
   const shared_ptr<XDriverList> &drivers) :
    XDriver(name, runtime, scalarentries, interfaces, thermometers, drivers)
{
}

void
XPrimaryDriver::finishWritingRaw(
    const XTime &time_awared, const XTime &time_recorded_org)
{
    XTime time_recorded = time_recorded_org;
	bool skipped = false;
    startRecording();
    if(time_recorded) {
	    *s_tl_pop_it = rawData().begin();
	    try {
	        analyzeRaw();
	    }
	    catch (XSkippedRecordError&) {
	    	skipped = true;
	    }
	    catch (XRecordError& e) {
	         time_recorded = XTime(); //record is invalid
	         e.print(getLabel() + ": " + KAME::i18n("Record Error, because "));
	    }
    }
    if(skipped)
    	abortRecording();
	else {
	    finishRecordingNReadLock(time_awared, time_recorded);
	    visualize();
	    readUnlockRecord();
	}
}

void
XPrimaryDriver::_push_char(char x, std::vector<char> &buf) {
    buf.push_back(x);
}
void
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
void
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
void
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
char
XPrimaryDriver::_pop_char(std::vector<char>::iterator &it) {
    char c = *(it++);
    return c;
}
short
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
int32_t
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
double
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
char XPrimaryDriver::pop() throw (XBufferUnderflowRecordError&) {
    if(*s_tl_pop_it == rawData().end()) throw XBufferUnderflowRecordError(__FILE__, __LINE__);
    return _pop_char(*s_tl_pop_it);
}
template <>
unsigned char XPrimaryDriver::pop() throw (XBufferUnderflowRecordError&) {
    if(*s_tl_pop_it == rawData().end()) throw XBufferUnderflowRecordError(__FILE__, __LINE__);
    return static_cast<unsigned char>(_pop_char(*s_tl_pop_it));
}
template <>
short XPrimaryDriver::pop() throw (XBufferUnderflowRecordError&) {
    if(*s_tl_pop_it == rawData().end()) throw XBufferUnderflowRecordError(__FILE__, __LINE__);
    return _pop_short(*s_tl_pop_it);
}
template <>
unsigned short XPrimaryDriver::pop() throw (XBufferUnderflowRecordError&) {
    if(*s_tl_pop_it == rawData().end()) throw XBufferUnderflowRecordError(__FILE__, __LINE__);
    return static_cast<unsigned short>(_pop_short(*s_tl_pop_it));
}
template <>
int32_t XPrimaryDriver::pop() throw (XBufferUnderflowRecordError&) {
    if(*s_tl_pop_it == rawData().end()) throw XBufferUnderflowRecordError(__FILE__, __LINE__);
    return _pop_int32(*s_tl_pop_it);
}
template <>
uint32_t XPrimaryDriver::pop() throw (XBufferUnderflowRecordError&) {
    if(*s_tl_pop_it == rawData().end()) throw XBufferUnderflowRecordError(__FILE__, __LINE__);
    return static_cast<uint32_t>(_pop_int32(*s_tl_pop_it));
}
template <>
float XPrimaryDriver::pop() throw (XBufferUnderflowRecordError&) {
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
double XPrimaryDriver::pop() throw (XBufferUnderflowRecordError&) {
    if(*s_tl_pop_it == rawData().end()) throw XBufferUnderflowRecordError(__FILE__, __LINE__);
    C_ASSERT(sizeof(double) == 8);
    return _pop_double(*s_tl_pop_it);
}

template <>
char XPrimaryDriver::pop(std::vector<char>::iterator &it) {
    return _pop_char(it);
}
template <>
unsigned char XPrimaryDriver::pop(std::vector<char>::iterator &it) {
    return static_cast<unsigned char>(_pop_char(it));
}
template <>
short XPrimaryDriver::pop(std::vector<char>::iterator &it) {
    return _pop_short(it);
}
template <>
unsigned short XPrimaryDriver::pop(std::vector<char>::iterator &it) {
    return static_cast<unsigned short>(_pop_short(it));
}
template <>
int32_t XPrimaryDriver::pop(std::vector<char>::iterator &it) {
    return _pop_int32(it);
}
template <>
uint32_t XPrimaryDriver::pop(std::vector<char>::iterator &it) {
    return static_cast<uint32_t>(_pop_int32(it));
}
template <>
float XPrimaryDriver::pop(std::vector<char>::iterator &it) {
    union {
        int32_t x;
        float y;
    } uni;
    C_ASSERT(sizeof(uni.x) == sizeof(uni.y));
    uni.x = _pop_int32(it);
    return uni.y;
}
template <>
double XPrimaryDriver::pop(std::vector<char>::iterator &it) {
    return _pop_double(it);
}
template <>
void XPrimaryDriver::push(char x) {
    _push_char(x, rawData());
}
template <>
void XPrimaryDriver::push(unsigned char x) {
    _push_char(static_cast<char>(x), rawData());
}
template <>
void XPrimaryDriver::push(short x) {
    _push_short(x, rawData());
}
template <>
void XPrimaryDriver::push(unsigned short x) {
    _push_short(static_cast<short>(x), rawData());
}
template <>
void XPrimaryDriver::push(int32_t x) {
    _push_int32(x, rawData());
}
template <>
void XPrimaryDriver::push(uint32_t x) {
    _push_int32(static_cast<int32_t>(x), rawData());
}
template <>
void XPrimaryDriver::push(float f) {
    union {
        int32_t x;
        float y;
    } uni;
    C_ASSERT(sizeof(uni.x) == sizeof(uni.y));
    uni.y = f;
    _push_int32(uni.x, rawData());
}
template <>
void XPrimaryDriver::push(double x) {
    _push_double(x, rawData());
}

template <>
void XPrimaryDriver::push(char x, std::vector<char> &buf) {
    _push_char(x, buf);
}
template <>
void XPrimaryDriver::push(unsigned char x, std::vector<char> &buf) {
    _push_char(static_cast<char>(x), buf);
}
template <>
void XPrimaryDriver::push(short x, std::vector<char> &buf) {
    _push_short(x, buf);
}
template <>
void XPrimaryDriver::push(unsigned short x, std::vector<char> &buf) {
    _push_short(static_cast<short>(x), buf);
}
template <>
void XPrimaryDriver::push(int32_t x, std::vector<char> &buf) {
    _push_int32(x, buf);
}
template <>
void XPrimaryDriver::push(uint32_t x, std::vector<char> &buf) {
    _push_int32(static_cast<int32_t>(x), buf);
}
template <>
void XPrimaryDriver::push(float f, std::vector<char> &buf) {
    union {
        int32_t x;
        float y;
    } uni;
    C_ASSERT(sizeof(uni.x) == sizeof(uni.y));
    uni.y =  f;
    _push_int32(uni.x, buf);
}
template <>
void XPrimaryDriver::push(double x, std::vector<char> &buf) {
    _push_double(x, buf);
}
