#include "primarydriver.h"
#include "interface.h"
#include <klocale.h>

XPrimaryDriver::XPrimaryDriver(const char *name, bool runtime, 
   const shared_ptr<XScalarEntryList> &scalarentries,
   const shared_ptr<XInterfaceList> &interfaces,
   const shared_ptr<XThermometerList> &thermometers,
   const shared_ptr<XDriverList> &drivers) :
    XDriver(name, runtime, scalarentries, interfaces, thermometers, drivers),
    m_interface(create<XInterface>("Interface", false,
            dynamic_pointer_cast<XDriver>(shared_from_this())))
{
    interfaces->insert(m_interface);
}
void
XPrimaryDriver::replaceInterface(const shared_ptr<XInterface> &replacement,
         const shared_ptr<XInterfaceList> &interfaces)
{
        releaseChild(interface());
        interfaces->releaseChild(interface());
        m_interface = replacement;
        interfaces->insert(interface());
}
void
XPrimaryDriver::readUnlockRaw() {
    m_rawLock.readUnlock();
}
void
XPrimaryDriver::readLockRaw() {
    m_rawLock.readLock();
}

void
XPrimaryDriver::startWritingRaw()
{
    m_rawLock.writeLock();
    m_rawData.clear();
}
void
XPrimaryDriver::finishWritingRaw(
    const XTime &time_awared, const XTime &time_recorded_org, bool success)
{
    if(success) {
        XTime time_recorded = time_recorded_org;
        m_rawLock.writeUnlockNReadLock();
        startRecording(time_awared);
        if(time_recorded) {
            m_pop_it = m_rawData.begin();
            try {
                analyzeRaw();
            }
            catch (XSkippedRecordError&) {
                 time_recorded = XTime(); //record is invalid
            }
            catch (XRecordError& e) {
                 time_recorded = XTime(); //record is invalid
                 e.print(getName() + ": " + i18n("Record Error, because "));
            }
        }
        readUnlockRaw();
        finishRecordingNReadLock(time_recorded);
        visualize();
        readUnlockRecord();
    }
    else
        m_rawLock.writeUnlock();
}

void
XPrimaryDriver::_push_char(char x, std::vector<char> &buf) {
    buf.push_back(x);
}
void
XPrimaryDriver::_push_short(short x, std::vector<char> &buf) {
    auto short y = x;
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
    auto int32_t y = x;
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
    auto double y = x;
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
    auto short x;
    char *p = reinterpret_cast<char *>(&x);
#ifdef __BIG_ENDIAN__
    for(char *z = p + sizeof(x) - 1; z >= p; z--) {
#else
    for(char *z = p; z < p + sizeof(x); z++) {
#endif
        *z = *(it++);
    }
    return x;
}
int32_t
XPrimaryDriver::_pop_int32(std::vector<char>::iterator &it) {
    auto int32_t x;
    char *p = reinterpret_cast<char *>(&x);
#ifdef __BIG_ENDIAN__
    for(char *z = p + sizeof(x) - 1; z >= p; z--) {
#else
    for(char *z = p; z < p + sizeof(x); z++) {
#endif
        *z = *(it++);
    }
    return x;
}
double
XPrimaryDriver::_pop_double(std::vector<char>::iterator &it) {
    auto double x;
    char *p = reinterpret_cast<char *>(&x);
#ifdef __BIG_ENDIAN__
    for(char *z = p + sizeof(x) - 1; z >= p; z--) {
#else
    for(char *z = p; z < p + sizeof(x); z++) {
#endif
        *z = *(it++);
    }
    return x;
}

template <>
char XPrimaryDriver::pop() throw (XBufferUnderflowRecordError&) {
    if(m_pop_it == m_rawData.end()) throw XBufferUnderflowRecordError(__FILE__, __LINE__);
    return _pop_char(m_pop_it);
}
template <>
unsigned char XPrimaryDriver::pop() throw (XBufferUnderflowRecordError&) {
    if(m_pop_it == m_rawData.end()) throw XBufferUnderflowRecordError(__FILE__, __LINE__);
    return static_cast<unsigned char>(_pop_char(m_pop_it));
}
template <>
short XPrimaryDriver::pop() throw (XBufferUnderflowRecordError&) {
    if(m_pop_it == m_rawData.end()) throw XBufferUnderflowRecordError(__FILE__, __LINE__);
    return _pop_short(m_pop_it);
}
template <>
unsigned short XPrimaryDriver::pop() throw (XBufferUnderflowRecordError&) {
    if(m_pop_it == m_rawData.end()) throw XBufferUnderflowRecordError(__FILE__, __LINE__);
    return static_cast<unsigned short>(_pop_short(m_pop_it));
}
template <>
int32_t XPrimaryDriver::pop() throw (XBufferUnderflowRecordError&) {
    if(m_pop_it == m_rawData.end()) throw XBufferUnderflowRecordError(__FILE__, __LINE__);
    return _pop_int32(m_pop_it);
}
template <>
uint32_t XPrimaryDriver::pop() throw (XBufferUnderflowRecordError&) {
    if(m_pop_it == m_rawData.end()) throw XBufferUnderflowRecordError(__FILE__, __LINE__);
    return static_cast<uint32_t>(_pop_int32(m_pop_it));
}
template <>
float XPrimaryDriver::pop() throw (XBufferUnderflowRecordError&) {
    if(m_pop_it == m_rawData.end()) throw XBufferUnderflowRecordError(__FILE__, __LINE__);
    int32_t x = _pop_int32(m_pop_it);
    C_ASSERT(sizeof(x) == 4);
    return *reinterpret_cast<float*>(&x);
}
template <>
double XPrimaryDriver::pop() throw (XBufferUnderflowRecordError&) {
    if(m_pop_it == m_rawData.end()) throw XBufferUnderflowRecordError(__FILE__, __LINE__);
    C_ASSERT(sizeof(double) == 8);
    return _pop_double(m_pop_it);
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
    auto int32_t x = _pop_int32(it);
    return *reinterpret_cast<float*>(&x);
}
template <>
double XPrimaryDriver::pop(std::vector<char>::iterator &it) {
    return _pop_double(it);
}
template <>
void XPrimaryDriver::push(char x) {
    _push_char(x, m_rawData);
}
template <>
void XPrimaryDriver::push(unsigned char x) {
    _push_char(static_cast<char>(x), m_rawData);
}
template <>
void XPrimaryDriver::push(short x) {
    _push_short(x, m_rawData);
}
template <>
void XPrimaryDriver::push(unsigned short x) {
    _push_short(static_cast<short>(x), m_rawData);
}
template <>
void XPrimaryDriver::push(int32_t x) {
    _push_int32(x, m_rawData);
}
template <>
void XPrimaryDriver::push(uint32_t x) {
    _push_int32(static_cast<int32_t>(x), m_rawData);
}
template <>
void XPrimaryDriver::push(float f) {
    _push_int32(*reinterpret_cast<int32_t*>(&f), m_rawData);
}
template <>
void XPrimaryDriver::push(double x) {
    _push_double(x, m_rawData);
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
    _push_int32(*reinterpret_cast<int32_t*>(&f), buf);
}
template <>
void XPrimaryDriver::push(double x, std::vector<char> &buf) {
    _push_double(x, buf);
}
