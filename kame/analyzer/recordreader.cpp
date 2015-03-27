/***************************************************************************
		Copyright (C) 2002-2014 Kentaro Kitagawa
		                   kitagawa@phys.s.u-tokyo.ac.jp
		
		This program is free software; you can redistribute it and/or
		modify it under the terms of the GNU Library General Public
		License as published by the Free Software Foundation; either
		version 2 of the License, or (at your option) any later version.
		
		You should have received a copy of the GNU Library General 
		Public License and a list of authors along with this program; 
		see the files COPYING and AUTHORS.
***************************************************************************/
#include "recordreader.h"
#include "analyzer.h"
#include "primarydriver.h"
#include "xtime.h"
#include "measure.h"

#include <zlib.h>
#include <vector>

#define IFSMODE std::ios::in
#define SPEED_FASTEST "Fastest"
#define SPEED_FAST "Fast"
#define SPEED_NORMAL "Normal"
#define SPEED_SLOW "Slow"

#define RECORDREADER_DELAY 20
#define RECORD_READER_NUM_THREADS 1

XRawStreamRecordReader::XIOError::XIOError(const char *file, int line)
	: XRecordError(i18n("IO Error"), file, line) {}
XRawStreamRecordReader::XIOError::XIOError(const XString &msg, const char *file, int line)
	: XRecordError(msg, file, line) {}
XRawStreamRecordReader::XBufferOverflowError::XBufferOverflowError(const char *file, int line)
	: XIOError(i18n("Buffer Overflow Error"), file, line) {}
XRawStreamRecordReader::XBrokenRecordError::XBrokenRecordError(const char *file, int line)
	: XRecordError(i18n("Broken Record Error"), file, line) {}
XRawStreamRecordReader::XNoDriverError::
XNoDriverError(const XString &driver_name, const char *file, int line)
	: XRecordError(i18n("No Driver Error: ") + driver_name, file, line),
	  name(driver_name) {}
         
XRawStreamRecordReader::XRawStreamRecordReader(const char *name, bool runtime, const shared_ptr<XDriverList> &driverlist)
	: XRawStream(name, runtime, driverlist),
	  m_speed(create<XComboNode>("Speed", true, true)),
	  m_fastForward(create<XBoolNode>("FastForward", true)),
	  m_rewind(create<XBoolNode>("Rewind", true)),
	  m_stop(create<XTouchableNode>("Stop", true)),
	  m_first(create<XTouchableNode>("First", true)),
	  m_next(create<XTouchableNode>("Next", true)),
	  m_back(create<XTouchableNode>("Back", true)),
	  m_posString(create<XStringNode>("PosString", true)),
	  m_periodicTerm(0) {

    for(Transaction tr( *this);; ++tr) {
        tr[ *m_speed].add(SPEED_FASTEST);
        tr[ *m_speed].add(SPEED_FAST);
        tr[ *m_speed].add(SPEED_NORMAL);
        tr[ *m_speed].add(SPEED_SLOW);
        tr[ *m_speed] = SPEED_FAST;

        m_lsnOnOpen = tr[ *filename()].onValueChanged().connectWeakly(
            shared_from_this(), &XRawStreamRecordReader::onOpen);
		m_lsnFirst = tr[ *m_first].onTouch().connectWeakly(
			shared_from_this(), &XRawStreamRecordReader::onFirst,
			XListener::FLAG_MAIN_THREAD_CALL | XListener::FLAG_AVOID_DUP | XListener::FLAG_DELAY_ADAPTIVE);
		m_lsnBack = tr[ *m_back].onTouch().connectWeakly(
			shared_from_this(), &XRawStreamRecordReader::onBack,
			XListener::FLAG_MAIN_THREAD_CALL | XListener::FLAG_AVOID_DUP | XListener::FLAG_DELAY_ADAPTIVE);
		m_lsnNext = tr[ *m_next].onTouch().connectWeakly(
			shared_from_this(), &XRawStreamRecordReader::onNext,
			XListener::FLAG_MAIN_THREAD_CALL | XListener::FLAG_AVOID_DUP | XListener::FLAG_DELAY_ADAPTIVE);
		m_lsnStop = tr[ *m_stop].onTouch().connectWeakly(
			shared_from_this(), &XRawStreamRecordReader::onStop,
			XListener::FLAG_MAIN_THREAD_CALL | XListener::FLAG_AVOID_DUP | XListener::FLAG_DELAY_ADAPTIVE);
	    m_lsnPlayCond = tr[ *m_fastForward].onValueChanged().connectWeakly(
			shared_from_this(),
			&XRawStreamRecordReader::onPlayCondChanged,
			XListener::FLAG_MAIN_THREAD_CALL | XListener::FLAG_AVOID_DUP | XListener::FLAG_DELAY_ADAPTIVE);
	    tr[ *m_rewind].onValueChanged().connect(m_lsnPlayCond);
	    tr[ *m_speed].onValueChanged().connect(m_lsnPlayCond);
		if(tr.commit())
			break;
    }
    
    m_threads.resize(RECORD_READER_NUM_THREADS);
    for(auto it = m_threads.begin(); it != m_threads.end(); it++) {
        it->reset(new XThread<XRawStreamRecordReader>(shared_from_this(),
													  &XRawStreamRecordReader::execute));
        ( *it)->resume();
    }
}
void
XRawStreamRecordReader::onOpen(const Snapshot &shot, XValueNodeBase *) {
	if(m_pGFD) gzclose(static_cast<gzFile>(m_pGFD));
	m_pGFD = gzopen(QString(( **filename())->to_str()).toLatin1().data(), "rb");
}
void
XRawStreamRecordReader::readHeader(void *_fd)
	throw (XRawStreamRecordReader::XRecordError &) {
	gzFile fd = static_cast<gzFile>(_fd);

	if(gzeof(fd))
		throw XIOError(__FILE__, __LINE__);
	uint32_t size =
		sizeof(uint32_t) //allsize
		+ sizeof(int32_t) //time().sec()
		+ sizeof(int32_t); //time().usec()
	std::vector<char> buf(size);
	XPrimaryDriver::RawDataReader reader(buf);
	if(gzread(fd, &buf[0], size) == -1) throw XIOError(__FILE__, __LINE__);
	m_allsize = reader.pop<uint32_t>();
	long sec = reader.pop<int32_t>();
	long usec = reader.pop<int32_t>();
	m_time = XTime(sec, usec);
}
void
XRawStreamRecordReader::parseOne(void *_fd, XMutex &mutex)
	throw (XRawStreamRecordReader::XRecordError &) {
	gzFile fd = static_cast<gzFile>(_fd);

	readHeader(fd);
	char name[256], sup[256];
	gzgetline(fd, (unsigned char*)name, 256, '\0');
	gzgetline(fd, (unsigned char*)sup, 256, '\0');
	if(strlen(name) == 0) {
		throw XBrokenRecordError(__FILE__, __LINE__);
	}
	shared_ptr<XNode> driver_precast = m_drivers->getChild(name);
	auto driver = dynamic_pointer_cast<XPrimaryDriver>(driver_precast);
	uint32_t size = 
		m_allsize - (
			sizeof(uint32_t) //allsize
			+ sizeof(int32_t) //time().sec()
			+ sizeof(int32_t) //time().usec()
			+ strlen(name) //name of driver
			+ strlen(sup) //reserved
			+ 2 //two null chars
			+ sizeof(uint32_t)  //allsize
			);
    // m_time must be copied before unlocking
    XTime time(m_time);
    trans( *m_posString) = time.getTimeStr();
    if( !driver || (size > MAX_RAW_RECORD_SIZE)) {
        if(gzseek(fd, size + sizeof(uint32_t), SEEK_CUR) == -1)
			throw XIOError(__FILE__, __LINE__);
		if(driver)
			throw XBrokenRecordError(__FILE__, __LINE__);
		if(driver_precast)
	        throw XNoDriverError(formatString_tr(I18N_NOOP("Typemismatch: %s"), name),
	         __FILE__, __LINE__);
		else
	        throw XNoDriverError(name, __FILE__, __LINE__);
    }
	shared_ptr<XPrimaryDriver::RawData> rawdata(new XPrimaryDriver::RawData());
	try {
		rawdata->resize(size);
		if(gzread(fd, &rawdata->at(0), size) == -1)
			throw XIOError(__FILE__, __LINE__);
		std::vector<char> buf(sizeof(uint32_t));
		if(gzread(fd, &buf[0], sizeof(uint32_t)) == -1)
			throw XIOError(__FILE__, __LINE__);
		XPrimaryDriver::RawDataReader reader(buf);
		uint32_t footer_allsize = reader.pop<uint32_t>();
		if(footer_allsize != m_allsize)
			throw XBrokenRecordError(__FILE__, __LINE__);
	}
	catch (XRecordError &e) {
		driver->finishWritingRaw(rawdata, XTime(), XTime());
		throw e;
	}
	mutex.unlock();
	{ XScopedLock<XMutex> lock(m_drivermutex);
	driver->finishWritingRaw(rawdata, XTime::now(), time);
	}
}
void
XRawStreamRecordReader::gzgetline(void* _fd, unsigned char*buf, unsigned int len, int del)
	throw (XIOError &) {
	gzFile fd = static_cast<gzFile>(_fd);

	int c;
	for(unsigned int i = 0; i < len; i++) {
		c = gzgetc(fd);
		if(c == -1) throw XIOError(__FILE__, __LINE__);
		*(buf++) = (unsigned char)c;
		if(c == del) return;
	}
	throw XBufferOverflowError(__FILE__, __LINE__);
}
void
XRawStreamRecordReader::first_(void *fd)
	throw (XRawStreamRecordReader::XIOError &) {
	gzrewind(static_cast<gzFile>(fd));
}
void
XRawStreamRecordReader::previous_(void *fd)
	throw (XRawStreamRecordReader::XRecordError &) {
	if(gzseek(static_cast<gzFile>(fd), -sizeof(uint32_t), SEEK_CUR) == -1) throw XIOError(__FILE__, __LINE__);
	goToHeader(fd);
}
void
XRawStreamRecordReader::next_(void *fd)
	throw (XRawStreamRecordReader::XRecordError &) {
	readHeader(fd);
	uint32_t headersize = sizeof(uint32_t) //allsize
		+ sizeof(int32_t) //time().sec()
		+ sizeof(int32_t); //time().usec()
	if(gzseek(static_cast<gzFile>(fd), m_allsize - headersize, SEEK_CUR) == -1) throw XIOError(__FILE__, __LINE__);
}
void
XRawStreamRecordReader::goToHeader(void *_fd)
	throw (XRawStreamRecordReader::XRecordError &) {
	gzFile fd = static_cast<gzFile>(_fd);

	if(gzeof(fd)) throw XIOError(__FILE__, __LINE__);
	std::vector<char> buf(sizeof(uint32_t));
	XPrimaryDriver::RawDataReader reader(buf);
	if(gzread(fd, &buf[0], sizeof(uint32_t)) == Z_NULL) throw XIOError(__FILE__, __LINE__);
	int allsize = reader.pop<uint32_t>();
	if(gzseek(fd, -allsize, SEEK_CUR) == -1) throw XIOError(__FILE__, __LINE__);
}
void
XRawStreamRecordReader::terminate() {
    m_periodicTerm = 0;
    for(auto it = m_threads.begin(); it != m_threads.end(); it++) {
        ( *it)->terminate();
    }
    XScopedLock<XCondition> lock(m_condition);
    m_condition.broadcast();
}

void
XRawStreamRecordReader::onPlayCondChanged(const Snapshot &shot, XValueNodeBase *) {
	Snapshot shot_this( *this);
    double ms = 1.0;
    if(shot_this[ *m_speed].to_str() == SPEED_FASTEST) ms = 0.1;
    if(shot_this[ *m_speed].to_str() == SPEED_FAST) ms = 10.0;
    if(shot_this[ *m_speed].to_str() == SPEED_NORMAL) ms = 30.0;
    if(shot_this[ *m_speed].to_str() == SPEED_SLOW) ms = 100.0;
    if( !shot_this[ *m_fastForward] && !shot_this[ *m_rewind]) ms = 0;
    if(shot_this[ *m_rewind]) ms = -ms;
    m_periodicTerm = ms;
    XScopedLock<XCondition> lock(m_condition);
    m_condition.broadcast();
}
void
XRawStreamRecordReader::onStop(const Snapshot &shot, XTouchableNode *) {
    m_periodicTerm = 0;
    g_statusPrinter->printMessage(i18n("Stopped"));
	for(Transaction tr( *this);; ++tr) {
		tr[ *m_fastForward] = false;
		tr[ *m_rewind] = false;
		tr.unmark(m_lsnPlayCond);
		if(tr.commit())
			break;
	}
}
void
XRawStreamRecordReader::onFirst(const Snapshot &shot, XTouchableNode *) {
	if(m_pGFD) {
		try {
			m_filemutex.lock();
			first_(m_pGFD);
			parseOne(m_pGFD, m_filemutex);
			g_statusPrinter->printMessage(i18n("First"));
		}
		catch (XRecordError &e) {
			m_filemutex.unlock();
			e.print(i18n("No Record, because "));
		}
	}
}
void
XRawStreamRecordReader::onNext(const Snapshot &shot, XTouchableNode *) {
	if(m_pGFD) {
		try {
			m_filemutex.lock(); 
			parseOne(m_pGFD, m_filemutex);
			g_statusPrinter->printMessage(i18n("Next"));
		}
		catch (XRecordError &e) {
			m_filemutex.unlock();
			e.print(i18n("No Record, because "));
		}
	}
}
void
XRawStreamRecordReader::onBack(const Snapshot &shot, XTouchableNode *) {
	if(m_pGFD) {
		try {
			m_filemutex.lock(); 
			previous_(m_pGFD);
			previous_(m_pGFD);
			parseOne(m_pGFD, m_filemutex);
			g_statusPrinter->printMessage(i18n("Previous"));
		}
		catch (XRecordError &e) {
			m_filemutex.unlock();
			e.print(i18n("No Record, because "));
		}
	}
}

void *XRawStreamRecordReader::execute(const atomic<bool> &terminated) {
	while( !terminated) {
		double ms = 0.0;
		{
			XScopedLock<XCondition> lock(m_condition);
			while((fabs((ms = m_periodicTerm)) < 1e-4) && !terminated)
				m_condition.wait();
		}
    
		if(terminated) break;
      
		try {
			m_filemutex.lock(); 
			if(ms < 0.0) {
				previous_(m_pGFD);
				previous_(m_pGFD);
			}
			parseOne(m_pGFD, m_filemutex);
		}
		catch (XNoDriverError &e) {
			m_filemutex.unlock();
			e.print(i18n("No such driver :") + e.name);
		}
		catch (XRecordError &e) {
			m_periodicTerm = 0.0;
			for(Transaction tr( *this);; ++tr) {
				tr[ *m_fastForward] = false;
				tr[ *m_rewind] = false;
				tr.unmark(m_lsnPlayCond);
				if(tr.commit())
					break;
			}
			m_filemutex.unlock();
			e.print(i18n("No Record, because "));
		}
     
		msecsleep(lrint(fabs(ms)));
	}
    return NULL;
}
