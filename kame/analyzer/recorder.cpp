/***************************************************************************
		Copyright (C) 2002-2015 Kentaro Kitagawa
		                   kitag@issp.u-tokyo.ac.jp
		
		This program is free software; you can redistribute it and/or
		modify it under the terms of the GNU Library General Public
		License as published by the Free Software Foundation; either
		version 2 of the License, or (at your option) any later version.
		
		You should have received a copy of the GNU Library General 
		Public License and a list of authors along with this program; 
		see the files COPYING and AUTHORS.
***************************************************************************/
//---------------------------------------------------------------------------

#include "recorder.h"
#include "analyzer.h"
#include "primarydriver.h"
#include "xtime.h"

#include <zlib.h>
#include <vector>

//---------------------------------------------------------------------------
#define OFSMODE std::ios::out | std::ios::app | std::ios::ate

XRawStream::XRawStream(const char *name, bool runtime, const shared_ptr<XDriverList> &driverlist)
	: XNode(name, runtime),
	  m_drivers(driverlist),
	  m_pGFD(0),
	  m_filename(create<XStringNode>("Filename", true)) {
}
XRawStream::~XRawStream() {
    if(m_pGFD) gzclose(static_cast<gzFile>(m_pGFD));
}    

XRawStreamRecorder::XRawStreamRecorder(const char *name, bool runtime, const shared_ptr<XDriverList> &driverlist)
	: XRawStream(name, runtime, driverlist),
	  m_recording(create<XBoolNode>("Recording", true)) {
    
    iterate_commit([=](Transaction &tr){
	    tr[ *recording()] = false;
	    m_lsnOnOpen = tr[ *filename()].onValueChanged().connectWeakly(
	        shared_from_this(), &XRawStreamRecorder::onOpen);
	    m_lsnOnFlush = tr[ *recording()].onValueChanged().connectWeakly(
	        shared_from_this(), &XRawStreamRecorder::onFlush);
    });
    m_drivers->iterate_commit([=](Transaction &tr){
        m_lsnOnCatch = tr[ *m_drivers].onCatch().connect( *this, &XRawStreamRecorder::onCatch);
        m_lsnOnRelease = tr[ *m_drivers].onRelease().connect( *this, &XRawStreamRecorder::onRelease);
    });
}
void
XRawStreamRecorder::onCatch(const Snapshot &shot, const XListNodeBase::Payload::CatchEvent &e) {
    auto driver = static_pointer_cast<XDriver>(e.caught);
    driver->iterate_commit([=](Transaction &tr){
        if(m_lsnOnRecord)
			tr[ *driver].onRecord().connect(m_lsnOnRecord);
		else
			m_lsnOnRecord = tr[ *driver].onRecord().connectWeakly(
				shared_from_this(), &XRawStreamRecorder::onRecord);
    });
}
void
XRawStreamRecorder::onRelease(const Snapshot &shot, const XListNodeBase::Payload::ReleaseEvent &e) {
    auto driver = static_pointer_cast<XDriver>(e.released);
    driver->iterate_commit([=](Transaction &tr){
        tr[ *driver].onRecord().disconnect(m_lsnOnRecord);
    });
}
void
XRawStreamRecorder::onOpen(const Snapshot &shot, XValueNodeBase *) {
	if(m_pGFD) gzclose(static_cast<gzFile>(m_pGFD));
	m_pGFD = gzopen(QString(( **filename())->to_str()).toLocal8Bit().data(), "wb");
}
void
XRawStreamRecorder::onFlush(const Snapshot &shot, XValueNodeBase *) {
	if( !***recording())
		if(m_pGFD) {
			m_filemutex.lock();    
			gzflush(static_cast<gzFile>(m_pGFD), Z_FULL_FLUSH);
			m_filemutex.unlock();    
		}
}
void
XRawStreamRecorder::onRecord(const Snapshot &shot, XDriver *d) {
    if( ***recording() && m_pGFD) {
        auto *driver = dynamic_cast<XPrimaryDriver*>(d);
        if(driver) {
        	const XPrimaryDriver::RawData &rawdata(shot[ *driver].rawData());
            uint32_t size = rawdata.size();
            if(size) {
                uint32_t headersize =
                    sizeof(uint32_t) //allsize
                    + sizeof(int32_t) //time().sec()
                    + sizeof(int32_t); //time().usec()            
                // size of raw record wrapped by header and footer
                uint32_t allsize =
                    headersize
                    + driver->getName().size() //name of driver
                    + 2 //two null chars
                    + size //rawData
                    + sizeof(uint32_t); //allsize
                XPrimaryDriver::RawData header;
                header.push((uint32_t)allsize);
                header.push((int32_t)shot[ *driver].time().sec());
                header.push((int32_t)shot[ *driver].time().usec());
                assert(header.size() == headersize);
    
                m_filemutex.lock();
                gzwrite(static_cast<gzFile>(m_pGFD), &header[0], header.size());
                gzprintf(static_cast<gzFile>(m_pGFD), "%s", (const char*)driver->getName().c_str());
                gzputc(static_cast<gzFile>(m_pGFD), '\0');
                gzputc(static_cast<gzFile>(m_pGFD), '\0'); //Reserved
                gzwrite(static_cast<gzFile>(m_pGFD), &rawdata[0], size);
                header.clear(); //using as a footer.
                header.push((uint32_t)allsize);
                gzwrite(static_cast<gzFile>(m_pGFD), &header[0], header.size());
                m_filemutex.unlock();
            }
        }
    } 
}


XTextWriter::XTextWriter(const char *name, bool runtime,
						 const shared_ptr<XDriverList> &driverlist, const shared_ptr<XScalarEntryList> &entrylist)
	: XNode(name, runtime),
	  m_drivers(driverlist),
	  m_entries(entrylist),
	  m_filename(create<XStringNode>("Filename", true)),
	  m_lastLine(create<XStringNode>("LastLine", true)),
	  m_recording(create<XBoolNode>("Recording", true)),
	  m_logFilename(create<XStringNode>("LogFilename", false)),
	  m_logRecording(create<XBoolNode>("LogRecording", false)),
	  m_logEvery(create<XUIntNode>("LogEvery", false))  {
  
    iterate_commit([=](Transaction &tr){
	    tr[ *recording()] = false;
	    tr[ *lastLine()].setUIEnabled(false);
	    tr[ *logRecording()] = false;
	    tr[ *logEvery()] = 300;
	    m_lsnOnFilenameChanged = tr[ *filename()].onValueChanged().connectWeakly(
	        shared_from_this(), &XTextWriter::onFilenameChanged);
	    m_lsnOnLogFilenameChanged = tr[ *logFilename()].onValueChanged().connectWeakly(
	        shared_from_this(), &XTextWriter::onLogFilenameChanged);
    });
    m_drivers->iterate_commit([=](Transaction &tr){
        m_lsnOnCatch = tr[ *m_drivers].onCatch().connect( *this, &XTextWriter::onCatch);
        m_lsnOnRelease = tr[ *m_drivers].onRelease().connect( *this, &XTextWriter::onRelease);
    });
}
void
XTextWriter::onCatch(const Snapshot &shot, const XListNodeBase::Payload::CatchEvent &e) {
    auto driver = static_pointer_cast<XDriver>(e.caught);
    driver->iterate_commit([=](Transaction &tr){
        if(m_lsnOnRecord)
			tr[ *driver].onRecord().connect(m_lsnOnRecord);
		else
			m_lsnOnRecord = tr[ *driver].onRecord().connectWeakly(
				shared_from_this(), &XTextWriter::onRecord);
    });
}
void
XTextWriter::onRelease(const Snapshot &shot, const XListNodeBase::Payload::ReleaseEvent &e) {
    auto driver = static_pointer_cast<XDriver>(e.released);
    driver->iterate_commit([=](Transaction &tr){
        tr[ *driver].onRecord().disconnect(m_lsnOnRecord);
    });
}
void
XTextWriter::onLastLineChanged(const Snapshot &shot, XValueNodeBase *) {
	XScopedLock<XRecursiveMutex> lock(m_filemutex);  
	if(m_stream.good()) {
		m_stream << shot[ *lastLine()].to_str()
				 << std::endl;
	}
}
void
XTextWriter::onRecord(const Snapshot &shot, XDriver *driver) {
	Snapshot shot_this( *this);
	XScopedLock<XRecursiveMutex> lock(m_logFilemutex);
	XTime logtime = XTime::now();
	XString logline;
	bool record_log = shot_this[ *logRecording()] &&
		(logtime - m_loggedTime > shot_this[ *logEvery()]);
	if(shot_this[ *recording()] || record_log) {
        if(shot[ *driver].time().isSet()) {
			for(;;) {
				Snapshot shot_entries( *m_entries);
				if( !shot_entries.size())
					break;
				const XNode::NodeList &entries_list( *shot_entries.list());
				//logger
				if(record_log) {
					m_loggedTime = logtime;
					for(auto it = entries_list.begin(); it != entries_list.end(); it++) {
						auto entry = static_pointer_cast<XScalarEntry>( *it);
                        logline.append(shot_entries[ *entry->value()].to_str() + KAME_DATAFILE_DELIMITER);
					}
					logline.append(m_loggedTime.getTimeFmtStr("%Y/%m/%d %H:%M:%S"));
				}
				if( !shot_this[ *recording()])
					break;
				//triggered writer
				bool triggered = false;
				for(auto it = entries_list.begin(); it != entries_list.end(); it++) {
					auto entry = static_pointer_cast<XScalarEntry>( *it);
                    if( !shot_entries[ *entry->store()]) continue;
                    shared_ptr<XDriver> d(entry->driver());
                    if( !d) continue;
                    try {
                        if((d.get() == driver) && shot.at( *entry).isTriggered()) {
                            triggered = true;
                            break;
                        }
                    }
                    catch (NodeNotFoundError &) {
                        fprintf(stderr, "Entry freed from driver was selected!\n");
                    }
				}
				if( !triggered)
					break;
				Transaction tr_entries(shot_entries);
				XString buf;
				for(auto it = entries_list.begin(); it != entries_list.end(); it++) {
					auto entry = static_pointer_cast<XScalarEntry>( *it);
					if( !shot_entries[ *entry->store()]) continue;
					entry->storeValue(tr_entries);
                    buf.append(shot_entries[ *entry->value()].to_str() + KAME_DATAFILE_DELIMITER);
				}
				buf.append(shot[ *driver].time().getTimeFmtStr("%Y/%m/%d %H:%M:%S"));
				if(tr_entries.commit()) {
					trans( *lastLine()) = buf;
					break;
				}
			}
		}
	}
	if(record_log) {
		if(m_logStream.good()) {
			m_logStream << logline
					 << std::endl;
		}
	}
}

void
XTextWriter::onFilenameChanged(const Snapshot &shot, XValueNodeBase *) {
	XScopedLock<XRecursiveMutex> lock(m_filemutex);  
	if(m_stream.is_open()) m_stream.close();
	m_stream.clear();
	m_stream.open((const char*)QString(shot[ *filename()].to_str()).toLocal8Bit().data(), OFSMODE);

	if(m_stream.good()) {
        iterate_commit([=](Transaction &tr){
			m_lsnOnFlush = tr[ *recording()].onValueChanged().connectWeakly(
				shared_from_this(), &XTextWriter::onFlush);
			m_lsnOnLastLineChanged = tr[ *lastLine()].onValueChanged().connectWeakly(
				shared_from_this(), &XTextWriter::onLastLineChanged);
        });
		lastLine()->setUIEnabled(true);

		XString buf;
		buf = "#";
		Snapshot shot_entries( *m_entries);
		if(shot_entries.size()) {
			const XNode::NodeList &entries_list( *shot_entries.list());
			for(auto it = entries_list.begin(); it != entries_list.end(); it++) {
				auto entry = static_pointer_cast<XScalarEntry>( *it);
				if( !shot_entries[ *entry->store()]) continue;
				buf.append(entry->getLabel());
                buf.append(KAME_DATAFILE_DELIMITER);
			}
		}
        buf.append("Date" KAME_DATAFILE_DELIMITER "Time" KAME_DATAFILE_DELIMITER "msec");
        trans( *lastLine()) = buf;
	}
	else {
		m_lsnOnFlush.reset();
		m_lsnOnLastLineChanged.reset();
		lastLine()->setUIEnabled(false);
		gErrPrint(i18n("Failed to open file."));
	}
}
void
XTextWriter::onFlush(const Snapshot &shot, XValueNodeBase *) {
    lastLine()->setUIEnabled( ***recording());
	if( !***recording()) {
		XScopedLock<XRecursiveMutex> lock(m_filemutex);  
		if(m_stream.good())
			m_stream.flush();
	}
}

void
XTextWriter::onLogFilenameChanged(const Snapshot &shot, XValueNodeBase *) {
	XScopedLock<XRecursiveMutex> lock(m_logFilemutex);
	if(m_logStream.is_open()) m_logStream.close();
	m_logStream.clear();
	m_logStream.open((const char*)QString(shot[ *logFilename()].to_str()).toLocal8Bit().data(), OFSMODE);

	if(m_logStream.good()) {
		XString buf;
		buf = "#";
		Snapshot shot_entries( *m_entries);
		if(shot_entries.size()) {
			const XNode::NodeList &entries_list( *shot_entries.list());
			for(auto it = entries_list.begin(); it != entries_list.end(); it++) {
				auto entry = static_pointer_cast<XScalarEntry>( *it);
				buf.append(entry->getLabel());
                buf.append(KAME_DATAFILE_DELIMITER);
			}
		}
        buf.append("Date" KAME_DATAFILE_DELIMITER "Time" KAME_DATAFILE_DELIMITER "msec");
        m_logStream << buf
				 << std::endl;
	}
	else {
        gWarnPrint(i18n("All-entry logger: Failed to open file."));
	}
}
