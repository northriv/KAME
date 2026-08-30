/***************************************************************************
		Copyright (C) 2002-2015 Kentaro Kitagawa
		                   kitag@issp.u-tokyo.ac.jp
		
		This program is free software; you can redistribute it and/or
		modify it under the terms of the GNU General Public
		License as published by the Free Software Foundation; either
		version 2 of the License, or (at your option) any later version.
		
		You should have received a copy of the GNU General 
		Public License and a list of authors along with this program; 
		see the files COPYING and AUTHORS.
***************************************************************************/
//---------------------------------------------------------------------------

#include "rawstream.h"
#include "primarydriver.h"
#include "xtime.h"

#include <zlib.h>
#include <vector>

//---------------------------------------------------------------------------

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
	m_lastFlushed = XTime();
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
                uint32_t headersize = KAMB_HEADER_SIZE;
                int32_t sec = shot[ *driver].time().sec();
                int32_t usec = shot[ *driver].time().usec();
                //timeAwared(), as microseconds from time(), in the field that
                //has been an empty string since the format was written.
                //
                //Every reader ever written accounts for its length, so putting
                //something there breaks neither direction: an older KAME reads
                //these files, and this one reads files that leave it empty.
                //Extending the binary header instead would have cost the first
                //of those.
                //
                //It matters because secondary drivers read the emitter's
                //timeAwared to decide whether a record is fresher than the
                //state it depends on, and one of them stamps its own record
                //with it -- and a replay currently hands them the time it is
                //replaying at.  Recording it is the half that cannot be done
                //later; feeding it back is a change of behaviour and waits.
                //\sa doc/design/PROVENANCE.md
                XString reserved;
                const XTime &ta = shot[ *driver].timeAwared();
                if(ta.isSet()) {
                    long long delta = ((long long)ta.sec() - sec) * 1000000LL
                        + ((long long)ta.usec() - usec);
                    if(delta)
                        reserved = formatString("t%lld", delta);
                }
                // size of raw record wrapped by header and footer
                uint32_t allsize =
                    headersize
                    + driver->getName().size() //name of driver
                    + reserved.size() //timeAwared, when it differs from time()
                    + 2 //two null chars
                    + size //rawData
                    + sizeof(uint32_t); //allsize
                XPrimaryDriver::RawData header;
                //The magic and the check come first so that a record can be
                //found by looking for it, rather than by guessing at every
                //offset which four bytes might be a length.
                //\sa kamb_record_check(), XJournalReader::seek_()
                header.push((uint32_t)KAMB_RECORD_MAGIC);
                header.push((uint32_t)kamb_record_check(allsize, sec, usec));
                header.push((uint32_t)allsize);
                header.push((int32_t)sec);
                header.push((int32_t)usec);
                assert(header.size() == headersize);
    
                m_filemutex.lock();
                gzwrite(static_cast<gzFile>(m_pGFD), &header[0], header.size());
                gzprintf(static_cast<gzFile>(m_pGFD), "%s", (const char*)driver->getName().c_str());
                gzputc(static_cast<gzFile>(m_pGFD), '\0');
                if(reserved.length())
                    gzprintf(static_cast<gzFile>(m_pGFD), "%s", reserved.c_str());
                gzputc(static_cast<gzFile>(m_pGFD), '\0'); //end of the reserved field
                gzwrite(static_cast<gzFile>(m_pGFD), &rawdata[0], size);
                header.clear(); //using as a footer.
                header.push((uint32_t)allsize);
                gzwrite(static_cast<gzFile>(m_pGFD), &header[0], header.size());
                //Z_FULL_FLUSH ends the deflate block AND resets the dictionary,
                //so everything up to here stays readable on its own: a KAME that
                //is killed mid-run leaves a .kamb missing at most the last
                //minute, rather than a stream that stops being decodable at the
                //last block boundary zlib happened to emit.  It also leaves
                //restart points a seek could one day use.
                //
                //A minute, not a second (which is what the journal does): this
                //is the high-rate file, and each flush costs compression on
                //every byte written after it.
                XTime now = XTime::now();
                if( !m_lastFlushed.isSet() || (now.diff_msec(m_lastFlushed) > 60000)) {
                    gzflush(static_cast<gzFile>(m_pGFD), Z_FULL_FLUSH);
                    m_lastFlushed = now;
                }
                m_filemutex.unlock();
                m_bytesWritten += allsize;
            }
        }
    } 
}

