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
    XString path = ( **filename())->to_str();
	m_pGFD = gzopen(QString(path).toLocal8Bit().data(), "wb");
	m_lastFlushed = XTime();
    m_writeFailSaid = false;   //!< a new file deserves to be believed again
    //A path that will not open used to be silent, and the switch stayed on:
    //KAME then said it was recording raw records while onRecord() skipped
    //every one of them for want of a handle.  The journal beside this one has
    //reported exactly this and cleared exactly this kind of switch since it
    //existed; here it is for the file the data itself goes in.
    if( !m_pGFD && path.length()) {
        gErrPrint(i18n("Cannot write ") + path);
        trans( *recording()) = false;
    }
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
                int32_t sec = shot[ *driver].time().sec();
                int32_t usec = shot[ *driver].time().usec();
                //timeAwared() -- when the phenomenon started being measured,
                //against time(), when it happened -- as microseconds from it,
                //and only when the two differ.  Secondary drivers read the
                //emitter's to decide whether a record is fresher than the
                //state it depends on, and one stamps its own record with it;
                //a replay that does not have it hands them the time it is
                //replaying at.  \sa doc/design/PROVENANCE.md
                long long awared = 0;
                const XTime &ta = shot[ *driver].timeAwared();
                if(ta.isSet())
                    awared = ((long long)ta.sec() - sec) * 1000000LL
                        + ((long long)ta.usec() - usec);
                //To the payload, name included: what a reader needs to skip a
                //header it does not fully understand.
                uint32_t headersize = KAMB_FIXED_SIZE + driver->getName().size() + 1;
                uint32_t allsize =
                    headersize
                    + size //rawData
                    + sizeof(uint32_t); //allsize, again, at the end
                XPrimaryDriver::RawData header;
                //The magic and the check come first so that a record can be
                //found by looking for it, rather than by guessing at every
                //offset which four bytes might be a length; the header's own
                //length comes second, where a reader can always find it.
                //\sa kamb_record_check(), XJournalReader::seek_()
                header.push((uint32_t)KAMB_RECORD_MAGIC);
                header.push((uint32_t)headersize);
                header.push((uint32_t)kamb_record_check(allsize, sec, usec));
                header.push((uint32_t)allsize);
                header.push((int32_t)sec);
                header.push((int32_t)usec);
                header.push((int64_t)awared);
    
                m_filemutex.lock();
                gzFile fd = static_cast<gzFile>(m_pGFD);
                //Checked, all of it.  gzwrite answers with the count and 0 for
                //an error, so a disk filling up mid-run shows as a short write
                //-- and unchecked it was invisible: the switch stayed on, the
                //byte counter went on climbing, and the file stopped growing.
                const XString &dname(driver->getName());
                bool ok = (gzwrite(fd, &header[0], header.size()) == (int)header.size());
                ok = ok && (gzwrite(fd, dname.c_str(), (unsigned)dname.size())
                        == (int)dname.size());
                ok = ok && (gzputc(fd, '\0') != -1);
                assert(header.size() + driver->getName().size() + 1 == headersize);
                ok = ok && (gzwrite(fd, &rawdata[0], size) == (int)size);
                header.clear(); //using as a footer.
                header.push((uint32_t)allsize);
                ok = ok && (gzwrite(fd, &header[0], header.size()) == (int)header.size());
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
                if(ok && ( !m_lastFlushed.isSet()
                    || (now.diff_msec(m_lastFlushed) > 60000))) {
                    ok = (gzflush(fd, Z_FULL_FLUSH) == Z_OK);
                    m_lastFlushed = now;
                }
                m_filemutex.unlock();
                if(ok)
                    m_bytesWritten += allsize;
                else if( !m_writeFailSaid) {
                    //A torn record ends the recording rather than being
                    //carried past: what follows it would be read as a header.
                    m_writeFailSaid = true;
                    gErrPrint(i18n("Writing failed (disk full?): ")
                        + ( **filename())->to_str());
                    trans( *recording()) = false;
                }
            }
        }
    } 
}

