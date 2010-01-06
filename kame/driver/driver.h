/***************************************************************************
		Copyright (C) 2002-2009 Kentaro Kitagawa
		                   kitag@issp.u-tokyo.ac.jp
		
		This program is free software; you can redistribute it and/or
		modify it under the terms of the GNU Library General Public
		License as published by the Free Software Foundation; either
		version 2 of the License, or (at your option) any later version.
		
		You should have received a copy of the GNU Library General 
		Public License and a list of authors along with this program; 
		see the files COPYING and AUTHORS.
***************************************************************************/
#ifndef driverH
#define driverH

#include "xnode.h"
#include "xlistnode.h"
#include <vector>
#include <set>

class XRecordDependency;
class XScalarEntryList;
class XInterfaceList;
class XThermometerList;
class XDriverList;

//! Base class for all instrument drivers
class XDriver : public XNode
{
	XNODE_OBJECT
protected:
	XDriver(const char *name, bool runtime,
			const shared_ptr<XScalarEntryList> &scalarentries,
			const shared_ptr<XInterfaceList> &interfaces,
			const shared_ptr<XThermometerList> &thermometers,
			const shared_ptr<XDriverList> &drivers);
public:
	virtual ~XDriver() {}

	//! Shows all forms belonging to the driver.
	virtual void showForms() = 0;
 
	//! Be called during parsing.
	XTalker<shared_ptr<XDriver> > &onRecord() {return m_tlkRecord;}
	//! Locks the analyzed members and entries
	void readLockRecord() const;
	void readUnlockRecord() const;
  
	//! Recorded time.
	//! It is a time stamp when a phenomenon occurred and recorded.
	//! Following analyses have to be based on this time.
	//! It is undefined if record is invalid.
	const XTime &time() const {return m_recordTime;}
	//! A time stamp when an operator (you) can see outputs.
	//! Manual operations (e.g. pushing a clear button) have to be based on this time.
	//! It is a time when a phenomenon starts if measurement is going on.
	//! It is a time when a record was read for a non-real-time analysis.
	//! It is undefined if record is invalid.
	const XTime &timeAwared() const {return m_awaredTime;}
  
	virtual const shared_ptr<XRecordDependency> dependency() const = 0;
protected:
	//! Throwing this exception will cause a reset of record time.
	//! And, prints error message.
	struct XRecordError : public XKameError {
		XRecordError(const XString &s, const char *file, int line) : XKameError(s, file, line) {}
	};
	//! Throwing this exception will skip signal emission, assuming record is kept valid.
	struct XSkippedRecordError : public XRecordError {
		XSkippedRecordError(const XString &s, const char *file, int line) : XRecordError(s, file, line) {}
		XSkippedRecordError(const char *file, int line) : XRecordError("", file, line) {}
	};
	//! The size of the raw record is not enough to continue analyzing.
	struct XBufferUnderflowRecordError : public XRecordError {
		XBufferUnderflowRecordError(const char *file, int line);
	};
 
	//! This is called after analyze() or analyzeRaw()
	//! The record will be read-locked.
	//! This might be called even if the record is broken (time() == false).
	virtual void visualize() = 0;
  
	//! Write-locks a record and read-locks all dependent drivers.
	//! \return true if locked.
	bool tryStartRecording();
	//! m_tlkRecord is invoked after unlocking
	//! \sa time(), timeAwared()
	void finishRecordingNReadLock(const XTime &time_awared, const XTime &time_recorded);
	//! leaves the existing record.
	void abortRecording();
	//! leaves the existing record.
	void abortRecordingNReadLock();
	//! Lock this record and dependent drivers
private:
	XTalker<shared_ptr<XDriver> > m_tlkRecord;
	//! mutex for record
	XRecursiveRWLock m_recordLock;
	//! \sa time()
	XTime m_recordTime;
	//! \sa timeAwared()
	XTime m_awaredTime;
};

//! When a record depends on other records, multiple delegations may cause a confilct of time stamps. This class can detect it.
class XRecordDependency
{
public:
    XRecordDependency();
    XRecordDependency(const shared_ptr<XRecordDependency> &);
    //! \return true if conflicted.
    //! Search for entry which has the same driver and different times.
    bool merge(const shared_ptr<const XDriver> &driver);
    void clear();
    
    bool isConflict() const {return m_bConflict;}
private:
    bool m_bConflict;
	struct tDependency {
		tDependency(const shared_ptr<const XDriver> &d, const XTime &time);
		bool operator<(const tDependency &d) const;
		shared_ptr<const XDriver> driver;    
		XTime time;
	};
	//! written on recording
	std::set<tDependency> m_dependency;
	typedef std::set<tDependency>::iterator tDependency_it;

};


class XDriverList : public XCustomTypeListNode<XDriver>
{
	XNODE_OBJECT
protected:
	XDriverList(const char *name, bool runtime,
				const shared_ptr<XScalarEntryList> &scalarentries,
				const shared_ptr<XInterfaceList> &interfaces,
				const shared_ptr<XThermometerList> &thermometers);
public:
	DEFINE_TYPE_HOLDER_EXTRA_PARAMS_4(
		const shared_ptr<XScalarEntryList> &,
		const shared_ptr<XInterfaceList> &,
		const shared_ptr<XThermometerList> &,
		const shared_ptr<XDriverList> &
		)
		virtual shared_ptr<XNode> createByTypename(const XString &type, const XString& name);
private:
	const shared_ptr<XScalarEntryList> m_scalarentries;
	const shared_ptr<XInterfaceList> m_interfaces;
	const shared_ptr<XThermometerList> m_thermometers;
};
//---------------------------------------------------------------------------
#endif
