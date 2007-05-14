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

	//! show all forms belonging to driver
	virtual void showForms() = 0;
 
	//! called during parsing
	XTalker<shared_ptr<XDriver> > &onRecord() {return m_tlkRecord;}
	//! lock analysed members and entries
	void readLockRecord() const;
	void readUnlockRecord() const;
  
	//! recorded time.
	//! time when phenomenon finished and recorded.
	//! analyses must be based on this time.
	//! zero if record is invalid.
	const XTime &time() const {return m_recordTime;}
	//! time when an operator (you) can see a phenomenon. 
	//! manual operations (ex. pushing clear button) must be based on this time.
	//! time when a phenomenon starts if measuement is on real-time.
	//! time when record is read if measuement is *not* on real-time.
	//! unknown if record is invalid.
	const XTime &timeAwared() const {return m_awaredTime;}
  
	virtual const shared_ptr<XRecordDependency> dependency() const = 0;
protected:
	//! throwing this exception will cause reset of record time
	//! And, print error message
	struct XRecordError : public XKameError {
		XRecordError(const QString &s, const char *file, int line) : XKameError(s, file, line) {}
	};
	//! throwing this exception will skip signal emission, assuming record is kept valid.
	struct XSkippedRecordError : public XRecordError {
		XSkippedRecordError(const char *file, int line) : XRecordError("", file, line) {}
	};
	//! size of raw record is not enough to continue analyzing 
	struct XBufferUnderflowRecordError : public XRecordError {
		XBufferUnderflowRecordError(const char *file, int line);
	};
 
	//! this is called after analyze() or analyzeRaw()
	//! record is readLocked
	virtual void visualize() = 0;
  
	//! writeLock record and readLock all dependent drivers
	//! \return true if locked.
	bool tryStartRecording();
	//! m_tlkRecord is invoked after unlocking
	//! \sa time(), timeAwared()
	void finishRecordingNReadLock(const XTime &time_awared, const XTime &time_recorded);
	//! leave existing record.
	void abortRecording();
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

class XRecordDependency
{
public:
    XRecordDependency();
    XRecordDependency(const shared_ptr<XRecordDependency> &);
    //! ret true if conflicted
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
		virtual shared_ptr<XNode> createByTypename(const std::string &type, const std::string& name);
private:
	const shared_ptr<XScalarEntryList> m_scalarentries;
	const shared_ptr<XInterfaceList> m_interfaces;
	const shared_ptr<XThermometerList> m_thermometers;
};
//---------------------------------------------------------------------------
#endif
