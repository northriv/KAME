/***************************************************************************
		Copyright (C) 2002-2010 Kentaro Kitagawa
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
#include "measure.h"
#include <vector>
#include <set>

class XRecordDependency;
class XScalarEntryList;
class XInterfaceList;
class XThermometerList;
class XDriverList;

//! Base class for all instrument drivers
class XDriver : public XNode {
public:
	XDriver(const char *name, bool runtime, Transaction &tr_meas, const shared_ptr<XMeasure> &meas);
	virtual ~XDriver() {}

	//! Shows all forms belonging to the driver.
	virtual void showForms() = 0;
 
	struct Payload : public XNode::Payload {
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

		Talker<XDriver*, XDriver*> &onRecord() {return m_tlkOnRecord;}
		const Talker<XDriver*, XDriver*> &onRecord() const {return m_tlkOnRecord;}
	private:
		friend class XDriver;

		//! \sa time()
		XTime m_recordTime;
		//! \sa timeAwared()
		XTime m_awaredTime;

		Talker<XDriver*, XDriver*> m_tlkOnRecord;
	};
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
 
	//! This function is called after committing XPrimaryDriver::analyzeRaw() or XSecondaryDriver::analyze().
	//! This might be called even if the record is invalid (time() == false).
	virtual void visualize(const Snapshot &shot) = 0;
  
	//! Sets time stamps.
	//! \sa time(), timeAwared()
	void record(Transaction &tr,
		const XTime &time_awared, const XTime &time_recorded);
private:
};

class XDriverList : public XCustomTypeListNode<XDriver> {
public:
	XDriverList(const char *name, bool runtime, const shared_ptr<XMeasure> &measure);

	DEFINE_TYPE_HOLDER_EXTRA_PARAMS_2(
		reference_wrapper<Transaction>,
		const shared_ptr<XMeasure> &
		)
		virtual shared_ptr<XNode> createByTypename(const XString &type, const XString& name);
private:
	const weak_ptr<XMeasure> m_measure;
};
//---------------------------------------------------------------------------
#endif
