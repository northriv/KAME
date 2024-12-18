/***************************************************************************
        Copyright (C) 2002-2024 Kentaro Kitagawa
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
class DECLSPEC_KAME XDriver : public XNode {
public:
	XDriver(const char *name, bool runtime, Transaction &tr_meas, const shared_ptr<XMeasure> &meas);
    virtual ~XDriver() = default;

	//! Shows all forms belonging to the driver.
	virtual void showForms() = 0;
 
    struct DECLSPEC_KAME Payload : public XNode::Payload {
		//! Recorded time.
		//! It is a time stamp when a phenomenon occurred and recorded.
		//! Following analyses have to be based on this time.
		//! It is undefined if record is invalid.
		const XTime &time() const {return m_recordTime;}
		//! A time stamp when an operator (ones) can see outputs.
		//! Manual operations (e.g. pushing a clear button) have to be based on this time.
		//! It is a time when a phenomenon starts if measurement is going on.
		//! It is a time when a record was read for a non-real-time analysis.
		//! It is undefined if record is invalid.
		const XTime &timeAwared() const {return m_awaredTime;}

        Talker<XDriver*> &onRecord() {return m_tlkOnRecord;}
        const Talker<XDriver*> &onRecord() const {return m_tlkOnRecord;}
	private:
		friend class XDriver;

		//! \sa time()
		XTime m_recordTime;
		//! \sa timeAwared()
		XTime m_awaredTime;

        Talker<XDriver*> m_tlkOnRecord;
	};
    //! Throwing this exception will cause a reset of record time.
    //! And, prints error message.
    struct DECLSPEC_KAME XRecordError : public XKameError {
        XRecordError(const XString &s, const char *file, int line) : XKameError(s, file, line) {}
        virtual ~XRecordError() = default;
    };
    //! Throwing this exception will skip signal emission, assuming record is kept valid.
    struct DECLSPEC_KAME XSkippedRecordError : public XRecordError {
        XSkippedRecordError(const XString &s, const char *file, int line) : XRecordError(s, file, line) {}
        XSkippedRecordError(const char *file, int line) : XRecordError("", file, line) {}
        virtual ~XSkippedRecordError() = default;
    };
    //! The size of the raw record is not enough to continue analyzing.
    struct DECLSPEC_KAME XBufferUnderflowRecordError : public XRecordError {
        XBufferUnderflowRecordError(const char *file, int line);
        virtual ~XBufferUnderflowRecordError() = default;
    };
protected:
 
	//! This function is called after committing XPrimaryDriver::analyzeRaw() or XSecondaryDriver::analyze().
	//! This might be called even if the record is invalid (time() == false).
	virtual void visualize(const Snapshot &shot) = 0;
  
	//! Sets time stamps.
	//! \sa time(), timeAwared()
	void record(Transaction &tr,
		const XTime &time_awared, const XTime &time_recorded);
};

class DECLSPEC_KAME XDriverList : public XCustomTypeListNode<XDriver> {
public:
	XDriverList(const char *name, bool runtime, const shared_ptr<XMeasure> &measure);

    DEFINE_TYPE_HOLDER(
        std::reference_wrapper<Transaction>,
		const shared_ptr<XMeasure> &
		)
    virtual shared_ptr<XNode> createByTypename(const XString &type, const XString& name) override;

    struct Payload : public XCustomTypeListNode<XDriver>::Payload {};
private:
	const weak_ptr<XMeasure> m_measure;
};
//---------------------------------------------------------------------------
#endif
