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
#ifndef dmmH
#define dmmH
//---------------------------------------------------------------------------
#include "primarydriverwiththread.h"
#include "xnodeconnector.h"

class XScalarEntry;
class QMainWindow;
class Ui_FrmDMM;
typedef QForm<QMainWindow, Ui_FrmDMM> FrmDMM;

class DECLSPEC_SHARED XDMM : public XPrimaryDriverWithThread {
public:
	XDMM(const char *name, bool runtime,
        Transaction &tr_meas, const shared_ptr<XMeasure> &meas,
        unsigned int max_num_channels = 1);

	//! usually nothing to do
	virtual ~XDMM() {}
	//! Shows all forms belonging to driver
	virtual void showForms();
	
	struct Payload : public XPrimaryDriver::Payload {
        double value(unsigned int ch = 0) const {
            if(m_var.size() <= ch)
                throw XInterface::XInterfaceError(i18n("Wrong Channel No."), __FILE__, __LINE__);
            return m_var[ch];
        }
        void write_(double var, unsigned int ch = 0) {
            if(m_var.size() <= ch)
                m_var.resize(ch + 1);
            m_var[ch] = var;
        }
	private:
        std::vector<double> m_var;
	};

    const shared_ptr<XComboNode> &function() const {return m_function;}
    const shared_ptr<XUIntNode> &waitInms() const {return m_waitInms;}
protected:
	//! This function will be called when raw data are written.
	//! Implement this function to convert the raw data to the record (Payload).
	//! \sa analyze()
	virtual void analyzeRaw(RawDataReader &reader, Transaction &tr);
	//! This function is called after committing XPrimaryDriver::analyzeRaw() or XSecondaryDriver::analyze().
	//! This might be called even if the record is invalid (time() == false).
	virtual void visualize(const Snapshot &shot);
  
protected:
    //! one-shot reading
	virtual double oneShotRead() = 0; 
	//! is called when m_function is changed
	virtual void changeFunction() = 0;
    //! one-shot multi-channel reading
    virtual std::deque<double> oneShotMultiRead() {return {};}
    unsigned int maxNumOfChannels() const {return m_maxNumOfChannels;}
private:
	//! is called when m_function is changed
	void onFunctionChanged(const Snapshot &shot, XValueNodeBase *node);
  
    std::deque<shared_ptr<XScalarEntry>> m_entries;
	const shared_ptr<XComboNode> m_function;
	const shared_ptr<XUIntNode> m_waitInms;
	shared_ptr<Listener> m_lsnOnFunctionChanged;
    std::deque<xqcon_ptr> m_conUIs;
 
	const qshared_ptr<FrmDMM> m_form;
    void *execute(const atomic<bool> &);
  
    const unsigned int m_maxNumOfChannels;
};

#endif
