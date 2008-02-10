/***************************************************************************
		Copyright (C) 2002-2008 Kentaro Kitagawa
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

#ifndef xnodeconnectorH
#define xnodeconnectorH

#include <support.h>
#include <xnode.h>
#include <xlistnode.h>
#include <xitemnode.h>
#include <xnodeconnector_prv.h>

//! need for making new forms
extern QWidget *g_pFrmMain;

//! Associate QWidget to XNode
//! use connectWeak() to make XListener.
//! use xqcon_create<T>() to make instances
class XQConnector : public QObject,
public enable_shared_from_this<XQConnector>
{
	//! don't forget this
	Q_OBJECT
	//! needed for XQConnector and cousins
	XQCON_OBJECT
protected:
	//! don't use this, use xqcon_create() instead
	//! \sa xqcon_create()
	XQConnector(const shared_ptr<XNode> &node, QWidget *item);
public:
	//! Here, disconnect all signals & slots
	virtual ~XQConnector();

	static shared_ptr<XNode> connectedNode(const QWidget *item);
private slots:
protected slots:
protected:
	bool isItemAlive() const {return m_pWidget;}
	shared_ptr<XListener> m_lsnUIEnabled;
	void onUIEnabled(const shared_ptr<XNode> &node);
	QWidget *m_pWidget;
};

class QButton;

class XQButtonConnector : public XQConnector
{
	Q_OBJECT
	XQCON_OBJECT
protected:
	XQButtonConnector(const shared_ptr<XNode> &node, QButton *item);
public:
	virtual ~XQButtonConnector();
private slots:
protected slots:
virtual void onClick();
protected:
	virtual void onTouch(const shared_ptr<XNode> &node);
	shared_ptr<XListener> m_lsnTouch;
	const shared_ptr<XNode> m_node;
	QButton *const m_pItem;
};

class XValueQConnector : public XQConnector
{
	Q_OBJECT
	XQCON_OBJECT
protected:
	XValueQConnector(const shared_ptr<XValueNodeBase> &node, QWidget *item);
public:
	virtual ~XValueQConnector();
private slots:
protected:
	shared_ptr<XListener> m_lsnBeforeValueChanged;
	shared_ptr<XListener> m_lsnValueChanged;
	virtual void beforeValueChanged(const shared_ptr<XValueNodeBase> &node) = 0;
	virtual void onValueChanged(const shared_ptr<XValueNodeBase> &node) = 0;
};

class QLineEdit;

class XQLineEditConnector : public XValueQConnector
{
	Q_OBJECT
	XQCON_OBJECT
protected:
	XQLineEditConnector(const shared_ptr<XValueNodeBase> &node,
		QLineEdit *item, bool forcereturn = true);
public:
	virtual ~XQLineEditConnector() {}
protected slots:
void onTextChanged(const QString &);
void onTextChanged2(const QString &);
void onReturnPressed();
void onExit();
protected:
	virtual void beforeValueChanged(const shared_ptr<XValueNodeBase> &) {}
	virtual void onValueChanged(const shared_ptr<XValueNodeBase> &);
	const shared_ptr<XValueNodeBase> m_node;
	QLineEdit *const m_pItem;
};

class QTextBrowser;

class XQTextBrowserConnector : public XValueQConnector
{
	Q_OBJECT
	XQCON_OBJECT
protected:
	XQTextBrowserConnector(const shared_ptr<XValueNodeBase> &node,
		QTextBrowser *item);
public:
	virtual ~XQTextBrowserConnector() {}
protected slots:
protected:
	virtual void beforeValueChanged(const shared_ptr<XValueNodeBase> &) {}
	virtual void onValueChanged(const shared_ptr<XValueNodeBase> &);
	const shared_ptr<XValueNodeBase> m_node;
	QTextBrowser *const m_pItem;
};

class QSpinBox;
class QSlider;

class XQSpinBoxConnector : public XValueQConnector
{
	Q_OBJECT
	XQCON_OBJECT
protected:
	XQSpinBoxConnector(const shared_ptr<XIntNode> &node,
		QSpinBox *item, QSlider *slider = 0L);
	XQSpinBoxConnector(const shared_ptr<XUIntNode> &node,
		QSpinBox *item, QSlider *slider = 0L);
public:
	virtual ~XQSpinBoxConnector() {}
protected slots:
void onChange(int val);
protected:
	virtual void beforeValueChanged(const shared_ptr<XValueNodeBase> &) {}
	virtual void onValueChanged(const shared_ptr<XValueNodeBase> &node);
	const shared_ptr<XIntNode> m_iNode;
	const shared_ptr<XUIntNode> m_uINode;
	QSpinBox *const m_pItem;
	QSlider *const m_pSlider;
};

class KDoubleNumInput;

class XKDoubleNumInputConnector : public XValueQConnector
{
	Q_OBJECT
	XQCON_OBJECT
protected:
	XKDoubleNumInputConnector(const shared_ptr<XDoubleNode> &node, 
		KDoubleNumInput *item);
public:
	virtual ~XKDoubleNumInputConnector() {}
protected slots:
void onChange(double val);
protected:
	virtual void beforeValueChanged(const shared_ptr<XValueNodeBase> &) {}
	virtual void onValueChanged(const shared_ptr<XValueNodeBase> &node);
	const shared_ptr<XDoubleNode> m_node;
	KDoubleNumInput *const m_pItem;
};

class KDoubleSpinBox;

class XKDoubleSpinBoxConnector : public XValueQConnector
{
	Q_OBJECT
	XQCON_OBJECT
protected:
	XKDoubleSpinBoxConnector(const shared_ptr<XDoubleNode> &node,
		KDoubleSpinBox *item);
public:
	virtual ~XKDoubleSpinBoxConnector() {}
protected slots:
void onChange(double val);
protected:
	virtual void beforeValueChanged(const shared_ptr<XValueNodeBase> &) {}
	virtual void onValueChanged(const shared_ptr<XValueNodeBase> &node);
	const shared_ptr<XDoubleNode> m_node;
	KDoubleSpinBox *const m_pItem;
};

class KURLRequester;

class XKURLReqConnector : public XValueQConnector
{
	Q_OBJECT
	XQCON_OBJECT
protected:
	XKURLReqConnector(const shared_ptr<XStringNode> &node, 
		KURLRequester *item, const char *filter, bool saving);
public:
	virtual ~XKURLReqConnector() {}
protected slots:
void onSelect( const QString& );
protected:
	virtual void beforeValueChanged(const shared_ptr<XValueNodeBase> &) {}
	virtual void onValueChanged(const shared_ptr<XValueNodeBase> &node);
	const shared_ptr<XStringNode> m_node;
	KURLRequester *const m_pItem;
};

class QLabel;

class XQLabelConnector : public XValueQConnector
{
	Q_OBJECT
	XQCON_OBJECT
protected:
	XQLabelConnector(const shared_ptr<XValueNodeBase> &node, 
		QLabel *item);
public:
	virtual ~XQLabelConnector() {}
protected slots:
protected:
	virtual void beforeValueChanged(const shared_ptr<XValueNodeBase> &) {}
	virtual void onValueChanged(const shared_ptr<XValueNodeBase> &node);
	const shared_ptr<XValueNodeBase> m_node;
	QLabel *const m_pItem;
};

class KLed;

class XKLedConnector : public XValueQConnector
{
	Q_OBJECT
	XQCON_OBJECT
protected:
	XKLedConnector(const shared_ptr<XBoolNode> &node,
		KLed *item);
public:
	virtual ~XKLedConnector() {}
protected slots:
protected:
	virtual void beforeValueChanged(const shared_ptr<XValueNodeBase> &) {}
	virtual void onValueChanged(const shared_ptr<XValueNodeBase> &node);
	const shared_ptr<XBoolNode> m_node;
	KLed *const m_pItem;
};

class QLCDNumber;

class XQLCDNumberConnector : public XValueQConnector
{
	Q_OBJECT
	XQCON_OBJECT
protected:
	XQLCDNumberConnector(const shared_ptr<XDoubleNode> &node,
		QLCDNumber *item);
public:
	virtual ~XQLCDNumberConnector() {}
protected:
	virtual void beforeValueChanged(const shared_ptr<XValueNodeBase> &) {}
	virtual void onValueChanged(const shared_ptr<XValueNodeBase> &node);
	const shared_ptr<XDoubleNode> m_node;
	QLCDNumber *const m_pItem;
};

class XQToggleButtonConnector : public XValueQConnector
{
	Q_OBJECT
	XQCON_OBJECT
protected:
	XQToggleButtonConnector(const shared_ptr<XBoolNode> &node,
		QButton *item);
public:
	virtual ~XQToggleButtonConnector() {}
protected slots:
void onClick();
protected:
	virtual void beforeValueChanged(const shared_ptr<XValueNodeBase> &) {}
	virtual void onValueChanged(const shared_ptr<XValueNodeBase> &node);
	const shared_ptr<XBoolNode> m_node;
	QButton *const m_pItem;
};

class QTable;
class XListQConnector : public XQConnector
{
	Q_OBJECT
	XQCON_OBJECT
protected:
	XListQConnector(const shared_ptr<XListNodeBase> &node, QTable *item);
public:
	virtual ~XListQConnector();
private slots:
protected slots:
void indexChange(int section, int fromIndex, int toIndex);
protected:
	shared_ptr<XListener> m_lsnMove;
	virtual void onMove(const XListNodeBase::MoveEvent &node);
	shared_ptr<XListener> m_lsnCatch;
	shared_ptr<XListener> m_lsnRelease;
	virtual void onCatch(const shared_ptr<XNode> &node) = 0;
	virtual void onRelease(const shared_ptr<XNode> &node) = 0;
	QTable *const m_pItem;
	const shared_ptr<XListNodeBase> m_list;
};

class XItemQConnector : public XValueQConnector
{
	Q_OBJECT
	XQCON_OBJECT
protected:
	XItemQConnector(const shared_ptr<XItemNodeBase> &node,
		QWidget *item);
public:
	virtual ~XItemQConnector();
private slots:
protected slots:
protected:
	shared_ptr<XListener>  m_lsnListChanged;
	virtual void onListChanged(const shared_ptr<XItemNodeBase> &) = 0;
	shared_ptr<const std::deque<XItemNodeBase::Item> > m_itemStrings;
};

class QComboBox;

class XQComboBoxConnector : public XItemQConnector
{
	Q_OBJECT
	XQCON_OBJECT
protected:
	XQComboBoxConnector(const shared_ptr<XItemNodeBase> &node,
		QComboBox *item);
public:
	virtual ~XQComboBoxConnector() {}
protected slots:
virtual void onSelect(int index);
protected:
	virtual void beforeValueChanged(const shared_ptr<XValueNodeBase> &) {}
	virtual void onValueChanged(const shared_ptr<XValueNodeBase> &);
	virtual void onListChanged(const shared_ptr<XItemNodeBase> &);
	const shared_ptr<XItemNodeBase> m_node;
	QComboBox *const m_pItem;
	int findItem(const QString &);
};

class QListBox;

class XQListBoxConnector : public XItemQConnector
{
	Q_OBJECT
	XQCON_OBJECT
protected:
	XQListBoxConnector(const shared_ptr<XItemNodeBase> &node,
		QListBox *item);
public:
	virtual ~XQListBoxConnector() {}
protected slots:
virtual void onSelect(int index);
protected:
	virtual void beforeValueChanged(const shared_ptr<XValueNodeBase> &) {}
	virtual void onValueChanged(const shared_ptr<XValueNodeBase> &);
	virtual void onListChanged(const shared_ptr<XItemNodeBase> &);
	const shared_ptr<XItemNodeBase> m_node;
	QListBox *const m_pItem;
};

class KColorButton;

class XKColorButtonConnector : public XValueQConnector
{
	Q_OBJECT
	XQCON_OBJECT
protected:
	XKColorButtonConnector(const shared_ptr<XHexNode> &node,
		KColorButton *item);
public:
	virtual ~XKColorButtonConnector() {}
protected slots:
void onClick(const QColor &newColor);
protected:
	virtual void beforeValueChanged(const shared_ptr<XValueNodeBase> &) {}
	virtual void onValueChanged(const shared_ptr<XValueNodeBase> &);
	const shared_ptr<XHexNode> m_node;
	KColorButton *const m_pItem;
};

class KColorCombo;

class XKColorComboConnector : public XValueQConnector
{
	Q_OBJECT
	XQCON_OBJECT
protected:
	XKColorComboConnector(const shared_ptr<XHexNode> &node, 
		KColorCombo *item);
public:
	virtual ~XKColorComboConnector() {}
protected slots:
void onClick(const QColor &newColor);
protected:
	virtual void beforeValueChanged(const shared_ptr<XValueNodeBase> &) {}
	virtual void onValueChanged(const shared_ptr<XValueNodeBase> &);
	const shared_ptr<XHexNode> m_node;
	KColorCombo *const m_pItem;
};

//! Show status
class QMainWindow;
class QStatusBar;
class KPassivePopup;
class XStatusPrinter : public enable_shared_from_this<XStatusPrinter>
{
protected:
	explicit XStatusPrinter(QMainWindow *window = NULL);
public:
	static shared_ptr<XStatusPrinter> create(QMainWindow *window = NULL);
	~XStatusPrinter();
	void printMessage(const QString &str, bool popup = true);
	void printWarning(const QString &str, bool popup = false);
	void printError(const QString &str, bool popup = true);
	void clear();
private:
	struct tstatus {std::string str; int ms; bool popup; enum {Normal, Warning, Error} type;};
	XTalker<tstatus> m_tlkTalker;
	shared_ptr<XListener> m_lsn;
	QMainWindow *m_pWindow;
	QStatusBar *m_pBar;
	KPassivePopup *m_pPopup;
	void print(const tstatus &status);
};

//---------------------------------------------------------------------------
#endif
